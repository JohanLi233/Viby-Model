"""Protected PSR: dense recurrent state and a zero-initialized logit residual.

No decoder injection, indexer teacher, rank loss, future-offset head or stopping
policy. Baseline features cross four explicit stop-gradient boundaries.
"""

from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
from mlx import nn
from .attention import _cos_sin
from .norms import RMSNorm
from .rope import rope_partial


def masked_softmax(scores, valid):
    p = mx.softmax(mx.where(valid, scores.astype(mx.float32), -1e30), axis=-1) * valid
    denominator = mx.sum(p, axis=-1, keepdims=True)
    return p / mx.where(denominator > 0, denominator, 1.0)


@dataclass(frozen=True)
class EvidenceMemory:
    values: mx.array  # [B,N,D], original rotated CED KV
    features: mx.array  # [B,T,dim], actual mHC boundary attention input
    positions: mx.array  # [B,T], global positions for features
    pad: Optional[mx.array] = None
    segments: Optional[mx.array] = None


@dataclass(frozen=True)
class ThinkingState:
    slots: mx.array  # [B,A,m,s]
    anchor: mx.array  # [B,A], -1 is padding
    segment: Optional[mx.array]
    valid: mx.array
    rounds: int
    horizon: int
    mode: str

    def for_decode(self):
        return self


@dataclass(frozen=True)
class ReasoningTrace:
    states: tuple
    full_scans: int
    rounds: int
    read_mode: str
    rho: tuple = ()
    read_error: tuple = ()
    error_bound: tuple = ()


class StateBlock(nn.Module):
    """Small dense slot communication + bounded residual update, shared in r."""

    def __init__(self, dim, eps, scale):
        super().__init__()
        self.norm = RMSNorm(dim, eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim, eps)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(2 * dim, dim, bias=False)
        self.scale = scale

    def __call__(self, slots):
        q, k, v = mx.split(self.qkv(self.norm(slots)), 3, axis=-1)
        p = mx.softmax(
            (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2))
            * q.shape[-1] ** -0.5,
            axis=-1,
        ).astype(v.dtype)
        slots = slots + self.scale * mx.tanh(self.out(p @ v))
        a, b = mx.split(self.up(self.ffn_norm(slots)), 2, axis=-1)
        return slots + self.scale * mx.tanh(self.down(nn.silu(a) * b))


class ProtectedPSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        s = config.psr_dim
        self.slots = mx.random.normal((config.psr_slots, s)) * s**-0.5
        self.init_proj = nn.Linear(config.dim, s, bias=False)
        self.norm = RMSNorm(s, config.norm_eps)
        self.read_query = nn.Linear(s, config.head_dim, bias=False)
        self.read_out = nn.Linear(config.head_dim, s, bias=False)
        self.blocks = [
            StateBlock(s, config.norm_eps, config.psr_update_scale)
            for _ in range(config.psr_blocks)
        ]
        self.query_norm = RMSNorm(config.dim, config.norm_eps)
        self.query = nn.Linear(config.dim, s, bias=False)
        self.key = nn.Linear(s, s, bias=False)
        self.value = nn.Linear(s, s, bias=False)
        self.offset = nn.Embedding(config.psr_horizon, s)
        self.output = nn.Linear(s, config.vocab_size, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.calibration_gate = mx.array(1.0)
        self.freeze(recurse=False, keys=["calibration_gate"])

    def _query(self, slots, anchors, cos, sin):
        q = self.read_query(self.norm(slots))
        pos = mx.broadcast_to(mx.maximum(anchors, 0)[:, None], slots.shape[:2])
        c, sn = _cos_sin(cos, sin, pos, True)
        return (
            rope_partial(q[..., None, :], c, sn, self.config.rope_head_dim)[..., 0, :],
            c,
            sn,
        )

    def _read(
        self, slots, values, visible, anchors, cos, sin, indices=None, diagnose=False
    ):
        q, c, sn = self._query(slots, anchors, cos, sin)
        rho = error = bound = None
        if indices is None or diagnose:
            scores = q.astype(mx.float32) @ values.astype(mx.float32).swapaxes(-1, -2)
            p = masked_softmax(scores * self.config.head_dim**-0.5, visible[:, None])
            dense = p.astype(values.dtype) @ values
        if indices is None:
            obs = dense
        else:
            in_range = (indices >= 0) & (indices < values.shape[1])
            safe = mx.clip(indices, 0, values.shape[1] - 1)
            gathered = values[mx.arange(values.shape[0])[:, None, None], safe]
            allowed = (
                visible[mx.arange(values.shape[0])[:, None, None], safe] & in_range
            )
            score = mx.sum(
                q.astype(mx.float32)[..., None, :] * gathered.astype(mx.float32),
                axis=-1,
            )
            selected_p = masked_softmax(score * self.config.head_dim**-0.5, allowed)
            obs = mx.sum(
                selected_p.astype(gathered.dtype)[..., None] * gathered, axis=-2
            )
            if diagnose:
                rho = mx.sum(mx.take_along_axis(p, safe, -1) * in_range, axis=-1)
                error = mx.sqrt(
                    mx.sum(
                        mx.square(dense.astype(mx.float32) - obs.astype(mx.float32)), -1
                    )
                )
                vmax = mx.max(
                    mx.where(
                        visible, mx.sqrt(mx.sum(values.astype(mx.float32) ** 2, -1)), 0
                    ),
                    -1,
                )
                bound = 2 * vmax[:, None] * (1 - rho)
        obs = rope_partial(
            obs[..., None, :], c, sn, self.config.rope_head_dim, inverse=True
        )[..., 0, :]
        return self.read_out(obs), rho, error, bound

    def __call__(
        self,
        memory,
        anchors,
        cos,
        sin,
        *,
        mode="recurrent",
        rounds=None,
        read_mode="dense",
        fixed_indices=None,
        record_trace=False,
        min_rho=0.95,
    ):
        cfg = self.config
        if mode not in ("state_only", "recurrent"):
            raise ValueError(
                "PSR forward accepts state_only/recurrent; off must bypass it"
            )
        rounds = cfg.psr_rounds if rounds is None else rounds
        if mode == "recurrent" and (
            not isinstance(rounds, int) or not 1 <= rounds <= cfg.psr_max_rounds
        ):
            raise ValueError(
                "recurrent mode requires R>=1; use state_only or off explicitly"
            )
        rounds = 0 if mode == "state_only" else rounds
        if read_mode not in ("dense", "fixed", "sparse_diagnostic"):
            raise ValueError("unknown PSR read mode")
        if read_mode != "dense" and fixed_indices is None:
            raise ValueError(
                "fixed/sparse_diagnostic needs explicit frozen selection indices"
            )
        if (
            anchors.ndim != 2
            or anchors.shape[0] != memory.values.shape[0]
            or not mx.issubdtype(anchors.dtype, mx.integer)
        ):
            raise ValueError("anchors must be an integer array [B,A]")
        b, a = anchors.shape
        n = memory.values.shape[1]
        local = anchors - memory.positions[:, :1]
        valid = (local >= 0) & (local < memory.features.shape[1])
        safe = mx.clip(local, 0, memory.features.shape[1] - 1).astype(mx.int32)
        row = mx.arange(b)[:, None]
        if memory.pad is not None:
            valid = valid & memory.pad[row, safe]
        segment = None if memory.segments is None else memory.segments[row, safe]
        visible = (mx.arange(n)[None, None, :] <= anchors[..., None]) & valid[..., None]
        if memory.pad is not None and memory.pad.shape[1] == n:
            visible = visible & memory.pad[:, None, :]
        if memory.segments is not None and memory.segments.shape[1] == n:
            visible = visible & (memory.segments[:, None, :] == segment[..., None])
        # Both branches are detached independently; do not rely on read-only semantics.
        initial = mx.stop_gradient(memory.features[row, safe])
        values = mx.stop_gradient(memory.values)
        slots = self.init_proj(initial)[..., None, :] + self.slots
        slots = slots.reshape(b * a, cfg.psr_slots, cfg.psr_dim)
        values = mx.broadcast_to(values[:, None], (b, a, n, cfg.head_dim)).reshape(
            b * a, n, cfg.head_dim
        )
        visible = visible.reshape(b * a, n)
        history, rhos, errors, bounds = [slots], [], [], []
        indices = None
        if fixed_indices is not None:
            indices = mx.stop_gradient(
                fixed_indices.reshape(b * a, cfg.psr_slots, -1).astype(mx.int32)
            )
            # Out-of-range sets are an input error, not a silent approximation.
            if indices.shape[-1] == 0:
                raise ValueError("empty fixed read set")
        for _ in range(rounds):
            obs, rho, error, bound = self._read(
                slots,
                values,
                visible,
                anchors.reshape(-1),
                cos,
                sin,
                indices,
                read_mode == "sparse_diagnostic",
            )
            slots = slots + cfg.psr_update_scale * mx.tanh(obs)
            for block in self.blocks:
                slots = block(slots)
            history.append(slots)
            if rho is not None:
                rhos.append(rho)
                errors.append(error)
                bounds.append(bound)
        if read_mode == "sparse_diagnostic" and rhos:
            required = valid.reshape(b * a, 1)
            observed = float(mx.min(mx.where(required, mx.stack(rhos), 1)))
            if not 0 <= min_rho <= 1:
                raise ValueError("min_rho must be in [0,1]")
            if observed < min_rho or not bool(mx.all(mx.isfinite(mx.stack(errors)))):
                raise ValueError(
                    f"sparse read quality failed: rho={observed:.6f} < {min_rho}; no prediction returned"
                )
        state = ThinkingState(
            slots.reshape(b, a, cfg.psr_slots, cfg.psr_dim),
            anchors,
            segment,
            valid,
            rounds,
            cfg.psr_horizon,
            mode,
        )
        scans = rounds if read_mode != "fixed" else 0
        trace = (
            ReasoningTrace(
                tuple(history),
                scans,
                rounds,
                read_mode,
                tuple(rhos),
                tuple(errors),
                tuple(bounds),
            )
            if record_trace
            else None
        )
        return state, trace

    def read_workspace(self, hidden, state, positions, segments=None, pad=None):
        b, t, _ = hidden.shape
        offsets = positions[:, None, :] - state.anchor[..., None]
        allowed = (offsets >= 0) & (offsets < state.horizon) & state.valid[..., None]
        if segments is not None and state.segment is not None:
            allowed = allowed & (segments[:, None] == state.segment[..., None])
        if pad is not None:
            allowed = allowed & pad[:, None].astype(mx.bool_)
        # Deterministic ownership for supplied overlapping anchors: latest wins.
        owner = mx.argmax(mx.where(allowed, state.anchor[..., None], -1), axis=1)
        active = mx.any(allowed, axis=1)
        row = mx.arange(b)[:, None]
        offset = mx.clip(positions - state.anchor[row, owner], 0, state.horizon - 1)
        q = self.query(self.query_norm(mx.stop_gradient(hidden))) + self.offset(
            offset.astype(mx.int32)
        )
        k, v = self.key(state.slots)[row, owner], self.value(state.slots)[row, owner]
        p = mx.softmax(
            mx.sum(q.astype(mx.float32)[..., None, :] * k.astype(mx.float32), -1)
            * q.shape[-1] ** -0.5,
            axis=-1,
        ).astype(v.dtype)
        output = mx.sum(p[..., None] * v, axis=-2)
        return mx.where(active[..., None], output, 0), active, offset


def residual_statistics(base_nll, corrected_nll, valid, covered, offsets, prefix=None):
    """Per-example sums/counts; callers aggregate sums, never batch means."""
    masks = [
        valid,
        valid * covered,
        valid * (~covered),
        valid * (mx.zeros_like(covered) if prefix is None else prefix),
    ]
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32)):
        masks.append(valid * covered * (offsets >= lo) * (offsets < hi))
    return mx.stack(
        [
            mx.stack(
                [
                    mx.sum(base_nll * m, -1),
                    mx.sum(corrected_nll * m, -1),
                    mx.sum(m, -1),
                ],
                -1,
            )
            for m in masks
        ],
        axis=1,
    )
