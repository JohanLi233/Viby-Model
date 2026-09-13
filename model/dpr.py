"""DPR-JEPA: causal particle prediction; future-only training targets.

No temporal state. Kernel/covariance reductions are FP32, covariance uses 1/N.
N=0 returns zero auxiliary terms with explicit coverage metrics; N>=1 retains
the regularizer, including rank-deficient batches (reported, never hidden).
"""

import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn


DPR_METRICS = (
    "latent_kernel_loss",
    "latent_cov_loss",
    "latent_mse_loss",
    "valid_targets",
    "target_capacity",
    "coverage",
    "cov_rank_deficient",
    "weight",
    "target_variance",
    "particle_spread",
    "prob_entropy",
    "target_mean_square",
    "target_second_moment",
    "target_effective_rank",
)


def target_moments(y, mask):
    """Decompose drift of detached targets; no eigenvalue decomposition."""
    rank = y.shape[-1]
    count = mask.sum()
    denom = mx.maximum(count, 1)
    flat_y, flat_m = y.reshape(-1, rank), mask.reshape(-1, 1)
    mu = mx.sum(flat_y * flat_m, axis=0) / denom
    centered = (flat_y - mu) * flat_m
    cov = centered.T @ centered / denom
    mean_square = mx.mean(mx.square(mu))
    reg = mean_square + mx.sum(mx.square(cov - mx.eye(rank))) / rank
    reg = mx.where(count > 0, reg, 0)
    trace = mx.trace(cov)
    variance = trace / rank
    effective_rank = trace * trace / mx.maximum(mx.sum(mx.square(cov)), 1e-20)
    return reg, variance, mean_square, variance + mean_square, effective_rank


def context_projection(dim, rank, seed):
    """Deterministic orthonormal rows; stored in the checkpoint as a buffer."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, rank)))
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
    return np.ascontiguousarray(q.T, dtype=np.float32)


def kernel_score(pi, z, y):
    pi, z, y = (a.astype(mx.float32) for a in (pi, z, y))
    r = z.shape[-1]
    pair = mx.sum(mx.square(z[..., :, None, :] - z[..., None, :, :]), axis=-1)
    cross = mx.sum(mx.square(z - y[..., None, :]), axis=-1)
    return (
        mx.sum(
            pi[..., :, None] * pi[..., None, :] * mx.exp(-pair / (2 * r)), axis=(-2, -1)
        )
        - 2 * mx.sum(pi * mx.exp(-cross / (2 * r)), axis=-1)
        + 1
    )


def future_mask(ids, k, pad_mask=None, segment_ids=None, pad_id=0, eos_id=2):
    b, t = ids.shape
    n = max(t - k, 0)
    live = ids != pad_id
    if pad_mask is not None:
        live = live & pad_mask.astype(mx.bool_)
    valid = live[:, :n]
    for j in range(1, k + 1):
        valid = valid & live[:, j : j + n] & (ids[:, j - 1 : j - 1 + n] != eos_id)
        if segment_ids is not None:
            valid = valid & (segment_ids[:, j : j + n] == segment_ids[:, :n])
    return valid


class FutureEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.offsets = [
            nn.Linear(cfg.dim, cfg.dpr_width, bias=False)
            for _ in range(cfg.dpr_horizon)
        ]
        self.bias = mx.zeros((cfg.dpr_width,))
        self.proj = nn.Linear(cfg.dpr_width, cfg.dpr_dim)

    def __call__(self, embeddings):
        e = mx.stop_gradient(embeddings)
        n = max(e.shape[1] - len(self.offsets), 0)
        u = self.bias
        for j, layer in enumerate(self.offsets, 1):
            u = u + layer(e[:, j : j + n])
        return math.sqrt(3) * mx.tanh(self.proj(nn.gelu(u)).astype(mx.float32))


class DistributionalPredictiveResidual(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.predict = nn.Linear(cfg.dim, cfg.dpr_particles * (cfg.dpr_dim + 1))
        self.output = nn.Linear(
            cfg.dpr_particles * (cfg.dpr_dim + 1), cfg.dim, bias=False
        )
        if cfg.dpr_variant == "legacy_v1":
            self.target = FutureEncoder(cfg)
        else:
            self.context_projection = mx.array(
                context_projection(cfg.dim, cfg.dpr_dim, cfg.dpr_seed)
            )
            self.freeze(keys=["context_projection"], recurse=False)
        self.output.weight = mx.zeros_like(self.output.weight)

    def contextual_targets(self, hidden):
        """Future groups from pre-injection CED states, detached in their entirety.

        Project before pooling (linear operations commute), so window work is
        O(B*T*k*r), not O(B*T*k*d). No teacher pass or target optimizer exists.
        """
        k = self.config.dpr_horizon
        n = max(hidden.shape[1] - k, 0)
        projected = (
            mx.stop_gradient(hidden).astype(mx.float32)
            @ mx.stop_gradient(self.context_projection).T
        )
        pooled = sum(projected[:, j : j + n] for j in range(1, k + 1)) / k
        normalized = pooled * mx.rsqrt(
            mx.mean(pooled * pooled, axis=-1, keepdims=True) + 1e-6
        )
        return mx.stop_gradient(math.sqrt(3) * mx.tanh(normalized))

    def __call__(self, a, intervention=None):
        cfg = self.config
        raw = self.predict(a).astype(mx.float32)
        raw = raw.reshape(*a.shape[:-1], cfg.dpr_particles, cfg.dpr_dim + 1)
        pi = mx.softmax(raw[..., 0], axis=-1)
        z = math.sqrt(3) * mx.tanh(raw[..., 1:])
        if intervention == "mean":
            z = mx.broadcast_to(
                mx.sum(pi[..., None] * z, axis=-2, keepdims=True), z.shape
            )
        elif intervention == "shuffle":
            if a.shape[0] < 2:
                raise ValueError(
                    "DPR shuffle requires at least two matched-length rows"
                )
            pi, z = mx.roll(pi, 1, axis=0), mx.roll(z, 1, axis=0)
        elif intervention is not None:
            raise ValueError("DPR intervention must be mean/shuffle or None")
        slots = mx.concatenate([pi[..., None], pi[..., None] * z], axis=-1)
        delta = self.output(slots.reshape(*a.shape[:-1], -1).astype(a.dtype))
        return delta, pi, z

    def auxiliary(self, pi, z, embeddings, ids, pad_mask=None, segment_ids=None):
        cfg = self.config
        n = max(ids.shape[1] - cfg.dpr_horizon, 0)
        if n == 0:
            metrics = mx.zeros((len(DPR_METRICS),), mx.float32)
            metrics = metrics.at[4].add(ids.size).at[6].add(1)
            return mx.array(0.0, mx.float32), metrics
        legacy = cfg.dpr_variant == "legacy_v1"
        y = self.target(embeddings) if legacy else self.contextual_targets(embeddings)
        pi, z = pi[:, :n], z[:, :n]
        mask = future_mask(
            ids,
            cfg.dpr_horizon,
            pad_mask,
            segment_ids,
            cfg.pad_token_id,
            cfg.eos_token_id,
        ).astype(mx.float32)
        count = mask.sum()
        denom = mx.maximum(count, 1)
        score = mx.sum(kernel_score(pi, z, y) * mask) / denom
        mean_z = mx.sum(pi[..., None] * z, axis=-2)
        mse = mx.sum(mx.mean(mx.square(mean_z - y), axis=-1) * mask) / denom
        reg, variance, mean_square, second_moment, effective_rank = target_moments(
            y, mask
        )
        spread = (
            mx.sum(
                mx.sum(
                    pi * mx.mean(mx.square(z - mean_z[..., None, :]), axis=-1), axis=-1
                )
                * mask
            )
            / denom
        )
        entropy = (
            mx.sum(-mx.sum(pi * mx.log(mx.maximum(pi, 1e-20)), axis=-1) * mask) / denom
        )
        metrics = mx.stack(
            [
                score,
                reg,
                mse,
                count,
                mx.array(ids.size, mx.float32),
                count / ids.size,
                (count <= cfg.dpr_dim).astype(mx.float32),
                mx.array(0.0, mx.float32),
                variance,
                spread,
                entropy,
                mean_square,
                second_moment,
                effective_rank,
            ]
        )
        # For detached contextual targets covariance is diagnostic only. Adding
        # it to the training loss cannot improve the predictor and obscures CE.
        return (score if cfg.dpr_objective == "kernel" else mse) + (
            reg if legacy else 0
        ), metrics
