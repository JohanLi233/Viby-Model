"""Predictive-state reasoning over a frozen CED memory.

Internal rounds never advance token positions. Exact global indexing is the
reference implementation; margin certificates are diagnostic-only (no proven
floating-point error enclosure). Targets are consumed only by psr_losses.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx import nn

from .attention import _cos_sin
from .norms import RMSNorm
from .rope import rope_partial

STOP, COMPUTE, READ = 0, 1, 2


@dataclass(frozen=True)
class EvidenceMemory:
    values: mx.array             # [B,N,head_dim], already RoPE-transformed
    keys: mx.array               # [B,N,index_head_dim], already transformed
    visible: mx.array            # [B,N], question prefix AND document AND pad
    anchor: mx.array             # [B], final observed token, never a round index


@dataclass(frozen=True)
class IndexReference:
    query: mx.array
    weights: mx.array
    indices: mx.array
    margin: mx.array
    key_norm: mx.array


@dataclass(frozen=True)
class ThinkingState:
    slots: mx.array
    anchor: mx.array
    segment: Optional[mx.array]
    valid: mx.array
    rounds: int
    remaining: int
    index_reference: Optional[IndexReference] = None

    def for_decode(self):
        # Decoding only needs the terminal workspace, not the index trace.
        return ThinkingState(self.slots, self.anchor, self.segment, self.valid,
                             self.rounds, self.remaining)


@dataclass(frozen=True)
class ReasoningTrace:
    states: mx.array             # [B,R+1,m,d], includes S0
    scores: tuple                # R entries: [B,m,N] on READ, None on COMPUTE
    indices: tuple               # R entries: selected addresses or None
    values: mx.array             # [B,R+1,3], downstream loss + future cost
    actions: tuple               # executed Python action IDs
    reuse_possible: tuple       # real-arithmetic diagnostic; never skips scan
    actual_cost: mx.array        # includes every executed scan/update
    full_scans: int
    memory: EvidenceMemory      # retained only when a trace is requested
    budget: int                 # original budget, even if policy stopped early
    index_distillation: Optional[mx.array] = None


def masked_softmax(scores, valid):
    """All-masked rows return zeros, including their gradients."""
    scores = mx.where(valid, scores.astype(mx.float32), -1e30)
    p = mx.softmax(scores, axis=-1) * valid
    return p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), 1e-30)


def exact_topk(scores, visible, k):
    """Stable score ordering: ties use the lowest address, never perturb scores."""
    n = scores.shape[-1]
    k = min(k, n)
    ranked = mx.where(visible[:, None, :], scores, -mx.inf)
    idx = mx.argsort(-mx.stop_gradient(ranked), axis=-1)[..., :k]
    idx = mx.sort(idx, axis=-1).astype(mx.int32)
    valid = mx.take_along_axis(mx.broadcast_to(visible[:, None, :], scores.shape), idx, -1)
    selected = mx.where(valid, idx, -1)
    ordered = mx.sort(mx.stop_gradient(ranked), axis=-1)
    # No outside valid key => no boundary to certify; conservative fallback.
    if k < n:
        gap = ordered[..., -k] - ordered[..., -k - 1]
        gap = mx.where(mx.isfinite(gap), gap, 0.0)
    else:
        gap = mx.zeros(scores.shape[:-1], dtype=mx.float32)
    return mx.stop_gradient(selected), gap


def index_change_bound(reference, query, weights):
    """ReLU multihead bound with scaling already included in weights.

    This floating-point evaluation is NOT a rounding-error proof.
    The caller must keep memory/visibility fixed across the comparison.
    """
    q0, w0 = reference.query.astype(mx.float32), reference.weights.astype(mx.float32)
    q1, w1 = query.astype(mx.float32), weights.astype(mx.float32)
    dq = mx.sqrt(mx.sum(mx.square(q1 - q0), axis=-1))
    nq = mx.sqrt(mx.sum(mx.square(q0), axis=-1))
    return reference.key_norm[:, None] * mx.sum(mx.abs(w1) * dq + mx.abs(w1 - w0) * nq, axis=-1)


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
        p = mx.softmax((q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2))
                       * q.shape[-1] ** -0.5, axis=-1).astype(v.dtype)
        slots = slots + self.scale * mx.tanh(self.out(p @ v))
        a, b = mx.split(self.up(self.ffn_norm(slots)), 2, axis=-1)
        return slots + self.scale * mx.tanh(self.down(nn.silu(a) * b))


class WorkspaceBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, s = config.dim, config.psr_dim
        self.norm = RMSNorm(d, config.norm_eps)
        self.query = nn.Linear(d, s, bias=False)
        self.key = nn.Linear(s, s, bias=False)
        self.value = nn.Linear(s, s, bias=False)
        self.out = nn.Linear(s, d, bias=False)
        self.gate = mx.array(config.psr_bridge_init, dtype=mx.float32)

    def __call__(self, hidden, state, token_pos, segment_ids=None, pad_mask=None):
        q = self.query(self.norm(hidden))
        k, v = self.key(state.slots), self.value(state.slots)
        p = mx.softmax(q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
                       * q.shape[-1] ** -0.5, axis=-1).astype(v.dtype)
        allowed = (token_pos >= state.anchor[:, None]) & state.valid[:, None]
        if segment_ids is not None and state.segment is not None:
            allowed = allowed & (segment_ids == state.segment[:, None])
        if pad_mask is not None:
            allowed = allowed & pad_mask
        delta = self.out(p @ v) * mx.tanh(self.gate).astype(hidden.dtype)
        return mx.where(allowed[..., None], delta, 0).astype(hidden.dtype)


class PredictiveStateReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, m = config.psr_dim, config.psr_slots
        self.slots = mx.random.normal((m, d)) * d ** -0.5
        self.init_proj = nn.Linear(config.dim, d, bias=False)
        self.norm = RMSNorm(d, config.norm_eps)
        self.index_query = nn.Linear(d, config.index_n_heads * config.index_head_dim, bias=False)
        self.index_weights = nn.Linear(d, config.index_n_heads, bias=False)
        self.read_query = nn.Linear(d, config.head_dim, bias=False)
        self.read_out = nn.Linear(config.head_dim, d, bias=False)
        self.blocks = [StateBlock(d, config.norm_eps, config.psr_update_scale)
                       for _ in range(config.psr_blocks)]
        self.budget_proj = nn.Linear(1, d, bias=False)
        self.value_norm = RMSNorm(d, config.norm_eps)
        self.value_head = nn.Linear(d, 3, bias=True)
        self.test_embed = nn.Embedding(config.psr_num_tests, d)
        self.test_head = nn.Linear(d, config.psr_test_classes, bias=False)

    def predictive_logits(self, states, tests):
        """U is a test ID, not a teacher state. [B,R+1,P] -> class logits."""
        u = self.test_embed(tests)
        s = self.norm(states)
        p = mx.softmax(u.astype(mx.float32) @ s.astype(mx.float32).swapaxes(-1, -2)
                       * s.shape[-1] ** -0.5, axis=-1).astype(s.dtype)
        return self.test_head(nn.silu(p @ s + u))

    def action_values(self, slots, budget):
        b = mx.full((slots.shape[0], 1), budget / self.config.psr_max_rounds, dtype=slots.dtype)
        return self.value_head(mx.mean(self.value_norm(slots), axis=1) + self.budget_proj(b))

    def _read(self, slots, memory, cos, sin, fixed_indices=None):
        cfg = self.config
        b, m, _ = slots.shape
        s = self.norm(slots)
        pos = mx.broadcast_to(memory.anchor[:, None], (b, m))
        c, sn = _cos_sin(cos, sin, pos, True)
        q = self.index_query(s).reshape(b, m, cfg.index_n_heads, cfg.index_head_dim)
        q = rope_partial(q, c, sn, cfg.rope_head_dim).astype(mx.float32)
        w = self.index_weights(s).astype(mx.float32) * (cfg.index_head_dim * cfg.index_n_heads) ** -0.5
        # Same multihead ReLU score as CSA2, independent PSR selection state.
        dots = q @ memory.keys.astype(mx.float32)[:, None, :, :].swapaxes(-1, -2)
        scores = mx.sum(mx.maximum(dots, 0) * w[..., None], axis=-2)
        scores = mx.where(memory.visible[:, None, :], scores, -1e30)
        indices, margin = exact_topk(scores, memory.visible, cfg.psr_topk)
        if fixed_indices is not None:
            indices = fixed_indices
        safe = mx.maximum(indices, 0)
        kv = memory.values[mx.arange(b)[:, None, None], safe]
        rq = self.read_query(s)
        rq = rope_partial(rq[..., None, :], c, sn, cfg.rope_head_dim)[..., 0, :]
        read_scores = mx.sum(rq.astype(mx.float32)[..., None, :] * kv.astype(mx.float32), axis=-1)
        p = masked_softmax(read_scores * cfg.head_dim ** -0.5, indices >= 0)
        obs = mx.sum(p.astype(kv.dtype)[..., None] * kv, axis=-2)
        # Return values to query-local coordinates before the learned update.
        obs = rope_partial(obs[..., None, :], c, sn, cfg.rope_head_dim, inverse=True)[..., 0, :]
        obs = self.read_out(obs)
        norm = mx.sqrt(mx.sum(mx.square(memory.keys.astype(mx.float32)), axis=-1))
        key_norm = mx.max(mx.where(memory.visible, norm, 0), axis=-1)
        ref = IndexReference(q, w, indices, margin, key_norm)
        return obs, scores, ref

    def advance(self, slots, memory, cos, sin, action=READ, fixed_indices=None):
        """One actual transition, also used for offline counterfactual rollouts."""
        scores, reference = None, None
        if action == READ:
            obs, scores, reference = self._read(slots, memory, cos, sin, fixed_indices)
            slots = slots + self.config.psr_update_scale * mx.tanh(obs)
        elif action != COMPUTE:
            raise ValueError("advance requires READ or COMPUTE")
        for block in self.blocks:
            slots = block(slots)
        return slots, scores, reference

    def retrieval_distribution(self, slots, memory, cos, sin):
        """Full read-attention teacher for text pretraining; no future targets.

        This trains the discrete indexer's scores to approximate the current
        read query. It is not an oracle for the next necessary evidence.
        """
        pos = mx.broadcast_to(memory.anchor[:, None], slots.shape[:2])
        c, sn = _cos_sin(cos, sin, pos, True)
        q = self.read_query(self.norm(slots))
        q = rope_partial(q[..., None, :], c, sn, self.config.rope_head_dim)[..., 0, :]
        score = q.astype(mx.float32) @ memory.values.astype(mx.float32).swapaxes(-1, -2)
        return mx.stop_gradient(masked_softmax(score * self.config.head_dim ** -0.5,
                                              memory.visible[:, None, :]))

    def __call__(self, initial, memory, segment, valid, cos, sin, *, rounds=None,
                 mode="fixed", policy_calibrated=False, actions=None,
                 selection_mode="adaptive", record_trace=False, distill_reads=False):
        cfg = self.config
        rounds = cfg.psr_rounds if rounds is None else rounds
        if not isinstance(rounds, int) or not 0 <= rounds <= cfg.psr_max_rounds:
            raise ValueError("thinking_rounds must be an integer within the configured budget")
        if mode not in ("fixed", "adaptive") or selection_mode not in ("adaptive", "fixed"):
            raise ValueError("invalid PSR mode or selection_mode")
        if mode == "adaptive" and (not policy_calibrated or self.training or initial.shape[0] != 1):
            raise ValueError("adaptive PSR requires eval(), batch=1 and a calibrated value policy")
        if actions is not None and (mode != "fixed" or len(actions) != rounds
                                    or any(a not in (COMPUTE, READ) for a in actions)):
            raise ValueError("fixed actions must contain one COMPUTE/READ per round")
        slots = self.init_proj(initial)[:, None, :] + self.slots[None, :, :]
        states, values, scores, indices, executed, reuse = [slots], [], [], [], [], []
        ref, first_indices = None, None
        actual_cost, scans, action = 0.0, 0, READ
        distillation = []
        for r in range(rounds):
            value = self.action_values(slots, rounds - r)
            values.append(value)
            if mode == "adaptive" and r % cfg.psr_group_size == 0:
                # One host decision per group; actual no-read/stop branches save work.
                group = min(cfg.psr_group_size, rounds - r)
                cost = mx.array([0, cfg.psr_compute_cost * group, cfg.psr_read_cost * group])
                action = int(mx.argmin(value[0].astype(mx.float32) + cfg.psr_cost_weight * cost).item())
                if action == STOP:
                    values.pop()
                    break
            elif mode == "fixed":
                action = READ if actions is None else actions[r]
            if action == READ:
                target = self.retrieval_distribution(slots, memory, cos, sin) if distill_reads else None
                slots, sc, new_ref = self.advance(slots, memory, cos, sin, READ,
                                                 first_indices if selection_mode == "fixed" else None)
                if target is not None:
                    logp = sc.astype(mx.float32) - mx.logsumexp(sc.astype(mx.float32), axis=-1, keepdims=True)
                    distillation.append(-mx.mean(mx.sum(target * logp, axis=-1)))
                possible = mx.zeros(slots.shape[:2], dtype=mx.bool_)
                if ref is not None and selection_mode == "adaptive":
                    delta = index_change_bound(ref, new_ref.query, new_ref.weights)
                    possible = (ref.margin > 2 * delta) & mx.isfinite(delta)
                ref = new_ref  # Most recent ACTUAL full scan; not adjacent unindexed drift.
                if first_indices is None:
                    first_indices = ref.indices
                scores.append(sc)
                indices.append(ref.indices)
                reuse.append(possible)
                scans += 1
                actual_cost += cfg.psr_read_cost
            else:
                slots, _, _ = self.advance(slots, memory, cos, sin, COMPUTE)
                scores.append(None)
                indices.append(None)
                reuse.append(None)
                actual_cost += cfg.psr_compute_cost
            executed.append(action)
            states.append(slots)
        used = len(executed)
        values.append(self.action_values(slots, rounds - used))
        state = ThinkingState(slots, memory.anchor, segment, valid, used, rounds - used, ref)
        trace = None
        if record_trace:
            trace = ReasoningTrace(mx.stack(states, axis=1), tuple(scores), tuple(indices),
                                   mx.stack(values, axis=1), tuple(executed), tuple(reuse),
                                   mx.array(actual_cost), scans, memory, rounds,
                                   sum(distillation) / len(distillation) if distillation else None)
        return state, trace


def _masked_ce(logits, targets):
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"target shape {targets.shape} != logits shape {logits.shape[:-1]}")
    valid = (targets >= 0) & (targets < logits.shape[-1])
    y = mx.clip(targets, 0, logits.shape[-1] - 1).astype(mx.int32)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
    return mx.sum(mx.where(valid, ce, 0)) / mx.maximum(mx.sum(valid), 1)


def psr_losses(reasoner, trace, targets):
    """Loss-only API: labels NEVER enter the reasoner's recurrent forward.

    address: [B,R,m], -1 ignores compute/padding/unsupervised slots.
    tests/results: [B,R+1,P], outcome CE supervises continuation behavior.
    values: [B,R+1,3], NaN masks unmeasured counterfactual actions. Targets are
      measured downstream terminal loss + future cost, EXCLUDING immediate
      action-group cost (the scheduler adds it). No one-step entropy labels.
    """
    cfg = reasoner.config
    losses = {}
    unknown = set(targets) - {"address", "tests", "results", "values", "terminal_tests", "terminal_results"}
    if unknown:
        raise ValueError(f"unknown PSR targets: {sorted(unknown)}")
    if "address" in targets:
        addr = targets["address"]
        b, rp1, m, _ = trace.states.shape
        if addr.shape != (b, rp1 - 1, m):
            raise ValueError("address targets must have shape [B,R,slots]")
        terms = []
        for r, sc in enumerate(trace.scores):
            if sc is not None:
                terms.append(_masked_ce(sc, addr[:, r]))
        losses["address"] = sum(terms, mx.array(0.0)) / max(len(terms), 1)
    if "tests" in targets or "results" in targets:
        if "tests" not in targets or "results" not in targets:
            raise ValueError("tests and results must be supplied together")
        tests = targets["tests"]
        if tests.shape[:2] != trace.states.shape[:2]:
            raise ValueError("tests must align with all R+1 states")
        losses["predictive"] = _masked_ce(reasoner.predictive_logits(trace.states, tests), targets["results"])
    if "terminal_tests" in targets or "terminal_results" in targets:
        if "predictive" in losses:
            raise ValueError("choose terminal-only or all-state predictive supervision")
        if "terminal_tests" not in targets or "terminal_results" not in targets:
            raise ValueError("terminal_tests and terminal_results must be supplied together")
        logits = reasoner.predictive_logits(trace.states[:, -1:], targets["terminal_tests"][:, None])[:, 0]
        losses["predictive"] = _masked_ce(logits, targets["terminal_results"])
    if "values" in targets:
        value_targets = mx.stop_gradient(targets["values"])
        if value_targets.shape != trace.values.shape:
            raise ValueError("value targets must have shape [B,R+1,3]")
        valid = mx.isfinite(value_targets)
        error = trace.values.astype(mx.float32) - mx.where(valid, value_targets, 0)
        losses["value"] = mx.sum(mx.where(valid, error * error, 0)) / mx.maximum(mx.sum(valid), 1)
    # Reporting executed cost is not a policy gradient. Value supervision is
    # what trains discrete decisions; do not differentiate a Python loop count.
    losses["cost"] = mx.stop_gradient(trace.actual_cost)
    if trace.index_distillation is not None:
        losses["index_distillation"] = trace.index_distillation
    losses["total"] = (cfg.psr_address_weight * losses.get("address", 0)
                       + cfg.psr_predictive_weight * losses.get("predictive", 0)
                       + cfg.psr_value_weight * losses.get("value", 0)
                       + cfg.psr_index_distill_weight * losses.get("index_distillation", 0)
                       + cfg.psr_cost_weight * losses["cost"])
    return losses


def counterfactual_value_targets(reasoner, trace, cos, sin, terminal_loss):
    """Offline finite-budget Bellman targets on the model's OWN states.

    terminal_loss(slots) -> per-example measured loss [B], e.g. restricted
    answer CE with ground truth. Enumerates READ/COMPUTE groups plus STOP;
    exponential in remaining groups, intended for small verified tasks only.
    Future labels are targets, never inputs to advance(). This is a realized
    sample backup; fitting over samples estimates expected decision value.
    No policy-calibrated flag is inferred from fitting these targets.
    """
    cfg = reasoner.config
    budget = trace.budget
    if budget > 8:
        raise ValueError("counterfactual enumeration is limited to eight rounds")

    def backup(slots, remaining):
        stop = terminal_loss(slots).astype(mx.float32)
        if remaining == 0:
            # No legal continuation. Only the STOP target is supervised.
            targets = mx.stack([stop, mx.full_like(stop, mx.nan), mx.full_like(stop, mx.nan)], axis=-1)
            return stop, targets
        group = min(cfg.psr_group_size, remaining)
        candidates = [stop]
        totals = [stop]
        for action, cost in ((COMPUTE, cfg.psr_compute_cost), (READ, cfg.psr_read_cost)):
            future = slots
            for _ in range(group):
                future, _, _ = reasoner.advance(future, trace.memory, cos, sin, action)
            value, _ = backup(future, remaining - group)
            candidates.append(value)
            totals.append(value + cfg.psr_cost_weight * cost * group)
        return mx.min(mx.stack(totals, axis=-1), axis=-1), mx.stack(candidates, axis=-1)

    targets = []
    for r in range(trace.states.shape[1]):
        _, value = backup(mx.stop_gradient(trace.states[:, r]), budget - r)
        targets.append(mx.stop_gradient(value))
    return mx.stack(targets, axis=1)
