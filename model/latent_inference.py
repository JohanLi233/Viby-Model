"""Task-anchored short-block latent inference, independent of DPR/PSR.

Only token marginal likelihood trains this module. The frozen-feature probe
uses FP32 branch parameters and checkpointed vocabulary computations. Candidate
membership is inherited from CED (up to 64 selected + 64 recent token keys),
deduplicated and revalidated; the two reads only reweight this fixed set.
"""

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
import numpy as np


@dataclass(frozen=True)
class LatentConfig:
    dim: int = 1024
    vocab: int = 6400
    rank: int = 64
    particles: int = 4
    horizon: int = 4
    rounds: int = 2
    candidates: int = 128
    epsilon: float = 1e-3
    version: str = "task_latent_v1"

    def __post_init__(self):
        if min(self.dim, self.vocab, self.rank, self.particles, self.horizon) < 1:
            raise ValueError("latent dimensions must be positive")
        if self.rounds != 2 or self.candidates != 128:
            raise ValueError("this experiment fixes two reads and 128 candidates")

    def to_dict(self):
        return asdict(self)


def rms(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + 1e-6)


def log_softmax(x):
    x = x.astype(mx.float32)
    return x - mx.logsumexp(x, -1, keepdims=True)


def boundary_candidates(shared, attention, batch, length, positions, decode=False):
    """Export existing token-resolution CED selection without changing dispatch."""
    if attention.ratio != 1:
        raise ValueError("latent evidence requires a ratio=1 CED boundary")
    if decode:
        idx = shared.topk_idx
        idx = mx.where(idx >= attention.window_size, idx - attention.window_size, -1)
    elif shared.sparse_selection is not None:
        ratio, (indices, lengths) = shared.sparse_selection
        if ratio != 1:
            raise ValueError("stale boundary selection")
        width = min(indices.shape[-1], 64)
        idx = indices[:, :width].reshape(batch, length, width)
        idx = mx.where(mx.arange(width) < lengths.reshape(batch, length, 1), idx, -1)
    elif shared.keep_mask is not None:
        idx = attention._bounded_visible_indices(shared.keep_mask, 64)
    else:
        raise ValueError("CED boundary did not expose its selection")
    idx = idx[..., :64]
    idx = mx.pad(idx, [(0, 0), (0, 0), (0, 64 - idx.shape[-1])], constant_values=-1)
    recent = positions[..., None] - mx.arange(64)
    idx = mx.sort(mx.concatenate([idx, mx.maximum(recent, -1)], -1), -1)
    unique = mx.concatenate([mx.ones((*idx.shape[:-1], 1), mx.bool_), idx[..., 1:] != idx[..., :-1]], -1)
    return mx.stop_gradient(mx.where(unique & (idx <= positions[..., None]), idx, -1))


def block_plan(input_ids, labels, loss_mask, segments, *, horizon=4, eos_id=2, pad_id=0):
    """Host layout; every valid NTP label is scored exactly once, tails included.

    Each row is a fresh context. Invalid labels terminate blocks; EOS is allowed
    as the final label, never as an input linking two documents. PAD is excluded.
    Returns flattened token rows with -1 padding and a causal evidence mask.
    """
    x, y, mask, seg = map(np.asarray, (input_ids, labels, loss_mask, segments))
    if x.ndim != 1 or any(a.shape != x.shape for a in (y, mask, seg)):
        raise ValueError("block_plan expects one sequence with matching arrays")
    valid = (mask > 0) & (x != pad_id) & (y != pad_id)
    groups, current = [], []
    for t in range(len(x)):
        if current and (not valid[t] or seg[t] != seg[current[-1]] or x[t] == eos_id):
            groups.append(current)
            current = []
        if valid[t]:
            current.append(t)
            if len(current) == horizon or y[t] == eos_id:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    rows = np.full((len(groups), horizon), -1, np.int32)
    for n, group in enumerate(groups):
        rows[n, :len(group)] = group
    return rows


def filter_nll(component_logp, log_prior, valid, *, static=False):
    """[blocks,k,M] observed-token log probabilities -> causal token NLL.

    The cumulative sum is EXCLUSIVE. No posterior or filtering weight is
    detached. Summing these token losses equals the exact block marginal NLL.
    """
    observed = mx.where(valid[..., None], component_logp, 0.0)
    before = mx.concatenate([mx.zeros_like(observed[:, :1]), mx.cumsum(observed, axis=1)[:, :-1]], 1)
    weights = log_prior[:, None, :] if static else log_softmax(log_prior[:, None, :] + before)
    return mx.where(valid, -mx.logsumexp(weights + observed, -1), 0.0)


class ShortBlockLatent(nn.Module):
    def __init__(self, config=LatentConfig()):
        super().__init__()
        self.config = config
        d, r, m = config.dim, config.rank, config.particles
        self.anchor = nn.Linear(d, r, bias=False)
        self.slots = mx.random.normal((m, r)) * r**-0.5
        self.key = nn.Linear(d, r, bias=False)
        self.value = nn.Linear(d, r, bias=False)
        self.query = nn.Linear(r, r, bias=False)
        self.read_out = nn.Linear(r, r, bias=False)
        self.ffn_up = nn.Linear(r, 4 * r, bias=False)
        self.ffn_down = nn.Linear(2 * r, r, bias=False)
        self.prior = nn.Linear(r, 1, bias=False)
        self.gate = nn.Linear(d, r, bias=False)
        self.output = nn.Linear(r, config.vocab, bias=False)
        w = self.output.weight.astype(mx.float32)
        self.output.weight = w / mx.maximum(mx.linalg.norm(w, axis=-1, keepdims=True), 1e-12) * (config.epsilon / r**0.5)

    def plan(self, anchors, evidence, candidates, visible, *, fixed_second_query=False, projected=None):
        """Two shared-weight reads. K/V are projected once per context.

        Evidence is the token-resolution mHC input to the CED boundary. Its
        contextual representation already carries backbone positional encoding;
        this first probe adds no separate latent RoPE or position embeddings.
        """
        safe = mx.maximum(candidates, 0)
        all_k, all_v = (self.key(evidence), self.value(evidence)) if projected is None else projected
        k, v = all_k[safe], all_v[safe]
        s = self.anchor(anchors)[:, None, :] + self.slots[None]
        first_query = self.query(rms(s))
        for u in range(self.config.rounds):
            q = first_query if u == 0 or fixed_second_query else self.query(rms(s))
            scores = (q @ k.swapaxes(-1, -2)) * self.config.rank**-0.5
            p = mx.softmax(mx.where(visible[:, None, :], scores, -1e30), -1) * visible[:, None, :]
            p = p / mx.maximum(p.sum(-1, keepdims=True), 1e-20)
            s = s + self.read_out(p @ v)
            up, gate = mx.split(self.ffn_up(rms(s)), 2, -1)
            s = s + self.ffn_down(nn.silu(gate) * up)
        return rms(s), log_softmax(self.prior(s)[..., 0])

    def component_logprobs(self, hidden, base_logits, slots):
        g = mx.sigmoid(self.gate(hidden))
        correction = self.output(g[..., None, :] * slots[:, None, :, :])
        return log_softmax(base_logits[..., None, :].astype(mx.float32) + correction)

    def observed_logprobs(self, hidden, base_logits, slots, labels):
        lp = self.component_logprobs(hidden, base_logits, slots)
        return mx.take_along_axis(lp, mx.stop_gradient(labels[..., None, None]), axis=-1)[..., 0]

    def token_losses(self, features, head_weight, *, static=False, fixed_second_query=False, rematerialize=True):
        z, prior = self.plan(features["a"], features["e"], features["candidates"], features["visible"], fixed_second_query=fixed_second_query)
        # At most 64 blocks (256 tokens) per vocabulary computation; checkpoint
        # drops softmax activations and recomputes them during backward.
        losses, base_losses = [], []
        for begin in range(0, z.shape[0], 64):
            end = begin + 64
            rows = features["rows"][begin:end]
            hidden = features["d"][mx.maximum(rows, 0)]
            labels = features["labels"][mx.maximum(rows, 0)]
            base = (hidden @ head_weight.T).astype(mx.float32)
            observed_fn = checkpoint(self, self.observed_logprobs) if rematerialize else self.observed_logprobs
            obs = observed_fn(hidden, base, z[begin:end], labels)
            valid = rows >= 0
            losses.append(filter_nll(obs, prior[begin:end], valid, static=static))
            base_losses.append(mx.where(valid, nn.losses.cross_entropy(base, labels, reduction="none"), 0.0))
        return mx.concatenate(losses), mx.concatenate(base_losses)

    def __call__(self, features, head_weight, *, static=False):
        nll, _ = self.token_losses(features, head_weight, static=static)
        return nll.sum() / mx.maximum((features["rows"] >= 0).sum(), 1)


@dataclass
class LatentFilterState:
    """A prompt boundary does not reset this state. Only EOS/block completion does."""
    slots: mx.array
    log_weights: mx.array
    count: int = 0
    pending_logprobs: mx.array | None = None

    def predict(self, branch, hidden, base_logits):
        lp = branch.component_logprobs(hidden[:, None], base_logits[:, None], self.slots)[:, 0]
        self.pending_logprobs = lp
        return mx.logsumexp(self.log_weights[..., None] + lp, axis=-2)

    def observe(self, token_ids, *, horizon=4, eos_id=2):
        if self.pending_logprobs is None:
            raise ValueError("predict must precede observe")
        observed = mx.take_along_axis(self.pending_logprobs, token_ids[:, None, None], -1)[..., 0]
        self.log_weights = log_softmax(self.log_weights + observed)
        self.count += 1
        self.pending_logprobs = None
        return (token_ids == eos_id) | (self.count >= horizon)


def prepare_features(a, evidence, hidden, candidates, x, labels, mask, segments, config, *, eos_id=2, pad_id=0):
    rows = block_plan(x, labels, mask, segments, horizon=config.horizon, eos_id=eos_id, pad_id=pad_id)
    if len(rows) == 0:
        raise ValueError("no valid next-token labels")
    starts = mx.array(rows[:, 0])
    seg, tokens = mx.array(segments), mx.array(x)
    selected = candidates[starts]
    safe = mx.clip(selected, 0, len(x) - 1)
    visible = (selected >= 0) & (selected <= starts[:, None]) & (selected < len(x))
    visible &= (seg[safe] == seg[starts, None]) & (tokens[safe] != pad_id)
    # Even callers with recycled segment ids may not read past an EOS reset.
    docs = mx.cumsum((tokens == eos_id).astype(mx.int32)) - (tokens == eos_id)
    visible &= docs[safe] == docs[starts, None]
    return dict(a=a[starts], e=evidence, d=hidden, candidates=safe, visible=visible, rows=mx.array(rows), labels=mx.array(labels))


class LatentRuntime:
    """Native single-sequence CED prefill/decode with persistent latent filtering.

    Outputs are normalized log PROBABILITIES, suitable for sampling as logits.
    Continuous batching, speculative rewind and cache sharing are not supported.
    EOS ends a document; the next input starts with fresh CED and latent caches.
    """

    def __init__(self, base, branch):
        self.base, self.branch = base, branch
        if (base.config.dim, base.config.vocab_size) != (branch.config.dim, branch.config.vocab):
            raise ValueError("base/branch dimensions differ")
        self.reset()

    def reset(self):
        from .cache import VibyCache
        self.cache = VibyCache(self.base.config, 1)
        self.filter = None
        self.projected = None
        self.after_eos = False

    def _consume(self, token, anchor, hidden, candidates, position):
        done = False
        if self.filter is not None:
            done = bool(self.filter.observe(token, horizon=self.branch.config.horizon, eos_id=self.base.config.eos_token_id)[0])
        if self.filter is None or done:
            safe = mx.clip(candidates, 0, self.projected[0].shape[0] - 1)
            valid = (candidates >= 0) & (candidates <= position)
            z, prior = self.branch.plan(anchor, None, safe, valid, projected=self.projected)
            self.filter = LatentFilterState(z, prior)
        self.after_eos = int(token[0]) == self.base.config.eos_token_id
        return self.filter.predict(self.branch, hidden.astype(mx.float32), self.base.logits(hidden).astype(mx.float32))

    def prefill(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("latent prefill requires one nonempty sequence")
        if bool(mx.any(input_ids == self.base.config.pad_token_id)):
            raise ValueError("native latent cache does not accept PAD")
        self.reset()
        if bool(mx.any(input_ids == self.base.config.eos_token_id)):
            return mx.stack([self.decode_step(input_ids[:, t]) for t in range(input_ids.shape[1])], 1)
        result = self.base.model(input_ids, cache=self.cache, collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
        hidden, new_prev, (a, e, candidates) = result[0], result[2], result[-1]
        self.projected = self.branch.key(e[0]), self.branch.value(e[0])
        outputs = [self._consume(input_ids[:, t], a[:, t], hidden[:, t], candidates[:, t], t) for t in range(input_ids.shape[1])]
        self.cache.start_pos = input_ids.shape[1]
        self.cache.engram_prev = new_prev
        return mx.stack(outputs, 1)

    def decode_step(self, token_ids):
        if token_ids.shape != (1,) or int(token_ids[0]) == self.base.config.pad_token_id:
            raise ValueError("native latent decode requires one non-PAD token")
        if self.after_eos:
            self.reset()
        pos = self.cache.start_pos
        if pos >= self.base.config.max_seq_len:
            raise ValueError("latent cache capacity exceeded; start a fresh context")
        self.cache.decode_max_pos = pos + 1
        result = self.base.model(token_ids[:, None], start_pos=pos, cache=self.cache, decode=pos > 0, prev_tokens=self.cache.engram_prev, collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
        hidden, new_prev, (a, e, candidates) = result[0], result[2], result[-1]
        projected = self.branch.key(e[0]), self.branch.value(e[0])
        self.projected = projected if self.projected is None else tuple(mx.concatenate([old, new], 0) for old, new in zip(self.projected, projected))
        out = self._consume(token_ids, a[:, 0], hidden[:, 0], candidates[:, 0], pos)
        self.cache.start_pos += 1
        self.cache.engram_prev = new_prev
        return out
