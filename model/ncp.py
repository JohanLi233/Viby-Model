"""NCP core port: pooled causal concepts, product codebooks and shifted feedback.

Sources: arXiv:2609.10715, Sec. 2.2-2.4/3; released HF modeling_ncp_olmo3.py
at 155259f2a1ccfd56e449e323360686aad5244d1c. Viby retains its token backbone
and mHC. Hierarchical residual routing is intentionally not part of this port.
"""

from dataclasses import dataclass
import mlx.core as mx
from mlx import nn
from .norms import RMSNorm
from .rope import precompute_freqs_cis, rope_partial


def safe_softmax(scores, mask):
    p = mx.softmax(mx.where(mask, scores.astype(mx.float32), -1e30), axis=-1) * mask
    denominator = mx.sum(p, axis=-1, keepdims=True)
    return p / mx.where(denominator > 0, denominator, 1.0)


class ConceptBlock(nn.Module):
    """Dense causal concept attention, QK RMSNorm and OLMo-style post norms."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.ncp_heads
        self.head_dim = d // self.heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.attn_norm = RMSNorm(d, cfg.norm_eps)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.up = nn.Linear(d, 2 * cfg.ncp_inter_dim, bias=False)
        self.down = nn.Linear(cfg.ncp_inter_dim, d, bias=False)

    def __call__(self, x, mask, cos, sin):
        b, n, d = x.shape

        def split(y):
            return y.reshape(b, n, self.heads, self.head_dim)

        q = self.q_norm(split(self.q(x)))
        k = self.k_norm(split(self.k(x)))
        c, s = cos[:n][None, :, None, :], sin[:n][None, :, None, :]
        q = rope_partial(q, c, s, self.head_dim).transpose(0, 2, 1, 3)
        k = rope_partial(k, c, s, self.head_dim).transpose(0, 2, 1, 3)
        v = split(self.v(x)).transpose(0, 2, 1, 3)
        score = (
            q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
        ) * self.head_dim**-0.5
        p = safe_softmax(score, mask[:, None])
        attention = (p.astype(v.dtype) @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        x = x + self.attn_norm(self.out(attention))
        a, bias = mx.split(self.up(x), 2, axis=-1)
        return x + self.ffn_norm(self.down(nn.silu(a) * bias))


@dataclass
class NCPResult:
    signal: mx.array
    active: mx.array
    predicted: mx.array
    targets: mx.array
    valid: mx.array
    ncp_loss: mx.array
    vq_loss: mx.array
    metrics: mx.array


@dataclass
class NCPRecomputeCache:
    """Reference inference: retains tokens and recomputes the full causal prefix."""

    input_ids: mx.array
    segment_ids: object = None
    start_pos: int = 0


class NextConceptModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        d, g = cfg.dim, cfg.ncp_codebooks
        self.input_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.layers = [ConceptBlock(cfg) for _ in range(cfg.ncp_layers)]
        self.final_norm = RMSNorm(d, cfg.norm_eps)
        self.prediction_heads = [
            nn.Linear(d, cfg.ncp_codebook_size, bias=True) for _ in range(g)
        ]
        self.codebooks = (
            mx.random.normal((g, cfg.ncp_codebook_size, d // g)) * (d // g) ** -0.5
        )
        self.feedback_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.feedback_scale = mx.array(cfg.ncp_fusion_init)
        self.freq_cos, self.freq_sin = precompute_freqs_cis(
            d // cfg.ncp_heads,
            cfg.max_seq_len // cfg.ncp_chunk_size + 1,
            0,
            cfg.ncp_rope_theta,
            1.0,
            32,
            1,
        )
        self.freeze(recurse=False, keys=["freq_cos", "freq_sin"])

    def reset_concept_rope(self):
        cfg = self.config
        dtype = self.freq_cos.dtype
        c, s = precompute_freqs_cis(
            cfg.dim // cfg.ncp_heads,
            cfg.max_seq_len // cfg.ncp_chunk_size + 1,
            0,
            cfg.ncp_rope_theta,
            1.0,
            32,
            1,
        )
        self.freq_cos, self.freq_sin = c.astype(dtype), s.astype(dtype)

    def predict(self, concepts, valid, segments):
        n = concepts.shape[1]
        reach = (mx.arange(n)[None, :] <= mx.arange(n)[:, None])[None]
        mask = reach & valid[:, None, :] & valid[:, :, None]
        if segments is not None:
            mask = mask & (segments[:, :, None] == segments[:, None, :])
        hidden = concepts
        for layer in self.layers:
            hidden = layer(hidden, mask, self.freq_cos, self.freq_sin)
        hidden = self.final_norm(hidden)
        logits = mx.stack([head(hidden) for head in self.prediction_heads], axis=2)
        weights = (
            mx.softmax(logits.astype(mx.float32), axis=-1)
            if self.config.ncp_merge == "softmax"
            else logits.astype(mx.float32)
        )
        # [B,N,G,K] @ [G,K,D/G]. Never replace by hard argmax on the NTP path.
        predicted = mx.einsum(
            "bngk,gkd->bngd", weights, self.codebooks.astype(mx.float32)
        )
        return predicted.reshape(*concepts.shape).astype(concepts.dtype)

    def losses(self, concepts, predicted, valid, segments):
        cfg = self.config
        b, n, d = concepts.shape
        g = cfg.ncp_codebooks
        target = mx.stop_gradient(concepts).astype(mx.float32)
        parts = target.reshape(b, n, g, d // g)
        books = self.codebooks.astype(mx.float32)
        distance = (
            mx.sum(parts**2, -1, keepdims=True)
            + mx.sum(books**2, -1)[None, None]
            - 2 * mx.einsum("bngd,gkd->bngk", parts, books)
        )
        indices = mx.stop_gradient(mx.argmin(distance, axis=-1).astype(mx.int32))
        selected = books[mx.arange(g)[None, None, :], indices]
        vq_per = mx.sum((selected - parts) ** 2, axis=(-1, -2))
        vq_scale = d if cfg.ncp_loss_reduction == "mean" else g
        vq = mx.sum(vq_per * valid) / mx.maximum(mx.sum(valid) * vq_scale, 1)
        pair = valid[:, :-1] & valid[:, 1:]
        if segments is not None:
            pair = pair & (segments[:, :-1] == segments[:, 1:])
        error = mx.sum((predicted[:, :-1].astype(mx.float32) - target[:, 1:]) ** 2, -1)
        scale = d if cfg.ncp_loss_reduction == "mean" else 1
        ncp = mx.sum(error * pair) / mx.maximum(mx.sum(pair) * scale, 1)
        # Counts are diagnostics, not another loss or a codebook-collapse penalty.
        counts = mx.sum(
            (indices[..., None] == mx.arange(cfg.ncp_codebook_size))
            * valid[..., None, None],
            axis=(0, 1),
        )
        usage = mx.mean((counts > 0).astype(mx.float32))
        return (
            ncp,
            vq,
            mx.stack(
                [
                    mx.sum(valid).astype(mx.float32),
                    mx.sum(pair).astype(mx.float32),
                    usage,
                    mx.sum(mx.sum(target**2, -1) * valid)
                    / mx.maximum(mx.sum(valid) * d, 1),
                    mx.sum(mx.sum(predicted.astype(mx.float32) ** 2, -1) * valid)
                    / mx.maximum(mx.sum(valid) * d, 1),
                ]
            ),
        )

    def __call__(
        self, encoder_hidden, segment_ids=None, pad_mask=None, compute_loss=False
    ):
        cfg = self.config
        b, t, d = encoder_hidden.shape
        k = cfg.ncp_chunk_size
        n = t // k
        zero = mx.array(0.0)
        if n == 0:
            empty = mx.zeros((b, 0, d), dtype=encoder_hidden.dtype)
            return NCPResult(
                mx.zeros_like(encoder_hidden),
                mx.zeros((b, t), mx.bool_),
                empty,
                empty,
                mx.zeros((b, 0), mx.bool_),
                zero,
                zero,
                mx.zeros((8,)),
            )
        pooled = mx.mean(encoder_hidden[:, : n * k].reshape(b, n, k, d), axis=2)
        concepts = self.input_norm(pooled)
        valid = mx.ones((b, n), mx.bool_)
        segments = None
        if segment_ids is not None:
            grouped = segment_ids[:, : n * k].reshape(b, n, k)
            segments = grouped[:, :, 0]
            valid = valid & mx.all(grouped == segments[..., None], axis=-1)
        if pad_mask is not None:
            valid = valid & mx.all(
                pad_mask[:, : n * k].reshape(b, n, k).astype(mx.bool_), axis=-1
            )
        predicted = self.predict(concepts, valid, segments)
        # Official released implementation: first prediction at token k-1,
        # where x[0:k] is known and its logit predicts x[k].
        group = (mx.arange(t) + 1) // k - 1
        safe = mx.clip(group, 0, n - 1)
        active = (group[None] >= 0) & mx.take(valid, safe, axis=1)
        if segments is not None:
            active = active & (mx.take(segments, safe, axis=1) == segment_ids)
        if pad_mask is not None:
            active = active & pad_mask.astype(mx.bool_)
        feedback = self.feedback_norm(mx.take(predicted, safe, axis=1))
        signal = mx.where(active[..., None], self.feedback_scale * feedback, 0).astype(
            encoder_hidden.dtype
        )
        ncp, vq, extra = (
            self.losses(concepts, predicted, valid, segments)
            if compute_loss
            else (zero, zero, mx.zeros((5,)))
        )
        metrics = mx.concatenate(
            [mx.stack([ncp, vq, mx.mean(active.astype(mx.float32))]), extra]
        )
        return NCPResult(signal, active, predicted, concepts, valid, ncp, vq, metrics)
