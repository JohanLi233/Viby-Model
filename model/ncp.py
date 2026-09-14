"""CED-aware next-concept prediction; only observed concepts enter shared KV.

Losses use per-dimension means. This is a shared-memory research architecture,
not a reproduction of the paper's per-layer concept Transformer.
"""

import mlx.core as mx
from mlx import nn

from .norms import RMSNorm
from .recurrent import build_anchor_plan, hold_anchors


NCP_METRICS = (
    "loss",
    "vq_loss",
    "groups",
    "pairs",
    "gate",
    "concept_rms",
    "concept_variance",
    "sample_mean_variance",
    "valid_samples",
    "pool_norm_weight_rms",
    "target_energy",
    "zero_mse",
    "previous_mse",
    "relative_mse_zero",
    "relative_mse_previous",
    "encoder_rms",
    "feedback_rms",
    "feedback_ratio",
)


def concept_diagnostics(concepts, predictions, valid, adjacent, pool_weight):
    """Detached per-microbatch statistics; all baselines use identical target pairs.

    concept_variance pools valid groups; sample_mean_variance compares per-row
    concept means (also reflects differing document contents/lengths).
    Empty populations report zero, with explicit group/pair/sample counts.
    """
    c = mx.stop_gradient(concepts.astype(mx.float32))
    p = mx.stop_gradient(predictions.astype(mx.float32))
    mask = valid.astype(mx.float32)
    count = mx.maximum(mx.sum(mask), 1)
    energy = mx.sum(mx.mean(c * c, axis=-1) * mask) / count
    mean = mx.sum(c * mask[..., None], axis=(0, 1)) / count
    variance = mx.maximum(energy - mx.mean(mean * mean), 0)
    row_count = mx.sum(mask, axis=1)
    rows = row_count > 0
    nrows = mx.sum(rows).astype(mx.float32)
    row_mean = mx.sum(c * mask[..., None], axis=1) / mx.maximum(row_count[:, None], 1)
    grand = mx.sum(row_mean * rows[:, None], axis=0) / mx.maximum(nrows, 1)
    between = mx.sum(mx.mean((row_mean - grand) ** 2, axis=-1) * rows) / mx.maximum(
        nrows, 1
    )
    pair_count = mx.maximum(mx.sum(adjacent), 1)

    def pair_mean(values):
        return mx.sum(mx.where(adjacent, values, 0)) / pair_count

    target_energy = pair_mean(mx.mean(c[:, 1:] ** 2, axis=-1))
    persistence = pair_mean(mx.mean((c[:, :-1] - c[:, 1:]) ** 2, axis=-1))
    prediction = pair_mean(mx.mean((p[:, :-1] - c[:, 1:]) ** 2, axis=-1))
    return mx.stack(
        [
            mx.sqrt(energy),
            variance,
            between,
            nrows,
            mx.sqrt(mx.mean(mx.stop_gradient(pool_weight.astype(mx.float32)) ** 2)),
            target_energy,
            target_energy,
            persistence,
            prediction / mx.maximum(target_energy, 1e-12),
            prediction / mx.maximum(persistence, 1e-12),
        ]
    )


def cache_signature(config):
    return tuple(
        getattr(config, name)
        for name in (
            "ncp_arch",
            "ncp_stride",
            "ncp_layers",
            "ncp_memory_dim",
            "ncp_heads",
            "ncp_groups",
            "ncp_codes",
            "dim",
        )
    )


def _rotate(x, positions):
    """RoPE at original completed-group token positions, for Q and shared K=V."""
    half = x.shape[-1] // 2
    angle = positions.astype(mx.float32)[..., None] * mx.exp(
        -mx.arange(half, dtype=mx.float32) * (mx.log(mx.array(10000.0)) / half)
    )
    if x.ndim == 4:
        angle = angle[:, None]
    c, s = mx.cos(angle).astype(x.dtype), mx.sin(angle).astype(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate([a * c - b * s, b * c + a * s], axis=-1)


class ConceptBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, w = config.dim, config.ncp_memory_dim
        self.heads = config.ncp_heads
        self.width = w
        self.attn_norm = RMSNorm(d, config.norm_eps)
        self.q_norm = RMSNorm(w, config.norm_eps)
        self.query = nn.Linear(d, self.heads * w, bias=False)
        self.output = nn.Linear(self.heads * w, d, bias=False)
        self.ffn_norm = RMSNorm(d, config.norm_eps)
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.gate = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def __call__(self, x, memory, positions, visible):
        b, n, _ = x.shape
        q = self.q_norm(
            self.query(self.attn_norm(x)).reshape(b, n, self.heads, self.width)
        )
        q = _rotate(q.transpose(0, 2, 1, 3), positions)
        # Explicit FP32 softmax and zeroed invalid rows: finite even with all PAD.
        score = (
            q.astype(mx.float32) @ memory[:, None].astype(mx.float32).swapaxes(-1, -2)
        ) * self.width**-0.5
        weights = mx.softmax(mx.where(visible[:, None], score, -1e30), axis=-1)
        weights = mx.where(visible[:, None], weights, 0).astype(memory.dtype)
        # K=V stores rotated values; undo the query rotation before projection,
        # preserving relative-position semantics under document offset shifts.
        out = _rotate(weights @ memory[:, None], -positions)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, -1)
        x = x + self.output(out)
        y = self.ffn_norm(x)
        return x + self.down(nn.silu(self.gate(y)) * self.up(y))


class ConceptCache:
    """One shared KV array, incomplete observed group, and held prediction.

    No per-layer hidden/KV history, predicted-history entries, or token prefix.
    Native cache API currently uses a common clock and unpadded single document.
    """

    def row_copy(self, row=0):
        result = ConceptCache()
        result.tokens = self.tokens
        result.signature = self.signature
        for field in ("memory", "pending", "prediction"):
            value = getattr(self, field)
            setattr(
                result, field, None if value is None else mx.array(value[row : row + 1])
            )
        return result

    def nbytes(self):
        return sum(
            int(x.nbytes)
            for x in (self.memory, self.pending, self.prediction)
            if x is not None
        )

    def __init__(self):
        self.memory = None
        self.pending = None
        self.prediction = None
        self.tokens = 0
        self.signature = None


class NextConceptPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pool_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.memory_project = nn.Linear(config.dim, config.ncp_memory_dim, bias=False)
        self.memory_norm = RMSNorm(config.ncp_memory_dim, config.norm_eps)
        self.layers = [ConceptBlock(config) for _ in range(config.ncp_layers)]
        self.predict_norm = RMSNorm(config.dim, config.norm_eps)
        self.predict = nn.Linear(
            config.dim, config.ncp_groups * config.ncp_codes, bias=False
        )
        self.codebook = (
            mx.random.normal(
                (config.ncp_groups, config.ncp_codes, config.dim // config.ncp_groups)
            )
            * 0.02
        )
        self.feedback_norm = RMSNorm(config.dim, config.norm_eps)
        self.feedback = nn.Linear(config.dim, config.dim, bias=False)
        self.feedback_gate = mx.zeros((1,))

    def project_memory(self, concepts, positions):
        return _rotate(self.memory_norm(self.memory_project(concepts)), positions)

    def predict_next(self, concepts, memory, positions, visible):
        state = concepts
        for layer in self.layers:
            state = layer(state, memory, positions, visible)
        cfg = self.config
        probabilities = mx.softmax(
            self.predict(self.predict_norm(state))
            .astype(mx.float32)
            .reshape(*state.shape[:2], cfg.ncp_groups, cfg.ncp_codes),
            axis=-1,
        )
        # Grouped weighted codewords remain differentiable to both predictor/table.
        prediction = mx.einsum(
            "bmgv,gvs->bmgs", probabilities, self.codebook.astype(mx.float32)
        )
        return prediction.reshape(*state.shape[:2], cfg.dim).astype(concepts.dtype)

    def auxiliary_losses(self, concepts, predictions, valid, adjacent):
        cfg = self.config
        target = mx.stop_gradient(concepts.astype(mx.float32))
        zero = mx.array(0.0, mx.float32)
        ncp = zero
        if concepts.shape[1] > 1:
            err = mx.mean(
                mx.square(predictions[:, :-1].astype(mx.float32) - target[:, 1:]),
                axis=-1,
            )
            ncp = mx.sum(mx.where(adjacent, err, 0)) / mx.maximum(mx.sum(adjacent), 1)
        pieces = target.reshape(
            *target.shape[:2], cfg.ncp_groups, cfg.dim // cfg.ncp_groups
        )
        book = self.codebook.astype(mx.float32)
        # Avoid [B,M,G,V,S] distance intermediates.
        distance = (
            mx.sum(pieces * pieces, axis=-1)[..., None]
            + mx.sum(book * book, axis=-1)[None, None]
            - 2 * mx.einsum("bmgs,gvs->bmgv", pieces, book)
        )
        nearest = mx.stop_gradient(mx.argmin(distance, axis=-1))
        selected = book[mx.arange(cfg.ncp_groups)[None, None, :], nearest]
        error = mx.mean(mx.square(selected - pieces), axis=(-1, -2))
        vq = mx.sum(mx.where(valid, error, 0)) / mx.maximum(mx.sum(valid), 1)
        return ncp, vq

    def __call__(
        self, encoder, segment_ids=None, pad_mask=None, cache=None, start_pos=0
    ):
        if cache is not None:
            return self.incremental(encoder, cache, start_pos)
        b, t, _ = encoder.shape
        cfg = self.config
        positions = mx.broadcast_to(mx.arange(t)[None], (b, t))
        # Contiguous run IDs also isolate repeated IDs and padding gaps.
        docs = mx.zeros((b, t), mx.int32) if segment_ids is None else segment_ids
        good = (
            mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_)
        )
        resets = mx.concatenate(
            [
                mx.ones((b, 1), mx.bool_),
                (docs[:, 1:] != docs[:, :-1]) | ~good[:, :-1] | ~good[:, 1:],
            ],
            axis=1,
        )
        runs = mx.cumsum(resets.astype(mx.int32), axis=1)
        plan = build_anchor_plan(positions, runs, good, cfg.ncp_stride)
        n = plan.indices.shape[1]
        zero = mx.array(0.0, mx.float32)
        if n == 0:
            return dict(
                signal=mx.zeros_like(encoder),
                ncp_loss=zero,
                vq_loss=zero,
                groups=zero,
                pairs=zero,
                diagnostics=mx.concatenate(
                    [
                        mx.zeros((4,)),
                        mx.sqrt(
                            mx.mean(
                                mx.stop_gradient(
                                    self.pool_norm.weight.astype(mx.float32)
                                )
                                ** 2
                            )
                        )[None],
                        mx.zeros((5,)),
                    ]
                ),
            )
        pooled = mx.zeros((b, n, cfg.dim), mx.float32)
        for offset in range(cfg.ncp_stride):
            ids = mx.maximum(plan.indices - offset, 0)
            pooled = pooled + mx.take_along_axis(
                encoder, ids[..., None], axis=1
            ).astype(mx.float32)
        concepts = self.pool_norm((pooled / cfg.ncp_stride).astype(encoder.dtype))
        concepts = mx.where(plan.valid[..., None], concepts, 0)
        memory = self.project_memory(concepts, plan.positions)
        visible = (
            plan.valid[:, :, None]
            & plan.valid[:, None, :]
            & (plan.segment_ids[:, :, None] == plan.segment_ids[:, None, :])
            & (plan.positions[:, None, :] <= plan.positions[:, :, None])
        )
        predictions = self.predict_next(concepts, memory, plan.positions, visible)
        adjacent = (
            plan.valid[:, :-1]
            & plan.valid[:, 1:]
            & (plan.segment_ids[:, :-1] == plan.segment_ids[:, 1:])
            & (plan.positions[:, 1:] - plan.positions[:, :-1] == cfg.ncp_stride)
        )
        ncp, vq = self.auxiliary_losses(concepts, predictions, plan.valid, adjacent)
        # Project at concept frequency, then hold through the next incomplete group.
        signals = self.feedback_gate * self.feedback(self.feedback_norm(predictions))
        return dict(
            signal=hold_anchors(signals, plan),
            ncp_loss=ncp,
            vq_loss=vq,
            groups=mx.sum(plan.valid),
            pairs=mx.sum(adjacent),
            diagnostics=concept_diagnostics(
                concepts, predictions, plan.valid, adjacent, self.pool_norm.weight
            ),
        )

    def incremental(self, encoder, cache, start_pos):
        if start_pos != cache.tokens:
            raise ValueError("NCP cache clock mismatch; use a fresh prefill")
        signature = cache_signature(self.config)
        if cache.signature not in (None, signature):
            raise ValueError("NCP execution changes require a fresh prefill")
        cache.signature = signature
        cfg = self.config
        b, t, d = encoder.shape
        pending = cache.pending
        old = 0 if pending is None else pending.shape[1]
        source = (
            encoder if pending is None else mx.concatenate([pending, encoder], axis=1)
        )
        n = source.shape[1] // cfg.ncp_stride
        predictions = None
        if n:
            concepts = self.pool_norm(
                mx.mean(
                    source[:, : n * cfg.ncp_stride]
                    .astype(mx.float32)
                    .reshape(b, n, cfg.ncp_stride, d),
                    axis=2,
                ).astype(encoder.dtype)
            )
            pos = mx.broadcast_to(
                (start_pos - old + mx.arange(1, n + 1) * cfg.ncp_stride - 1)[None],
                (b, n),
            )
            new_memory = self.project_memory(concepts, pos)
            previous = 0 if cache.memory is None else cache.memory.shape[1]
            memory = (
                new_memory
                if cache.memory is None
                else mx.concatenate([cache.memory, new_memory], axis=1)
            )
            visible = mx.broadcast_to(
                mx.arange(previous + n)[None, None, :]
                <= (previous + mx.arange(n))[None, :, None],
                (b, n, previous + n),
            )
            predictions = self.predict_next(concepts, memory, pos, visible)
            cache.memory = memory
        fallback = (
            mx.zeros((b, 1, d), encoder.dtype)
            if cache.prediction is None
            else cache.prediction
        )
        pool = (
            fallback
            if predictions is None
            else mx.concatenate([fallback, predictions], axis=1)
        )
        owners = (old + mx.arange(t) + 1) // cfg.ncp_stride
        held = pool[:, owners]
        signal = self.feedback_gate * self.feedback(self.feedback_norm(held))
        # Learned norm bias must never generate feedback before the first concept.
        if cache.prediction is None:
            signal = mx.where((owners > 0)[None, :, None], signal, 0)
        if predictions is not None:
            cache.prediction = predictions[:, -1:]
        cache.pending = source[:, n * cfg.ncp_stride :]
        cache.tokens += t
        zero = mx.array(0.0, mx.float32)
        return dict(
            signal=signal,
            ncp_loss=zero,
            vq_loss=zero,
            groups=mx.array(n * b),
            pairs=zero,
            diagnostics=mx.zeros(
                (10,), mx.float32
            ),  # no auxiliary diagnostic population in cache mode
        )
