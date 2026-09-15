"""Shared-address iterative evidence reading, exact parallelism over token queries.

Each read result replaces the state and becomes the next query. K/V stay fixed
through the depth loop. This is not a temporal pipeline or a scan approximation.
"""

import mlx.core as mx
from mlx import nn

from .norms import rms_unit


def unit(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)


class IterativeEvidenceReader(nn.Module):
    """Q/K share an address map; values must use the input-state coordinate space."""

    def __init__(self, dim):
        super().__init__()
        self.address = nn.Linear(dim, dim, bias=False)

    def project_keys(self, keys):
        return unit(keys.astype(mx.float32) @ self.address.weight.astype(mx.float32).T)

    def read_once(self, state, keys, values, visible=None):
        query = self.project_keys(state)
        scores = 8 * (query @ keys.swapaxes(-1, -2))
        if visible is not None:
            scores = mx.where(visible, scores, -1e30)
        weights = mx.softmax(scores, axis=-1)
        if visible is not None:
            weights = mx.where(visible, weights, 0)
        return weights @ values.astype(mx.float32), weights

    def iterate(
        self,
        initial,
        keys,
        values,
        steps,
        visible=None,
        fixed_query=False,
        donor=None,
        return_trace=False,
    ):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Thinking steps must be a positive integer")
        state = initial.astype(mx.float32)
        trace = []
        for step in range(steps):
            state, weights = self.read_once(
                initial if fixed_query else state, keys, values, visible
            )
            if return_trace:
                trace.append(weights)
            if donor is not None and step == 0:
                state = state[donor]
        return (state, mx.stack(trace, axis=1)) if return_trace else state

    def __call__(self, initial, keys, values, steps, **kwargs):
        return self.iterate(initial, self.project_keys(keys), values, steps, **kwargs)


class IterativeThinking(nn.Module):
    """CED adapter; the encoder must learn addresses/values from natural text.

    The structured probe supplies typed entity fields; this adapter does not.
    Its language-model quality therefore requires separate validation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.dim, config.thinking_dim, bias=False)
        self.value = nn.Linear(config.dim, config.thinking_dim, bias=False)
        self.reader = IterativeEvidenceReader(config.thinking_dim)
        self.output = (
            None
            if config.thinking_arch == "ced_iterative_tied_v1"
            else nn.Linear(config.thinking_dim, config.dim, bias=False)
        )

    def __call__(
        self,
        encoder,
        segment_ids=None,
        pad_mask=None,
        cache=None,
        start_pos=0,
        intervention="normal",
    ):
        from .thinking import cache_signature

        if intervention not in ("normal", "off", "reset", "swap"):
            raise ValueError("Unknown thinking intervention")
        b, t, _ = encoder.shape
        if t < 1:
            raise ValueError("Thinking requires a nonempty sequence")
        if cache is not None:
            if segment_ids is not None or pad_mask is not None:
                raise ValueError(
                    "Cached thinking requires unpadded single-document inputs"
                )
            if cache.tokens != start_pos or cache.signature not in (
                None,
                cache_signature(self.config),
            ):
                raise ValueError(
                    "Thinking cache clock/signature mismatch; use a fresh prefill"
                )
            if start_pos and (
                cache.memory is None or cache.memory.shape[1] != start_pos
            ):
                raise ValueError("Thinking cache is missing complete evidence")
        elif start_pos != 0:
            raise ValueError("Uncached thinking requires a full prefix")
        good = (
            mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_)
        )
        docs = mx.zeros((b, t), mx.int32) if segment_ids is None else segment_ids
        resets = mx.concatenate(
            [
                mx.ones((b, 1), mx.bool_),
                (docs[:, 1:] != docs[:, :-1]) | ~good[:, 1:] | ~good[:, :-1],
            ],
            axis=1,
        )
        runs = mx.cumsum(resets.astype(mx.int32), axis=1)
        base = rms_unit(encoder)
        local = self.input(base)
        keys = self.reader.project_keys(local)
        values = self.value(base).astype(mx.float32)
        memory = mx.concatenate([keys, values], axis=-1)
        if cache is not None and cache.memory is not None:
            memory = mx.concatenate([cache.memory, memory], axis=1)
        w = self.config.thinking_dim
        keys, values = memory[..., :w], memory[..., w:]
        positions = start_pos + mx.arange(t)
        visible = mx.arange(memory.shape[1])[None, None] <= positions[None, :, None]
        if cache is None:
            visible = (
                visible
                & good[:, :, None]
                & good[:, None, :]
                & (runs[:, :, None] == runs[:, None, :])
            )
        donor = None
        if intervention == "swap":
            if b < 2 or segment_ids is not None or pad_mask is not None:
                raise ValueError(
                    "Thinking swap needs two unpadded single-document rows"
                )
            donor = mx.concatenate([mx.arange(1, b), mx.array([0])])
        state = self.reader.iterate(
            local,
            keys,
            values,
            self.config.thinking_steps,
            visible=visible,
            fixed_query=intervention == "reset",
            donor=donor,
        )
        # A bounded residual interface preserves the original full-token backbone.
        writeback = self.input.weight if self.output is None else self.output.weight.T
        projected = rms_unit(state @ writeback.astype(mx.float32))
        amplitude = mx.sqrt(
            mx.mean(encoder.astype(mx.float32) ** 2, axis=-1, keepdims=True)
            + self.config.norm_eps
        )
        signal = self.config.thinking_scale * amplitude * projected
        signal = mx.where(good[..., None], signal, 0).astype(encoder.dtype)
        if intervention == "off":
            signal = mx.zeros_like(signal)
        if cache is not None:
            cache.memory, cache.previous = memory, None
            cache.tokens, cache.signature = start_pos + t, cache_signature(self.config)
        return signal
