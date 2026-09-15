"""Two-stage causal latent pipeline at the encoder/decoder boundary.

One shared nonlinear block; stage 2 consumes the preceding token's stage 1.
No future target, auxiliary loss, or recursive decoder rollout. The extra compact
K=V is projected from clean encoder states, independently of CED's main KV.
"""

import mlx.core as mx
from mlx import nn

from .ncp import _rotate
from .norms import rms_unit


def cache_signature(config):
    signature = (
        config.thinking_arch,
        config.dim,
        config.thinking_dim,
        config.thinking_scale,
    )
    if config.thinking_arch in ("ced_iterative_v2", "ced_iterative_tied_v1"):
        signature += (config.thinking_steps,)
    return signature


class ThinkingCache:
    def __init__(self):
        self.tokens = 0
        self.signature = None
        self.memory = None
        self.previous = None

    def row_copy(self, row=0):
        other = ThinkingCache()
        other.tokens, other.signature = self.tokens, self.signature
        for key in ("memory", "previous"):
            value = getattr(self, key)
            setattr(
                other, key, None if value is None else mx.array(value[row : row + 1])
            )
        return other

    def nbytes(self):
        return sum(int(x.nbytes) for x in (self.memory, self.previous) if x is not None)


class ThinkingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, 2 * dim, bias=False)
        self.gate = nn.Linear(dim, 2 * dim, bias=False)
        self.down = nn.Linear(2 * dim, dim, bias=False)

    def __call__(self, hidden, memory, positions, visible):
        dtype, dim = hidden.dtype, hidden.shape[-1]
        q = _rotate(rms_unit(self.query(rms_unit(hidden))), positions)
        scores = q.astype(mx.float32) @ memory.astype(mx.float32).swapaxes(-1, -2)
        weights = mx.softmax(mx.where(visible, scores * dim**-0.5, -1e30), axis=-1)
        weights = mx.where(visible, weights, 0).astype(dtype)
        read = _rotate(weights @ memory, -positions)
        state = hidden.astype(mx.float32) + self.output(read).astype(mx.float32)
        y = rms_unit(state).astype(dtype)
        delta = self.down(nn.silu(self.gate(y)) * self.up(y))
        # FP32 residual and unit-RMS carried states prevent amplitude accumulation
        # between pipeline stages; this is not a claim about Jacobian bounds.
        return rms_unit(state + delta.astype(mx.float32)).astype(dtype)


class LatentThinking(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.dim, config.thinking_dim, bias=False)
        self.block = ThinkingBlock(config.thinking_dim)
        self.output = nn.Linear(config.thinking_dim, config.dim, bias=False)

    def __call__(
        self,
        encoder,
        segment_ids=None,
        pad_mask=None,
        cache=None,
        start_pos=0,
        intervention="normal",
    ):
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
        elif start_pos != 0:
            raise ValueError("Uncached thinking requires a full prefix")
        positions = mx.broadcast_to((start_pos + mx.arange(t))[None], (b, t))
        good = (
            mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_)
        )
        docs = mx.zeros((b, t), mx.int32) if segment_ids is None else segment_ids
        reset = mx.concatenate(
            [
                mx.ones((b, 1), mx.bool_),
                (docs[:, 1:] != docs[:, :-1]) | ~good[:, :-1] | ~good[:, 1:],
            ],
            axis=1,
        )
        runs = mx.cumsum(reset.astype(mx.int32), axis=1)
        local = rms_unit(self.input(rms_unit(encoder)))
        new_memory = _rotate(local, positions)
        memory = (
            new_memory
            if cache is None or cache.memory is None
            else mx.concatenate([cache.memory, new_memory], axis=1)
        )
        memory_pos = mx.arange(memory.shape[1])
        visible = memory_pos[None, None] <= positions[..., None]
        if cache is None:
            visible = (
                visible
                & good[:, :, None]
                & good[:, None, :]
                & (runs[:, :, None] == runs[:, None, :])
            )
        previous = mx.zeros((b, 1, local.shape[-1]), local.dtype)
        if cache is not None and cache.previous is not None:
            previous = cache.previous
        if cache is not None and t == 1 and intervention == "normal":
            # Both stage inputs are ready. Pack query rows for one block call.
            joined = mx.concatenate([local, (local + previous) * 2**-0.5], axis=1)
            both = self.block(
                joined,
                memory,
                mx.concatenate([positions, positions], axis=1),
                mx.broadcast_to(visible, (b, 2, memory.shape[1])),
            )
            first, second = both[:, :1], both[:, 1:]
        else:
            first = self.block(local, memory, positions, visible)
            shifted = mx.concatenate([previous, first[:, :-1]], axis=1)
            if cache is None:
                shifted = mx.where((~reset & good)[..., None], shifted, 0)
            if intervention == "reset":
                shifted = mx.zeros_like(shifted)
            elif intervention == "swap":
                if b < 2 or segment_ids is not None or pad_mask is not None:
                    raise ValueError(
                        "Thinking swap needs two unpadded single-document rows"
                    )
                shifted = mx.concatenate([shifted[1:], shifted[:1]], axis=0)
            second = self.block((local + shifted) * 2**-0.5, memory, positions, visible)
        projected = rms_unit(self.output(second).astype(mx.float32))
        energy = mx.mean(encoder.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        amplitude = mx.sqrt(energy + self.config.norm_eps)
        signal = self.config.thinking_scale * amplitude * projected
        signal = mx.where(good[..., None], signal, 0).astype(encoder.dtype)
        if intervention == "off":
            signal = mx.zeros_like(signal)
        if cache is not None:
            cache.memory = memory
            cache.previous = first[:, -1:]
            cache.tokens = start_pos + t
            cache.signature = cache_signature(self.config)
        return signal
