"""Context-written delta memory, followed by two dependent reads (NTP only).

The inclusive affine scan is work-efficient O(T H r^3), not a Python token
loop or Hillis-Steele O(T log T) scan. Sequential reference is retained.
"""
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass(frozen=True)
class BindingConfig:
    dim: int = 1024
    rank: int = 64
    banks: int = 2
    seed: int = 20260914
    version: str = "binding_workspace_v1"

    def to_dict(self):
        return asdict(self)


def ball(x):
    x = x.astype(mx.float32)
    # Clamp the squared norm before sqrt: finite derivative at x=0.
    return x * mx.rsqrt(mx.maximum(mx.sum(x*x, -1, keepdims=True), 1.0))


def compose(left, right):
    a, b = left
    c, d = right
    return batch_mm(a, c), batch_mm(b, c) + d


def batch_mm(a, b):
    """Materialize strided scan batches before batched GEMM and its VJP."""
    shape = a.shape
    a = mx.contiguous(a).reshape(-1, shape[-2], shape[-1])
    b = mx.contiguous(b).reshape(-1, b.shape[-2], b.shape[-1])
    return (a @ b).reshape(shape)


def affine_scan(a, b):
    """Inclusive scan over axis 1; associative in real arithmetic.

    Reduce adjacent pairs; recursively scan pairs; reconstruct even prefixes.
    Each level processes half as many items, avoiding a T*log(T) work factor.
    """
    n = a.shape[1]
    if n == 1:
        return a, b
    if n % 2:
        eye = mx.broadcast_to(mx.eye(a.shape[-1]), a[:, :1].shape)
        a = mx.concatenate([a, eye], 1)
        b = mx.concatenate([b, mx.zeros_like(b[:, :1])], 1)
    pairs = compose((a[:, ::2], b[:, ::2]), (a[:, 1::2], b[:, 1::2]))
    if a.shape[1] == 2:
        # Avoid empty matrix batches in the reconstruction base case.
        return mx.concatenate([a[:, :1], pairs[0]], 1)[:, :n], mx.concatenate([b[:, :1], pairs[1]], 1)[:, :n]
    odd_a, odd_b = affine_scan(*pairs)
    even_a, even_b = compose((odd_a[:, :-1], odd_b[:, :-1]), (a[:, 2::2], b[:, 2::2]))
    even_a = mx.concatenate([a[:, :1], even_a], 1)
    even_b = mx.concatenate([b[:, :1], even_b], 1)
    shape = (a.shape[0], a.shape[1], *a.shape[2:])
    return (mx.stack([even_a, odd_a], 2).reshape(shape)[:, :n],
            mx.stack([even_b, odd_b], 2).reshape(shape)[:, :n])


def memory_scan(k, v, beta, valid, reset, initial=None, *, reference=False):
    """k/v [B,T,r], beta [B,T,H], state [B,H,r,r] (value,key)."""
    k, v, beta = (x.astype(mx.float32) for x in (k, v, beta))
    b, t, r = k.shape
    h = beta.shape[-1]
    if initial is None:
        initial = mx.zeros((b, h, r, r), mx.float32)
    rates = mx.where(valid[..., None], beta, 0)
    if reference:
        state, states = initial, []
        for i in range(t):
            state = mx.where(reset[:, i, None, None, None], 0, state)
            residual = v[:, i, None, :] - (state @ k[:, i, None, :, None])[..., 0]
            state = state + rates[:, i, :, None, None] * residual[..., None] * k[:, i, None, None, :]
            states.append(state)
        return mx.stack(states, 1)
    return _memory_scan(k, v, beta, valid, reset, initial)


@mx.custom_function
def _memory_scan(k, v, beta, valid, reset, initial):
    r = k.shape[-1]
    rates = mx.where(valid[..., None], beta, 0)
    a = mx.eye(r) - rates[..., None, None] * k[:, :, None, :, None] * k[:, :, None, None, :]
    update = rates[..., None, None] * v[:, :, None, :, None] * k[:, :, None, None, :]
    a = mx.where(reset[..., None, None, None], 0, a)
    prefix_a, prefix_b = affine_scan(a, update)
    return batch_mm(mx.broadcast_to(initial[:, None], prefix_a.shape), prefix_a) + prefix_b


@_memory_scan.vjp
def _memory_vjp(primals, cotangent, states):
    """Analytic delta adjoint; reverse associative scan, no token bwd loop.

    MLX 0.32.2 automatic differentiation through the recursive interleaved scan
    disagrees with finite differences. Keep the primitive's VJP explicit and
    compare all four floating-point input gradients with sequential reference.
    """
    k, v, beta, valid, reset, initial = primals
    rates = mx.where(valid[..., None], beta, 0)
    a = mx.eye(k.shape[-1]) - rates[..., None, None] * k[:, :, None, :, None] * k[:, :, None, None, :]
    a = mx.where(reset[..., None, None, None], 0, a)
    next_a = mx.concatenate([a[:, 1:], mx.zeros_like(a[:, :1])], 1).swapaxes(-1,-2)
    _, adjoint = affine_scan(next_a[:, ::-1], cotangent[:, ::-1])
    adjoint = adjoint[:, ::-1]
    previous = mx.concatenate([initial[:, None], states[:, :-1]], 1)
    previous = mx.where(reset[..., None, None, None], 0, previous)
    adj_k = mx.sum(adjoint * k[:, :, None, None, :], -1)
    old_k = mx.sum(previous * k[:, :, None, None, :], -1)
    residual = v[:, :, None, :] - old_k
    gv = mx.sum(rates[..., None] * adj_k, axis=2)
    gb = mx.where(valid[..., None], mx.sum(residual * adj_k, -1), 0)
    left = mx.sum(adjoint * residual[..., :, None], -2)
    right = mx.sum(previous * adj_k[..., :, None], -2)
    gk = mx.sum(rates[..., None] * (left - right), axis=2)
    gi = batch_mm(adjoint[:, 0], a[:, 0].swapaxes(-1,-2))
    return gk, gv, gb, mx.zeros_like(valid), mx.zeros_like(reset), gi


class BindingWorkspace(nn.Module):
    def __init__(self, config=BindingConfig(), address_mode="full"):
        super().__init__()
        if address_mode not in ("full", "fixed"):
            raise ValueError("address_mode must be full or fixed")
        self.config, self.address_mode = config, address_mode
        d, r, h = config.dim, config.rank, config.banks
        if 2*r > d:
            raise ValueError("orthonormal bridge needs 2r <= d")
        self.key = nn.Linear(d, r, bias=False)
        self.value = nn.Linear(d, r, bias=False)
        self.query = nn.Linear(d, r, bias=False)
        self.write_gate = nn.Linear(d, h, bias=True)
        self.controllers = [nn.Linear(d, h, bias=True) for _ in range(2)]
        self.state_control = nn.Linear(r, h, bias=False)
        # Explicit normal fan-in, independent NumPy RNG; no backbone RNG use.
        rng = np.random.default_rng(config.seed)
        for layer in [self.key, self.value, self.query, self.write_gate, *self.controllers, self.state_control]:
            layer.weight = mx.array(rng.normal(0, layer.weight.shape[1]**-.5, layer.weight.shape).astype(np.float32))
            if "bias" in layer:
                layer.bias = mx.zeros_like(layer.bias)
        q, _ = np.linalg.qr(rng.normal(size=(d, 2*r)))
        self.bridge = mx.array(q.astype(np.float32))
        self.freeze(recurse=False, keys=["bridge"])

    def __call__(self, evidence, anchor, tokens, pad=None, segments=None,
                 initial=None, previous_token=None, *, reference=False,
                 address_override=None, write_mask=None, eos_id=2, pad_id=0):
        valid = tokens != pad_id if pad is None else pad.astype(mx.bool_) & (tokens != pad_id)
        first = mx.ones_like(tokens[:, :1], mx.bool_) if initial is None else mx.zeros_like(tokens[:, :1], mx.bool_)
        if previous_token is not None:
            first = first | (previous_token[:, None] == eos_id)
        reset = mx.concatenate([first, tokens[:, :-1] == eos_id], 1)
        if segments is not None:
            reset = reset | mx.concatenate([mx.zeros_like(first), segments[:, 1:] != segments[:, :-1]], 1)
        writes = valid if write_mask is None else valid & write_mask
        k, v = ball(self.key(evidence)), ball(self.value(evidence))
        beta = mx.sigmoid(self.write_gate(evidence.astype(mx.float32)))
        states = memory_scan(k, v, beta, writes, reset, initial, reference=reference)
        q0 = ball(self.query(anchor))
        registers, q = [], q0
        for j in range(2):
            rho = mx.softmax(self.controllers[j](anchor.astype(mx.float32)) + self.state_control(q), -1)
            address = q0 if j == 1 and self.address_mode == "fixed" else q
            if j == 1 and address_override is not None:
                address = address_override
            read = (states @ address[..., None, :, None])[..., 0]
            q = ball(mx.sum(rho[..., None] * read, axis=-2))
            registers.append(q)
        z = mx.concatenate(registers, -1)
        injection = (z @ self.bridge.T) * valid[..., None]
        return injection, states[:, -1], tuple(registers)
