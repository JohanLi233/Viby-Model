"""RMSNorm（V4.1 口径：走 mx.fast.rms_norm，统计量在核内 fp32）。"""

import mlx.core as mx
from mlx import nn

_ONES = {}


def _ones(dim: int, dtype) -> mx.array:
    key = (dim, str(dtype))
    w = _ONES.get(key)
    if w is None:
        w = mx.ones((dim,), dtype=dtype)
        _ONES[key] = w
    return w


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


def rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    """无参数的 RMS 归一化（mHC 的 rsqrt 口径）。"""
    return mx.fast.rms_norm(x, _ones(x.shape[-1], x.dtype), eps)
