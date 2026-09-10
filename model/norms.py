"""RMSNorm（V4.1 口径：fp32 统计量，权重最后相乘，回原 dtype）。"""

import mlx.core as mx
from mlx import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        y = xf * mx.rsqrt(var + self.eps)
        return (self.weight.astype(mx.float32) * y).astype(dt)


def rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    """无参数的 RMS 归一化（mHC 的 rsqrt 口径，沿最后一维、fp32）。"""
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
