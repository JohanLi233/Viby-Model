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
        """统计量走 fp32，元素乘法留在原 dtype（省掉两次 [B,T,D] 的 fp32 往返）。"""
        dt = x.dtype
        var = mx.mean((x * x).astype(mx.float32), axis=-1, keepdims=True)
        rstd = mx.rsqrt(var + self.eps).astype(dt)
        return (x * rstd) * self.weight.astype(dt)


def rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    """无参数的 RMS 归一化（mHC 的 rsqrt 口径）。

    统计量在 fp32 上算（平方先按原 dtype 乘、再升 fp32 求均值），但不把整个
    [B,T,hc*D] 激活升到 fp32：mHC 每层调用 2 次、每次 [4,1024,4096]，
    原实现光 `.astype(fp32)` + fp32 乘法就是 ~130MB/次（24 次/前向）。
    """
    dt = x.dtype
    var = mx.mean((x * x).astype(mx.float32), axis=-1, keepdims=True)
    return x * mx.rsqrt(var + eps).astype(dt)
