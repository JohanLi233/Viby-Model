"""mHC：Manifold-Constrained Hyper-Connections（V4.1 §2.2）。

残差流是 hc_mult 条并行副本 [B, T, hc_mult, dim]。每个子层（attn / MoE）
前后的读写都由一组 (pre, post, comb) 系数决定，系数从流本身现算：

    mixes = Linear(RMSNorm_flat(x))                       # [(2+hc)·hc]
    pre   = sigmoid(mixes[:hc] * s0 + b0) + eps           # 收敛子层输入
    post  = 2 * sigmoid(mixes[hc:2hc] * s1 + b1)          # 子层写出强度
    comb  = Sinkhorn(softmax(mixes[2hc:] * s2 + b2))      # 双随机残差混合

Sinkhorn–Knopp 迭代把 comb 投影到双随机矩阵，保证跨层信号传播非扩张。
"""

import mlx.core as mx
from mlx import nn

from .norms import rms_unit


def hc_split(mixes: mx.array, scale: mx.array, base: mx.array, hc_mult: int, eps: float = 1e-6):
    """mixes: [..., (2+hc)·hc] → (pre [...,hc], post [...,hc], comb [...,hc·hc])。

    eps 用 config.hc_eps：官方 hc_split_sinkhorn 的 pre 分支就是 sigmoid(...) + eps。
    """
    hc = hc_mult
    m = mixes
    pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = m[..., 2 * hc :] * scale[2] + base[2 * hc :]
    return pre, post, comb


def sinkhorn(comb: mx.array, iters: int, eps: float) -> mx.array:
    """comb: [..., hc, hc] → 双随机矩阵（softmax 起手，再交替行列归一化）。"""
    m = mx.max(comb, axis=-1, keepdims=True)
    c = mx.exp(comb - m)
    c = c / (mx.sum(c, axis=-1, keepdims=True) + eps)
    # 列归一化放在前面："comb / (comb.sum(-2) + eps)" 首步
    c = c / (mx.sum(c, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (mx.sum(c, axis=-1, keepdims=True) + eps)
        c = c / (mx.sum(c, axis=-2, keepdims=True) + eps)
    return c


class HyperConnection(nn.Module):
    """一条子层（attn 或 MoE）的 mHC 读写参数。"""

    def __init__(self, dim: int, hc_mult: int, sinkhorn_iters: int = 20, eps: float = 1e-6,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.norm_eps = norm_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # 与参考实现同形状：[mix_hc, hc_mult * dim]（等价 Linear(hc_dim, mix_hc, bias=False)）
        self.fn = nn.Linear(hc_dim, mix_hc, bias=False)
        self.scale = mx.ones((3,), dtype=mx.float32)
        # 起步 ≈ 标准 pre-norm 残差：pre 近 one-hot(0)、post=1、comb 均匀。
        base = mx.zeros((mix_hc,), dtype=mx.float32)
        base = base.at[:hc_mult].add(mx.array([4.0] + [-4.0] * (hc_mult - 1), dtype=mx.float32))
        self.base = base

    def mixes(self, x: mx.array):
        """x: [B, T, hc, dim] → (pre, post, comb)，fp32。"""
        flat = x.reshape(*x.shape[:2], -1)
        mixes = self.fn(rms_unit(flat, self.norm_eps).astype(x.dtype)).astype(mx.float32)
        pre, post, comb = hc_split(mixes, self.scale, self.base, self.hc_mult, self.eps)
        comb = sinkhorn(
            comb.reshape(*comb.shape[:-1], self.hc_mult, self.hc_mult),
            self.sinkhorn_iters,
            self.eps,
        )
        return pre, post, comb


def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    """把 hc 条流按 pre_mix 收敛成一条：[B,T,hc,d] × [B,T,hc] → [B,T,d]。"""
    y = mx.sum(pre_mix[..., None] * x, axis=-2)
    return y.astype(x.dtype)


def hc_post(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """子层输出展开回 hc 条流 + 用 comb 混入残差。

    x:[B,T,d] residual:[B,T,hc,d] post:[B,T,hc] comb:[B,T,hc,hc]
        out[m] = post[m]·x + Σ_j comb[m, j]·residual[j]
    """
    y = post[..., None] * x[..., None, :] + mx.sum(
        comb[..., None] * residual[..., None, :, :], axis=-2
    )
    return y.astype(x.dtype)


def identity_pre_mix(x: mx.array, hc_mult: int) -> mx.array:
    """起始读取：只取第 0 条流（对应参考实现 make_identity_pre_mix）。"""
    pre = mx.zeros((x.shape[0], x.shape[1], hc_mult), dtype=mx.float32)
    return pre.at[:, :, 0].add(1.0)
