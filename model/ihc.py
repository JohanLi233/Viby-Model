"""identity Hyper-Connections（iHC）：H_res = I，token 门读/写 M 条残差流。

Hy4 preview（hc_mult=4）与 Chimera §3.6 同构：
  x̃ = Σ_m h_pre,m(R) R_m
  R'_m = R_m + h_post,m(R) Δ
h_pre ∈ (0,1)^M，h_post ∈ (0,2)^M。α=0 且流为副本时，读是均匀平均、写门=1，
等价普通残差 h ← h+Δ。出口默认流上均值；identity 只取流 0。没有 mHC 的
Sinkhorn / 流间混合。
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .norms import _rms_unit

# Chimera：α 从小值起步，输入相关路由逐渐出现。0 时严格恒等（测用）。
_IHC_ALPHA0 = 0.01


def ihc_expand(h: mx.array, n_streams: int) -> mx.array:
    """(B, T, D) → (B, T, M, D)，broadcast 复制，不要 concat/slice。"""
    m = int(n_streams)
    if m < 2:
        raise ValueError("iHC 流数必须 >= 2")
    d = int(h.shape[-1])
    return mx.broadcast_to(h[..., None, :], h.shape[:-1] + (m, d))


def ihc_collapse(r: mx.array, mode: str = "mean") -> mx.array:
    """(B, T, M, D) → (B, T, D)。mean=流上均值；identity=只取流 0。"""
    if mode == "identity":
        return r[..., 0, :]
    if mode != "mean":
        raise ValueError("ihc_collapse mode 必须是 mean 或 identity")
    return mx.mean(r, axis=-2)


def ihc_add_to_stream(r: mx.array, delta: mx.array, stream: int) -> mx.array:
    """把 (B, T, D) 加到第 stream 条流，其余流不动。"""
    m = int(r.shape[-2])
    s = int(stream)
    if s < 0 or s >= m:
        raise ValueError("ihc 流下标越界")
    mask = mx.zeros((m,), dtype=r.dtype)
    mask = mask.at[s].add(1.0)
    lead = (1,) * (r.ndim - 2)
    return r + delta[..., None, :] * mask.reshape(lead + (m, 1))


class IHCGate(nn.Module):
    """一个 sublayer 的读/写门：W_h ∈ R^{2M × MD}，α 标量，bias 两段。"""

    def __init__(self, dim: int, n_streams: int, eps: float = 1e-6):
        super().__init__()
        self.n_streams = int(n_streams)
        self.eps = float(eps)
        md = self.n_streams * int(dim)
        std = md**-0.5
        self.weight = mx.random.uniform(-std, std, (2 * self.n_streams, md))
        logit = math.log(1.0 / (self.n_streams - 1))
        self.bias_pre = mx.full((self.n_streams,), logit)
        self.bias_post = mx.zeros((self.n_streams,))
        self.alpha = mx.array([_IHC_ALPHA0])

    def gates(self, r: mx.array) -> tuple[mx.array, mx.array]:
        """h_pre, h_post：(B, T, M)。都从当前 R 算出（写前）。"""
        b, t, m, d = r.shape
        v = r.reshape(b, t, m * d)
        n = _rms_unit(v, self.eps)
        logits = self.alpha.astype(r.dtype) * (n @ self.weight.T)
        pre, post = mx.split(logits, 2, axis=-1)
        pre = pre + self.bias_pre.astype(r.dtype)
        post = post + self.bias_post.astype(r.dtype)
        h_pre = mx.sigmoid(pre.astype(mx.float32)).astype(r.dtype)
        h_post = (2.0 * mx.sigmoid(post.astype(mx.float32))).astype(r.dtype)
        return h_pre, h_post

    def mix(self, r: mx.array) -> tuple[mx.array, mx.array]:
        """读 + 写门：x̃ 与 h_post。优先融合核。"""
        from .kernels import ihc_fused

        o = ihc_fused.mix(
            r, self.weight, self.alpha, self.bias_pre, self.bias_post, self.eps
        )
        if o is not None:
            return o
        h_pre, h_post = self.gates(r)
        return self.read(r, h_pre), h_post

    def read(self, r: mx.array, h_pre: mx.array) -> mx.array:
        return (h_pre[..., None] * r).sum(axis=2)

    def write(self, r: mx.array, delta: mx.array, h_post: mx.array) -> mx.array:
        from .kernels import ihc_fused

        o = ihc_fused.write(r, delta, h_post)
        if o is not None:
            return o
        return r + h_post[..., None] * delta[..., None, :]
