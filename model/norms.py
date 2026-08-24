import mlx.core as mx
import mlx.nn as nn

from .kernels.gated_norm import gated_norm_decode


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # MLX 融合 RMSNorm kernel（bf16 路径比手动 float32 实现快数倍）
        return mx.fast.rms_norm(x, self.weight, self.eps)


class GatedNorm(nn.Module):
    """GatedNorm：RMSNorm 输出后接一个 rank-128 bottleneck 逐维门。

    y = RMSNorm(x)；g = 2·sigmoid(silu(y @ W_down) @ W_up)；输出 y * g。
    W_down (d, rank) / W_up (rank, d)，无 bias。

    初始化：W_up 零初始化 + W_down 小随机均匀初始化（与 _StackedExperts
    同口径）。初始 silu(y@W_down) @ 0 = 0 → 门 = 2·sigmoid(0) = 1，严格
    恒等于裸 RMSNorm 起步；W_up 的梯度 dz/dW_up = silu(...) ≠ 0 从
    step 0 即可学，W_down 的梯度经 W_up 回传、随其离地恢复（ΔW·V=0
    式零初始化的标准启动动力学）。两矩阵同零会把双向梯度都堵死
    （dz/dW_up = silu(0)=0 且 dz/dW_down ∝ W_up=0），故不采用；
    sigmoid 裸零初始化（门=0.5）会把激活砍半，也不采用——2·sigmoid
    使初始门恰为 1。
    W_down/W_up 为 ndim=2 → 自动进 Muon 组；内部 RMSNorm 的 1-D gain
    留 AdamW 标量组（embed_norm 的两个矩阵同样进 Muon，见
    trainer/muon.py 分组规则）。
    """

    def __init__(self, dim: int, eps: float = 1e-5, rank: int = 128):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        std = dim**-0.5
        self.gate_down = mx.random.uniform(-std, std, (dim, rank))
        self.gate_up = mx.zeros((rank, dim))

    def __call__(self, x: mx.array) -> mx.array:
        # 推理小批量：融合 Metal kernel（1 kernel 替代 ~7 个，无梯度）。
        # 训练/大行数走可微 eager 链。
        if not self.training:
            r = gated_norm_decode(
                x, self.norm.weight, self.gate_down, self.gate_up, self.norm.eps
            )
            if r is not None:
                return r
        y = self.norm(x)
        g = 2.0 * mx.sigmoid(nn.silu(y @ self.gate_down) @ self.gate_up)
        return y * g.astype(y.dtype)


_rms_unit_weights: dict = {}


def _rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    """无权重 RMS 单位化。用融合 RMSNorm kernel（ones 权重）替代手写
    f32 平方/均值/rsqrt 链：数学等价，Metal 单 kernel 且不必物化
    (..., D) 级 f32 中间量。"""
    w = _rms_unit_weights.get((x.shape[-1], x.dtype, eps))
    if w is None:
        w = mx.ones((x.shape[-1],), dtype=x.dtype)
        _rms_unit_weights[(x.shape[-1], x.dtype, eps)] = w
    return mx.fast.rms_norm(x, w, eps).astype(x.dtype)
