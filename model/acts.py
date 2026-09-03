"""逐点激活（与 MoE / decode kernel 共用，避免循环导入）。"""

import os

import mlx.core as mx

# 0 时 situ_glu_gu 退回“切片 + 两参数核”，用于同口径 A/B 定位性能回退。
_PACKED = os.environ.get("VIBY_SITU_PACKED", "1") != "0"

# 0 时 silu_glu_gu 退回“切片 + 两参数核”（同 _PACKED 的用途，独立开关）。
_PACKED_SILU = os.environ.get("VIBY_SILU_PACKED", "1") != "0"

# 0 时 SiTU-GLU 退回无界 SwiGLU（g·σ(g)·u），用于 r082 回退归因 A/B。
# 只影响训练/eager 前向；MoE decode 的融合 Metal 核不受此开关控制。
_SITU = os.environ.get("VIBY_SITU", "1") != "0"

# K3 SiTU-GLU：softcap(x,β)=β tanh(x/β) 分别封 Swish 的线性因子与 up 支。
SITU_BETA1 = 4.0
SITU_BETA2 = 25.0


def situ_glu_eager(
    g: mx.array, u: mx.array, beta1: float = SITU_BETA1, beta2: float = SITU_BETA2
):
    """SiTU-GLU 公式参考（融合 kernel 对照 / 非常数 β）。"""
    gate = beta1 * mx.tanh(g / beta1) * mx.sigmoid(g)
    up = beta2 * mx.tanh(u / beta2)
    return gate * up


def situ_glu(
    g: mx.array, u: mx.array, beta1: float = SITU_BETA1, beta2: float = SITU_BETA2
):
    """SiTU-GLU(g,u) = [β1 tanh(g/β1) ⊙ σ(g)] ⊙ [β2 tanh(u/β2)]。

    近原点一阶同 SwiGLU；|输出| ≤ β1 β2 = 100。默认 β 走融合 kernel。
    """
    if not _SITU:
        return g * mx.sigmoid(g) * u
    if beta1 == SITU_BETA1 and beta2 == SITU_BETA2:
        from .kernels.situ import situ_glu as _fused

        return _fused(g, u)
    return situ_glu_eager(g, u, beta1, beta2)


def situ_glu_gu(h: mx.array):
    """situ_glu(h[..., :I], h[..., I:])，I = h.shape[-1] // 2。

    gate/up 合并成一次 GEMM 后天然产出 [g | u] 拼接布局。两参数入口要
    先把两个跨步视图各物化成连续副本（fwd 两次、bwd 反过来还要一次
    拼接），打包核直接在核内寻址两半省掉这些搬运。
    """
    if not _PACKED or not _SITU:
        I = h.shape[-1] // 2  # noqa: E741
        return situ_glu(h[..., :I], h[..., I:])
    from .kernels.situ import situ_glu_packed

    return situ_glu_packed(h)


def silu_glu_eager(g: mx.array, u: mx.array) -> mx.array:
    """SiLU-GLU（SwiGLU）：silu(g)·u = g·σ(g)·u。

    与 situ_glu_eager 同接口；无 tanh 软帽，|输出| 无 ≤ β1β2 上界。
    """
    return g * mx.sigmoid(g) * u


def silu_glu(g: mx.array, u: mx.array) -> mx.array:
    """SiLU-GLU；与 situ_glu 同接口。暂不走手写 fused kernel（fused 核
    见 kernels/silu.py）：默认 dtype 走融合 kernel，失败回退 eager。
    """
    from .kernels.silu import silu_glu as _fused

    return _fused(g, u)


def silu_glu_gu(h: mx.array) -> mx.array:
    """silu_glu(h[..., :I], h[..., I:])，I = h.shape[-1] // 2。

    与 situ_glu_gu 同理：gate/up 合并成一次 GEMM 后天然产出 [g | u] 拼接
    布局，打包核在核内直接寻址两半省掉跨步视图物化与反向拼接。
    """
    if not _PACKED_SILU:
        I = h.shape[-1] // 2  # noqa: E741
        return silu_glu(h[..., :I], h[..., I:])
    from .kernels.silu import silu_glu_packed

    return silu_glu_packed(h)


def glu_gu(h: mx.array, hidden_act: str) -> mx.array:
    """按 config.hidden_act 选激活：'silu'（默认）走 SiLU-GLU，否则 SiTU-GLU。"""
    return silu_glu_gu(h) if hidden_act == "silu" else situ_glu_gu(h)
