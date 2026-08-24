"""Marin Hero 初始化：2D 矩阵 ~ TruncNormal(0, (0.5/√hidden)²)，|z|≤2。"""

import math

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def trunc_normal(shape, std: float, bound: float = 2.0):
    """标准正态截断在 ±bound 后乘 std（init 期允许少量 host 判断）。"""
    x = mx.random.normal(shape)
    for _ in range(4):
        if not bool((mx.abs(x) > bound).any().item()):
            break
        x = mx.where(mx.abs(x) > bound, mx.random.normal(shape), x)
    return mx.clip(x, -bound, bound) * std


def _skip_init(path: str, arr: mx.array) -> bool:
    if arr.ndim < 2:
        return True
    # 零初始化：attn_gate、GatedNorm.gate_up、AttnRes 伪查询（1D 已跳过）
    if "attn_gate" in path:
        return True
    # 只跳过 GatedNorm.gate_up，不要误伤专家堆叠 gate_up_w
    leaf = path.rsplit(".", 1)[-1]
    if leaf == "gate_up":
        return True
    # ShortConv / KDA conv：身份或专用小核，不套 0.5/√hidden
    if any(
        s in path
        for s in (
            ".k_conv.",
            ".out_conv.",
            ".mlp_out_conv.",
            ".q_conv.",
            ".v_conv.",
        )
    ):
        return True
    return False


def apply_trunc_normal_init(module, hidden: int):
    """覆盖 2D 可训练矩阵为 Marin 口径 0.5/√hidden 截断正态。

    跳过零初始化门、卷积核、1-D gain。A_log / dt_bias 是 1-D，保持
    各自构造时的特殊初始化。
    """
    std = 0.5 / math.sqrt(float(hidden))
    flat = dict(tree_flatten(module.parameters()))
    new = []
    for path, arr in flat.items():
        if _skip_init(path, arr):
            new.append((path, arr))
        else:
            new.append((path, trunc_normal(arr.shape, std).astype(arr.dtype)))
    module.update(tree_unflatten(new))
