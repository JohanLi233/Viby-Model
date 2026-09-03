"""Marin Hero 初始化：矩阵 ~ TruncNormal(0, (0.5/√fan_in)²)，|z|≤2。"""

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


def _fan_in(path: str, arr: mx.array, hidden: int) -> float:
    """矩阵 fan-in：Linear / 专家堆叠是最后一维；GatedNorm.gate_down 是 x@W 无转置。"""
    leaf = path.rsplit(".", 1)[-1]
    if leaf == "gate_down":
        return float(arr.shape[0])
    if arr.ndim >= 2:
        return float(arr.shape[-1])
    return float(hidden)


def _skip_init(path: str, arr: mx.array) -> bool:
    if arr.ndim < 2:
        return True
    # 零初始化：attn_gate、KDA g_proj、GatedNorm.gate_up、AttnRes 伪查询（1D 已跳过）
    if "attn_gate" in path:
        return True
    if ".g_proj." in path:
        return True
    # 零初始化 scale：moe_write_spread 的 write_scale（s=0 ⇒ 额外写出恒等）
    if "write_scale" in path:
        return True
    # 只跳过 GatedNorm.gate_up，不要误伤专家堆叠 gate_up_w
    leaf = path.rsplit(".", 1)[-1]
    if leaf == "gate_up":
        return True
    # ShortConv / KDA conv：身份或专用小核，不套 TruncNormal(0.5/√fan_in)
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
    """覆盖 2D/3D 可训练矩阵为 Marin 截断正态，std=0.5/√fan_in。

    fan_in 取 Linear / 专家堆叠的最后一维（x @ W.T）；GatedNorm.gate_down
    是 x@W，fan_in 取第 0 维。方阵与 hidden 宽的投影和旧 0.5/√hidden 相同；
    down / lat_up / 专家 down 按真实输入维放大，MuonH 半径才对得上。
    跳过零初始化门、卷积核、1-D gain。A_log / dt_bias 是 1-D，保持
    各自构造时的特殊初始化。
    """
    flat = dict(tree_flatten(module.parameters()))
    new = []
    for path, arr in flat.items():
        if _skip_init(path, arr):
            new.append((path, arr))
        else:
            std = 0.5 / math.sqrt(_fan_in(path, arr, hidden))
            new.append((path, trunc_normal(arr.shape, std).astype(arr.dtype)))
    module.update(tree_unflatten(new))
