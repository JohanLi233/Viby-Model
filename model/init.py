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


def _skip_init(path: str, arr: mx.array) -> bool:
    """跳过 1-D 量与专用初始化：norm 权重、attn_sink（零）、mHC scale/base、
    RoPE 表（freq_cos/freq_sin）、Engram 的 q/k_weight、NCP 零初始化残差头。"""
    if path in ("output.weight",):
        return True
    # NCP copy-residual 输出投影：零初始化是语义（初始预测 = 精确 copy），
    # 全局重新随机化会无声破坏它（与 feedback_gate 等 1-D 零门不同，它是 2-D）。
    if path.endswith("residual_out.weight"):
        return True
    if arr.ndim < 2:
        return True
    if path.endswith("attn_sink"):
        return True
    # RoPE 表（freq_cos / freq_sin）是 config 决定的常量，不是可学矩阵：
    # 覆盖它们会把位置编码换成随机噪声（且它们是 2-D，光靠 ndim<2 拦不住）。
    if path.endswith("freq_cos") or path.endswith("freq_sin"):
        return True
    # Engram：q_weight / k_weight 必须保持 ones（初始门 = 纯归一化点积），
    # 查表 embed 是稀疏寻址的大表，二次随机化纯属浪费。
    if "engram_layers" in path and (
        path.endswith("q_weight") or path.endswith("k_weight") or ".embed." in path
    ):
        return True
    return False


def apply_trunc_normal_init(module, hidden: int):
    """覆盖 2D/3D 可训练矩阵为 Marin 截断正态，std=0.5/√fan_in。

    fan_in 取最后一维（前向都是 x @ W.T）；专家堆叠 (E, out, in) 同理，
    逐 expert 独立同分布。1-D gain / 门控保持构造时的专用初始化。
    """
    flat = dict(tree_flatten(module.parameters()))
    new = []
    for path, arr in flat.items():
        if _skip_init(path, arr):
            new.append((path, arr))
        else:
            std = 0.5 / math.sqrt(float(arr.shape[-1]))
            new.append((path, trunc_normal(arr.shape, std).astype(arr.dtype)))
    module.update(tree_unflatten(new))
