"""
Muon 混合优化器（MLX 单设备版）

基于 mlx.optimizers.Muon / mlx.optimizers.MultiOptimizer 实现混合优化器：
- Muon 仅用于 ndim >= 2 且非嵌入/输出头的核心权重矩阵（Newton-Schulz
  正交化动量），weight_decay=0
- 嵌入/输出头与其余标量参数使用 AdamW
"""

import mlx.optimizers as optim
from mlx.utils import tree_flatten


def create_adamw_optimizer(model, args, training_type="pretrain"):
    """全参数 AdamW（用于优化器杠杆对照实验；lr 需单独调优）。"""
    from .utils import Logger

    Logger("正在为优化器进行参数分组 (pure AdamW)")
    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        return "embed" in path or "lm_head" in path

    embed_count = sum(1 for p, a in trainable if _is_embed(p, a))
    other_count = len(trainable) - embed_count
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - 其余参数组: {other_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, other_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, other_lr_mult, adam_wd = 1.0, 1.0, 0.1

    adamw_embed = optim.AdamW(
        learning_rate=args.learning_rate * embed_lr_mult,
        betas=[0.9, 0.95], eps=1e-8, weight_decay=adam_wd,
    )
    adamw_other = optim.AdamW(
        learning_rate=args.learning_rate * other_lr_mult,
        betas=[0.9, 0.95], eps=1e-8, weight_decay=adam_wd,
    )
    adamw_embed.base_lr = args.learning_rate * embed_lr_mult
    adamw_other.base_lr = args.learning_rate * other_lr_mult
    return optim.MultiOptimizer(
        [adamw_embed, adamw_other], filters=[_is_embed]
    )


def create_mixed_optimizer(model, args, training_type="pretrain"):
    """
    创建混合优化器（mlx.optimizers.MultiOptimizer）

    参数分组：
    - Muon：ndim >= 2 且非嵌入/输出头的核心权重矩阵，wd=0
    - AdamW(embed)：lr = args.learning_rate，wd=0.1
    - AdamW(scalar)：lr = args.learning_rate，wd=0.1
    （SFT 时 embed/scalar lr 分别乘 0.1/0.3，wd=0.01）
    """
    from .utils import Logger

    Logger("正在为优化器进行参数分组")

    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        return "embed" in path or "lm_head" in path

    def _is_muon(path, arr):
        return arr.ndim >= 2 and not _is_embed(path, arr)

    muon_count = sum(1 for p, a in trainable if _is_muon(p, a))
    embed_count = sum(1 for p, a in trainable if _is_embed(p, a) and not _is_muon(p, a))
    scalar_count = sum(1 for p, a in trainable if not _is_muon(p, a) and not _is_embed(p, a))

    Logger("参数分组完成：")
    Logger(f"  - Muon 参数组 (核心权重): {muon_count} 个张量")
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - 标量参数组: {scalar_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, scalar_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, scalar_lr_mult, adam_wd = 1.0, 1.0, 0.1

    muon_opt = optim.Muon(
        learning_rate=args.learning_rate,  # Muon 使用基础学习率 (e.g., 0.01)
        momentum=0.95,
        weight_decay=0.0,
    )
    adamw_embed = optim.AdamW(
        learning_rate=args.learning_rate * embed_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_scalar = optim.AdamW(
        learning_rate=args.learning_rate * scalar_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )

    # 为学习率调度器存储初始学习率
    muon_opt.base_lr = args.learning_rate
    adamw_embed.base_lr = args.learning_rate * embed_lr_mult
    adamw_scalar.base_lr = args.learning_rate * scalar_lr_mult

    # MultiOptimizer: filters 数量 = len(optimizers) - 1，按顺序首个命中生效，
    # 未命中任何 filter 的参数落到最后一组
    return optim.MultiOptimizer(
        [muon_opt, adamw_embed, adamw_scalar],
        filters=[_is_muon, _is_embed],
    )
