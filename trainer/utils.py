"""
训练共享工具函数模块
包含预训练和SFT训练共享的功能
"""

import os
import math
import time
import torch
import torch.distributed as dist
from torch import optim
from typing import Tuple
from .muon import Muon, SingleDeviceMuon


def Logger(content):
    """统一的日志输出函数"""
    if (
        not hasattr(Logger, "_ddp_initialized")
        or not Logger._ddp_initialized
        or dist.get_rank() == 0
    ):
        print(content)


def set_ddp_flag(is_ddp: bool):
    """设置DDP标志，供Logger使用"""
    Logger._ddp_initialized = is_ddp


def get_lr_and_momentum(
    step: int,
    total_steps: int,
    warmup_steps: int,
    initial_momentum: float = 0.85,
    final_momentum: float = 0.95,
    momentum_warmup_steps: int = 300,
) -> Tuple[float, float]:
    """
    计算当前步骤的学习率乘子和动量。
    - 学习率: Warmup + Cosine Decay
    - 动量: Linear Warmup
    """
    # --- 学习率调度 ---
    if step < warmup_steps:
        # 线性预热阶段
        lr_multiplier = float(step) / float(max(1, warmup_steps))
    elif step >= total_steps:
        # 训练结束，使用最小学习率
        lr_multiplier = 0.0  # 或者一个很小的值
    else:
        # 余弦衰减阶段
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        lr_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    # --- 动量调度 (仅用于Muon) ---
    momentum = final_momentum
    if step < momentum_warmup_steps:
        frac = float(step) / float(max(1, momentum_warmup_steps))
        momentum = (1 - frac) * initial_momentum + frac * final_momentum

    return lr_multiplier, momentum


def save_checkpoint(
    model, optimizer, scaler, epoch, step, args, lm_config, training_type="pretrain"
):
    """统一的检查点保存函数"""
    model.eval()

    # 根据训练类型确定文件名
    moe_path = "_moe" if getattr(lm_config, "use_moe", False) else ""
    if training_type == "sft":
        ckp_filename = f"full_sft_{lm_config.hidden_size}{moe_path}.pth"
    else:
        ckp_filename = f"pretrain_{lm_config.hidden_size}{moe_path}.pth"

    ckp = os.path.join(args.save_dir, ckp_filename)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # 获取原始模型（跳过DDP和compile包装）
        original_model = model.module
        if hasattr(original_model, "_orig_mod"):
            original_model = original_model._orig_mod
        state_dict = original_model.state_dict()
    else:
        # 获取原始模型（跳过compile包装）
        original_model = model
        if hasattr(original_model, "_orig_mod"):
            original_model = original_model._orig_mod
        state_dict = original_model.state_dict()

    # 保存检查点，包含更多信息
    checkpoint = {
        "model_state_dict": {k: v.half() for k, v in state_dict.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "config": lm_config.__dict__,
        "training_type": training_type,
    }

    torch.save(checkpoint, ckp)
    Logger(f"Checkpoint saved: {ckp}")

    # 保存最新的检查点路径 (使用绝对路径)
    latest_ckp = os.path.join(args.save_dir, "latest_checkpoint.txt")
    with open(latest_ckp, "w") as f:
        f.write(os.path.abspath(ckp))

    model.train()


def load_checkpoint(checkpoint_path, model, optimizer, scaler, args):
    """统一的检查点加载函数"""
    Logger(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # 加载模型状态到原始模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # 获取原始模型（跳过DDP和compile包装）
        original_model = model.module
        if hasattr(original_model, "_orig_mod"):
            original_model = original_model._orig_mod
        original_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 获取原始模型（跳过compile包装）
        original_model = model
        if hasattr(original_model, "_orig_mod"):
            original_model = original_model._orig_mod
        original_model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器与scaler状态（如未指定重置）
    if not getattr(args, "reset_optimizer", False):
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    last_finished_step = checkpoint.get("step", 0)

    # 从下一步继续，避免重复已完成的 step
    start_step = int(last_finished_step) + 1

    # 如要求重置优化器，则从step 0开始
    if getattr(args, "reset_optimizer", False):
        start_step = 0
        Logger("reset_optimizer set: optimizer/scaler states not loaded; start_step reset to 0")

    Logger(f"Resumed from epoch {start_epoch}, next_step {start_step}")

    return start_epoch, start_step


def find_latest_checkpoint(save_dir):
    """查找最新的检查点文件"""
    latest_file = os.path.join(save_dir, "latest_checkpoint.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            return f.read().strip()
    return None


def init_distributed_mode():
    """初始化分布式训练模式"""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        return False, 0, "cuda:0"

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

    return True, ddp_local_rank, device


class MultiOptimizer(torch.optim.Optimizer):
    """多优化器包装类，支持混合优化器"""

    def __init__(self, optimizers):
        # 收集所有参数组
        params = []
        for opt in optimizers:
            for param_group in opt.param_groups:
                params.extend(param_group["params"])

        # 调用父类构造函数
        super().__init__(params, {})

        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self):
        return {
            f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            key = f"optimizer_{i}"
            if key in state_dict:
                opt.load_state_dict(state_dict[key])

    # 支持scaler的unscale操作
    def unscale_gradients(self, scaler):
        for opt in self.optimizers:
            scaler.unscale_(opt)


def create_mixed_optimizer(model, args, ddp=False, training_type="pretrain"):
    """
    创建混合优化器（Muon + AdamW）

    Args:
        model: 模型
        args: 参数配置
        ddp: 是否分布式训练
        training_type: 训练类型 ("pretrain" 或 "sft")
    """
    Logger("正在为优化器进行参数分组...")

    # 统一枚举参数，便于精确分组并避免重复
    named_params = list(model.named_parameters())

    # 1. Muon 优化器处理的参数：所有中间层的 2D 权重矩阵 + Canon 权重
    # 通过名称排除嵌入层、输出头
    muon_params = [
        p
        for n, p in named_params
        if (p.ndim == 2 and "embed" not in n and "lm_head" not in n)
        or ("canon_" in n and p.ndim > 1)
    ]

    # 2. 嵌入层参数
    embed_params = [p for n, p in named_params if "embed" in n]

    # 3. 标量参数（如 LayerNorm、偏置项等），排除 embed，但包含 canon 的 bias
    scalar_params = [
        p for n, p in named_params if p.ndim < 2 and "embed" not in n
    ]

    Logger("参数分组完成：")
    Logger(f"  - Muon 参数组: {len(muon_params)} 个张量")
    Logger(f"  - 嵌入层参数组: {len(embed_params)} 个张量")
    Logger(f"  - 标量参数组: {len(scalar_params)} 个张量")

    # 创建混合优化器
    optimizers = []

    # 初始化 Muon 优化器
    if muon_params:
        if training_type == "sft":
            # SFT: args.learning_rate 已经设置为较小值 (0.001)，相比预训练 (0.01) 降低了10倍
            muon_lr = args.learning_rate
        else:
            # 预训练: 使用标准学习率 (0.01)
            muon_lr = args.learning_rate

        if ddp:
            muon_optimizer = Muon(
                muon_params, lr=muon_lr, momentum=0.95, weight_decay=0
            )
        else:
            muon_optimizer = SingleDeviceMuon(
                muon_params, lr=muon_lr, momentum=0.95, weight_decay=0
            )
        optimizers.append(muon_optimizer)

    # 初始化 AdamW 优化器，包含多个精细分组
    adamw_param_groups = []

    if training_type == "sft":
        # SFT 的学习率倍数设置 - 使用较小的学习率防止灾难性遗忘
        embed_lr_mult = 0.1  # 嵌入层需要特别小心，避免破坏预训练的词汇表征
        scalar_lr_mult = 0.3  # 标量参数（如 LayerNorm）可以适度调整
        weight_decay = 0.01  # SFT 使用较小的权重衰减
    else:
        # 预训练的学习率倍数设置 - 从零开始学习，需要较大学习率
        embed_lr_mult = 1.0  # 嵌入层需要充分学习词汇表征（权重绑定，输出头共享此参数）
        scalar_lr_mult = 1.0  # 标量参数使用基准学习率
        weight_decay = 0.1  # 预训练使用较大的权重衰减防止过拟合

    if embed_params:
        adamw_param_groups.append(
            {"params": embed_params, "lr": args.learning_rate * embed_lr_mult}
        )
    if scalar_params:
        adamw_param_groups.append(
            {"params": scalar_params, "lr": args.learning_rate * scalar_lr_mult}
        )

    if adamw_param_groups:
        adamw_optimizer = optim.AdamW(
            adamw_param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        optimizers.append(adamw_optimizer)

    optimizer = MultiOptimizer(optimizers)

    # 为学习率调度器存储初始学习率
    Logger("为学习率调度器存储初始学习率...")
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer


def apply_lr_schedule(optimizer, global_step, total_training_steps, warmup_iters):
    """应用学习率调度"""
    lr_multiplier, current_momentum = get_lr_and_momentum(
        global_step,
        warmup_steps=warmup_iters,
        total_steps=total_training_steps,
    )

    # 遍历所有参数组，应用调度
    for param_group in optimizer.param_groups:
        # 应用学习率调度
        initial_lr = param_group.get("initial_lr", param_group["lr"])
        param_group["lr"] = initial_lr * lr_multiplier

        # 仅当参数组是 Muon 的组时，才应用动量调度
        if "momentum" in param_group:
            param_group["momentum"] = current_momentum


def log_training_progress(
    epoch,
    step,
    iter_per_epoch,
    current_loss,
    optimizer,
    start_time,
    args,
    wandb=None,
    grad_norm=0.0,
    max_logit=0.0,
    base_step_offset: int = 0,
):
    """统一的训练进度日志记录"""
    spend_time = time.time() - start_time
    # 使用相对步数计算速率，避免 resume 后跳过的步导致速率异常
    effective_steps_done = max(1, (step - base_step_offset + 1))
    steps_per_sec = effective_steps_done / spend_time if spend_time > 0 else 0.0
    tokens_per_sec = steps_per_sec * args.batch_size * args.max_seq_len

    Logger(
        "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.2e} grad_norm:{:.3f} max_logit:{:.3f} step/s:{:.2f} tokens/s:{:.0f} eta:{}min".format(
            epoch + 1,
            args.epochs,
            step,
            iter_per_epoch,
            current_loss,
            optimizer.param_groups[-1]["lr"],
            grad_norm,
            max_logit,
            steps_per_sec,
            tokens_per_sec,
            int((iter_per_epoch - step - 1) / max(steps_per_sec, 1e-8) / 60),
        )
    )

    if wandb is not None:
        wandb.log(
            {
                "loss": current_loss,
                "lr": optimizer.param_groups[-1]["lr"],
                "steps_per_sec": steps_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "grad_norm": grad_norm,
                "max_logit": max_logit,
                "epoch": epoch + 1,
            }
        )
