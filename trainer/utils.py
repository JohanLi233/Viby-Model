"""
训练共享工具函数模块（MLX 单设备版）
包含预训练和SFT训练共享的功能
"""

import os
import json
import math
import time
from typing import Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import AutoTokenizer
from model.model import VibyForCausalLM


def Logger(content):
    """统一的日志输出函数（单设备，直接打印）"""
    print(content)


def set_ddp_flag(is_ddp: bool):
    """保留的兼容接口，单设备下为 no-op"""
    pass


def convert_model_dtype(model, dtype_name):
    """按 args.dtype 将模型参数整体转换为 bfloat16/float16（MLX 无 autocast）。

    非持久 buffer（freqs_cos/freqs_sin）也在 parameters() 里，
    一并转换无妨；加载权重时结构匹配即可。
    """
    if dtype_name == "bfloat16":
        target = mx.bfloat16
    elif dtype_name == "float16":
        target = mx.float16
    else:
        return model

    params = model.parameters()

    def cast(p):
        return p.astype(target) if mx.issubdtype(p.dtype, mx.floating) else p

    model.update(tree_map(cast, params))
    mx.eval(model.parameters())
    return model


def log_parameter_count(model):
    total_params = model.num_parameters()
    trainable_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    Logger(
        f"总参数量：{total_params / 1e6:.3f}M, 可训练参数量：{trainable_params / 1e6:.3f}M"
    )


# 这些 key 是按当前 config 重新计算的非持久状态，跨 checkpoint
# 加载时不应因 shape 不一致而报错，也不应覆盖当前模型中的版本。
_NON_STRICT_WEIGHT_KEYS = (
    "freqs_cos",
    "freqs_sin",
)


def _is_non_strict_key(key: str) -> bool:
    return any(key == k or key.endswith("." + k) for k in _NON_STRICT_WEIGHT_KEYS)


def load_model_weights(
    model,
    checkpoint_path,
    strict=True,
    label="checkpoint",
    allow_dim0_slice=False,
):
    """安全地加载模型权重。

    - `strict=True` 时，除 `_NON_STRICT_WEIGHT_KEYS` 之外缺少/多余/形状不一致
      的参数都会抛错，避免静默得到随机初始化的部分模型。
    - `freqs_cos/freqs_sin` buffer 即使 shape 不匹配也会被跳过，
      保留当前模型按新 config 计算的版本。
    - `allow_dim0_slice=True` 仅用于 loop_k_override 推理：形状只在第 0 维
      （loop 步数）不同的参数，按重叠步数做前缀加载，其余步保留当前模型
      初始化值（FiLM 为 0、dw_g/ws 为 1，语义正确）。
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        Logger(f"Warning: {label} {checkpoint_path} not found")
        return False

    weights = dict(mx.load(checkpoint_path).items())
    model_shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}

    loaded = {}
    skipped = []
    for key, value in weights.items():
        if key not in model_shapes:
            if strict and not _is_non_strict_key(key):
                raise ValueError(
                    f"{label} {checkpoint_path} 包含模型中没有的参数: {key}"
                )
            skipped.append(key)
            continue
        if value.shape != model_shapes[key]:
            if _is_non_strict_key(key):
                skipped.append(key)
                continue
            if (
                allow_dim0_slice
                and value.ndim >= 1
                and value.shape[1:] == model_shapes[key][1:]
                and value.shape[0] != model_shapes[key][0]
            ):
                # loop_k_override：只切第 0 维（步数），尾部保留当前模型
                # 初始化值；后续 model.update 统一写入。
                current = dict(tree_flatten(model.parameters())).get(key)
                if current is not None:
                    n = min(value.shape[0], model_shapes[key][0])
                    value_f = value.astype(current.dtype)
                    current_f = current.astype(value_f.dtype)
                    padded = mx.concatenate([value_f[:n], current_f[n:]], axis=0)
                    loaded[key] = padded.astype(current.dtype)
                    continue
            if not strict:
                # 宽松加载（如 eval 侧 loop_k_override）：形状不一致且无法
                # 按步数切片的张量跳过，保留当前模型对应参数的初始化值。
                skipped.append(key)
                continue
            raise ValueError(
                f"{label} {checkpoint_path} 参数 {key} 形状不一致: "
                f"checkpoint={value.shape}, model={model_shapes[key]}"
            )
        loaded[key] = value

    missing = [
        key
        for key, shape in model_shapes.items()
        if key not in loaded and not _is_non_strict_key(key)
    ]
    if strict and missing:
        raise ValueError(
            f"{label} {checkpoint_path} 缺少模型参数 "
            f"({len(missing)} 个): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    if loaded:
        model.update(tree_unflatten(list(loaded.items())))
        mx.eval(model.parameters())
    if skipped:
        Logger(
            f"Warning: 跳过 {label} 中 {len(skipped)} 个不匹配/非持久参数: "
            f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
        )
    Logger(f"Loaded {label}: {checkpoint_path}")
    return True


def build_model_and_tokenizer(
    lm_config,
    args,
    checkpoint_name=None,
    checkpoint_label="checkpoint",
    strict=True,
):
    """构造模型与 tokenizer，可选加载权重、可选 dtype 转换并打印参数量。

    基座权重（SFT 加载 pretrain、DPO 加载 SFT）默认 strict=True 且缺失时报错，
    避免静默从随机权重开始训练。

    mx.compile 对 loss 函数的包装在 BaseTrainer 中按 args.compile_model 处理
    （默认启用）。
    """
    model_path = getattr(args, "model_path", "./model/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = VibyForCausalLM(lm_config)

    if checkpoint_name is not None:
        checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
        loaded = load_model_weights(
            model,
            checkpoint_path,
            strict=strict,
            label=checkpoint_label,
        )
        if not loaded:
            raise FileNotFoundError(
                f"{checkpoint_label} 不存在: {checkpoint_path}。"
                "SFT/DPO 必须在已有基座权重上训练，请先完成前置阶段训练或"
                "用 --pretrain_checkpoint/--sft_checkpoint 指定正确路径。"
            )

    convert_model_dtype(model, getattr(args, "dtype", ""))

    log_parameter_count(model)
    return model, tokenizer


# 模型结构相关的 CLI 参数：SFT/DPO 中默认值为 None，表示"从基座 checkpoint 的
# sidecar config 继承"；显式传入时覆盖 sidecar（hidden_size 除外——它还用于
# 推导默认 checkpoint 文件名，保持固定默认值）。
_ARCH_ARG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "head_dim",
    "vocab_size",
    "intermediate_size",
    "use_value_res",
    "use_attn_gate",
    "mtp_depth",
    "mtp_loss_weight",
    "loop_k",
    "dw_rank",
    "ws_loop",
    "hrm_H_cycles",
    "hrm_L_cycles",
    "hrm_bp_cycles",
    "hrm_emb_scale",
    "ffn_type",
    "zero_centered_norm",
    "use_res_gate",
    "sandwich_norm",
    "san_res_init",
    "emb_scale",
    "engram_layers",
    "engram_orders",
    "engram_heads",
    "engram_slots",
    "engram_sub_dim",
)


def load_checkpoint_config(save_dir, checkpoint_name):
    """读取 checkpoint 同名 sidecar JSON 中的 config 字段，不存在则返回 None。"""
    if not checkpoint_name:
        return None
    base = checkpoint_name
    if base.endswith(".safetensors"):
        base = base[: -len(".safetensors")]
    meta_path = os.path.join(save_dir, f"{base}.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    config = meta.get("config")
    if config:
        Logger(f"已从 {meta_path} 继承模型结构配置")
    return config


def build_config_from_sidecar(args, checkpoint_name):
    """以基座 checkpoint 的 sidecar config 为底，CLI 显式参数（非 None）覆盖。

    返回 (config_kwargs, found_sidecar)。无 sidecar 时仅含 CLI 参数，
    其余字段由 VibyConfig 默认值补齐。
    """
    sidecar = load_checkpoint_config(args.save_dir, checkpoint_name)
    cfg = dict(sidecar) if sidecar else {}
    for key in _ARCH_ARG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg, sidecar is not None


def init_wandb(args, trainer):
    if not args.use_wandb or getattr(trainer, "ddp", False):
        return None
    try:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        return wandb
    except ImportError:
        Logger("Warning: wandb not installed, logging disabled")
        return None


def get_lr_and_momentum(
    step: int,
    total_steps: int,
    warmup_steps: int,
    initial_momentum: float = 0.85,
    final_momentum: float = 0.95,
    momentum_warmup_steps: int = 300,
    min_lr_ratio: float = 0.1,
) -> Tuple[float, float]:
    """
    计算当前步骤的学习率乘子和动量。
    - 学习率: Warmup + Cosine Decay（下限 min_lr_ratio × 峰值，0 为衰减到 0）
    - 动量: Linear Warmup
    """
    # --- 学习率调度 ---
    if step < warmup_steps:
        # 线性预热阶段
        lr_multiplier = float(step) / float(max(1, warmup_steps))
    elif step >= total_steps:
        # 训练结束，使用最小学习率
        lr_multiplier = min_lr_ratio
    else:
        # 余弦衰减阶段
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    # --- 动量调度 (仅用于Muon) ---
    momentum = final_momentum
    if step < momentum_warmup_steps:
        frac = float(step) / float(max(1, momentum_warmup_steps))
        momentum = (1 - frac) * initial_momentum + frac * final_momentum

    return lr_multiplier, momentum


def _sub_optimizers(optimizer):
    return getattr(optimizer, "optimizers", [optimizer])


def get_current_lr(optimizer):
    """从优化器读取当前主学习率（MultiOptimizer 时取第一个，即 Muon 组）。"""
    opt = _sub_optimizers(optimizer)[0]
    lr = opt.learning_rate
    if callable(lr):
        # schedule callable：返回存储的 base lr 浮点值
        return float(getattr(opt, "base_lr", 0.0))
    return float(lr)


def _optimizer_hparams(optimizer):
    """提取优化器超参数（lr/momentum 是 python 属性，不在 state 树里）。"""
    hparams = []
    for opt in _sub_optimizers(optimizer):
        h = {}
        lr = opt.learning_rate
        h["learning_rate"] = None if callable(lr) else float(lr)
        if hasattr(opt, "momentum"):
            h["momentum"] = float(opt.momentum)
        hparams.append(h)
    return hparams


def save_checkpoint(model, optimizer, epoch, step, args, lm_config, training_type="pretrain"):
    """统一的检查点保存函数（safetensors 格式）"""
    model.eval()

    # 根据训练类型确定文件名
    prefixes = {
        "pretrain": "pretrain",
        "sft": "full_sft",
        "full_sft": "full_sft",
        "dpo": "dpo",
    }
    prefix = prefixes.get(training_type, training_type)
    ckp_name = f"{prefix}_{lm_config.hidden_size}"

    ckp = os.path.join(args.save_dir, f"{ckp_name}.safetensors")

    # 模型权重
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(ckp, weights)

    # 中间检查点保真：save_interval 命中时额外写一份 step 版本模型快照
    # （仅模型权重、不含优化器），供 (N,D) 缩放律取 D 切面；主 ckp 保持
    # 单文件覆盖以兼容 resume/latest_checkpoint。
    save_interval = int(getattr(args, "save_interval", 0) or 0)
    if save_interval > 0 and step > 0 and step % save_interval == 0:
        step_ckp = os.path.join(args.save_dir, f"{ckp_name}_step{step}.safetensors")
        mx.save_safetensors(step_ckp, weights)
        Logger(f"Step checkpoint saved: {step_ckp}")

    # 优化器状态（打平为扁平字典后保存）
    opt_path = os.path.join(args.save_dir, f"{ckp_name}.optimizer.safetensors")
    mx.save_safetensors(opt_path, dict(tree_flatten(optimizer.state)))

    # 元信息
    meta = {
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "config": lm_config.to_dict(),
        "training_type": training_type,
        "optimizer": _optimizer_hparams(optimizer),
    }
    meta_path = os.path.join(args.save_dir, f"{ckp_name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    Logger(f"Checkpoint saved: {ckp}")

    # 保存最新的检查点路径 (使用绝对路径)
    latest_ckp = os.path.join(args.save_dir, "latest_checkpoint.txt")
    with open(latest_ckp, "w") as f:
        f.write(os.path.abspath(ckp))

    model.train()


def load_checkpoint(checkpoint_path, model, optimizer, args):
    """统一的检查点加载函数（safetensors 格式）"""
    Logger(f"Loading checkpoint from: {checkpoint_path}")

    # 加载模型权重（严格校验，buffer shape 不匹配时保留当前 config 的版本）
    if not load_model_weights(model, checkpoint_path, strict=True, label="checkpoint"):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    # 推导配套文件路径
    base = checkpoint_path
    if base.endswith(".safetensors"):
        base = base[: -len(".safetensors")]
    opt_path = f"{base}.optimizer.safetensors"
    meta_path = f"{base}.json"

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # 加载优化器状态（如未指定重置）
    if not getattr(args, "reset_optimizer", False):
        if os.path.exists(opt_path):
            optimizer.state = tree_unflatten(list(mx.load(opt_path).items()))
            mx.eval(optimizer.state)
            # 恢复 lr/momentum 等 python 属性
            for sub_opt, h in zip(
                _sub_optimizers(optimizer), meta.get("optimizer", [])
            ):
                if h.get("learning_rate") is not None:
                    sub_opt.learning_rate = h["learning_rate"]
                if h.get("momentum") is not None and hasattr(sub_opt, "momentum"):
                    sub_opt.momentum = h["momentum"]

    start_epoch = meta.get("epoch", 0)
    last_finished_step = meta.get("step", 0)

    # 从下一步继续，避免重复已完成的 step
    start_step = int(last_finished_step) + 1

    # 如要求重置优化器，则从step 0开始
    if getattr(args, "reset_optimizer", False):
        start_step = 0
        Logger(
            "reset_optimizer set: optimizer states not loaded; start_step reset to 0"
        )

    Logger(f"Resumed from epoch {start_epoch}, next_step {start_step}")

    return start_epoch, start_step


def find_latest_checkpoint(save_dir):
    """查找最新的检查点文件"""
    latest_file = os.path.join(save_dir, "latest_checkpoint.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            return f.read().strip()
    return None


def apply_lr_schedule(optimizer, global_step, total_training_steps, warmup_iters, min_lr_ratio=0.1):
    """应用学习率调度（每微批 step 调用一次）"""
    lr_multiplier, current_momentum = get_lr_and_momentum(
        global_step,
        warmup_steps=warmup_iters,
        total_steps=total_training_steps,
        min_lr_ratio=min_lr_ratio,
    )

    for opt in _sub_optimizers(optimizer):
        base_lr = getattr(opt, "base_lr", None)
        if base_lr is None:
            lr = opt.learning_rate
            base_lr = 0.0 if callable(lr) else float(lr)
            opt.base_lr = base_lr
        opt.learning_rate = base_lr * lr_multiplier
        # 仅 Muon 有 momentum 属性
        if hasattr(opt, "momentum"):
            opt.momentum = current_momentum


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
    base_step_offset: int = 0,
    mtp_loss=None,
):
    """统一的训练进度日志记录

    current_loss 是含 MTP 加权的总 loss；mtp_loss 为未加权分量（可选）。
    展示格式保持 `loss:<总>` 开头以兼容 results.tsv 的解析。
    """
    spend_time = time.time() - start_time
    # 使用相对步数计算速率，避免 resume 后跳过的步导致速率异常
    effective_steps_done = max(1, (step - base_step_offset + 1))
    steps_per_sec = effective_steps_done / spend_time if spend_time > 0 else 0.0
    tokens_per_sec = steps_per_sec * args.batch_size * args.max_seq_len
    current_lr = get_current_lr(optimizer)

    loss_str = f"{current_loss:.3f}"
    main_loss = None
    if mtp_loss is not None:
        main_loss = current_loss - args.mtp_loss_weight * mtp_loss
        loss_str += f"(main:{main_loss:.3f},mtp:{mtp_loss:.3f})"

    log_msg = "Epoch:[{}/{}]({}/{}) loss:{} lr:{:.2e} grad_norm:{:.3f} step/s:{:.2f} tokens/s:{:.0f} eta:{}min".format(
        epoch + 1,
        args.epochs,
        step,
        iter_per_epoch,
        loss_str,
        current_lr,
        grad_norm,
        steps_per_sec,
        tokens_per_sec,
        int((iter_per_epoch - step - 1) / max(steps_per_sec, 1e-8) / 60),
    )

    Logger(log_msg)

    if wandb is not None:
        log_dict = {
            "loss": current_loss,
            "lr": current_lr,
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "grad_norm": grad_norm,
            "epoch": epoch + 1,
        }
        if mtp_loss is not None:
            log_dict["mtp_loss"] = mtp_loss
            log_dict["main_loss"] = main_loss

        wandb.log(log_dict)
