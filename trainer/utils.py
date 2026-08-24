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
    """按 args.dtype 将模型参数整体转换为 bfloat16/float16（MLX 无 autocast）。"""
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


# 旧 checkpoint 里的非持久 buffer（当前架构已无 RoPE，这些 buffer 不复存在；
# 保留跳过逻辑仅为加载旧权重时不报错）。
_NON_STRICT_WEIGHT_KEYS = (
    "freqs_cos",
    "freqs_sin",
)

# 已从架构中删除的子系统（MLA / HRM / Engram / value-res / CycleDelta /
# RoPE）遗留在旧 checkpoint 里的参数标记：加载时直接跳过，不进 strict 报错。
_LEGACY_WEIGHT_MARKERS = (
    "kv_down",
    "kv_up",
    "k_rope",
    "qkv_proj",
    "rope_freqs",
    ".engrams.",
    "hrm_",
    "v_res_lambda",
    "cycle_v",
    "cycle_g",
    "cycle_u",
    "l_module.",
    "h_module.",
)


def _is_non_strict_key(key: str) -> bool:
    return any(key == k or key.endswith("." + k) for k in _NON_STRICT_WEIGHT_KEYS)


def load_model_weights(
    model,
    checkpoint_path,
    strict=True,
    label="checkpoint",
):
    """安全地加载模型权重。

    - `strict=True` 时，除 `_NON_STRICT_WEIGHT_KEYS` 之外缺少/多余/形状不一致
      的参数都会抛错，避免静默得到随机初始化的部分模型。
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
            # 已删除子系统（MLA/HRM/Engram/value-res/CycleDelta）的旧参数
            # 属于已知可丢弃参数。
            if any(m in key for m in _LEGACY_WEIGHT_MARKERS):
                skipped.append(key)
                continue
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
            if not strict:
                # 宽松加载：形状不一致的张量跳过，保留当前模型对应参数的
                # 初始化值。
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
    "head_dim",
    "vocab_size",
    "intermediate_size",
    "use_attn_gate",
    "mtp_depth",
    "mtp_loss_weight",
    "n_routed_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "moe_intermediate_size",
    "routed_scaling_factor",
    "moe_router_logit_norm",
    "moe_router_logit_temp",
    "moe_diversity_loss_weight",
    "z_loss_weight",
    "moe_latent_dim",
    "tie_word_embeddings",
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


def init_swanlab(args, trainer):
    if not args.use_swanlab or getattr(trainer, "ddp", False):
        return None
    try:
        import swanlab
    except ImportError:
        Logger("Warning: swanlab not installed, logging disabled")
        return None

    # 详细上报超参数：全部 CLI 参数 + 模型配置 + 参数量
    def _clean(d):
        return {
            k: v
            for k, v in d.items()
            if v is None or isinstance(v, (str, int, float, bool))
        }

    config = {f"args.{k}": v for k, v in _clean(vars(args)).items()}
    lm_cfg = getattr(trainer, "lm_config", None)
    if lm_cfg is not None:
        config.update({f"model.{k}": v for k, v in _clean(vars(lm_cfg)).items()})
    try:
        config["model.params_total_m"] = round(trainer.model.num_parameters() / 1e6, 3)
    except Exception:
        pass

    # auto-resume 时尽量接回同一个 swanlab run；id 持久化在 out_dir 下。
    run_id_path = os.path.join(args.save_dir, "swanlab_run_id.txt")
    run_id = None
    if os.path.exists(run_id_path):
        run_id = open(run_id_path, "r", encoding="utf-8").read().strip() or None
    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": args.swanlab_run_name,
        "config": config,
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
        Logger(f"SwanLab resume run: {run_id}")
    else:
        Logger("SwanLab 新建 run（未找到 swanlab_run_id.txt）")
    swanlab.init(**init_kwargs)

    # 首次 init 或新 run 后把 id 落盘，下次 --auto_resume 继续接回。
    try:
        current_run = swanlab.get_run()
        if current_run is not None and getattr(current_run, "id", None):
            with open(run_id_path, "w", encoding="utf-8") as f:
                f.write(str(current_run.id))
    except Exception:
        pass
    return swanlab


def finish_training(swanlab=None, interrupted=False):
    """收掉 swanlab；Ctrl-C 路径用 os._exit 跳过解释器 teardown。

    MLX / SwanLab / 任何漏网的 daemon 线程在 Py_Finalize 阶段再碰 Python
    会 Fatal `PyThreadState_Get`（GIL 已释放）。正常跑完仍走常规退出。
    """
    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass
    if interrupted:
        os._exit(0)


def compute_scaled_hparams(tokens: float, tpb: float, hidden: int):
    """Hyperball 口径的 compute 缩放公式，返回 (adam_lr, muon_lr, beta2, eps)。

    tokens = 总训练 token 预算；tpb = 每个优化器步的 token 数
    （batch_size × accumulation_steps × max_seq_len）；hidden = hidden_size。
    eps 只用于 Adam/AdamH 组，Muon 的 eps 保持 1e-8。
    """
    adam_lr = 0.087571 * tokens**-0.3461 * hidden**-0.3448 * math.sqrt(tpb)
    muon_lr = (13.0 / 3.0) * adam_lr
    beta2 = min(max(0.999 ** (tpb / 131072.0), 0.95), 0.9999)
    eps = 9.676e-18 * math.sqrt(tokens / tpb)
    return adam_lr, muon_lr, beta2, eps


def resolve_compute_scaled_hparams(args, iter_per_epoch: int):
    """解析 pretrain 的 lr/beta2/eps 并写回 args（仅 train_pretrain 调用）。

    优先级：显式 --learning_rate > lr_scale_auto 公式 > 旧常数 0.01。
    - auto 开 + 无手动 lr：adam_lr/muon_lr/beta2/eps 全部按公式；
    - auto 开 + 手动 lr：adam_lr 取手动值，muon_lr 仍 13/3× 派生，
      beta2/eps 按公式；
    - auto 关：单一 lr（手动值或 0.01），betas (0.9, 0.95)，eps 1e-8（旧行为）。

    写回字段：learning_rate（= adam 基础 lr）、muon_lr、adam_beta2、adam_eps、
    token_budget_resolved、tokens_per_batch。SFT/DPO 不调用本函数，
    create_mixed_optimizer 内以 getattr 兜底回退单一 lr 旧行为。
    """
    tpb = args.batch_size * args.accumulation_steps * args.max_seq_len
    tokens = getattr(args, "token_budget", None)
    if tokens is None:
        opt_steps = (args.epochs * iter_per_epoch) // args.accumulation_steps
        tokens = float(opt_steps) * tpb
    tokens = float(tokens)

    manual_lr = args.learning_rate  # None = 用户未显式传入
    if getattr(args, "lr_scale_auto", False):
        adam_lr, muon_lr, beta2, eps = compute_scaled_hparams(
            tokens, tpb, args.hidden_size
        )
        if manual_lr is not None:
            adam_lr = float(manual_lr)
            muon_lr = (13.0 / 3.0) * adam_lr
            src = "手动 --learning_rate 覆盖 adam_lr（beta2/eps 仍按公式）"
        else:
            src = "公式自动推导"
    else:
        adam_lr = muon_lr = float(manual_lr) if manual_lr is not None else 0.01
        beta2, eps = 0.95, 1e-8
        src = "旧常数（--no-lr_scale_auto）"

    args.learning_rate = adam_lr
    args.muon_lr = muon_lr
    args.adam_beta2 = beta2
    args.adam_eps = eps
    args.token_budget_resolved = tokens
    args.tokens_per_batch = tpb
    Logger(
        f"[lr_scale] {src}: tokens={tokens:.3e} tpb={tpb} "
        f"adam_lr={adam_lr:.4e} muon_lr={muon_lr:.4e} beta2={beta2:.5f} eps={eps:.3e}"
    )
    return args


def resolve_warmup_iters(args, total_steps: int):
    """warmup_iters 为 None 时按总步数的 1%（Marin / K3）。显式值不覆盖。"""
    if getattr(args, "warmup_iters", None) is None:
        args.warmup_iters = max(1, int(round(0.01 * float(max(1, total_steps)))))
    return args


def get_lr_and_momentum(
    step: int,
    total_steps: int,
    warmup_steps: int,
    initial_momentum: float = 0.85,
    final_momentum: float = 0.95,
    momentum_warmup_steps: int = 300,
    min_lr_ratio: float = 0.05,
) -> Tuple[float, float]:
    """
    计算当前步骤的学习率乘子和动量。
    - 学习率: 线性 warmup + 线性衰减到 min_lr_ratio（Marin Hero 口径）
    - 动量: Linear Warmup
    """
    # --- 学习率调度 ---
    if step < warmup_steps:
        lr_multiplier = float(step) / float(max(1, warmup_steps))
    elif step >= total_steps:
        lr_multiplier = min_lr_ratio
    else:
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        lr_multiplier = 1.0 - (1.0 - min_lr_ratio) * progress

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


def save_checkpoint(
    model, optimizer, epoch, step, args, lm_config, training_type="pretrain"
):
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


def apply_lr_schedule(
    optimizer, global_step, total_training_steps, warmup_iters, min_lr_ratio=0.05
):
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
    swanlab=None,
    grad_norm=0.0,
    base_step_offset: int = 0,
    mtp_loss=None,
    main_loss=None,
    diversity_loss=None,
    z_loss=None,
    unemb_lar=None,
    extra=None,
):
    """统一的训练进度日志记录

    current_loss 是总 loss（CE + MTP + diversity + z-loss，均已加权）；
    main_loss 是纯语言建模 CE；mtp/diversity/z 为已加权分量（可选）。
    展示格式保持 `loss:<总>` 开头以兼容 results.tsv 的解析。
    extra：调用方附加的详细指标 dict（MoE 负载/内存等），一并上报 swanlab。
    """
    spend_time = time.time() - start_time
    # 使用相对步数计算速率，避免 resume 后跳过的步导致速率异常
    effective_steps_done = max(1, (step - base_step_offset + 1))
    steps_per_sec = effective_steps_done / spend_time if spend_time > 0 else 0.0
    tokens_per_sec = steps_per_sec * args.batch_size * args.max_seq_len
    current_lr = get_current_lr(optimizer)

    loss_str = f"{current_loss:.3f}"
    if main_loss is None and mtp_loss is not None:
        main_loss = current_loss - args.mtp_loss_weight * mtp_loss
    if main_loss is not None:
        loss_str += f"(main:{main_loss:.3f}"
        if mtp_loss is not None:
            loss_str += f",mtp:{mtp_loss:.3f}"
        if diversity_loss not in (None, 0.0):
            loss_str += f",div:{diversity_loss:.4f}"
        if z_loss not in (None, 0.0):
            loss_str += f",z:{z_loss:.5f}"
        loss_str += ")"
    lar_str = ""
    if unemb_lar is not None:
        lar_str = f" unemb_lar:{unemb_lar:.4f}"

    log_msg = "Epoch:[{}/{}]({}/{}) loss:{}{} lr:{:.2e} grad_norm:{:.3f} step/s:{:.2f} tokens/s:{:.0f} eta:{}min".format(
        epoch + 1,
        args.epochs,
        step,
        iter_per_epoch,
        loss_str,
        lar_str,
        current_lr,
        grad_norm,
        steps_per_sec,
        tokens_per_sec,
        int((iter_per_epoch - step - 1) / max(steps_per_sec, 1e-8) / 60),
    )

    Logger(log_msg)

    if swanlab is not None:
        log_dict = {
            "loss": current_loss,
            "lr": current_lr,
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "eta_min": int((iter_per_epoch - step - 1) / max(steps_per_sec, 1e-8) / 60),
            "grad_norm": grad_norm,
            "epoch": epoch + 1,
        }
        if main_loss is not None:
            log_dict["main_loss"] = main_loss
        if mtp_loss is not None:
            log_dict["mtp_loss"] = mtp_loss
        if diversity_loss is not None:
            log_dict["diversity_loss"] = diversity_loss
        if z_loss is not None:
            log_dict["z_loss"] = z_loss
        if unemb_lar is not None:
            log_dict["unembedding_lar"] = unemb_lar
        if extra:
            log_dict.update(extra)

        swanlab.log(log_dict, step=epoch * iter_per_epoch + step)
