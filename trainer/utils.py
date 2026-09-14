"""
训练共享工具函数模块（MLX 单设备版）
包含预训练和SFT训练共享的功能
"""

import os
import json
import math
import time
import hashlib
import random
import subprocess
import struct
from pathlib import Path
import numpy as np
from typing import Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoTokenizer
from model.config import VibyConfig
from model.model import VibyForCausalLM
from .flops import (
    DEFAULT_PEAK_TFLOPS,
    model_flops_utilization,
    training_flops_per_token,
)


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

    keep_router = getattr(model.config, "router_fp32", True)
    model.update(
        tree_unflatten(
            [
                (
                    path,
                    p.astype(
                        mx.float32
                        if keep_router
                        and path.endswith(("router.weight", "router.bias"))
                        else target
                    )
                    if mx.issubdtype(p.dtype, mx.floating)
                    else p,
                )
                for path, p in tree_flatten(params)
            ]
        )
    )
    mx.eval(model.parameters())
    return model


def log_parameter_count(model, args=None):
    total_params = model.num_parameters()
    trainable_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    active_params = model.num_active_parameters()
    # Engram 检索表是稀疏寻址的大表，单列出来避免"总参"里看不出它的占比
    engram_params = model.ngram_lookup_parameters()
    if engram_params:
        Logger(
            f"总参数量：{total_params / 1e6:.3f}M"
            f"（含 Engram n-gram 检索表 {engram_params / 1e6:.3f}M）, "
            f"可训练参数量：{trainable_params / 1e6:.3f}M, "
            f"每次激活：{active_params / 1e6:.3f}M（不含检索表）"
        )
    else:
        Logger(
            f"总参数量：{total_params / 1e6:.3f}M, "
            f"可训练参数量：{trainable_params / 1e6:.3f}M, "
            f"每次激活：{active_params / 1e6:.3f}M"
        )
    if args is None:
        return
    seq = int(getattr(args, "max_seq_len", 0) or 0)
    peak = float(getattr(args, "peak_tflops", None) or DEFAULT_PEAK_TFLOPS)
    args.peak_tflops = peak
    flops = training_flops_per_token(model, seq) if seq > 0 else 0
    args.flops_per_token = flops
    Logger(
        f"训练 FLOPs/token 估算：{flops / 1e9:.3f}G（6×激活 GEMM + 名义 sparse top-k，"
        f"未实测文档掩码/阈值并列；DSpark 按槽数折算），"
        f"peak {peak:.1f} TFLOPS"
    )


# 非持久 buffer（RoPE 表按 config 重算；加载时允许缺失/形状不一致）。
# 新模型的 Attention 把表存成 freq_cos/freq_sin 属性，形状随 max_seq_len 变，
# 跨阶段（pretrain→SFT→DPO 的上下文长度不同）必须允许按当前 config 重算。
_NON_STRICT_WEIGHT_KEYS = (
    "freqs_cos",
    "freqs_sin",
    "rope_freqs",
    "freq_cos",
    "freq_sin",
)

# 已从架构中删除的子系统（HRM / Engram / value-res / CycleDelta）
# 遗留在旧 checkpoint 里的参数标记：加载时直接跳过，不进 strict 报错。
# MLA 的 qkv_proj / kv_up_proj 已恢复，不能再当废弃键跳过。
_LEGACY_WEIGHT_MARKERS = (
    "kv_down",
    "k_rope",
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
    allow_fresh_prefixes=(),
    weights=None,
):
    """安全地加载模型权重。

    - `strict=True` 时，除 `_NON_STRICT_WEIGHT_KEYS` 之外缺少/多余/形状不一致
      的参数都会抛错，避免静默得到随机初始化的部分模型。
    - `allow_fresh_prefixes`：这些前缀下的参数允许 checkpoint 里没有（从零
      初始化）。DSpark 独立阶段（报告 §2.4.3）就是用它：基座预训练时不含
      MTP 模块，草稿层在这一阶段才引入。
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        Logger(f"Warning: {label} {checkpoint_path} not found")
        return False

    weights = dict(mx.load(checkpoint_path).items()) if weights is None else weights
    model_shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}

    loaded = {}
    skipped = []
    for key, value in weights.items():
        if key not in model_shapes:
            # 已删除子系统（HRM/Engram/value-res/CycleDelta）的旧参数
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
        if getattr(model.config, "router_fp32", True) and key.endswith(
            ("router.weight", "router.bias")
        ):
            value = value.astype(mx.float32)
        loaded[key] = value

    missing = [
        key
        for key, shape in model_shapes.items()
        if key not in loaded
        and not _is_non_strict_key(key)
        and not any(key.startswith(pref) for pref in allow_fresh_prefixes)
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


def _reset_rope_tables(model, lm_config):
    """按 config 重算 Attention 的 RoPE 表并冻结。

    新模型把 precompute_freqs_cis 的结果直接存成模块属性（freq_cos/freq_sin），
    它们因此既是 parameters() 里的 2-D 张量、又会被 model/init.py 的
    apply_trunc_normal_init 当成普通矩阵重新随机化（skip_init=False 的从零训练
    路径），还会拿到优化器更新。RoPE 表是由 config 决定的常量：这里按构造期的
    同一口径重算并 freeze（config 侧的等价修法是把它们移出参数树，旧实现用
    object.__setattr__）。冻结只影响 trainable_parameters()，权重仍随 checkpoint
    存盘，strict 加载照旧。
    """
    from model.rope import precompute_freqs_cis

    def _fix(m):
        if not (hasattr(m, "freq_cos") and hasattr(m, "freq_sin")):
            return
        if hasattr(m, "reset_concept_rope"):
            m.reset_concept_rope()
            return
        # 与 Attention.__init__ 同口径：压缩层用 compress_rope_theta + YaRN，
        # 纯滑窗层用 rope_theta 且不做 YaRN
        seq, theta = (
            (lm_config.original_seq_len, lm_config.compress_rope_theta)
            if getattr(m, "ratio", 0)
            else (0, lm_config.rope_theta)
        )
        c, s = precompute_freqs_cis(
            int(m.rope_head_dim),
            int(lm_config.max_seq_len),
            int(seq),
            float(theta),
            float(lm_config.rope_factor),
            int(lm_config.beta_fast),
            int(lm_config.beta_slow),
        )
        # 保持 dtype 转换后的口径（convert_model_dtype 已把表转成 bf16/fp16）
        dtype = m.freq_cos.dtype
        m.freq_cos, m.freq_sin = c.astype(dtype), s.astype(dtype)

    model.apply_to_modules(lambda name, m: _fix(m))
    model.freeze(keys=["freq_cos", "freq_sin"])
    mx.eval(model.parameters())


def freeze_backbone_for_dspark(model):
    """DSpark 独立训练阶段：只留 mtp_modules.* 可训练（V4.1 §2.4.3）。

    先整树 freeze 再解冻草稿层。freeze 只改 trainable_parameters()：checkpoint
    的保存/加载仍覆盖全部权重。trainer 把 trainable 参数显式喂给被 compile 的
    loss 函数，冻结后参数叶子变少会自动触发图重建，无需额外处理。
    """
    model.freeze()
    for stage in getattr(model, "mtp_modules", []):
        stage.unfreeze()
    return model


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
    # 严格加载基座时全部参数都会被 checkpoint 覆盖，跳过随机初始化
    model = VibyForCausalLM(lm_config, skip_init=checkpoint_name is not None and strict)

    if checkpoint_name is not None:
        checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
        validate_checkpoint_execution(checkpoint_path, lm_config, args)
        loaded = load_model_weights(
            model,
            checkpoint_path,
            strict=strict,
            label=checkpoint_label,
            # DSpark 独立阶段：基座 checkpoint 没有草稿层参数，从零初始化
            allow_fresh_prefixes=("mtp_modules.",)
            if getattr(args, "freeze_backbone", False)
            else (),
        )
        if not loaded:
            raise FileNotFoundError(
                f"{checkpoint_label} 不存在: {checkpoint_path}。"
                "SFT/DPO 必须在已有基座权重上训练，请先完成前置阶段训练或"
                "用 --pretrain_checkpoint/--sft_checkpoint 指定正确路径。"
            )

    convert_model_dtype(model, getattr(args, "dtype", ""))

    # RoPE 表是常量：重算 + 冻结（见 _reset_rope_tables 的说明）
    _reset_rope_tables(model, lm_config)
    if getattr(args, "freeze_backbone", False):
        freeze_backbone_for_dspark(model)
        n_train = len(tree_flatten(model.trainable_parameters()))
        Logger(
            "--freeze_backbone：只训练 mtp_modules.*（论文 §2.4.3 的 DSpark "
            f"独立训练阶段），可训练张量 {n_train} 个"
        )

    log_parameter_count(model, args)
    return model, tokenizer


# 模型结构相关的 CLI 参数 → VibyConfig 字段名。唯一真源：trainer/config.py 用
# 它生成 parser（默认值取 VibyConfig 的 ≈1B 配方），这里用它把 CLI 覆盖写进
# sidecar config / 组成 VibyConfig 的 kwargs。SFT/DPO 的 parser 把这些参数默认
# 设为 None：表示"从基座 checkpoint 的 sidecar config 继承"，显式传入才覆盖。
ARCH_ARG_TO_FIELD = {
    **{
        name: name
        for name in (
            "ced_recurrent_enabled",
            "ced_recurrent_stride",
            "ced_recurrent_rounds",
        )
    },
    "hidden_size": "dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "head_dim": "head_dim",
    "rope_head_dim": "rope_head_dim",
    "q_lora_rank": "q_lora_rank",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    "window_size": "window_size",
    "use_xsa": "use_xsa",
    "xsa_last_n": "xsa_last_n",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "compress_ratios": "compress_ratios",
    "kv_source_layers": "kv_source_layers",
    "index_source_layers": "index_source_layers",
    "candidate_source_layer": "candidate_source_layer",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "rope_theta": "rope_theta",
    "compress_rope_theta": "compress_rope_theta",
    "original_seq_len": "original_seq_len",
    "rope_factor": "rope_factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "n_routed_experts": "n_routed_experts",
    "num_experts_per_tok": "n_activated_experts",
    "n_shared_experts": "n_shared_experts",
    "moe_intermediate_size": "moe_inter_dim",
    "score_func": "score_func",
    "gate_temp": "gate_temp",
    "route_scale": "route_scale",
    "swiglu_limit": "swiglu_limit",
    "bias_update_rate": "bias_update_rate",
    "moe_balance_method": "moe_balance_method",
    "qb_update_rate": "qb_update_rate",
    "qb_stats_rows": "qb_stats_rows",
    "router_fp32": "router_fp32",
    "aux_balance_loss_weight": "aux_balance_loss_weight",
    "engram_layer_ids": "engram_layer_ids",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "mtp_depth": "n_mtp_layers",
    "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_n_activated_experts": "dspark_n_activated_experts",
    "mtp_loss_weight": "mtp_loss_weight",
    "z_loss_weight": "z_loss_weight",
    "norm_topk_prob": "norm_topk_prob",
    "tie_word_embeddings": "tie_word_embeddings",
    "vocab_size": "vocab_size",
}

# 旧名别名：命令行选项名 → ARCH_ARG_TO_FIELD 里的 dest
# （--routed_scaling_factor 与 --route_scale 共用一个 dest，这里只用于
#  --preset 判断"用户是否显式传过"）
ARCH_ARG_ALIASES = {
    "routed_scaling_factor": "route_scale",
    "ced_recurrent": "ced_recurrent_enabled",
    "no_ced_recurrent": "ced_recurrent_enabled",
}

# 派生字段 → 生成它们的 CLI 结构参数。sidecar 里的派生字段是按当时的结构算好的；
# 只要用户显式改了生成它的结构参数（如 --num_hidden_layers / --mtp_depth），
# 就必须丢掉旧值让 VibyConfig 重算，否则会出现 compress_ratios 长度与新层数
# 不符、dspark_target_layer_ids 越界这类自相矛盾。
DERIVED_FIELD_TRIGGERS = {
    "compress_ratios": ("num_hidden_layers", "mtp_depth"),
    "kv_source_layers": ("num_hidden_layers", "mtp_depth"),
    "index_source_layers": ("num_hidden_layers", "mtp_depth"),
    "candidate_source_layer": ("num_hidden_layers", "mtp_depth"),
    "dspark_target_layer_ids": ("num_hidden_layers", "mtp_depth"),
    "engram_layer_ids": ("num_hidden_layers",),
    "xsa_last_n": ("num_hidden_layers",),
    "engram_num_embeddings": (
        "engram_layer_ids",
        "engram_max_ngram_size",
        "engram_vocab_size",
        "engram_n_heads",
    ),
}

# 列表类结构参数：命令行/JSON 里是逗号或空格分隔的整数串
ARCH_LIST_ARGS = (
    "compress_ratios",
    "kv_source_layers",
    "index_source_layers",
    "engram_layer_ids",
    "dspark_target_layer_ids",
)


def _as_int_tuple(value):
    """列表类结构参数统一成 tuple[int]（VibyConfig 接受 list/tuple 两种）。"""
    if value is None or isinstance(value, tuple):
        return value
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    return tuple(int(x) for x in value)


def build_model_kwargs(args) -> dict:
    """CLI 结构参数 → VibyConfig kwargs（pretrain 从零建模型的路径）。

    只带显式传入（非 None）的字段与 --preset：其余交给 VibyConfig 的默认值 /
    preset 预设，CLI 不复制一份配方（避免两处漂移）。
    """
    kw = {}
    preset = getattr(args, "preset", None)
    if preset:
        kw["preset"] = preset
    for arg, field in ARCH_ARG_TO_FIELD.items():
        value = getattr(args, arg, None)
        if value is not None:
            kw[field] = _as_int_tuple(value) if arg in ARCH_LIST_ARGS else value
    # 序列长度：RoPE 表按 --max_seq_len 建（数据侧用的是同一个参数；
    # 不放进 ARCH_ARG_TO_FIELD 是为了不让 SFT/DPO 把它抹成 None）
    seq = getattr(args, "max_seq_len", None)
    if seq:
        kw["max_seq_len"] = int(seq)
    return kw


def base_checkpoint_name(args, lm_config, explicit_attr, prefix):
    """基座 checkpoint 文件名：显式传入优先，否则按 config 的 dim 推导。"""
    name = getattr(args, explicit_attr, None)
    if name:
        return name
    return f"{prefix}_{int(lm_config.dim)}.safetensors"


def sidecar_checkpoint_hint(args, explicit_attr, prefix):
    """查找 sidecar 用的默认文件名（此刻 lm_config 还不存在）。

    未显式传入 checkpoint 名时，用 --hidden_size（没传就用 VibyConfig 默认
    dim）推导；换过 hidden_size 的基座用显式 --pretrain_checkpoint/--sft_checkpoint
    指定即可。
    """
    name = getattr(args, explicit_attr, None)
    if name:
        return name
    dim = getattr(args, "hidden_size", None) or VibyConfig().dim
    return f"{prefix}_{int(dim)}.safetensors"


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


def checkpoint_execution(config):
    """Execution identity independent of shared parameter names/shapes."""
    cfg = config if isinstance(config, dict) else vars(config)
    if not cfg.get("ced_recurrent_enabled", False):
        return {"kind": "token_ced_v1"}
    return {
        "kind": cfg.get("ced_recurrent_arch", "residual_lift_v1"),
        "stride": int(cfg.get("ced_recurrent_stride", 4)),
        "rounds": int(cfg.get("ced_recurrent_rounds", 3)),
    }


def validate_checkpoint_execution(checkpoint_path, config, args, *, automatic=False):
    """Require an explicit optimizer-reset warm start when CED execution changes.

    Legacy sidecars without recurrent fields denote token CED. Missing metadata
    cannot attest to recurrent execution and thus also requires conversion.
    Returns True for a conversion, before any model/optimizer mutation occurs.
    """
    meta_path = os.path.splitext(os.fspath(checkpoint_path))[0] + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)
    source = meta.get("execution") or checkpoint_execution(meta.get("config", {}))
    target = checkpoint_execution(config)
    if source == target:
        return False
    explicit = (
        not automatic
        and not getattr(args, "auto_resume", False)
        and bool(getattr(args, "reset_optimizer", False))
    )
    if not explicit:
        raise ValueError(
            f"Checkpoint CED execution {source} differs from requested {target}. "
            "Automatic resume cannot convert execution. Use an explicit checkpoint "
            "with --reset_optimizer for a warm start in a fresh output directory."
        )
    Logger(f"Explicit CED warm start: {source} -> {target}; optimizer/progress reset")
    return True


def build_config_from_sidecar(args, checkpoint_name):
    """以基座 checkpoint 的 sidecar config 为底，CLI 显式参数（非 None）覆盖。

    返回 (config_kwargs, found_sidecar)。sidecar 的键已经是 VibyConfig 的字段名
    （to_dict 的输出），CLI 侧经 ARCH_ARG_TO_FIELD 映射；无 sidecar 时只含 CLI
    结构参数与 --preset，其余字段由 VibyConfig 默认值补齐。
    """
    sidecar = load_checkpoint_config(args.save_dir, checkpoint_name)
    cfg = dict(sidecar) if sidecar else {}
    given = set()
    for arg, field in ARCH_ARG_TO_FIELD.items():
        value = getattr(args, arg, None)
        if value is not None:
            cfg[field] = _as_int_tuple(value) if arg in ARCH_LIST_ARGS else value
            given.add(arg)
    if sidecar:
        # 结构生成参数被显式改动时，sidecar 里的派生字段作废（见
        # DERIVED_FIELD_TRIGGERS 的说明）；用户显式传了的派生字段保留
        for field, triggers in DERIVED_FIELD_TRIGGERS.items():
            if (
                field in cfg
                and field not in given
                and any(t in given for t in triggers)
            ):
                cfg.pop(field)
    else:
        preset = getattr(args, "preset", None)
        if preset:
            cfg["preset"] = preset
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
        config["model.params_active_m"] = round(
            trainer.model.num_active_parameters() / 1e6, 3
        )
        ngram_m = trainer.model.ngram_lookup_parameters() / 1e6
        if ngram_m:
            config["model.params_ngram_m"] = round(ngram_m, 3)
        flops = getattr(args, "flops_per_token", None)
        if flops:
            config["model.flops_per_token"] = int(flops)
        peak = getattr(args, "peak_tflops", None)
        if peak:
            config["train.peak_tflops"] = float(peak)
    except Exception:
        pass

    # auto-resume 时尽量接回同一个 swanlab run；id 按训练阶段持久化在 out_dir
    # 下——pretrain/SFT/DPO 共享 out_dir，共用一个 run 会把不同阶段的
    # loss 曲线混到同一块面板上。旧的无阶段后缀文件仅 pretrain 兼容回退。
    training_type = getattr(trainer, "training_type", "pretrain")
    run_id_path = os.path.join(args.save_dir, f"swanlab_run_id_{training_type}.txt")
    if not os.path.exists(run_id_path) and training_type == "pretrain":
        legacy = os.path.join(args.save_dir, "swanlab_run_id.txt")
        if os.path.exists(legacy):
            run_id_path = legacy
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

    SwanLab 0.10 在 SIGINT handler 里已经把 run 标成 aborted 并 finish
    （日志里的 "KeyboardInterrupt by user" / "Upload complete"）。这里再
    调一次 finish() 会打 `Run has already finished or has not started`。
    """
    if swanlab is not None:
        try:
            has_run = getattr(swanlab, "has_run", None)
            if callable(has_run) and not has_run():
                pass
            else:
                swanlab.finish()
        except Exception:
            pass
    if interrupted:
        os._exit(0)


# Marin Hero MoeHeuristic.max_learning_rate（issue #8435）：公式推出的
# adam_lr / muon_lr 都截到该上限；手动 --learning_rate 不截。
_MARIN_MAX_LR = 0.05


def compute_scaled_hparams(tokens: float, tpb: float, hidden: int):
    """Hyperball / Marin Hero 口径的 compute 缩放公式，返回
    (adam_lr, muon_lr, beta2, eps)。

    tokens = 本轮实际 token 预算；tpb = 每个优化器步的 token 数
    （batch_size × accumulation_steps × max_seq_len）；hidden = hidden_size。
    公式与 Marin ``MoeHeuristic`` 一致（#7856 / #8003 / #8435）：
    adam_lr = 0.087571 · tokens^-0.3461 · hidden^-0.3448 · √tpb，
    muon_lr = 13/3 × adam_lr，二者再截到 0.05。
    eps 只用于 Adam/AdamH 组，Muon 的 eps 保持 1e-8。
    """
    adam_lr = 0.087571 * tokens**-0.3461 * hidden**-0.3448 * math.sqrt(tpb)
    adam_lr = min(_MARIN_MAX_LR, adam_lr)
    muon_lr = min(_MARIN_MAX_LR, (13.0 / 3.0) * adam_lr)
    beta2 = min(max(0.999 ** (tpb / 131072.0), 0.95), 0.9999)
    eps = 9.676e-18 * math.sqrt(tokens / tpb)
    return adam_lr, muon_lr, beta2, eps


def resolve_lr_horizon(args, epoch_steps: int) -> int:
    """学习率日程所用的总微批步数（与日志 (step/N) 同口径）。

    对齐 Marin #8435：linear · warmup 1% · 终点落到 min_lr_ratio（默认 0.05）。
    缩短 run 时重算衰减斜率，结束时仍是 0.05×peak——不是按原全量步数走、
    以至于短跑从不退火。80% 处分相是 datamix，不是 LR 平台（那是 WSD）。

    优先级：显式 --lr_decay_steps（按该长度排程，即使长于本轮数据；
    用来复现「长 run 的前缀」）> --max_steps（截到 epochs×每轮，即实际
    会跑的步数）> epochs×每轮步数。
    """
    epoch_total = int(getattr(args, "epochs", 1)) * int(epoch_steps)
    lr_decay_steps = getattr(args, "lr_decay_steps", None)
    if lr_decay_steps:
        return int(lr_decay_steps)
    max_steps = getattr(args, "max_steps", None)
    if max_steps:
        return min(int(max_steps), epoch_total)
    return epoch_total


def _token_budget_from_horizon(args, iter_per_epoch: int) -> float:
    """未传 --token_budget 时，用与 LR 日程相同的 horizon 推 token 数。

    Marin ladder 每档：tokens = num_train_steps × batch × seq。
    显式 --token_budget 则保留「原计划预算」（缩短 horizon 时只改衰减斜率、
    峰值仍按原预算——Hero 中途砍 token 上限的做法）。
    """
    horizon = resolve_lr_horizon(args, iter_per_epoch)
    accum = max(1, int(getattr(args, "accumulation_steps", 1)))
    tpb = args.batch_size * accum * args.max_seq_len
    opt_steps = max(1, int(horizon) // accum)
    return float(opt_steps) * float(tpb)


def resolve_compute_scaled_hparams(args, iter_per_epoch: int):
    """解析 pretrain 的 lr/beta2/eps 并写回 args（仅 train_pretrain 调用）。

    优先级：显式 --learning_rate > lr_scale_auto 公式 > 旧常数 0.01。
    - auto 开 + 无手动 lr：adam_lr/muon_lr/beta2/eps 全部按公式；
    - auto 开 + 手动 lr：adam_lr 取手动值，muon_lr 仍 13/3× 派生，
      beta2/eps 按公式（手动值不套 0.05 上限）；
    - auto 关：单一 lr（手动值或 0.01），betas (0.9, 0.95)，eps 1e-8（旧行为）。

    未传 --token_budget 时预算与 LR 日程 horizon 一致（max_steps /
    lr_decay_steps / 全量 epoch）。写回字段：learning_rate（= adam 基础 lr）、
    muon_lr、adam_beta2、adam_eps、token_budget_resolved、tokens_per_batch。
    SFT/DPO 不调用本函数，create_mixed_optimizer 内以 getattr 兜底回退单一 lr。
    """
    tpb = args.batch_size * args.accumulation_steps * args.max_seq_len
    tokens = getattr(args, "token_budget", None)
    if tokens is None:
        tokens = _token_budget_from_horizon(args, iter_per_epoch)
        tok_src = "horizon"
    else:
        tok_src = "token_budget"
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
        f"[lr_scale] {src} ({tok_src}): tokens={tokens:.3e} tpb={tpb} "
        f"adam_lr={adam_lr:.4e} muon_lr={muon_lr:.4e} beta2={beta2:.5f} eps={eps:.3e}"
    )
    return args


def resolve_warmup_iters(args, total_steps: int):
    """warmup_iters 为 None 时按本轮 horizon 的 1%（Marin #8435）。显式值不覆盖。"""
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
    schedule: str = "linear",
    wsd_decay_frac: float = 0.2,
    wsd_decay_shape: str = "linear",
) -> Tuple[float, float]:
    """
    计算当前步骤的学习率乘子和动量。
    - linear: warmup 后立刻线性收到 min_lr_ratio（Marin Hero #8435：
      ``lr_schedule=linear``、``warmup=0.01``、``decay=None``、``min_lr_ratio=0.05``。
      80% 处分相是 datamix 换源，不是 LR 平台。）
    - wsd: 线性 warmup + 平台期 + 末尾 wsd_decay_frac 线性收到 min_lr_ratio
    - 动量: Linear Warmup

    函数默认 schedule=linear，避免旧单测无参调用改义；CLI / trainer
    默认同样是 linear（见 --lr_schedule）。min_lr_ratio 默认 0.05 而不是 0，
    因为后面还有 context extension / 后训练；decay-to-zero 只适合不再更新
    权重的单阶段预训练。
    """
    # --- 学习率调度 ---
    if step < warmup_steps:
        lr_multiplier = float(step) / float(max(1, warmup_steps))
    elif step >= total_steps:
        lr_multiplier = min_lr_ratio
    elif schedule == "wsd":
        decay_start = int(round(float(total_steps) * (1.0 - float(wsd_decay_frac))))
        decay_start = max(decay_start, warmup_steps)
        if step < decay_start:
            lr_multiplier = 1.0
        else:
            progress = float(step - decay_start) / float(
                max(1, total_steps - decay_start)
            )
            if wsd_decay_shape == "cosine":
                # 报告 §4.2.2：平台后用余弦从 peak 衰减到 min_lr_ratio
                import math as _math

                cosine = 0.5 * (1.0 + _math.cos(_math.pi * min(progress, 1.0)))
                lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            else:
                lr_multiplier = 1.0 - (1.0 - min_lr_ratio) * progress
    else:
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        lr_multiplier = 1.0 - (1.0 - min_lr_ratio) * progress

    # --- 动量调度 (仅用于Muon) ---
    # 动量 warmup（0.85→0.95 前 300 步）默认关闭：隔离 probe 实测单独
    # 无害也无益（probe_p9 与基线逐点重合），而 r081 健康基线本就没有
    # 它——"早期无害"≠"长程无害"，保持与已验证基线一致。
    # VIBY_MUONH_MOM_WARMUP=1 可打开（消融用）。
    if os.environ.get("VIBY_MUONH_MOM_WARMUP", "0") != "1":
        momentum_warmup_steps = 0
    momentum = final_momentum
    if step < momentum_warmup_steps:
        frac = float(step) / float(max(1, momentum_warmup_steps))
        momentum = (1 - frac) * initial_momentum + frac * final_momentum

    return lr_multiplier, momentum


def _sub_optimizers(optimizer):
    return getattr(optimizer, "optimizers", [optimizer])


def get_optimizer_steps(optimizer):
    """Read leaf clocks; MultiOptimizer has states[], not a top-level step.

    Keep all group clocks in logs instead of assuming they must be identical.
    This also works before the first update and after state restoration.
    """
    children = getattr(optimizer, "optimizers", None)
    if children is not None:
        return [step for child in children for step in get_optimizer_steps(child)]
    return [int(optimizer.step)]


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
        h = {"class": type(opt).__name__}
        for key, value in vars(opt).items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, (tuple, list))
                and all(isinstance(x, (int, float)) for x in value)
            ):
                h[key] = value
        lr = opt.learning_rate
        h["learning_rate"] = None if callable(lr) else float(lr)
        if hasattr(opt, "momentum"):
            h["momentum"] = float(opt.momentum)
        hparams.append(h)
    return hparams


def _artifact_sha256(path, common_only=False):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        if not common_only:
            for chunk in iter(lambda: file.read(8 << 20), b""):
                digest.update(chunk)
        else:
            size = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(size))
            for name, entry in sorted(header.items()):
                if name == "__metadata__":
                    continue
                digest.update(
                    json.dumps([name, entry["dtype"], entry["shape"]]).encode()
                )
                start, end = entry["data_offsets"]
                file.seek(8 + size + start)
                remaining = end - start
                while remaining:
                    chunk = file.read(min(remaining, 8 << 20))
                    if not chunk:
                        raise ValueError("truncated checkpoint while hashing")
                    digest.update(chunk)
                    remaining -= len(chunk)
    return digest.hexdigest()


def _restore_optimizer_hparams(optimizer, records):
    for opt, record in zip(_sub_optimizers(optimizer), records or []):
        for key, value in record.items():
            if key in vars(opt) and key != "_initialized":
                old = getattr(opt, key)
                if isinstance(old, (str, int, float, bool, list, tuple)):
                    setattr(opt, key, tuple(value) if isinstance(old, tuple) else value)
        if record.get("learning_rate") is not None:
            opt.learning_rate = record["learning_rate"]


def save_checkpoint(
    model, optimizer, epoch, step, args, lm_config, training_type="pretrain"
):
    """统一的检查点保存函数（safetensors 格式）"""
    if getattr(args, "no_save", False):
        return
    model.eval()

    # 根据训练类型确定文件名
    prefixes = {
        "pretrain": "pretrain",
        "sft": "full_sft",
        "full_sft": "full_sft",
        "dpo": "dpo",
    }
    prefix = prefixes.get(training_type, training_type)
    ckp_name = f"{prefix}_{int(lm_config.dim)}"

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

    # sidecar / latest 必须在优化器之前：optimizer 常数 GB，Ctrl-C 或写失败
    # 会留下有权重无 json，eval 只能用残缺 CLI 配置建 MoE 然后炸掉。
    meta = {
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "config": lm_config.to_dict(),
        "execution": checkpoint_execution(lm_config),
        "training_type": training_type,
        "optimizer": _optimizer_hparams(optimizer),
    }
    if training_type == "sft" and getattr(args, "sft_algorithm", "standard") == "tail":
        meta["rng"] = {
            "mlx_key": list(mx.random.state)[0].tolist(),
            "python": random.getstate(),
            "numpy_epoch_start": getattr(args, "epoch_shuffle_state", None),
        }
    if any(
        getattr(lm_config, flag, False)
        for flag in ("ced_recurrent_enabled",)
    ):
        meta["code_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        meta["dirty_diff_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "diff"])
        ).hexdigest()
        root = Path(__file__).resolve().parents[1]
        meta["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("model", "trainer")
            for p in sorted((root / folder).rglob("*.py"))
        }
        meta["tokenizer_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(getattr(args, "model_path", "model")).glob("*token*.json")
        }
        meta["rng"] = {
            "mlx_key": list(mx.random.state)[0].tolist(),
            "python": random.getstate(),
            "numpy_epoch_start": getattr(args, "epoch_shuffle_state", None),
        }
        filters = getattr(optimizer, "filters", [lambda *_: True])
        group_key = "optimizer_parameter_groups"
        meta[group_key] = [[] for _ in filters]
        for key, value in tree_flatten(model.trainable_parameters()):
            for i, fn in enumerate(filters):
                if fn(key, value):
                    meta[group_key][i].append(key)
                    break
    meta_path = os.path.join(args.save_dir, f"{ckp_name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if save_interval > 0 and step > 0 and step % save_interval == 0:
        # A shared-weight recurrent snapshot cannot be identified by tensor keys.
        with open(os.path.splitext(step_ckp)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    latest_ckp = os.path.join(args.save_dir, "latest_checkpoint.txt")
    with open(latest_ckp, "w") as f:
        f.write(os.path.abspath(ckp))

    opt_path = os.path.join(args.save_dir, f"{ckp_name}.optimizer.safetensors")
    mx.save_safetensors(opt_path, dict(tree_flatten(optimizer.state)))

    if any(
        getattr(lm_config, flag, False)
        for flag in ("ced_recurrent_enabled",)
    ):
        meta["common_weights_sha256"] = _artifact_sha256(ckp, common_only=True)
        meta["optimizer_sha256"] = _artifact_sha256(opt_path)
        with open(meta_path, "w") as file:
            json.dump(meta, file, indent=2, default=str)
    Logger(f"Checkpoint saved: {ckp}")

    model.train()


def load_checkpoint(checkpoint_path, model, optimizer, args):
    """统一的检查点加载函数（safetensors 格式）"""
    Logger(f"Loading checkpoint from: {checkpoint_path}")
    validate_checkpoint_execution(checkpoint_path, model.config, args)
    weights = dict(mx.load(checkpoint_path).items())
    fresh_prefixes = (
        ("mtp_modules.",) if getattr(args, "freeze_backbone", False) else ()
    )

    # 加载模型权重（严格校验，buffer shape 不匹配时保留当前 config 的版本）。
    # --freeze_backbone 的 DSpark 独立阶段（报告 §2.4.3）允许基座 checkpoint
    # 里没有 mtp_modules.*：草稿层在该阶段才引入，从零初始化。
    if not load_model_weights(
        model,
        checkpoint_path,
        strict=True,
        label="checkpoint",
        allow_fresh_prefixes=fresh_prefixes,
        weights=weights,
    ):
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

    # 加载优化器状态（如未指定重置）。--freeze_backbone 的 DSpark 阶段可训练集合
    # 与基座训练不同（只有 mtp_modules.*），优化器分组天然不一致，必须重置。
    if getattr(args, "freeze_backbone", False) and not getattr(
        args, "reset_optimizer", False
    ):
        Logger(
            "--freeze_backbone：可训练集合与基座不同，自动跳过优化器状态（等价 --reset_optimizer）"
        )
    if not getattr(args, "reset_optimizer", False) and not getattr(
        args, "freeze_backbone", False
    ):
        if os.path.exists(opt_path):
            optimizer.state = tree_unflatten(list(mx.load(opt_path).items()))
            mx.eval(optimizer.state)
            _restore_optimizer_hparams(optimizer, meta.get("optimizer", []))

    start_epoch = meta.get("epoch", 0)
    last_finished_step = meta.get("step", 0)

    # 从下一步继续，避免重复已完成的 step
    start_step = int(last_finished_step) + 1

    # 如要求重置优化器，则从step 0开始
    if getattr(args, "reset_optimizer", False):
        start_epoch = 0
        start_step = 0
        Logger(
            "reset_optimizer set: optimizer states not loaded; start_step reset to 0"
        )

    if not getattr(args, "reset_optimizer", False) and meta.get("rng"):
        rng = meta["rng"]
        high, low = rng["mlx_key"]
        mx.random.seed((int(high) << 32) | int(low))

        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value

        random.setstate(tuples(rng["python"]))
        state = rng.get("numpy_epoch_start")
        if state:
            np.random.set_state(
                (state[0], np.asarray(state[1], dtype=np.uint32), *state[2:])
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
    optimizer,
    global_step,
    total_training_steps,
    warmup_iters,
    min_lr_ratio=0.05,
    schedule="linear",
    wsd_decay_frac=0.2,
    wsd_decay_shape="linear",
):
    """应用学习率调度（每微批 step 调用一次）。

    函数默认 schedule=linear / wsd_decay_shape=linear（保持旧单测语义）；
    CLI 默认是 wsd + cosine（对齐报告 §4.2.2）。
    """
    lr_multiplier, current_momentum = get_lr_and_momentum(
        global_step,
        warmup_steps=warmup_iters,
        total_steps=total_training_steps,
        min_lr_ratio=min_lr_ratio,
        schedule=schedule,
        wsd_decay_frac=wsd_decay_frac,
        wsd_decay_shape=wsd_decay_shape,
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


# 实时速率统计（模块级状态）：以上一次日志为窗口计算瞬时 step/s，
# 再做 EMA 平滑，避免“从训练开始以来的累计平均”掩盖近期速度变化。
_speed_state = {"last_step": None, "last_time": None, "ema": None}
_SPEED_EMA_ALPHA = 0.3  # EMA 权重：越大越贴近瞬时值，越小越平滑


def _realtime_steps_per_sec(step, cumulative_steps_per_sec):
    """根据相邻两次日志的步数差/时间差计算实时 step/s（EMA 平滑）。

    首次调用或跨 epoch step 重置时窗口无效，回退到累计平均速率。
    """
    now = time.time()
    last_step = _speed_state["last_step"]
    last_time = _speed_state["last_time"]
    if last_step is not None and last_time is not None:
        d_step = step - last_step
        d_time = now - last_time
        if d_step > 0 and d_time > 0:
            inst = d_step / d_time
            ema = _speed_state["ema"]
            ema = (
                inst
                if ema is None
                else _SPEED_EMA_ALPHA * inst + (1 - _SPEED_EMA_ALPHA) * ema
            )
            _speed_state["ema"] = ema
    _speed_state["last_step"] = step
    _speed_state["last_time"] = now
    ema = _speed_state["ema"]
    return ema if ema is not None else cumulative_steps_per_sec


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
    cumulative_steps_per_sec = (
        effective_steps_done / spend_time if spend_time > 0 else 0.0
    )
    # 实时速率：相邻两次日志窗口的瞬时速率做 EMA，首次/跨 epoch 回退累计平均
    steps_per_sec = _realtime_steps_per_sec(step, cumulative_steps_per_sec)
    tokens_per_sec = steps_per_sec * args.batch_size * args.max_seq_len
    flops_per_token = float(getattr(args, "flops_per_token", 0) or 0)
    peak_tflops = float(getattr(args, "peak_tflops", None) or DEFAULT_PEAK_TFLOPS)
    mfu = model_flops_utilization(tokens_per_sec, flops_per_token, peak_tflops)
    achieved_tflops = (
        tokens_per_sec * flops_per_token / 1e12 if flops_per_token else 0.0
    )
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

    mfu_str = f" mfu:{mfu:.1%}" if flops_per_token else ""
    log_msg = "Epoch:[{}/{}]({}/{}) loss:{}{} lr:{:.2e} grad_norm:{:.3f} step/s:{:.2f} tokens/s:{:.0f}{} eta:{}min".format(
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
        mfu_str,
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
        if flops_per_token:
            log_dict["mfu"] = mfu
            log_dict["tflops"] = achieved_tflops
        if extra:
            log_dict.update(extra)

        swanlab.log(log_dict, step=epoch * iter_per_epoch + step)
