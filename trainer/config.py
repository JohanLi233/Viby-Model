"""
训练配置和参数解析模块
"""

import argparse
from .utils import Logger


def add_common_args(parser):
    """添加通用参数"""
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--device",
        type=str,
        default="mlx",
        help="仅作信息展示；MLX 使用统一内存，无设备概念",
    )
    # 模型结构参数（与 eval_model.py 保持一致，默认 768/8）
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument(
        "--kv_lora_rank",
        type=int,
        default=192,
        help="MLA 的 KV 低秩潜在维度",
    )
    parser.add_argument(
        "--qk_rope_head_dim",
        type=int,
        default=32,
        help="MLA 解耦 RoPE 键的维度（跨 head 共享）",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="默认 hidden_size // num_attention_heads",
    )
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="默认按 hidden_size 自动计算",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=True,
        help="使用 mx.compile 编译 loss 函数（默认开启）",
    )
    parser.add_argument("--no_compile", action="store_false", dest="compile_model")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument("--pin_memory", action="store_true", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--cache_limit_gb",
        type=float,
        default=0,
        help="Metal 分配器空闲块缓存上限（GB），0=不限制。"
        "上限内的释放块常驻复用、不归还 OS；过大在 bs16x640 以上的"
        "大配置会挤占活跃内存（实测最优点 24G，峰值+缓存≈40G）",
    )
    parser.add_argument(
        "--max_train_minutes",
        type=float,
        default=None,
        help="最长训练时长（分钟）。到时后在当前梯度累积窗口边界停止并保存 "
        "checkpoint（不会因 resume 丢梯度）；默认不限制",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="最多训练多少个微批 step（与日志 (step/N) 同口径）。到步数后在当前"
        "梯度累积窗口边界停止并保存 checkpoint；默认不限制。lr 调度不压缩，"
        "便于与同调度全量跑的早期 step 直接对比",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=None,
        help="LR cosine 衰减的总步数覆盖（短时限时训练时对齐到实际步数，"
        "让 lr 在结束时衰减到 0；默认用 epochs×每轮步数）",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="cosine 衰减的下限比例（相对峰值），默认 0.1；0 表示衰减到 0",
    )
    parser.add_argument(
        "--pack_sequences",
        action="store_true",
        default=False,
        help="预训练数据打包：文档用 eos 拼接成定长块，消除 padding 浪费"
        "（文档远短于 max_seq_len 时真实 token/步接近翻倍）",
    )
    parser.add_argument(
        "--doc_mask",
        action="store_true",
        default=False,
        help="打包序列加文档边界掩码：注意力不允许跨文档（与逐篇 PPL 评估"
        "口径对齐），并屏蔽跨文档边界位置的 loss。需配合 --pack_sequences",
    )
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="When resuming, do not load optimizer states and restart from step 0",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1337, help="全局随机种子")
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument(
        "--use_value_res",
        action="store_true",
        default=False,
        help="Value Residual Learning：后续层以可学习比例混入第一层 V",
    )
    parser.add_argument("--no_value_res", action="store_false", dest="use_value_res")
    parser.add_argument(
        "--use_attn_gate",
        action="store_true",
        default=False,
        help="注意力输出门：输入条件逐 head sigmoid 门（零初始化）",
    )
    parser.add_argument("--no_attn_gate", action="store_false", dest="use_attn_gate")
    parser.add_argument("--mtp_depth", type=int, default=1)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="muon",
        choices=["muon", "adamw"],
        help="muon = Muon+AdamW 混合（核心矩阵 Muon）；adamw = 全参数 AdamW",
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Muon Newton-Schulz 迭代步数（默认 5 对齐原版；减到 3 整步快 ~5%，"
        "但正交化精度下降、训练动力学改变）",
    )
    parser.add_argument(
        "--loop_k",
        type=int,
        default=1,
        help="深度循环次数：layer 栈重复执行 loop_k 次；>1 时启用 per-step FiLM",
    )
    parser.add_argument(
        "--dw_rank",
        type=int,
        default=0,
        help="ΔW-Loop 步条件低秩权重再生的秩；>0 且 loop_k>1 时启用"
        "（参数代价 ≈ r·(in+out)+k·r 每矩阵）",
    )
    parser.add_argument(
        "--ws_loop",
        type=int,
        default=0,
        help="W-Scale-Loop 步条件对角权重缩放；>0 且 loop_k>1 时启用"
        "（参数代价 ≈ k·(in+out) 每矩阵，s 全 1 初始化）",
    )
    parser.add_argument(
        "--hrm_H_cycles",
        type=int,
        default=0,
        help="HRM 模式高循环次数；>0 时启用双状态层次循环，"
        "num_hidden_layers 表示每个 stack 的真实层数",
    )
    parser.add_argument(
        "--hrm_L_cycles",
        type=int,
        default=3,
        help="HRM 每个高循环内的低循环次数",
    )
    parser.add_argument(
        "--hrm_bp_cycles",
        type=str,
        default=None,
        help="HRM 训练梯度路由：逗号分隔的每 H cycle 尾部回传 L cycle 数；默认全部回传",
    )
    parser.add_argument(
        "--hrm_emb_scale",
        type=float,
        default=1.0,
        help="HRM 输入 embedding 缩放；默认 1.0（MLX 初始化与 HF 不同）",
    )
    parser.add_argument(
        "--hrm_cycle_router",
        type=int,
        default=0,
        help="HRM×MoE CycleRouter：router 增加 per-cycle 专家偏置（零初始化），"
        "让专家按迭代特化；仅 HRM 模式生效",
    )
    parser.add_argument(
        "--hrm_cycle_film",
        type=int,
        default=0,
        help="HRM CycleFiLM：每次 stack 调用前做 per-cycle scale/shift（零初始化）；"
        "仅 HRM 模式生效",
    )
    parser.add_argument(
        "--ffn_type",
        type=str,
        default="swiglu",
        choices=["swiglu", "none", "hadamard"],
        help="FFN 类型：swiglu=标准 FFN；none=SAN 纯注意力（FFN 参数重配到深度）；"
        "hadamard=固定 Hadamard 变换 + 对角缩放（Needle 2）",
    )
    parser.add_argument(
        "--zero_centered_norm",
        action="store_true",
        default=False,
        help="SAN 组件：ZCN(z)=(1+γ)z/RMS(z)，γ 零初始化",
    )
    parser.add_argument(
        "--use_res_gate",
        action="store_true",
        default=False,
        help="SAN 组件：注意力残差标量门 y=x+σ(g)·attn_out，g 零初始化",
    )
    parser.add_argument(
        "--sandwich_norm",
        action="store_true",
        default=False,
        help="Post-attention sandwich norm：注意力输出先 norm 再加残差"
        "（SAN 论文消融唯一正收益变体，-0.009 nats）",
    )
    parser.add_argument(
        "--san_res_init",
        action="store_true",
        default=False,
        help="SAN 组件：o_proj 深度缩放初始化（方差 ∝ 1/(2L)）",
    )
    parser.add_argument(
        "--emb_scale",
        type=float,
        default=1.0,
        help="非 HRM 模式的 embedding 缩放（SAN 用 0.02*hidden_size 对齐 "
        "Needle 的 normal(0.02)*sqrt(d) 有效尺度）",
    )
    parser.add_argument(
        "--engram_layers",
        type=str,
        default="",
        help="Engram n-gram 记忆注入的层下标，逗号分隔（如 1,4）；空串关闭",
    )
    parser.add_argument(
        "--engram_orders",
        type=str,
        default="2,3",
        help="Engram 的 n-gram 阶数，逗号分隔",
    )
    parser.add_argument("--engram_slots", type=int, default=8192)
    parser.add_argument(
        "--engram_heads",
        type=int,
        default=0,
        help="Engram 每 order 的表头数；0=按 hidden_size 自动",
    )
    parser.add_argument("--engram_sub_dim", type=int, default=128)
    parser.add_argument(
        "--n_routed_experts",
        type=int,
        default=0,
        help="DeepSeekMoE 路由专家数；>0 时第 n_dense_layers 层起 FFN 换为 MoE",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=6,
        help="MoE 每 token 激活的路由专家数（V4 为 6）",
    )
    parser.add_argument(
        "--n_shared_experts",
        type=int,
        default=1,
        help="MoE 共享专家数（中间维 = moe_intermediate_size × n_shared_experts）",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=None,
        help="MoE 单个路由专家中间维；默认取 intermediate_size",
    )
    parser.add_argument(
        "--n_dense_layers",
        type=int,
        default=1,
        help="MoE 模型的前若干 dense FFN 层数（V3/V4 风格）",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        type=float,
        default=2.5,
        help="MoE 路由权重缩放因子（sigmoid 归一化后乘该系数）",
    )
    parser.add_argument(
        "--moe_bias_update_rate",
        type=float,
        default=0.001,
        help="无辅助损失负载均衡的偏置更新步长 u（每优化器步按负载统计更新）；<=0 关闭",
    )


def get_pretrain_parser():
    """获取预训练参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Pretraining")
    add_common_args(parser)

    # 预训练特定参数
    parser.set_defaults(
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        accumulation_steps=8,
        max_seq_len=2048,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Pretrain")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")

    return parser


def get_sft_parser():
    """获取SFT参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Full SFT")
    add_common_args(parser)

    # SFT特定参数
    parser.set_defaults(
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        accumulation_steps=1,
        max_seq_len=2048,
    )
    # 结构参数默认 None：优先从 pretrain checkpoint 的 sidecar config 继承，
    # 仅在用户显式传入时覆盖（sidecar 缺失时回退 VibyConfig 默认值）。
    parser.set_defaults(
        num_hidden_layers=None,
        num_attention_heads=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
    )
    parser.set_defaults(
        use_value_res=None,
        use_attn_gate=None,
        loop_k=None,
        dw_rank=None,
        ws_loop=None,
        hrm_H_cycles=None,
        hrm_L_cycles=None,
        hrm_bp_cycles=None,
        hrm_emb_scale=None,
        hrm_cycle_router=None,
        hrm_cycle_film=None,
        ffn_type=None,
        zero_centered_norm=None,
        use_res_gate=None,
        sandwich_norm=None,
        san_res_init=None,
        emb_scale=None,
        engram_layers=None,
        engram_orders=None,
        engram_heads=None,
        engram_slots=None,
        engram_sub_dim=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_shared_experts=None,
        moe_intermediate_size=None,
        n_dense_layers=None,
        routed_scaling_factor=None,
        moe_bias_update_rate=None,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Full-SFT")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl")
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="预训练检查点文件名（默认按 hidden_size 自动推导）",
    )

    # YaRN scaling parameters
    parser.add_argument(
        "--enable_yarn", action="store_true", help="Enable YaRN scaling"
    )
    parser.add_argument(
        "--yarn_scaling_factor", default=2.0, type=float, help="YaRN scaling factor"
    )
    parser.add_argument(
        "--original_max_seq_len",
        default=None,
        type=int,
        help="Original context length before scaling（默认取 pretrain checkpoint "
        "sidecar 中的实际上下文长度，无 sidecar 时为 1024）",
    )
    parser.add_argument(
        "--yarn_beta_fast", default=32.0, type=float, help="YaRN beta_fast parameter"
    )
    parser.add_argument(
        "--yarn_beta_slow", default=1.0, type=float, help="YaRN beta_slow parameter"
    )
    parser.add_argument(
        "--yarn_attention_factor",
        default=1.0,
        type=float,
        help="YaRN attention factor（mscale）",
    )

    return parser


def get_dpo_parser():
    """获取DPO参数解析器"""
    parser = argparse.ArgumentParser(description="Viby DPO Training")
    add_common_args(parser)

    # DPO特定参数
    parser.set_defaults(
        epochs=2,
        batch_size=4,
        learning_rate=1e-8,  # DPO学习率通常很小
        accumulation_steps=1,
        max_seq_len=1024,
    )
    # 结构参数默认 None：优先从 SFT checkpoint 的 sidecar config 继承
    # （含 rope_scaling/YaRN 配置），仅在用户显式传入时覆盖。
    parser.set_defaults(
        num_hidden_layers=None,
        num_attention_heads=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
    )
    parser.set_defaults(
        use_value_res=None,
        use_attn_gate=None,
        loop_k=None,
        dw_rank=None,
        ws_loop=None,
        hrm_H_cycles=None,
        hrm_L_cycles=None,
        hrm_bp_cycles=None,
        hrm_emb_scale=None,
        hrm_cycle_router=None,
        hrm_cycle_film=None,
        ffn_type=None,
        zero_centered_norm=None,
        use_res_gate=None,
        sandwich_norm=None,
        san_res_init=None,
        emb_scale=None,
        engram_layers=None,
        engram_orders=None,
        engram_heads=None,
        engram_slots=None,
        engram_sub_dim=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_shared_experts=None,
        moe_intermediate_size=None,
        n_dense_layers=None,
        routed_scaling_factor=None,
        moe_bias_update_rate=None,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-DPO")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
    parser.add_argument(
        "--dpo_beta", type=float, default=0.1, help="DPO beta parameter"
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="SFT 检查点文件名（默认按 hidden_size 自动推导）",
    )

    return parser


def build_sft_rope_scaling(args):
    """Build YaRN config for SFT and update args.max_seq_len when enabled."""
    if not (args.max_seq_len >= 2048 or args.enable_yarn):
        return None

    if not hasattr(args, "yarn_scaling_factor") or args.yarn_scaling_factor == 2.0:
        args.yarn_scaling_factor = args.max_seq_len / args.original_max_seq_len

    rope_scaling = {
        "type": "yarn",
        "factor": args.yarn_scaling_factor,
        "original_max_position_embeddings": args.original_max_seq_len,
        "beta_fast": getattr(args, "yarn_beta_fast", 32.0),
        "beta_slow": getattr(args, "yarn_beta_slow", 1.0),
        "attention_factor": getattr(args, "yarn_attention_factor", 1.0),
    }
    Logger(
        f"[YaRN] 启用上下文扩展: {args.original_max_seq_len} → {args.max_seq_len} (scaling factor: {args.yarn_scaling_factor})"
    )
    return rope_scaling


def setup_training_args(args, training_type="pretrain"):
    """设置训练参数"""
    import os

    # 基础参数校验
    if args.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if args.accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须大于 0")
    if args.save_interval <= 0:
        raise ValueError("save_interval 必须大于 0")
    if args.log_interval <= 0:
        raise ValueError("log_interval 必须大于 0")
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len 必须大于 0")
    if args.max_train_minutes is not None and args.max_train_minutes <= 0:
        raise ValueError("max_train_minutes 必须大于 0（或不传入以禁用时间限制）")
    if getattr(args, "max_steps", None) is not None and args.max_steps <= 0:
        raise ValueError("max_steps 必须大于 0（或不传入以禁用步数限制）")

    # checkpoint 保存点必须落在梯度累积窗口边界上：窗口中间保存的 checkpoint
    # 不含已累加但未更新的梯度，resume 时这部分梯度会永久丢失
    if args.save_interval % args.accumulation_steps != 0:
        adjusted = (
            (args.save_interval + args.accumulation_steps - 1)
            // args.accumulation_steps
        ) * args.accumulation_steps
        Logger(
            f"Warning: save_interval ({args.save_interval}) 不是 accumulation_steps "
            f"({args.accumulation_steps}) 的整数倍，已自动调整为 {adjusted}，"
            "避免 checkpoint 落在梯度累积窗口中间导致 resume 丢失梯度"
        )
        args.save_interval = adjusted

    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置 swanlab 运行名称：out_dir 基名（通常即实验轮次名）+ 关键配置
    run_tag = os.path.basename(os.path.normpath(args.out_dir))
    args.swanlab_run_name = (
        f"{run_tag}-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
    )

    # 设置随机种子（MLX 单设备）
    base_seed = getattr(args, "seed", 1337)
    import random

    import mlx.core as mx
    import numpy as np

    random.seed(base_seed)
    np.random.seed(base_seed)
    mx.random.seed(base_seed)

    # DataLoader / 分布式相关参数在 MLX 单设备下被忽略
    ignored = []
    if args.num_workers != 1:
        ignored.append(f"num_workers={args.num_workers}")
    if args.pin_memory:
        ignored.append("pin_memory")
    if args.prefetch_factor != 2:
        ignored.append(f"prefetch_factor={args.prefetch_factor}")
    if not args.persistent_workers:
        ignored.append("persistent_workers=False")
    if args.ddp or getattr(args, "local_rank", -1) not in (-1, 0):
        ignored.append("ddp/local_rank")
    if ignored:
        Logger(f"Warning: MLX 单设备训练忽略以下参数: {', '.join(ignored)}")
    args.pin_memory = False

    return args
