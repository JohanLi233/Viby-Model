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
    # default None 作为"用户未显式传入"的哨兵：pretrain 下由
    # resolve_compute_scaled_hparams 决定最终 lr（手动 > 公式 > 0.01 兜底），
    # SFT/DPO 由各自 parser 的 set_defaults 给出固定默认值。
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="基础学习率。pretrain 且 --lr_scale_auto 开启时，显式传入会覆盖"
        "公式推出的 adam_lr（muon_lr 仍按 13/3× 派生）；不传则全自动",
    )
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
    parser.add_argument(
        "--tie_word_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="绑定输入/输出 embedding（True 时 lm_head 复用 embed_tokens，"
        "无额外参数）；默认 False：单独创建 nn.Linear，muonh 下走 AdamH 组。"
        "解绑增加约 vocab_size × hidden_size 参数，但给 unembedding 独立梯度",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=True,
        help="使用 mx.compile 编译 loss 函数（默认开启）",
    )
    parser.add_argument("--no_compile", action="store_false", dest="compile_model")
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--no_swanlab", dest="use_swanlab", action="store_false")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="梯度范数裁剪阈值；0=不裁剪（默认，对齐 Hyperball 配方）",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=None,
        help="线性 warmup 步数；默认按总步数的 1%（Marin / K3）",
    )
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
        help="LR 线性衰减的总步数覆盖（短时限时训练时对齐到实际步数，"
        "让 lr 在结束时落到 min_lr_ratio；默认用 epochs×每轮步数）",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.05,
        help="线性衰减的下限比例（相对峰值），默认 0.05（Marin）；0 表示衰减到 0",
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
        "--use_attn_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="注意力输出门：2·σ(W_g x)，零初始化门=1（Marin，默认开）",
    )
    parser.add_argument("--no_attn_gate", action="store_false", dest="use_attn_gate")
    parser.add_argument("--mtp_depth", type=int, default=1)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument(
        "--z_loss_weight",
        type=float,
        default=1e-4,
        help="logit z-loss 权重：loss += z_loss_weight * mean(lse²)（Marin），"
        "主 LM 与 MTP 各 head 同权重；0 关闭",
    )
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
        help="Muon Newton-Schulz 迭代步数（默认 5 对齐原版；减到 3 整步快 ~5%%，"
        "但正交化精度下降、训练动力学改变）",
    )
    parser.add_argument(
        "--n_routed_experts",
        type=int,
        default=0,
        help="DeepSeekMoE 路由专家数（必须 >0，所有层均为 MoE）",
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
        help="独立共享专家个数；每个中间维 = moe_intermediate_size，输出相加",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=None,
        help="MoE 单个路由专家中间维；默认取 intermediate_size",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        type=float,
        default=2.5,
        help="MoE 路由权重缩放因子（sigmoid 归一化后乘该系数）",
    )
    parser.add_argument(
        "--moe_router_logit_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="路由 logits 逐 token 标准化（默认开启，防 sigmoid 饱和）",
    )
    parser.add_argument(
        "--moe_router_logit_temp",
        type=float,
        default=1.0,
        help="logits 标准化后的温度/标准差",
    )
    parser.add_argument(
        "--moe_diversity_loss_weight",
        type=float,
        default=0.0,
        help="router 输入 token 多样性正则权重；0 关闭",
    )
    parser.add_argument(
        "--moe_latent_dim",
        type=int,
        default=None,
        help="Latent MoE：路由专家在压缩后的 latent 空间计算——共享 "
        "lat_down(hidden→d)/lat_up(d→hidden) 投影包住路由专家，lat_down 后"
        "接 learnable latent_norm；router 仍在全维打分，共享专家仍走全宽。"
        "默认 None → hidden_size//2（开启）；显式 0 关闭（消融）",
    )
    parser.add_argument(
        "--lr_scale_auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute 缩放超参（Hyperball 口径，仅 pretrain 生效）：adam_lr = "
        "0.087571·tokens^-0.3461·hidden^-0.3448·sqrt(tpb)，muon_lr = 13/3×adam_lr，"
        "beta2 = clip(0.999^(tpb/131072), 0.95, 0.9999)，"
        "eps = 9.676e-18·sqrt(tokens/tpb)（Adam/AdamH 组）。默认开启；"
        "--no-lr_scale_auto 回到旧常数（单一 lr 默认 0.01、betas (0.9,0.95)、eps 1e-8）。"
        "优先级：显式 --learning_rate > 公式 > 0.01 兜底",
    )
    parser.add_argument(
        "--token_budget",
        type=float,
        default=None,
        help="总训练 token 预算（供 lr_scale_auto 公式）；不传则按"
        "epochs × 每轮步数 × batch_size × max_seq_len 推导",
    )
    parser.add_argument(
        "--muonh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MuonH/AdamH/Adam 体系（Marin 口径）：Muon 组更新加 Frobenius "
        "范数球投影（方向/范数解耦）；3D 堆叠专家逐专家 NS 进 MuonH（默认 "
        "AdamW）；lm_head（非 tied）走 AdamH（Adam 方向+范数球投影，lr=muon "
        "基础 lr）。默认开启，--no-muonh 回退旧分组。注意：开关改变优化器分组，"
        "不能用于续跑旧 checkpoint（optimizer state 分组对不上），只用于新 run。"
        "VIBY_MUONH_EXPERTS=0 只开范数球投影、专家保持 AdamW（消融/省逐专家 "
        "NS 开销）。Temporal 默认开：VIBY_MUONH_CACHE_Q=1 + EVERY=8，命中 Q@U；"
        "VIBY_MUONH_CACHE_Q_RES 残差超限则本步重跑 NS5（0 关闭）",
    )


def get_pretrain_parser():
    """获取预训练参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Pretraining")
    add_common_args(parser)

    # 预训练特定参数
    # learning_rate 保持 None 哨兵：由 resolve_compute_scaled_hparams 在
    # train_pretrain.py 中按 手动 > 公式 > 0.01 兜底 的优先级解析
    parser.set_defaults(
        epochs=1,
        batch_size=32,
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
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
    )
    parser.set_defaults(
        use_attn_gate=None,
        tie_word_embeddings=None,
        moe_router_logit_norm=None,
        moe_router_logit_temp=None,
        moe_diversity_loss_weight=None,
        z_loss_weight=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_shared_experts=None,
        moe_intermediate_size=None,
        routed_scaling_factor=None,
        moe_latent_dim=None,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Full-SFT")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl")
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="预训练检查点文件名（默认按 hidden_size 自动推导）",
    )

    return parser


def get_draft_parser():
    """获取 EAGLE-3 草稿微调参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Draft (EAGLE-3) Finetune")
    add_common_args(parser)

    # 草稿微调特定参数
    parser.set_defaults(
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        accumulation_steps=1,
        max_seq_len=2048,
    )
    # 结构参数默认 None：优先从基座 checkpoint 的 sidecar config 继承，
    # 仅在用户显式传入时覆盖（sidecar 缺失时回退 VibyConfig 默认值）。
    parser.set_defaults(
        num_hidden_layers=None,
        num_attention_heads=None,
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
    )
    parser.set_defaults(
        use_attn_gate=None,
        tie_word_embeddings=None,
        moe_router_logit_norm=None,
        moe_router_logit_temp=None,
        moe_diversity_loss_weight=None,
        z_loss_weight=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_shared_experts=None,
        moe_intermediate_size=None,
        routed_scaling_factor=None,
        moe_latent_dim=None,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Draft")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="基座检查点文件名（默认按 hidden_size 自动推导）",
    )
    parser.add_argument(
        "--draft_ttt_steps",
        type=int,
        default=4,
        help="EAGLE-3 training-time test (TTT) rollout 步数：step 1 消费目标"
        "多层特征 + 真实下一 token 嵌入；step≥2 消费草稿自身上一步 hidden + "
        "自预测 token 嵌入；各步 CE 平均",
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
    # 结构参数默认 None：优先从 SFT checkpoint 的 sidecar config 继承，
    # 仅在用户显式传入时覆盖。
    parser.set_defaults(
        num_hidden_layers=None,
        num_attention_heads=None,
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
    )
    parser.set_defaults(
        use_attn_gate=None,
        tie_word_embeddings=None,
        moe_router_logit_norm=None,
        moe_router_logit_temp=None,
        moe_diversity_loss_weight=None,
        z_loss_weight=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        n_shared_experts=None,
        moe_intermediate_size=None,
        routed_scaling_factor=None,
        moe_latent_dim=None,
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
    # learning_rate 为 None 时（pretrain 哨兵，稍后由 lr_scale_auto 解析）标 auto
    run_tag = os.path.basename(os.path.normpath(args.out_dir))
    lr_tag = args.learning_rate if args.learning_rate is not None else "auto"
    args.swanlab_run_name = f"{run_tag}-E{args.epochs}-BS{args.batch_size}-LR{lr_tag}"

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
