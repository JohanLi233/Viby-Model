"""
训练配置和参数解析模块
"""

import argparse
import os
import sys

from model.config import VibyConfig

from .flops import DEFAULT_PEAK_TFLOPS
from .utils import ARCH_ARG_ALIASES, ARCH_ARG_TO_FIELD, Logger

# 结构参数的 parser 默认值直接取自 VibyConfig（≈1B 配方），CLI 不复制配方；
# --preset 只在用户未显式传入某字段时把它换成预设值（见 apply_preset）。
_DEFAULT_CFG = VibyConfig()


def _int_list(text: str):
    """逗号/空格分隔的整数串 → tuple[int]（与 utils._as_int_tuple 同口径）。"""
    return tuple(int(x) for x in text.replace(",", " ").split())


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
    # ---- 模型结构参数（DeepSeek-V4.1 缩放版）----
    # 默认值全部取自 VibyConfig 的 ≈1B 配方；--preset 只补"未显式传入"的字段。
    # SFT/DPO 的 parser 会把这些默认抹成 None，让基座 checkpoint 的 sidecar 生效。
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "base"],
        default=None,
        help="结构预设：tiny 是小配置（单测/冒烟）；base 与不传等价，都是 "
        "VibyConfig 的 ≈1B 配方。命令行显式传入的结构参数优先于预设",
    )
    parser.add_argument("--hidden_size", type=int, default=_DEFAULT_CFG.dim)
    parser.add_argument("--num_hidden_layers", type=int, default=_DEFAULT_CFG.n_layers)
    parser.add_argument("--num_attention_heads", type=int, default=_DEFAULT_CFG.n_heads)
    parser.add_argument("--head_dim", type=int, default=_DEFAULT_CFG.head_dim)
    parser.add_argument("--rope_head_dim", type=int, default=_DEFAULT_CFG.rope_head_dim)
    parser.add_argument("--q_lora_rank", type=int, default=_DEFAULT_CFG.q_lora_rank)
    parser.add_argument("--o_groups", type=int, default=_DEFAULT_CFG.o_groups)
    parser.add_argument("--o_lora_rank", type=int, default=_DEFAULT_CFG.o_lora_rank)
    parser.add_argument("--vocab_size", type=int, default=_DEFAULT_CFG.vocab_size)
    parser.add_argument(
        "--window_size",
        type=int,
        default=_DEFAULT_CFG.window_size,
        help="滑窗分支窗口（V4.1-Flash 为 128）",
    )
    parser.add_argument(
        "--use_xsa",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CFG.use_xsa,
        help="Gated XSA（可学习 Exclusive Self-Attention，arXiv:2603.09078 / modded-nanogpt record #82）：逐 head 学 tanh(α) 扣掉注意力输出中与自身 V 平行的分量 z = y − tanh(α)·(yᵀv/‖v‖²)·v，α=0 初始化 ⇒ 起步恒等。作用在最深 --xsa_last_n 层（本模型是共享 K=V 的 MQA，v 对所有 head 相同，无需 GQA 的 V-扩展）。默认开",
    )
    parser.add_argument(
        "--xsa_last_n",
        type=int,
        default=0,
        help="启用 gated XSA 的最深 N 层；0=自动按配方取最深 ≈1/3 层（max(1, n_layers // 3)）。取 n_layers 则全主干层应用（DSpark 草稿层始终不挂）。仅 --use_xsa 时生效",
    )
    parser.add_argument(
        "--hc_mult",
        type=int,
        default=_DEFAULT_CFG.hc_mult,
        help="mHC 残差流条数（V4.1 扩展因子 4）",
    )
    parser.add_argument(
        "--hc_sinkhorn_iters", type=int, default=_DEFAULT_CFG.hc_sinkhorn_iters
    )
    parser.add_argument("--hc_eps", type=float, default=_DEFAULT_CFG.hc_eps)
    parser.add_argument(
        "--compress_ratios",
        type=_int_list,
        default=None,
        help="逐层压缩率，逗号分隔（如 0,0,2,2,1,1）；默认按 n_layers 推导"
        "（浅层纯滑窗、编码器 r=2、解码器 r=1）",
    )
    parser.add_argument(
        "--kv_source_layers",
        type=_int_list,
        default=None,
        help="压缩 KV 的产出层（逗号分隔、严格递增、不含第 0 层）；默认按 "
        "n_layers 推导",
    )
    parser.add_argument(
        "--index_source_layers",
        type=_int_list,
        default=None,
        help="Lightning Indexer 的产出层；默认按 n_layers 推导",
    )
    parser.add_argument(
        "--candidate_source_layer",
        type=int,
        default=None,
        help="分层稀疏索引的候选块产出层（更深的 indexer 只在候选池里打分）；"
        "默认 CED 边界层",
    )
    parser.add_argument(
        "--candidate_topk_blocks",
        type=int,
        default=_DEFAULT_CFG.candidate_topk_blocks,
        help="候选块个数（V4.1-Flash 2048，缩放版默认 64）",
    )
    parser.add_argument(
        "--candidate_block_size", type=int, default=_DEFAULT_CFG.candidate_block_size
    )
    parser.add_argument("--index_n_heads", type=int, default=_DEFAULT_CFG.index_n_heads)
    parser.add_argument(
        "--index_head_dim", type=int, default=_DEFAULT_CFG.index_head_dim
    )
    parser.add_argument("--index_topk", type=int, default=_DEFAULT_CFG.index_topk)
    parser.add_argument(
        "--psr",
        "--psr_enabled",
        dest="psr_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用隔离的 PSR 纠错分支；预训练默认启用，--no-psr 选择基线",
    )
    parser.add_argument(
        "--dpr",
        dest="dpr_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DPR-JEPA; requires --no-psr --mtp_depth 0",
    )
    for name in ("dpr_particles", "dpr_dim", "dpr_width", "dpr_horizon", "dpr_seed"):
        parser.add_argument("--" + name, type=int, default=getattr(_DEFAULT_CFG, name))
    for name in ("dpr_loss_weight", "dpr_warmup_fraction"):
        parser.add_argument(
            "--" + name, type=float, default=getattr(_DEFAULT_CFG, name)
        )
    parser.add_argument("--dpr_objective", choices=("kernel", "mse"), default="kernel")
    parser.add_argument(
        "--ced-recurrent",
        dest="ced_recurrent_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="残差提升式循环 CED 研究路径；配合 --no-psr --mtp_depth 0",
    )
    parser.add_argument(
        "--ced-recurrent-stride",
        "--ced_recurrent_stride",
        type=int,
        default=4,
        help="每个文档完成多少 token 后产生一个锚点",
    )
    parser.add_argument(
        "--ced-recurrent-rounds",
        "--ced_recurrent_rounds",
        type=int,
        default=3,
        help="共享中间 decoder 栈的循环次数",
    )
    for name in (
        "psr_slots",
        "psr_dim",
        "psr_blocks",
        "psr_topk",
        "psr_rounds",
        "psr_max_rounds",
        "psr_horizon",
        "psr_train_anchors",
    ):
        parser.add_argument("--" + name, type=int, default=getattr(_DEFAULT_CFG, name))
    parser.add_argument("--psr_learning_rate", type=float, default=1e-4)
    parser.add_argument("--psr_weight_decay", type=float, default=0.0)
    parser.add_argument("--psr_grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--psr_freeze_base", action="store_true", help="冻结基座，只训练隔离纠错头"
    )
    parser.add_argument(
        "--psr_training_mode",
        choices=["off", "state_only", "recurrent"],
        default="recurrent",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=_DEFAULT_CFG.rope_theta,
        help="纯滑窗层的 RoPE base",
    )
    parser.add_argument(
        "--compress_rope_theta",
        type=float,
        default=_DEFAULT_CFG.compress_rope_theta,
        help="压缩分支的 RoPE base（配 YaRN）",
    )
    parser.add_argument(
        "--original_seq_len", type=int, default=_DEFAULT_CFG.original_seq_len
    )
    parser.add_argument("--rope_factor", type=float, default=_DEFAULT_CFG.rope_factor)
    parser.add_argument("--beta_fast", type=int, default=_DEFAULT_CFG.beta_fast)
    parser.add_argument("--beta_slow", type=int, default=_DEFAULT_CFG.beta_slow)
    parser.add_argument(
        "--n_routed_experts",
        type=int,
        default=_DEFAULT_CFG.n_routed_experts,
        help="MoE 路由专家数（V4.1 主干全层 MoE，默认 ≈1B 配方 96）",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=_DEFAULT_CFG.n_activated_experts,
        help="每 token 激活的路由专家数（V4.1-Flash 为 6）",
    )
    parser.add_argument(
        "--n_shared_experts",
        type=int,
        default=_DEFAULT_CFG.n_shared_experts,
        help="共享专家个数（V4.1 固定 1）",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=_DEFAULT_CFG.moe_inter_dim,
        help="单个路由专家的中间维",
    )
    parser.add_argument(
        "--score_func",
        type=str,
        choices=["sqrtsoftplus", "softmax", "sigmoid"],
        default=_DEFAULT_CFG.score_func,
        help="路由亲和度（V4.1 用 sqrtsoftplus）",
    )
    parser.add_argument("--gate_temp", type=float, default=_DEFAULT_CFG.gate_temp)
    parser.add_argument(
        "--moe_balance_method",
        choices=["qb", "noaux_tc"],
        default=_DEFAULT_CFG.moe_balance_method,
        help="专家负载均衡：qb=分位数偏置（默认）；noaux_tc=固定 sign 步长",
    )
    parser.add_argument(
        "--qb_update_rate",
        type=float,
        default=_DEFAULT_CFG.qb_update_rate,
        help="每个优化器窗口向 QB 目标偏置移动的比例，0=冻结，1=全量更新",
    )
    parser.add_argument(
        "--qb_stats_rows",
        type=int,
        default=_DEFAULT_CFG.qb_stats_rows,
        help="每层每累积窗口的 QB token 样本预算（默认 8192）",
    )
    parser.add_argument(
        "--router_fp32",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CFG.router_fp32,
        help="保留路由权重和打分 GEMM 为 fp32（默认开）",
    )
    parser.add_argument(
        "--aux_balance_loss_weight",
        type=float,
        default=_DEFAULT_CFG.aux_balance_loss_weight,
        help="归一化的序列级均衡损失权重；0=仅使用偏置均衡",
    )
    parser.add_argument(
        "--route_scale",
        type=float,
        default=_DEFAULT_CFG.route_scale,
        help="路由权重缩放（论文 route scale，默认 1.5）",
    )
    parser.add_argument(
        "--routed_scaling_factor",
        dest="route_scale",
        type=float,
        default=None,
        help="--route_scale 的旧名别名（保留兼容；两者都传时后者生效）",
    )
    parser.add_argument(
        "--swiglu_limit",
        type=float,
        default=_DEFAULT_CFG.swiglu_limit,
        help="专家 clamped SwiGLU 的 clamp 上界（V4.1-Flash 为 10）",
    )
    parser.add_argument(
        "--bias_update_rate",
        type=float,
        default=_DEFAULT_CFG.bias_update_rate,
        help="noaux_tc 的 e_score_correction_bias 更新步长 γ："
        "仅 noaux_tc 使用：b -= γ·sign(load_frac − 1/E)，随后减均值",
    )
    parser.add_argument(
        "--engram_layer_ids",
        type=_int_list,
        default=None,
        help="挂 Engram 的主干层下标（逗号分隔）；默认第 2 层 + 约 35%% 深度处",
    )
    parser.add_argument(
        "--engram_max_ngram_size",
        type=int,
        default=_DEFAULT_CFG.engram_max_ngram_size,
        help="最大 n-gram 阶（至少 2）",
    )
    parser.add_argument(
        "--engram_vocab_size",
        type=int,
        default=_DEFAULT_CFG.engram_vocab_size,
        help="n-gram 素数桶的模数上界（压缩 id 空间大小，决定表行数）",
    )
    parser.add_argument(
        "--engram_n_heads", type=int, default=_DEFAULT_CFG.engram_n_heads
    )
    parser.add_argument(
        "--engram_head_dim", type=int, default=_DEFAULT_CFG.engram_head_dim
    )
    parser.add_argument(
        "--mtp_depth",
        type=int,
        default=_DEFAULT_CFG.n_mtp_layers,
        help="DSpark 草稿层数（n_mtp_layers，V4.1-Flash 为 3）",
    )
    parser.add_argument(
        "--dspark_block_size",
        type=int,
        default=_DEFAULT_CFG.dspark_block_size,
        help="DSpark 半自回归草稿块大小（V4.1-Flash 为 5：1 锚点 + 4 草稿槽）",
    )
    parser.add_argument(
        "--dspark_noise_token_id",
        type=int,
        default=_DEFAULT_CFG.dspark_noise_token_id,
        help="草稿槽的噪声 token（默认 pad_token_id）",
    )
    parser.add_argument(
        "--dspark_target_layer_ids",
        type=_int_list,
        default=None,
        help="DSpark 读取的主干目标层（逗号分隔，取这些层的注意力输入）；默认最深 3 层",
    )
    parser.add_argument(
        "--dspark_markov_rank", type=int, default=_DEFAULT_CFG.dspark_markov_rank
    )
    parser.add_argument(
        "--dspark_n_routed_experts",
        type=int,
        default=_DEFAULT_CFG.dspark_n_routed_experts,
        help="draft 层 MoE 的路由专家数（比主干窄）",
    )
    parser.add_argument(
        "--dspark_n_activated_experts",
        type=int,
        default=_DEFAULT_CFG.dspark_n_activated_experts,
    )
    parser.add_argument(
        "--mtp_loss_weight",
        type=float,
        default=_DEFAULT_CFG.mtp_loss_weight,
        help="DSpark 草稿损失权重（主干 loss + w·mtp_loss）",
    )
    parser.add_argument(
        "--z_loss_weight",
        type=float,
        default=_DEFAULT_CFG.z_loss_weight,
        help="logit z-loss 权重：loss += z_loss_weight * mean(lse²)（Marin），"
        "主 LM 与 DSpark 各 head 同权重；0 关闭",
    )
    parser.add_argument(
        "--norm_topk_prob",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CFG.norm_topk_prob,
        help="top-k 路由权重归一化（V4.1 开启）",
    )
    parser.add_argument(
        "--tie_word_embeddings",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_CFG.tie_word_embeddings,
        help="绑定输入/输出 embedding（True 时 lm_head 复用 embed，无额外参数）；"
        "默认 False：单独创建 nn.Linear，muonh 下走 AdamH 组",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="冻结主干，只训练 mtp_modules.*：对应论文 §2.4.3 的 DSpark 独立"
        "训练阶段（DSpark 目标本就不回传主干，这里连其余参数也不更新）",
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
        help="线性 warmup 微批步数；默认按本轮 LR horizon 的 1%%（Marin #8435）",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument("--pin_memory", action="store_true", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="吞吐/短探针：不写 checkpoint、sidecar、latest_checkpoint，"
        "也不开 swanlab。终端 loss/tokens/s 仍按 --log_interval 打印",
    )
    parser.add_argument(
        "--peak_tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help="硬件峰值 TFLOPS，用于 MFU。默认 13.5 对齐 M4 Max bf16 "
        "稠密 GEMM 实测峰值（MLX_PERF.md 12.9~14）。换机请改此值",
    )
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
        "梯度累积窗口边界停止并保存 checkpoint。未显式传 --lr_decay_steps / "
        "--token_budget 时，LR 日程和 compute 缩放的 token 预算都按该步数"
        "（Marin #8435：缩短 horizon 仍在终点落到 min_lr_ratio）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=None,
        help="LR 日程所用的总微批步数覆盖。比 --max_steps 优先；可长于本轮"
        "数据（复现长 run 前缀、峰值仍按该 horizon 衰减）。默认："
        "max_steps 与数据长度的较小值，否则 epochs×每轮",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.05,
        help="日程衰减下限比例（相对峰值），默认 0.05（Marin Hero；后面还有 "
        "context extension / 后训练，不收到 0）",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="wsd",
        choices=["linear", "wsd"],
        help="学习率日程。wsd（默认，对齐报告 §4.2.2：线性预热 → 平台 → 末尾"
        "余弦衰减到 min_lr_ratio）；linear（Marin Hero #8435）= warmup 后立刻线性"
        "收到 min_lr_ratio，无平台",
    )
    parser.add_argument(
        "--wsd_decay_frac",
        type=float,
        default=0.2,
        help="WSD 末尾衰减占总步数的比例，默认 0.2；仅 --lr_schedule wsd 生效",
    )
    parser.add_argument(
        "--wsd_decay_shape",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="WSD 末尾衰减形状：cosine（默认，报告 §4.2.2 用余弦）或 linear",
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
        "--no-doc_align",
        action="store_false",
        dest="doc_align",
        default=True,
        help="关闭文档边界对齐（默认开）：打包时每块首 token 对齐到文档开头，"
        "长文档按 max_doc_len 截断，丢弃跨块尾部位以换取边界对齐、减少跨文档"
        "无效 attention；仅 --pack_sequences 生效。",
    )
    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=None,
        help="打包时单篇文档最大 token 数（不含 eos），默认=max_seq_len；"
        "配合文档边界对齐，值越小跨块截断的尾部越短、边界越干净。",
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
        "--sinkhorn_steps",
        type=int,
        default=None,
        help="Sinkhorn 均衡更新的归一化步数 K（报告 §4.2.2 = 11，必须奇数）："
        "奇数步收尾在行归一化，U^(K) 才是单位行 ℓ₂ 范数、乘 √n 后满足式 (7) "
        "的行 RMS≈1。不传则读 env VIBY_SINKHORN_K，仍无则 11。"
        "（Sinkhorn 组 = Engram 检索表 / token 嵌入 / 预测头）",
    )
    parser.add_argument(
        "--lr_scale_auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute 缩放超参（Marin Hero / Hyperball 口径，仅 pretrain）："
        "adam_lr = min(0.05, 0.087571·tokens^-0.3461·hidden^-0.3448·sqrt(tpb))，"
        "muon_lr = min(0.05, 13/3×adam_lr)，"
        "beta2 = clip(0.999^(tpb/131072), 0.95, 0.9999)，"
        "eps = 9.676e-18·sqrt(tokens/tpb)（Adam/AdamH 组）。默认开启；"
        "--no-lr_scale_auto 回到旧常数（单一 lr 默认 0.01、betas (0.9,0.95)、eps 1e-8）。"
        "优先级：显式 --learning_rate > 公式 > 0.01 兜底。"
        "tokens 未传 --token_budget 时与 LR 日程 horizon 一致",
    )
    parser.add_argument(
        "--token_budget",
        type=float,
        default=None,
        help="峰值 LR 所用的 token 预算（lr_scale_auto）。显式传入则缩短 "
        "max_steps 时只改衰减斜率、峰值仍按该预算（Hero 中途砍上限）。"
        "不传则与 LR 日程 horizon 一致：max_steps 或 epochs×每轮 × tpb",
    )
    parser.add_argument(
        "--muonh",
        dest="muonh",
        action="store_true",
        default=True,
        help="MuonH/AdamH/Adam 体系（Marin 口径）：Muon 组更新加 Frobenius "
        "范数球投影（方向/范数解耦）；3D 堆叠专家逐专家 NS 进 MuonH（"
        "`--no_muonh` 或 VIBY_MUONH_EXPERTS=0 时专家回 AdamW）；lm_head（非 tied）"
        "走 AdamH（Adam 方向+范数球投影，lr=muon 基础 lr）。默认开启。"
        "注意：开关改变优化器分组，不能用于续跑旧 checkpoint（optimizer state "
        "分组对不上），只用于新 run。正交化固定每步全量重算：NS 降频复用/"
        "Temporal Q 缓存是 r082 回退的最大元凶（早期 −0.4~0.5 nat），机制已删除，"
        "勿重引入。",
    )
    parser.add_argument(
        "--no_muonh",
        dest="muonh",
        action="store_false",
        help="关闭 MuonH：专家不进逐专家 NS，回普通 Muon + AdamW 分组",
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
        # 报告 §2.1：骨干预训练阶段省略 MTP 模块；DSpark 在预训练之后用一个
        # 专门阶段单独训练（冻结主干），见 §2.4.3。要训练草稿层就显式传
        # --mtp_depth N（配合 --resume 基座 + --freeze_backbone 即官方口径）。
        mtp_depth=0,
        psr_enabled=True,
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Pretrain")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")

    return parser


def get_sft_parser():
    """获取SFT参数解析器"""
    parser = argparse.ArgumentParser(
        description="Viby TailSFT (default) / standard SFT"
    )
    add_common_args(parser)

    # SFT特定参数
    parser.add_argument(
        "--sft_algorithm",
        choices=["tail", "standard"],
        default="tail",
        help="默认 TailSFT；standard 保留普通 SFT 对照",
    )
    parser.add_argument(
        "--tail_sft_filter_fraction",
        type=float,
        default=0.5,
        help="每个前向微批过滤进步最大的序列比例 [0,1)，默认 0.5",
    )
    parser.add_argument(
        "--tail_sft_schedule",
        choices=["static", "ramp"],
        default="static",
        help="static=固定比例；ramp=从 0 线性升至目标比例",
    )
    parser.add_argument(
        "--tail_sft_cache",
        default=None,
        help="初始模型逐序列损失缓存；默认在 .cache 中按数据/基座指纹复用",
    )
    parser.add_argument(
        "--empty_think_ratio",
        type=float,
        default=0.0,
        help="空 <think> 块保留阈值：0.0=始终清掉空 think 占位块（默认）；"
        ">0 时按概率保留（0.2≈MiniMind 原版行为，八成概率删除）",
    )
    parser.set_defaults(
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        accumulation_steps=1,
        max_seq_len=2048,
    )
    # 结构参数默认 None：优先从基座 checkpoint 的 sidecar config 继承，
    # 仅在用户显式传入时覆盖（无 sidecar 时回退 VibyConfig 默认值 / --preset）。
    parser.set_defaults(**{key: None for key in ARCH_ARG_TO_FIELD})

    parser.add_argument("--swanlab_project", type=str, default="Viby-Full-SFT")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl")
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="预训练检查点文件名（默认按 hidden_size 自动推导）",
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
    # 仅在用户显式传入时覆盖（无 sidecar 时回退 VibyConfig 默认值 / --preset）。
    parser.set_defaults(**{key: None for key in ARCH_ARG_TO_FIELD})

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


def _explicit_cli_keys() -> set:
    """命令行上真正出现过的选项名（dest 口径）。

    argparse 分不清"默认值"与"用户显式传了默认值"，而 --preset 的语义是
    "只补未显式传入的字段"：直接扫 argv 是唯一无歧义的判据。
    """
    seen = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            name = tok[2:].split("=", 1)[0]
            if name:
                seen.add(name.replace("-", "_"))
    return seen


# 由 (n_layers, n_mtp_layers) 推导、preset 不该固定的结构参数（见 apply_preset）
_PRESET_DERIVED_ARGS = {
    "ced_recurrent_enabled",
    "ced_recurrent_stride",
    "ced_recurrent_rounds",
    "psr_enabled",
    "mtp_depth",
    "xsa_last_n",
    "compress_ratios",
    "kv_source_layers",
    "index_source_layers",
    "candidate_source_layer",
    "dspark_target_layer_ids",
}


def apply_preset(args):
    """--preset：把未显式传入的结构参数换成预设值（显式值优先）。

    只在 pretrain 调用：SFT/DPO 的结构参数默认 None 表示"继承基座 sidecar"，
    预设由 build_config_from_sidecar 在没有 sidecar 时并入。
    """
    preset = getattr(args, "preset", None)
    if not preset:
        return args
    cfg = VibyConfig(preset=preset)
    explicit = _explicit_cli_keys()
    # 旧名别名（--routed_scaling_factor → route_scale）同样算显式传入
    explicit |= {dest for alias, dest in ARCH_ARG_ALIASES.items() if alias in explicit}
    # 派生字段（压缩率/各源层/候选层/draft 目标层）与 mtp_depth 不由 preset 固定：
    # 它们由 VibyConfig 从 (n_layers, n_mtp_layers) 推导，而 mtp_depth 决定训练阶段
    # ——报告 §2.1 要求预训练不带 MTP，DSpark 在之后单独一阶段训练（§2.4.3）。
    # 显式传参仍然优先。
    for arg, field in ARCH_ARG_TO_FIELD.items():
        if arg in _PRESET_DERIVED_ARGS and arg not in explicit:
            continue
        if arg not in explicit:
            setattr(args, arg, getattr(cfg, field))
    return args


def setup_training_args(args, training_type="pretrain"):
    """设置训练参数"""
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
    if training_type == "sft":
        from .tail_sft import validate_args

        validate_args(args)

    # 结构预设：只覆盖未显式传入的结构参数（pretrain 从零建模型；
    # SFT/DPO 的结构参数默认 None，交给 sidecar 继承逻辑）
    if training_type == "pretrain":
        apply_preset(args)
        if args.dpr_enabled and (
            args.psr_enabled
            or args.ced_recurrent_enabled
            or args.mtp_depth
            or args.freeze_backbone
            or args.psr_freeze_base
        ):
            raise ValueError(
                "DPR requires --no-psr --no-ced-recurrent --mtp_depth 0 and unfrozen backbone"
            )
        if args.dpr_enabled and args.pack_sequences and not args.doc_mask:
            raise ValueError("DPR packed training requires --doc_mask")
        if args.ced_recurrent_enabled:
            if args.psr_enabled or args.psr_freeze_base:
                raise ValueError(
                    "Recurrent CED requires --no-psr and an unfrozen backbone"
                )
            if args.mtp_depth != 0:
                raise ValueError(
                    "Recurrent CED requires --mtp_depth 0 on both comparison sides"
                )
        if (
            args.psr_learning_rate <= 0
            or args.psr_weight_decay < 0
            or args.psr_grad_clip < 0
        ):
            raise ValueError("invalid isolated PSR optimizer settings")

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
    if getattr(args, "no_save", False):
        args.use_swanlab = False
        if getattr(args, "auto_resume", False):
            Logger("Warning: --no_save 已关闭落盘，忽略 --auto_resume")
            args.auto_resume = False
        Logger("--no_save：不写 checkpoint / sidecar / swanlab")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)

    # 设置 swanlab 运行名称：out_dir 基名（通常即实验轮次名）+ 关键配置
    # learning_rate 为 None 时（pretrain 哨兵，稍后由 lr_scale_auto 解析）标 auto
    run_tag = os.path.basename(os.path.normpath(args.out_dir))
    lr_tag = args.learning_rate if args.learning_rate is not None else "auto"
    args.swanlab_run_name = (
        f"{run_tag}-{training_type}-E{args.epochs}-BS{args.batch_size}-LR{lr_tag}"
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
