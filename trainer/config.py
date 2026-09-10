"""
训练配置和参数解析模块
"""

import argparse
from model.flops import DEFAULT_PEAK_TFLOPS
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
        "--use_linear_attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Kimi Linear 注意力（KDA local + NoPE GQA global + ShortConv）。"
        "默认关：全层 MLA full softmax，无线性注意力、无短卷积",
    )
    parser.add_argument(
        "--kv_lora_rank",
        type=int,
        default=192,
        help="MLA 的 KV 低秩潜在维度（仅 --no-use_linear_attn，默认）",
    )
    parser.add_argument(
        "--qk_rope_head_dim",
        type=int,
        default=32,
        help="MLA 解耦 RoPE 键的维度（跨 head 共享）",
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
        default="linear",
        choices=["linear", "wsd"],
        help="学习率日程。linear（默认，Marin Hero #8435）= warmup 1%% 后立刻"
        "线性收到 min_lr_ratio，无平台；wsd=warmup + 平台 + 末尾衰减。"
        "数据 80%% 处分相不是 LR 平台，不要用 wsd 去模拟",
    )
    parser.add_argument(
        "--wsd_decay_frac",
        type=float,
        default=0.2,
        help="WSD 末尾线性衰减占总步数的比例，默认 0.2；仅 --lr_schedule wsd 生效",
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
        "--use_attn_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="注意力输出门：2·σ(W_g x)，零初始化门=1（Marin，默认开）",
    )
    parser.add_argument("--no_attn_gate", action="store_false", dest="use_attn_gate")
    parser.add_argument(
        "--use_xsa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gated XSA（可学习 Exclusive Self-Attention，arXiv:2603.09078 / "
        "modded-nanogpt record #82）：逐 head 学 tanh(α) 扣掉注意力输出中 "
        "与自身 V 平行的分量 z = y − tanh(α)·(yᵀv/‖v‖²)·v，α=0 初始化 ⇒ "
        "恒等。作用在最深 --xsa_last_n 层（MLA 头 1:1，无需 GQA 的 V-扩展）。"
        "默认开",
    )
    parser.add_argument(
        "--xsa_last_n",
        type=int,
        default=0,
        help="启用 gated XSA 的最深 N 层；0=自动按配方取最深 ≈1/3 层"
        "（`max(1, num_hidden_layers // 3)`）。取 num_hidden_layers 则全层应用。"
        "仅 --use_xsa 时生效",
    )
    parser.add_argument(
        "--hidden_act",
        choices=["silu", "situ"],
        default="silu",
        help="FFN 激活：silu（默认，SwiGLU/silu(g)·u）或 situ（SiTU-GLU "
        "β1=4, β2=25 的 tanh 软帽版，走 SiTU 融合核）",
    )
    parser.add_argument(
        "--attn_res_window",
        type=int,
        default=4,
        help="AttnRes 只混合最近 W 个残差（默认 4）。0=论文全历史混合。"
        "窗口把每层重读全部历史 v 的 O(L²) 流量收到 O(L·W)，是 MFU 主杠杆",
    )
    parser.add_argument(
        "--attn_res_register",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="寄存器残差：hidden 用加法写回（h ← h+F），AttnRes 只混合 "
        "[h]+近 W 个写入作为下一子层的读。默认关（论文替换式 AttnRes）。"
        "与 r086 对照时打开此开关即可，其余超参保持不变",
    )
    parser.add_argument(
        "--attn_res_read_h",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="寄存器读侧用 h 本身（prenorm(h)），不再 softmax 混合 [h]+写入。"
        "需要 --attn_res_register。默认关。旧 sidecar 缺键保持混合读",
    )
    parser.add_argument(
        "--ihc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="identity Hyper-Connections：M 条残差流，读 Σ h_pre R、写 R+h_post Δ，"
        "H_res=I（无 Sinkhorn）。开时接管残差，AttnRes merge/read 不走。"
        "默认关。旧 sidecar 缺键保持关",
    )
    parser.add_argument(
        "--ihc_streams",
        type=int,
        default=4,
        help="iHC 残差流数 M（Hy4 hc_mult=4）。仅 --ihc 时生效",
    )
    parser.add_argument(
        "--ihc_typed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="分型 iHC：流 0 恒等、n-gram 只进 --ihc_ngram_stream 且在进栈前注入，"
        "collapse 默认 identity。需要 --ihc。默认关",
    )
    parser.add_argument(
        "--ihc_collapse",
        choices=["mean", "identity"],
        default=None,
        help="iHC 出口：mean=流上均值（默认），identity=只读流 0。"
        "--ihc_typed 且未显式指定时为 identity",
    )
    parser.add_argument(
        "--ihc_ngram_stream",
        type=int,
        default=1,
        help="分型 iHC 下 n-gram 写入的流下标（默认 1）。identity 塌缩时不能为 0",
    )
    parser.add_argument(
        "--loop_span",
        type=int,
        default=0,
        help="SMELT 中间层 loop 跨度（arXiv:2609.01343）：居中内部 span 层"
        "连续执行 --loop_count 次（权重共享，参数量不变，effective depth ="
        " L + (r−1)·span）。0（默认）=关闭；需 >0 且 loop_count>1 才生效。"
        "loop 跨度 KV 翻倍，KV 预算对齐可调低 --kv_lora_rank（如 256→192）",
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=2,
        help="SMELT loop 次数 r（默认 2，SMELT/Loopie 消融均表明 r=2 最优）。"
        "仅 --loop_span > 0 时生效",
    )
    parser.add_argument(
        "--loop_res_scale",
        choices=["rsqrt", "r", "none"],
        default="rsqrt",
        help="loop 跨度内 sublayer 残差写入缩放（防 weight-tied 更新吹大"
        "残差流，SMELT Eq.8/9）：rsqrt=r**-0.5（默认）/ r=1/r / none=不缩放（消融）",
    )
    parser.add_argument(
        "--loop_grad_mode",
        choices=["full", "jfb"],
        default="full",
        help="loop 跨度反向模式：full（默认，全展开反传）/ jfb（单步梯度，"
        "0th-order IFT，HRM/TRM 口径）：前 r−1 次 visit 断梯度，只反传末次"
        "visit + x_entry 恒等通路。前向数值与 full 逐位一致；反向算力与"
        "激活内存降到约 1 次 visit（r=2 时 span 反向减半）",
    )
    parser.add_argument(
        "--loop_extrap",
        type=float,
        default=0.0,
        help="Richardson 外推 λ：末次 visit 后 h += λ(h − h_prev_visit)"
        "（h_i=第 i 次 visit 结束的 hidden，r>2 取最后两次）。0（默认）=关；"
        "λ=1 是经典 Richardson（误差随 visit 减半）。固定标量不加参数，"
        "可对旧 checkpoint 直接做推理探针。需要 --loop_span>0 且 loop_count>=2",
    )
    parser.add_argument(
        "--loop_anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="loop 锚点读出（默认开，仅 replace 模式 AttnRes 生效）：span "
        "区段的 merge 常驻 [span 入口 hidden] + [各 visit 出口摘要] + 全部 "
        "pre-span 写入，窗口只裁 span 内写入——修复滑窗把输入侧/跨 visit "
        "历史挤出后 span 闭环递归、主 loss 卡平台的问题。不加参数、不改残差"
        "语义。--no_loop_anchor 做消融",
    )
    parser.add_argument("--mtp_depth", type=int, default=1)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument(
        "--mtp_steps",
        type=int,
        default=2,
        help="Qwen3.8-Next MTP 预训练展开步数：同一层 teacher-forced "
        "预测 t+2, t+3, …，各步 CE 取平均。1=只预测下一额外 token",
    )
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
        help="DeepSeekMoE 路由专家数（必须 >0，所有层均为 MoE；1080M 配方 256）",
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
        "--moe_write_spread",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LatentMoE 专家写出基扩展：共享 lat_up 之外按专家对角缩放 "
        "tile(h_k) 写入残差补空间。s 零初始化 ⇒ 起步与关掉逐位一致。"
        "需要 moe_latent_dim > 0。默认关",
    )
    parser.add_argument(
        "--moe_route_scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LatentMoE 路由尺度：latent_out_norm 之后乘未归一化 top-k "
        "sigmoid 和。关掉时 RMSNorm 把 routed_scaling_factor 乘成 1。"
        "需要 moe_latent_dim > 0。默认关",
    )
    parser.add_argument(
        "--first_k_dense_replace",
        type=int,
        default=0,
        help="前 K 层 FFN 用 dense SwiGLU，其后保持 MoE。0=全层 MoE（默认）。"
        "1=第 0 层 dense stem。旧 sidecar 缺键保持 0",
    )
    parser.add_argument(
        "--dense_intermediate_size",
        type=int,
        default=None,
        help="浅层 dense 中间维。未传且 --first_k_dense_replace>0 时取 "
        "moe_intermediate_size（小 stem，与单个专家同宽）",
    )
    parser.add_argument(
        "--kda_v_head_ratio",
        type=int,
        default=2,
        help="KDA V 头 / QK 头比（Qwen GDN 口径）。默认 2：8 个 QK 头 → "
        "16 个 V 头；1=旧正方形。只用于新训练",
    )
    parser.add_argument(
        "--ngram_table_size",
        type=int,
        default=65536,
        help="Engram 哈希检索表总桶数（0 关闭）。全部 (阶×头) 子表共用此"
        "预算，各自素数取整后实际略小；每头维度 = d_mem/(阶数×头数)。"
        "表走独立 Adam 组（5x lr, wd=0）",
    )
    parser.add_argument(
        "--ngram_heads",
        type=int,
        default=8,
        help="Engram 每个 n-gram 阶的哈希头数（各头不同滚动乘子，碰撞去"
        "相关，检索向量拼接）。默认 8，对齐 Engram (arXiv 2601.07372)",
    )
    parser.add_argument(
        "--ngram_layer",
        type=int,
        default=2,
        help="n-gram 注入层（1-indexed）。默认 2，对齐 Qwen3.8-Flash-Next",
    )
    parser.add_argument(
        "--ngram_logit_skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="n-gram 表向量（不过 hidden 门）经 lm_head 加到 logits，"
        "标量尺度零初始化。默认关。与 r086 对照时打开此开关即可",
    )
    parser.add_argument(
        "--ngram_conf_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="n-gram 置信门控：用表向量 unembed 的 max-softmax 缩放每层 "
        "attn/mlp 写入（g=1−s·stopgrad(max p)，s 零初始化）。默认关。"
        "A0 只缩放写入、不跳过 FLOP。需要 --ngram_table_size > 0",
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
    )

    parser.add_argument("--swanlab_project", type=str, default="Viby-Pretrain")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")

    return parser


def get_sft_parser():
    """获取SFT参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Full SFT")
    add_common_args(parser)

    # SFT特定参数
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
    # 结构参数默认 None：优先从 pretrain checkpoint 的 sidecar config 继承，
    # 仅在用户显式传入时覆盖（sidecar 缺失时回退 VibyConfig 默认值）。
    parser.set_defaults(
        num_hidden_layers=None,
        num_attention_heads=None,
        vocab_size=None,
        mtp_depth=None,
        mtp_loss_weight=None,
        mtp_steps=None,
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
        moe_write_spread=None,
        moe_route_scale=None,
        first_k_dense_replace=None,
        dense_intermediate_size=None,
        kda_v_head_ratio=None,
        ngram_table_size=None,
        ngram_layer=None,
        ngram_heads=None,
        ngram_logit_skip=None,
        ngram_conf_gate=None,
        use_linear_attn=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        attn_res_register=None,
        attn_res_read_h=None,
        ihc=None,
        ihc_streams=None,
        ihc_typed=None,
        ihc_collapse=None,
        ihc_ngram_stream=None,
        loop_span=None,
        loop_count=None,
        loop_res_scale=None,
        loop_grad_mode=None,
        loop_extrap=None,
        loop_anchor=None,
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
        mtp_steps=None,
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
        moe_write_spread=None,
        moe_route_scale=None,
        first_k_dense_replace=None,
        dense_intermediate_size=None,
        kda_v_head_ratio=None,
        ngram_table_size=None,
        ngram_layer=None,
        ngram_heads=None,
        ngram_logit_skip=None,
        ngram_conf_gate=None,
        use_linear_attn=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        attn_res_register=None,
        attn_res_read_h=None,
        ihc=None,
        ihc_streams=None,
        ihc_typed=None,
        ihc_collapse=None,
        ihc_ngram_stream=None,
        loop_span=None,
        loop_count=None,
        loop_res_scale=None,
        loop_grad_mode=None,
        loop_extrap=None,
        loop_anchor=None,
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
        mtp_steps=None,
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
        moe_write_spread=None,
        moe_route_scale=None,
        first_k_dense_replace=None,
        dense_intermediate_size=None,
        kda_v_head_ratio=None,
        ngram_table_size=None,
        ngram_layer=None,
        ngram_heads=None,
        ngram_logit_skip=None,
        ngram_conf_gate=None,
        use_linear_attn=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        attn_res_register=None,
        attn_res_read_h=None,
        ihc=None,
        ihc_streams=None,
        ihc_typed=None,
        ihc_collapse=None,
        ihc_ngram_stream=None,
        loop_span=None,
        loop_count=None,
        loop_res_scale=None,
        loop_grad_mode=None,
        loop_extrap=None,
        loop_anchor=None,
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
