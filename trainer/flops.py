"""训练 FLOPs / MFU 口径（DeepSeek-V4.1 缩放版）。

MFU = (tokens/s × 每 token 训练 FLOPs) / 硬件峰值。

每 token 训练 FLOPs = 6 × 激活 GEMM 参数 + 注意力二次项：

- **6N**：fwd 2N + bwd 4N。N 只数每 token 真正参与 GEMM 的权重：
  路由专家按 top-k/E 折算；token embedding 与 Engram n-gram 检索表是纯
  寻址（稀疏 gather），不计；DSpark 在每个 anchor 展开 block 个草稿槽，
  草稿层与输出头按实际槽数折算。
- **注意力**：QKᵀ + AV 没有对应权重，按每层实际参与打分的长度 L 计
  6 · n_heads · (head_dim + head_dim) · L（fwd+bwd；K=V 共享 MQA，
  d_qk = d_v = head_dim）。默认按 sparse top-k 的名义长度估算。
  真实长度依赖文档掩码、候选池、阈值并列；验收需传入实测逐层平均长度。
  这是 6N 近似口径，不是精确算子 FLOPs；不计 Indexer 打分、池化、逐元素
  算子与重算，也不将稠密 fallback 的无效 mask 位置充作 useful FLOPs。
- 不含优化器（Muon Newton-Schulz）FLOPs；墙钟含优化器，因此 MFU 会低于
  纯 GEMM 利用率。这是训练 MFU 的标准口径。
- PSR 启用时，单独按固定预算文本预训练的一次前缀思考摊销到 token，
  包含全量索引/读取教师、循环与桥接；不能把循环权重当成每 token 一次 GEMM。

默认峰值 13.5 TFLOPS：M4 Max bf16 稠密 GEMM 实测中位
（research/MLX_PERF.md §0：12.9~14）。换机用 --peak_tflops。
"""

from mlx.utils import tree_flatten

# M4 Max bf16 稠密 GEMM 实测峰值中位（12.9~14 TFLOPS）
DEFAULT_PEAK_TFLOPS = 13.5


def _is_lookup_table(path: str, tied: bool) -> bool:
    """纯寻址表：token embedding 与 Engram n-gram 检索表。

    embedding 绑定 lm_head 时它同时是 readout，按「每 token 一次 GEMM」计数
    （在 _is_unembedding 里处理），此处不再跳过。
    """
    if "engram_layers" in path and path.endswith(".embed.weight"):
        return True
    if ".markov_head.embed.weight" in path:
        return True
    if path.endswith("model.embed.weight") or path == "embed.weight":
        return not tied
    return False


def _is_unembedding(path: str, tied: bool) -> bool:
    if "lm_head" in path:
        return True
    return tied and (path.endswith("model.embed.weight") or path == "embed.weight")


def _active_size(path: str, value, cfg) -> int:
    """堆叠专家 (E, out, in) 按 top-k/E 折算每 token 真正参与的行。"""
    size = int(value.size)
    if ".experts." in path and getattr(value, "ndim", 0) >= 3 and value.shape[0] > 0:
        e = int(value.shape[0])
        # draft 层（DSpark）用的是更窄的 MoE：E/top-k 都与主干不同
        if path.startswith("mtp_modules"):
            k = int(getattr(cfg, "dspark_n_activated_experts", 0) or 0)
        else:
            k = int(getattr(cfg, "n_activated_experts", 0) or 0)
        size = size // e * min(max(k, 0), e)
    return size


def gemm_active_params(model) -> int:
    """6N 的可训练权重近似（不含 frozen buffers，DSpark 按槽数折算）。

    面向全参数训练；冻结主干/LoRA 的前向成本需要另行逐算子审计。
    """
    cfg = model.config
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    mtp_on = int(getattr(cfg, "n_mtp_layers", 0) or 0) > 0
    n = 0
    block = int(getattr(cfg, "dspark_block_size", 1)) if mtp_on else 0
    for path, value in tree_flatten(model.trainable_parameters()):
        # PSR runs once per prefix, with R shared transitions. Counting its
        # matrices once per token overstates work and omits repeated work.
        if path.startswith(("psr.",)):
            continue
        # 1-D（norm gain / mHC scale·base / attn_sink / router bias）不是 GEMM 权重
        if getattr(value, "ndim", 0) < 2:
            continue
        if _is_lookup_table(path, tied):
            continue
        size = _active_size(path, value, cfg)
        if path.startswith("mtp_modules") and ".main_proj." not in path:
            size *= block
        if _is_unembedding(path, tied) and mtp_on:
            size *= 1 + block
        n += size
    return n


def attn_fwdbwd_flops_per_token(config, seq_len: int, attention_lengths=None) -> int:
    """CSA2 注意力的 QKᵀ + AV fwd+bwd FLOPs / token（不含投影 GEMM）。"""
    t = int(seq_len)
    if t <= 0:
        return 0
    heads = int(config.n_heads)
    head_dim = int(config.head_dim)
    window = int(config.window_size)
    ratios = tuple(config.compress_ratios)
    layers = int(config.n_layers) + int(config.n_mtp_layers)
    if attention_lengths is not None and len(attention_lengths) != layers:
        raise ValueError("attention_lengths must contain one mean per backbone/draft layer")
    total = 0.0
    for i in range(layers):
        draft = i >= int(config.n_layers)
        multiplier = int(config.dspark_block_size) if draft else 1
        length_t = multiplier if draft else t
        r = ratios[i] if i < len(ratios) else 0
        w = min(length_t, window)
        # Average causal window length, before document/padding masks.
        length = w - w * (w - 1) / (2 * length_t)
        if attention_lengths is not None:
            length = float(attention_lengths[i])
            if length < 0:
                raise ValueError("attention lengths must be nonnegative")
        elif r > 0:
            length += min(int(config.index_topk), (length_t + r - 1) // r)
        total += multiplier * 6 * heads * (head_dim + head_dim) * length
    return int(total)


def training_flops_per_token(model, seq_len: int, attention_lengths=None) -> int:
    """训练一步（fwd+bwd）每个 token 的近似 FLOPs，不含优化器。"""
    return 6 * gemm_active_params(model) + attn_fwdbwd_flops_per_token(
        model.config, seq_len, attention_lengths
    ) + psr_pretrain_flops_per_token(model.config, seq_len)


def psr_pretrain_flops_per_token(cfg, seq_len):
    """Nominal dense protected-side work; no indexer/teacher/value objectives.

    Includes forward-only detached baseline vocabulary logits. Sorting, gather,
    norm, scalar ops and optimizer are not GEMM FLOPs; profile them separately.
    """
    t = int(seq_len)
    if t <= 0 or not getattr(cfg, "psr_enabled", False):
        return 0
    a,m,s,r,d = min(cfg.psr_train_anchors,t),cfg.psr_slots,cfg.psr_dim,cfg.psr_rounds,cfg.dim
    hd = cfg.head_dim
    total = 6*a*d*s + 6*a*r*m*(2*s*hd + cfg.psr_blocks*10*s*s)
    total += 12*a*r*m*t*hd + cfg.psr_blocks*12*a*r*m*m*s
    total += 6*(t*d*s + 2*a*m*s*s + t*s*cfg.vocab_size) + 12*t*m*s
    total += 2*t*d*cfg.vocab_size
    return int(total/t)


def model_flops_utilization(
    tokens_per_sec: float,
    flops_per_token: float,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS,
) -> float:
    """返回 0~∞ 的利用率（1.0 = 100% MFU）。"""
    peak = float(peak_tflops) * 1e12
    if peak <= 0 or tokens_per_sec <= 0 or flops_per_token <= 0:
        return 0.0
    return float(tokens_per_sec) * float(flops_per_token) / peak
