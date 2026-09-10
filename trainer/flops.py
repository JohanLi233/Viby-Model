"""训练 FLOPs / MFU 口径（DeepSeek-V4.1 缩放版）。

MFU = (tokens/s × 每 token 训练 FLOPs) / 硬件峰值。

每 token 训练 FLOPs = 6 × 激活 GEMM 参数 + 注意力二次项：

- **6N**：fwd 2N + bwd 4N。N 只数每 token 真正参与 GEMM 的权重：
  路由专家按 top-k/E 折算；token embedding 与 Engram n-gram 检索表是纯
  寻址（稀疏 gather），不计；MTP/DSpark 草稿层每 token 只多走一遍前向
  （block 内各槽共享同一次前向），lm_head 则会被草稿 logits 再用一次。
- **注意力**：QKᵀ + AV 没有对应权重，按每层实际参与打分的长度 L 计
  6 · n_heads · (head_dim + head_dim) · L（fwd+bwd；K=V 共享 MQA，
  d_qk = d_v = head_dim）。训练走稠密路径：压缩分支全量计算、top-k 掩码
  只改可见性，故 L = min(T, window_size) + T/ratio（ratio=0 的纯滑窗层
  只有窗口）。indexer 的 [B,T,N] 小打分矩阵与压缩器池化不计入（占比 <1%）。
- 不含优化器（Muon Newton-Schulz）FLOPs；墙钟含优化器，因此 MFU 会低于
  纯 GEMM 利用率。这是训练 MFU 的标准口径。

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
    """每 token 参与 GEMM 的激活参数量（MTP/lm_head 的重复前向已折算）。"""
    cfg = model.config
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    mtp_on = int(getattr(cfg, "n_mtp_layers", 0) or 0) > 0
    n = 0
    for path, value in tree_flatten(model.parameters()):
        # 1-D（norm gain / mHC scale·base / attn_sink / router bias）不是 GEMM 权重
        if getattr(value, "ndim", 0) < 2:
            continue
        if _is_lookup_table(path, tied):
            continue
        size = _active_size(path, value, cfg)
        if _is_unembedding(path, tied) and mtp_on:
            # 主干 CE 一次 + DSpark 草稿 logits 一次（block 各槽共享同一次前向）
            size *= 2
        n += size
    return n


def attn_fwdbwd_flops_per_token(config, seq_len: int) -> int:
    """CSA2 注意力的 QKᵀ + AV fwd+bwd FLOPs / token（不含投影 GEMM）。"""
    t = int(seq_len)
    if t <= 0:
        return 0
    heads = int(config.n_heads)
    head_dim = int(config.head_dim)
    window = int(config.window_size)
    ratios = tuple(config.compress_ratios)
    total = 0
    for i in range(int(config.n_layers) + int(config.n_mtp_layers)):
        r = ratios[i] if i < len(ratios) else 0
        length = min(t, window)
        if r > 0:
            # 训练是稠密 prefill：压缩分支的 N = T/ratio 全部算、top-k 只裁可见性
            length = min(t, window + (t + r - 1) // r)
        total += 6 * heads * (head_dim + head_dim) * length
    return total


def training_flops_per_token(model, seq_len: int) -> int:
    """训练一步（fwd+bwd）每个 token 的近似 FLOPs，不含优化器。"""
    return 6 * gemm_active_params(model) + attn_fwdbwd_flops_per_token(
        model.config, seq_len
    )


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
