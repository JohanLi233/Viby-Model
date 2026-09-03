"""训练 FLOPs / MFU 口径。

MFU = (tokens/s × 每 token 训练 FLOPs) / 硬件峰值。

每 token 训练 FLOPs = 6 × 激活 GEMM 参数 + softmax 注意力二次项：

- 6N：PaLM / nanoGPT 的 fwd+bwd 矩阵乘（前向 2N、反向 4N）。
  N 用每次前向真正参与 GEMM 的参数：路由专家按 top-k/E 折算；
  embedding / n-gram 查找表不算；MTP 模块按 ``mtp_steps`` 展开，
  lm_head 再加同样次数（主 CE 已经计过一次）。
- 注意力：QKᵀ + AV 没有对应权重，按 nanoGPT 二次项
  ``6 · n_heads · (d_qk + d_v) · T`` 计入（fwd+bwd，不算因果半边）。
  MLA 的 d_qk = head_dim + qk_rope_head_dim；线性注意力路径只计 GQA
  global 层，KDA 的递推已含在投影的 6N 里。
- 不含优化器（Muon NS）FLOPs；墙钟含优化器，因此 MFU 会低于纯 GEMM
  利用率。这是训练 MFU 的标准口径。

默认峰值 13.5 TFLOPS：M4 Max bf16 稠密 GEMM 实测中位
（``research/MLX_PERF.md`` §0：12.9~14）。换机用 ``--peak_tflops``。
"""

from mlx.utils import tree_flatten

# M4 Max bf16 稠密 GEMM 实测峰值中位（12.9~14 TFLOPS）
DEFAULT_PEAK_TFLOPS = 13.5

_SKIP_LEAVES = frozenset({"expert_bias", "freqs_cos", "freqs_sin", "rope_freqs"})


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _is_lookup_table(path: str, tied: bool) -> bool:
    if "embed_tokens" in path:
        return not tied
    return path.endswith("ngram.table") or ".ngram.table" in path


def _is_unembedding(path: str, tied: bool) -> bool:
    if path.startswith("lm_head") or ".lm_head." in path:
        return True
    return tied and "embed_tokens" in path


def _active_size(path: str, value, top_k: int) -> int:
    size = int(value.size)
    if ".experts." in path and getattr(value, "ndim", 0) >= 3 and value.shape[0] > 0:
        e = int(value.shape[0])
        size = size // e * min(top_k, e)
    return size


def gemm_active_params(model) -> int:
    """每 token 参与 GEMM 的激活参数量（MTP 已按展开步数加权）。"""
    cfg = model.config
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    mtp_on = int(getattr(cfg, "mtp_depth", 0) or 0) > 0
    mtp_steps = int(getattr(cfg, "mtp_steps", 1) or 1) if mtp_on else 0
    n = 0
    for path, value in tree_flatten(model.parameters()):
        if _leaf(path) in _SKIP_LEAVES:
            continue
        if _is_lookup_table(path, tied):
            continue
        size = _active_size(path, value, top_k)
        if "mtp_modules" in path:
            size *= mtp_steps
        elif _is_unembedding(path, tied) and mtp_on:
            size *= 1 + mtp_steps
        n += size
    return n


def _n_gqa_layers(n_layers: int) -> int:
    return sum(1 for i in range(n_layers) if (i + 1) % 4 == 0 or i == n_layers - 1)


def attn_fwdbwd_flops_per_token(config, seq_len: int) -> int:
    """softmax 注意力 QKᵀ+AV 的 fwd+bwd FLOPs / token（不含投影 GEMM）。"""
    t = int(seq_len)
    if t <= 0:
        return 0
    heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)
    n_layers = int(config.num_hidden_layers)
    mtp_extra = (
        int(config.mtp_steps) if int(getattr(config, "mtp_depth", 0) or 0) > 0 else 0
    )
    if not getattr(config, "use_linear_attn", False):
        qk = head_dim + int(config.qk_rope_head_dim)
        n_softmax = n_layers + mtp_extra
        return n_softmax * 6 * heads * (qk + head_dim) * t
    n_gqa = _n_gqa_layers(n_layers) + mtp_extra
    return n_gqa * 6 * heads * (head_dim + head_dim) * t


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
