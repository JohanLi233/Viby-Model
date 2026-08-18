import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn


class VibyConfig:
    model_type = "viby"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        # MLA（DeepSeek V2/V3 风格）：KV 先压缩到 kv_lora_rank 维潜在向量，
        # K/V 使用时再上投影；位置信息由 qk_rope_head_dim 维解耦 RoPE 键携带
        # （跨 head 共享）。Q/K 逐 head 维度 = head_dim + qk_rope_head_dim。
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 192)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)
        # absorbed MLA decode（FlashMLA 思想，仅推理）：decode 时不上投影
        # 历史 K/V，q 侧吸收 W_k_up 后在 kv_lora_rank 维 latent 空间打分，
        # W_v_up 预融进 o_proj；KV cache 改存 (c_kv, k_rope, k_rinv)，
        # 每 token 每层 KV 读取从 H*(head_dim+rope)+H*head_dim 维降到
        # kv_lora_rank + rope + H 维。1=开（默认），0=关。
        self.absorbed_decode = kwargs.get("absorbed_decode", 1)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.original_max_position_embeddings = kwargs.get(
            "original_max_position_embeddings", 2048
        )
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling is None and self.inference_rope_scaling:
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": self.original_max_position_embeddings,
                "attention_factor": 1.0,
                "type": "yarn",
            }

        # Value Residual Learning：第一层 attention 的 V 作为跨层"值残差"，
        # 后续层以可学习比例 λ 混入（v' = (1-λ)v + λ·v_0），缓解深层
        # 注意力过度集中与信息稀释。
        self.use_value_res = kwargs.get("use_value_res", False)
        # 注意力输出门（modded-nanogpt）：对 attention 输出施加逐 head 的
        # 输入条件 sigmoid 门 gate=σ(W_g·x)，W_g 零初始化（初始门 0.5）。
        self.use_attn_gate = kwargs.get("use_attn_gate", False)
        self.mtp_depth = kwargs.get("mtp_depth", 0)
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight", 0.3)
        # HRM 模式（Hierarchical Recurrent Model，对齐 HF HrmText 的 H/L
        # 双状态层次循环）：hrm_H_cycles>0 时启用。num_hidden_layers 表示
        # 每个 stack 的真实层数 P；每个 token 的层求值次数 =
        # H_cycles*(L_cycles+1)*P。训练前向与推理前向完全一致（不展开推理）。
        # L stack（快状态 z_L）与 H stack（慢状态 z_H）互相注入：
        #   z_L <- L_stack(z_L + z_H)  重复 L_cycles 次
        #   z_H <- H_stack(z_H + z_L)  每 H cycle 一次
        # 每层权重只存一份，但每轮循环的 attention K/V 有独立 cache slot
        # （按调用顺序展开 past_key_values），因此不是无状态循环。
        self.hrm_H_cycles = kwargs.get("hrm_H_cycles", 0)
        self.hrm_L_cycles = kwargs.get("hrm_L_cycles", 3)
        # 训练时每个 H cycle 里允许回传梯度的尾部 L cycle 数；与 HF 默认
        # L_bp_cycles 同义。None 表示全部回传（小尺度筛选用）。
        self.hrm_bp_cycles = kwargs.get("hrm_bp_cycles", None)
        self.hrm_emb_scale = kwargs.get("hrm_emb_scale", 1.0)
        # CycleRouter（HRM×MoE 创新，research/HRM_MOE.md）：每个 MoE router
        # 增加 cycle_emb (H*(L+1), hidden)，router 输入叠加第 c 次迭代的
        # 嵌入，让专家按迭代特化——循环第 c 步与第 c' 步的有效权重不再
        # 同一，打破循环引理 F_looped(k,N) ⊆ F_untied(k·N) 的包含上界。
        # 零初始化 ⇒ 初始严格等价无 CycleRouter 的 HRM。仅 HRM 模式生效。
        self.hrm_cycle_router = kwargs.get("hrm_cycle_router", 0)
        # CycleFiLM：每次 stack 调用前对注入状态做 per-cycle scale/shift
        # （零初始化），给每个 cycle 显式"时间身份"。仅 HRM 模式生效。
        self.hrm_cycle_film = kwargs.get("hrm_cycle_film", 0)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.emb_scale = kwargs.get("emb_scale", 1.0)
        # Engram（Cactus Needle）：哈希 n-gram 键值记忆。在指定层把可学习
        # n-gram 表查出的 value 经余弦相似度门控注入残差流，目标是用结构化
        # 记忆替代 dense FFN 的一部分知识存储（攻击 B1 的 bit/param 下界）。
        self.engram_layers = kwargs.get("engram_layers", ())
        self.engram_orders = kwargs.get("engram_orders", (2, 3))
        self.engram_heads = kwargs.get("engram_heads", 0)
        self.engram_slots = kwargs.get("engram_slots", 8192)
        self.engram_sub_dim = kwargs.get("engram_sub_dim", 128)
        # DeepSeekMoE（V3/V4 风格）：n_routed_experts>0 时，第 n_dense_layers
        # 层起的 FFN 替换为 n_shared_experts 个共享专家 + E 个细粒度路由专家。
        # 路由：sigmoid 打分，选择分 = sigmoid 分 + expert_bias（无辅助损失
        # 负载均衡偏置，非梯度参数，由训练循环按负载统计逐步更新）；
        # top-k 命中后用原始 sigmoid 分归一化并乘 routed_scaling_factor。
        self.n_routed_experts = kwargs.get("n_routed_experts", 0)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 6)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", None)
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 2.5)
        # 无辅助损失偏置更新步长 u（V3 大部分训练阶段用 1e-3）；<=0 关闭更新
        self.moe_bias_update_rate = kwargs.get("moe_bias_update_rate", 0.001)

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.n_routed_experts > 0:
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok 必须大于 0")
            if self.num_experts_per_tok > self.n_routed_experts:
                raise ValueError(
                    "num_experts_per_tok 不能大于 n_routed_experts，"
                    "否则 top-k 会重复选中同一专家"
                )
            if self.n_shared_experts < 0:
                raise ValueError("n_shared_experts 不能为负数")
            moe_in = self.moe_intermediate_size or self.intermediate_size
            if moe_in <= 0:
                raise ValueError("moe_intermediate_size/intermediate_size 必须大于 0")
            if self.n_dense_layers < 0 or self.n_dense_layers > self.num_hidden_layers:
                raise ValueError("n_dense_layers 必须在 [0, num_hidden_layers] 范围内")

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "VibyConfig":
        # 旧 checkpoint 的 sidecar 可能带有已删除机制的键，__init__ 通过
        # **kwargs 静默忽略，天然兼容
        data = dict(data)
        data.pop("model_type", None)
        return cls(**data)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_dict(json.load(f))


@dataclass
class CausalLMOutput:
    loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    past_key_values: Optional[list] = None
    hidden_states: Optional[mx.array] = None
    # MTP 辅助 loss 分量（未加权，仅日志展示用；无 MTP 时为 None）
    mtp_loss: Optional[mx.array] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # MLX 融合 RMSNorm kernel（bf16 路径比手动 float32 实现快数倍）
        return mx.fast.rms_norm(x, self.weight, self.eps)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> tuple[mx.array, mx.array]:
    freqs = 1.0 / (
        rope_base ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
    )
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:

            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = mx.clip(
                (mx.arange(dim // 2).astype(mx.float32) - low) / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = mx.arange(end).astype(mx.float32)
    freqs = mx.outer(t, freqs)
    freqs_cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1) * attn_factor
    freqs_sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    # q: (bsz, seq_len, heads, rope_dim)；k: (bsz, seq_len, 1, rope_dim)
    # （MLA 解耦 RoPE 键跨 head 共享）；cos/sin: (seq_len, rope_dim)
    def rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    cos = cos[:, None, :]
    sin = sin[:, None, :]
    q_embed = (q * cos + rotate_half(q) * sin).astype(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).astype(k.dtype)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# 融合 cross-entropy kernel：逐行 logsumexp + 选中项 + softmax-onehot 反向，
# 各一个 Metal kernel。原版链要物化 f32 (M,V) logits、logsumexp、广播减法、
# take_along 等多趟 105MB 级读写（M=B*T, V=6400）；融合后 fwd 只读一遍
# bf16 logits，bwd 读一遍写一遍梯度。行载入 threadgroup memory（V*4 字节
# ≤ 32KB，V=6400 时 25.6KB），全程 f32 累加。手写 VJP 经 mx.custom_function
# 接入 autodiff。labels=-100 的行 loss/梯度均为 0（与原实现一致）。
# 编译失败或 V 超限自动回退原版链。
# ---------------------------------------------------------------------------

_ce_kernel_cache: dict = {}
_CE_KERNEL_DISABLED = False


def _build_ce_kernels(V: int, dtype):
    key = (V, dtype)
    if key in _ce_kernel_cache:
        return _ce_kernel_cache[key]
    mt = _MOE_METAL_TYPE[dtype]
    # 行归约：256 线程先各自扫描，再 threadgroup 树形归约 max / sum
    reduce_src = """
        uint lane = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        threadgroup float row[V_];
        threadgroup float part[256];
        size_t base = (size_t)m * V_;
        for (uint v = lane; v < V_; v += 256) row[v] = float(logits[base + v]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lmax = -INFINITY;
        for (uint v = lane; v < V_; v += 256) lmax = metal::max(lmax, row[v]);
        part[lane] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (lane < s) part[lane] = metal::max(part[lane], part[lane + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mmax = part[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lsum = 0.0f;
        for (uint v = lane; v < V_; v += 256) lsum += metal::exp(row[v] - mmax);
        part[lane] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (lane < s) part[lane] = part[lane] + part[lane + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv_l = 1.0f / part[0];
        int lbl = labels[m];
        bool valid = (lbl >= 0) && (lbl < int(V_));
    """
    fwd_src = (
        reduce_src
        + """
        if (lane == 0) {
            ce[m] = valid ? (mmax + metal::log(part[0]) - row[lbl]) : 0.0f;
        }
    """
    )
    bwd_src = (
        reduce_src
        + f"""
        float c = cot[m];
        for (uint v = lane; v < V_; v += 256) {{
            float p = metal::exp(row[v] - mmax) * inv_l;
            if (v == uint(lbl)) p -= 1.0f;
            g[base + v] = {mt}(valid ? c * p : 0.0f);
        }}
    """
    )
    fwd_src = fwd_src.replace("V_", str(V))
    bwd_src = bwd_src.replace("V_", str(V))
    fwd = mx.fast.metal_kernel(
        name=f"ce_fwd_{V}_{mt}",
        input_names=["logits", "labels"],
        output_names=["ce"],
        source=fwd_src,
    )
    bwd = mx.fast.metal_kernel(
        name=f"ce_bwd_{V}_{mt}",
        input_names=["logits", "labels", "cot"],
        output_names=["g"],
        source=bwd_src,
    )
    _ce_kernel_cache[key] = (fwd, bwd)
    return fwd, bwd


@mx.custom_function
def _ce_rows(logits: mx.array, labels: mx.array) -> mx.array:
    """逐行融合 CE：(M,V) logits + (M,) int32 labels -> (M,) f32 行 loss。"""
    fwd, _ = _build_ce_kernels(logits.shape[-1], logits.dtype)
    M = logits.shape[0]
    return fwd(
        inputs=[logits, labels],
        output_shapes=[(M,)],
        output_dtypes=[mx.float32],
        grid=(256, M, 1),
        threadgroup=(256, 1, 1),
    )[0]


def _ce_rows_vjp(primals, cotangents, outputs):
    logits, labels = primals
    _, bwd = _build_ce_kernels(logits.shape[-1], logits.dtype)
    g = bwd(
        inputs=[logits, labels, cotangents],
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(256, logits.shape[0], 1),
        threadgroup=(256, 1, 1),
    )[0]
    return g, None


_ce_rows.vjp(_ce_rows_vjp)


def cross_entropy(
    logits: mx.array,
    labels: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    global _CE_KERNEL_DISABLED
    V = logits.shape[-1]
    # 行 buffer V*4 字节需放进 32KB threadgroup memory
    if not _CE_KERNEL_DISABLED and V * 4 <= 28 * 1024:
        try:
            flat_labels = labels.reshape(-1).astype(mx.int32)
            ce = _ce_rows(logits.reshape(-1, V), flat_labels)
            if mask is None:
                # 与原链的 mean 语义差异：原链 mean 含 -100 行（其 ce=0），
                # 这里逐行 ce 同样为 0，mean 结果一致
                return mx.mean(ce)
            mask_flat = mask.reshape(-1).astype(ce.dtype)
            return mx.sum(ce * mask_flat) / mx.maximum(mx.sum(mask_flat), mx.array(1.0))
        except Exception:
            _CE_KERNEL_DISABLED = True
    logits_f = logits.astype(mx.float32)
    log_z = mx.logsumexp(logits_f, axis=-1)
    labels_safe = mx.where(labels == -100, 0, labels)
    logp = logits_f - log_z[..., None]
    picked = mx.take_along_axis(logp, labels_safe[..., None], axis=-1).squeeze(-1)
    ce = mx.where(labels == -100, 0.0, -picked)

    ce = ce.reshape(-1)
    if mask is None:
        return mx.mean(ce)

    mask_flat = mask.reshape(-1).astype(ce.dtype)
    return mx.sum(ce * mask_flat) / mx.maximum(mx.sum(mask_flat), mx.array(1.0))


class Attention(nn.Module):
    """MLA（Multi-head Latent Attention，DeepSeek V2/V3 风格）。

    K/V 不按 head 直接投影：先压缩到低秩潜在向量 c = W_DKV·x
    （kv_lora_rank 维），K/V 使用时从 c 上投影；位置信息由独立的解耦
    RoPE 键携带（qk_rope_head_dim 维，跨 head 共享），Q 侧同样拼接
    nope（内容）+ rope（位置）两段。逐 head QK 维度 = head_dim +
    qk_rope_head_dim，V 维度 = head_dim。
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = self.head_dim + self.rope_dim
        self.kv_rank = config.kv_lora_rank
        self.is_causal = True
        # 合并投影：q+kv_down+k_rope 共用一个 (D→1248) GEMM、
        # k_up+v_up 共用一个 (rank→2*H*hd) GEMM——输入相同，合并减少 kernel
        # 数并放大 GEMM 形状（小 GEMM 在 M4 Max 上只跑 3-8 TFLOPS）。
        # 数学严格等价（同 in_features、同均匀初始化分布）。Muon 侧由
        # BatchedMuon.segment_map 按行段分别 NS，保持逐矩阵语义。
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.n_heads * self.qk_dim + config.kv_lora_rank + self.rope_dim,
            bias=False,
        )
        self.kv_up_proj = nn.Linear(
            config.kv_lora_rank, 2 * self.n_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        # QK-norm 只作用于 nope（内容）段，rope 段保持原始 RoPE 几何。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = config.flash_attn
        # Value Residual：可学习混合系数 θ（λ = sigmoid(θ)，θ=-2 → λ≈0.12，
        # 初始以本层 V 为主，温和偏离基线）
        self.v_res_lambda = (
            mx.array(-2.0) if getattr(config, "use_value_res", False) else None
        )
        # 注意力输出门：输入条件的逐 head sigmoid 门，零初始化（初始 0.5）。
        # 加在 o_proj 之前，逐 head 调制 attention 输出幅度。
        self.attn_gate = None
        if getattr(config, "use_attn_gate", False):
            self.attn_gate = nn.Linear(config.hidden_size, self.n_heads, bias=True)
            self.attn_gate.weight = mx.zeros_like(self.attn_gate.weight)
            self.attn_gate.bias = mx.zeros_like(self.attn_gate.bias)
        # absorbed decode（FlashMLA 思想，仅推理、默认开）：decode 时不上
        # 投影 K/V——q 侧吸收 W_k_up 后在 192 维 latent 空间直接打分，输出
        # 侧把 W_v_up 预融进 o_proj（(H*192,768) 单矩阵 A）；KV cache 改存
        # (c_kv, k_rope, k_rinv)，每 token 每层的 KV 读取从 H×224 维降到
        # 225 维（~7×）。当前 token 的逐 head rms 由 W_k@c 直算后取
        # rsqrt(mean+eps)（与上投影路径同序，无需额外常驻权重）。
        # 与 use_value_res 兼容：cache 存混合前 c_pre（K 侧语义），V 侧
        # 按 v'=(1-λ)v_ℓ+λv_0 在 latent 空间展开——o_lat = (1-λ)(α@c_ℓ)
        # + λ(α@c_0)，λ 项使用 layer0 的 latent cache 与跨层折叠矩阵
        # W_UV0ℓ = W_oℓ·W_v0（VibyModel 建模后接线 _abs_attn0 模块引用）。
        # 折叠权重在首次 decode 时物化一次
        # （eval/generate 加载后权重不再变；训练路径不会进入本分支）。
        self.absorbed_decode = bool(
            getattr(config, "absorbed_decode", 1)
        ) and not getattr(config, "hrm_H_cycles", 0)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        past_key_value: Optional[tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        layer_idx: Optional[int] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        value_residual: Optional[list] = None,
    ) -> tuple[mx.array, Optional[tuple[mx.array, mx.array]]]:
        bsz, seq_len, _ = x.shape
        if (
            self.absorbed_decode
            and use_cache
            and (seq_len == 1 or past_key_value is not None)
        ):
            # decode / 带 past 的续算（含投机验证）：全程 latent 空间，
            # 历史 K/V 不上投影。纯 prefill（past=None 且 seq>1）仍走下面
            # 的 flash 路径，但按 latent 格式落 cache。
            return self._absorbed_call(
                x,
                position_embeddings,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                value_residual=value_residual,
            )
        qkv = self.qkv_proj(x)
        q_flat, c_kv, k_rope_flat = mx.split(
            qkv,
            [self.n_heads * self.qk_dim, self.n_heads * self.qk_dim + self.kv_rank],
            axis=-1,
        )
        xq = q_flat.reshape(bsz, seq_len, self.n_heads, self.qk_dim)
        k_rope = k_rope_flat[:, :, None, :]  # (B, T, 1, rope_dim) 跨 head 共享
        kv = self.kv_up_proj(c_kv)
        xk, xv = mx.split(kv, 2, axis=-1)
        xk = xk.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        q_nope, q_rope = mx.split(xq, [self.head_dim], axis=-1)

        # Value Residual Learning：第一层把本层 V 存入共享 list，后续层以
        # 可学习比例 λ = sigmoid(θ) 混入第一层 V（v' = (1-λ)v + λ·v_0）。
        # list 是跨层共享载体（compile 兼容）。
        if value_residual is not None:
            if len(value_residual) == 0:
                value_residual.append(xv)
            else:
                lam = mx.sigmoid(self.v_res_lambda).astype(xv.dtype)
                xv = (1.0 - lam) * xv + lam * value_residual[0]
        if self.absorbed_decode and use_cache:
            # latent cache 的逐 head 归一化倒数（fp32，与 fast.rms_norm
            # 内部同精度）；decode 时直接乘在 latent 打分上
            k_rinv = mx.rsqrt(
                mx.mean(mx.square(xk.astype(mx.float32)), axis=-1) + self.k_norm.eps
            )  # (B, T, H)
        q_nope, xk = self.q_norm(q_nope), self.k_norm(xk)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        if self.absorbed_decode and use_cache:
            k_rope_lat = k_rope[:, :, 0, :]  # (B, T, rope_dim) 共享键
        k_rope = mx.broadcast_to(k_rope, (bsz, seq_len, self.n_heads, self.rope_dim))
        xq = mx.concatenate([q_nope, q_rope], axis=-1)
        xk = mx.concatenate([xk, k_rope], axis=-1)

        if past_key_value is not None:
            xk = mx.concatenate([past_key_value[0], xk], axis=1)
            xv = mx.concatenate([past_key_value[1], xv], axis=1)
        if use_cache:
            # absorbed 模式 prefill 落 latent cache（past 必为 None，
            # 带 past 的续算已在入口分支拦截）；普通模式落上投影后的 K/V。
            if self.absorbed_decode:
                past_kv = (c_kv, k_rope_lat, k_rinv)
            else:
                past_kv = (xk, xv)
        else:
            past_kv = None

        # (bsz, heads, seq_len, qk_dim)
        xq = xq.transpose(0, 2, 1, 3)
        if mask_is_full is None:
            mask_is_full = attention_mask is None or bool(
                mx.all(attention_mask == 1).item()
            )
        scale = 1.0 / math.sqrt(self.qk_dim)
        if (
            self.flash
            and seq_len > 1
            and past_key_value is None
            and self.dropout == 0.0
            and (mask_is_full or causal_bias is not None)
        ):
            # MLA 各 head 独立上投影 K/V，等价于 MHA，可直接走 flash。
            # 有 padding/doc_mask 时使用预构造的 causal+pad/seg 融合 mask。
            mask = "causal" if mask_is_full else causal_bias
            output = mx.fast.scaled_dot_product_attention(
                xq,
                xk.transpose(0, 2, 1, 3),
                xv.transpose(0, 2, 1, 3),
                scale=scale,
                mask=mask,
            )
        else:
            xk = xk.transpose(0, 2, 1, 3)
            xv = xv.transpose(0, 2, 1, 3)

            scores = (xq @ mx.swapaxes(xk, -1, -2)) * scale

            if self.is_causal:
                causal_mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(
                    scores.dtype
                )
                scores = scores.at[..., -seq_len:].add(causal_mask)

            if attention_mask is not None:
                key_len = scores.shape[-1]
                am = attention_mask
                if am.shape[1] < key_len:
                    pad = mx.ones((am.shape[0], key_len - am.shape[1]), dtype=am.dtype)
                    am = mx.concatenate([am, pad], axis=1)
                elif am.shape[1] > key_len:
                    am = am[:, -key_len:]
                scores = (
                    scores + (1.0 - am[:, None, None, :].astype(scores.dtype)) * -1e9
                )

            # causal_bias（doc_mask 段掩码 / pad 融合掩码）在 eager 分支同样生效；
            # 与上面的 pad 处理重叠时只是 -1e9 叠加，语义不变
            if causal_bias is not None:
                scores = scores + causal_bias.astype(scores.dtype)

            attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                xq.dtype
            )
            attn_weights = self.attn_dropout(attn_weights)
            output = attn_weights @ xv

        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        if self.attn_gate is not None:
            gate = mx.sigmoid(self.attn_gate(x).astype(mx.float32))
            output = output * mx.repeat(
                gate.astype(output.dtype), self.head_dim, axis=-1
            )
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

    def _absorbed_call(
        self,
        x: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        past_key_value: Optional[tuple] = None,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        value_residual: Optional[list] = None,
    ) -> tuple[mx.array, tuple]:
        """Absorbed MLA decode：历史 K/V 不上投影，latent 空间打分。

        cache = (c_kv (B,T,rank), k_rope (B,T,rope), k_rinv (B,T,H))。
        q 侧吸收 W_k_up 与 k_norm 的 γ：q_abs[h] = (q̂[h]⊙γ) @ Wk[h]；
        内容分 = (q_abs[h]·c_t)·rinv[h,t]，位置分用跨 head 共享的 k_rope。
        输出侧 o_lat[h] = Σ_t α[h,t]·c_t，再过预融合矩阵
        W_UV（W_v 融进 W_o，(hidden, H*rank)）。打分/softmax 在 fp32，
        与上投影路径的精度口径对齐。
        """
        bsz, seq_len, _ = x.shape
        if not hasattr(self, "_abs_wk"):
            # 折叠权重懒物化（object.__setattr__ 避免进入 parameters /
            # save_weights 树；首次 decode 时权重已冻结，只算一次）
            d, h, r = self.head_dim, self.n_heads, self.kv_rank
            wk = self.kv_up_proj.weight[: h * d].reshape(h, d, r)
            wv = self.kv_up_proj.weight[h * d :].reshape(h, d, r)
            wuv = mx.concatenate(
                [
                    self.o_proj.weight[:, i * d : (i + 1) * d].astype(mx.float32)
                    @ wv[i].astype(mx.float32)
                    for i in range(h)
                ],
                axis=1,
            ).astype(self.o_proj.weight.dtype)  # (hidden, H*rank)
            object.__setattr__(self, "_abs_wk", mx.contiguous(wk))
            object.__setattr__(self, "_abs_wuv", mx.contiguous(wuv))
            if hasattr(self, "_abs_attn0"):
                # v_res 兼容：layer0 的 W_v0 与本层 W_o 的跨层折叠，
                # 供 λ·v_0 项在 latent 空间直接求值。此刻（首次 decode）
                # 才读 layer0 的当前权重，load_weights 后取值才正确。
                wv0 = self._abs_attn0.kv_up_proj.weight[h * d :].reshape(h, d, r)
                wuv0 = mx.concatenate(
                    [
                        self.o_proj.weight[:, i * d : (i + 1) * d].astype(mx.float32)
                        @ wv0[i].astype(mx.float32)
                        for i in range(h)
                    ],
                    axis=1,
                ).astype(self.o_proj.weight.dtype)
                object.__setattr__(self, "_abs_wuv0", mx.contiguous(wuv0))
                mx.eval(self._abs_wk, self._abs_wuv, self._abs_wuv0)
            else:
                mx.eval(self._abs_wk, self._abs_wuv)

        qkv = self.qkv_proj(x)
        q_flat, c_kv, k_rope_flat = mx.split(
            qkv,
            [self.n_heads * self.qk_dim, self.n_heads * self.qk_dim + self.kv_rank],
            axis=-1,
        )
        xq = q_flat.reshape(bsz, seq_len, self.n_heads, self.qk_dim)
        q_nope, q_rope = mx.split(xq, [self.head_dim], axis=-1)
        q_nope = self.q_norm(q_nope)

        # 当前 token 的逐 head rms 由 W_k@c 直算（与上投影路径同序）
        wk_flat = self.kv_up_proj.weight[: self.n_heads * self.head_dim]
        k_cur = (c_kv @ wk_flat.T).reshape(bsz, seq_len, self.n_heads, self.head_dim)
        k_rinv = mx.rsqrt(
            mx.mean(mx.square(k_cur.astype(mx.float32)), axis=-1) + self.k_norm.eps
        )  # (B, S, H) fp32

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(
            q_rope, k_rope_flat[:, :, None, :], cos, sin
        )
        k_rope = k_rope[:, :, 0, :]  # (B, S, rope_dim)

        if past_key_value is not None:
            c_all = mx.concatenate([past_key_value[0], c_kv], axis=1)
            kr_all = mx.concatenate([past_key_value[1], k_rope], axis=1)
            rinv_all = mx.concatenate([past_key_value[2], k_rinv], axis=1)
        else:
            c_all, kr_all, rinv_all = c_kv, k_rope, k_rinv
        past_kv = (c_all, kr_all, rinv_all)
        key_len = c_all.shape[1]

        # value residual（latent 版）：layer0 把全量 c cache（含当前 token）
        # 挂上共享 list；后续层的 λ·v_0 项用它在 latent 空间求值。
        # cache 存的是混合前 c_pre（K 侧语义保持上投影路径一致）。
        vres_base = (
            self.v_res_lambda is not None
            and value_residual is not None
            and len(value_residual) == 0
        )
        if vres_base:
            value_residual.append(c_all)
        vres_mix = (
            self.v_res_lambda is not None
            and value_residual is not None
            and not vres_base
            and hasattr(self, "_abs_attn0")
        )
        c0_all = value_residual[0] if vres_mix else None

        # q 吸收：q_abs[h] = (q̂[h]⊙γ) @ Wk[h] → (B, H, S, rank)
        gamma = self.k_norm.weight
        qg = (q_nope * gamma.astype(q_nope.dtype)).transpose(0, 2, 1, 3)
        q_abs = mx.matmul(qg, self._abs_wk)  # (B,H,S,96)@(H,96,rank)

        # 内容分 + 共享 rope 分，fp32 打分（对齐上投影路径 softmax 口径）
        scores = mx.matmul(q_abs, c_all.transpose(0, 2, 1)[:, None]).astype(mx.float32)
        scores = scores * rinv_all.transpose(0, 2, 1)[:, :, None, :]
        scores = scores + mx.matmul(
            q_rope.transpose(0, 2, 1, 3), kr_all.transpose(0, 2, 1)[:, None]
        ).astype(mx.float32)
        scores = scores * (1.0 / math.sqrt(self.qk_dim))

        if mask_is_full is None:
            mask_is_full = attention_mask is None or bool(
                mx.all(attention_mask == 1).item()
            )
        if self.is_causal and seq_len > 1:
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(
                scores.dtype
            )
            scores = scores.at[..., -seq_len:].add(causal_mask)
        if attention_mask is not None and not mask_is_full:
            am = attention_mask
            if am.shape[1] < key_len:
                pad = mx.ones((am.shape[0], key_len - am.shape[1]), dtype=am.dtype)
                am = mx.concatenate([am, pad], axis=1)
            elif am.shape[1] > key_len:
                am = am[:, -key_len:]
            scores = scores + (1.0 - am[:, None, None, :].astype(scores.dtype)) * -1e9
        if causal_bias is not None:
            scores = scores + causal_bias.astype(scores.dtype)

        attn = mx.softmax(scores, axis=-1).astype(c_all.dtype)
        # 输出侧：o_lat[h] = Σ_t α[h,t]·c_t → (B,H,S,rank)
        o_lat = mx.matmul(attn, c_all[:, None, :, :])
        if vres_mix:
            # v' = (1-λ)v_ℓ + λ·v_0 的 latent 展开：两路分别过各自的
            # 折叠矩阵（W_oℓ·W_vℓ 与 W_oℓ·W_v0）
            lam = mx.sigmoid(self.v_res_lambda).astype(o_lat.dtype)
            o_lat0 = mx.matmul(attn, c0_all[:, None, :, :])
        o_flat = o_lat.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        if vres_mix:
            o0_flat = o_lat0.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        if self.attn_gate is not None:
            # 逐 head 门与 o_proj 线性可交换：直接乘在 latent 输出上
            gate = mx.sigmoid(self.attn_gate(x).astype(mx.float32))
            gate = mx.repeat(gate.astype(o_flat.dtype), self.kv_rank, axis=-1)
            o_flat = o_flat * gate
            if vres_mix:
                o0_flat = o0_flat * gate
        if vres_mix:
            output = (1.0 - lam) * (o_flat @ self._abs_wuv.T) + lam * (
                o0_flat @ self._abs_wuv0.T
            )
        else:
            output = o_flat @ self._abs_wuv.T
        output = self.resid_dropout(output)
        return output, past_kv


_ACT2FN = {"silu": nn.silu, "gelu": nn.gelu, "relu": nn.relu}


class FeedForward(nn.Module):
    def __init__(
        self,
        config: VibyConfig,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = _ACT2FN[config.hidden_act]

    def __call__(self, x: mx.array, step_idx: Optional[int] = None) -> mx.array:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(h)


class MoEGate(nn.Module):
    """DeepSeek V3/V4 风格 MoE router：sigmoid 打分 + 无辅助损失偏置均衡。

    选择分 = sigmoid(x·W) + expert_bias；top-k 命中后取原始 sigmoid 分
    （不含 bias）在命中集上归一化并乘 routed_scaling_factor 作为合并权重。
    expert_bias 是 freeze 的非梯度参数，由训练循环按 V3 规则更新：
    超载专家 bias 减小、欠载增大（b_i ← b_i − u·sign(load_i − mean_load)）。

    collect_stats（python 属性，compile 期常量）开启时，forward 额外记录
    每专家 token 计数到 self.last_load（图节点，随 loss 一同输出），
    供训练循环做偏置更新；eval/生成侧保持关闭，零开销。
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.n_routed = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.scaling = float(getattr(config, "routed_scaling_factor", 2.5))
        std = config.hidden_size**-0.5
        self.weight = mx.random.uniform(-std, std, (self.n_routed, config.hidden_size))
        self.expert_bias = mx.zeros((self.n_routed,))
        self.freeze(recurse=False, keys=["expert_bias"])
        # CycleRouter（HRM×MoE，research/HRM_MOE.md）：router 输入叠加第 c
        # 次迭代的嵌入 cycle_emb[c]（零初始化 ⇒ 初始严格等价）。不走
        # "logit 偏置"形式：选择经 argpartition+stop_gradient 不可微，
        # logit 偏置永远收不到梯度；嵌入形式经 sigmoid 分到 w 回传，
        # 可微、且同样改变离散选择。专家借此按迭代特化，打破循环引理
        # F_looped(k,N) ⊆ F_untied(k·N) 的包含上界。
        # n_cycles = H*(L+1)：L 调用 c=h_idx*(L+1)+l_idx，H 调用占每段
        # 最后一位 c=h_idx*(L+1)+L。仅 HRM 模式且 hrm_cycle_router 开启。
        n_cycles = int(getattr(config, "hrm_H_cycles", 0) or 0) * (
            int(getattr(config, "hrm_L_cycles", 3) or 0) + 1
        )
        self.cycle_emb = (
            mx.zeros((n_cycles, config.hidden_size))
            if getattr(config, "hrm_cycle_router", 0) and n_cycles > 0
            else None
        )
        self.collect_stats = False
        self.last_load = None

    def __call__(
        self, x: mx.array, step_idx: Optional[int] = None
    ) -> tuple[mx.array, mx.array]:
        # 路由分数在 fp32 下计算（V3 约定，bf16 模型下保证 top-k 选择稳定）
        if self.cycle_emb is not None and step_idx is not None:
            x = x + self.cycle_emb[step_idx].astype(x.dtype)
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        sel = scores + self.expert_bias.astype(mx.float32)
        idx = mx.argpartition(-sel, self.top_k - 1, axis=-1)[..., : self.top_k]
        # 路由选择是离散操作、不可微；不断开的话 idx 会经 argpartition 链回
        # router 权重，使 autodiff 向使用 idx 的 gather/scatter 算子请求
        # indices 的 VJP（不支持）
        idx = mx.stop_gradient(idx)
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            w = w / mx.maximum(w.sum(axis=-1, keepdims=True), mx.array(1e-9))
        w = w * self.scaling
        if self.collect_stats:
            # object.__setattr__ 绕过 Module 的参数注册：stats 是图节点，
            # 不能进入 parameters()（否则 compile trace 的占位数组会污染
            # mx.eval(model.parameters()) 与 checkpoint）
            # HRM 模式下同一 gate 每次前向被调用 H*(L+1) 次：累加而非覆盖，
            # 让 bias 均衡看到全部迭代的负载（每次前向开始时由
            # VibyForCausalLM 统一清零）。
            counts = (
                mx.zeros((self.n_routed,), dtype=mx.float32)
                .at[idx.reshape(-1)]
                .add(1.0)
            )
            if self.last_load is not None:
                counts = counts + self.last_load
            object.__setattr__(self, "last_load", counts)
        return idx.astype(mx.int32), w.astype(x.dtype)


class _StackedExperts(nn.Module):
    """堆叠路由专家权重：(E, out, in) 张量，前向用广播 matmul 批量计算。

    gate/up 投影合并存为单张量 gate_up_w (E, 2*I, D)：前向一次 GEMM 出
    (E, C, 2I) 再 split，比两次独立 GEMM 的 kernel 数少、单 GEMM 更大
    （GPU 利用率高），数学严格等价（同分布独立初始化）。

    独立模块以便优化器按路径名（"*.experts.*"）把 3D 专家权重分进
    AdamW 组——Muon 对 ndim>2 参数会 reshape 成 (E, out*in) 整体正交化，
    跨专家耦合尺度，不是逐专家语义。
    """

    def __init__(self, n_routed: int, moe_in: int, dim: int):
        super().__init__()
        std_in = dim**-0.5
        std_moe = moe_in**-0.5
        self.gate_up_w = mx.random.uniform(-std_in, std_in, (n_routed, 2 * moe_in, dim))
        self.down_w = mx.random.uniform(-std_moe, std_moe, (n_routed, dim, moe_in))


# ---------------------------------------------------------------------------
# 融合 Metal kernel：decode/极小批量 MoE 路由专家前向（仅推理，无 autodiff）
# ---------------------------------------------------------------------------
# decode（T=1）时 _dense_forward 的全专家广播 matmul 要读全部 E 个专家权重
# （100M 配置 ~15.3MB/层）且 router+专家每层 ~18 个 kernel 发射；decode 的
# 主瓶颈是 GPU 侧大量小 kernel 的固定调度开销。这里把整个路由 FFN 融合成
# 3 个 kernel，且只读 top-k 命中专家的权重（~2.9MB/层）：
#   moe_router: scores=sigmoid(x·W^T)+expert_bias，lane 分专家算分，
#               threadgroup barrier 后 lane0 串行 top-k，归一化×scaling；
#   moe_up:     h[m,k,i] = silu(x[m]·Wgu[e,i,:]) * (x[m]·Wgu[e,I+i,:])
#   moe_down:   out[m,d] = Σ_k w[m,k] · (h[m,k,:] · down_w[e,d,:])
# 内存访问按 simdgroup（32 lane）协作布局：连续 lane 读连续地址（合并访问），
# 点积经 simd_sum 归约；D/I/K 编译期注入为常量（循环可 unroll）。
# 调用约定（mlx 0.32 实测）：
# - 不传 template（host 下发 0.65us vs 3.11us），Metal 类型名直接注入源码；
# - 小输入会被放 constant 地址空间（随尺寸变化），body 内一律直接下标索引，
#   不声明局部 device/constant 指针；
# - JIT 编译是 lazy 的，首次调用后需 mx.eval 触发，失败则整体回退稠密路径。

_MOE_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_moe_decode_kernel_cache: dict = {}


def _build_moe_decode_kernels(D, moe_in, K, E, dtype, norm_topk, scaling):
    """构建 3 个融合 kernel：router（打分+top-k 选择）、up（SwiGLU 前半）、
    down（加权合并）。按 (D,moe_in,K,E,dtype,norm,scaling) 缓存。"""
    key = (D, moe_in, K, E, dtype, norm_topk, scaling)
    if key in _moe_decode_kernel_cache:
        return _moe_decode_kernel_cache[key]
    mt = _MOE_METAL_TYPE[dtype]
    norm_code = (
        f"float scale = {scaling}f / metal::max(wsum, 1e-9f);"
        if norm_topk
        else f"float scale = {scaling}f;"
    )
    # 每 token 一个 simdgroup：lane 分专家算分进 threadgroup 数组，
    # barrier 后 lane0 串行 K 轮 argmax（E≤1024；选择分连续浮点 tie 概率 0，
    # tie 时与 argpartition 的选择可能不同——数学上等价的合法 top-k）。
    router_src = f"""
        uint lane = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        threadgroup float sel[{E}];
        threadgroup float scr[{E}];
        size_t xb = (size_t)m * {D};
        for (uint e = lane; e < {E}; e += 32) {{
            size_t wb = (size_t)e * {D};
            float acc = 0.0f;
            for (uint d = 0; d < {D}; d++) {{
                acc += float(x[xb + d]) * float(weight[wb + d]);
            }}
            float s = 1.0f / (1.0f + metal::exp(-acc));
            scr[e] = s;
            sel[e] = s + float(bias[e]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {{
            float wsum = 0.0f;
            float wk[{K}];
            int ik[{K}];
            for (uint k = 0; k < {K}; k++) {{
                int best = 0;
                float bv = sel[0];
                for (uint e = 1; e < {E}; e++) {{
                    if (sel[e] > bv) {{ bv = sel[e]; best = e; }}
                }}
                sel[best] = -1.0f;
                ik[k] = best;
                wk[k] = scr[best];
                wsum += scr[best];
            }}
            {norm_code}
            for (uint k = 0; k < {K}; k++) {{
                idx[m * {K} + k] = ik[k];
                w[m * {K} + k] = {mt}(wk[k] * scale);
            }}
        }}
    """
    up_src = f"""
        uint lane = thread_position_in_grid.x;
        uint ki = thread_position_in_grid.y;
        uint m = thread_position_in_grid.z;
        uint k = ki / {moe_in};
        uint i = ki % {moe_in};
        int e = idx[m * {K} + k];
        size_t xb = (size_t)m * {D};
        size_t eb = (size_t)e * {2 * moe_in * D} + (size_t)i * {D};
        float g = 0.0f, u = 0.0f;
        for (uint d = lane; d < {D}; d += 32) {{
            float xv = float(x[xb + d]);
            g += xv * float(gate_up_w[eb + d]);
            u += xv * float(gate_up_w[eb + {moe_in * D} + d]);
        }}
        g = metal::simd_sum(g);
        u = metal::simd_sum(u);
        if (lane == 0) {{
            h[m * {K * moe_in} + k * {moe_in} + i] = {mt}(g / (1.0f + metal::exp(-g)) * u);
        }}
    """
    down_src = f"""
        uint lane = thread_position_in_grid.x;
        uint d = thread_position_in_grid.y;
        uint m = thread_position_in_grid.z;
        float acc = 0.0f;
        for (uint k = 0; k < {K}; k++) {{
            int e = idx[m * {K} + k];
            size_t db = (size_t)e * {D * moe_in} + (size_t)d * {moe_in};
            size_t hb = (size_t)m * {K * moe_in} + k * {moe_in};
            float inner = 0.0f;
            for (uint i = lane; i < {moe_in}; i += 32) {{
                inner += float(h[hb + i]) * float(down_w[db + i]);
            }}
            acc += float(w[m * {K} + k]) * metal::simd_sum(inner);
        }}
        if (lane == 0) {{
            out[m * {D} + d] = {mt}(acc);
        }}
    """
    router = mx.fast.metal_kernel(
        name=f"moe_router_{D}_{E}_{K}_{mt}",
        input_names=["x", "weight", "bias"],
        output_names=["idx", "w"],
        source=router_src,
    )
    up = mx.fast.metal_kernel(
        name=f"moe_up_{D}_{moe_in}_{K}_{mt}",
        input_names=["x", "gate_up_w", "idx"],
        output_names=["h"],
        source=up_src,
    )
    down = mx.fast.metal_kernel(
        name=f"moe_down_{D}_{moe_in}_{K}_{mt}",
        input_names=["h", "down_w", "w", "idx"],
        output_names=["out"],
        source=down_src,
    )
    _moe_decode_kernel_cache[key] = (router, up, down)
    return router, up, down


class MoEFeedForward(nn.Module):
    """DeepSeekMoE FFN：共享专家（每 token 必走）+ top-k 路由专家。

    三路径实现（按 (token,choice) 对数 G = B*T*K 静态选择）：
    - G <= _KERNEL_MAX_PAIRS（decode / 极小批量）：融合 Metal kernel 路径——
      3 个手写 kernel 完成 router（打分+top-k）、SwiGLU 前半、加权合并，
      只读 top-k 命中专家权重，kernel 数与权重读取量都远小于稠密路径
      （见上方注释）。不可微，仅推理前向；JIT 编译失败自动永久回退。
      router.collect_stats 开启（训练负载统计）时不走本路径。
      注意：训练 batch 必须保证 G > _KERNEL_MAX_PAIRS（即
      bs*seq*top_k > 512，正常训练恒成立），否则会走不可微路径。
    - G <= _DENSE_MAX_PAIRS（小 prefill）：稠密批量路径——堆叠
      权重广播 matmul 一次算完全部专家，按路由权重（非命中为 0）加权
      合并。无 host sync、可微、可与 mx.compile 共存。
    - G 更大（训练 / 大 prefill）：稀疏桶路径——(专家, 桶内序号) 散布到
      (E, C, D) padded 桶，三次 batched GEMM 算完。每 token 只算 top-k
      个专家（FLOPs ≈ 1.1×G，负载均衡时），内存 (E,C,D) 有界；每层
      ~14 个胖算子 + 一次 host sync。数据依赖形状 → MoE 训练不走
      mx.compile（BaseTrainer 检测到 MoE 自动回退 eager）。
    注：曾评估 mx.gather_mm 逐对稀疏路径，其反向会对专家权重物化
    (G,I,D) 临时张量（48×640×6 时 ~29GB/矩阵，超 Metal 单 buffer 上限），
    且 1×D 微 GEMM 利用率极低（mlx 0.32 实测），故不采用。
    """

    _DENSE_MAX_PAIRS = 4096  # G 小于该值走稠密批量路径
    _KERNEL_MAX_PAIRS = 512  # G 小于该值走融合 kernel 路径（decode/极小批量）
    _KERNEL_VERIFIED: set = set()  # 编译验证通过的 (D,I,K,dtype)
    _KERNEL_DISABLED = False  # 任一 shape 编译失败则整体禁用 kernel 路径

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.n_routed = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        moe_in = int(config.moe_intermediate_size or config.intermediate_size)
        self.moe_in = moe_in
        self.router = MoEGate(config)
        self.experts = _StackedExperts(self.n_routed, moe_in, config.hidden_size)
        n_shared = int(getattr(config, "n_shared_experts", 1))
        self.shared = (
            FeedForward(config, intermediate_size=moe_in * n_shared)
            if n_shared > 0
            else None
        )
        # VIBY_DEBUG_MEM 用：本模块训练期见过的最大对齐桶容量 C（python int，
        # 不进图不进参数），由训练循环在日志点读取并清零
        self._c_max_seen = 0

    def _dense_forward(self, x, idx, w):
        """稠密批量路径：全专家广播 matmul + 路由权重加权合并。"""
        B, T, D = x.shape
        M = B * T
        # 路由权重稠密化：S (M, E)，每行 top-k 个非零（top-k 内专家不重复）
        S = (
            mx.zeros((M, self.n_routed), dtype=x.dtype)
            .at[mx.arange(M)[:, None], idx.reshape(M, self.top_k)]
            .add(w.reshape(M, self.top_k).astype(x.dtype))
        )
        xf = x.reshape(M, D)
        # 广播 matmul：(M,D) @ (E,D,2I) -> (E,M,2I)，split 得 gate/up 两半
        g, u = mx.split(xf @ self.experts.gate_up_w.swapaxes(-1, -2), 2, axis=-1)
        h = nn.silu(g) * u
        y = h @ self.experts.down_w.swapaxes(-1, -2)  # (E,M,D)
        return (y * S.T[..., None].astype(y.dtype)).sum(axis=0).reshape(B, T, D)

    def _sparse_forward(self, x, idx, w):
        """稀疏桶路径：按专家 argsort 后散布到 (E, C, D) padded 桶，
        三次 batched GEMM 算完，输出端 scatter-add 直接累加回 token。
        C = 本批最大桶容量（host sync 取标量）；padding 行保持 0 且
        永不读回，数学上严格等价。

        排名用 argsort 而非 (G,E) one-hot+cumsum：后者在 G=24576 时要
        物化 (G,E) 并做逐列 scan，实测 1.1ms/层（全层最碎的一段）；
        argsort 是 GPU radix sort，加 bincount 共 ~0.1ms。排序还让同
        专家的 pair 在桶内连续，scatter/gather 访问局部性更好。
        输出端用 f32 累加器的 scatter-add 替代「padded 加权 → gather →
        reshape(M,K,D).sum」，省掉 wb padded 缓冲和一次全量读写。
        相比逐专家 python 循环：每层 ~12 个胖算子 vs ~200 个小算子分发。
        FLOPs = E×C ≈ 1.1×G（负载均衡时），内存 (E,C,D) 有界。路由极度
        倾斜时 C 趋近 G，退化到 ~E×G/E 行计算，仍正确只是变慢——bias
        均衡机制防止出现。数据依赖形状 → MoE 训练不走 mx.compile。
        """
        B, T, D = x.shape
        K = self.top_k
        E = self.n_routed
        M = B * T
        G = M * K
        exps = idx.reshape(G)  # (G,)
        xf = x.reshape(M, D)

        order = mx.argsort(exps)  # 同专家的 pair 连续
        exps_s = exps[order]
        counts = mx.zeros((E,), dtype=mx.int32).at[exps].add(1)  # bincount
        C = int(mx.max(counts).item())  # 每层唯一一次 host sync
        # 容量向上对齐 512：C 随路由负载逐层逐步漂移，裸值使 Metal 分配器
        # 缓存块尺寸持续失配；对齐后每层仅 1-2 个离散尺寸，缓存稳定复用。
        # padding 行恒为 0 且永不读回，数学严格不变。
        C = (C + 511) // 512 * 512
        if C > self._c_max_seen:
            self._c_max_seen = C
        offsets = mx.cumsum(counts) - counts  # (E,) 各专家桶起始
        # 排序后位置 p 在其专家桶内的序号 = p - 桶起始；slot 全局唯一
        slot = exps_s * C + (mx.arange(G, dtype=mx.int32) - offsets[exps_s])
        tok_s = (order // K).astype(mx.int32)  # 排序后各 pair 的源 token
        w_s = w.reshape(-1)[order]

        xb = mx.zeros((E * C, D), dtype=x.dtype).at[slot].add(xf[tok_s])
        xb = xb.reshape(E, C, D)
        g, u = mx.split(xb @ self.experts.gate_up_w.swapaxes(-1, -2), 2, axis=-1)
        h = nn.silu(g) * u
        y = h @ self.experts.down_w.swapaxes(-1, -2)  # (E,C,D)
        y_s = y.reshape(E * C, D)[slot]  # (G,D) 排序序收回
        # 加权乘在 bf16 做（autodiff 只需驻留 y_s/w_s 两个小量）；
        # 原先 y_s/w_s 先升 f32 再乘，反向要驻留 (G,D) f32 乘子 ≈300MB/层。
        # 累加器仍是 f32（精度关键的求和项不丢精度）。
        yw = (y_s * w_s[:, None].astype(y_s.dtype)).astype(mx.float32)
        # K 个专家输出按 token 累加（f32 累加器，精度优于 bf16 原子加）
        out = mx.zeros((M, D), dtype=mx.float32).at[tok_s].add(yw)
        return out.astype(x.dtype).reshape(B, T, D)

    def _kernel_forward(self, x):
        """融合 Metal kernel 路径：3 个 kernel 完成 router 打分+top-k 选择、
        SwiGLU 前半、加权合并（无梯度）。router kernel 输出 idx 按分数降序，
        与 MoEGate 的 argpartition 无序输出在加权求和下数学等价。"""
        B, T, D = x.shape
        M = B * T
        K, inter, E = self.top_k, self.moe_in, self.n_routed
        g = self.router
        router, up, down = _build_moe_decode_kernels(
            D, inter, K, E, x.dtype, g.norm_topk_prob, g.scaling
        )
        xf = x.reshape(M, D)
        idx, w = router(
            inputs=[xf, g.weight, g.expert_bias.astype(mx.float32)],
            output_shapes=[(M, K), (M, K)],
            output_dtypes=[mx.int32, x.dtype],
            grid=(32, M, 1),
            threadgroup=(32, 1, 1),
        )
        h = up(
            inputs=[xf, self.experts.gate_up_w, idx],
            output_shapes=[(M * K * inter,)],
            output_dtypes=[x.dtype],
            grid=(32, K * inter, M),
            threadgroup=(32, 1, 1),
        )[0]
        out = down(
            inputs=[h, self.experts.down_w, w, idx],
            output_shapes=[(M, D)],
            output_dtypes=[x.dtype],
            grid=(32, D, M),
            threadgroup=(32, 1, 1),
        )[0]
        return out.reshape(B, T, D)

    def __call__(self, x: mx.array, step_idx: Optional[int] = None) -> mx.array:
        B, T, D = x.shape
        G = B * T * self.top_k
        # collect_stats 开启（训练侧负载统计）时禁走 kernel 路径：
        # kernel 不可微且不含统计逻辑，此时回退可微的稠密/稀疏路径。
        # CycleRouter（cycle_emb 非空且本次调用带 cycle 编号）同样禁走：
        # 融合 kernel 不含 cycle_emb 项，否则 decode 与训练选择不一致。
        use_cycle = self.router.cycle_emb is not None and step_idx is not None
        if (
            G <= self._KERNEL_MAX_PAIRS
            and not self.router.collect_stats
            and not use_cycle
        ):
            cls = type(self)
            if cls._KERNEL_DISABLED or x.dtype not in _MOE_METAL_TYPE:
                idx, w = self.router(x, step_idx=step_idx)
                out = self._dense_forward(x, idx, w)
            else:
                kk = (D, self.moe_in, self.top_k, self.n_routed, x.dtype)
                try:
                    out = self._kernel_forward(x)
                    if kk not in cls._KERNEL_VERIFIED:
                        mx.eval(out)  # 触发 JIT 编译，编译失败走 except 回退
                        cls._KERNEL_VERIFIED.add(kk)
                except Exception:
                    cls._KERNEL_DISABLED = True
                    idx, w = self.router(x, step_idx=step_idx)
                    out = self._dense_forward(x, idx, w)
        else:
            idx, w = self.router(x, step_idx=step_idx)  # (B,T,K) int32 / (B,T,K)
            if G <= self._DENSE_MAX_PAIRS:
                out = self._dense_forward(x, idx, w)
            else:
                out = self._sparse_forward(x, idx, w)
        if self.shared is not None:
            out = out + self.shared(x)
        return out


def _rms_unit(x: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)


def _shift_right_tokens(x: mx.array, offset: int) -> mx.array:
    """沿序列维右移（左填 0），用于 n-gram 窗口与因果卷积。"""
    if offset == 0:
        return x
    pad = mx.zeros_like(x[:, :offset, ...])
    return mx.concatenate([pad, x[:, :-offset, ...]], axis=1)


_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193
_ENGRAM_CONV_TAPS = 4


def engram_indices_mx(
    tokens: mx.array,
    orders: tuple,
    heads: int,
    slots: int,
) -> mx.array:
    """Needle 同款哈希 n-gram 索引：uint32 xor/multiply 混合，逐 order/head
    独立 seed，得到 (B, T, len(orders)*heads) 的表下标。"""
    u = tokens.astype(mx.uint32)
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = int((_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF)
            acc = mx.full_like(u, mx.array(seed, dtype=mx.uint32))
            for j in range(order):
                acc = mx.bitwise_xor(acc, _shift_right_tokens(u, j)) * mx.array(
                    _ENGRAM_PRIME, dtype=mx.uint32
                )
            acc = mx.bitwise_xor(
                acc, mx.right_shift(acc, mx.array(15, dtype=mx.uint32))
            )
            idx.append((acc % mx.array(slots, dtype=mx.uint32)).astype(mx.int32))
    return mx.stack(idx, axis=-1)


class Engram(nn.Module):
    """Cactus Needle 的可学习 n-gram 键值记忆（注入残差流用）。

    表：num_tables x slots x sub_dim；每个位置按因果 n-gram 哈希查表，
    经 key/value 投影与 4-tap 扩张卷积得到 (ek, ev)。注入方：
        α = σ(rms(x)·rms(ek)/√d)
        x = x + site_flag · α · ev
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        orders = tuple(config.engram_orders)
        heads = config.engram_heads or max(
            1, config.hidden_size // (len(orders) * config.engram_sub_dim)
        )
        self.orders = orders
        self.heads = heads
        self.slots = int(config.engram_slots)
        self.sub_dim = int(config.engram_sub_dim)
        self.num_tables = len(orders) * heads
        self.dilation = max(orders)
        std = getattr(config, "initializer_range", 0.02)
        self.table = mx.random.normal(
            (self.num_tables, self.slots, self.sub_dim), scale=std
        )
        self.key_proj = nn.Linear(
            self.num_tables * self.sub_dim, config.hidden_size, bias=False
        )
        # value_proj 零初始化：冷启动时 α≈σ(0)=0.5，随机 ev 会向残差流注入
        # 噪声（r064/r065 两个尺度一致小幅负收益）；零初始化使 engram 从严格
        # 恒等出发，与 ΔW V=0 / res_gate 的约定一致。
        self.value_proj = nn.Linear(
            self.num_tables * self.sub_dim, config.hidden_size, bias=False
        )
        self.value_proj.weight = mx.zeros_like(self.value_proj.weight)
        # 因果卷积：identity 初始化（taps[0]=1，其余 0）
        taps = mx.zeros((_ENGRAM_CONV_TAPS, config.hidden_size))
        taps = taps.at[0].add(1.0)  # MLX ArrayAt 只支持 add/subtract 等
        self.taps = taps

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array],
        segment_ids: Optional[mx.array],
    ) -> tuple[mx.array, mx.array]:
        idx = engram_indices_mx(input_ids, self.orders, self.heads, self.slots)
        B, T, K = idx.shape
        fetched = []
        for k in range(K):
            fetched.append(mx.take(self.table[k], idx[:, :, k], axis=0))
        e = mx.stack(fetched, axis=2)  # (B,T,K,D)
        e = e.reshape(B, T, K * self.sub_dim)

        # 因果 n-gram 有效掩码：窗口内所有 token 有效且同文档。
        base_ok = (input_ids != 0).astype(mx.float32)
        if attention_mask is not None:
            am = attention_mask
            if am.shape[1] > T:
                # 缓存解码时 attention_mask 是完整序列长度，而 input_ids
                # 是尾部 n-gram 窗口，二者尾部对齐。
                am = am[:, -T:]
            elif am.shape[1] < T:
                # 投机验证/短 prompt 解码：attention_mask 只覆盖到 prompt
                # 末尾，窗口多出的新生成 token 均为有效位置（generate 的
                # 投机路径只接受全 1 mask），左侧补 1。
                am = mx.concatenate(
                    [mx.ones((am.shape[0], T - am.shape[1]), dtype=am.dtype), am],
                    axis=1,
                )
            base_ok = base_ok * am.astype(mx.bool_).astype(mx.float32)
        # 每个 order 独立计算自己的因果有效掩码（窗口内 token 全部有效且
        # 同文档），再按 engram_indices_mx 的布局（order-major, head-minor）
        # 拼成 (B,T,K)。旧实现先对所有 order 做交集、再广播到全部 K 个
        # 切片：开头不足 max(order) 个 token 时，会把本已有效的低阶
        # n-gram 也一并清零。
        order_ok_masks = []
        for order in self.orders:
            ok = mx.ones((B, T), dtype=mx.float32)
            for j in range(order):
                if segment_ids is not None:
                    seg_prev = _shift_right_tokens(segment_ids, j)
                    same_doc = segment_ids == seg_prev
                else:
                    same_doc = mx.ones_like(input_ids).astype(mx.bool_)
                ok = ok * _shift_right_tokens(base_ok, j) * same_doc.astype(mx.float32)
            ok = ok * base_ok
            order_ok_masks.append(
                mx.repeat(ok[:, :, None], self.heads, axis=-1)  # (B,T,H)
            )
        ngram_ok = mx.concatenate(order_ok_masks, axis=-1)  # (B,T,O*H)
        ngram_ok = mx.repeat(ngram_ok[..., None], self.sub_dim, axis=-1)
        ngram_ok = ngram_ok.reshape(B, T, K * self.sub_dim)
        # 降回 e 的 dtype 再乘：f32 掩码会把 key/value 投影与卷积整条链
        # 抬成 f32（带宽翻倍）
        e = e * ngram_ok.astype(e.dtype)

        ek = self.key_proj(e)
        ev = self.value_proj(e)
        # 4-tap 扩张卷积，全部沿因果方向。
        ev_conv = ev * self.taps[0].astype(ev.dtype)
        for j in range(1, _ENGRAM_CONV_TAPS):
            shift = j * self.dilation
            ev_conv = ev_conv + _shift_right_tokens(ev, shift) * self.taps[j].astype(
                ev.dtype
            )
        return ek, ev_conv


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # DeepSeekMoE：n_routed_experts>0 且 layer_id >= n_dense_layers 时
        # FFN 替换为共享+路由专家（V3/V4 前若干层保持 dense）。
        self.is_moe = getattr(
            config, "n_routed_experts", 0
        ) > 0 and layer_id >= getattr(config, "n_dense_layers", 1)
        self.mlp = MoEFeedForward(config) if self.is_moe else FeedForward(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        past_key_value: Optional[tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        layer_idx: Optional[int] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        value_residual: Optional[list] = None,
        step_idx: Optional[int] = None,
        engram_ev: Optional[tuple[mx.array, mx.array]] = None,
    ):
        if engram_ev is not None:
            ek, ev = engram_ev
            # 对齐窗口与当前输入：训练/prefill 时等长（全位置注入）；
            # 缓存解码时窗口宽于输入（标准 decode T=1 / 投机验证 T=D），
            # 窗口尾部 T 个位置即当前 token 各自的 n-gram 记忆（早期
            # 位置窗口不足时由 Engram 内的因果有效性掩码置零）。
            if ek.shape[1] > hidden_states.shape[1]:
                ek = ek[:, -hidden_states.shape[1] :, :]
                ev = ev[:, -hidden_states.shape[1] :, :]
            # 余弦相似度门控（Needle：α = σ(rms(x)·rms(ek)/√d)）
            alpha = mx.sigmoid(
                mx.sum(_rms_unit(hidden_states) * _rms_unit(ek), axis=-1)
                / math.sqrt(self.self_attn.qk_dim)
            )
            # alpha 是 f32（_rms_unit 内部升精度）；乘加前降回残差 dtype，
            # 否则残差流从本层起被抬成 f32，后续所有 GEMM/逐元素算子
            # 带宽翻倍（实测整步慢 ~42ms/18%）
            alpha = alpha.astype(hidden_states.dtype)
            hidden_states = hidden_states + alpha[..., None] * ev.astype(
                hidden_states.dtype
            )
        residual = hidden_states
        attn_in = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            attn_in,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            value_residual=value_residual,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        mlp_in = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_in, step_idx=step_idx)
        hidden_states = residual + mlp_output
        return hidden_states, present_key_value


class VibyStack(nn.Module):
    """HRM 模式的一个 transformer stack：P 层 + 尾部 RMSNorm。

    对应 HF HrmTextStack。L/H 两个 stack 结构完全相同、参数独立。
    """

    def __init__(self, config: VibyConfig, n_layers: int):
        super().__init__()
        self.layers = [VibyBlock(layer, config) for layer in range(n_layers)]
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        value_residual: Optional[list] = None,
        step_idx: Optional[int] = None,
        cache_offset: int = 0,
        engram_evs: Optional[dict] = None,
    ) -> tuple[mx.array, list]:
        presents = []
        flat_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            pv = (
                past_key_values[flat_idx + cache_offset]
                if past_key_values is not None
                else None
            )
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=pv,
                use_cache=use_cache,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                value_residual=value_residual,
                step_idx=step_idx,
                engram_ev=engram_evs.get(layer_idx) if engram_evs else None,
            )
            presents.append(present)
            flat_idx += 1
        hidden_states = self.final_norm(hidden_states)
        return hidden_states, presents


class MTPModule(nn.Module):
    """DeepSeek-V3 style Multi-Token Prediction module.

    At depth k, predicts token t+k+1 from the previous depth's hidden state
    h_t and the embedding of token t+k:
        h' = proj([RMSNorm(h_t); RMSNorm(Emb(t+k))])
        h_k = TransformerBlock(h')
    The output is scored with the shared lm_head of the main model.
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.norm_h = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_e = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # layer_id 取 num_hidden_layers：MoE 开启时 MTP 块同样是 MoE 层
        # （V3/V4 的 MTP 模块与主干同构），且永远落在 dense 前缀之外。
        self.block = VibyBlock(config.num_hidden_layers, config)
        # HRM CycleRouter：MTP 块的 router 复用最后一个 cycle 槽位
        # （其 cycle_emb 与主循环的 gate 相互独立，槽位语义私有）；
        # 不传 step_idx 会让 cycle_emb 整参数无梯度通路（value_and_grad
        # 会因 None 叶报错）。
        n_cyc = int(getattr(config, "hrm_H_cycles", 0) or 0) * (
            int(getattr(config, "hrm_L_cycles", 3) or 0) + 1
        )
        self.cycle_slot = (
            n_cyc - 1 if getattr(config, "hrm_cycle_router", 0) and n_cyc > 0 else None
        )

    def __call__(
        self,
        h_prev: mx.array,
        token_emb: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple[mx.array, Optional[tuple]]:
        x = mx.concatenate([self.norm_h(h_prev), self.norm_e(token_emb)], axis=-1)
        x = self.proj(x)
        out, present = self.block(
            x,
            position_embeddings,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            step_idx=self.cycle_slot,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        return out, present


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.use_value_res = getattr(config, "use_value_res", False)
        self.hrm_H = int(getattr(config, "hrm_H_cycles", 0) or 0)
        self.hrm_L = int(getattr(config, "hrm_L_cycles", 3) or 0)
        self.hrm_emb_scale = float(getattr(config, "hrm_emb_scale", 1.0) or 1.0)
        self.emb_scale = float(getattr(config, "emb_scale", 1.0) or 1.0)
        self.l_module = None
        self.h_module = None
        if self.hrm_H > 0:
            # num_hidden_layers 在 HRM 模式下表示每个 stack 的真实层数 P；
            # 一个 token 的层求值次数 = H*(L+1)*P，训练/推理前向完全一致。
            self.l_module = VibyStack(config, config.num_hidden_layers)
            self.h_module = VibyStack(config, config.num_hidden_layers)
            self.layers = None
            raw_bp = list(getattr(config, "hrm_bp_cycles", None) or [self.hrm_L])
            self.hrm_bp_padded = [1] * max(0, self.hrm_H - len(raw_bp)) + raw_bp
        else:
            self.layers = [
                VibyBlock(layer, config) for layer in range(config.num_hidden_layers)
            ]
            self.hrm_bp_padded = None
            # absorbed+v_res 兼容接线：layer ℓ≥1 的 λ·v_0 项需要 layer0 的
            # V 上投影权重做跨层折叠（W_UV0ℓ = W_oℓ·W_v0）。存的是 layer0
            # 注意力模块的引用而非权重切片——load_weights 会整体替换权重
            # 数组，折叠在首次 decode 时才物化，必须届时取当前权重。
            # 引用经 object.__setattr__ 隐藏，不进 parameters/不重复计数。
            if self.use_value_res and self.layers:
                attn0 = self.layers[0].self_attn
                if attn0.absorbed_decode:
                    for layer in self.layers[1:]:
                        if layer.self_attn.absorbed_decode:
                            object.__setattr__(layer.self_attn, "_abs_attn0", attn0)
        # Engram n-gram 记忆位点。普通栈按 layer_idx 注入；HRM 模式注入
        # L-module（快状态栈）的同名层，每个循环调用都会重读一次表。
        engram_layers = tuple(
            int(x) for x in (getattr(config, "engram_layers", ()) or ())
        )
        self.engram_layers = engram_layers
        if engram_layers:
            if not tuple(getattr(config, "engram_orders", ()) or ()):
                raise ValueError("engram_layers 非空时 engram_orders 不能为空")
            if int(getattr(config, "engram_slots", 0)) <= 0:
                raise ValueError("engram_slots 必须大于 0")
            if int(getattr(config, "engram_sub_dim", 0)) <= 0:
                raise ValueError("engram_sub_dim 必须大于 0")
        # 越界位点会建表但永远注入不到（按 layer_idx 匹配），直接拒绝，
        # 防止静默的 no-op 参数（test_engram 曾因此漏检第二个位点）。
        bad = [s for s in self.engram_layers if not 0 <= s < config.num_hidden_layers]
        if bad:
            raise ValueError(
                f"engram_layers {bad} 超出栈范围 [0, {config.num_hidden_layers})"
            )
        self.engrams = (
            [Engram(config) for _ in self.engram_layers] if self.engram_layers else []
        )
        # CycleFiLM（HRM×MoE，research/HRM_MOE.md）：每次 stack 调用前对
        # 注入状态做 per-cycle scale/shift，给每个 cycle 显式时间身份。
        # 零初始化 ⇒ 初始严格等价。编号与 CycleRouter 共享：
        # c = h_idx*(L+1)+l_idx（L 调用）/ h_idx*(L+1)+L（H 调用）。
        self.hrm_n_cycles = self.hrm_H * (self.hrm_L + 1)
        if self.hrm_H > 0 and getattr(config, "hrm_cycle_film", 0):
            self.hrm_film_scale = mx.zeros((self.hrm_n_cycles, config.hidden_size))
            self.hrm_film_shift = mx.zeros((self.hrm_n_cycles, config.hidden_size))
        else:
            self.hrm_film_scale = None
            self.hrm_film_shift = None
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope_scaling = config.rope_scaling
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault(
                "original_max_position_embeddings",
                config.original_max_position_embeddings,
            )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.qk_rope_head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        self.freeze(recurse=False, keys=["freqs_cos", "freqs_sin"])

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        engram_input_ids: Optional[mx.array] = None,
        engram_decode: bool = False,
        **kwargs,
    ) -> tuple[mx.array, list]:
        batch_size, seq_length = input_ids.shape
        hrm_mode = self.hrm_H > 0
        if hrm_mode:
            n_layers_per_stack = self.config.num_hidden_layers
            n_effective_layers = self.hrm_H * (self.hrm_L + 1) * n_layers_per_stack
        else:
            n_effective_layers = len(self.layers)
        if past_key_values is None:
            past_key_values = [None] * n_effective_layers
        elif len(past_key_values) != n_effective_layers:
            raise ValueError(
                f"past_key_values 层数 {len(past_key_values)} 与有效层数 "
                f"{n_effective_layers} 不一致"
            )
        first_cache = past_key_values[0]
        if first_cache is None:
            start_pos = 0
        else:
            start_pos = first_cache[0].shape[1]
        if start_pos + seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"输入长度 {start_pos + seq_length} 超过模型最大上下文长度 "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if hrm_mode and self.hrm_emb_scale != 1.0:
            hidden_states = hidden_states * self.hrm_emb_scale
        elif not hrm_mode and self.emb_scale != 1.0:
            hidden_states = hidden_states * self.emb_scale
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_length]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_length]
        if freqs_cos.dtype != hidden_states.dtype:
            freqs_cos = freqs_cos.astype(hidden_states.dtype)
            freqs_sin = freqs_sin.astype(hidden_states.dtype)
        position_embeddings = (freqs_cos, freqs_sin)

        # 每微批只 sync/构造一次 causal+pad 融合 mask，所有层共享：
        # 有 padding 时普通 Attention 也能走 flash，而不是逐层退化到 O(T²)。
        # mask_has_pad 可由调用方在 eager 侧预先算好传入（mx.compile 图内
        # 不允许 .item() host sync）；未传入时按原逻辑现场判断。
        mask_is_full = True
        causal_bias = None
        if mask_has_pad is None:
            mask_has_pad = attention_mask is not None and bool(
                mx.any(attention_mask != 1).item()
            )
        if attention_mask is not None and mask_has_pad:
            mask_is_full = False
            am = attention_mask.astype(mx.bool_)
            pad_bias = mx.where(am, 0.0, -1e9)  # (B, T)
            causal = mx.triu(mx.full((seq_length, seq_length), -1e9), k=1)
            causal_bias = (
                causal[None, None, :, :] + pad_bias[:, None, None, :]
            ).astype(hidden_states.dtype)

        # 文档边界掩码（doc_mask 打包训练）：注意力限制在同文档内因果可见，
        # 消除跨文档泄漏，与逐篇 PPL 评估口径对齐。仅在完整前向（训练）传入。
        if segment_ids is not None and first_cache is None and seq_length > 1:
            same_doc = segment_ids[:, :, None] == segment_ids[:, None, :]
            causal_tril = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
            allowed = same_doc & causal_tril[None, :, :]
            seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(
                hidden_states.dtype
            )
            causal_bias = seg_bias if causal_bias is None else causal_bias + seg_bias
            mask_is_full = False

        # value residual 的跨层共享载体：第一层写入 v_0，后续层读取混合
        value_residual = [] if self.use_value_res else None

        # Engram 注入位点：完整前向（训练 / 无 cache 的 PPL 评估）默认启用；
        # 缓存解码（generate）仅在 engram_decode=True 时启用：prefill 用完整
        # input_ids 全位置注入，后续 decode 步用调用方传入的 n-gram 窗口
        # （engram_input_ids，长度 ≥ max(orders) + 卷积感受野）只注入窗口
        # 尾部与当前输入等长的位置（VibyBlock 内按形状收窄）。不传窗口时
        # 保持原语义（跳过），未接线的调用方行为不变。
        engram_evs = {}
        if self.engrams and (not use_cache or engram_decode):
            for site_layer, eng in zip(self.engram_layers, self.engrams):
                src = (
                    input_ids
                    if (not use_cache or first_cache is None)
                    else engram_input_ids
                )
                if src is not None:
                    # 缓存解码时 src 是 n-gram 窗口，长度通常小于完整
                    # segment_ids；直接把完整 segment_ids 传给 Engram 会
                    # 广播失败（segment_ids 的文档掩码只在完整前向生效，
                    # 缓存解码的 attention 侧本来也不使用它）。
                    seg_for_engram = (
                        segment_ids
                        if segment_ids is not None
                        and segment_ids.shape[1] == src.shape[1]
                        else None
                    )
                    engram_evs[site_layer] = eng(src, attention_mask, seg_for_engram)

        presents = []
        if hrm_mode:
            # HRM 双状态层次循环。z_H = 慢/高状态，z_L = 快/低状态；
            # 每个 token 的训练前向与推理前向完全一致，没有推理侧额外展开。
            z_h = hidden_states
            z_l = mx.zeros_like(z_h)

            def _cycle_in(x, c):
                # CycleFiLM：per-cycle scale/shift（零初始化时恒等）
                if self.hrm_film_scale is None:
                    return x
                s = self.hrm_film_scale[c].astype(x.dtype)
                b = self.hrm_film_shift[c].astype(x.dtype)
                return x * (1.0 + s) + b

            for h_idx in range(self.hrm_H):
                num_grad = int(
                    self.hrm_bp_padded[h_idx] if h_idx < len(self.hrm_bp_padded) else 1
                )
                grad_threshold = self.hrm_L - num_grad
                for l_idx in range(self.hrm_L):
                    cyc = h_idx * (self.hrm_L + 1) + l_idx
                    cache_offset = cyc * n_layers_per_stack
                    z_l, p = self.l_module(
                        _cycle_in(z_l + z_h, cyc),
                        position_embeddings,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        attention_mask=attention_mask,
                        causal_bias=causal_bias,
                        mask_is_full=mask_is_full,
                        value_residual=value_residual,
                        step_idx=cyc,
                        cache_offset=cache_offset,
                        engram_evs=engram_evs,
                    )
                    presents.extend(p)
                    # L_bp_cycles 梯度路由：早期 L cycle 只做前向，
                    # 仅尾部 num_grad 个 cycle 回传梯度（与 HF HrmText 一致）。
                    # 早期 cycle 的 CycleFiLM/CycleRouter 行与 L 权重同享
                    # 该语义：输出被 detach ⇒ 这些行不收梯度。
                    if l_idx < grad_threshold:
                        z_l = mx.stop_gradient(z_l)

                cyc = h_idx * (self.hrm_L + 1) + self.hrm_L
                cache_offset = cyc * n_layers_per_stack
                z_h, p = self.h_module(
                    _cycle_in(z_h + z_l, cyc),
                    position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    causal_bias=causal_bias,
                    mask_is_full=mask_is_full,
                    value_residual=value_residual,
                    step_idx=cyc,
                    cache_offset=cache_offset,
                )
                presents.extend(p)

            hidden_states = z_h
            return hidden_states, presents

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values[layer_idx],
                use_cache=use_cache,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                value_residual=value_residual,
                engram_ev=engram_evs.get(layer_idx),
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


def _transform_logits_mx(
    logits: mx.array,
    seen_ids: Optional[mx.array],
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
) -> mx.array:
    """采样前的 logits 变换（repetition penalty / top_k / top_p），全程 MLX。

    logits: (..., V) 或 (V,)；seen_ids: (N,) 或 (B, N) 已见 token。
    返回与旧 numpy 实现等价的变换后 logits，不离开 GPU。
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None]
        if seen_ids is not None:
            seen_ids = seen_ids[None]

    logits = logits / temperature

    if repetition_penalty != 1.0 and seen_ids is not None and seen_ids.size > 0:
        rows = []
        for b in range(logits.shape[0]):
            row = logits[b]
            seen = seen_ids[b] if seen_ids.ndim == 2 else seen_ids
            # 每个唯一 token id 只惩罚一次：重复索引经 scatter-max 归并为 0/1 掩码，
            # 无需 sort、无需 CPU 同步（.at 对重复索引是累加语义，不能直接 gather-scatter）。
            mask = (
                mx.zeros(row.shape[-1], dtype=mx.int32)
                .at[seen.astype(mx.int32)]
                .maximum(1)
            )
            scale = mx.where(row > 0, 1.0 / repetition_penalty, repetition_penalty)
            row = mx.where(mask.astype(mx.bool_), row * scale, row)
            rows.append(row)
        logits = mx.stack(rows)

    if do_sample and top_k > 0:
        kth = mx.partition(logits, -top_k, axis=-1)[..., -top_k : -top_k + 1]
        logits = mx.where(logits < kth, -mx.inf, logits)
    if do_sample and top_p < 1.0:
        order = mx.argsort(-logits, axis=-1).astype(mx.int32)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        exp_shifted = mx.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        sorted_probs = exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)
        cumprobs = mx.cumsum(sorted_probs, axis=-1)
        remove = cumprobs > top_p
        # 与 numpy 一致：第一个越过阈值的 token 保留，其后移除
        remove = mx.concatenate(
            [mx.zeros(remove.shape[:-1] + (1,), dtype=mx.bool_), remove[..., :-1]],
            axis=-1,
        )
        masked = mx.where(remove, -mx.inf, sorted_logits)
        logits = mx.put_along_axis(logits, order, masked, axis=-1)

    return logits[0] if squeeze else logits


def _probs_mx(logits: mx.array) -> mx.array:
    """softmax，返回概率分布（与 logits 同形状）。"""
    exp_shifted = mx.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _sample_from_logits_mx(logits: mx.array, do_sample: bool) -> mx.array:
    """从（已变换的）logits 采样一个 token 下标，全程 MLX。

    do_sample=True 时按概率采样（GPU 随机数），否则取 argmax。
    """
    if do_sample:
        return mx.random.categorical(logits, axis=-1)
    return mx.argmax(logits, axis=-1)


class VibyForCausalLM(nn.Module):
    def __init__(self, config: Optional[VibyConfig] = None):
        super().__init__()
        config = config or VibyConfig()
        self.config = config
        self.model = VibyModel(config)
        if config.tie_word_embeddings:
            # Tied lm_head: computed with the embedding weight, no extra param.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_modules = (
            [MTPModule(config) for _ in range(config.mtp_depth)]
            if config.mtp_depth > 0
            else []
        )

    def _lm_logits(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states @ self.model.embed_tokens.weight.T

    def _mtp_loss(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        labels: mx.array,
        loss_mask: Optional[mx.array],
        attention_mask: Optional[mx.array],
        mask_has_pad: Optional[bool] = None,
    ) -> mx.array:
        """DeepSeek-V3 MTP loss: average CE over the chained MTP depths.

        Depth k consumes h_t^(k-1) and Emb(t+k) to predict labels[:, k:]
        (i.e. token t+k+1, since labels are next-token shifted).
        """
        seq_len = input_ids.shape[1]
        token_emb = self.model.embed_tokens(input_ids)
        h_prev = hidden_states
        terms = []
        for k, module in enumerate(self.mtp_modules, start=1):
            if seq_len <= k:
                break
            sub = seq_len - k
            position_embeddings = (
                self.model.freqs_cos[:sub].astype(hidden_states.dtype),
                self.model.freqs_sin[:sub].astype(hidden_states.dtype),
            )
            am = attention_mask[:, :sub] if attention_mask is not None else None
            # mask_has_pad 由调用方在 eager 侧算好传入（compile 图内不允许 .item()）
            if mask_has_pad is None:
                mask_is_full = am is None or bool(mx.all(am == 1).item())
            else:
                mask_is_full = am is None or not mask_has_pad
            causal_bias = None
            if am is not None and not mask_is_full:
                pad_bias = mx.where(am.astype(mx.bool_), 0.0, -1e9)
                causal = mx.triu(mx.full((sub, sub), -1e9), k=1)
                causal_bias = (
                    causal[None, None, :, :] + pad_bias[:, None, None, :]
                ).astype(hidden_states.dtype)
            h_k, _ = module(
                h_prev[:, :sub, :],
                token_emb[:, k:, :],
                position_embeddings,
                attention_mask=am,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
            )
            logits_k = self._lm_logits(h_k)
            terms.append(
                cross_entropy(
                    logits_k,
                    labels[:, k:],
                    mask=loss_mask[:, k:] if loss_mask is not None else None,
                )
            )
            h_prev = h_k
        return sum(terms) / len(terms) if terms else mx.array(0.0)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        labels: Optional[mx.array] = None,
        loss_mask: Optional[mx.array] = None,
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        engram_input_ids: Optional[mx.array] = None,
        engram_decode: bool = False,
        **kwargs,
    ) -> CausalLMOutput:
        # MoE 负载统计按"次前向"重置：HRM 模式下同一 gate 每次前向被
        # H*(L+1) 次调用并跨循环累加（见 MoEGate.__call__）；非 HRM 模式
        # 每 gate 每次前向只调一次，重置后累加 ≡ 原覆盖语义。
        for g in self.moe_gates():
            if g.collect_stats:
                g.last_load = None
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            mask_has_pad=mask_has_pad,
            segment_ids=segment_ids,
            engram_input_ids=engram_input_ids,
            engram_decode=engram_decode,
            **kwargs,
        )
        hidden_states_full = hidden_states
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self._lm_logits(hidden_states)

        loss = None
        mtp_loss = None
        if labels is not None:
            loss = cross_entropy(
                logits,
                labels,
                mask=loss_mask,
            )
            if self.config.mtp_depth > 0 and self.mtp_modules:
                mtp_loss = self._mtp_loss(
                    hidden_states_full,
                    input_ids,
                    labels,
                    loss_mask,
                    attention_mask,
                    mask_has_pad=mask_has_pad,
                )
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            mtp_loss=mtp_loss,
        )

    def generate(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = 50,
        eos_token_id: Optional[int] = 2,
        streamer=None,
        use_cache: bool = True,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        return_kv: bool = False,
        use_mtp_speculative: bool = False,
        num_speculative_tokens: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 必须是 2 维 (batch, seq)，实际为 {input_ids.ndim} 维"
            )
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences 必须 >= 1")
        if top_k < 0:
            raise ValueError("top_k 不能为负数")
        if top_k > self.config.vocab_size:
            raise ValueError(
                f"top_k ({top_k}) 不能超过 vocab_size ({self.config.vocab_size})"
            )

        # 限制生成长度，避免超出 RoPE 预计算范围后出现晦涩的广播错误。
        prompt_len = input_ids.shape[1]
        if prompt_len > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt 长度 {prompt_len} 超过模型最大上下文 "
                f"{self.config.max_position_embeddings}"
            )
        max_new_tokens = min(
            max_new_tokens, self.config.max_position_embeddings - prompt_len
        )

        # engram 缓存解码：模型带 engram 时生成全程按训练口径注入——
        # prefill 全位置，标准 decode / 投机验证步携带 n-gram 窗口。
        # 窗口长度 = max(orders) + (taps-1)*dilation：既要覆盖当前 token 的
        # n-gram 上下文，也要覆盖 4-tap 扩张卷积在 t-3d, t-6d, t-9d 处读取的
        # 历史 ev（其 n-gram 上下文同样要落在窗口内），否则训练中卷积学会
        # 非恒等 taps 后，生成侧会与训练口径不一致。
        engram_win_len = 0
        if self.model.engrams:
            orders = tuple(self.config.engram_orders or ())
            max_order = max(orders) if orders else 0
            if max_order:
                taps = self.model.engrams[0].taps.shape[0]
                engram_win_len = max_order + (taps - 1) * max_order
        engram_decode_enabled = engram_win_len > 0

        # 投机解码只支持全 1 的 attention_mask（无 padding）。
        mask_ok = attention_mask is None or bool(mx.all(attention_mask == 1).item())
        if use_mtp_speculative:
            can_speculate = (
                len(self.mtp_modules) > 0
                and input_ids.shape[0] == 1
                and num_return_sequences == 1
                and use_cache
                and mask_ok
            )
            if can_speculate:
                generated = self._generate_speculative(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=eos_token_id,
                    streamer=streamer,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    num_speculative_tokens=num_speculative_tokens,
                    engram_win_len=engram_win_len,
                    engram_decode_enabled=engram_decode_enabled,
                )
                if return_kv:
                    return {
                        "generated_ids": generated,
                        "past_kv": getattr(self, "_last_spec_past", None),
                    }
                return generated
            import warnings

            warnings.warn(
                "use_mtp_speculative requires mtp_depth > 0, batch size 1, "
                "use_cache=True 且 attention_mask 全为 1；已回退到标准生成。"
            )

        if num_return_sequences > 1:
            input_ids = mx.concatenate([input_ids] * num_return_sequences, axis=0)
            if attention_mask is not None:
                attention_mask = mx.concatenate(
                    [attention_mask] * num_return_sequences, axis=0
                )
        past_key_values = None
        batch = input_ids.shape[0]
        finished = mx.zeros(batch, dtype=mx.bool_)

        if streamer:
            streamer.put(input_ids)

        if max_new_tokens <= 0:
            if streamer:
                streamer.end()
            if return_kv:
                return {"generated_ids": input_ids, "past_kv": None}
            return input_ids

        for _ in range(max_new_tokens):
            if past_key_values:
                past_len = past_key_values[0][0].shape[1]
            else:
                past_len = 0
            current_input_ids = (
                input_ids[:, past_len:]
                if past_len < input_ids.shape[1]
                else input_ids[:, -1:]
            )
            engram_window = None
            if engram_decode_enabled and past_len > 0:
                engram_window = input_ids[:, -engram_win_len:]
            outputs = self(
                current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                engram_input_ids=engram_window,
                engram_decode=engram_decode_enabled,
            )
            logits_mx = outputs.logits[:, -1, :]
            if attention_mask is not None:
                attention_mask = mx.concatenate(
                    [attention_mask, mx.ones((batch, 1), dtype=attention_mask.dtype)],
                    axis=-1,
                )

            seen_ids = input_ids if repetition_penalty != 1.0 else None
            transformed = _transform_logits_mx(
                logits_mx,
                seen_ids,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            )
            next_token = _sample_from_logits_mx(transformed, do_sample)

            if eos_token_id is not None:
                next_token = mx.where(
                    finished,
                    mx.full_like(next_token, eos_token_id),
                    next_token,
                )

            input_ids = mx.concatenate(
                [input_ids, next_token[:, None].astype(input_ids.dtype)], axis=-1
            )
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token[:, None])

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if bool(mx.all(finished).item()):
                    break

        if streamer:
            streamer.end()
        if (
            return_kv
            and use_cache
            and past_key_values is not None
            and past_key_values[0][0].shape[1] < input_ids.shape[1]
        ):
            # 标准循环每步先 forward 当前 token、再采样并 append，因此循环
            # 结束时 cache 还差最后一个已生成 token。这里补一次纯前向，让
            # return_kv 的 cache 与 generated_ids 对齐（与投机路径一致）。
            missing = input_ids[:, past_key_values[0][0].shape[1] :]
            engram_window = None
            if engram_decode_enabled:
                engram_window = input_ids[:, -(engram_win_len - 1 + missing.shape[1]) :]
            outputs = self(
                missing,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                engram_input_ids=engram_window,
                engram_decode=engram_decode_enabled,
            )
            past_key_values = outputs.past_key_values
        if return_kv:
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids

    def _mtp_draft(
        self,
        h_last: mx.array,
        first_token: int,
        pos_idx: int,
        draft_len: int,
        seen_ids: mx.array,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        mtp_past: Optional[tuple] = None,
    ) -> tuple[list[int], list[mx.array], Optional[tuple]]:
        """Draft tokens with the chained MTP modules.

        h_last: (1, 1, hidden) main-model hidden state at position pos_idx.
        first_token: the bonus token at position pos_idx + 1.
        draft_len: 每轮草稿的 token 数。超过模块数时循环复用 MTP 模块
        （与 vLLM 的 MTP self-speculation 一致，业界 depth=1 但草稿 3~7 个）。
        mtp_past: 首个 MTP 模块的注意力 KV cache（训练口径的上下文，
        由 _generate_speculative 在 prefill 时教师强制构建、每轮验证后
        回滚追平）；循环复用产生的中间条目随返回一并带出，由调用方截断。
        Returns (draft_token_ids, draft_probs, mtp_past)，len(drafts) ==
        draft_len，全程 MLX，logits 不离开 GPU。
        """
        token = mx.array([[first_token]])
        h = h_last
        drafts: list[int] = []
        draft_probs: list[mx.array] = []
        n_modules = len(self.mtp_modules)
        # 自复用链中每个草稿都等价于训练时“同一输出位置、更深一层”的 MTP：
        # 始终消费同一 h 流位置 (pos_idx) 上的表示，只把 token 逐层后移，
        # 因此 RoPE 全程固定为 freqs[pos_idx]，与训练约定一致。
        cos = self.model.freqs_cos[pos_idx : pos_idx + 1]
        sin = self.model.freqs_sin[pos_idx : pos_idx + 1]
        for i in range(draft_len):
            module_idx = i % n_modules
            module = self.mtp_modules[module_idx]
            emb = self.model.embed_tokens(token)
            # KV cache 只接首个模块（depth-1 流，唯一被训练的深度）；
            # 更深的复用步保持无 cache 旧语义（其输入本就 OOD）。
            own_cache = mtp_past is not None and module_idx == 0
            h, present = module(
                h,
                emb,
                (cos.astype(h.dtype), sin.astype(h.dtype)),
                past_key_value=mtp_past if own_cache else None,
                use_cache=True,
            )
            if own_cache:
                mtp_past = present
            logits = self._lm_logits(h)[0, 0]
            # repetition penalty 的上下文应包含 bonus token 与已生成的草稿
            seen_cur = mx.concatenate(
                [seen_ids, mx.array([first_token] + drafts, dtype=mx.int32)]
            )
            transformed = _transform_logits_mx(
                logits,
                seen_cur,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            )
            probs = _probs_mx(transformed)
            tok = int(_sample_from_logits_mx(transformed, do_sample).item())
            drafts.append(tok)
            draft_probs.append(probs)
            token = mx.array([[tok]])
        return drafts, draft_probs, mtp_past

    def _generate_speculative(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_id: Optional[int],
        streamer,
        do_sample: bool,
        repetition_penalty: float,
        num_speculative_tokens: Optional[int] = None,
        engram_win_len: int = 0,
        engram_decode_enabled: bool = False,
    ) -> mx.array:
        """MTP speculative decoding: draft with MTP modules, verify with the
        main model in parallel, accept the longest matching prefix.

        Greedy mode reproduces the main model's greedy output exactly;
        sampling mode uses standard rejection sampling against draft probs.

        注意：greedy + repetition_penalty!=1 且候选 logits 非常接近（~bf16
        噪声量级）时，批量验证与顺序解码的 KV 舍入差异可能翻转 argmax，
        导致投机路径比标准路径差一个 token。这是 bf16 精度问题，非逻辑错误；
        采样模式不受影响（拒绝采样保证分布等价）。
        """
        # 每轮草稿长度：默认等于 MTP 模块数，也可通过 num_speculative_tokens
        # 循环复用模块草稿更多 token（对齐 DeepSeek V4 等主流配置）。
        draft_len = num_speculative_tokens or len(self.mtp_modules)
        seq_len = input_ids.shape[1]
        prompt_ids = input_ids[0].astype(mx.int32)

        # Prefill the main model.
        # 与标准路径一致：prefill 全位置注入 engram（engram_input_ids=None
        # 时 VibyModel 回退用完整 input_ids）。
        out = self(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            engram_decode=engram_decode_enabled,
        )
        past = out.past_key_values
        h_last = out.hidden_states[:, -1:, :]
        logits_last = out.logits[0, -1]

        # MTP 草稿层 KV cache（对齐 vLLM/SGLang 的 NextN 推理）：训练时
        # MTP block 对整流做因果注意力，草稿若拿不到上下文会显著偏离
        # （r060 实测 argmax 命中率 45%→34%）。prefill 时用主模型 hidden
        # states + 真实 token 嵌入教师强制构建；之后每轮验证后回滚草稿
        # 追加、再按被接受位置追平。仅维护首个模块（depth-1 流）；有
        # padding 时不启用（speculative 本就限 batch=1，eval 恒无 pad）。
        mtp_past = None
        if attention_mask is None and seq_len > 1:
            h_pref = out.hidden_states[:, : seq_len - 1, :]
            e_pref = self.model.embed_tokens(input_ids[:, 1:])
            rope_pref = (
                self.model.freqs_cos[: seq_len - 1].astype(h_pref.dtype),
                self.model.freqs_sin[: seq_len - 1].astype(h_pref.dtype),
            )
            _, mtp_past = self.mtp_modules[0](h_pref, e_pref, rope_pref, use_cache=True)

        if streamer:
            streamer.put(input_ids)

        generated: list[int] = []
        stats = {"accepted": 0, "drafted": 0}
        while len(generated) < max_new_tokens:
            seen = mx.concatenate([prompt_ids, mx.array(generated, dtype=mx.int32)])
            # 1. Bonus token from the main model (always accepted).
            bonus_logits = _transform_logits_mx(
                logits_last,
                seen,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            )
            bonus = int(_sample_from_logits_mx(bonus_logits, do_sample).item())
            # 2. Draft with the MTP chain.
            # 剩余额度/上下文位置有限时收紧草稿数：bonus 必占 1 个，
            # full-accept 的额外 tail 还要再占 1 个位置，避免越过
            # max_position_embeddings（前向里有显式越界检查）。
            remaining = max_new_tokens - len(generated)
            iter_draft_len = min(draft_len, max(0, remaining - 2))
            mtp_past_len = mtp_past[0].shape[1] if mtp_past is not None else 0
            drafts, draft_probs = [], []
            if iter_draft_len > 0:
                drafts, draft_probs, mtp_past = self._mtp_draft(
                    h_last,
                    bonus,
                    seq_len - 1,
                    iter_draft_len,
                    seen,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                    mtp_past=mtp_past,
                )
            stats["drafted"] += len(drafts)
            # 3. Verify the chain in parallel with the main model.
            verify_tokens = mx.array([[bonus] + drafts], dtype=mx.int32)
            past_before_verify = past
            # engram 窗口：验证步一次送入 1+D 个新 token，每个都需要自己的
            # n-gram 前文与卷积感受野，因此携带“上下文尾部 win_len-1 + D”
            # 个 token；VibyBlock 只取窗口尾部 D 个位置注入（与验证 token
            # 一一对应）。
            engram_window = None
            if engram_decode_enabled:
                win = mx.concatenate([seen, mx.array([bonus] + drafts, dtype=mx.int32)])
                win_w = engram_win_len - 1 + len(verify_tokens[0])
                engram_window = win[-win_w:][None, :]
            vout = self(
                verify_tokens,
                attention_mask=attention_mask,
                past_key_values=past_before_verify,
                use_cache=True,
                engram_input_ids=engram_window,
                engram_decode=engram_decode_enabled,
            )
            past_full = vout.past_key_values
            vlogits = vout.logits[0]  # (1+D, V)

            # 4. Accept the longest matching prefix.
            n_acc = 0
            for i, d in enumerate(drafts):
                seen_i = mx.concatenate(
                    [seen, mx.array([bonus] + drafts[:i], dtype=mx.int32)]
                )
                p_logits = _transform_logits_mx(
                    vlogits[i],
                    seen_i,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                if do_sample:
                    p = _probs_mx(p_logits)
                    q = draft_probs[i]
                    ratio = float(p[d].item()) / max(float(q[d].item()), 1e-12)
                    if float(mx.random.uniform().item()) < min(1.0, ratio):
                        n_acc += 1
                    else:
                        break
                else:
                    if int(mx.argmax(p_logits).item()) == d:
                        n_acc += 1
                    else:
                        break
            stats["accepted"] += n_acc

            accepted_prefix = [bonus] + drafts[:n_acc]
            seen_tail = mx.concatenate(
                [seen, mx.array(accepted_prefix, dtype=mx.int32)]
            )
            if n_acc == iter_draft_len and iter_draft_len > 0:
                # All drafts accepted: sample an extra token from the tail.
                tail_logits = _transform_logits_mx(
                    vlogits[iter_draft_len],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                tail = int(_sample_from_logits_mx(tail_logits, do_sample).item())
            elif iter_draft_len == 0:
                # 上下文/剩余额度边界：只产出 bonus，不补 tail
                tail = None
            elif do_sample:
                # Reject: resample from the positive residual (p - q)+.
                p_logits = _transform_logits_mx(
                    vlogits[n_acc],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                p = _probs_mx(p_logits)
                q = draft_probs[n_acc]
                resid = mx.clip(p - q, 0.0, None)
                if float(resid.sum().item()) > 0:
                    tail = int(mx.random.categorical(mx.log(resid + 1e-12)).item())
                else:
                    tail = int(mx.argmax(p_logits).item())
            else:
                # greedy：与标准路径一致，施加变换后取 argmax
                tail_logits = _transform_logits_mx(
                    vlogits[n_acc],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    False,
                    repetition_penalty,
                )
                tail = int(mx.argmax(tail_logits).item())

            new_tokens = accepted_prefix + ([tail] if tail is not None else [])

            # 5. Stop at EOS (keep it, drop anything after).
            stop = False
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
                stop = True

            # 6. 回滚 cache 到实际接受的前缀，再前向 tail token。
            seq_len += len(new_tokens)
            generated.extend(new_tokens)
            if streamer:
                streamer.put(mx.array([new_tokens]))
            if stop:
                # EOS 截断后无需再前向 tail：直接截取已验证 cache 到最终长度。
                # 若继续走下面的 tail_win，seen_tail 中已包含 EOS 再拼一次
                # EOS，会把 engram n-gram 窗口整体前移一格，污染 return_kv。
                past = [tuple(c[:, :seq_len] for c in pkv) for pkv in past_full]
                break

            # MTP cache：回滚草稿阶段的追加，再教师强制追平本轮新确定的
            # (hidden, next-token) 对。流位置 = 追加前 cache 长度（与训练
            # 口径一致：条目 t = 主模型 h[t] + token t+1 的嵌入，RoPE 按
            # 绝对流位置切片）。
            if mtp_past is not None:
                mtp_past = tuple(c[:, :mtp_past_len] for c in mtp_past)
                n_e = len(new_tokens)
                h_in = mx.concatenate(
                    [h_last, vout.hidden_states[:, : n_e - 1, :]], axis=1
                )
                e_in = self.model.embed_tokens(mx.array([new_tokens], dtype=mx.int32))
                rope_c = (
                    self.model.freqs_cos[mtp_past_len : mtp_past_len + n_e].astype(
                        h_in.dtype
                    ),
                    self.model.freqs_sin[mtp_past_len : mtp_past_len + n_e].astype(
                        h_in.dtype
                    ),
                )
                _, mtp_past = self.mtp_modules[0](
                    h_in, e_in, rope_c, past_key_value=mtp_past, use_cache=True
                )

            keep = seq_len - 1  # old_seq_len + len(new_tokens[:-1])
            # cache 兼容 (k, v) 与 absorbed (c, k_rope, k_rinv) 两种格式
            past = [tuple(c[:, :keep] for c in pkv) for pkv in past_full]
            # tail token 的 n-gram 窗口：尾部 win_len 个 token（含 tail 自身），
            # 与标准 decode 步口径一致，保证该位置的缓存 K/V 与下一轮 bonus
            # logits 都带 engram 记忆。iter_draft_len==0 时 tail 为 None，
            # new_tokens[-1] 就是已验证的 bonus，此时窗口必须从 seen（bonus
            # 之前的上下文）截取；用 seen_tail 会重复 bonus，使 n-gram 前文
            # 少读一个真实 token。
            tail_ctx = seen if tail is None else seen_tail
            tail_win = None
            if engram_decode_enabled:
                tail_win = (
                    mx.concatenate(
                        [
                            tail_ctx,
                            mx.array([new_tokens[-1]], dtype=mx.int32),
                        ]
                    )
                )[-engram_win_len:][None, :]
            tout = self(
                mx.array([[new_tokens[-1]]]),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                engram_input_ids=tail_win,
                engram_decode=engram_decode_enabled,
            )
            past = tout.past_key_values
            h_last = tout.hidden_states[:, -1:, :]
            logits_last = tout.logits[0, -1]

        generated = generated[:max_new_tokens]
        if streamer:
            streamer.end()
        out_ids = mx.concatenate(
            [input_ids, mx.array([generated], dtype=input_ids.dtype)], axis=1
        )

        # 当达到 max_new_tokens 提前截断 generated 时，让返回的 cache 与
        # out_ids 长度严格一致。
        target_len = input_ids.shape[1] + len(generated)
        if past:
            # 时间维在 axis 1；兼容 (k, v) 与 absorbed 三元组两种格式
            past = [tuple(c[:, :target_len] for c in pkv) for pkv in past]
        object.__setattr__(self, "_last_spec_stats", stats)
        object.__setattr__(self, "_last_spec_past", past)
        return out_ids

    def moe_gates(self) -> list:
        """模型内全部 MoE router（含 MTP 块内的），遍历顺序固定。"""
        return [m for m in self.modules() if isinstance(m, MoEGate)]

    def moe_load_stats(self) -> Optional[mx.array]:
        """各 router 本步的每专家 token 计数拼接向量（无 MoE/未收集时 None）。

        仅在 router 开启 collect_stats 且 forward 执行后有效；返回的是图节点，
        可作为 compile 图的额外输出一并物化。
        """
        stats = [g.last_load for g in self.moe_gates() if g.last_load is not None]
        return mx.concatenate(stats) if stats else None

    def update_moe_biases(self, load_stats: mx.array, rate: float):
        """无辅助损失负载均衡（V3/V4 规则）：按负载统计更新路由偏置。

        b_i ← b_i − u·sign(load_i − mean_load)：超载专家选择分降低、
        欠载升高。bias 保持 fp32 累积（bf16 下 1e-3 步长会迅速被
        舍入吞掉），参与选择时按需 astype。
        """
        gates = self.moe_gates()
        if not gates or load_stats is None or load_stats.size == 0:
            return
        offset = 0
        for g in gates:
            n = g.n_routed
            load = load_stats[offset : offset + n].astype(mx.float32)
            offset += n
            err = load - mx.mean(load)
            g.expert_bias = g.expert_bias.astype(mx.float32) - rate * mx.sign(err)

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        self.save_weights(os.path.join(path, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyForCausalLM":
        config = VibyConfig.from_pretrained(path)
        model = cls(config)
        model.load_weights(os.path.join(path, "model.safetensors"))
        return model
