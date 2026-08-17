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
        # 深度循环：整个 layer 栈被重复执行 loop_k 次（ALBERT/Universal
        # Transformer 风格）。loop_k>1 时启用 per-step FiLM 调制以打破
        # 不同迭代之间的完全同质性（scale/shift 零初始化）。
        self.loop_k = kwargs.get("loop_k", 1)
        # ΔW-Loop（步条件低秩权重再生）：loop_k>1 且 dw_rank>0 时，块内
        # 每个 Linear 获得跨步共享的低秩基 U/V 与每步系数 g_step，使
        # W_eff(step) = W + U·diag(g_step)·V——把循环的每步多样性从激活
        # 空间（FiLM）提升到权重空间。V 零初始化：初始严格等价于基线；
        # 参数代价 ≈ r·(in+out) + k·r 每矩阵。loop_k=1 时不创建任何参数。
        self.dw_rank = kwargs.get("dw_rank", 0)
        # W-Scale-Loop（步条件对角权重缩放）：loop_k>1 且 ws_loop>0 时，块内
        # 每个 Linear 获得跨步独立的输入/输出对角缩放参数
        # W_eff(step) = diag(s_out[step])·W·diag(s_in[step])，用等价激活空间
        # 形式 y = ((x·s_in[step])Wᵀ)·s_out[step] 实现（不重建大权重，推理
        # 每 token 增量 ~2(in+out) FLOPs，参数代价 ≈ k·(in+out) 每矩阵）。
        # s 全 1 初始化 → 初始严格等价基线。可与 dw_rank 叠加（ΔW 的
        # delta 分支在缩放后的输入上计算，并同样经过 s_out 缩放）。
        self.ws_loop = kwargs.get("ws_loop", 0)
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
        # SAN（Simple Attention Network / Cactus Needle，arXiv:2607.18363）：
        # 去掉 FFN，把参数预算全部重配到注意力深度。同参数下 SAN 与 FFN
        # transformer 差距仅 ~0.006 nats；缺口集中在低上下文/权重知识。
        # 这里提供 SAN 的最小必要机制：ffn_type=none、zero-centered norm、
        # 标量残差门、o_proj 深度缩放初始化、embedding 缩放。
        self.ffn_type = kwargs.get("ffn_type", "swiglu")
        self.zero_centered_norm = kwargs.get("zero_centered_norm", 0)
        self.use_res_gate = kwargs.get("use_res_gate", 0)
        # Post-attention sandwich norm（SAN 论文消融中唯一正收益变体，
        # -0.009 nats）：注意力输出先过 norm 再加残差。
        self.sandwich_norm = kwargs.get("sandwich_norm", 0)
        self.san_res_init = kwargs.get("san_res_init", 0)
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

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

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


class ZCRMSNorm(nn.Module):
    """Zero-centered RMSNorm（SAN / Cactus Needle 论文组件）。

    ZCN(z) = (1 + γ) ⊙ z / RMS(z)，γ 初始化为 0。与普通 RMSNorm 在
    Adam 类优化器下参数化只差一个常数平移（优化等价），但便于深度
    attention-only 栈的初始化分析。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


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
                (mx.arange(dim // 2).astype(mx.float32) - low)
                / max(high - low, 0.001),
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


def cross_entropy(
    logits: mx.array,
    labels: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
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


class StepDeltaLinear(nn.Module):
    """步条件低秩权重再生（ΔW-Loop）+ 可选对角权重缩放（W-Scale-Loop）。

    激活空间等价形式（避免重建大权重矩阵）：
        y = ((x·s_in)@Wᵀ)·s_out + ((((x·s_in)@Vᵀ)·g_step)@Uᵀ)·s_out
    即 W_eff(step) = diag(s_out)·(W + U·diag(g_step)·V)·diag(s_in)。

    初始化：V=0、s_in=s_out=1 → 初始严格等于基线 Linear（g 取任意值均
    无贡献）；g=1 → 各步初始相同，退化为一个共享 LoRA，随训练各步系数
    逐步分化。step_idx=None（非循环调用，如 MTP 块）时严格等价于 base。
    rank=0 时只启用 W-Scale（ws_loop>0）；ws_loop=0 时只启用 ΔW。
    """

    def __init__(
        self,
        base: nn.Linear,
        loop_k: int,
        rank: int = 0,
        ws_loop: bool = False,
    ):
        super().__init__()
        self.base = base
        out_f, in_f = base.weight.shape
        self.rank = int(rank or 0)
        self.ws_loop = int(ws_loop or 0)
        if self.rank > 0:
            self.dw_u = mx.random.normal((out_f, rank)) * (1.0 / math.sqrt(rank))
            self.dw_v = mx.zeros((rank, in_f))
            self.dw_g = mx.ones((loop_k, rank))
        if self.ws_loop:
            self.ws_in = mx.ones((loop_k, in_f))
            self.ws_out = mx.ones((loop_k, out_f))

    def __call__(self, x: mx.array, step_idx: Optional[int] = None) -> mx.array:
        if step_idx is None:
            return self.base(x)

        x_in = x
        s_out = None
        if self.ws_loop:
            s_in = self.ws_in[step_idx].astype(x.dtype)
            x_in = x * s_in
            y = self.base(x_in)
            s_out = self.ws_out[step_idx].astype(y.dtype)
            y = y * s_out
        else:
            y = self.base(x)

        if self.rank > 0:
            # 大 tensor 全程保持 x.dtype；只有 (B,T,r) 小瓶颈过 f32。
            # 之前对整个 x/delta 上采样 f32 会把激活显存推高数倍，在长序列
            # 大 batch 的训练编译图里直接导致交换内存爬行
            g = self.dw_g[step_idx].astype(mx.float32)
            z = x_in @ self.dw_v.T.astype(x_in.dtype)
            z = (z.astype(mx.float32) * g).astype(x_in.dtype)
            delta = z @ self.dw_u.T.astype(x_in.dtype)
            if s_out is not None:
                delta = delta * s_out
            y = y + delta
        return y


def _linear(mod, x, step_idx=None):
    """ΔW/W-Scale 开启时 StepDeltaLinear 需要 step_idx；普通 nn.Linear 直接调用。"""
    return mod(x, step_idx) if isinstance(mod, StepDeltaLinear) else mod(x)


class Attention(nn.Module):
    """MLA（Multi-head Latent Attention，DeepSeek V2/V3 风格）。

    K/V 不按 head 直接投影：先压缩到低秩潜在向量 c = W_DKV·x
    （kv_lora_rank 维），K/V 使用时从 c 上投影；位置信息由独立的解耦
    RoPE 键携带（qk_rope_head_dim 维，跨 head 共享），Q 侧同样拼接
    nope（内容）+ rope（位置）两段。逐 head QK 维度 = head_dim +
    qk_rope_head_dim，V 维度 = head_dim。
    """

    def __init__(
        self,
        config: VibyConfig,
        use_dw: bool = False,
        use_ws: bool = False,
    ):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = self.head_dim + self.rope_dim
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.qk_dim, bias=False
        )
        self.kv_down_proj = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=False
        )
        self.k_up_proj = nn.Linear(
            config.kv_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.v_up_proj = nn.Linear(
            config.kv_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.k_rope_proj = nn.Linear(config.hidden_size, self.rope_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        # SAN 深度缩放初始化：o_proj 方差 ∝ 1/(2L)，保证深层 attention-only
        # 栈的残差流二阶矩不随深度增长（arXiv:2607.18363 Theorem 1）。
        if getattr(config, "san_res_init", 0):
            std = getattr(config, "initializer_range", 0.02) / math.sqrt(
                2.0 * max(1, config.num_hidden_layers)
            )
            self.o_proj.weight = mx.random.normal(self.o_proj.weight.shape, scale=std)
        # QK-norm 只作用于 nope（内容）段，rope 段保持原始 RoPE 几何。
        # SAN 模式使用 zero-centered RMSNorm（γ 零初始化）。
        norm_cls = ZCRMSNorm if getattr(config, "zero_centered_norm", 0) else RMSNorm
        self.q_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
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
            self.attn_gate = nn.Linear(
                config.hidden_size, self.n_heads, bias=True
            )
            self.attn_gate.weight = mx.zeros_like(self.attn_gate.weight)
            self.attn_gate.bias = mx.zeros_like(self.attn_gate.bias)
        # ΔW-Loop / W-Scale-Loop：块内注意力投影包上步条件低秩再生与/或
        # 对角权重缩放（V 零初始化、s 全 1 初始化 → 初始严格等价基线）；
        # 仅 loop_k>1 且对应开关开启时由上层启用
        if use_dw or use_ws:
            k = config.loop_k
            r = config.dw_rank if use_dw else 0
            ws = bool(use_ws and getattr(config, "ws_loop", 0))
            self.q_proj = StepDeltaLinear(self.q_proj, k, r, ws)
            self.kv_down_proj = StepDeltaLinear(self.kv_down_proj, k, r, ws)
            self.k_up_proj = StepDeltaLinear(self.k_up_proj, k, r, ws)
            self.v_up_proj = StepDeltaLinear(self.v_up_proj, k, r, ws)
            self.k_rope_proj = StepDeltaLinear(self.k_rope_proj, k, r, ws)
            self.o_proj = StepDeltaLinear(self.o_proj, k, r, ws)

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
        step_idx: Optional[int] = None,
    ) -> tuple[mx.array, Optional[tuple[mx.array, mx.array]]]:
        bsz, seq_len, _ = x.shape
        xq = _linear(self.q_proj, x, step_idx).reshape(
            bsz, seq_len, self.n_heads, self.qk_dim
        )
        q_nope, q_rope = mx.split(xq, [self.head_dim], axis=-1)

        c_kv = _linear(self.kv_down_proj, x, step_idx)  # (B, T, kv_lora_rank) 潜在向量
        xk = _linear(self.k_up_proj, c_kv, step_idx).reshape(
            bsz, seq_len, self.n_heads, self.head_dim
        )
        xv = _linear(self.v_up_proj, c_kv, step_idx).reshape(
            bsz, seq_len, self.n_heads, self.head_dim
        )
        k_rope = _linear(self.k_rope_proj, x, step_idx)[
            :, :, None, :
        ]  # (B, T, 1, rope_dim) 跨 head 共享

        # Value Residual Learning：第一层把本层 V 存入共享 list，后续层以
        # 可学习比例 λ = sigmoid(θ) 混入第一层 V（v' = (1-λ)v + λ·v_0）。
        # list 是跨层共享载体（compile 兼容）。
        if value_residual is not None:
            if len(value_residual) == 0:
                value_residual.append(xv)
            else:
                lam = mx.sigmoid(self.v_res_lambda).astype(xv.dtype)
                xv = (1.0 - lam) * xv + lam * value_residual[0]
        q_nope, xk = self.q_norm(q_nope), self.k_norm(xk)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        k_rope = mx.broadcast_to(k_rope, (bsz, seq_len, self.n_heads, self.rope_dim))
        xq = mx.concatenate([q_nope, q_rope], axis=-1)
        xk = mx.concatenate([xk, k_rope], axis=-1)

        if past_key_value is not None:
            xk = mx.concatenate([past_key_value[0], xk], axis=1)
            xv = mx.concatenate([past_key_value[1], xv], axis=1)
        past_kv = (xk, xv) if use_cache else None

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
                causal_mask = mx.triu(
                    mx.full((seq_len, seq_len), -mx.inf), k=1
                ).astype(scores.dtype)
                scores = scores.at[..., -seq_len:].add(causal_mask)

            if attention_mask is not None:
                key_len = scores.shape[-1]
                am = attention_mask
                if am.shape[1] < key_len:
                    pad = mx.ones((am.shape[0], key_len - am.shape[1]), dtype=am.dtype)
                    am = mx.concatenate([am, pad], axis=1)
                elif am.shape[1] > key_len:
                    am = am[:, -key_len:]
                scores = scores + (1.0 - am[:, None, None, :].astype(scores.dtype)) * -1e9

            # causal_bias（doc_mask 段掩码 / pad 融合掩码）在 eager 分支同样生效；
            # 与上面的 pad 处理重叠时只是 -1e9 叠加，语义不变
            if causal_bias is not None:
                scores = scores + causal_bias.astype(scores.dtype)

            attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(xq.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            output = attn_weights @ xv

        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        if self.attn_gate is not None:
            gate = mx.sigmoid(self.attn_gate(x).astype(mx.float32))
            output = output * mx.repeat(
                gate.astype(output.dtype), self.head_dim, axis=-1
            )
        output = self.resid_dropout(_linear(self.o_proj, output, step_idx))
        return output, past_kv


_ACT2FN = {"silu": nn.silu, "gelu": nn.gelu, "relu": nn.relu}


def _walsh_matrix(n: int) -> mx.array:
    """自然序 Walsh-Hadamard 矩阵，归一化 1/sqrt(n)。（测试/参考用）"""
    import numpy as np

    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return mx.array(h / np.sqrt(n))


def _fwht(x: mx.array) -> mx.array:
    """快速 Walsh-Hadamard 变换（O(n log n)，自然序，含 1/sqrt(n) 归一化）。

    与稠密 matmul(x, _walsh_matrix(n)) 数值等价（已对 n=4/8/16 验证）。
    """
    lead = x.shape[:-1]
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = mx.concatenate([a + b, a - b], axis=-1)
        h *= 2
    x = x.reshape(*lead, n)
    return x / math.sqrt(n)


class HadamardMLP(nn.Module):
    """Needle 2 的参数高效 channel mixer（arXiv:2607.18363 / cactus-compute）。

    z = pad(x)；z = (d1·z) H；z = SiLU(d2·z) H；out = d3·z
    其中 H 用快速 Walsh-Hadamard 变换实现（O(n log n)，无权重）。
    只学习 3n 个对角标量。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n = 1 << (dim - 1).bit_length()
        self.d1 = mx.ones((self.n,))
        self.d2 = mx.ones((self.n,))
        self.d3 = mx.full((self.n,), 0.02)

    def __call__(self, x: mx.array) -> mx.array:
        pad = self.n - self.dim
        if pad:
            z = mx.pad(x, ((0, 0), (0, 0), (0, pad)))
        else:
            z = x
        z = _fwht(self.d1.astype(z.dtype) * z)
        z = _fwht(nn.silu(self.d2.astype(z.dtype) * z))
        z = self.d3.astype(z.dtype) * z
        return z[..., : self.dim]


class FeedForward(nn.Module):
    def __init__(
        self,
        config: VibyConfig,
        intermediate_size: Optional[int] = None,
        use_dw: bool = False,
        use_ws: bool = False,
    ):
        super().__init__()
        self.ffn_type = getattr(config, "ffn_type", "swiglu")
        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.hadamard = None
        if self.ffn_type == "hadamard":
            self.hadamard = HadamardMLP(config.hidden_size)
            return
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = _ACT2FN[config.hidden_act]
        # ΔW-Loop / W-Scale-Loop：FFN 三个投影的步条件低秩再生与/或
        # 对角权重缩放（同 Attention）
        if use_dw or use_ws:
            k = config.loop_k
            r = config.dw_rank if use_dw else 0
            ws = bool(use_ws and getattr(config, "ws_loop", 0))
            self.gate_proj = StepDeltaLinear(self.gate_proj, k, r, ws)
            self.down_proj = StepDeltaLinear(self.down_proj, k, r, ws)
            self.up_proj = StepDeltaLinear(self.up_proj, k, r, ws)

    def __call__(self, x: mx.array, step_idx: Optional[int] = None) -> mx.array:
        if self.hadamard is not None:
            return self.hadamard(x)
        h = self.act_fn(_linear(self.gate_proj, x, step_idx)) * _linear(
            self.up_proj, x, step_idx
        )
        return _linear(self.down_proj, h, step_idx)

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
                acc = (
                    mx.bitwise_xor(acc, _shift_right_tokens(u, j))
                    * mx.array(_ENGRAM_PRIME, dtype=mx.uint32)
                )
            acc = mx.bitwise_xor(acc, mx.right_shift(acc, mx.array(15, dtype=mx.uint32)))
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
        self.key_proj = nn.Linear(self.num_tables * self.sub_dim, config.hidden_size, bias=False)
        # value_proj 零初始化：冷启动时 α≈σ(0)=0.5，随机 ev 会向残差流注入
        # 噪声（r064/r065 两个尺度一致小幅负收益）；零初始化使 engram 从严格
        # 恒等出发，与 ΔW V=0 / res_gate 的约定一致。
        self.value_proj = nn.Linear(self.num_tables * self.sub_dim, config.hidden_size, bias=False)
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
        ngram_ok = mx.ones((B, T), dtype=mx.float32)
        for order in self.orders:
            ok = mx.ones((B, T), dtype=mx.float32)
            for j in range(order):
                if segment_ids is not None:
                    seg_prev = _shift_right_tokens(segment_ids, j)
                    same_doc = segment_ids == seg_prev
                else:
                    same_doc = mx.ones_like(input_ids).astype(mx.bool_)
                ok = ok * _shift_right_tokens(base_ok, j) * same_doc.astype(mx.float32)
            ngram_ok = ngram_ok * ok
        ngram_ok = ngram_ok * base_ok
        ngram_ok = mx.repeat(ngram_ok[:, :, None], self.heads, axis=-1)
        ngram_ok = mx.repeat(ngram_ok, len(self.orders), axis=-1)  # (B,T,K)
        ngram_ok = mx.repeat(ngram_ok[..., None], self.sub_dim, axis=-1)
        ngram_ok = ngram_ok.reshape(B, T, K * self.sub_dim)
        e = e * ngram_ok

        ek = self.key_proj(e)
        ev = self.value_proj(e)
        # 4-tap 扩张卷积，全部沿因果方向。
        ev_conv = ev * self.taps[0].astype(ev.dtype)
        for j in range(1, _ENGRAM_CONV_TAPS):
            shift = j * self.dilation
            ev_conv = ev_conv + _shift_right_tokens(ev, shift) * self.taps[j].astype(ev.dtype)
        return ek, ev_conv


class VibyBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: VibyConfig,
        use_dw: bool = False,
        use_ws: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(config, use_dw=use_dw, use_ws=use_ws)
        norm_cls = ZCRMSNorm if getattr(config, "zero_centered_norm", 0) else RMSNorm
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn_type = getattr(config, "ffn_type", "swiglu")
        self.mlp = (
            None
            if self.ffn_type == "none"
            else FeedForward(config, use_dw=use_dw, use_ws=use_ws)
        )
        # SAN 标量残差门：y = x + σ(g)·attn_out，g 零初始化 → 初始 0.5。
        self.use_res_gate = getattr(config, "use_res_gate", 0)
        self.res_gate = mx.array(0.0) if self.use_res_gate else None
        # Post-attention sandwich norm（注意力输出在加残差前再过一次 norm）
        self.sandwich_norm = getattr(config, "sandwich_norm", 0)
        self.post_attn_norm = (
            norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            if self.sandwich_norm
            else None
        )

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
            # 余弦相似度门控（Needle：α = σ(rms(x)·rms(ek)/√d)）
            alpha = mx.sigmoid(
                mx.sum(_rms_unit(hidden_states) * _rms_unit(ek), axis=-1)
                / math.sqrt(self.self_attn.qk_dim)
            )
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
            step_idx=step_idx,
        )
        if self.sandwich_norm:
            hidden_states = self.post_attn_norm(hidden_states)
        if self.use_res_gate:
            hidden_states = residual + mx.sigmoid(self.res_gate) * hidden_states
        else:
            hidden_states = hidden_states + residual
        if self.mlp is None:
            return hidden_states, present_key_value
        residual = hidden_states
        mlp_in = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_in, step_idx=step_idx)
        hidden_states = residual + mlp_output
        return hidden_states, present_key_value


class VibyStack(nn.Module):
    """HRM 模式的一个 transformer stack：P 层 + 尾部 RMSNorm。

    对应 HF HrmTextStack。L/H 两个 stack 结构完全相同、参数独立。
    """

    def __init__(
        self,
        config: VibyConfig,
        n_layers: int,
        use_dw: bool = False,
        use_ws: bool = False,
    ):
        super().__init__()
        self.layers = [
            VibyBlock(layer, config, use_dw=use_dw, use_ws=use_ws)
            for layer in range(n_layers)
        ]
        norm_cls = ZCRMSNorm if getattr(config, "zero_centered_norm", 0) else RMSNorm
        self.final_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

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
        self.block = VibyBlock(0, config)

    def __call__(
        self,
        h_prev: mx.array,
        token_emb: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
    ) -> mx.array:
        x = mx.concatenate([self.norm_h(h_prev), self.norm_e(token_emb)], axis=-1)
        x = self.proj(x)
        out, _ = self.block(
            x,
            position_embeddings,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
        )
        return out


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.use_value_res = getattr(config, "use_value_res", False)
        self.loop_k = int(getattr(config, "loop_k", 1) or 1)
        self.hrm_H = int(getattr(config, "hrm_H_cycles", 0) or 0)
        self.hrm_L = int(getattr(config, "hrm_L_cycles", 3) or 0)
        self.hrm_emb_scale = float(getattr(config, "hrm_emb_scale", 1.0) or 1.0)
        self.emb_scale = float(getattr(config, "emb_scale", 1.0) or 1.0)
        # ΔW-Loop / W-Scale-Loop 仅在普通循环开启时生效（loop_k=1 没有步概念，
        # 不创建参数，严格向后兼容）；HRM 模式不使用这两个机制。
        use_dw = getattr(config, "dw_rank", 0) > 0 and self.loop_k > 1
        use_ws = getattr(config, "ws_loop", 0) > 0 and self.loop_k > 1
        self.l_module = None
        self.h_module = None
        if self.hrm_H > 0:
            if self.loop_k != 1:
                raise ValueError("hrm_H_cycles 与 loop_k 互斥：HRM 模式请用 loop_k=1")
            # num_hidden_layers 在 HRM 模式下表示每个 stack 的真实层数 P；
            # 一个 token 的层求值次数 = H*(L+1)*P，训练/推理前向完全一致。
            self.l_module = VibyStack(config, config.num_hidden_layers)
            self.h_module = VibyStack(config, config.num_hidden_layers)
            self.layers = None
            raw_bp = list(getattr(config, "hrm_bp_cycles", None) or [self.hrm_L])
            self.hrm_bp_padded = [1] * max(0, self.hrm_H - len(raw_bp)) + raw_bp
        else:
            self.layers = [
                VibyBlock(layer, config, use_dw=use_dw, use_ws=use_ws)
                for layer in range(config.num_hidden_layers)
            ]
            self.hrm_bp_padded = None
        # Engram n-gram 记忆位点。普通栈按 layer_idx 注入；HRM 模式注入
        # L-module（快状态栈）的同名层，每个循环调用都会重读一次表。
        engram_layers = tuple(
            int(x) for x in (getattr(config, "engram_layers", ()) or ())
        )
        self.engram_layers = engram_layers
        # 越界位点会建表但永远注入不到（按 layer_idx 匹配），直接拒绝，
        # 防止静默的 no-op 参数（test_engram 曾因此漏检第二个位点）。
        bad = [s for s in self.engram_layers if not 0 <= s < config.num_hidden_layers]
        if bad:
            raise ValueError(
                f"engram_layers {bad} 超出栈范围 [0, {config.num_hidden_layers})"
            )
        self.engrams = (
            [Engram(config) for _ in self.engram_layers]
            if self.engram_layers
            else []
        )
        # 深度循环参数。step_scale/step_shift 零初始化：loop_k=1 或初始时
        # 严格等价于不循环的基线（loop_k=1 不创建这两个参数）。
        if not self.hrm_H and self.loop_k > 1:
            self.step_scale = mx.zeros((self.loop_k, config.hidden_size))
            self.step_shift = mx.zeros((self.loop_k, config.hidden_size))
        else:
            self.step_scale = None
            self.step_shift = None
        norm_cls = ZCRMSNorm if getattr(config, "zero_centered_norm", 0) else RMSNorm
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
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
        **kwargs,
    ) -> tuple[mx.array, list]:
        batch_size, seq_length = input_ids.shape
        hrm_mode = self.hrm_H > 0
        if hrm_mode:
            n_layers_per_stack = self.config.num_hidden_layers
            n_effective_layers = (
                self.hrm_H * (self.hrm_L + 1) * n_layers_per_stack
            )
        else:
            n_passes = self.loop_k
            n_effective_layers = len(self.layers) * n_passes
        if past_key_values is None:
            past_key_values = [None] * n_effective_layers
        elif (
            not hrm_mode
            and len(past_key_values) == len(self.layers)
            and not use_cache
        ):
            # 兼容旧调用方：无 cache 时展开为 loop_k 份 None
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
            causal = mx.triu(
                mx.full((seq_length, seq_length), -1e9), k=1
            )
            causal_bias = (
                causal[None, None, :, :] + pad_bias[:, None, None, :]
            ).astype(hidden_states.dtype)

        # 文档边界掩码（doc_mask 打包训练）：注意力限制在同文档内因果可见，
        # 消除跨文档泄漏，与逐篇 PPL 评估口径对齐。仅在完整前向（训练）传入。
        if segment_ids is not None and first_cache is None and seq_length > 1:
            same_doc = segment_ids[:, :, None] == segment_ids[:, None, :]
            causal_tril = mx.tril(
                mx.ones((seq_length, seq_length), dtype=mx.bool_)
            )
            allowed = same_doc & causal_tril[None, :, :]
            seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(
                hidden_states.dtype
            )
            causal_bias = (
                seg_bias if causal_bias is None else causal_bias + seg_bias
            )
            mask_is_full = False

        # value residual 的跨层共享载体：第一层写入 v_0，后续层读取混合
        value_residual = [] if self.use_value_res else None

        # Engram 注入位点：仅完整前向（训练 / 无 cache 的 PPL 评估）启用；
        # cache 解码暂不注入（跨 chunk 的 n-gram 前文需要额外 carry）。
        engram_evs = {}
        if self.engrams and not use_cache and first_cache is None:
            for site_layer, eng in zip(self.engram_layers, self.engrams):
                engram_evs[site_layer] = eng(
                    input_ids, attention_mask, segment_ids
                )

        presents = []
        if hrm_mode:
            # HRM 双状态层次循环。z_H = 慢/高状态，z_L = 快/低状态；
            # 每个 token 的训练前向与推理前向完全一致，没有推理侧额外展开。
            z_h = hidden_states
            z_l = mx.zeros_like(z_h)
            for h_idx in range(self.hrm_H):
                num_grad = int(
                    self.hrm_bp_padded[h_idx]
                    if h_idx < len(self.hrm_bp_padded)
                    else 1
                )
                grad_threshold = self.hrm_L - num_grad
                for l_idx in range(self.hrm_L):
                    cache_offset = (
                        h_idx * (self.hrm_L + 1) + l_idx
                    ) * n_layers_per_stack
                    z_l, p = self.l_module(
                        z_l + z_h,
                        position_embeddings,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        attention_mask=attention_mask,
                        causal_bias=causal_bias,
                        mask_is_full=mask_is_full,
                        value_residual=value_residual,
                        step_idx=None,
                        cache_offset=cache_offset,
                        engram_evs=engram_evs,
                    )
                    presents.extend(p)
                    # L_bp_cycles 梯度路由：早期 L cycle 只做前向，
                    # 仅尾部 num_grad 个 cycle 回传梯度（与 HF HrmText 一致）。
                    if l_idx < grad_threshold:
                        z_l = mx.stop_gradient(z_l)

                cache_offset = (
                    h_idx * (self.hrm_L + 1) + self.hrm_L
                ) * n_layers_per_stack
                z_h, p = self.h_module(
                    z_h + z_l,
                    position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    causal_bias=causal_bias,
                    mask_is_full=mask_is_full,
                    value_residual=value_residual,
                    step_idx=None,
                    cache_offset=cache_offset,
                )
                presents.extend(p)

            hidden_states = z_h
            return hidden_states, presents

        flat_idx = 0
        for step in range(n_passes):
            if self.step_scale is not None:
                scale = self.step_scale[step].astype(hidden_states.dtype)
                shift = self.step_shift[step].astype(hidden_states.dtype)
                hidden_states = hidden_states * (1.0 + scale) + shift
            for layer_idx, layer in enumerate(self.layers):
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_values[flat_idx],
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    causal_bias=causal_bias,
                    mask_is_full=mask_is_full,
                    value_residual=value_residual,
                    step_idx=step,
                    engram_ev=engram_evs.get(layer_idx),
                )
                presents.append(present)
                flat_idx += 1

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
            mask = mx.zeros(row.shape[-1], dtype=mx.int32).at[
                seen.astype(mx.int32)
            ].maximum(1)
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
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
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
            h_k = module(
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
        **kwargs,
    ) -> CausalLMOutput:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            mask_has_pad=mask_has_pad,
            segment_ids=segment_ids,
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
            raise ValueError(f"input_ids 必须是 2 维 (batch, seq)，实际为 {input_ids.ndim} 维")
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
            outputs = self(
                current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits_mx = outputs.logits[:, -1, :]
            if attention_mask is not None:
                attention_mask = mx.concatenate(
                    [attention_mask, mx.ones((batch, 1), dtype=attention_mask.dtype)],
                    axis=-1,
                )

            seen_ids = input_ids if repetition_penalty != 1.0 else None
            transformed = _transform_logits_mx(
                logits_mx, seen_ids, temperature, top_p, top_k,
                do_sample, repetition_penalty,
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
    ) -> tuple[list[int], list[mx.array]]:
        """Draft tokens with the chained MTP modules.

        h_last: (1, 1, hidden) main-model hidden state at position pos_idx.
        first_token: the bonus token at position pos_idx + 1.
        draft_len: 每轮草稿的 token 数。超过模块数时循环复用 MTP 模块
        （与 vLLM 的 MTP self-speculation 一致，业界 depth=1 但草稿 3~7 个）。
        Returns (draft_token_ids, draft_probs) with len == draft_len，
        全程 MLX，logits 不离开 GPU。
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
            module = self.mtp_modules[i % n_modules]
            emb = self.model.embed_tokens(token)
            h = module(h, emb, (cos.astype(h.dtype), sin.astype(h.dtype)))
            logits = self._lm_logits(h)[0, 0]
            # repetition penalty 的上下文应包含 bonus token 与已生成的草稿
            seen_cur = mx.concatenate(
                [seen_ids, mx.array([first_token] + drafts, dtype=mx.int32)]
            )
            transformed = _transform_logits_mx(
                logits, seen_cur, temperature, top_p, top_k,
                do_sample, repetition_penalty,
            )
            probs = _probs_mx(transformed)
            tok = int(_sample_from_logits_mx(transformed, do_sample).item())
            drafts.append(tok)
            draft_probs.append(probs)
            token = mx.array([[tok]])
        return drafts, draft_probs

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
        out = self(input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        h_last = out.hidden_states[:, -1:, :]
        logits_last = out.logits[0, -1]

        if streamer:
            streamer.put(input_ids)

        max_pos = self.config.max_position_embeddings
        generated: list[int] = []
        stats = {"accepted": 0, "drafted": 0}
        while len(generated) < max_new_tokens:
            seen = mx.concatenate([prompt_ids, mx.array(generated, dtype=mx.int32)])
            # 1. Bonus token from the main model (always accepted).
            bonus_logits = _transform_logits_mx(
                logits_last, seen, temperature, top_p, top_k,
                do_sample, repetition_penalty,
            )
            bonus = int(_sample_from_logits_mx(bonus_logits, do_sample).item())
            # 2. Draft with the MTP chain.
            # 剩余额度/上下文位置有限时收紧草稿数：bonus 必占 1 个，
            # full-accept 的额外 tail 还要再占 1 个位置，避免越过
            # max_position_embeddings（前向里有显式越界检查）。
            remaining = max_new_tokens - len(generated)
            iter_draft_len = min(draft_len, max(0, remaining - 2))
            drafts, draft_probs = [], []
            if iter_draft_len > 0:
                drafts, draft_probs = self._mtp_draft(
                    h_last, bonus, seq_len - 1, iter_draft_len, seen,
                    temperature, top_p, top_k,
                    do_sample, repetition_penalty,
                )
            stats["drafted"] += len(drafts)
            # 3. Verify the chain in parallel with the main model.
            verify_tokens = mx.array([[bonus] + drafts], dtype=mx.int32)
            past_before_verify = past
            vout = self(
                verify_tokens,
                attention_mask=attention_mask,
                past_key_values=past_before_verify,
                use_cache=True,
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
                    vlogits[i], seen_i, temperature, top_p, top_k,
                    do_sample, repetition_penalty,
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
                    vlogits[iter_draft_len], seen_tail, temperature, top_p, top_k,
                    do_sample, repetition_penalty,
                )
                tail = int(_sample_from_logits_mx(tail_logits, do_sample).item())
            elif iter_draft_len == 0:
                # 上下文/剩余额度边界：只产出 bonus，不补 tail
                tail = None
            elif do_sample:
                # Reject: resample from the positive residual (p - q)+.
                p_logits = _transform_logits_mx(
                    vlogits[n_acc], seen_tail, temperature, top_p, top_k,
                    do_sample, repetition_penalty,
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
                    vlogits[n_acc], seen_tail, temperature, top_p, top_k,
                    False, repetition_penalty,
                )
                tail = int(mx.argmax(tail_logits).item())

            new_tokens = accepted_prefix + ([tail] if tail is not None else [])

            # 5. Stop at EOS (keep it, drop anything after).
            stop = False
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
                stop = True

            # 6. 回滚 cache 到实际接受的前缀，再前向 tail token。
            cache_prefix = new_tokens[:-1]
            if cache_prefix:
                keep = seq_len + len(cache_prefix)
                past = [(k[:, :keep], v[:, :keep]) for k, v in past_full]
            else:
                past = past_before_verify
            seq_len += len(new_tokens)
            tout = self(
                mx.array([[new_tokens[-1]]]),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = tout.past_key_values
            h_last = tout.hidden_states[:, -1:, :]
            logits_last = tout.logits[0, -1]

            generated.extend(new_tokens)
            if streamer:
                streamer.put(mx.array([new_tokens]))
            if stop:
                break

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
            past = [(k[:, :, :target_len], v[:, :, :target_len]) for k, v in past]
        object.__setattr__(self, "_last_spec_stats", stats)
        object.__setattr__(self, "_last_spec_past", past)
        return out_ids

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
