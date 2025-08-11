from transformers.configuration_utils import PretrainedConfig
import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin  # type: ignore
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from causal_conv1d_mps import short_conv_fused
    HAS_CAUSAL_CONV1D_MPS = True
except ImportError:
    HAS_CAUSAL_CONV1D_MPS = False
    short_conv_fused = None


class VibyConfig(PretrainedConfig):
    model_type = "viby"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 640,
        intermediate_size: int = 1280,
        max_position_embeddings: int = 32768,
        original_max_position_embeddings: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 16,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        z_loss_factor: float = 0.0001,
        sliding_window: int = 128,
        canon_set: str = "ABCD",
        canon_bias: bool = True,
        canon_activation: bool = True,
        canon_kernel: int = 4,
        canon_residual: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.z_loss_factor = z_loss_factor
        self.attn_softmax_temp = kwargs.get("attn_softmax_temp", 1.0)
        # SWA sliding window length; 参考 gpt-oss，默认开启 128；在注意力层内每隔一层生效
        self.sliding_window = sliding_window
        # Canon layers configuration
        self.canon_set = canon_set
        self.canon_bias = canon_bias
        self.canon_activation = canon_activation
        self.canon_kernel = canon_kernel
        self.canon_residual = canon_residual


def z_loss_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, z_loss_factor: float = 0.0
) -> torch.Tensor:
    """Cross entropy with Z-loss for training stability."""
    ce = F.cross_entropy(
        logits.flatten(end_dim=-2), labels.flatten(end_dim=-1), reduction="none"
    )  # (B*T,)

    if z_loss_factor > 0.0:
        # z = logsumexp over vocab for each position
        # shape (B, T)
        z = torch.logsumexp(logits, dim=-1)
        # square penalty, then mean
        z_term = z_loss_factor * (z.square())
        # match CE shape (B*T,) for joint mean
        z_term = z_term.flatten()
        loss = (ce + z_term).mean()
    else:
        loss = ce.mean()
    return loss


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def find_correction_range(
    low_freq_factor: float,
    high_freq_factor: float,
    dim: int,
    base: float,
    original_max_pos_embeds: int,
) -> torch.Tensor:
    """
    Determines which dimensions should be scaled differently based on their frequency.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    low_freq_wavelen = original_max_pos_embeds / low_freq_factor
    high_freq_wavelen = original_max_pos_embeds / high_freq_factor

    return torch.logical_and(
        inv_freq < high_freq_wavelen, inv_freq > low_freq_wavelen
    ).float()


def precompute_freqs_cis_yarn(
    dim: int,
    end: int,
    theta: float,
    scaling_factor: float,
    original_max_pos_embeds: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
):
    """
    Precompute RoPE frequencies with YaRN scaling ("NTK-by-parts" interpolation).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    correction_range = find_correction_range(
        beta_slow, beta_fast, dim, theta, original_max_pos_embeds
    ).to(t.device)

    # Standard position interpolation scaling
    t_pi = t / scaling_factor

    # NTK-aware scaling (extrapolation)
    ntk_scaling_factor = (
        scaling_factor ** (dim / (dim - 2)) if dim > 2 else scaling_factor
    )
    t_ntk = t / ntk_scaling_factor

    # Blend the two based on the correction mask
    corrected_t = (
        t_pi[:, None] * (1 - correction_range) + t_ntk[:, None] * correction_range
    )

    freqs = corrected_t * freqs

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


class CanonLayer(nn.Module):
    """Canon layer using causal 1D convolution for efficient sequence modeling."""
    
    def __init__(self, hidden_size: int, config: VibyConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = config.canon_kernel
        self.bias = config.canon_bias
        self.activation = config.canon_activation
        self.residual = config.canon_residual
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.randn(hidden_size, self.kernel_size))
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias_param', None)
            
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, seq_len) with 1s for valid tokens
        """
        # Try to use MPS kernel first if available and on MPS device
        if HAS_CAUSAL_CONV1D_MPS and x.device.type == "mps":
            try:
                output = short_conv_fused(
                    x=x,
                    weight=self.weight,
                    bias=self.bias_param,
                    attention_mask=attention_mask,
                    activation=self.activation,
                    residual=self.residual
                )
                return output
            except Exception as e:
                print(f"Warning: MPS kernel failed ({e}), falling back to PyTorch implementation")
        
        # Fallback to PyTorch implementation
        residual = x if self.residual else 0
        
        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        # Convert to (batch, hidden_size, seq_len) for conv1d
        x_conv = x.transpose(1, 2)
        
        # Manual grouped convolution using F.conv1d
        # Create weight tensor in correct format for F.conv1d: (out_channels, in_channels//groups, kernel_size)
        conv_weight = self.weight.unsqueeze(1)  # (hidden_size, 1, kernel_size)
        
        # Apply causal padding
        x_padded = F.pad(x_conv, (self.kernel_size - 1, 0))
        
        # Grouped convolution
        x_conv = F.conv1d(x_padded, conv_weight, bias=self.bias_param, groups=self.hidden_size)
        
        # Remove padding to make it causal (keep only first seq_len outputs)
        x_conv = x_conv[..., :x.shape[1]]
        
        # Apply activation if specified
        if self.activation:
            x_conv = F.silu(x_conv)
        
        # Convert back to (batch, seq_len, hidden_size)
        output = x_conv.transpose(1, 2)
        
        return output + residual


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 GQA 的 key/value 头以匹配 query 头的数量。
    """
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=2)


class Attention(nn.Module):
    def __init__(self, args: VibyConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.attn_softmax_temp = args.attn_softmax_temp
        # SWA attention sink (per-head learnable logit slot), follow gpt-oss style
        self.sinks = nn.Parameter(torch.zeros(self.n_local_heads))
        self.sliding_window_default = args.sliding_window

        # QK normalization layers (like Qwen3)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        
        # Canon B layer for QKV projections
        if "B" in args.canon_set:
            total_dim = (
                args.num_attention_heads * self.head_dim
                + 2 * args.num_key_value_heads * self.head_dim
            )
            self.canon_b = CanonLayer(total_dim, args)
        else:
            self.canon_b = None

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        
        # Apply Canon B if enabled
        if self.canon_b is not None:
            # Flatten back to 2D for Canon layer
            xq_flat = xq.view(bsz, seq_len, -1)
            xk_flat = xk.view(bsz, seq_len, -1) 
            xv_flat = xv.view(bsz, seq_len, -1)
            
            # Apply Canon B to concatenated QKV
            qkv_concat = torch.cat([xq_flat, xk_flat, xv_flat], dim=-1)
            qkv_processed = self.canon_b(qkv_concat, attention_mask)
            
            # Split back to Q, K, V
            q_dim = self.n_local_heads * self.head_dim
            kv_dim = self.n_local_kv_heads * self.head_dim
            xq_flat, xk_flat, xv_flat = qkv_processed.split([q_dim, kv_dim, kv_dim], dim=-1)
            
            # Reshape back to multi-head format
            xq = xq_flat.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk_flat.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv_flat.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            # 将过去的 k, v 与当前的 k, v 连接起来
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        # GQA: 重复 key 和 value 以匹配 query 的头数
        key = repeat_kv(xk, self.n_rep)
        value = repeat_kv(xv, self.n_rep)

        # 维度转换以适配 scaled_dot_product_attention
        # (bsz, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 构建掩码与缩放，改为手写注意力以注入 SWA sink

        # Dropout 概率（与 F.sdpa 语义一致，只在训练时生效）
        dropout_p = self.dropout if self.training else 0.0

        # YaRN 温度缩放：等价于对 Q 做温度缩放
        if self.attn_softmax_temp != 1.0:
            xq = xq / self.attn_softmax_temp

        # 手写 attention：QK^T / sqrt(d)
        d = self.head_dim
        scale = 1.0 / math.sqrt(d)
        # xq: [B, H, Tq, D], key: [B, H, Tk, D]
        # 使用 einsum 代替 batched matmul，等价于: (B,H,Tq,D) x (B,H,Tk,D)^T -> (B,H,Tq,Tk)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", xq, key) * scale

        # 始终应用因果掩码（decoder-only）
        tq = attn_scores.size(-2)
        tk = attn_scores.size(-1)
        past_len = tk - tq
        causal_mask = torch.ones((tq, tk), device=attn_scores.device, dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1 + past_len)
        attn_scores = attn_scores.masked_fill(
            causal_mask, torch.finfo(attn_scores.dtype).min
        )

        # Key padding mask：由 attention_mask 推导（1=保留，0=填充 → True 表示屏蔽）
        if attention_mask is not None:
            # 目标 key 长度
            tk = attn_scores.size(-1)

            if attention_mask.dim() == 2:
                # [B, T] → 调整到与 tk 一致
                am = attention_mask
                if am.size(1) != tk:
                    if am.size(1) < tk:
                        pad_len = tk - am.size(1)
                        pad = torch.ones((am.size(0), pad_len), dtype=am.dtype, device=am.device)
                        am = torch.cat([am, pad], dim=1)
                    else:
                        # 超长则取末尾的 tk 段（通常与当前 K 对齐）
                        am = am[:, -tk:]
                key_padding_mask = (am == 0).unsqueeze(1).unsqueeze(1)  # [B,1,1,Tk]
            else:
                # 已经是更高维度的掩码时，尽力广播并对齐 tk
                kpm = attention_mask.to(torch.bool)
                if kpm.dim() == 3:
                    kpm = kpm.unsqueeze(1)  # [B,1,Tq,Tk] or [B,1,1,Tk]
                if kpm.size(-1) != tk:
                    if kpm.size(-1) < tk:
                        pad_shape = list(kpm.shape)
                        pad_shape[-1] = tk - kpm.size(-1)
                        pad = torch.zeros(pad_shape, dtype=kpm.dtype, device=kpm.device)
                        kpm = torch.cat([kpm, pad], dim=-1)
                    else:
                        kpm = kpm[..., -tk:]
                key_padding_mask = kpm

            attn_scores = attn_scores.masked_fill(
                key_padding_mask, torch.finfo(attn_scores.dtype).min
            )

        # Sliding Window Attention（参考 gpt-oss），仅当窗口 > 0 时启用
        win = self.sliding_window_default if sliding_window is None else sliding_window
        if win is not None and win > 0:
            tq = attn_scores.size(-2)
            tk = attn_scores.size(-1)
            past_len = tk - tq
            q_abs = torch.arange(tq, device=attn_scores.device) + past_len
            k_abs = torch.arange(tk, device=attn_scores.device)
            # 仅允许关注最近 win 个 key：mask 位置为 True 表示被屏蔽
            allow = k_abs.unsqueeze(0) >= (q_abs.unsqueeze(1) - win)
            sw_mask = ~allow  # [Tq, Tk]
            attn_scores = attn_scores.masked_fill(
                sw_mask, torch.finfo(attn_scores.dtype).min
            )

        # SWA sink：按 gpt-oss，在 softmax 归一化前拼接额外列，再 softmax，之后丢弃该列
        sink_logits = self.sinks.view(1, self.n_local_heads, 1, 1).to(attn_scores.dtype)
        sink_logits = sink_logits.expand(bsz, -1, attn_scores.size(-2), 1)
        attn_scores = torch.cat([attn_scores, sink_logits], dim=-1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        # 丢弃 sink 概率列，仅对真实 token 做加权
        attn_weights = attn_weights[..., :-1]
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)

        # 与 V 做乘积
        # 使用 einsum 代替 batched matmul，(B,H,Tq,Tk) x (B,H,Tk,D) -> (B,H,Tq,D)
        output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, value)

        # 回转维度
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        # keep original act for compatibility but we will use swiglu_clamp path
        self.act_fn = ACT2FN.get(config.hidden_act, torch.nn.functional.silu)
        # SwiGLU parameters
        self.swiglu_alpha = 1.702
        self.swiglu_limit = 7.0
        
        # Canon D layer for gate and up projections
        if "D" in config.canon_set:
            # gate_proj outputs intermediate_size * 2, up_proj outputs intermediate_size
            # Total concatenated dimension: intermediate_size * 3
            self.canon_d = CanonLayer(config.intermediate_size * 3, config)
        else:
            self.canon_d = None

    @staticmethod
    def swiglu_clamp(
        x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
    ) -> torch.Tensor:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        # Use clamp-ed SwiGLU with gate-up structure
        gated = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply Canon D if enabled
        if self.canon_d is not None:
            gate_up_concat = torch.cat([gated, up], dim=-1)
            gate_up_processed = self.canon_d(gate_up_concat, attention_mask)
            # Split back: gated (intermediate_size * 2) + up (intermediate_size)
            gated = gate_up_processed[..., :gated.shape[-1]]
            up = gate_up_processed[..., gated.shape[-1]:]
        
        h = self.swiglu_clamp(gated, alpha=self.swiglu_alpha, limit=self.swiglu_limit)
        out = self.down_proj(h * up)
        return self.dropout(out)


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.default_sliding_window = config.sliding_window
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)
        
        # Canon A layer (after input layernorm, before attention)
        if "A" in config.canon_set:
            self.canon_a = CanonLayer(config.hidden_size, config)
        else:
            self.canon_a = None
            
        # Canon C layer (after post-attention layernorm, before MLP)
        if "C" in config.canon_set:
            self.canon_c = CanonLayer(config.hidden_size, config)
        else:
            self.canon_c = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        layer_id: int,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        
        # Pre-attention processing
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Apply Canon A if enabled
        if self.canon_a is not None:
            normed_hidden_states = self.canon_a(normed_hidden_states, attention_mask)
        
        # sliding window 策略：参考 gpt-oss，仅对偶数层启用，默认窗口 128
        sliding_window = self.default_sliding_window if (layer_id % 2 == 0) else 0
        attn_output, present_key_value = self.self_attn(
            normed_hidden_states,
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
            sliding_window=sliding_window,
        )
        hidden_states = residual + attn_output
        
        # Pre-MLP processing
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply Canon C if enabled
        if self.canon_c is not None:
            normed_hidden_states = self.canon_c(normed_hidden_states, attention_mask)
            
        mlp_output = self.mlp(normed_hidden_states, attention_mask)
        hidden_states = residual + mlp_output
        
        return hidden_states, present_key_value


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        if config.rope_scaling and config.rope_scaling.get("type") == "yarn":
            scaling_factor = config.rope_scaling["factor"]

            # Dynamically calculate attention temperature based on scaling factor
            config.attn_softmax_temp = 0.5 * math.log(scaling_factor) + 1.0
            print(
                f"[YaRN] Enabled with scaling factor {scaling_factor}. New attention temp: {config.attn_softmax_temp:.3f}"
            )

            freqs_cos, freqs_sin = precompute_freqs_cis_yarn(
                dim=config.hidden_size // config.num_attention_heads,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
                scaling_factor=scaling_factor,
                original_max_pos_embeds=config.original_max_position_embeddings,
                beta_fast=config.rope_scaling.get("beta_fast", 32.0),
                beta_slow=config.rope_scaling.get("beta_slow", 1.0),
            )
        else:
            # Fallback to original RoPE if YaRN is not configured
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=config.hidden_size // config.num_attention_heads,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [VibyBlock(layer, config) for layer in range(self.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)  # type: ignore
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],  # type: ignore
            self.freqs_sin[start_pos : start_pos + seq_length],  # type: ignore
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                layer_id=layer_idx,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class VibyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VibyConfig

    def __init__(self, config: VibyConfig):
        self.config = config or VibyConfig()
        super().__init__(self.config)
        self.model = VibyModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # Initialize canon layers if present
        self.apply(self._init_canon_layers)
        
    def _init_canon_layers(self, module):
        if isinstance(module, CanonLayer):
            module.reset_parameters()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,  # 增加了 labels 以便计算 loss
        **args,
    ):
        hidden_states, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Use Z-loss if enabled
            loss = z_loss_cross_entropy(
                shift_logits, shift_labels, self.config.z_loss_factor
            )

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )

    # 让 HF generate 正确地只喂最后一个 token（当使用缓存时），并传递好 attention_mask
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 当有缓存时，仅传入最后一个 token，避免重复累积 K/V
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    # Beam search 等会调用该方法以重排缓存
    def _reorder_cache(self, past_key_values, beam_idx: torch.LongTensor):
        if past_key_values is None:
            return past_key_values
        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
                continue
            k, v = layer_past
            k = k.index_select(0, beam_idx)
            v = v.index_select(0, beam_idx)
            reordered.append((k, v))
        return reordered
