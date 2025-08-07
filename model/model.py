import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin  # type: ignore
from transformers.modeling_outputs import CausalLMOutputWithPast


class VibyConfig(PretrainedConfig):
    model_type = "viby"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 640,
        intermediate_size: int = 640,
        max_position_embeddings: int = 32768,
        original_max_position_embeddings: int = 640,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 16,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        z_loss_factor: float = 0.0001,
        # --- New Features (MoE and SWA) ---
        # MoE configurations
        moe_layers: List[
            int
        ] = [],  # Indices of layers to be MoE (e.g., [1, 3, 5, ...])
        num_experts: int = 4,  # Total number of experts (e.g., 8). Set > 0 to enable MoE.
        num_experts_per_tok: int = 2,  # Number of experts selected per token (e.g., 2)
        moe_router_loss_weight: float = 0.01,  # Weight for auxiliary load balancing loss
        # Sliding Window Attention (SWA) / Sink Attention
        sliding_window: Optional[
            int
        ] = 512,  # Local attention window size (e.g., 4096)
        sink_size: int = 4,  # Number of initial tokens (sink) to always attend to
        # ------------------------------------
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.dropout = dropout
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

        # MoE parameters
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_router_loss_weight = moe_router_loss_weight

        # SWA parameters
        self.sliding_window = sliding_window
        self.sink_size = sink_size


# --- Loss Functions ---


def z_loss_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, z_loss_factor: float = 0.0
) -> torch.Tensor:
    """Cross entropy with Z-loss for training stability. Handles padding masks correctly."""
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )  # (B*T,)

    if z_loss_factor > 0.0:
        # z = logsumexp over vocab for each position
        z = torch.logsumexp(shift_logits, dim=-1)
        # square penalty
        z_term = z_loss_factor * (z.square())
        z_term = z_term.flatten()

        # We must mask the terms where labels are ignored
        mask = shift_labels.view(-1) != -100

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        ce = ce[mask]
        z_term = z_term[mask]

        # Calculate mean loss over valid tokens
        loss = (ce.sum() + z_term.sum()) / mask.sum()
    else:
        # If z_loss is off, standard mean CE (F.cross_entropy handles the averaging over valid tokens)
        loss = ce.mean()
    return loss


def load_balancing_loss_func(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Computes auxiliary load balancing loss for MoE models (Switch Transformers style).
    """
    if router_logits is None:
        return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

    # router_logits: (batch_size * seq_len, num_experts)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

    # Calculate expert utilization (f_i) using top-1 choice for stability (as per GShard/Switch)
    expert_indices = torch.argmax(router_probs, dim=-1)
    expert_mask = F.one_hot(expert_indices, num_classes=num_experts).to(torch.float)

    # Handle padding tokens if attention_mask is provided
    if attention_mask is not None:
        # attention_mask: (batch_size, seq_len) -> (batch_size * seq_len, 1)
        # We must ensure the mask matches the input_ids length corresponding to the router logits
        if attention_mask.shape[1] * attention_mask.shape[0] != router_logits.shape[0]:
            # If shapes don't match (e.g. labels were longer than input_ids), slice the mask
            expected_len = router_logits.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask[:, :expected_len]

        flat_mask = attention_mask.reshape(-1).unsqueeze(-1).to(torch.float)
        expert_mask = expert_mask * flat_mask
        tokens_per_expert = expert_mask.sum(dim=0)
        total_tokens = flat_mask.sum()
    else:
        tokens_per_expert = expert_mask.sum(dim=0)
        total_tokens = expert_mask.shape[0]

    if total_tokens == 0:
        return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

    # Fraction of tokens per expert (f)
    f = tokens_per_expert / total_tokens

    # Average probability allocated to each expert (P)
    if attention_mask is not None:
        router_probs_masked = router_probs * flat_mask
        P = router_probs_masked.sum(dim=0) / total_tokens
    else:
        P = router_probs.mean(dim=0)

    # Load balancing loss: L_aux = N * sum(f_i * P_i)
    loss = (f * P).sum() * num_experts
    return loss.to(router_logits.dtype)


# --- RoPE Embeddings ---
# (YaRN implementations optimized to return separate cos/sin)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Keep cos/sin separate for optimized application
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def find_correction_range(
    low_freq_factor: float,
    high_freq_factor: float,
    dim: int,
    base: float,
    original_max_pos_embeds: int,
) -> torch.Tensor:
    # (Implementation remains the same)
    # Note: The 'dim' here usually refers to the full head dimension when calculating inv_freq base
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
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    correction_range = find_correction_range(
        beta_slow, beta_fast, dim, theta, original_max_pos_embeds
    ).to(t.device)

    t_pi = t / scaling_factor
    ntk_scaling_factor = (
        scaling_factor ** (dim / (dim - 2)) if dim > 2 else scaling_factor
    )
    t_ntk = t / ntk_scaling_factor

    corrected_t = (
        t_pi[:, None] * (1 - correction_range) + t_ntk[:, None] * correction_range
    )

    freqs = corrected_t * freqs

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embeddings (RoPE) efficiently using slicing (Optimized).
    xq, xk shape: (bsz, seq_len, num_heads, head_dim)
    cos, sin shape: (seq_len, head_dim // 2)
    """

    # Reshape cos/sin for broadcasting with (B, T, H, D/2): (1, T, 1, D/2)
    # Insert batch dim first, then head dim to match xq/xk (..., H, D/2)
    cos = cos.unsqueeze(0).unsqueeze(2).contiguous()
    sin = sin.unsqueeze(0).unsqueeze(2).contiguous()

    # Separate even and odd indices
    xq_even = xq[..., ::2]
    xq_odd = xq[..., 1::2]
    xk_even = xk[..., ::2]
    xk_odd = xk[..., 1::2]

    # Apply rotation: Real = (x_even * cos) - (x_odd * sin); Imag = (x_odd * cos) + (x_even * sin)
    xq_rotated_even = (xq_even * cos) - (xq_odd * sin)
    xq_rotated_odd = (xq_odd * cos) + (xq_even * sin)
    xk_rotated_even = (xk_even * cos) - (xk_odd * sin)
    xk_rotated_odd = (xk_odd * cos) + (xk_even * sin)

    # Interleave back together
    xq_embed = torch.stack([xq_rotated_even, xq_rotated_odd], dim=-1).flatten(-2)
    xk_embed = torch.stack([xk_rotated_even, xk_rotated_odd], dim=-1).flatten(-2)

    return xq_embed, xk_embed


# --- Attention Mechanisms (SWA/Sink) ---


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=2)


def _make_sliding_window_and_sink_mask(
    bsz: int,
    tgt_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int,
    sink_size: int,
) -> torch.Tensor:
    """
    Creates a causal mask combined with Sliding Window Attention (SWA) and Sink Attention.
    Used during training/prefill (tgt_len > 1).
    """
    seq_len = past_len + tgt_len

    # 1. Initialize boolean mask (True means attend)
    attn_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    # 2. Causal Mask (Lower Triangular)
    causal_mask = torch.tril(attn_mask, diagonal=0)

    # 3. Sliding Window Mask
    if seq_len > sliding_window:
        # triu(diagonal=-(sliding_window-1)) ensures attention is only allowed within the window
        band_mask = torch.triu(causal_mask, diagonal=-(sliding_window - 1))
    else:
        band_mask = causal_mask

    # 4. Sink Attention (Always attend to the first sink_size tokens)
    if sink_size > 0:
        band_mask[:, :sink_size] = True

    # Combine Causal and Band/Sink mask (Ensure causality is always respected)
    final_mask_bool = causal_mask & band_mask

    # 5. Convert to SDPA format (0 for attend, -inf for don't attend)
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    mask.masked_fill_(~final_mask_bool, torch.finfo(dtype).min)

    # Select the relevant part of the mask for the current step
    # Queries: [past_len : past_len + tgt_len], Keys: [0 : seq_len]
    mask = mask[past_len : past_len + tgt_len, :seq_len]

    # Expand mask for SDPA format: (bsz, 1, tgt_len, seq_len)
    return mask.expand(bsz, 1, tgt_len, seq_len)


class Attention(nn.Module):
    def __init__(self, args: VibyConfig):
        super().__init__()
        # (GQA initialization remains the same)
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

        # (Projections remain the same)
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

        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout_p = args.dropout
        self.attn_softmax_temp = args.attn_softmax_temp

        # QK normalization layers (like Qwen/Gemma)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # SWA/Sink configuration
        self.sliding_window = args.sliding_window
        self.sink_size = args.sink_size

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Expected (B, 1, T_q, T_k) SDPA format
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # QK Norm
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV Cache Management
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[1]
            # Append current k, v to the cache
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # Optimization: Trim the KV cache during generation if SWA is enabled
        if use_cache and self.sliding_window is not None:
            # Keep sink + sliding window
            cache_keep_len = self.sliding_window + self.sink_size
            # Trim if the cache exceeds the maximum length
            if xk.shape[1] > cache_keep_len:
                xk = xk[:, -cache_keep_len:]
                xv = xv[:, -cache_keep_len:]

        past_kv = (xk, xv) if use_cache else None

        # GQA: Repeat key and value
        key = repeat_kv(xk, self.n_rep)
        value = repeat_kv(xv, self.n_rep)

        # Transpose for SDPA: (bsz, num_heads, seq_len_q/k, head_dim)
        xq = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # --- Attention Mask Handling (SWA/Sink Integration) ---
        final_attn_mask = None
        is_causal = False

        # Condition 1: Sliding Window Attention enabled (during training/prefill, seq_len > 1)
        if self.sliding_window is not None and seq_len > 1:
            # Generate the SWA/Sink mask.
            final_attn_mask = _make_sliding_window_and_sink_mask(
                bsz,
                seq_len,
                past_len,
                x.dtype,
                x.device,
                self.sliding_window,
                self.sink_size,
            )
            # If an input attention_mask (for padding, already in SDPA format) was provided, combine them
            if attention_mask is not None:
                # SDPA masks are additive (0 + -inf = -inf)
                final_attn_mask = final_attn_mask + attention_mask

        # Condition 2: Input attention mask provided (e.g., for padding during generation)
        elif attention_mask is not None:
            final_attn_mask = attention_mask

        # Condition 3: Dense Causal Attention
        else:
            # If no mask is provided and SWA is off.
            # is_causal=True only works if T_q = T_k (past_len=0) and no mask is provided.
            if past_len == 0 and seq_len > 1 and final_attn_mask is None:
                is_causal = True

        # ----------------------------------------------------

        dropout_p = self.dropout_p if self.training else 0.0

        # Apply temperature scaling (used by YaRN)
        if self.attn_softmax_temp != 1.0:
            xq = xq / self.attn_softmax_temp

        # Efficient Attention (Flash Attention via SDPA)
        # Use a context manager to enable Flash Attention if available
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            output = F.scaled_dot_product_attention(
                xq,
                key,
                value,
                attn_mask=final_attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


# --- FeedForward and MoE ---


class FeedForward(nn.Module):
    """Standard SwiGLU MLP (Also serves as the Expert in MoE)."""

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class MoEBlock(nn.Module):
    """
    Mixture of Experts (MoE) block. Replaces FeedForward in selected layers.
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        if self.num_experts == 0:
            raise ValueError("MoEBlock initialized but num_experts is 0.")

        # Gating mechanism (Router)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Reshape for routing: (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_size)

        # Compute router logits
        router_logits = self.gate(hidden_states)

        # Top-k Gating (use float32 for stability)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Normalize the weights for the selected experts
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # Convert back to model dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Initialize output tensor
        final_hidden_states = torch.zeros_like(hidden_states)

        # Efficiently route tokens to experts (iterate over experts, not tokens)
        # Create masks: (num_experts, top_k, batch_size * seq_len)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # (top_k, batch_size * seq_len)
            mask = expert_mask[expert_idx]

            # Find the indices of tokens routed to this expert
            # top_k_indices: the rank (0 to top_k-1)
            # idx: indices in the flattened input tensor
            top_k_indices, idx = torch.where(mask)

            if idx.shape[0] == 0:
                continue

            # Select the inputs and the corresponding weights
            current_state = hidden_states[idx]
            # (num_routed_tokens, 1)
            current_weight = routing_weights[idx, top_k_indices].unsqueeze(-1)

            # Process through the expert
            expert_output = expert_layer(current_state)

            # Weight the output
            weighted_output = expert_output * current_weight

            # Accumulate the results using index_add_
            final_hidden_states.index_add_(0, idx, weighted_output)

        # Reshape back
        final_hidden_states = final_hidden_states.reshape(
            batch_size, seq_len, hidden_size
        )

        # Return the output and the router logits (for auxiliary loss)
        return final_hidden_states, router_logits


# --- Transformer Block ---


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig, is_moe: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Determine if this block uses MLP or MoE
        self.is_moe = is_moe
        if self.is_moe:
            self.mlp = MoEBlock(config)
        else:
            self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:

        # Attention (Pre-Norm)
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            normed_hidden_states,
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + attn_output

        # MLP/MoE (Pre-Norm)
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = None
        if self.is_moe:
            mlp_output, router_logits = self.mlp(normed_hidden_states)
        else:
            mlp_output = self.mlp(normed_hidden_states)

        hidden_states = residual + mlp_output

        return hidden_states, present_key_value, router_logits


# --- Main Model ---


class VibyModel(PreTrainedModel):
    config_class = VibyConfig

    def __init__(self, config: VibyConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        # Initialize RoPE (including YaRN scaling if configured)
        self._init_rope(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize layers, determining MoE vs Dense
        self.layers = nn.ModuleList()
        for l in range(self.num_hidden_layers):
            is_moe = (config.num_experts > 0) and (l in config.moe_layers)
            self.layers.append(VibyBlock(l, config, is_moe=is_moe))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_rope(self, config: VibyConfig):
        head_dim = config.hidden_size // config.num_attention_heads

        if config.rope_scaling and config.rope_scaling.get("type") == "yarn":
            scaling_factor = config.rope_scaling["factor"]

            # Dynamically calculate attention temperature (YaRN mscale stabilization)
            # Formula from the YaRN paper Section 3.4
            config.attn_softmax_temp = 0.1 * math.log(scaling_factor) + 1.0
            print(
                f"[YaRN] Enabled with scaling factor {scaling_factor}. New attention temp (mscale): {config.attn_softmax_temp:.3f}"
            )

            freqs_cos, freqs_sin = precompute_freqs_cis_yarn(
                dim=head_dim,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
                scaling_factor=scaling_factor,
                original_max_pos_embeds=config.original_max_position_embeddings,
                beta_fast=config.rope_scaling.get("beta_fast", 32.0),
                beta_slow=config.rope_scaling.get("beta_slow", 1.0),
            )
        else:
            # Fallback to original RoPE
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=head_dim,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _prepare_sdpa_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # Prepare mask for SDPA (B, H, T_q, T_k) format from a standard (B, T) mask.
        # Handles padding masks. Causal/SWA masking is handled within the Attention module.

        if attention_mask is not None:
            bsz, tgt_len = input_shape
            src_len = tgt_len + past_key_values_length

            # Ensure the mask covers the full sequence length (T_k)
            if attention_mask.shape[1] < src_len:
                # If mask only covers current input, expand it assuming past tokens are attended
                if attention_mask.shape[1] == tgt_len and past_key_values_length > 0:
                    past_mask = torch.ones(
                        (bsz, past_key_values_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([past_mask, attention_mask], dim=1)
                else:
                    # If still short, pad with zeros (mask)
                    pad_len = src_len - attention_mask.shape[1]
                    pad_mask = torch.zeros(
                        (bsz, pad_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
            elif attention_mask.shape[1] > src_len:
                attention_mask = attention_mask[:, :src_len]

            # Expand (B, T_k) -> (B, 1, T_q, T_k)
            # We want the mask to apply to the target length queries.
            expanded_mask = attention_mask[:, None, None, :].expand(
                bsz, 1, tgt_len, src_len
            )

            # Convert 1s (attend) to 0s, and 0s (mask) to -inf (SDPA format)
            mask = torch.zeros_like(expanded_mask, dtype=inputs_embeds.dtype)
            mask.masked_fill_(expanded_mask == 0, torch.finfo(inputs_embeds.dtype).min)
            return mask
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # (B, T) bool/int mask (1=attend, 0=ignore)
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Tuple], List[torch.Tensor]]:
        batch_size, seq_length = input_ids.shape

        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            start_pos = 0
        else:
            # Determine the length of the context processed so far based on the cache length.
            # Note: If using SWA with KV trimming, this start_pos might underestimate the true position.
            # For full robustness with SWA, position_ids should ideally be passed externally.
            start_pos = (
                past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
            )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # RoPE selection based on position
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length].contiguous(),
            self.freqs_sin[start_pos : start_pos + seq_length].contiguous(),
        )

        # Prepare attention mask for SDPA format (handles padding)
        # The Attention module will combine this with the SWA/Sink mask if applicable.
        sdpa_attention_mask = self._prepare_sdpa_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, start_pos
        )

        presents = []
        all_router_logits = []

        # Transformer blocks forward pass
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present, router_logits = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=sdpa_attention_mask,
            )
            if use_cache:
                presents.append(present)
            if router_logits is not None:
                # Collect router logits for aux loss
                all_router_logits.append(router_logits)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents, all_router_logits


class VibyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VibyConfig
    _supports_cache_class = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Optional[VibyConfig] = None):
        config = config or VibyConfig()
        super().__init__(config)
        self.model = VibyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and tie them (handled by post_init in PreTrainedModel)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Standard preparation for generation (handling KV cache)
        if past_key_values is not None:
            # Only pass the last token if we have a cache
            input_ids = input_ids[:, -1:]

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # BaseModel forward pass
        hidden_states, past_kvs, all_router_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # LM Head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 1. Calculate Causal LM Loss (with Z-loss)
            lm_loss = z_loss_cross_entropy(logits, labels, self.config.z_loss_factor)
            loss = lm_loss

            # 2. Calculate MoE Auxiliary Load Balancing Loss
            # Only calculate during training if MoE layers exist
            if len(all_router_logits) > 0 and self.training:
                aux_loss = 0.0

                # We need an attention mask corresponding to the input_ids length
                # to correctly calculate utilization without padding tokens.
                current_attention_mask = attention_mask

                # If attention_mask is missing, try deriving it from labels (assuming -100 is padding)
                if current_attention_mask is None and labels is not None:
                    # Create mask matching input_ids length
                    current_attention_mask = (
                        labels[:, : input_ids.shape[1]] != -100
                    ).long()

                # Calculate loss for each MoE layer
                for router_logits in all_router_logits:
                    # The router_logits correspond to the input tokens processed in this step (B*T, N_experts)

                    layer_aux_loss = load_balancing_loss_func(
                        router_logits,
                        self.config.num_experts,
                        self.config.num_experts_per_tok,
                        current_attention_mask,
                    )
                    aux_loss += layer_aux_loss

                # Average the auxiliary loss across MoE layers
                if len(all_router_logits) > 0:
                    aux_loss = aux_loss / len(all_router_logits)

                # Add weighted auxiliary loss to the main loss
                loss += self.config.moe_router_loss_weight * aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )
