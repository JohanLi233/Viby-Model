from transformers.configuration_utils import PretrainedConfig
import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List
import torch.nn.functional as F
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
        hidden_size: int = 512,
        intermediate_size: int = 1792,
        max_position_embeddings: int = 32768,
        original_max_position_embeddings: int = 512,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        z_loss_factor: float = 0.0001,
        use_moe: bool = True,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        use_cache: bool = True,
        window_length: int = 512,
        swiglu_limit: float = 7.0,
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
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.window_length = window_length
        self.swiglu_limit = swiglu_limit
        self.use_cache = use_cache


def z_loss_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, z_loss_factor: float = 0.0
) -> torch.Tensor:
    """Cross entropy with Z-loss for training stability."""
    ce = F.cross_entropy(
        logits.flatten(end_dim=-2), labels.flatten(end_dim=-1), reduction="none"
    )

    if z_loss_factor > 0.0:
        z = torch.logsumexp(logits, dim=-1)
        z_term = z_loss_factor * (z.square())
        z_term = z_term.flatten()
        loss = (ce + z_term).mean()
    else:
        loss = ce.mean()
    return loss


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # Duplicate for rotating the whole head dim
    return torch.cat([freqs_cos, freqs_cos], dim=-1), torch.cat(
        [freqs_sin, freqs_sin], dim=-1
    )


# NEW: Re-implemented YaRN precomputation based on gpt_oss logic
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
    Precompute RoPE frequencies with YaRN scaling, adapted from gpt_oss.
    This version uses a smooth ramp for interpolation/extrapolation.
    """
    d_half = dim // 2
    freq = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)

    # YaRN concentration factor
    concentration = 0.1 * math.log(scaling_factor) + 1.0

    # NTK-by-parts boundary calculation
    low = (
        d_half
        * math.log(original_max_pos_embeds / (beta_fast * 2 * math.pi))
        / math.log(theta)
    )
    high = (
        d_half
        * math.log(original_max_pos_embeds / (beta_slow * 2 * math.pi))
        / math.log(theta)
    )
    assert 0 < low < high < d_half - 1

    # Positional Interpolation and NTK-aware scaling
    interpolation = 1.0 / (scaling_factor * freq)
    extrapolation = 1.0 / freq

    # Smooth ramp and mask for blending
    ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
    mask = 1 - ramp.clamp(0, 1)

    inv_freq = interpolation * (1 - mask) + extrapolation * mask

    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    # Apply concentration and prepare for rotation
    freqs_cos = freqs.cos() * concentration
    freqs_sin = freqs.sin() * concentration

    # Duplicate for rotating the whole head dim
    return torch.cat([freqs_cos, freqs_cos], dim=-1), torch.cat(
        [freqs_sin, freqs_sin], dim=-1
    )


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# MODIFIED: Attention module completely refactored to use learnable sinks
class Attention(nn.Module):
    def __init__(self, args: VibyConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
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

        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)

        # QK normalization layers
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Sliding window and learnable sink parameters from gpt_oss
        self.sliding_window = args.window_length
        self.sinks = nn.Parameter(
            torch.empty(self.n_local_heads)
        )  # One sink value per head
        nn.init.normal_(self.sinks)  # Initialize sinks

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Will be used to create the combined mask
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            # Concatenate with past keys/values for generation
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        key = repeat_kv(xk, self.n_rep)
        value = repeat_kv(xv, self.n_rep)

        # Get dimensions for attention calculation
        query_len, key_len = xq.size(1), key.size(1)

        # Transpose for batch matrix multiplication
        # (bsz, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Manual Attention Calculation with Learnable Sinks
        # 1. Compute QK scores
        scores = torch.matmul(xq, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 2. Create attention mask
        # Start with causal mask
        mask = torch.full((query_len, key_len), float("-inf"), device=scores.device)
        mask = torch.triu(mask, diagonal=1)

        # Add sliding window mask
        if self.sliding_window > 0:
            mask += torch.tril(
                mask.new_full((query_len, key_len), float("-inf")),
                diagonal=-self.sliding_window,
            )

        # Add padding mask if provided
        if attention_mask is not None:
            # Expected shape (bsz, 1, query_len, key_len)
            mask = mask.unsqueeze(0).unsqueeze(0) + attention_mask

        scores += mask

        # 3. Add learnable sinks
        # Reshape sinks to be concatenated with scores
        sinks_reshaped = self.sinks.reshape(1, self.n_local_heads, 1, 1).expand(
            bsz, -1, query_len, 1
        )
        scores = torch.cat([scores, sinks_reshaped], dim=-1)

        # 4. Softmax and Dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)

        # Drop the sink weights, they are only for normalization
        attn_weights = attn_weights[..., :-1]
        attn_weights = F.dropout(attn_weights, p=self.dropout if self.training else 0.0)

        # 5. Compute output
        output = torch.matmul(attn_weights, value)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


# MODIFIED: FeedForward with SwiGLU clamping
class FeedForward(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config  # Store config
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
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # NEW: Add clamping for stability, from gpt_oss
        if self.config.swiglu_limit > 0:
            limit = self.config.swiglu_limit
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)

        return self.dropout(self.down_proj(self.act_fn(gate) * up))


class MoEGate(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                FeedForward(config)  # Pass config to FeedForward
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    FeedForward(config)  # Pass config to FeedForward
                    for _ in range(config.n_shared_experts)
                ]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )
        return expert_cache


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            normed_hidden_states,
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states += residual

        return hidden_states, present_key_value


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        if config.rope_scaling and config.rope_scaling.get("type") == "yarn":
            scaling_factor = config.rope_scaling["factor"]
            # Dynamically calculate attention temperature based on scaling factor
            # Note: gpt_oss applies 'concentration' directly to cos/sin, which has a similar effect.
            # We keep this temp scaling for now as it's a valid technique.
            # config.attn_softmax_temp = 0.5 * math.log(scaling_factor) + 1.0 (This is from the old implementation, concentration in new RoPE replaces it)

            print(
                f"[YaRN] Enabled with scaling factor {scaling_factor} using gpt_oss logic."
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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[
                2
            ]  # Shape is (bsz, n_kv_heads, seq_len, head_dim)

        # Prepare attention mask for manual attention
        # This is for padding. Causal/sliding window mask is handled in Attention module.
        attn_mask_for_forward = None
        if attention_mask is not None:
            # HuggingFace masks are [bsz, seq_len] with 1 for non-padded, 0 for padded.
            # We need [bsz, 1, query_len, key_len] with 0 for non-padded, -inf for padded.
            attn_mask_for_forward = attention_mask[:, None, None, :].to(
                dtype=self.embed_tokens.weight.dtype
            )
            attn_mask_for_forward = (1.0 - attn_mask_for_forward) * torch.finfo(
                attn_mask_for_forward.dtype
            ).min
            if past_key_values_length > 0:
                # Expand mask to include past keys
                past_mask = torch.zeros(
                    (batch_size, 1, 1, past_key_values_length),
                    device=input_ids.device,
                    dtype=attn_mask_for_forward.dtype,
                )
                attn_mask_for_forward = torch.cat(
                    [past_mask, attn_mask_for_forward], dim=-1
                )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[
                past_key_values_length : past_key_values_length + seq_length
            ],
            self.freqs_sin[
                past_key_values_length : past_key_values_length + seq_length
            ],
        )

        presents = [] if use_cache else None
        for i, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values or [None] * len(self.layers))
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attn_mask_for_forward,
            )
            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class VibyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VibyConfig

    def __init__(self, config: VibyConfig):
        super().__init__(config)
        self.model = VibyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        hidden_states, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = z_loss_cross_entropy(
                shift_logits, shift_labels, self.config.z_loss_factor
            )
            if aux_loss > 0:
                loss += aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )
