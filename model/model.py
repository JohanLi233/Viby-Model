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
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        # Sink attention parameters
        num_sink_tokens: int = 4,
        window_length: int = 512,
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
        # Sink attention config
        self.num_sink_tokens = num_sink_tokens
        self.window_length = window_length


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

        # QK normalization layers (like Qwen3)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        
        # Sink attention parameters
        self.num_sink_tokens = getattr(args, 'num_sink_tokens', 4)
        self.window_length = getattr(args, 'window_length', 512)
        
        # Sink tokens cache for each head
        self.sink_cache = {}  # Will store sink key/value pairs
        self.window_cache = {}  # Will store recent window key/value pairs
        self.cache_initialized = False

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
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

        # Enhanced sink attention cache management
        if use_cache:
            xk, xv = self._update_sink_cache(xk, xv, past_key_value)
        elif past_key_value is not None:
            # Standard KV cache concatenation
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

        final_attn_mask = None
        # is_causal 用于在训练 prefill 阶段自动生成上三角掩码
        # 当提供了 attention_mask (例如处理padding)，需要手动构建掩码
        is_causal = attention_mask is None
        if attention_mask is not None:
            final_attn_mask = attention_mask.to(torch.bool)
        # ----------------------------------------------------

        dropout_p = self.dropout if self.training else 0.0

        # Apply temperature scaling to the query for YaRN
        if self.attn_softmax_temp != 1.0:
            xq = xq / self.attn_softmax_temp

        output = F.scaled_dot_product_attention(
            xq,
            key,
            value,
            attn_mask=final_attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,  # 当没有提供 mask 时，自动应用因果掩码
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
    
    def _update_sink_cache(self, xk, xv, past_key_value):
        """Update sink attention cache with new keys and values"""
        if past_key_value is not None:
            # Concatenate with past keys/values
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        seq_len = xk.size(1)
        
        # If sequence is shorter than sink tokens, return as-is
        if seq_len <= self.num_sink_tokens:
            return xk, xv
        
        # Separate sink tokens and sliding window tokens
        sink_k = xk[:, :self.num_sink_tokens]  # First few tokens as sink
        sink_v = xv[:, :self.num_sink_tokens]
        
        # Keep recent tokens within window length
        if seq_len > self.num_sink_tokens + self.window_length:
            # Keep sink + recent window
            recent_start = seq_len - self.window_length
            recent_k = xk[:, recent_start:]
            recent_v = xv[:, recent_start:]
            
            # Combine sink and recent tokens
            xk = torch.cat([sink_k, recent_k], dim=1)
            xv = torch.cat([sink_v, recent_v], dim=1)
        
        return xk, xv
    
    def reset_cache(self):
        """Reset sink attention cache"""
        self.sink_cache = {}
        self.window_cache = {}
        self.cache_initialized = False


class FeedForward(nn.Module):
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
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


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
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

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
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
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
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

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
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
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
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
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
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
    
    def reset_cache(self):
        """Reset attention cache for this block"""
        if hasattr(self.self_attn, 'reset_cache'):
            self.self_attn.reset_cache()


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
            [VibyBlock(l, config) for l in range(self.num_hidden_layers)]
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
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss
    
    def reset_cache(self):
        """Reset all attention caches"""
        for layer in self.layers:
            if hasattr(layer, 'reset_cache'):
                layer.reset_cache()


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

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,  # 增加了 labels 以便计算 loss
        **args,
    ):
        hidden_states, past_kvs, aux_loss = self.model(
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
            if aux_loss > 0:
                loss = loss + aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
        )
    
    def reset_cache(self):
        """Reset all sink attention caches"""
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
