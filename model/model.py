import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class VibyConfig(PretrainedConfig):
    model_type = "viby"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
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
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }

        self.z_loss_factor = kwargs.get("z_loss_factor", 0.0)

        # MoE specific configs, ignored when use_moe is False.
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.router_scoring_func = kwargs.get("router_scoring_func", "softmax")
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.swiglu_limit = kwargs.get("swiglu_limit", None)
        self.moe_packed_experts = kwargs.get("moe_packed_experts", False)
        self.moe_sorted_dispatch = kwargs.get("moe_sorted_dispatch", False)
        self.n_shared_experts = kwargs.get("n_shared_experts", 0)
        self.first_k_dense_replace = kwargs.get("first_k_dense_replace", 0)
        self.use_deepseek_v4_attention = kwargs.get("use_deepseek_v4_attention", False)
        self.q_lora_rank = kwargs.get("q_lora_rank", self.hidden_size)
        self.o_groups = kwargs.get("o_groups", 4)
        self.o_lora_rank = kwargs.get(
            "o_lora_rank", max(1, (self.hidden_size * 5) // 8)
        )
        self.attention_sink = kwargs.get("attention_sink", False)
        self.use_mhc = kwargs.get("use_mhc", False)
        self.hc_mult = kwargs.get("hc_mult", 1)
        self.hc_sinkhorn_iters = kwargs.get("hc_sinkhorn_iters", 1)
        self.hc_eps = kwargs.get("hc_eps", 1e-4)
        self.mtp_depth = kwargs.get("mtp_depth", 0)
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight", 0.3)

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.hidden_size % self.o_groups != 0:
            raise ValueError("hidden_size must be divisible by o_groups")
        if self.hc_mult != 1:
            raise ValueError("hc_mult > 1 requires widened residual streams and is not enabled in this implementation yet")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            inv_dim = lambda b: (  # noqa: E731
                dim * math.log(orig_max / (b * 2 * math.pi))
            ) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)
    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


def z_loss_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    z_loss_factor: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.flatten(end_dim=-2),
        labels.flatten(end_dim=-1),
        ignore_index=-100,
        reduction="none",
    )

    if z_loss_factor > 0.0:
        z_term = z_loss_factor * torch.logsumexp(logits, dim=-1).square()
        ce = ce + z_term.flatten()

    if mask is None:
        return ce.mean()

    mask_flat = mask.flatten().to(ce.dtype)
    return (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)


class GroupedLowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, groups: int):
        super().__init__()
        if in_features % groups != 0 or out_features % groups != 0:
            raise ValueError("in_features and out_features must be divisible by groups")
        self.groups = groups
        self.in_per_group = in_features // groups
        self.out_per_group = out_features // groups
        self.rank = rank
        self.a_proj = nn.Parameter(torch.empty(groups, rank, self.in_per_group))
        self.b_proj = nn.Parameter(torch.empty(groups, self.out_per_group, rank))
        self.reset_parameters()

    def reset_parameters(self):
        for group_id in range(self.groups):
            nn.init.kaiming_uniform_(self.a_proj[group_id], a=math.sqrt(5))
            nn.init.zeros_(self.b_proj[group_id])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.groups, self.in_per_group)
        hidden = torch.einsum("bgi,gri->bgr", x, self.a_proj)
        out = torch.einsum("bgr,gor->bgo", hidden, self.b_proj)
        return out.reshape(*original_shape, self.groups * self.out_per_group)


class HyperConnection(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.num_residuals = config.hc_mult
        self.hidden_size = config.hidden_size
        self.eps = config.hc_eps
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        if self.num_residuals <= 1:
            self.static_beta = None
            self.dynamic_alpha_fn = None
            return
        self.static_beta = nn.Parameter(torch.ones(self.num_residuals))
        self.dynamic_alpha_fn = nn.Linear(
            config.hidden_size,
            self.num_residuals * self.num_residuals,
            bias=False,
        )

    def _normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.float()
        for _ in range(self.sinkhorn_iters):
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            alpha = alpha / alpha.sum(dim=-2, keepdim=True).clamp_min(self.eps)
        return alpha

    def width_connection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_residuals <= 1:
            return hidden_states
        residuals = hidden_states.view(
            *hidden_states.shape[:-1], self.num_residuals, self.hidden_size
        )
        alpha = self.dynamic_alpha_fn(residuals.mean(dim=-2))
        alpha = alpha.view(*hidden_states.shape[:-1], self.num_residuals, self.num_residuals)
        alpha = self._normalize_alpha(alpha).to(hidden_states.dtype)
        mixed = torch.einsum("...ij,...jh->...ih", alpha, residuals)
        return mixed.reshape_as(hidden_states)

    def depth_connection(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_residuals <= 1:
            return residual + update
        residual = residual.view(*residual.shape[:-1], self.num_residuals, self.hidden_size)
        update = update.view(*update.shape[:-1], self.num_residuals, self.hidden_size)
        beta = self.static_beta.view(*([1] * (residual.ndim - 2)), self.num_residuals, 1)
        return (residual + beta.to(update.dtype) * update).reshape(*update.shape[:-2], -1)


class Attention(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.use_deepseek_v4_attention = config.use_deepseek_v4_attention
        self.num_key_value_heads = (
            1
            if self.use_deepseek_v4_attention
            else config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        if self.use_deepseek_v4_attention:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )
            self.kv_proj = nn.Linear(
                config.hidden_size,
                2 * self.head_dim,
                bias=False,
            )
            self.o_proj = GroupedLowRankLinear(
                config.num_attention_heads * self.head_dim,
                config.hidden_size,
                rank=config.o_lora_rank,
                groups=config.o_groups,
            )
            self.attn_sinks = (
                nn.Parameter(torch.zeros(config.num_attention_heads))
                if config.attention_sink
                else None
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
            )
            self.k_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
            )
            self.attn_sinks = None
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn
        self._collect_attention_stats = False

    def enable_attention_stats_collection(self):
        self._collect_attention_stats = True

    def disable_attention_stats_collection(self):
        self._collect_attention_stats = False

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        if self.use_deepseek_v4_attention:
            xq = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
            xk, xv = self.kv_proj(x).chunk(2, dim=-1)
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, 1, self.head_dim)
            xv = xv.view(bsz, seq_len, 1, self.head_dim)
        else:
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        if xk.size(1) == 1 and xq.size(1) != 1:
            xk = xk.expand(-1, xq.size(1), -1, -1)
            xv = xv.expand(-1, xq.size(1), -1, -1)

        full_attention = attention_mask is None or torch.all(attention_mask == 1)
        if (
            self.flash
            and seq_len > 1
            and (not self.is_causal or past_key_value is None)
            and full_attention
            and not self._collect_attention_stats
            and self.attn_sinks is None
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if layer_idx is not None and self._collect_attention_stats:
                max_logits = torch.max(
                    scores.view(bsz, self.n_local_heads, -1).to(torch.float32),
                    dim=-1,
                )[0]
                self._current_attention_logits = {
                    "layer_idx": layer_idx,
                    "max_logits": torch.max(max_logits, dim=0)[0],
                }

            if self.is_causal:
                causal_mask = torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                ).triu(1)
                scores[:, :, :, -seq_len:] += causal_mask

            if attention_mask is not None:
                key_len = scores.size(-1)
                if attention_mask.dim() == 2:
                    am = attention_mask
                    if am.size(1) < key_len:
                        pad = torch.ones(
                            am.size(0),
                            key_len - am.size(1),
                            dtype=am.dtype,
                            device=am.device,
                        )
                        am = torch.cat([am, pad], dim=1)
                    elif am.size(1) > key_len:
                        am = am[:, -key_len:]
                    scores += (1.0 - am[:, None, None, :].to(scores.dtype)) * -1e9
                else:
                    scores += attention_mask.to(scores.dtype)

            if self.attn_sinks is not None:
                sink_logits = self.attn_sinks.view(1, self.n_local_heads, 1, 1)
                sink_logits = sink_logits.to(scores.dtype).expand(
                    bsz, -1, scores.size(-2), 1
                )
                scores = torch.cat([scores, sink_logits], dim=-1)

            attn_weights = self.attn_dropout(
                F.softmax(scores.float(), dim=-1).type_as(xq)
            )
            if self.attn_sinks is not None:
                attn_weights = attn_weights[..., :-1]
            output = attn_weights @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: VibyConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        if config.moe_packed_experts:
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                )
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
            )
            self.experts = None
        else:
            self.register_parameter("gate_up_proj", None)
            self.register_parameter("down_proj", None)
            self.experts = nn.ModuleList(
                [
                    FeedForward(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(config.num_experts)
                ]
            )
        self.shared_experts = (
            FeedForward(
                config,
                intermediate_size=config.n_shared_experts * config.moe_intermediate_size,
            )
            if config.n_shared_experts > 0
            else None
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.aux_loss = torch.tensor(0.0)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        if not self.config.moe_packed_experts:
            return
        for expert_id in range(self.config.num_experts):
            nn.init.kaiming_uniform_(self.gate_up_proj[expert_id], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.down_proj[expert_id], a=math.sqrt(5))

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.router_scoring_func == "sqrtsoftplus":
            return torch.sqrt(F.softplus(logits.float())).to(logits.dtype)
        if self.config.router_scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        if self.config.router_scoring_func == "softmax":
            return F.softmax(logits, dim=-1)
        raise ValueError(f"Unsupported router_scoring_func: {self.config.router_scoring_func}")

    def _apply_expert_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.config.swiglu_limit is not None and self.config.swiglu_limit > 0:
            gate = gate.clamp(max=self.config.swiglu_limit)
            up = up.clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
        return self.act_fn(gate) * up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        logits = self.gate(x_flat)
        scores = self._score(logits)
        topk_idx = torch.topk(
            scores + self.e_score_correction_bias,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        ).indices
        topk_weight = scores.gather(1, topk_idx)

        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.config.routed_scaling_factor

        if (
            self.config.moe_sorted_dispatch
            and self.config.num_experts_per_tok == 1
        ):
            y = self._forward_top1_sorted(x_flat, topk_idx[:, 0], topk_weight[:, 0])
        else:
            y = self._forward_mask_dispatch(x_flat, topk_idx, topk_weight)

        if self.training and self.config.router_aux_loss_coef > 0:
            load = torch.bincount(
                topk_idx.flatten(),
                minlength=self.config.num_experts,
            ).to(scores.dtype)
            load = load / topk_idx.numel()
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = scores.new_zeros(())

        y = y.view(batch_size, seq_len, hidden_dim)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y

    def _run_expert(self, expert_id: int, selected: torch.Tensor) -> torch.Tensor:
        if self.config.moe_packed_experts:
            gate_up = F.linear(selected, self.gate_up_proj[expert_id])
            return F.linear(
                self._apply_expert_gate(gate_up),
                self.down_proj[expert_id],
            )
        return self.experts[expert_id](selected)

    def _forward_top1_sorted(
        self,
        x_flat: torch.Tensor,
        expert_idx: torch.Tensor,
        expert_weight: torch.Tensor,
    ) -> torch.Tensor:
        sorted_experts, order = torch.sort(expert_idx)
        sorted_x = x_flat.index_select(0, order)
        counts = torch.bincount(sorted_experts, minlength=self.config.num_experts)
        y = torch.zeros_like(x_flat)
        start = 0
        for i in range(self.config.num_experts):
            count = counts[i]
            end = start + count
            if count > 0:
                out = self._run_expert(i, sorted_x[start:end])
                weight = expert_weight.index_select(0, order[start:end]).unsqueeze(-1)
                y.index_copy_(0, order[start:end], (out * weight).to(y.dtype))
            elif self.training:
                if self.config.moe_packed_experts:
                    y[0, 0] += 0 * (
                        self.gate_up_proj[i].sum() + self.down_proj[i].sum()
                    )
                else:
                    y[0, 0] += 0 * sum(p.sum() for p in self.experts[i].parameters())
            start = end
        return y

    def _forward_mask_dispatch(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.zeros_like(x_flat)
        for i in range(self.config.num_experts):
            mask = topk_idx == i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                topk_pos, gather_idx = torch.where(mask[token_idx])
                selected = x_flat.index_select(0, token_idx)
                expert_out = self._run_expert(i, selected)
                weight = topk_weight[mask].view(-1, 1)
                if self.config.num_experts_per_tok == 1:
                    y.index_add_(0, token_idx, (expert_out * weight).to(y.dtype))
                else:
                    y.index_add_(
                        0,
                        token_idx.index_select(0, gather_idx),
                        (expert_out.index_select(0, topk_pos) * weight).to(y.dtype),
                    )
            elif self.training:
                if self.config.moe_packed_experts:
                    y[0, 0] += 0 * (
                        self.gate_up_proj[i].sum() + self.down_proj[i].sum()
                    )
                else:
                    y[0, 0] += 0 * sum(p.sum() for p in self.experts[i].parameters())
        return y


class VibyBlock(nn.Module):
    def __init__(self, layer_id: int, config: VibyConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_hyper_connection = HyperConnection(config) if config.use_mhc else None
        self.post_attention_hyper_connection = (
            HyperConnection(config) if config.use_mhc else None
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        use_moe = config.use_moe and layer_id >= config.first_k_dense_replace
        self.mlp = FeedForward(config) if not use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        attn_input = (
            self.input_hyper_connection.width_connection(hidden_states)
            if self.input_hyper_connection is not None
            else hidden_states
        )
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(attn_input),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
        )
        hidden_states = (
            self.input_hyper_connection.depth_connection(residual, hidden_states)
            if self.input_hyper_connection is not None
            else hidden_states + residual
        )

        residual = hidden_states
        mlp_input = (
            self.post_attention_hyper_connection.width_connection(hidden_states)
            if self.post_attention_hyper_connection is not None
            else hidden_states
        )
        mlp_output = self.mlp(
            self.post_attention_layernorm(mlp_input)
        )
        hidden_states = (
            self.post_attention_hyper_connection.depth_connection(residual, mlp_output)
            if self.post_attention_hyper_connection is not None
            else residual + mlp_output
        )
        return hidden_states, present_key_value


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [VibyBlock(layer, config) for layer in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _ensure_rope(self, hidden_states: torch.Tensor):
        if self.freqs_cos.device != hidden_states.device:
            self.freqs_cos = self.freqs_cos.to(hidden_states.device)
            self.freqs_sin = self.freqs_sin.to(hidden_states.device)
        if (
            self.freqs_cos.dtype != hidden_states.dtype
            and hidden_states.dtype.is_floating_point
        ):
            self.freqs_cos = self.freqs_cos.to(hidden_states.dtype)
            self.freqs_sin = self.freqs_sin.to(hidden_states.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, list, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        self._ensure_rope(hidden_states)
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
                layer_idx=layer_idx,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            (
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(layer.mlp, MOEFeedForward)
            ),
            hidden_states.new_zeros(()),
        )
        return hidden_states, presents, aux_loss


class VibyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VibyConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Optional[VibyConfig] = None):
        config = config or VibyConfig()
        super().__init__(config)
        self.config = config
        self.model = VibyModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.register_buffer(
            "attention_max_logits",
            torch.zeros(
                (config.num_hidden_layers, config.num_attention_heads),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.post_init()

    def enable_attention_stats_collection(self):
        self.attention_logit_stats = {}
        for layer in self.model.layers:
            layer.self_attn.enable_attention_stats_collection()

    def disable_attention_stats_collection(self):
        for layer in self.model.layers:
            layer.self_attn.disable_attention_stats_collection()

    def get_attention_stats(self):
        if not hasattr(self, "attention_logit_stats"):
            self.attention_logit_stats = {}
        return self.attention_logit_stats

    def update_attention_stats_from_forward(self):
        current_batch_max_logits = []
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            if hasattr(attention, "_current_attention_logits"):
                logit_info = attention._current_attention_logits
                if logit_info["layer_idx"] == layer_idx:
                    current_batch_max_logits.append(
                        logit_info["max_logits"].detach().to(torch.float32)
                    )

        if current_batch_max_logits:
            batch_stats = torch.stack(current_batch_max_logits, dim=0).to(
                self.attention_max_logits.device
            )
            self.attention_max_logits.data = torch.maximum(
                self.attention_max_logits.data, batch_stats
            )

            if not hasattr(self, "attention_logit_stats"):
                self.attention_logit_stats = {}
            for layer_idx, max_logits in enumerate(current_batch_max_logits):
                layer_key = f"layer_{layer_idx}"
                self.attention_logit_stats[layer_key] = {
                    f"head_{head_idx}": max_logits[head_idx].item()
                    for head_idx in range(len(max_logits))
                }

        for layer in self.model.layers:
            if hasattr(layer.self_attn, "_current_attention_logits"):
                delattr(layer.self_attn, "_current_attention_logits")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else logits_to_keep
            if not isinstance(logits_to_keep, int)
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = z_loss_cross_entropy(
                logits,
                labels,
                self.config.z_loss_factor,
                mask=loss_mask,
            )
            if self.config.mtp_depth > 0:
                mtp_loss = logits.new_zeros(())
                mtp_terms = 0
                for offset in range(1, self.config.mtp_depth + 1):
                    if logits.size(1) <= offset:
                        break
                    mtp_logits = logits[:, :-offset, :]
                    mtp_labels = labels[:, offset:]
                    mtp_mask = loss_mask[:, offset:] if loss_mask is not None else None
                    mtp_loss = mtp_loss + z_loss_cross_entropy(
                        mtp_logits,
                        mtp_labels,
                        self.config.z_loss_factor,
                        mask=mtp_mask,
                    )
                    mtp_terms += 1
                if mtp_terms > 0:
                    loss = loss + self.config.mtp_loss_weight * (mtp_loss / mtp_terms)
            if self.config.use_moe:
                loss = loss + aux_loss

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Union[list, tuple]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values, beam_idx: torch.LongTensor):
        if past_key_values is None:
            return None
        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
            else:
                reordered.append(
                    tuple(
                        past_state.index_select(0, beam_idx)
                        for past_state in layer_past
                    )
                )
        return reordered

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        if input_ids is None:
            raise ValueError("input_ids or inputs must be provided")
        input_ids = input_ids.repeat(num_return_sequences, 1)
        attention_mask = (
            attention_mask.repeat(num_return_sequences, 1)
            if attention_mask is not None
            else None
        )
        past_key_values = kwargs.pop("past_key_values", None)
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )

        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            current_input_ids = (
                input_ids[:, past_len:]
                if past_len < input_ids.shape[1]
                else input_ids[:, -1:]
            )
            outputs = self.forward(
                current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(attention_mask.shape[0], 1),
                    ],
                    dim=-1,
                )

            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float(
                    "inf"
                )
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = (
                    torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                )
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = (
                torch.multinomial(probs, num_samples=1)
                if do_sample
                else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
