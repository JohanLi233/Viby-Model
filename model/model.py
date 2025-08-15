from transformers.configuration_utils import PretrainedConfig
import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin  # type: ignore
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

try:
    from causal_conv1d_mps import short_conv_fused, short_conv_update  # type: ignore

    HAS_CAUSAL_CONV1D_MPS = True
except ImportError:
    HAS_CAUSAL_CONV1D_MPS = False
    short_conv_fused = None
    short_conv_update = None


class VibyConfig(PretrainedConfig):
    model_type = "viby"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "gelu_pytorch_tanh",
        hidden_size: int = 640,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 32768,
        original_max_position_embeddings: int = 1024,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 18,
        num_key_value_heads: int = 1,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        z_loss_factor: float = 0.0,
        sliding_window: int = 512,
        canon_set: str = "ABCD",
        canon_bias: bool = False,
        canon_activation: bool = False,
        canon_kernel: int = 4,
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
        self.sliding_window = sliding_window
        self.canon_set = canon_set
        self.canon_bias = canon_bias
        self.canon_activation = canon_activation
        self.canon_kernel = canon_kernel


# (z_loss_cross_entropy and RoPE helper functions remain the same)
def z_loss_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    z_loss_factor: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy with Z-loss for training stability."""
    ce = F.cross_entropy(
        logits.flatten(end_dim=-2), labels.flatten(end_dim=-1), reduction="none"
    )

    if z_loss_factor > 0.0:
        z = torch.logsumexp(logits, dim=-1)
        z_term = z_loss_factor * (z.square())
        z_term = z_term.flatten()
        loss_unreduced = ce + z_term
    else:
        loss_unreduced = ce

    if mask is not None:
        mask_flat = mask.flatten()
        # 使用 clamp(min=1) 避免除以零
        loss = (loss_unreduced * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    else:
        loss = loss_unreduced.mean()

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


# [FIXED] CanonLayer with robust caching
class CanonLayer(nn.Module):
    """Canon layer using causal 1D convolution for efficient sequence modeling."""

    def __init__(self, hidden_size: int, config: VibyConfig, name: str):
        super().__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.kernel_size = config.canon_kernel
        self.bias = config.canon_bias
        self.activation = config.canon_activation

        self.weight = nn.Parameter(torch.randn(hidden_size, 1, self.kernel_size))
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias_param", None)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            in_channels = self.weight.size(1)
            kernel_size = self.weight.size(2)
            fan_in = in_channels * kernel_size
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def _get_canon_cache(self, cache: Cache, layer_idx: int, cache_name: str):
        """Get Canon cache state for this layer"""
        if not hasattr(cache, cache_name):
            max_layers = getattr(cache, "_max_layers", 256)
            setattr(cache, cache_name, [None] * max_layers)
        return getattr(cache, cache_name)[layer_idx]

    def _set_canon_cache(
        self, cache: Cache, layer_idx: int, cache_name: str, state: torch.Tensor
    ):
        """Set Canon cache state for this layer"""
        if not hasattr(cache, cache_name):
            max_layers = getattr(cache, "_max_layers", 256)
            setattr(cache, cache_name, [None] * max_layers)
        getattr(cache, cache_name)[layer_idx] = state

    def _get_canon_seqlens(self, cache: Cache, layer_idx: int, cache_name: str):
        """Get per-batch sequence lengths for MPS short-conv update (int32)."""
        seqlens_name = f"{cache_name}_seqlens"
        if not hasattr(cache, seqlens_name):
            max_layers = getattr(cache, "_max_layers", 256)
            setattr(cache, seqlens_name, [None] * max_layers)
        return getattr(cache, seqlens_name)[layer_idx]

    def _set_canon_seqlens(
        self, cache: Cache, layer_idx: int, cache_name: str, seqlens: torch.Tensor
    ):
        """Set per-batch sequence lengths for MPS short-conv update (int32)."""
        seqlens_name = f"{cache_name}_seqlens"
        if not hasattr(cache, seqlens_name):
            max_layers = getattr(cache, "_max_layers", 256)
            setattr(cache, seqlens_name, [None] * max_layers)
        getattr(cache, seqlens_name)[layer_idx] = seqlens

    # FIX: Updated signature to return the state and handle initialization.
    def _step_with_cache(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single token step with cache state. Returns output and updated state."""
        batch_size, seq_len, hidden_size = x.shape
        assert seq_len == 1, "Step function only supports single token"

        x_token = x.squeeze(1)  # (B, D)

        # [FIX] Apply attention mask if provided (Crucial for batched generation with padding)
        if attention_mask is not None:
            # Extract the mask for the current token (mask usually covers the whole sequence)
            if attention_mask.dim() == 2 and attention_mask.shape[0] == batch_size:
                current_mask = attention_mask[:, -1]  # (B,)
                # Mask the input token before processing and state update
                x_token = x_token * current_mask.unsqueeze(-1).to(x_token.dtype)

        # Initialize cache state if None (e.g., first token generation)
        if conv_state is None:
            state_len = max(8, self.kernel_size)
            conv_state = torch.zeros(
                batch_size, hidden_size, state_len, device=x.device, dtype=x.dtype
            )

        # Try MPS kernel for single token update (only supported for kernel_size == 4)
        if (
            HAS_CAUSAL_CONV1D_MPS
            and x.device.type == "mps"
            and short_conv_update is not None
            and self.kernel_size == 4
        ):
            try:
                cache_name = f"canon_cache_{self.name}"
                # Read existing seqlens from cache if available
                if cache is not None and layer_idx is not None:
                    cache_seqlens = self._get_canon_seqlens(
                        cache, layer_idx, cache_name
                    )
                else:
                    cache_seqlens = None
                # Default seqlens when absent
                if cache_seqlens is None:
                    cache_seqlens = torch.zeros(
                        batch_size, dtype=torch.int32, device=x.device
                    )
                output = short_conv_update(
                    x=x_token,  # Use the masked token
                    conv_state=conv_state,
                    weight=self.weight.squeeze(1),
                    bias=self.bias_param,
                    cache_seqlens=cache_seqlens,
                    activation=self.activation,
                    residual=True,
                )
                # Increment seqlens for next step
                if cache is not None and layer_idx is not None:
                    if (
                        attention_mask is not None
                        and attention_mask.dim() == 2
                        and attention_mask.shape[0] == batch_size
                    ):
                        inc = attention_mask[:, -1].to(torch.int32).to(x.device)
                    else:
                        inc = torch.ones(batch_size, dtype=torch.int32, device=x.device)
                    new_seqlens = cache_seqlens + inc
                    self._set_canon_seqlens(cache, layer_idx, cache_name, new_seqlens)
                return output.unsqueeze(1), conv_state  # Return state
            except Exception as e:
                print(
                    f"Warning: MPS update kernel failed ({e}), falling back to PyTorch"
                )

        # Fallback PyTorch implementation
        # _pytorch_step modifies conv_state in-place using x_token (which is now masked).
        output = self._pytorch_step(x_token, conv_state).unsqueeze(1)
        return x + output, conv_state  # Return state

    def _pytorch_step(self, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of single step"""
        batch_size, hidden_size = x.shape
        kernel_size = self.kernel_size

        # Initialize result with bias
        if self.bias_param is not None:
            # Use clone for safety with in-place operations
            result = self.bias_param.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            result = torch.zeros_like(x)

        # Rolling buffer update: Roll the state left by 1 and insert new input at the end
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x

        # Compute convolution using only the last kernel_size elements
        weight_2d = self.weight.squeeze(1)  # (D, W)
        conv_input = conv_state[:, :, -kernel_size:]  # (B, D, W)
        conv_output = torch.sum(conv_input * weight_2d.unsqueeze(0), dim=-1)
        result = result + conv_output

        # Apply activation
        if self.activation:
            result = F.silu(result)

        return result

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        cache_name = f"canon_cache_{self.name}"
        conv_state = None

        # Get cache state if available
        if cache is not None and layer_idx is not None:
            conv_state = self._get_canon_cache(cache, layer_idx, cache_name)

        # [FIX] Handle single token generation (Decoding path)
        # We use the optimized step if cache is enabled and seq_len is 1, even if state is None (handles initialization).
        if cache is not None and layer_idx is not None and seq_len == 1:
            result, conv_state = self._step_with_cache(
                x, conv_state, attention_mask, cache=cache, layer_idx=layer_idx
            )
            self._set_canon_cache(cache, layer_idx, cache_name, conv_state)
            return result

        # [FIX] Try MPS kernel ONLY if conv_state is None (no history).
        if HAS_CAUSAL_CONV1D_MPS and x.device.type == "mps" and conv_state is None:
            try:
                # (Attention mask processing logic from original code is preserved here)
                processed_attention_mask = attention_mask
                if attention_mask is not None:
                    batch_size, seq_len = x.shape[:2]
                    if attention_mask.shape != (batch_size, seq_len):
                        if attention_mask.shape[0] == batch_size:
                            if attention_mask.shape[1] > seq_len:
                                processed_attention_mask = attention_mask[:, -seq_len:]
                            elif attention_mask.shape[1] < seq_len:
                                pad_len = seq_len - attention_mask.shape[1]
                                pad = torch.ones(
                                    (batch_size, pad_len),
                                    device=attention_mask.device,
                                    dtype=attention_mask.dtype,
                                )
                                processed_attention_mask = torch.cat(
                                    [attention_mask, pad], dim=1
                                )
                        else:
                            processed_attention_mask = None

                # Ensure mask on MPS device
                if (
                    processed_attention_mask is not None
                    and processed_attention_mask.device != x.device
                ):
                    processed_attention_mask = processed_attention_mask.to(
                        device=x.device
                    )
                output = short_conv_fused(
                    x=x,
                    weight=self.weight.squeeze(1),
                    bias=self.bias_param,
                    attention_mask=processed_attention_mask,
                    activation=self.activation,
                    residual=True,
                )

                # [FIX] Update cache state for prefill stage
                if cache is not None and layer_idx is not None:
                    # Initialize cache state (conv_state is None here)
                    state_len = max(8, self.kernel_size)
                    conv_state = torch.zeros(
                        batch_size,
                        self.hidden_size,
                        state_len,
                        device=x.device,
                        dtype=x.dtype,
                    )

                    # Use masked inputs to populate state to avoid padding leakage
                    if processed_attention_mask is not None:
                        masked_x = x * processed_attention_mask.unsqueeze(-1)
                    else:
                        masked_x = x
                    x_conv = masked_x.transpose(1, 2)  # (B, D, T)

                    # [FIX] Update cache with the processed sequence (Right-aligned)
                    if seq_len <= state_len:
                        # Store at the end
                        conv_state[:, :, -seq_len:] = x_conv
                    else:
                        # If sequence is longer than cache, store the last part
                        conv_state[:, :, :] = x_conv[:, :, -state_len:]

                    self._set_canon_cache(cache, layer_idx, cache_name, conv_state)

                    # Initialize per-batch seqlens for MPS update based on valid tokens
                    if processed_attention_mask is not None:
                        init_seqlens = processed_attention_mask.sum(dim=1).to(
                            dtype=torch.int32, device=x.device
                        )
                    else:
                        init_seqlens = torch.full(
                            (batch_size,),
                            fill_value=seq_len,
                            dtype=torch.int32,
                            device=x.device,
                        )
                    self._set_canon_seqlens(cache, layer_idx, cache_name, init_seqlens)

                return output
            except Exception as e:
                print(
                    f"Warning: MPS kernel failed ({e}), falling back to PyTorch implementation"
                )

        # Fallback to PyTorch implementation (Handles Prefill and Chunked Decode)

        # Apply attention mask if provided (ensure correct slice is taken)
        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.shape == (
                batch_size,
                seq_len,
            ):
                x = x * attention_mask.unsqueeze(-1)
            elif (
                attention_mask.dim() == 2
                and attention_mask.size(0) == batch_size
                and attention_mask.size(1) >= seq_len
            ):
                current_mask = attention_mask[:, -seq_len:]
                x = x * current_mask.unsqueeze(-1)

        x_conv = x.transpose(1, 2)  # (B, D, T)

        # [FIX] Handle history concatenation for correct convolution computation
        history = None
        padding_len = self.kernel_size - 1

        if conv_state is not None and padding_len > 0:
            # Use the last (K-1) elements as history
            history = conv_state[:, :, -padding_len:]

        # Prepare input for convolution
        if history is not None:
            # Prepend history if available
            x_conv_input = torch.cat([history, x_conv], dim=-1)
        else:
            # Otherwise, use causal padding (zeros) - Start of sequence
            x_conv_input = F.pad(x_conv, (padding_len, 0))

        # [FIX] Update the cache state robustly for the next iteration
        if cache is not None and layer_idx is not None:
            # If conv_state was None, initialize it now.
            if conv_state is None:
                state_len = max(8, self.kernel_size)
                conv_state = torch.zeros(
                    batch_size,
                    self.hidden_size,
                    state_len,
                    device=x.device,
                    dtype=x.dtype,
                )

            state_len = conv_state.size(-1)

            # Determine how to update the state buffer (fixed-size sliding window)
            # We check if history was None (before initialization) to determine if it's start of sequence.
            if history is None:
                # Start of sequence (Prefill)
                if seq_len <= state_len:
                    # Store right-aligned
                    conv_state[:, :, -seq_len:] = x_conv
                else:
                    # Longer than state, store the last part
                    conv_state[:, :, :] = x_conv[:, :, -state_len:]
            else:
                # Continuation (Chunked Decode)
                # We need to append x_conv to the existing state and shift.

                if seq_len >= state_len:
                    # Input is large, the new state is just the end of x_conv.
                    conv_state[:, :, :] = x_conv[:, :, -state_len:]
                else:
                    # Input is small (T < S). Shift existing state left by T, and append x_conv.
                    # Shift left by T using roll (in-place update)
                    conv_state.copy_(torch.roll(conv_state, shifts=-seq_len, dims=-1))
                    # Insert new input at the end
                    conv_state[:, :, -seq_len:] = x_conv

            self._set_canon_cache(cache, layer_idx, cache_name, conv_state)

        # Grouped convolution
        x_conv_output = F.conv1d(
            x_conv_input, self.weight, bias=self.bias_param, groups=self.hidden_size
        )

        # Apply activation if specified
        if self.activation:
            x_conv_output = F.silu(x_conv_output)

        # Convert back to (batch, seq_len, hidden_size)
        output = x_conv_output.transpose(1, 2)

        return x + output


# (The rest of the classes: Attention, FeedForward, VibyBlock, VibyModel, VibyForCausalLM remain unchanged from the user's provided code, as the fixes were localized to CanonLayer. They are included below for completeness.)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 GQA 的 key/value 头以匹配 query 头的数量。
    输入格式: (batch, num_kv_heads, seq_len, head_dim)
    输出格式: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=1)


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
            self.canon_b = CanonLayer(total_dim, args, name="b")
        else:
            self.canon_b = None
            
        # QK-Clip: 标志是否收集注意力统计信息
        self._collect_attention_stats = False
    
    
    def enable_attention_stats_collection(self):
        """启用注意力统计信息收集"""
        self._collect_attention_stats = True
    
    def disable_attention_stats_collection(self):
        """禁用注意力统计信息收集"""
        self._collect_attention_stats = False

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply Canon B if enabled
        if self.canon_b is not None and layer_idx is not None:
            # Flatten back to 2D for Canon layer
            xq_flat = xq.view(bsz, seq_len, -1)
            xk_flat = xk.view(bsz, seq_len, -1)
            xv_flat = xv.view(bsz, seq_len, -1)

            # Concatenate Q, K, V
            qkv_concat = torch.cat(
                [xq_flat, xk_flat, xv_flat], dim=-1
            )  # (B, T, total_dim)

            # Apply Canon B with HF Cache
            qkv_processed = self.canon_b(
                qkv_concat,
                attention_mask=attention_mask,
                cache=past_key_value,
                layer_idx=layer_idx,
            )

            # Split back to Q, K, V
            q_dim = self.n_local_heads * self.head_dim
            kv_dim = self.n_local_kv_heads * self.head_dim
            xq_flat, xk_flat, xv_flat = qkv_processed.split(
                [q_dim, kv_dim, kv_dim], dim=-1
            )

            # Reshape back to multi-head format
            xq = xq_flat.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk_flat.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv_flat.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Convert to format expected by HF Cache: (batch, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)  # (bsz, n_local_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_local_kv_heads, seq_len, head_dim)
        xv = xv.transpose(1, 2)  # (bsz, n_local_kv_heads, seq_len, head_dim)

        # Handle KV caching with HF Cache system
        if past_key_value is not None and use_cache:
            # Update the cache and get updated key/value states
            cache_kwargs = {}  # Can add position info if needed
            xk, xv = past_key_value.update(xk, xv, layer_idx, cache_kwargs)

        past_kv = past_key_value if use_cache else None

        # GQA: 重复 key 和 value 以匹配 query 的头数
        key = repeat_kv(xk, self.n_rep)
        value = repeat_kv(xv, self.n_rep)

        # Dropout 概率（与 F.sdpa 语义一致，只在训练时生效）
        dropout_p = self.dropout if self.training else 0.0

        # YaRN 温度缩放：等价于对 Q 做温度缩放
        if self.attn_softmax_temp != 1.0:
            temp_scale = torch.tensor(self.attn_softmax_temp, dtype=xq.dtype, device=xq.device)
            xq = xq / temp_scale

        # 手写 attention：QK^T / sqrt(d)
        d = self.head_dim
        # 确保 scale 与张量有相同的数据类型和设备
        scale = torch.tensor(1.0 / math.sqrt(d), dtype=xq.dtype, device=xq.device)
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", xq, key) * scale
        
        # QK-Clip: 计算并存储每个头的最大 logit 值
        # 注意：为了与 torch.compile 兼容，我们延迟统计收集到训练循环中
        if layer_idx is not None and getattr(self, '_collect_attention_stats', False):
            # 获取当前批次中每个头的最大 logit
            # FIX: Cast to float32 before max to avoid inductor bug on MPS with bfloat16
            max_logits_per_head = torch.max(attn_scores.view(bsz, self.n_local_heads, -1).to(torch.float32), dim=-1)[0]
            batch_max_logits = torch.max(max_logits_per_head, dim=0)[0]  # (n_local_heads,)
            
            # 存储到模型属性中，供后续使用（避免在编译图中处理）
            self._current_attention_logits = {
                'layer_idx': layer_idx,
                'max_logits': batch_max_logits
            }

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
                        pad = torch.ones(
                            (am.size(0), pad_len), dtype=am.dtype, device=am.device
                        )
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

            # Apply key padding mask
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
        output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, value)

        # 回转维度
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN.get(config.hidden_act, torch.nn.functional.silu)

        # Canon D layer for gate and up projections
        if "D" in config.canon_set:
            self.canon_d = CanonLayer(config.intermediate_size * 2, config, name="d")
        else:
            self.canon_d = None

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
    ):
        gated = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply Canon D if enabled
        if self.canon_d is not None and layer_idx is not None:
            gate_up_concat = torch.cat([gated, up], dim=-1)
            gate_up_processed = self.canon_d(
                gate_up_concat,
                attention_mask=attention_mask,
                cache=cache,
                layer_idx=layer_idx,
            )
            # Split back
            gated = gate_up_processed[..., : gated.shape[-1]]
            up = gate_up_processed[..., gated.shape[-1] :]

        # Standard SwiGLU: SiLU(gate) * up
        h = F.silu(gated)
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

        # Canon A layer
        if "A" in config.canon_set:
            self.canon_a = CanonLayer(config.hidden_size, config, name="a")
        else:
            self.canon_a = None

        # Canon C layer
        if "C" in config.canon_set:
            self.canon_c = CanonLayer(config.hidden_size, config, name="c")
        else:
            self.canon_c = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        layer_id: int,
        past_key_value: Optional[Cache] = None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states

        # Pre-attention processing - normalize first
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Apply Canon A after normalization if enabled
        if self.canon_a is not None:
            normed_hidden_states = self.canon_a(
                normed_hidden_states,
                attention_mask=attention_mask,
                cache=past_key_value,
                layer_idx=layer_id,
            )

        # Sliding window strategy
        sliding_window = 0 if (layer_id + 1) % 6 == 0 else self.default_sliding_window
        attn_output, present_attn_kv = self.self_attn(
            normed_hidden_states,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            sliding_window=sliding_window,
            layer_idx=layer_id,
        )
        hidden_states = residual + attn_output

        # Pre-MLP processing
        residual = hidden_states

        # Normalize first
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply Canon C after normalization if enabled
        if self.canon_c is not None:
            normed_hidden_states = self.canon_c(
                normed_hidden_states,
                attention_mask=attention_mask,
                cache=past_key_value,
                layer_idx=layer_id,
            )

        mlp_output = self.mlp(
            normed_hidden_states,
            attention_mask,
            cache=past_key_value,
            layer_idx=layer_id,
        )
        hidden_states = residual + mlp_output

        # Return current cache state
        present_key_value = present_attn_kv if use_cache else None

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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            # Set the actual number of layers for Canon cache optimization
            past_key_values._max_layers = self.num_hidden_layers

        # Calculate start_pos from cache
        start_pos = 0
        if past_key_values is not None:
            start_pos = past_key_values.get_seq_length()

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],  # type: ignore
            self.freqs_sin[start_pos : start_pos + seq_length],  # type: ignore
        )

        presents = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                layer_id=layer_idx,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values if use_cache else None


class VibyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VibyConfig

    def __init__(self, config: VibyConfig):
        # Initialize config first
        config = config or VibyConfig()
        super().__init__(config)
        self.config = config
        self.model = VibyModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # Tie weights
        self.model.embed_tokens.weight = self.lm_head.weight

        # 为 MuonClip 优化器初始化 QK-Clip 统计缓冲区
        # 形状: (num_layers, num_heads)。使用 float32 以保证精度和稳定性。
        self.register_buffer(
            "attention_max_logits",
            torch.zeros(
                (config.num_hidden_layers, config.num_attention_heads),
                dtype=torch.float32
            ),
            persistent=False # 不保存到checkpoint
        )

        # Initialize canon layers if present
        self.apply(self._init_canon_layers)

    def _init_canon_layers(self, module):
        if isinstance(module, CanonLayer):
            module.reset_parameters()
    
    def enable_attention_stats_collection(self):
        """为所有注意力层启用统计信息收集"""
        # 初始化全局统计存储
        self.attention_logit_stats = {}
        
        for layer in self.model.layers:
            layer.self_attn.enable_attention_stats_collection()
            # 设置统计收集回调
            layer.self_attn._stats_callback = self._collect_attention_stats
    
    def disable_attention_stats_collection(self):
        """为所有注意力层禁用统计信息收集"""
        for layer in self.model.layers:
            layer.self_attn.disable_attention_stats_collection()
            if hasattr(layer.self_attn, '_stats_callback'):
                delattr(layer.self_attn, '_stats_callback')
    
    def _collect_attention_stats(self, layer_idx, stats):
        """收集注意力统计信息的回调函数"""
        if not hasattr(self, 'attention_logit_stats'):
            self.attention_logit_stats = {}
        
        layer_key = f'layer_{layer_idx}'
        self.attention_logit_stats[layer_key] = stats
    
    def get_attention_stats(self):
        """获取注意力统计信息"""
        if not hasattr(self, 'attention_logit_stats'):
            self.attention_logit_stats = {}
        return self.attention_logit_stats
    
    def update_attention_stats_from_forward(self):
        """
        为 QK-Clip 更新和聚合注意力统计数据（最大logits）。
        """
        if not hasattr(self, 'attention_max_logits'):
            return

        current_batch_max_logits = []
        
        # 从各层收集统计数据
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            if hasattr(attention, '_current_attention_logits'):
                logit_info = attention._current_attention_logits
                if logit_info['layer_idx'] == layer_idx:
                    # 分离并确保为 float32
                    max_logits = logit_info['max_logits'].detach().to(torch.float32)
                    current_batch_max_logits.append(max_logits)

        if current_batch_max_logits:
            # 堆叠成一个张量 (num_layers, num_heads)
            batch_stats_tensor = torch.stack(current_batch_max_logits, dim=0).to(self.attention_max_logits.device)
            
            # 关键：使用 torch.maximum 进行梯度累积聚合。
            # 使用 .data 对缓冲区进行原地更新，而不跟踪 autograd。
            self.attention_max_logits.data = torch.maximum(
                self.attention_max_logits.data,
                batch_stats_tensor
            )

            # 向后兼容：同时更新字典格式的统计信息用于日志记录
            if not hasattr(self, 'attention_logit_stats'):
                self.attention_logit_stats = {}
            
            for layer_idx, max_logits in enumerate(current_batch_max_logits):
                layer_key = f'layer_{layer_idx}'
                if layer_key not in self.attention_logit_stats:
                    self.attention_logit_stats[layer_key] = {}
                
                for head_idx in range(len(max_logits)):
                    head_key = f'head_{head_idx}'
                    self.attention_logit_stats[layer_key][head_key] = max_logits[head_idx].item()

        # 清理注意力模块上的临时存储
        for layer in self.model.layers:
             if hasattr(layer.self_attn, '_current_attention_logits'):
                 delattr(layer.self_attn, '_current_attention_logits')

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,  # 增加了 labels 以便计算 loss
        loss_mask: Optional[torch.Tensor] = None,
        **args,
    ):
        hidden_states, past_key_values_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = z_loss_cross_entropy(
                logits,
                labels,
                self.config.z_loss_factor,
                mask=loss_mask,
            )

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits,
            past_key_values=past_key_values_out,
            hidden_states=hidden_states,
        )

    # 让 HF generate 正确地只喂最后一个 token（当使用缓存时），并传递好 attention_mask
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # When we have cache, only pass the last token to avoid recomputing K/V
        if past_key_values is not None:
            seq_len = past_key_values.get_seq_length()
            if seq_len > 0:
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            # Pass the full attention mask; CanonLayer handles slicing if needed.
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    # Beam search 等会调用该方法以重排缓存
    def _reorder_cache(self, past_key_values, beam_idx: torch.LongTensor):
        if past_key_values is None:
            return None

        # 1. Reorder standard KV cache if the object supports it
        if hasattr(past_key_values, "reorder_cache"):
            # HF Cache objects handle reordering internally for KV cache
            past_key_values.reorder_cache(beam_idx)

        # 2. Reorder custom Canon cache states manually
        if hasattr(past_key_values, "__dict__"):
            for attr_name in list(past_key_values.__dict__.keys()):
                if attr_name.startswith("canon_cache_") and not attr_name.endswith(
                    "_seqlens"
                ):
                    cache_list = getattr(past_key_values, attr_name)
                    # Canon cache lists are always initialized as lists
                    for layer_idx in range(len(cache_list)):
                        if cache_list[layer_idx] is not None:
                            # Reorder the batch dimension using beam_idx
                            # Use index_select for robust reordering
                            cache_list[layer_idx] = cache_list[layer_idx].index_select(
                                0, beam_idx
                            )
                # Keep per-layer seqlens in sync
                if attr_name.startswith("canon_cache_") and attr_name.endswith(
                    "_seqlens"
                ):
                    seqlens_list = getattr(past_key_values, attr_name)
                    # Seqlens lists are always initialized as lists
                    for layer_idx in range(len(seqlens_list)):
                        if seqlens_list[layer_idx] is not None:
                            seqlens_list[layer_idx] = seqlens_list[
                                layer_idx
                            ].index_select(0, beam_idx)

        return past_key_values
