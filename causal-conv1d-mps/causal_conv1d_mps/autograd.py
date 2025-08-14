"""
PyTorch Autograd Functions for CausalConv1D MPS Implementation

This module provides autograd-enabled functions that support backward pass
for causal convolution operations on Apple Silicon devices using Metal.
"""

import torch
from typing import Optional, Tuple, Any
from . import _C


class CausalConv1dMPSFunction(torch.autograd.Function):
    """
    PyTorch autograd function for causal 1D convolution with MPS backend.
    
    This function wraps the MPS implementation and provides automatic 
    differentiation support by implementing both forward and backward passes.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = False,
        final_states_out: Optional[torch.Tensor] = None,
        silu_activation: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for causal convolution.
        
        Args:
            ctx: Autograd context for saving tensors
            x: Input tensor of shape (batch, dim, seqlen)
            weight: Weight tensor of shape (dim, width)
            bias: Optional bias tensor of shape (dim,)
            seq_idx: Optional sequence indices (not implemented in MPS version)
            initial_states: Optional initial states (not implemented in MPS version)
            return_final_states: Whether to return final states
            final_states_out: Optional output tensor for final states
            silu_activation: Whether to apply SiLU activation
            
        Returns:
            Output tensor of shape (batch, dim, seqlen)
        """
        # Validate inputs
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this device")

        if x.device.type != "mps":
            raise ValueError("Input tensor must be on MPS device")

        if weight.device.type != "mps":
            raise ValueError("Weight tensor must be on MPS device")

        if bias is not None and bias.device.type != "mps":
            raise ValueError("Bias tensor must be on MPS device")

        # For now, ignore advanced features not implemented in MPS version
        if seq_idx is not None:
            raise NotImplementedError("seq_idx not supported in MPS version")
        
        if initial_states is not None:
            raise NotImplementedError("initial_states not supported in MPS version")
            
        if return_final_states:
            raise NotImplementedError("final states not supported in MPS version")

        # Validate tensor shapes
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, dim, seqlen), got {x.dim()}D")

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight tensor (dim, width), got {weight.dim()}D")

        batch, dim, seqlen = x.shape
        weight_dim, width = weight.shape

        if dim != weight_dim:
            raise ValueError(f"Input dim {dim} does not match weight dim {weight_dim}")

        if bias is not None:
            if bias.dim() != 1:
                raise ValueError(f"Expected 1D bias tensor, got {bias.dim()}D")
            if bias.shape[0] != dim:
                raise ValueError(f"Bias dim {bias.shape[0]} does not match input dim {dim}")

        # Align dtypes
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)

        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        else:
            # Create empty tensor for C++ interface
            bias = torch.tensor([], device=x.device, dtype=x.dtype)

        # Save tensors for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.silu_activation = silu_activation
        
        # Call forward kernel
        output = _C.causal_conv1d_fwd(x, weight, bias, silu_activation)
        
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for causal convolution.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output tensor
            
        Returns:
            Tuple of gradients w.r.t. inputs: (dx, dweight, dbias, None, None, None, None, None)
        """
        x, weight, bias = ctx.saved_tensors
        silu_activation = ctx.silu_activation
        
        # Ensure grad_output is contiguous
        if grad_output.stride(2) != 1 and grad_output.stride(1) != 1:
            grad_output = grad_output.contiguous()
        
        # Call backward kernel
        dx, dweight, dbias = _C.causal_conv1d_bwd(
            x, weight, bias, grad_output, silu_activation
        )
        
        # Return gradients in the same order as forward arguments
        # (x, weight, bias, seq_idx, initial_states, return_final_states, final_states_out, silu_activation)
        return (
            dx,
            dweight,
            dbias if (bias.numel() > 0) else None,
            None,  # seq_idx
            None,  # initial_states
            None,  # return_final_states
            None,  # final_states_out
            None,  # silu_activation
        )


class ShortConvFusedMPSFunction(torch.autograd.Function):
    """
    PyTorch autograd function for fused short convolution with MPS backend.
    
    This function combines masking, convolution, and activation
    in a single fused kernel for efficiency.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        activation: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for fused short convolution.
        
        Args:
            ctx: Autograd context for saving tensors
            x: Input tensor of shape (batch, seqlen, dim)
            weight: Weight tensor of shape (dim, width)
            bias: Optional bias tensor of shape (dim,)
            attention_mask: Optional attention mask of shape (batch, seqlen)
            activation: Whether to apply SiLU activation
            
        Returns:
            Output tensor of shape (batch, seqlen, dim) - pure convolution result
        """
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this device")

        if x.device.type != "mps":
            raise ValueError("Input tensor must be on MPS device")

        if weight.device.type != "mps":
            raise ValueError("Weight tensor must be on MPS device")

        if bias is not None and bias.device.type != "mps":
            raise ValueError("Bias tensor must be on MPS device")

        if attention_mask is not None and attention_mask.device.type != "mps":
            raise ValueError("Attention mask must be on MPS device")

        # Validate tensor shapes
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (batch, seqlen, dim), got {x.dim()}D")

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight tensor (dim, width), got {weight.dim()}D")

        batch, seqlen, dim = x.shape
        weight_dim, width = weight.shape

        if dim != weight_dim:
            raise ValueError(f"Input dim {dim} does not match weight dim {weight_dim}")

        if bias is not None:
            if bias.dim() != 1:
                raise ValueError(f"Expected 1D bias tensor, got {bias.dim()}D")
            if bias.shape[0] != dim:
                raise ValueError(f"Bias dim {bias.shape[0]} does not match input dim {dim}")

        # Align dtypes to x
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)

        # Ensure tensors are contiguous and prepare empty tensors if needed
        x = x.contiguous()
        weight = weight.contiguous()

        if bias is not None:
            bias = bias.contiguous()
        else:
            bias = torch.tensor([], device=x.device, dtype=x.dtype)

        # Handle attention mask processing (normalized to (B, T))
        if attention_mask is not None:
            B, T, D = x.shape
            m = attention_mask
            # dtype alignment
            if m.dtype == torch.bool:
                m = m.to(dtype=x.dtype)
            else:
                m = m.to(dtype=x.dtype)
            # Try to squeeze size=1 dimensions
            if m.dim() > 2:
                m = m.squeeze()
            # Now accept 1D or 2D
            if m.dim() == 1:
                if m.numel() == T:
                    m = m.view(1, T).expand(B, T)
                elif m.numel() == B:
                    m = m.view(B, 1).expand(B, T)
                elif m.numel() == B * T:
                    m = m.view(B, T)
                else:
                    raise ValueError(
                        f"Attention mask 1D length {m.numel()} cannot be normalized to (B, T)={(B, T)}"
                    )
            elif m.dim() == 2:
                if m.shape == (B, T):
                    pass
                elif m.shape == (1, T):
                    m = m.expand(B, T)
                elif m.shape == (B, 1):
                    m = m.expand(B, T)
                elif m.shape == (T, B):
                    m = m.t().contiguous()
                else:
                    raise ValueError(
                        f"Attention mask 2D shape {tuple(m.shape)} must be broadcastable to (B, T)={(B, T)}"
                    )
            else:
                raise ValueError(
                    f"Attention mask dim {m.dim()} not supported; expected 1D/2D convertible to (B, T)"
                )

            # Handle sequence length mismatch
            if m.shape[0] != B:
                raise ValueError(f"Attention mask batch {m.shape[0]} != input batch {B}")
            if m.shape[1] != T:
                if m.shape[1] > T:
                    # Often occurs in generate step (T=1) but passing full history mask
                    m = m[:, -T:]
                elif m.shape[1] == 1:
                    m = m.expand(B, T)
                else:
                    raise ValueError(f"Attention mask time {m.shape[1]} cannot match input time {T}")
            attention_mask = m.contiguous()
        else:
            # Create full ones mask when not provided
            B, T, _ = x.shape
            attention_mask = torch.ones((B, T), device=x.device, dtype=x.dtype)

        # Save tensors for backward pass
        ctx.save_for_backward(x, weight, bias, attention_mask)
        ctx.activation = activation
        
        # Call forward kernel
        output = _C.short_conv_fused(x, weight, bias, attention_mask, activation)
        
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for fused short convolution.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output tensor
            
        Returns:
            Tuple of gradients w.r.t. inputs
        """
        x, weight, bias, attention_mask = ctx.saved_tensors
        activation = ctx.activation
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call backward kernel
        dx, dweight, dbias = _C.short_conv_fused_bwd(
            x, weight, bias, attention_mask, 
            grad_output, activation
        )
        
        # Return gradients for (x, weight, bias, attention_mask, activation)
        return (
            dx,
            dweight, 
            dbias if (bias.numel() > 0) else None,
            None,  # attention_mask (no gradient needed)
            None,  # activation
        )


class ShortConvUpdateMPSFunction(torch.autograd.Function):
    """
    PyTorch autograd function for short convolution update with MPS backend.
    
    This function is used for single-token inference with stateful convolution.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        activation: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for convolution update.
        
        Args:
            ctx: Autograd context for saving tensors
            x: Input tensor of shape (batch, dim) - single token input
            conv_state: Convolution state cache of shape (batch, dim, state_len)
            weight: Weight tensor of shape (dim, width)
            bias: Optional bias tensor of shape (dim,)
            cache_seqlens: Current sequence lengths for each batch item, shape (batch,)
            activation: Whether to apply SiLU activation
            
        Returns:
            Output tensor of shape (batch, dim) - single token output
        """
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this device")

        if x.device.type != "mps":
            raise ValueError("Input tensor must be on MPS device")

        if conv_state.device.type != "mps":
            raise ValueError("Convolution state tensor must be on MPS device")

        if weight.device.type != "mps":
            raise ValueError("Weight tensor must be on MPS device")

        if bias is not None and bias.device.type != "mps":
            raise ValueError("Bias tensor must be on MPS device")

        # Validate tensor shapes
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch, dim), got {x.dim()}D")

        if conv_state.dim() != 3:
            raise ValueError(f"Expected 3D conv_state tensor (batch, dim, state_len), got {conv_state.dim()}D")

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight tensor (dim, width), got {weight.dim()}D")

        batch, dim = x.shape
        conv_batch, conv_dim, state_len = conv_state.shape
        weight_dim, width = weight.shape

        if batch != conv_batch or dim != conv_dim:
            raise ValueError(f"Input shape {x.shape} doesn't match conv_state batch/dim {(conv_batch, conv_dim)}")

        if dim != weight_dim:
            raise ValueError(f"Input dim {dim} does not match weight dim {weight_dim}")

        if width != 4:
            raise ValueError(f"Only width=4 is supported for update kernel, got {width}")

        if bias is not None:
            if bias.dim() != 1:
                raise ValueError(f"Expected 1D bias tensor, got {bias.dim()}D")
            if bias.shape[0] != dim:
                raise ValueError(f"Bias dim {bias.shape[0]} does not match input dim {dim}")

        # Handle cache_seqlens
        if cache_seqlens is None:
            cache_seqlens = torch.arange(batch, device=x.device, dtype=torch.int32)
        else:
            if cache_seqlens.device.type != "mps":
                raise ValueError("cache_seqlens must be on MPS device")
            if cache_seqlens.dim() != 1:
                raise ValueError(f"Expected 1D cache_seqlens, got {cache_seqlens.dim()}D")
            if cache_seqlens.shape[0] != batch:
                raise ValueError(f"cache_seqlens batch size {cache_seqlens.shape[0]} != input batch {batch}")
            if cache_seqlens.dtype != torch.int32:
                cache_seqlens = cache_seqlens.to(torch.int32)

        # Align dtypes to x
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)

        # Ensure tensors are contiguous
        x = x.contiguous()
        conv_state = conv_state.contiguous()  # This will be modified in-place by the kernel
        weight = weight.contiguous()

        if bias is not None:
            bias = bias.contiguous()
        else:
            bias = torch.tensor([], device=x.device, dtype=x.dtype)

        cache_seqlens = cache_seqlens.contiguous()

        # Save tensors for backward pass (note: conv_state is modified in-place)
        # We need to save the original conv_state before modification
        original_conv_state = conv_state.clone()
        ctx.save_for_backward(x, original_conv_state, weight, bias, cache_seqlens)
        ctx.activation = activation
        
        # Call forward kernel (modifies conv_state in-place)
        output = _C.short_conv_update(
            x, conv_state, weight, bias, cache_seqlens, activation
        )
        
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for convolution update.
        
        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output tensor
            
        Returns:
            Tuple of gradients w.r.t. inputs
        """
        x, conv_state, weight, bias, cache_seqlens = ctx.saved_tensors
        activation = ctx.activation
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call backward kernel
        dx, dconv_state, dweight, dbias = _C.short_conv_update_bwd(
            x, conv_state, weight, bias, 
            cache_seqlens, grad_output, activation
        )
        
        # Return gradients for (x, conv_state, weight, bias, cache_seqlens, activation)
        return (
            dx,
            dconv_state,
            dweight,
            dbias if (bias.numel() > 0) else None,
            None,  # cache_seqlens (no gradient needed)
            None,  # activation
        )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Causal 1D convolution with automatic differentiation support.
    
    Args:
        x: Input tensor of shape (batch, dim, seqlen)
        weight: Weight tensor of shape (dim, width)  
        bias: Optional bias tensor of shape (dim,)
        seq_idx: Optional sequence indices (not supported in MPS version)
        initial_states: Optional initial states (not supported in MPS version)
        return_final_states: Whether to return final states (not supported)
        final_states_out: Optional output for final states (not supported)
        activation: Activation function ("silu", "swish", or None)
        
    Returns:
        Output tensor of shape (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    silu_activation = activation in ["silu", "swish"]
    
    return CausalConv1dMPSFunction.apply(
        x, weight, bias, seq_idx, initial_states, 
        return_final_states, final_states_out, silu_activation
    )


def short_conv_fused_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    """
    Fused short convolution with automatic differentiation support.
    
    Args:
        x: Input tensor of shape (batch, seqlen, dim)
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        attention_mask: Optional attention mask of shape (batch, seqlen)
        activation: Whether to apply SiLU activation
        
    Returns:
        Output tensor of shape (batch, seqlen, dim) - pure convolution result
    """
    return ShortConvFusedMPSFunction.apply(
        x, weight, bias, attention_mask, activation
    )


def short_conv_update_fn(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    """
    Convolution update with automatic differentiation support.
    
    Args:
        x: Input tensor of shape (batch, dim) - single token input
        conv_state: Convolution state cache of shape (batch, dim, state_len)
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        cache_seqlens: Current sequence lengths for each batch item, shape (batch,)
        activation: Whether to apply SiLU activation
        
    Returns:
        Output tensor of shape (batch, dim) - single token output
    """
    return ShortConvUpdateMPSFunction.apply(
        x, conv_state, weight, bias, cache_seqlens, activation
    )