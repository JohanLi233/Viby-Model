"""
CausalConv1D MPS Implementation

A Metal Performance Shaders (MPS) implementation of causal 1D convolution for PyTorch.
Provides high-performance GPU acceleration on Apple Silicon devices.
"""

import torch
from typing import Optional

try:
    from . import _C
except ImportError:
    raise ImportError(
        "CausalConv1D MPS extension not found. Please build the package with: pip install -e ."
    )

__version__ = "0.1.0"
__all__ = ["causal_conv1d_fwd", "short_conv_fused"]


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
) -> torch.Tensor:
    """
    Causal 1D convolution forward pass using MPS.

    Args:
        x: Input tensor of shape (batch, dim, seqlen)
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        silu_activation: Whether to apply SiLU activation

    Returns:
        Output tensor of shape (batch, dim, seqlen)

    Raises:
        RuntimeError: If MPS is not available
        ValueError: If tensor shapes are invalid
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this device")

    if x.device.type != "mps":
        raise ValueError("Input tensor must be on MPS device")

    if weight.device.type != "mps":
        raise ValueError("Weight tensor must be on MPS device")

    if bias is not None and bias.device.type != "mps":
        raise ValueError("Bias tensor must be on MPS device")

    # Validate tensor shapes
    if x.dim() != 3:
        raise ValueError(
            f"Expected 3D input tensor (batch, dim, seqlen), got {x.dim()}D"
        )

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

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        # Create empty tensor for C++ interface
        bias = torch.tensor([], device=x.device, dtype=x.dtype)

    return _C.causal_conv1d_fwd(x, weight, bias, silu_activation)


def short_conv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    activation: bool = True,
    residual: bool = True,
) -> torch.Tensor:
    """
    Fused short convolution operation (optimized for HuggingFace-style usage).

    Args:
        x: Input tensor of shape (batch, seqlen, dim)
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        attention_mask: Optional attention mask of shape (batch, seqlen)
        activation: Whether to apply SiLU activation
        residual: Whether to add residual connection

    Returns:
        Output tensor of shape (batch, seqlen, dim)

    Raises:
        RuntimeError: If MPS is not available
        ValueError: If tensor shapes are invalid
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
        raise ValueError(
            f"Expected 3D input tensor (batch, seqlen, dim), got {x.dim()}D"
        )

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

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(f"Expected 2D attention mask, got {attention_mask.dim()}D")
        if attention_mask.shape != (batch, seqlen):
            raise ValueError(
                f"Attention mask shape {attention_mask.shape} does not match input shape ({batch}, {seqlen})"
            )

    # Ensure tensors are contiguous and prepare empty tensors if needed
    x = x.contiguous()
    weight = weight.contiguous()

    if bias is not None:
        bias = bias.contiguous()
    else:
        bias = torch.tensor([], device=x.device, dtype=x.dtype)

    if attention_mask is not None:
        attention_mask = attention_mask.to(torch.float32).contiguous()
    else:
        attention_mask = torch.tensor([], device=x.device, dtype=torch.float32)

    return _C.short_conv_fused(x, weight, bias, attention_mask, activation, residual)
