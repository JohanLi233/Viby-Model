"""
CausalConv1D MPS Implementation

A Metal Performance Shaders (MPS) implementation of causal 1D convolution for PyTorch.
Provides high-performance GPU acceleration on Apple Silicon devices.
"""

import os
from pathlib import Path
import torch
from typing import Optional

# Import autograd functions
from .autograd import (
    causal_conv1d_fn,
    causal_conv1d_update,
    CausalConv1dMPSFunction,
    ShortConvFusedMPSFunction,
    ShortConvUpdateMPSFunction,
)


def _ensure_metal_path_env() -> None:
    """Ensure CAUSAL_CONV1D_METAL_PATH is set to a valid .metal file.

    Search order:
    1) Existing env var (if points to an existing file)
    2) Repo layout: parent of this package dir (../causal_conv1d.metal)
    3) Current working directory: ./causal_conv1d.metal
    """
    env_key = "CAUSAL_CONV1D_METAL_PATH"
    existing = os.environ.get(env_key)
    if existing and Path(existing).exists():
        return

    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "causal_conv1d.metal",  # package directory (current location)
        pkg_dir.parent / "causal_conv1d.metal",  # repo layout
        Path.cwd() / "causal_conv1d.metal",  # current working dir
    ]
    for p in candidates:
        if p.exists():
            os.environ[env_key] = str(p)
            break


_ensure_metal_path_env()

try:
    from . import _C
except ImportError:
    raise ImportError(
        "CausalConv1D MPS extension not found. Please build the package with: pip install -e ."
    )

__version__ = "0.1.0"


__all__ = [
    # Low-level forward functions (direct kernel calls)
    "causal_conv1d_fwd",
    # Main API functions (match CUDA version)
    "causal_conv1d_fn",
    "causal_conv1d_update",
    # Short conv functions
    "short_conv_fused",
    # Autograd function classes
    "CausalConv1dMPSFunction",
    "ShortConvFusedMPSFunction",
    "ShortConvUpdateMPSFunction",
]


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

    # Align dtypes: kernel requires weight/bias to match x.dtype
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

    return _C.causal_conv1d_fwd(x, weight, bias, silu_activation)




# =====================================================================================
# Recommended API
# =====================================================================================

# Main API (matches CUDA version)
causal_conv1d = causal_conv1d_fn


def short_conv_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    activation: bool = True,
    residual: bool = True,
) -> torch.Tensor:
    """
    Fused short convolution with masking, activation, and residual connection.
    
    Args:
        x: Input tensor of shape (batch, seqlen, dim)
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        attention_mask: Optional attention mask of shape (batch, seqlen)
        activation: Whether to apply SiLU activation
        residual: Whether to add residual connection
        
    Returns:
        Output tensor of shape (batch, seqlen, dim)
    """
    return ShortConvFusedMPSFunction.apply(
        x, weight, bias, attention_mask, activation, residual
    )
