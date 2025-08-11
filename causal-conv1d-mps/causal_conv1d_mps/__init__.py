"""
CausalConv1D MPS Implementation

A Metal Performance Shaders (MPS) implementation of causal 1D convolution for PyTorch.
Provides high-performance GPU acceleration on Apple Silicon devices.
"""

import os
from pathlib import Path
import torch
from typing import Optional


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
        pkg_dir.parent / "causal_conv1d.metal",  # repo layout
        Path.cwd() / "causal_conv1d.metal",      # current working dir
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
__all__ = ["causal_conv1d_fwd", "short_conv_fused", "short_conv_update"]


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

    if attention_mask is not None:
        # 统一 mask 的 dtype 与形状到 (B, T)
        B, T, D = x.shape
        m = attention_mask
        # dtype 对齐
        if m.dtype == torch.bool:
            m = m.to(dtype=x.dtype)
        else:
            m = m.to(dtype=x.dtype)
        # 尝试压缩 size=1 维度
        if m.dim() > 2:
            m = m.squeeze()
        # 现在接受 1D 或 2D
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

        # 若仍与 (B, T) 的第二维不一致，则进行截断/扩展（优先截取末尾对齐当前序列长度）
        if m.shape[0] != B:
            raise ValueError(
                f"Attention mask batch {m.shape[0]} != input batch {B}"
            )
        if m.shape[1] != T:
            if m.shape[1] > T:
                # 通常出现在 generate 步（T=1）但传入完整历史 mask 的情况
                m = m[:, -T:]
            elif m.shape[1] == 1:
                m = m.expand(B, T)
            else:
                # 若更短且不为 1，则无法可靠广播
                raise ValueError(
                    f"Attention mask time {m.shape[1]} cannot match input time {T}"
                )
        attention_mask = m.contiguous()
    else:
        # 未提供 mask 时，创建满 1 的 (B, T) 掩码，避免旧版扩展对空张量形状检查失败
        B, T, _ = x.shape
        attention_mask = torch.ones((B, T), device=x.device, dtype=x.dtype)

    return _C.short_conv_fused(x, weight, bias, attention_mask, activation, residual)


def short_conv_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    activation: bool = True,
    residual: bool = True,
) -> torch.Tensor:
    """
    Single-token causal convolution update for efficient inference.

    Args:
        x: Input tensor of shape (batch, dim) - single token input
        conv_state: Convolution state cache of shape (batch, dim, state_len)
                   This tensor will be modified in-place to update the state
        weight: Weight tensor of shape (dim, width)
        bias: Optional bias tensor of shape (dim,)
        cache_seqlens: Current sequence lengths for each batch item, shape (batch,)
                      If None, assumes all batches are at the same position
        activation: Whether to apply SiLU activation
        residual: Whether to add residual connection

    Returns:
        Output tensor of shape (batch, dim) - single token output

    Raises:
        RuntimeError: If MPS is not available
        ValueError: If tensor shapes are invalid
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
        raise ValueError(
            f"Expected 2D input tensor (batch, dim), got {x.dim()}D"
        )

    if conv_state.dim() != 3:
        raise ValueError(
            f"Expected 3D conv_state tensor (batch, dim, state_len), got {conv_state.dim()}D"
        )

    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor (dim, width), got {weight.dim()}D")

    batch, dim = x.shape
    conv_batch, conv_dim, state_len = conv_state.shape
    weight_dim, width = weight.shape

    if batch != conv_batch or dim != conv_dim:
        raise ValueError(
            f"Input shape {x.shape} doesn't match conv_state batch/dim {(conv_batch, conv_dim)}"
        )

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
        # If not provided, create a simple increasing sequence
        # This assumes we're processing tokens sequentially for all batches
        cache_seqlens = torch.arange(batch, device=x.device, dtype=torch.int32)
    else:
        if cache_seqlens.device.type != "mps":
            raise ValueError("cache_seqlens must be on MPS device")
        if cache_seqlens.dim() != 1:
            raise ValueError(f"Expected 1D cache_seqlens, got {cache_seqlens.dim()}D")
        if cache_seqlens.shape[0] != batch:
            raise ValueError(
                f"cache_seqlens batch size {cache_seqlens.shape[0]} != input batch {batch}"
            )
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

    return _C.short_conv_update(
        x, conv_state, weight, bias, cache_seqlens, activation, residual
    )
