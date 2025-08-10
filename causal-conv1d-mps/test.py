#!/usr/bin/env python3
"""
基于pytest的causal_conv1d MPS实现测试
改造自canon/causal-conv1d/tests/test_causal_conv1d.py
"""

import torch
import torch.nn.functional as F
import pytest
import causal_conv1d_mps


def causal_conv1d_reference(x, weight, bias=None, silu_activation=False):
    """
    使用 PyTorch 实现的参考版本（CPU）
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,) or None
    """
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # 转换为CPU进行参考计算
    x_cpu = x.detach().cpu().float()
    weight_cpu = weight.detach().cpu().float()
    bias_cpu = bias.detach().cpu().float() if bias is not None else None

    # 使用 F.conv1d 实现因果卷积
    # 添加 padding，然后截取正确的部分
    x_padded = F.pad(x_cpu, (width - 1, 0))  # 在左侧填充 width-1 个零

    # 使用分组卷积
    out = F.conv1d(
        x_padded, weight_cpu.unsqueeze(1), bias=bias_cpu, groups=dim, padding=0
    )

    # 截取到原始序列长度
    out = out[:, :, :seqlen]

    # 应用 SiLU 激活函数
    if silu_activation:
        out = F.silu(out)

    return out


@pytest.mark.parametrize("itype", [torch.float32])  # MPS主要支持float32
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [1, 2, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_causal_conv1d_mps(dim, seqlen, width, has_bias, silu_activation, itype):
    """测试基本的causal conv1d功能"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    
    # 设置随机种子
    torch.random.manual_seed(42)
    batch = 2
    
    # 创建测试数据
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32)
    else:
        bias = None
    
    # MPS实现
    out_mps = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, silu_activation)
    
    # 参考实现
    out_ref = causal_conv1d_reference(x, weight, bias, silu_activation)
    out_ref = out_ref.to(device)
    
    print(f"Output max diff: {(out_mps - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_mps - out_ref).abs().mean().item()}")
    
    # 验证结果
    assert torch.allclose(out_mps, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [8, 16, 32, 64])
@pytest.mark.parametrize("dim", [64, 128])
def test_short_conv_fused(dim, seqlen, width, has_bias, silu_activation, itype):
    """测试融合的short conv操作（HuggingFace风格）"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    
    # 设置随机种子
    torch.random.manual_seed(42)
    batch = 2
    
    # 创建测试数据 - 注意：这里是 (batch, seqlen, dim) 格式
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32)
    else:
        bias = None
    
    # 创建注意力掩码
    attention_mask = torch.ones(batch, seqlen, device=device, dtype=torch.float32)
    # 随机设置一些padding位置
    for b in range(batch):
        valid_len = torch.randint(seqlen//2, seqlen, (1,)).item()
        attention_mask[b, valid_len:] = 0
    
    # MPS融合实现
    out_mps = causal_conv1d_mps.short_conv_fused(
        x, weight, bias, attention_mask, activation=silu_activation, residual=True
    )
    
    # 参考实现（手工实现相同的操作）
    x_masked = x * attention_mask.unsqueeze(-1)
    x_transposed = x_masked.transpose(-1, -2).contiguous()  # (batch, dim, seqlen)
    
    conv_out = causal_conv1d_reference(x_transposed, weight, bias, silu_activation)
    conv_out = conv_out.transpose(-1, -2)  # 转回 (batch, seqlen, dim)
    
    out_ref = x + conv_out  # residual connection
    
    print(f"Output max diff: {(out_mps - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_mps - out_ref).abs().mean().item()}")
    
    # 验证结果
    assert torch.allclose(out_mps, out_ref, rtol=rtol, atol=atol)


def test_edge_cases():
    """测试边界情况"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = "mps"
    
    # 测试最小尺寸
    x = torch.randn(1, 1, 1, device=device, dtype=torch.float32)
    weight = torch.randn(1, 2, device=device, dtype=torch.float32)
    bias = torch.randn(1, device=device, dtype=torch.float32)
    
    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, False)
    assert result.shape == (1, 1, 1)
    
    # 测试无偏置
    x = torch.randn(2, 3, 5, device=device, dtype=torch.float32)
    weight = torch.randn(3, 4, device=device, dtype=torch.float32)
    
    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, None, False)
    assert result.shape == (2, 3, 5)


def test_error_handling():
    """测试错误处理"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = "mps"
    
    # 测试维度不匹配
    x = torch.randn(2, 4, 8, device=device, dtype=torch.float32)
    weight = torch.randn(5, 4, device=device, dtype=torch.float32)  # 错误的dim
    
    with pytest.raises(ValueError, match="does not match"):
        causal_conv1d_mps.causal_conv1d_fwd(x, weight, None, False)
    
    # 测试错误的tensor维度
    x_2d = torch.randn(4, 8, device=device, dtype=torch.float32)  # 应该是3D
    weight = torch.randn(4, 4, device=device, dtype=torch.float32)
    
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        causal_conv1d_mps.causal_conv1d_fwd(x_2d, weight, None, False)


def test_different_dtypes():
    """测试不同数据类型的支持"""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = "mps"
    batch, dim, seqlen, width = 2, 64, 32, 4
    
    # 测试float32
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    
    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, False)
    assert result.dtype == torch.float32
    assert result.shape == (batch, dim, seqlen)


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__])
