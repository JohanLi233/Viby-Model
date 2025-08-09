#!/usr/bin/env python3
"""
测试 causal_conv1d 的 MPS 实现正确性
"""

import torch
import torch.nn.functional as F
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import my_mps_extension._C as mps_ext


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


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")

    # 测试参数
    batch_size = 2
    dim = 4
    seqlen = 8
    width = 4

    # 创建测试数据
    torch.manual_seed(42)
    x = torch.randn(batch_size, dim, seqlen, device="mps", dtype=torch.float32)
    weight = torch.randn(dim, width, device="mps", dtype=torch.float32)
    bias = torch.randn(dim, device="mps", dtype=torch.float32)

    print(f"输入形状: x {x.shape}, weight {weight.shape}, bias {bias.shape}")

    try:
        # MPS 实现
        print("运行 MPS 实现...")
        result_mps = mps_ext.causal_conv1d_fwd(x, weight, bias, False)
        print(f"MPS 结果形状: {result_mps.shape}")

        # 参考实现
        print("运行参考实现...")
        result_ref = causal_conv1d_reference(x, weight, bias, False)
        result_ref_mps = result_ref.to("mps")
        print(f"参考结果形状: {result_ref_mps.shape}")

        # 比较结果
        diff = torch.abs(result_mps - result_ref_mps)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")

        # 检查是否在合理范围内
        tolerance = 1e-4
        if max_diff < tolerance:
            print("✅ 基本功能测试通过")
            return True
        else:
            print("❌ 基本功能测试失败，差异过大")
            print("MPS 结果（前几个元素）:")
            print(result_mps[0, 0, :5])
            print("参考结果（前几个元素）:")
            print(result_ref_mps[0, 0, :5])
            return False

    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False


def test_silu_activation():
    """测试 SiLU 激活函数"""
    print("\n=== 测试 SiLU 激活函数 ===")

    # 简单测试数据
    batch_size = 1
    dim = 2
    seqlen = 4
    width = 4

    torch.manual_seed(123)
    x = torch.randn(batch_size, dim, seqlen, device="mps", dtype=torch.float32)
    weight = torch.randn(dim, width, device="mps", dtype=torch.float32)
    bias = torch.randn(dim, device="mps", dtype=torch.float32)

    try:
        # 测试不带激活
        result_no_act = mps_ext.causal_conv1d_fwd(x, weight, bias, False)
        ref_no_act = causal_conv1d_reference(x, weight, bias, False).to("mps")

        # 测试带激活
        result_with_act = mps_ext.causal_conv1d_fwd(x, weight, bias, True)
        ref_with_act = causal_conv1d_reference(x, weight, bias, True).to("mps")

        # 比较
        diff_no_act = torch.max(torch.abs(result_no_act - ref_no_act)).item()
        diff_with_act = torch.max(torch.abs(result_with_act - ref_with_act)).item()

        print(f"无激活最大差异: {diff_no_act:.6f}")
        print(f"有激活最大差异: {diff_with_act:.6f}")

        tolerance = 1e-4
        if diff_no_act < tolerance and diff_with_act < tolerance:
            print("✅ SiLU 激活测试通过")
            return True
        else:
            print("❌ SiLU 激活测试失败")
            return False

    except Exception as e:
        print(f"❌ SiLU 激活测试出错: {e}")
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    results = []

    # 测试最小尺寸
    try:
        print("测试最小尺寸...")
        x = torch.randn(1, 1, 1, device="mps", dtype=torch.float32)
        weight = torch.randn(1, 4, device="mps", dtype=torch.float32)
        bias = torch.randn(1, device="mps", dtype=torch.float32)

        result = mps_ext.causal_conv1d_fwd(x, weight, bias, False)
        print(f"最小尺寸结果: {result.shape}")
        results.append(True)
    except Exception as e:
        print(f"❌ 最小尺寸测试失败: {e}")
        results.append(False)

    # 测试无偏置
    try:
        print("测试无偏置...")
        x = torch.randn(2, 3, 5, device="mps", dtype=torch.float32)
        weight = torch.randn(3, 4, device="mps", dtype=torch.float32)
        bias = torch.tensor([], device="mps", dtype=torch.float32)  # 空张量

        # MPS 实现应该能处理空的 bias
        result = mps_ext.causal_conv1d_fwd(x, weight, bias, False)
        print(f"无偏置结果: {result.shape}")
        results.append(True)
    except Exception as e:
        print(f"❌ 无偏置测试失败: {e}")
        results.append(False)

    all_passed = all(results)
    if all_passed:
        print("✅ 边界情况测试通过")
    else:
        print("❌ 部分边界情况测试失败")

    return all_passed


def main():
    """主测试函数"""
    print("开始测试 Causal Conv1D MPS 实现...")

    if not torch.backends.mps.is_available():
        print("❌ MPS 不可用，跳过测试")
        return

    results = []

    # 运行各项测试
    results.append(test_basic_functionality())
    results.append(test_silu_activation())
    results.append(test_edge_cases())

    # 总结
    print(f"\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，需要进一步调试")


if __name__ == "__main__":
    main()
