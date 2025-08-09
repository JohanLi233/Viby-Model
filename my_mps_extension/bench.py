import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# 设置环境变量和路径
os.environ["CAUSAL_CONV1D_METAL_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "causal_conv1d.metal"
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def bench_robust(fn, warmup=10, iters=50, runs=5):
    """更稳定的性能测试函数"""
    results = []

    for run in range(runs):
        # 每次运行前都预热
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()

        # 测量多次迭代
        times = []
        for _ in range(iters):
            t0 = time.time()
            fn()
            torch.mps.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

        # 去掉最高和最低值，计算平均值
        times = sorted(times)[2:-2]  # 去掉前后各2个极值
        avg_time = sum(times) / len(times)
        results.append(avg_time)

    # 去掉最高和最低的运行，返回中位数
    results = sorted(results)[1:-1]
    return sum(results) / len(results)


def bench_robust_stable(fn, warmup=25, iters=100, runs=5, desc=""):
    """
    一个更稳定、更健壮的性能测试函数。

    主要改进:
    1. 在所有运行(runs)开始前进行一次充分的预热。
    2. 每次测量都严格使用 torch.mps.synchronize() 包裹。
    3. 返回多次运行(runs)的【中位数】时间，它对异常值不敏感。
    4. 同时返回标准差，用于评估结果的稳定性。
    """
    # 集中预热：在所有计时开始前，让GPU达到稳定工作状态
    if desc:
        print(f"{desc:<20} {'预热中...':<10}", end="\r")
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    run_times = []
    for _ in range(runs):
        # 每次运行都独立计时，更能抵抗系统干扰
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        t1 = time.time()

        # 计算单次迭代的平均时间
        avg_iter_time = (t1 - t0) / iters
        run_times.append(avg_iter_time)

    # 统计分析：计算中位数和标准差
    median_time = np.median(run_times)
    std_dev = np.std(run_times)

    return median_time, std_dev


def causal_conv1d_reference(x, weight, bias=None, silu_activation=False):
    """PyTorch 参考实现"""
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # 使用 F.conv1d 实现因果卷积
    x_padded = F.pad(x, (width - 1, 0))  # 左侧填充
    out = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=dim, padding=0)
    out = out[:, :, :seqlen]  # 截取到原始长度

    if silu_activation:
        out = F.silu(out)

    return out


def canon_forward_reference(x_btd, weight_dw, bias_d=None, activation: bool = True):
    """
    参考版 Canon 前向：输入 [B, T, D]，与 lingua 的 Canon 一致。
    - depthwise 组卷积 (groups=D)，kernel 权重形状为 [D, W]
    - 因果填充 padd_left=W-1
    - 可选 SiLU 激活
    返回同形状 [B, T, D]
    """
    b, t, d = x_btd.shape
    w = weight_dw.shape[1]
    x_bdt = x_btd.movedim(-1, -2)  # [B, D, T]
    x_pad = F.pad(x_bdt, (w - 1, 0))
    y = F.conv1d(x_pad, weight_dw.unsqueeze(1), bias=bias_d, groups=d)
    y = y[..., :t]
    if activation:
        y = F.silu(y)
    return y.movedim(-2, -1)


def main():
    print("🚀 Causal Conv1D MPS 性能测试")

    assert torch.backends.mps.is_available(), "MPS not available"

    try:
        import my_mps_extension._C as mps_ext

        print("✅ 成功加载 MPS 扩展")
    except ImportError as e:
        print(f"❌ 无法加载扩展: {e}")
        return

    device = torch.device("mps")

    # 测试配置：(batch, dim, seqlen, width)
    test_configs = [
        (1, 64, 128, 4),  # 小规模
        (2, 128, 256, 4),  # 中等规模
        (4, 256, 512, 4),  # 大规模
        (1, 512, 1024, 4),  # 超大规模
        (8, 64, 128, 4),  # 大批量
    ]

    print(
        f"{'Config':<20} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 80)

    for batch, dim, seqlen, width in test_configs:
        # 创建测试数据
        torch.manual_seed(42)
        x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)
        bias = torch.randn(dim, device=device, dtype=torch.float32)

        config_str = f"{batch}×{dim}×{seqlen}×{width}"

        try:
            # MPS 实现
            def run_mps():
                return mps_ext.causal_conv1d_fwd(
                    x.contiguous(), weight.contiguous(), bias.contiguous(), False
                )

            # PyTorch 参考实现
            def run_torch():
                return causal_conv1d_reference(x, weight, bias, False)

            # 性能测试（使用更稳定的方法）
            t_mps, std_mps = bench_robust_stable(
                run_mps, warmup=1000, iters=500, runs=21, desc=config_str
            )
            t_torch, _ = bench_robust_stable(
                run_torch, warmup=1000, iters=500, runs=21, desc=config_str
            )

            # 正确性验证
            result_mps = run_mps()
            result_torch = run_torch()
            max_diff = torch.max(torch.abs(result_mps - result_torch)).item()
            is_correct = max_diff < 1e-4

            speedup = t_torch / t_mps
            std_percent_mps = (std_mps / t_mps) * 100 if t_mps > 0 else 0
            print(
                f"{config_str:<20} {t_mps * 1000:<10.2f} {t_torch * 1000:<12.2f} {speedup:<10.2f} {std_percent_mps:<15.2f} {'✅' if is_correct else '❌':<8}"
            )

            if not is_correct:
                print(f"  ⚠️  最大差异: {max_diff:.6f}")

            # 在不同配置之间稍微休息，缓解温度影响
            time.sleep(1)

        except Exception as e:
            print(
                f"{config_str:<20} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<15} {'❌':<8}"
            )
            print(f"  错误: {e}")

    # SiLU 激活函数性能测试
    print("\n🔥 SiLU 激活函数性能测试")
    print(
        f"{'Config':<20} {'MPS+SiLU(ms)':<14} {'PyTorch+SiLU(ms)':<17} {'Speedup':<10} {'MPS_StdDev(%)':<15}"
    )
    print("-" * 90)

    # 选择中等规模测试激活函数
    batch, dim, seqlen, width = 2, 128, 256, 4
    config_str = f"{batch}×{dim}×{seqlen}×{width}"

    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)

    try:

        def run_mps_silu():
            return mps_ext.causal_conv1d_fwd(
                x.contiguous(), weight.contiguous(), bias.contiguous(), True
            )

        def run_torch_silu():
            return causal_conv1d_reference(x, weight, bias, True)

        t_mps_silu, std_mps_silu = bench_robust_stable(
            run_mps_silu, warmup=1000, iters=400, runs=21, desc=config_str
        )
        t_torch_silu, _ = bench_robust_stable(
            run_torch_silu, warmup=1000, iters=400, runs=21, desc=config_str
        )
        speedup_silu = t_torch_silu / t_mps_silu
        std_percent_mps_silu = (
            (std_mps_silu / t_mps_silu) * 100 if t_mps_silu > 0 else 0
        )

        print(
            f"{config_str:<20} {t_mps_silu * 1000:<14.2f} {t_torch_silu * 1000:<17.2f} {speedup_silu:<10.2f} {std_percent_mps_silu:<15.2f}"
        )

    except Exception as e:
        print(f"{config_str:<20} {'ERROR':<12} {'ERROR':<15} {'ERROR':<10}")
        print(f"  错误: {e}")

    print("\n📊 性能测试完成！")
    print(
        "💡 提示: Speedup > 1.0 表示 MPS 实现更快。StdDev(%) 越小，表示测试结果越稳定。"
    )

    # =====================
    # Canon 场景（B, T, D）基准
    # =====================
    print("\n🧪 Canon 使用场景基准 (B,T,D 接口)")
    print(
        f"{'Config':<24} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 96)

    device = torch.device("mps")
    torch.manual_seed(123)
    canon_configs = [
        # (B, T, D, W)
        (1, 128, 512, 4),
        (2, 256, 768, 4),
        (4, 512, 1024, 4),
    ]

    try:
        import my_mps_extension._C as mps_ext
    except Exception as e:
        print(f"无法加载扩展，跳过 Canon 场景测试: {e}")
        return

    for bsz, seqlen, dim, width in canon_configs:
        x_btd = torch.randn(bsz, seqlen, dim, device=device)
        weight_dw = torch.randn(dim, width, device=device)
        bias_d = torch.randn(dim, device=device)

        cfg = f"B{bsz} T{seqlen} D{dim} W{width}"

        def run_mps_canon():
            # 转为 [B, D, T] 路径以复用核
            x_bdt = x_btd.movedim(-1, -2).contiguous()
            y_bdt = mps_ext.causal_conv1d_fwd(
                x_bdt, weight_dw.contiguous(), bias_d.contiguous(), True
            )
            return y_bdt.movedim(-2, -1)

        def run_ref_canon():
            return canon_forward_reference(x_btd, weight_dw, bias_d, activation=True)

        t_ref, _ = bench_robust_stable(
            run_ref_canon, warmup=1000, iters=500, runs=21, desc=cfg
        )

        t_mps, std_mps = bench_robust_stable(
            run_mps_canon, warmup=1000, iters=500, runs=21, desc=cfg
        )
        y_mps = run_mps_canon()
        y_ref = run_ref_canon()
        max_diff = torch.max(torch.abs(y_mps - y_ref)).item()
        is_ok = max_diff < 1e-4

        sp = t_ref / t_mps
        std_pct = (std_mps / t_mps) * 100 if t_mps > 0 else 0
        print(
            f"{cfg:<24} {t_mps*1000:<10.2f} {t_ref*1000:<12.2f} {sp:<10.2f} {std_pct:<15.2f} {'✅' if is_ok else '❌':<8}"
        )

        if not is_ok:
            print(f"  ⚠️ 最大差异: {max_diff:.6f}")


if __name__ == "__main__":
    main()
