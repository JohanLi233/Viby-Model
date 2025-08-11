import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import causal_conv1d_mps as ccmps


# 全局 dtype 与容差（默认使用 bf16）
DTYPE = torch.bfloat16
ATOL_BY_DTYPE = {
    torch.float32: 1e-4,
    torch.float16: 5e-3,
    torch.bfloat16: 1e-2,
}


def get_tolerance_for(dtype: torch.dtype) -> float:
    return ATOL_BY_DTYPE.get(dtype, 1e-4)


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


def generate_attention_mask(batch: int, seqlen: int, device: torch.device, min_fill_ratio: float = 0.5):
    """
    生成 HuggingFace 风格的 2D attention_mask（1 表示有效，0 表示 padding）。
    每个样本的有效长度在 [min_fill_ratio*seqlen, seqlen] 之间随机。
    返回: mask (B, T) - float32
    """
    min_len = max(1, int(seqlen * min_fill_ratio))
    lengths = torch.randint(low=min_len, high=seqlen + 1, size=(batch,), device=device)
    mask = torch.zeros(batch, seqlen, device=device, dtype=torch.float32)
    for b in range(batch):
        mask[b, : lengths[b].item()] = 1.0
    return mask


def short_conv_hf_reference(
    x_btd: torch.Tensor,
    weight_dw: torch.Tensor,
    bias_d: torch.Tensor | None,
    activation: bool = True,
    residual: bool = True,
    attention_mask: torch.Tensor | None = None,
):
    """
    模拟 HuggingFace `ShortConvolution.forward` 的关键路径：
    - 输入为 [B, T, D]
    - 先按 mask 将 padding 位置置零
    - 做 depthwise causal conv（可选 SiLU）
    - 可选 residual（将结果与原始 x 相加）
    返回 [B, T, D]
    """
    x_in = x_btd
    if attention_mask is not None:
        m = attention_mask.to(dtype=x_btd.dtype)
        x_btd = x_btd * m.unsqueeze(-1)
    y = canon_forward_reference(x_btd, weight_dw, bias_d, activation)
    if residual:
        y = x_in + y
    return y


def short_conv_mps_like(
    x_btd: torch.Tensor,
    weight_dw: torch.Tensor,
    bias_d: torch.Tensor | None,
    activation: bool = True,
    residual: bool = True,
    attention_mask: torch.Tensor | None = None,
):
    """
    使用自研 MPS 核模拟 HF 的 `ShortConvolution.forward`：
    - 输入 [B, T, D]，按 mask 置零
    - 走 [B, D, T] 路径调用 mps_ext.causal_conv1d_fwd
    - 可选 residual
    返回 [B, T, D]
    """
    x_in = x_btd
    if attention_mask is not None:
        x_btd = x_btd * attention_mask.unsqueeze(-1)
    x_bdt = x_btd.movedim(-1, -2).contiguous()
    y_bdt = ccmps.causal_conv1d_fwd(
        x_bdt, weight_dw.contiguous(), bias_d.contiguous() if bias_d is not None else None, activation
    )
    y = y_bdt.movedim(-2, -1)
    if residual:
        y = x_in + y
    return y


def short_conv_mps_optimized(
    x_btd: torch.Tensor,
    weight_dw: torch.Tensor,
    bias_d: torch.Tensor | None,
    activation: bool = True,
    residual: bool = True,
    attention_mask: torch.Tensor | None = None,
):
    """
    使用优化的 FUSED MPS 内核。输入 (B, T, D)。避免布局更改并融合所有操作。
    """
    x_contig = x_btd.contiguous()
    w_contig = weight_dw.contiguous()

    y = ccmps.short_conv_fused(
        x_contig, w_contig, bias_d, attention_mask, activation, residual
    )
    return y

def run_hf_like_canon_ac_bench():
    """
    模拟 HF 中 CanonA/C 的使用：
    - 输入 [B, T, D]
    - 2D attention_mask (B, T)
    - SiLU 激活 + 残差
    对比 MPS 与 PyTorch 参考实现的性能与正确性。
    """
    device = torch.device("mps")
    torch.manual_seed(202)
    configs = [
        # (B, T, D, W)
        (2, 256, 768, 4),
        (4, 512, 1024, 4),
    ]

    print("\n🧪 HF 场景 (Optimized Fused)：CanonA/C（B,T,D + Fused Kernel）")
    print(f"{'Config':<28} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}")
    print("-" * 104)

    for bsz, seqlen, dim, width in configs:
        x_btd = torch.randn(bsz, seqlen, dim, device=device, dtype=DTYPE)
        weight_dw = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias_d = torch.randn(dim, device=device, dtype=DTYPE)
        mask = generate_attention_mask(bsz, seqlen, device)

        cfg = f"B{bsz} T{seqlen} D{dim} W{width}"

        def run_ref():
            return short_conv_hf_reference(
                x_btd, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )

        def run_mps():
            return short_conv_mps_optimized(
                x_btd, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )

        t_ref, _ = bench_robust_stable(run_ref, warmup=800, iters=300, runs=150, desc=cfg)
        t_mps, std_mps = bench_robust_stable(run_mps, warmup=800, iters=300, runs=150, desc=cfg)

        y_ref = run_ref()
        y_mps = run_mps()
        max_diff = torch.max(torch.abs(y_ref - y_mps)).item()
        is_ok = max_diff < get_tolerance_for(DTYPE)

        sp = t_ref / t_mps
        std_pct = (std_mps / t_mps) * 100 if t_mps > 0 else 0
        print(
            f"{cfg:<28} {t_mps * 1000:<10.2f} {t_ref * 1000:<12.2f} {sp:<10.2f} {std_pct:<15.2f} {'✅' if is_ok else '❌':<8}"
        )
        if not is_ok:
            print(f"  ⚠️ 最大差异: {max_diff:.6f}")


def run_hf_like_canon_b_bench():
    """
    模拟 HF 中 CanonB 在 Attention 里的用法：
    - 将 Q, K, V 在最后维度拼接，做 depthwise causal conv（SiLU + residual）
    - 不执行注意力，仅基准化这一步
    """
    device = torch.device("mps")
    torch.manual_seed(203)
    configs = [
        # (B, T, num_heads, num_kv_heads, head_dim, W)
        (2, 256, 12, 4, 64, 4),  # Dq=768, Dk=256, Dv=256, Dtotal=1280
        (2, 512, 16, 8, 64, 4),  # Dq=1024, Dk=512, Dv=512, Dtotal=2048
    ]

    print("\n🧪 HF 场景 (Optimized Fused)：CanonB（QKV 连接 + Fused Kernel）")
    print(f"{'Config':<40} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}")
    print("-" * 118)

    for bsz, seqlen, n_heads, n_kv, head_dim, width in configs:
        dq = n_heads * head_dim
        dk = n_kv * head_dim
        dv = n_kv * head_dim
        d_total = dq + dk + dv

        x_q = torch.randn(bsz, seqlen, dq, device=device, dtype=DTYPE)
        x_k = torch.randn(bsz, seqlen, dk, device=device, dtype=DTYPE)
        x_v = torch.randn(bsz, seqlen, dv, device=device, dtype=DTYPE)
        x_cat = torch.cat([x_q, x_k, x_v], dim=-1)

        weight_dw = torch.randn(d_total, width, device=device, dtype=DTYPE)
        bias_d = torch.randn(d_total, device=device, dtype=DTYPE)
        mask = generate_attention_mask(bsz, seqlen, device)

        cfg = f"B{bsz} T{seqlen} H{n_heads} KV{n_kv} hd{head_dim} W{width}"

        def run_ref():
            return short_conv_hf_reference(
                x_cat, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )

        def run_mps():
            return short_conv_mps_optimized(
                x_cat, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )

        t_ref, _ = bench_robust_stable(run_ref, warmup=800, iters=300, runs=150, desc=cfg)
        t_mps, std_mps = bench_robust_stable(run_mps, warmup=800, iters=300, runs=150, desc=cfg)

        y_ref = run_ref()
        y_mps = run_mps()
        max_diff = torch.max(torch.abs(y_ref - y_mps)).item()
        is_ok = max_diff < get_tolerance_for(DTYPE)

        sp = t_ref / t_mps
        std_pct = (std_mps / t_mps) * 100 if t_mps > 0 else 0
        print(
            f"{cfg:<40} {t_mps * 1000:<10.2f} {t_ref * 1000:<12.2f} {sp:<10.2f} {std_pct:<15.2f} {'✅' if is_ok else '❌':<8}"
        )
        if not is_ok:
            print(f"  ⚠️ 最大差异: {max_diff:.6f}")


def run_hf_like_canon_d_bench():
    """
    模拟 HF 中 CanonD 在 MLP 里的用法：
    - 将 gate_proj 和 up_proj 的输出在最后维度拼接
    - 做 depthwise causal conv（SiLU + residual）
    - 然后分割回原来的维度用于后续的 down_proj
    """
    device = torch.device("mps")
    torch.manual_seed(204)
    configs = [
        # (B, T, hidden_size, intermediate_size, W) - 模拟 MLP 配置
        (2, 256, 768, 2048, 4),   # 小模型配置
        (2, 512, 1024, 4096, 4),  # 中等模型配置
    ]

    print("\n🧪 HF 场景 (Optimized Fused)：CanonD（MLP Gate&Up 连接 + Fused Kernel）")
    print(f"{'Config':<35} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}")
    print("-" * 113)

    for bsz, seqlen, hidden_size, intermediate_size, width in configs:
        # 模拟 MLP 中 gate_proj 和 up_proj 的输出
        gate_output = torch.randn(bsz, seqlen, intermediate_size, device=device, dtype=DTYPE)
        up_output = torch.randn(bsz, seqlen, intermediate_size, device=device, dtype=DTYPE)
        
        # CanonD 应用在连接后的输出上 (intermediate_size * 2)
        x_cat = torch.cat([gate_output, up_output], dim=-1)
        
        weight_dw = torch.randn(intermediate_size * 2, width, device=device, dtype=DTYPE)
        bias_d = torch.randn(intermediate_size * 2, device=device, dtype=DTYPE)
        mask = generate_attention_mask(bsz, seqlen, device)

        cfg = f"B{bsz} T{seqlen} H{hidden_size} I{intermediate_size} W{width}"

        def run_ref():
            # CanonD 参考实现：对连接后的张量做卷积，然后分割
            conv_out = short_conv_hf_reference(
                x_cat, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )
            # 分割回 gate 和 up 部分
            gate_conv, up_conv = conv_out.chunk(2, dim=-1)
            return gate_conv, up_conv

        def run_mps():
            # CanonD MPS 实现
            conv_out = short_conv_mps_optimized(
                x_cat, weight_dw, bias_d, activation=True, residual=True, attention_mask=mask
            )
            # 分割回 gate 和 up 部分
            gate_conv, up_conv = conv_out.chunk(2, dim=-1)
            return gate_conv, up_conv

        t_ref, _ = bench_robust_stable(run_ref, warmup=800, iters=300, runs=150, desc=cfg)
        t_mps, std_mps = bench_robust_stable(run_mps, warmup=800, iters=300, runs=150, desc=cfg)

        gate_ref, up_ref = run_ref()
        gate_mps, up_mps = run_mps()
        
        # 比较两个输出的差异
        max_diff_gate = torch.max(torch.abs(gate_mps - gate_ref)).item()
        max_diff_up = torch.max(torch.abs(up_mps - up_ref)).item()
        max_diff = max(max_diff_gate, max_diff_up)
        is_ok = max_diff < get_tolerance_for(DTYPE)

        sp = t_ref / t_mps
        std_pct = (std_mps / t_mps) * 100 if t_mps > 0 else 0
        print(
            f"{cfg:<35} {t_mps * 1000:<10.2f} {t_ref * 1000:<12.2f} {sp:<10.2f} {std_pct:<15.2f} {'✅' if is_ok else '❌':<8}"
        )
        if not is_ok:
            print(f"  ⚠️ 最大差异: gate={max_diff_gate:.6f}, up={max_diff_up:.6f}")


def main():
    print("🚀 Causal Conv1D MPS 性能测试")

    assert torch.backends.mps.is_available(), "MPS not available"


    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CausalConv1D MPS Benchmarks")
    parser.add_argument("--only-hf", action="store_true", help="仅运行 HuggingFace 风格的 CanonA/C、CanonB 与 CanonD 基准")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="测试数据类型（默认 bf16）",
    )
    args = parser.parse_args()

    # 根据参数设置全局 DTYPE 与容差
    global DTYPE
    if args.dtype == "bf16":
        DTYPE = torch.bfloat16
    elif args.dtype == "fp16":
        DTYPE = torch.float16
    else:
        DTYPE = torch.float32

    if args.only_hf:
        run_hf_like_canon_ac_bench()
        run_hf_like_canon_b_bench()
        run_hf_like_canon_d_bench()
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
        x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
        weight = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias = torch.randn(dim, device=device, dtype=DTYPE)

        config_str = f"{batch}×{dim}×{seqlen}×{width}"

        try:
            # MPS 实现
            def run_mps():
                return ccmps.causal_conv1d_fwd(
                    x.contiguous(), weight.contiguous(), bias.contiguous(), False
                )

            # PyTorch 参考实现
            def run_torch():
                return causal_conv1d_reference(x, weight, bias, False)

            # 性能测试（使用更稳定的方法）
            t_mps, std_mps = bench_robust_stable(
                run_mps, warmup=1000, iters=500, runs=210, desc=config_str
            )
            t_torch, _ = bench_robust_stable(
                run_torch, warmup=1000, iters=500, runs=210, desc=config_str
            )

            # 正确性验证
            result_mps = run_mps()
            result_torch = run_torch()
            max_diff = torch.max(torch.abs(result_mps - result_torch)).item()
            is_correct = max_diff < get_tolerance_for(DTYPE)

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
    x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
    weight = torch.randn(dim, width, device=device, dtype=DTYPE)
    bias = torch.randn(dim, device=device, dtype=DTYPE)

    try:

        def run_mps_silu():
            return ccmps.causal_conv1d_fwd(
                x.contiguous(), weight.contiguous(), bias.contiguous(), True
            )

        def run_torch_silu():
            return causal_conv1d_reference(x, weight, bias, True)

        t_mps_silu, std_mps_silu = bench_robust_stable(
            run_mps_silu, warmup=1000, iters=400, runs=210, desc=config_str
        )
        t_torch_silu, _ = bench_robust_stable(
            run_torch_silu, warmup=1000, iters=400, runs=210, desc=config_str
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

    for bsz, seqlen, dim, width in canon_configs:
        x_btd = torch.randn(bsz, seqlen, dim, device=device, dtype=DTYPE)
        weight_dw = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias_d = torch.randn(dim, device=device, dtype=DTYPE)

        cfg = f"B{bsz} T{seqlen} D{dim} W{width}"

        def run_mps_canon():
            # 转为 [B, D, T] 路径以复用核
            x_bdt = x_btd.movedim(-1, -2).contiguous()
            y_bdt = ccmps.causal_conv1d_fwd(
                x_bdt, weight_dw.contiguous(), bias_d.contiguous(), True
            )
            return y_bdt.movedim(-2, -1)

        def run_ref_canon():
            return canon_forward_reference(x_btd, weight_dw, bias_d, activation=True)

        t_ref, _ = bench_robust_stable(
            run_ref_canon, warmup=1000, iters=500, runs=210, desc=cfg
        )

        t_mps, std_mps = bench_robust_stable(
            run_mps_canon, warmup=1000, iters=500, runs=210, desc=cfg
        )
        y_mps = run_mps_canon()
        y_ref = run_ref_canon()
        max_diff = torch.max(torch.abs(y_mps - y_ref)).item()
        is_ok = max_diff < get_tolerance_for(DTYPE)

        sp = t_ref / t_mps
        std_pct = (std_mps / t_mps) * 100 if t_mps > 0 else 0
        print(
            f"{cfg:<24} {t_mps * 1000:<10.2f} {t_ref * 1000:<12.2f} {sp:<10.2f} {std_pct:<15.2f} {'✅' if is_ok else '❌':<8}"
        )

        if not is_ok:
            print(f"  ⚠️ 最大差异: {max_diff:.6f}")

    # =====================
    # HF-like 场景补充：CanonA/C、CanonB 与 CanonD
    # =====================
    run_hf_like_canon_ac_bench()
    run_hf_like_canon_b_bench()
    run_hf_like_canon_d_bench()


if __name__ == "__main__":
    main()
