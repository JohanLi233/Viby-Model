import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import causal_conv1d_mps as ccmps


# Global dtype and tolerance (default bf16)
DTYPE = torch.bfloat16
ATOL_BY_DTYPE = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-1,
}


def get_tolerance_for(dtype: torch.dtype) -> float:
    return ATOL_BY_DTYPE.get(dtype, 1e-4)


def bench_robust(fn, warmup=10, iters=50, runs=5):
    """More stable performance test function"""
    results = []

    for run in range(runs):
        # Warmup before each run
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()

        # Measure multiple iterations
        times = []
        for _ in range(iters):
            t0 = time.time()
            fn()
            torch.mps.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

        # Remove highest and lowest values, calculate average
        times = sorted(times)[2:-2]  # Remove 2 extreme values from both ends
        avg_time = sum(times) / len(times)
        results.append(avg_time)

    # Remove highest and lowest runs, return median
    results = sorted(results)[1:-1]
    return sum(results) / len(results)


def bench_robust_stable(fn, warmup=25, iters=100, runs=5, desc=""):
    """
    A more stable and robust performance test function.

    Main improvements:
    1. Warmup before all runs.
    2. Use torch.mps.synchronize() for each measurement.
    3. Return median time of multiple runs, which is less sensitive to outliers.
    4. Return standard deviation for stability evaluation.
    """
    # Centralized warmup: before all timing starts, let GPU reach stable working state
    if desc:
        print(f"{desc:<20} {'Warming up...':<10}", end="\r")
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    run_times = []
    for _ in range(runs):
        # Each run is timed independently, more resistant to system interference
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        t1 = time.time()

        # Calculate average time per iteration
        avg_iter_time = (t1 - t0) / iters
        run_times.append(avg_iter_time)

    # Statistical analysis: calculate median and standard deviation
    median_time = np.median(run_times)
    std_dev = np.std(run_times)

    return median_time, std_dev


def causal_conv1d_reference(x, weight, bias=None, silu_activation=False):
    """PyTorch reference implementation"""
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # Use F.conv1d to implement causal convolution
    x_padded = F.pad(x, (width - 1, 0))  # Left padding

    out = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=dim, padding=0)
    out = out[:, :, :seqlen]  # Crop to original length

    if silu_activation:
        out = F.silu(out)

    return out


def canon_forward_reference(x_btd, weight_dw, bias_d=None, activation: bool = True):
    """
    Canon reference forward: input [B, T, D].
    - Depthwise group convolution (groups=D), kernel weights shape [D, W]
    - Causal padding padd_left=W-1
    - Optional SiLU activation
    Return shape [B, T, D]
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


def generate_attention_mask(
    batch: int, seqlen: int, device: torch.device, min_fill_ratio: float = 0.5
):
    """
    Generate a 2D attention_mask (1 for valid positions, 0 for padding).
    Each sample's valid length is uniformly sampled from [min_fill_ratio*seqlen, seqlen].
    Return a float32 tensor with shape (B, T).
    """
    min_len = max(1, int(seqlen * min_fill_ratio))
    lengths = torch.randint(low=min_len, high=seqlen + 1, size=(batch,), device=device)
    mask = torch.zeros(batch, seqlen, device=device, dtype=torch.float32)
    for b in range(batch):
        mask[b, : lengths[b].item()] = 1.0
    return mask


def short_conv_masked_reference(
    x_btd: torch.Tensor,
    weight_dw: torch.Tensor,
    bias_d: torch.Tensor | None,
    activation: bool = True,
    residual: bool = True,
    attention_mask: torch.Tensor | None = None,
):
    """
    Reference short-convolution path (B,T,D interface):
    - Input is [B, T, D]
    - First, zero out padding positions according to mask
    - Then, do depthwise causal conv (optional SiLU)
    - Optional residual (add result to original x)
    Return [B, T, D]
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
    x_in = x_btd
    if attention_mask is not None:
        x_btd = x_btd * attention_mask.unsqueeze(-1)
    x_bdt = x_btd.movedim(-1, -2).contiguous()
    y_bdt = ccmps.causal_conv1d_fn(
        x_bdt,
        weight_dw.contiguous(),
        bias_d.contiguous() if bias_d is not None else None,
        activation="silu" if activation else None,
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
    x_contig = x_btd.contiguous()
    w_contig = weight_dw.contiguous()

    y = ccmps.short_conv_fused(
        x_contig, w_contig, bias_d, attention_mask, activation, residual
    )
    return y


def run_canon_ac_bench():
    device = torch.device("mps")
    torch.manual_seed(202)
    configs = [
        # (B, T, D, W)
        (2, 256, 768, 4),
        (4, 512, 1024, 4),
    ]

    print("\n🧪 Scene (Optimized Fused): CanonA/C (B,T,D + Fused Kernel)")
    print(
        f"{'Config':<28} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 104)

    for bsz, seqlen, dim, width in configs:
        x_btd = torch.randn(bsz, seqlen, dim, device=device, dtype=DTYPE)
        weight_dw = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias_d = torch.randn(dim, device=device, dtype=DTYPE)
        mask = generate_attention_mask(bsz, seqlen, device)

        cfg = f"B{bsz} T{seqlen} D{dim} W{width}"

        def run_ref():
            return short_conv_masked_reference(
                x_btd,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )

        def run_mps():
            return short_conv_mps_optimized(
                x_btd,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )

        t_ref, _ = bench_robust_stable(
            run_ref, warmup=800, iters=300, runs=150, desc=cfg
        )
        t_mps, std_mps = bench_robust_stable(
            run_mps, warmup=800, iters=300, runs=150, desc=cfg
        )

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
            print(f"  ⚠️ Max difference: {max_diff:.6f}")


def run_canon_b_bench():
    device = torch.device("mps")
    torch.manual_seed(203)
    configs = [
        # (B, T, num_heads, num_kv_heads, head_dim, W)
        (2, 256, 12, 4, 64, 4),  # Dq=768, Dk=256, Dv=256, Dtotal=1280
        (2, 512, 16, 8, 64, 4),  # Dq=1024, Dk=512, Dv=512, Dtotal=2048
    ]

    print("\n🧪 Scene (Optimized Fused): CanonB (QKV concat + Fused Kernel)")
    print(
        f"{'Config':<40} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
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
            return short_conv_masked_reference(
                x_cat,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )

        def run_mps():
            return short_conv_mps_optimized(
                x_cat,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )

        t_ref, _ = bench_robust_stable(
            run_ref, warmup=800, iters=300, runs=150, desc=cfg
        )
        t_mps, std_mps = bench_robust_stable(
            run_mps, warmup=800, iters=300, runs=150, desc=cfg
        )

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
            print(f"  ⚠️ Max difference: {max_diff:.6f}")


def run_canon_d_bench():
    """
    Simulate a typical CanonD usage in MLP:
    - Concat gate_proj and up_proj outputs in last dimension
    - Do depthwise causal conv (SiLU + residual)
    - Then split back to original dimensions for down_proj
    """
    device = torch.device("mps")
    torch.manual_seed(204)
    configs = [
        # (B, T, hidden_size, intermediate_size, W) - Simulate MLP configs
        (2, 256, 768, 2048, 4),  # Small model config
        (2, 512, 1024, 4096, 4),  # Medium model config
    ]

    print("\n🧪 Scene (Optimized Fused): CanonD (MLP Gate&Up concat + Fused Kernel)")
    print(
        f"{'Config':<35} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 113)

    for bsz, seqlen, hidden_size, intermediate_size, width in configs:
        # Simulate MLP's gate_proj and up_proj outputs
        gate_output = torch.randn(
            bsz, seqlen, intermediate_size, device=device, dtype=DTYPE
        )
        up_output = torch.randn(
            bsz, seqlen, intermediate_size, device=device, dtype=DTYPE
        )

        # CanonD applied to concatenated output (intermediate_size * 2)
        x_cat = torch.cat([gate_output, up_output], dim=-1)

        weight_dw = torch.randn(
            intermediate_size * 2, width, device=device, dtype=DTYPE
        )
        bias_d = torch.randn(intermediate_size * 2, device=device, dtype=DTYPE)
        mask = generate_attention_mask(bsz, seqlen, device)

        cfg = f"B{bsz} T{seqlen} H{hidden_size} I{intermediate_size} W{width}"

        def run_ref():
            # Reference: do conv on concatenated tensor and then split into two parts
            conv_out = short_conv_masked_reference(
                x_cat,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )
            # Split back to gate and up parts
            gate_conv, up_conv = conv_out.chunk(2, dim=-1)
            return gate_conv, up_conv

        def run_mps():
            # MPS fused implementation for CanonD
            conv_out = short_conv_mps_optimized(
                x_cat,
                weight_dw,
                bias_d,
                activation=True,
                residual=True,
                attention_mask=mask,
            )
            # Split back to gate and up parts
            gate_conv, up_conv = conv_out.chunk(2, dim=-1)
            return gate_conv, up_conv

        t_ref, _ = bench_robust_stable(
            run_ref, warmup=800, iters=300, runs=150, desc=cfg
        )
        t_mps, std_mps = bench_robust_stable(
            run_mps, warmup=800, iters=300, runs=150, desc=cfg
        )

        gate_ref, up_ref = run_ref()
        gate_mps, up_mps = run_mps()

        # Compare differences between two outputs
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
            print(f"  ⚠️ Max difference: gate={max_diff_gate:.6f}, up={max_diff_up:.6f}")


def main():
    print("🚀 Causal Conv1D MPS performance test")

    assert torch.backends.mps.is_available(), "MPS not available"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CausalConv1D MPS Benchmarks")
    parser.add_argument(
        "--only-scenarios",
        action="store_true",
        help="Only run scenario benchmarks (A/C, B, and D)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Test data type (default bf16)",
    )
    args = parser.parse_args()

    # Set global DTYPE and tolerance based on parameters
    global DTYPE
    if args.dtype == "bf16":
        DTYPE = torch.bfloat16
    elif args.dtype == "fp16":
        DTYPE = torch.float16
    else:
        DTYPE = torch.float32

    if args.only_scenarios:
        run_canon_ac_bench()
        run_canon_b_bench()
        run_canon_d_bench()
        return

    device = torch.device("mps")

    # Test configs: (batch, dim, seqlen, width)
    test_configs = [
        (1, 64, 128, 4),  # Small scale
        (2, 128, 256, 4),  # Medium scale
        (4, 256, 512, 4),  # Large scale
        (1, 512, 1024, 4),  # Huge scale
        (8, 64, 128, 4),  # Huge batch
    ]

    print(
        f"{'Config':<20} {'MPS(ms)':<10} {'PyTorch(ms)':<12} {'Speedup':<10} {'MPS_StdDev(%)':<15} {'Correct':<8}"
    )
    print("-" * 80)

    for batch, dim, seqlen, width in test_configs:
        # Create test data
        torch.manual_seed(42)
        x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
        weight = torch.randn(dim, width, device=device, dtype=DTYPE)
        bias = torch.randn(dim, device=device, dtype=DTYPE)

        config_str = f"{batch}×{dim}×{seqlen}×{width}"

        try:
            # MPS implementation
            def run_mps():
                return ccmps.causal_conv1d_fn(
                    x.contiguous(), weight.contiguous(), bias.contiguous()
                )

            # PyTorch reference implementation
            def run_torch():
                return causal_conv1d_reference(x, weight, bias, False)

            # Performance test (using more stable method)
            t_mps, std_mps = bench_robust_stable(
                run_mps, warmup=1000, iters=500, runs=210, desc=config_str
            )
            t_torch, _ = bench_robust_stable(
                run_torch, warmup=1000, iters=500, runs=210, desc=config_str
            )

            # Correctness verification
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
                print(f"  ⚠️ Max difference: {max_diff:.6f}")

            # Take a break between different configurations to mitigate temperature effects
            time.sleep(1)

        except Exception as e:
            print(
                f"{config_str:<20} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<15} {'❌':<8}"
            )
            print(f"  Error: {e}")

    # SiLU activation function performance test
    print("\n🔥 SiLU activation function performance test")
    print(
        f"{'Config':<20} {'MPS+SiLU(ms)':<14} {'PyTorch+SiLU(ms)':<17} {'Speedup':<10} {'MPS_StdDev(%)':<15}"
    )
    print("-" * 90)

    # Test activation function with medium scale
    batch, dim, seqlen, width = 2, 128, 256, 4
    config_str = f"{batch}×{dim}×{seqlen}×{width}"

    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=DTYPE)
    weight = torch.randn(dim, width, device=device, dtype=DTYPE)
    bias = torch.randn(dim, device=device, dtype=DTYPE)

    try:

        def run_mps_silu():
            return ccmps.causal_conv1d_fn(
                x.contiguous(), weight.contiguous(), bias.contiguous(), activation="silu"
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
        print(f"  Error: {e}")

    print("\n📊 Performance test completed!")
    print(
        "💡 Tip: Speedup > 1.0 means MPS is faster. StdDev(%) smaller means more stable test results."
    )

    # =====================
    # Canon scene (B, T, D) benchmark
    # =====================
    print("\n🧪 Canon scene (B,T,D interface) benchmark")
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
            # Convert to [B, D, T] path to reuse kernel
            x_bdt = x_btd.movedim(-1, -2).contiguous()
            y_bdt = ccmps.causal_conv1d_fn(
                x_bdt, weight_dw.contiguous(), bias_d.contiguous(), activation="silu"
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
            print(f"  ⚠️ Max difference: {max_diff:.6f}")

    # =====================
    # Scene supplement: CanonA/C, CanonB, and CanonD
    # =====================
    run_canon_ac_bench()
    run_canon_b_bench()
    run_canon_d_bench()


if __name__ == "__main__":
    main()
