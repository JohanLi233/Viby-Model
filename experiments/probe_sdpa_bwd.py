"""标定 MLX SDPA 反向的现状：它是不是就等于朴素 autodiff。

动手写 Metal flash 反向之前要先知道天花板在哪：
- 若 SDPA 的 bwd ≈ 朴素逐算子实现，说明 mlx 根本没有反向 kernel，
  手写的收益 = 因果性省一半 FLOPs + 避免物化 (B,H,T,T) 中间量；
- 若明显快于朴素，说明已有部分优化，收益要重新估。

同时给出「理想 flash 反向」的 FLOPs 下界，用于判断手写值不值。

用法: uv run experiments/probe_sdpa_bwd.py
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B = int(os.environ.get("VIBY_BENCH_B", 12))
H = int(os.environ.get("VIBY_BENCH_H", 8))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
DK = int(os.environ.get("VIBY_BENCH_DK", 128))
DV = int(os.environ.get("VIBY_BENCH_DV", 96))


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.3).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.3).astype(mx.bfloat16)
    C = mx.random.normal((B, H, T, DV)) * 0.3
    scale = 1.0 / math.sqrt(DK)
    vpad = mx.concatenate([v, mx.zeros((B, H, T, DK - DV), dtype=v.dtype)], axis=-1)
    mx.eval(q, k, v, C, vpad)

    def sdpa(q_, k_, v_):
        o = mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale, mask="causal")
        return (o[..., :DV].astype(mx.float32) * C).sum()

    def naive(q_, k_, v_):
        s = (q_ @ mx.swapaxes(k_, -1, -2)).astype(mx.float32) * scale
        m = mx.triu(mx.full((T, T), -mx.inf), k=1).astype(mx.float32)
        p = mx.softmax(s + m, axis=-1).astype(q_.dtype)
        o = p @ v_
        return (o.astype(mx.float32) * C).sum()

    vg_s = mx.value_and_grad(sdpa, argnums=(0, 1, 2))
    vg_n = mx.value_and_grad(naive, argnums=(0, 1, 2))

    arms = {
        "SDPA fwd": lambda: sdpa(q, k, vpad),
        "SDPA fwd+bwd": lambda: vg_s(q, k, vpad),
        "朴素 fwd": lambda: naive(q, k, v),
        "朴素 fwd+bwd": lambda: vg_n(q, k, v),
    }
    med = {}
    samples = {n: [] for n in arms}
    for rnd in range(7):
        for n, f in arms.items():
            t0 = time.perf_counter()
            mx.eval(f())
            samples[n].append(time.perf_counter() - t0)
    med = {n: statistics.median(s[2:]) for n, s in samples.items()}

    print(f"B={B} H={H} T={T} d_qk={DK} d_v={DV}（causal）")
    for n, t in med.items():
        print(f"  {n:<14}{t * 1e3:>8.2f}ms")
    sb = med["SDPA fwd+bwd"] - med["SDPA fwd"]
    nb = med["朴素 fwd+bwd"] - med["朴素 fwd"]
    print(
        f"\n  SDPA 反向净额 {sb * 1e3:.2f}ms   朴素反向净额 {nb * 1e3:.2f}ms"
        f"   比值 {nb / sb:.2f}x"
    )

    half = B * H * (T * T / 2)
    # 理想 flash 反向：dQ kernel 三个半单元（S/dP/dQ），dKV kernel 四个
    # （S/dP/dV/dK）。d 按各自参与的维度算。
    fl = (
        (half * DK * 2) * 2  # S 重算两次（dQ、dKV 各一次）
        + (half * DV * 2) * 2  # dP = dO·Vᵀ 两次
        + (half * DK * 2)  # dQ = dS·K
        + (half * DV * 2)  # dV = Pᵀ·dO
        + (half * DK * 2)  # dK = dSᵀ·Q
    )
    print(f"\n  理想 flash 反向 FLOPs {fl / 1e9:.1f} GFLOP")
    print(f"  当前 SDPA 反向等效 {fl / sb / 1e12:.2f} TFLOPS（按该 FLOPs 口径）")
    for tf in (8, 12, 16, 20):
        t = fl / (tf * 1e12)
        print(
            f"    若手写 kernel 达 {tf:>2} TFLOPS → {t * 1e3:>5.2f}ms  （{sb / t:.2f}x）"
        )


if __name__ == "__main__":
    main()
