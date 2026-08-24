"""Gram-NS 在专家栈形状上的开销分解与分块变体。

prof_muon：刷新步 ns5_gram (2304,640,384) 664ms、(2304,384,320) 780ms，
合计 1444ms；而按实测 batched GEMM 吞吐（12.6~13.2 TFLOPS）算的裸 GEMM
下界只有 420 + 222 = 642ms。差额 ~800ms 疑似来自 (2304,384,384) 级中间
张量（679MB/份）的物化与分配抖动。

批维在 NS 里完全无耦合（逐矩阵正交化），所以沿 batch 分块是**数学恒等**
变换。本脚本对照：整批 / 各种分块大小，并给出与整批的数值差。

用法: uv run python experiments/probe_ns_gram.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import BatchedMuon

SHAPES = [
    (2304, 640, 384),
    (2304, 384, 320),
]
if os.environ.get("VIBY_NS_SHAPES"):
    SHAPES = [
        tuple(int(v) for v in s.split(","))
        for s in os.environ["VIBY_NS_SHAPES"].split(";")
    ]


def timed(fn, it=5, w=2):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


opt = BatchedMuon(learning_rate=1e-3, hyperball=True, ns_bf16=True)


def gram_chunked(X, chunk):
    """沿 batch 分块跑 Gram-NS（批维无耦合，逐块与整批逐矩阵同结果）。"""
    N = X.shape[0]
    if chunk >= N:
        return opt._ns5_gram(X)
    outs = [opt._ns5_gram(X[i : i + chunk]) for i in range(0, N, chunk)]
    return mx.concatenate(outs, axis=0)


for N, r, c in SHAPES:
    mx.random.seed(0)
    X = (mx.random.normal((N, r, c)) * 0.1).astype(mx.bfloat16)
    mx.eval(X)
    n_short, k_long = min(r, c), max(r, c)
    # 裸 GEMM 下界：R 初始化 + 5×(R@R) + 4×(Q@Z) + 4×2(Z@R@Z) + Y
    gemm_flops = (
        2 * N * n_short * n_short * k_long  # R = X@Xᵀ
        + 5 * 2 * N * n_short**3  # R@R
        + 4 * 2 * N * n_short**3  # Q@Z
        + 8 * 2 * N * n_short**3  # Z@R@Z
        + 2 * N * n_short * n_short * k_long  # Y = Q@X
    )
    print(f"\n=== ({N}, {r}, {c}) bf16 ===")
    print(
        f"X {X.nbytes / 2**20:.0f}MB  中间 (N,{n_short},{n_short}) "
        f"{N * n_short * n_short * 2 / 2**20:.0f}MB/份  "
        f"GEMM {gemm_flops / 1e12:.2f} TFLOP"
    )
    ref = None
    print(f"{'变体':<20}{'ms':>9}{'有效TFLOPS':>12}{'vs整批 rel':>12}")
    for chunk in (N, N // 2, N // 4, N // 8, N // 16, N // 32, 64):
        if chunk < 1:
            continue
        t = timed(lambda ch=chunk: gram_chunked(X, ch))
        y = gram_chunked(X, chunk)
        mx.eval(y)
        if ref is None:
            ref = y
            rel = 0.0
        else:
            rel = (
                (y.astype(mx.float32) - ref.astype(mx.float32)).abs().max()
                / (ref.astype(mx.float32).abs().max() + 1e-12)
            ).item()
        label = "整批" if chunk >= N else f"分块 {chunk}"
        print(f"{label:<20}{t:>9.1f}{gemm_flops / (t / 1e3) / 1e12:>12.2f}{rel:>12.1e}")
        mx.clear_cache()
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
