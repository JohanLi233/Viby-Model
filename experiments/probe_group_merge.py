"""固定总 FLOPs，只改 batched GEMM 的 batch 维：拆得越碎损失多少。

probe_moe_gemm 显示同样 129024 行、同样 M/N/K，36 次 batch-8 调用只有
7.52 TFLOPS，1 次 batch-288 有 10.95。稀疏桶按 EG=8 分组是为了压 padding
（组容量取组内峰值最大值），代价就是把 GEMM 拆碎。

如果损失主要来自「调用次数」而非「batch 维大小」，那么把容量相近的组量化到
同一档、合并成更宽的 batched GEMM 就能同时拿到低 padding 和高 GEMM 效率
——这值得改造。如果曲线在 batch 32~144 之间是平的、只在 288 处跳变，那说明
是别的机制，改造白做。

用法: uv run experiments/probe_group_merge.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

D = int(os.environ.get("VIBY_BENCH_D", 768))
I = int(os.environ.get("VIBY_BENCH_I", 104))  # noqa: E741
E = int(os.environ.get("VIBY_BENCH_E", 288))
CAP = int(os.environ.get("VIBY_BENCH_CAP", 448))


def timed(fn, it=10, w=3):
    for _ in range(w):
        fn()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    rows = E * CAP
    fl = 2 * rows * D * 2 * I
    print(f"E={E} cap={CAP} D={D} 2I={2 * I}  总行数={rows}  {fl / 1e9:.1f} GFLOP")
    print(f"{'batch':>7}{'调用数':>8}{'ms':>9}{'TFLOPS':>9}{'相对最优':>10}")

    x = mx.random.normal((E, CAP, D)).astype(mx.bfloat16)
    w = mx.random.normal((E, D, 2 * I)).astype(mx.bfloat16)
    mx.eval(x, w)

    res = []
    for eg in (8, 16, 32, 48, 72, 96, 144, 288):
        if E % eg:
            continue
        n = E // eg

        def fn(eg=eg, n=n):
            mx.eval(
                [x[i * eg : (i + 1) * eg] @ w[i * eg : (i + 1) * eg] for i in range(n)]
            )

        t = timed(fn)
        res.append((eg, n, t, fl / t / 1e12))
    best = max(r[3] for r in res)
    for eg, n, t, tf in res:
        print(f"{eg:>7}{n:>8}{t * 1e3:>9.2f}{tf:>9.2f}{tf / best:>10.2f}")

    # 对照：同样的组数，但每组容量不同（真实场景），看是否还有额外损失
    print("\n对照：36 组 batch-8，但每组容量按 Zipf 递减（真实容量表形状）")
    caps = [max(64, int(CAP * 2.2 / (1 + 0.35 * g))) // 64 * 64 for g in range(36)]
    xs = [mx.random.normal((8, c, D)).astype(mx.bfloat16) for c in caps]
    ws = [w[i * 8 : (i + 1) * 8] for i in range(36)]
    mx.eval(xs)
    tot = sum(8 * c for c in caps)
    fl2 = 2 * tot * D * 2 * I

    def fn2():
        mx.eval([a @ b for a, b in zip(xs, ws)])

    t = timed(fn2)
    print(
        f"  总行数={tot}  {fl2 / 1e9:.1f} GFLOP  {t * 1e3:.2f}ms  {fl2 / t / 1e12:.2f} TFLOPS"
    )


if __name__ == "__main__":
    main()
