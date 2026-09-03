"""QB 分位数：全量 mx.sort vs mx.partition + 尾部 min 的等价性与耗时。

update_moe_biases 只要沿 token 轴的两个相邻序统计量（lo / lo+1）。
partition(kth=lo) 数学上是 introselect，位置 lo 上落的就是精确的第 lo 小，
且 lo 之后全体 ≥ 它，故 min(p[:, lo+1:]) 即精确的第 lo+1 小。

实测（mlx 0.32 Metal）：Partition::eval_gpu 直接转进 gpu_merge_sort，
目标形状墙钟与 sort 持平（~34ms）；两次 partition 则 2×。QB 的 q≈0.979，
尾部 ~2% 行，min 那趟几乎免费。
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import mlx.core as mx

G, N, E, K, NE = 13, 24576, 384, 8, 384
Q = 1.0 - K / NE


def bench(fn, n=7):
    fn()
    mx.eval(mx.array(0))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def by_sort(x, lo, hi, frac):
    s = mx.sort(x, axis=1)
    return s[:, lo].astype(mx.float32) * (1 - frac) + s[:, hi].astype(mx.float32) * frac


def by_partition(x, lo, hi, frac):
    p = mx.partition(x, lo, axis=1)
    v_lo = p[:, lo].astype(mx.float32)
    v_hi = v_lo if hi == lo else mx.min(p[:, lo + 1 :], axis=1).astype(mx.float32)
    return v_lo * (1 - frac) + v_hi * frac


def main():
    pos = Q * (N - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, N - 1)
    frac = pos - lo
    print(f"(G,N,E)=({G},{N},{E})  q={Q:.5f}  lo={lo} hi={hi} frac={frac:.4f}")
    print(f"尾部行数 {N - lo - 1}（占 {(N - lo - 1) / N * 100:.1f}%）")

    for dt in (mx.bfloat16, mx.float32):
        x = (mx.random.normal((G, N, E)) * 0.1).astype(dt)
        mx.eval(x)
        a = by_sort(x, lo, hi, frac)
        b = by_partition(x, lo, hi, frac)
        mx.eval(a, b)
        same = bool(mx.all(a == b))
        maxdiff = float(mx.max(mx.abs(a - b)))
        t_s = bench(lambda: by_sort(x, lo, hi, frac))
        t_p = bench(lambda: by_partition(x, lo, hi, frac))
        print(
            f"\n{str(dt):<18} 逐位相同={same} 最大差={maxdiff:.3e}\n"
            f"  sort      {t_s:7.1f}ms\n"
            f"  partition {t_p:7.1f}ms  ({t_s / max(t_p, 1e-9):.1f}× 快, "
            f"省 {t_s - t_p:.1f}ms)"
        )
        del x


if __name__ == "__main__":
    main()
