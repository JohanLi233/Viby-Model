"""扫 flash 反向 kernel 的分块参数（线程数 × 流过块高）。

首版取 NT=128/STR=32 只有 2.7 TFLOPS，远低于 simdgroup MMA 该有的水平。
threadgroup 内存占到 24~26KB（上限 32KB）意味着每核只驻留一个 threadgroup、
只有 4 个 simdgroup 在飞，藏不住访存延迟——所以要连着线程数一起扫。

NT 决定常驻块高 RES=NT/4（每 simdgroup 固定管 8 行），STR 是流过块高。
tile_ok 先按 32KB 上限筛掉不合法组合。

用法: uv run experiments/sweep_flash_tile.py
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_fused as af

B = int(os.environ.get("VIBY_BENCH_B", 12))
H = int(os.environ.get("VIBY_BENCH_H", 8))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
DK = int(os.environ.get("VIBY_BENCH_DK", 128))
DV = int(os.environ.get("VIBY_BENCH_DV", 96))


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(DK)
    vpad = mx.concatenate([v, mx.zeros((B, H, T, DK - DV), dtype=v.dtype)], axis=-1)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    dmask = mx.where((same & tril)[:, None], 0.0, -1e9).astype(mx.bfloat16)
    o = mx.fast.scaled_dot_product_attention(q, k, vpad, scale=scale, mask="causal")[
        ..., :DV
    ]
    mx.eval(q, k, v, do, dmask, o)

    half = B * H * (T * T / 2)
    fl = half * DK * 2 * 4 + half * DV * 2 * 3

    cands = []
    for nt in (128, 256, 512, 1024):
        for strb in (8, 16, 32, 64):
            split = af._split_ok(DK, DV, nt)
            res = af._res_split(nt) if split else af._res(nt)
            if T % res or T % strb or not af.tile_ok(DK, DV, nt, strb):
                continue
            cands.append((nt, strb))

    print(f"B={B} H={H} T={T} d_qk={DK} d_v={DV}   反向 FLOPs {fl / 1e9:.1f} GFLOP")
    print(
        f"{'NT':>6}{'RES':>5}{'STR':>5}{'tg内存':>9}{'causal':>10}{'doc_mask':>10}{'TFLOPS':>9}"
    )
    best = None
    for nt, strb in cands:
        split = af._split_ok(DK, DV, nt)
        res = af._res_split(nt) if split else af._res(nt)
        if split:
            strl = af._lse_str(nt, strb)
            tg = max(
                af._split.tgmem(kd, DK, DV, nt, strb, strl)
                for kd in ("lse", "dq", "dkv")
            )
            tg = max(tg, af._split.tgmem_combined(DK, DV, nt, strb))
        else:
            tg = max(af._tgmem(kd, DK, DV, res, strb) for kd in ("lse", "dq", "dkv"))
        try:
            for _ in range(2):
                mx.eval(af.flash_backward(q, k, v, o, do, scale, None, nt, strb))
                mx.eval(af.flash_backward(q, k, v, o, do, scale, dmask, nt, strb))
        except Exception as exc:
            print(f"{nt:>6}{res:>5}{strb:>5}{tg / 1024:>8.1f}K   失败 {str(exc)[:40]}")
            continue
        tc, tm = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            mx.eval(af.flash_backward(q, k, v, o, do, scale, None, nt, strb))
            t1 = time.perf_counter()
            mx.eval(af.flash_backward(q, k, v, o, do, scale, dmask, nt, strb))
            t2 = time.perf_counter()
            tc.append(t1 - t0)
            tm.append(t2 - t1)
        c, m = statistics.median(tc), statistics.median(tm)
        print(
            f"{nt:>6}{res:>5}{strb:>5}{tg / 1024:>8.1f}K{c * 1e3:>10.2f}{m * 1e3:>10.2f}"
            f"{fl / c / 1e12:>9.2f}"
        )
        if best is None or c < best[0]:
            best = (c, nt, strb)
    if best:
        print(
            f"\n最优 NT={best[1]} STR={best[2]}  causal {best[0] * 1e3:.2f}ms"
            f"  {fl / best[0] / 1e12:.2f} TFLOPS"
        )
        print(f"  对照 mlx autodiff 反向 15.9ms ⇒ 加速 {15.9e-3 / best[0]:.2f}x")


if __name__ == "__main__":
    main()
