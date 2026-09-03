"""目标形状 flash 反向分块：Dk=112 Dv=64 B12 H12 T1024。

默认 STR=16 是旧形状（Dk=128 Dv=96）的 32KB 上限产物。本脚本扫
NT×STR，含 STR=24（lcm 与 1024 不对齐、会 pad）。

用法: .venv/bin/python experiments/sweep_flash_target.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_fused as af

B, H, T, DK, DV = 12, 12, 1024, 112, 64


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    scale = DK**-0.5
    o, lse = af.flash_forward(q, k, v, scale, None)
    mx.eval(q, k, v, do, o, lse)
    half = B * H * (T * T / 2)
    fl = half * DK * 2 * 4 + half * DV * 2 * 3  # bwd approx
    print(f"B={B} H={H} T={T} Dk={DK} Dv={DV}  bwd ~{fl / 1e9:.1f} GFLOP")
    print(
        f"{'NT':>6}{'RES':>5}{'STR':>5}{'align':>6}{'tgKB':>7}{'fwd':>8}{'bwd':>8}{'f+b':>8}{'TF_b':>7}"
    )
    best = None
    for nt in (128, 256, 512):
        for strb in (8, 16, 24, 32):
            if not af._split_ok(DK, DV, nt):
                continue
            if not af.tile_ok(DK, DV, nt, strb):
                continue
            res = af._res_split(nt)
            align = af._align(nt, strb, True)
            strl = af._lse_str(nt, strb)
            tg = max(
                af._split.tgmem(kd, DK, DV, nt, strb, strl)
                for kd in ("lse", "dq", "dkv")
            )
            tg = max(tg, af._split.tgmem_combined(DK, DV, nt, strb))
            fstr = af._split.fwd_str(DK, DV, nt)
            try:
                for _ in range(2):
                    mx.eval(af.flash_forward(q, k, v, scale, None, nt, fstr))
                    mx.eval(
                        af.flash_backward(
                            q, k, v, o, do, scale, None, nt, strb, lse=lse
                        )
                    )
            except Exception as e:
                print(f"{nt:>6}{res:>5}{strb:>5}  fail {e}")
                continue
            tf, tb = [], []
            for _ in range(5):
                t0 = time.perf_counter()
                mx.eval(af.flash_forward(q, k, v, scale, None, nt, fstr))
                t1 = time.perf_counter()
                mx.eval(
                    af.flash_backward(q, k, v, o, do, scale, None, nt, strb, lse=lse)
                )
                t2 = time.perf_counter()
                tf.append(t1 - t0)
                tb.append(t2 - t1)
            f = statistics.median(tf) * 1e3
            b = statistics.median(tb) * 1e3
            print(
                f"{nt:>6}{res:>5}{strb:>5}{align:>6}{tg / 1024:>7.1f}{f:>8.2f}{b:>8.2f}"
                f"{f + b:>8.2f}{fl / (b / 1e3) / 1e12:>7.2f}"
            )
            if best is None or f + b < best[0]:
                best = (f + b, nt, strb, f, b)
    print(
        f"\n最优 NT={best[1]} STR={best[2]}  fwd {best[3]:.2f} bwd {best[4]:.2f} f+b {best[0]:.2f}ms"
    )


if __name__ == "__main__":
    main()
