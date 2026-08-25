"""①补充：PE 逐迭代系数在不同步数下的精度/墙钟权衡 + 生产形状实测。

问题：PE4 的精度是否 ≥ 经典 NS5（3.4445 系数 5 步）？若是，则同等训练
质量下 NS 墙钟 −20%（系数替换零成本，少一步迭代）。这是「更优多项式」
而不是「降低精度」——但仍需训练 probe 实证，此处先给数值口径。

另测生产批量形状下 std/gram 两种形式的墙钟基线。

用法: .venv/bin/python experiments/probe_pe_wallclock.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from probe_polar_express import COEFFS, NS_CLASSIC, iterate, metrics, make_momentum


def bench(fn, X, iters=10):
    for _ in range(3):
        mx.eval(fn(X))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(X)
    mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def batched_iterate(M, coeffs_seq, bf16):
    """生产 std 形式（批量 (N,r,c)）。"""
    return iterate(
        M, coeffs_seq, bf16, norm_scale=1.0 if coeffs_seq[0] == NS_CLASSIC else 1.01
    )


def main():
    # ---- 精度：PE4 / PE5 / NS5 ----
    print("精度对比（残差 ||DᵀD−I||_F/√n，中位 over 5 seeds，gauss/lowrank 混合）:")
    variants = [
        ("NS5-bf16", lambda M: iterate(M, [NS_CLASSIC] * 5, True, 1.0)),
        ("PE4-bf16", lambda M: iterate(M, COEFFS[:4], True, 1.01)),
        ("PE5-bf16", lambda M: iterate(M, COEFFS[:5], True, 1.01)),
    ]
    for kind in ("gauss", "lowrank"):
        for r, c in ((640, 384), (384, 384), (768, 768)):
            row = f"  {kind:>8} {r}x{c}:"
            for _, fn in variants:
                res = []
                for s in range(5):
                    M = make_momentum(s * 1000 + r + c, r, c, kind)
                    res.append(metrics(fn(M))[0])
                res.sort()
                row += f"  {res[2]:.3e}"
            print(row)

    # ---- 墙钟：生产形状 ----
    print("\n墙钟（生产批量形状，bf16，std 形式，ms/call）:")
    shapes = [(2304, 384, 384), (2304, 768, 384), (59, 768, 768), (2, 6400, 768)]
    for N, r, c in shapes:
        X = mx.random.normal((N, r, c), dtype=mx.bfloat16)
        mx.eval(X)
        t_ns5 = bench(lambda x: batched_iterate(x, [NS_CLASSIC] * 5, True), X)
        t_pe5 = bench(lambda x: batched_iterate(x, COEFFS[:5], True), X)
        t_pe4 = bench(lambda x: batched_iterate(x, COEFFS[:4], True), X)
        print(
            f"  ({N},{r},{c}): NS5 {t_ns5:.1f}  PE5 {t_pe5:.1f}  "
            f"PE4 {t_pe4:.1f}  (PE4/NS5 = {t_pe4 / t_ns5:.2f})"
        )


if __name__ == "__main__":
    main()
