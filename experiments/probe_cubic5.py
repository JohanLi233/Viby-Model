"""cubic5（NVIDIA arXiv 2606.00371）系数推导 + 数值/墙钟验证。

论文闭式推导（§2.2 Chen–Chow scaled cubic）：
  f_t(x) = (u/2)(3α_t x − α_t³ x³)，α_t = 1/k_t，
  k_t² = (r_t² + r_t·l_t + l_t²)/3，u = 1.3，
  l₀ = 7e-3（bf16 有效下界），r₀ = 1，t≥1 后 r_t = u，
  l_{t+1} = f_t(l_t)（端点等值性质：f(l_t)=f(r_t)）。
  ⇒ a_t = 3uα_t/2，b_t = −uα_t³/2，每步 X ← aX + b(XXᵀ)X（2 GEMM，
  比 quintic 少一个 Gram 平方项）。5 步后 l 达到松弛带下界 0.7。

本脚本：闭式生成系数 → 自检 l 轨迹 → MLX 迭代 → 与 NS5/PE5 同口径
对比残差 + 生产形状墙钟。

用法: .venv/bin/python experiments/probe_cubic5.py
"""

import os
import sys
import time
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from probe_polar_express import COEFFS, NS_CLASSIC, iterate, metrics, make_momentum

U_PEAK = 1.3
L0 = 7e-3


def cubic5_coeffs(steps=5, l0=L0, u=U_PEAK):
    coeffs = []
    l, r = l0, 1.0
    for t in range(steps):
        if t > 0:
            r = u
        k2 = (r * r + r * l + l * l) / 3
        alpha = 1.0 / sqrt(k2)
        a = 1.5 * u * alpha
        b = -0.5 * u * alpha**3
        coeffs.append((a, b))
        # 端点等值：新下界 = f(l)
        l = a * l + b * l**3
    return coeffs, l


CUBIC5, L_FINAL = cubic5_coeffs()


def iterate_cubic(M, coeffs_seq, bf16):
    """X ← aX + b(XXᵀ)X，每步 2 GEMM（无 Gram 平方项）。"""
    X = M.astype(mx.bfloat16) if bf16 else M.astype(mx.float32)
    dt0 = X.dtype
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.swapaxes(-1, -2)
    nrm = mx.linalg.norm(X.astype(mx.float32))
    X = X / (nrm + 1e-7).astype(dt0)
    for a, b in coeffs_seq:
        A = X @ X.swapaxes(-1, -2)
        X = a * X + b * (A @ X)
    if tr:
        X = X.swapaxes(-1, -2)
    return X.astype(mx.float32)


def bench(fn, X, iters=10):
    for _ in range(3):
        mx.eval(fn(X))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(X)
    mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print("cubic5 系数（a, b）与下界轨迹:")
    l, r = L0, 1.0
    for t, (a, b) in enumerate(CUBIC5):
        print(f"  t={t}: a={a:.6f} b={b:.6f}   [l={l:.5f}, r={r:.3f}]")
        if t > 0:
            r = U_PEAK
        l = a * l + b * l**3
    print(f"  5 步后 l = {L_FINAL:.4f}（目标 ≥0.7）\n")

    print("精度对比（残差 ||DᵀD−I||_F/√n，中位 over 5 seeds）:")
    variants = [
        ("NS5-bf16   ", lambda M: iterate(M, [NS_CLASSIC] * 5, True, 1.0)),
        ("PE5-bf16   ", lambda M: iterate(M, COEFFS[:5], True, 1.01)),
        ("cubic5-bf16", lambda M: iterate_cubic(M, CUBIC5, True)),
        ("cubic5-f32 ", lambda M: iterate_cubic(M, CUBIC5, False)),
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
    print("\nσ 范围（lowrank 640x384）:")
    M = make_momentum(7, 640, 384, "lowrank")
    for name, fn in variants:
        res, smax, smin = metrics(fn(M))
        print(f"  {name}: σ∈[{smin:.3f},{smax:.3f}] res={res:.3e}")

    print("\n墙钟（生产批量形状，bf16，ms/call）:")
    for N, r, c in [(2304, 384, 384), (2304, 768, 384), (59, 768, 768)]:
        X = mx.random.normal((N, r, c), dtype=mx.bfloat16)
        mx.eval(X)
        t_ns5 = bench(lambda x: iterate(x, [NS_CLASSIC] * 5, True, 1.0), X)
        t_pe5 = bench(lambda x: iterate(x, COEFFS[:5], True, 1.01), X)
        t_cb5 = bench(lambda x: iterate_cubic(x, CUBIC5, True), X)
        print(
            f"  ({N},{r},{c}): NS5 {t_ns5:.1f}  PE5 {t_pe5:.1f}  "
            f"cubic5 {t_cb5:.1f}  (cubic5/NS5 = {t_cb5 / t_ns5:.2f})"
        )


if __name__ == "__main__":
    main()
