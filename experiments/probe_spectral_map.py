"""谱指数 Muon 设计工具：现有方法的有效谱图 + x^p 单发 minimax 多项式设计。

背景：实测质量排序对「谱均衡强度」非单调（无 NS 5.669 ≫ cubic5 5.245
< classic 5.328 < PE5 5.465），最优点在完全均衡（polar, p=0）与完全
不均衡（归一化, p=1）之间。Muon-p（arXiv 2606.13867）提出 US^pVᵀ 并
证明固定单变量多项式迭代无法收敛到分数幂——其方案是双变量递推
（f(x,y)=x+c(y−x³)，2 GEMM/步，小 σ 处定点吸引子导数→1，截断下
小 σ 远未收敛）。本脚本走他们定理管不到的路线：**非迭代、一发成型**
的 minimax 奇多项式 p(x)≈x^p（松弛带哲学：只要求 bf16 噪声地板内
的相对精度，不求收敛）。

Part A: classic/PE5/cubic5 的标量复合谱图 + 有效指数拟合（log-log 斜率）。
Part B: Lawson 加权迭代设计 h(t)≈t^{(p-1)/2}（deg k），p(x)=x·h(x²)，
        报告 [l0,1] 上的最大相对误差（按目标加权）。
Part C: MLX bf16 实测 realized map vs 目标（矩阵级，含数值噪声）。

用法: .venv/bin/python experiments/probe_spectral_map.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

from probe_polar_express import COEFFS, NS_CLASSIC
from probe_cubic5 import CUBIC5

L0 = 7e-3  # bf16 有效下界（cubic5 论文口径）


# ---------- Part A: 现有方法的标量谱图 ----------


def scalar_map(coeffs_seq, norm_scale, x):
    """quintic/cubic 复合的标量版：x ← a x + b x³ + c x⁵。"""
    x = x / norm_scale
    for abc in coeffs_seq:
        a, b = abc[0], abc[1]
        c = abc[2] if len(abc) > 2 else 0.0
        x = a * x + b * x**3 + c * x**5
    return x


def eff_exponent(xs, ys, lo=0.03, hi=0.5):
    """bulk 区间内的 log-log 斜率（有效均衡指数 p_eff：0=polar，1=归一化）。"""
    m = (xs >= lo) & (xs <= hi)
    lx, ly = np.log(xs[m]), np.log(np.maximum(ys[m], 1e-12))
    # 逐点差分斜率的中位数，抗端点平台
    sl = np.diff(ly) / np.diff(lx)
    return float(np.median(sl)), float(np.polyfit(lx, ly, 1)[0])


def part_a():
    xs = np.logspace(-4, 0, 400)
    methods = {
        "classic": ([NS_CLASSIC] * 5, 1.0),
        "PE5    ": (COEFFS[:5], 1.01),
        "cubic5 ": (CUBIC5, 1.0),
    }
    print("Part A: 有效谱图（标量复合，输入 σ 已 F-范数归一）")
    print(f"{'方法':<8} {'p_eff(中位)':>12} {'p_eff(LSQ)':>11}  关键点 t(σ):")
    for name, (seq, ns) in methods.items():
        ys = scalar_map(seq, ns, xs)
        med, lsq = eff_exponent(xs, ys)
        pts = "  ".join(
            f"t({s:g})={float(scalar_map(seq, ns, np.array([s]))[0]):.3f}"
            for s in (1e-3, 1e-2, 0.1, 0.5, 1.0)
        )
        print(f"{name:<8} {med:>12.3f} {lsq:>11.3f}  {pts}")
    print("  参考：t(σ)=σ^p 时 p_eff=p；p=0 即 polar 全平，p=1 即不归一化\n")


# ---------- Part B: Lawson minimax 设计 ----------


def design_frac(p, k, s=1e-4, iters=400, n_grid=8000):
    """奇多项式 p(x)=x·h(x²)（h deg k）逼近正则化幂律 x·(x²+s)^{(p-1)/2}。

    s>0 消掉 0 点奇点：x≪√s 时 t≈x·s^{(p-1)/2}（比例区），x≫√s 时
    t≈x^p（幂律区），√s 是膝盖。s→0 退化为 Muon-p 的 x^p；p=0 即
    soft-polar（高通平坦、低通比例）——正是 cubic5 胜出形状的连续化。
    口径：x 空间绝对 minimax；Chebyshev 节点 + 阻尼 Lawson。
    """
    q = (p - 1.0) / 2.0
    u = np.cos(np.pi * (np.arange(n_grid) + 0.5) / n_grid)
    x = 0.5 + 0.5 * u  # [0,1] 全域（无奇点，从 0 起拟合）
    t = x * x
    tgt = (t + s) ** q
    w_fit = x  # |x·h(x²) − x(x²+s)^q| = x·|h(t) − (t+s)^q|
    V = np.stack([t**i for i in range(k + 1)], axis=1)
    w = np.ones(n_grid)
    coef = None
    for _ in range(iters):
        W = (w * w_fit)[:, None] * V
        coef = np.linalg.lstsq(W, (w * w_fit) * tgt, rcond=None)[0]
        err = np.abs((V @ coef - tgt) * w_fit)
        w = w * np.sqrt(err + 1e-15)
        w /= w.sum()
    maxabs = float(err.max())
    return coef, maxabs


def part_b():
    print(
        "Part B: 单发奇多项式 p(x)=x·h(x²)，目标正则化幂律 x(x²+s)^{(p-1)/2}"
        "（x 空间绝对 minimax）"
    )
    print(f"{'p':>5} {'s':>8} {'deg':>4} {'GEMM':>5} {'max abs':>10} {'max|coef|':>10}")
    for p, s in ((0.0, 1e-4), (0.25, 1e-4), (0.5, 1e-4), (0.5, 1e-2), (0.7, 1e-4)):
        for k in (2, 3, 4):
            coef, mr = design_frac(p, k, s)
            gemm = k + 2  # A + k 次 Horner@A + 最后 H@X
            print(
                f"{p:>5.2f} {s:>8.0e} {2 * k + 1:>4} {gemm:>5} {mr:>10.2e} "
                f"{float(np.max(np.abs(coef))):>10.2f}  "
                + " ".join(f"{c:.4f}" for c in coef)
            )
    print()


# ---------- Part C: MLX bf16 realized map ----------


def frac_apply(M, coef, bf16=True):
    """X·h(XXᵀ)，h 为 deg-k 多项式（Horner 在 Gram 上）。"""
    coef = [float(c) for c in coef]
    X = M.astype(mx.bfloat16) if bf16 else M.astype(mx.float32)
    dt0 = X.dtype
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.swapaxes(-1, -2)
    nrm = mx.linalg.norm(X.astype(mx.float32))
    X = X / (nrm + 1e-7).astype(dt0)
    A = X @ X.swapaxes(-1, -2)
    eye = mx.eye(A.shape[-1], dtype=dt0)
    H = coef[-1] * eye
    for c in reversed(coef[:-1]):
        H = H @ A + c * eye
    X = H @ X
    if tr:
        X = X.swapaxes(-1, -2)
    return X.astype(mx.float32)


def part_c(picks):
    print("Part C: bf16 realized map vs 目标正则化幂律（矩阵级，U diag(σ) Vᵀ）")
    r, c = 640, 384
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((r, c)))
    V, _ = np.linalg.qr(rng.standard_normal((c, c)))
    s = np.logspace(np.log10(L0), 0, c)
    M = U @ np.diag(s) @ V.T
    M = M / np.linalg.norm(M)
    Mx = mx.array(M.astype(np.float32))
    for p, k, s_reg in picks:
        coef, mr = design_frac(p, k, s_reg)
        D = np.array(frac_apply(Mx, coef, bf16=True))
        sout = np.linalg.svd(D, compute_uv=False)
        sin = np.linalg.svd(M, compute_uv=False)
        tgt = sin * (sin**2 + s_reg) ** ((p - 1.0) / 2.0)
        abs_err = np.abs(sout - tgt)
        bulk = sin >= 0.05
        rel_bulk = abs_err[bulk] / np.maximum(tgt[bulk], 1e-12)
        mono = bool(np.all(np.diff(sout[::-1]) >= -1e-3))
        print(
            f"  p={p:.2f} s={s_reg:g} deg={2 * k + 1}: 设计 max|Δ| {mr:.2e}；bf16 "
            f"max|Δ| {abs_err.max():.2e}，bulk rel 中位 "
            f"{np.median(rel_bulk):.2e} p90 {np.quantile(rel_bulk, 0.9):.2e}，"
            f"单调 {mono}"
        )


if __name__ == "__main__":
    part_a()
    part_b()
    part_c([(0.5, 3, 1e-4), (0.0, 3, 1e-4)])
