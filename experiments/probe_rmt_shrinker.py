"""RMT 最优谱映射 t* 的数值验证（理论推导见 research/SPECTRAL_THEORY.md）。

理论：动量 = 持久信号 + 新鲜噪声的 spiked 模型下，等变无状态更新规则的
最优谱映射有闭式（shrink-then-whiten / Wiener-polar）：
    t*(σ) = cosθ_u(σ)·cosθ_v(σ)   （σ > λ₊ 时；σ ≤ λ₊ 置 0）
其中 cosθ 是 BGN（Benaych-Georges–Nadakuditi 2012）奇异向量重合度，
λ₊ = √w·(1/√m + 1/√n) 是 F-归一口径下的 Marchenko–Pastur 噪声 bulk 上边缘
（w = 动量里噪声的 Frobenius 能量占比）。

Part A  合成 spiked 矩阵对拍 BGN 渐近公式（σ_obs 与左右奇异向量重合度）
Part B  生产形状的 λ₊ 表 + t* vs classic/PE5/cubic5/cubic5b05 标量谱图对比
Part C  EMA 动量玩具模型：奇异方向-信号子空间重合度是否在预测 edge 坍缩，
        以及动量噪声水平是否符合 ESS=(1+β)/(1−β) 预测
Part D  平滑 t* 的 Chebyshev-on-Gram 单发拟合（deg vs 误差 vs 噪声泄漏）

用法: .venv/bin/python experiments/probe_rmt_shrinker.py
"""

import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_polar_express import COEFFS, NS_CLASSIC  # noqa: E402
from probe_cubic5 import CUBIC5  # noqa: E402
from probe_spectral_map import scalar_map  # noqa: E402
from trainer.muon import _CUBIC5B05_COEFFS  # noqa: E402


# ---------- BGN 渐近公式（口径：噪声 G/√n，G iid N(0,1)，c = m/n ≥ 1）----------


def bgn_sigma_obs(theta, c):
    """spike 强度 θ（> c^{1/4}）对应的观测奇异值。"""
    return np.sqrt((1.0 + theta**2) * (c + theta**2)) / theta


def bgn_overlaps(theta, c):
    """左（长边 m）/右（短边 n）奇异向量重合度 cos²θ_u, cos²θ_v。"""
    t4 = theta**4
    return (t4 - c) / (t4 + c * theta**2), (t4 - c) / (t4 + theta**2)


def theta_hat(sig_scaled, c):
    """由观测 σ（噪声单位）反演 spike 强度 θ̂；bulk 内返回 nan。"""
    y = sig_scaled**2
    disc = (y - 1.0 - c) ** 2 - 4.0 * c
    th2 = np.where(
        disc >= 0, ((y - 1.0 - c) + np.sqrt(np.maximum(disc, 0))) / 2, np.nan
    )
    with np.errstate(invalid="ignore"):
        return np.sqrt(th2)


def t_star(sigma, m, n, w):
    """F-归一口径（Σσ²=1）下的最优谱映射。w = 噪声 Frobenius 能量占比。"""
    m, n = max(m, n), min(m, n)
    c = m / n
    scale = np.sqrt(w / m)  # = σ_e·√n，σ_e² = w/(mn)
    s = np.asarray(sigma, dtype=np.float64) / scale
    edge = 1.0 + np.sqrt(c)
    th = theta_hat(s, c)
    with np.errstate(invalid="ignore"):
        val = (th**4 - c) / (th**3 * s)
    return np.where(np.asarray(sigma) / scale > edge, np.nan_to_num(val), 0.0)


def lam_plus(m, n, w):
    return np.sqrt(w) * (1.0 / np.sqrt(m) + 1.0 / np.sqrt(n))


# ---------- Part A ----------


def part_a():
    m, n = 640, 384
    c = m / n
    thc = c**0.25
    print(f"Part A: BGN 公式对拍（m={m}, n={n}, c={c:.3f}, 8 seeds 均值）")
    print(
        f"{'θ/θc':>6} {'σ理论':>8} {'σ实测':>8} {'φu理论':>8} {'φu实测':>8} "
        f"{'φv理论':>8} {'φv实测':>8}"
    )
    for ratio in (1.05, 1.2, 1.5, 2.5, 5.0):
        th = ratio * thc
        so, fu, fv = [], [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            u = rng.standard_normal(m)
            u /= np.linalg.norm(u)
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            X = th * np.outer(u, v) + rng.standard_normal((m, n)) / np.sqrt(n)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            so.append(S[0])
            fu.append(float(np.dot(U[:, 0], u) ** 2))
            fv.append(float(np.dot(Vt[0], v) ** 2))
        pu, pv = bgn_overlaps(th, c)
        print(
            f"{ratio:>6.2f} {bgn_sigma_obs(th, c):>8.4f} {np.mean(so):>8.4f} "
            f"{pu:>8.4f} {np.mean(fu):>8.4f} {pv:>8.4f} {np.mean(fv):>8.4f}"
        )
    print()


# ---------- Part B ----------

PROD_SHAPES = [
    ("专家栈 gu/dw", 768, 384),
    ("probe 基准", 640, 384),
    ("o_proj", 768, 768),
    ("qkv 合并", 1232, 768),
    ("kv_up", 1536, 192),
    ("per-head 切分", 768, 64),
]


def knee50(map_fn):
    """标量谱图首次越过 0.5 的 σ（二分）。"""
    lo, hi = 1e-5, 1.0
    if map_fn(np.array([hi]))[0] < 0.5:
        return float("nan")
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if map_fn(np.array([mid]))[0] >= 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def part_b():
    print("Part B1: 各生产形状的预测膝盖 λ₊ = √w(1/√m+1/√n)")
    print(f"{'形状':<14} {'w=1.0':>8} {'w=0.3':>8} {'w=0.1':>8} {'w=0.025':>8}")
    for name, m, n in PROD_SHAPES:
        row = "  ".join(f"{lam_plus(m, n, w):>7.4f}" for w in (1.0, 0.3, 0.1, 0.025))
        print(f"{name:<14} {row}")

    m, n = 640, 384
    methods = {
        "classic ": lambda x: scalar_map([NS_CLASSIC] * 5, 1.0, x),
        "PE5     ": lambda x: scalar_map(COEFFS[:5], 1.01, x),
        "cubic5  ": lambda x: scalar_map(CUBIC5, 1.0, x),
        "cubic5b05": lambda x: scalar_map(_CUBIC5B05_COEFFS, 1.0, x),
    }
    print(f"\nPart B2: 现役方法膝盖（t=0.5 交点）vs 预测 λ₊（{m}x{n}）")
    for name, fn in methods.items():
        print(f"  {name} knee50 = {knee50(fn):.4f}")
    for w in (1.0, 0.3, 0.1, 0.025):
        print(f"  λ₊(w={w:<5}) = {lam_plus(m, n, w):.4f}")

    print(f"\nPart B3: t*(σ) vs 现役映射（{m}x{n}，关键 σ 点）")
    pts = np.array([1e-3, 3e-3, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0])
    hdr = "  ".join(f"{p:>7g}" for p in pts)
    print(f"{'σ':<16} {hdr}")
    for w in (0.3, 0.1, 0.025):
        row = "  ".join(f"{v:>7.3f}" for v in t_star(pts, m, n, w))
        print(f"{'t*  (w=' + str(w) + ')':<16} {row}")
    for name, fn in methods.items():
        row = "  ".join(f"{v:>7.3f}" for v in fn(pts))
        print(f"{name:<16} {row}")
    print()


# ---------- Part C ----------


def part_c():
    m, n, k, beta, T = 640, 384, 8, 0.95, 240
    c = m / n
    rng = np.random.default_rng(0)
    Us = np.linalg.qr(rng.standard_normal((m, k)))[0]
    Vs = np.linalg.qr(rng.standard_normal((n, k)))[0]
    thc = c**0.25
    ratios = np.array([0.6, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 6.0])
    th = ratios * thc  # 动量口径 spike 强度（动量噪声单位 σ_e√n = 1）
    ess = (1 + beta) / (1 - beta)
    nu_g = np.sqrt(ess) / np.sqrt(n)  # 梯度噪声 → 动量噪声 1/√n
    M = np.zeros((m, n))
    for _ in range(T):
        G = (Us * th) @ Vs.T + nu_g * rng.standard_normal((m, n))
        M = beta * M + (1 - beta) * G
    resid = M - (Us * th) @ Vs.T * (1 - beta**T)
    print(
        f"Part C: EMA 动量模型（β={beta}, T={T}, ESS 预测噪声 std "
        f"{1 / np.sqrt(n):.5f}，实测 {resid.std() * np.sqrt(n) / np.sqrt(n):.5f}"
        f"，比值 {resid.std() / (1 / np.sqrt(n)):.3f}）"
    )
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    edge = 1.0 + np.sqrt(c)
    print(f"  观测 bulk 上边缘（第 {k + 1} 个 σ）= {S[k]:.4f}，MP 预测 = {edge:.4f}")
    print(f"{'σ_i':>8} {'σ/edge':>8} {'重合度实测':>10} {'t*预测':>8}")
    for i in range(k + 2):
        ov = float(np.linalg.norm(Us.T @ U[:, i]) * np.linalg.norm(Vs.T @ Vt[i]))
        thi = theta_hat(np.array([S[i]]), c)[0]
        pred = 0.0
        if S[i] > edge and np.isfinite(thi):
            pu, pv = bgn_overlaps(thi, c)
            pred = float(np.sqrt(max(pu, 0) * max(pv, 0)))
        print(f"{S[i]:>8.3f} {S[i] / edge:>8.3f} {ov:>10.3f} {pred:>8.3f}")
    print()


# ---------- Part D ----------


def part_d():
    m, n, w = 640, 384, 0.1
    lam = lam_plus(m, n, w)
    sig = np.linspace(1e-4, 1.0, 4000)
    g = t_star(sig, m, n, w)
    # 高斯平滑膝盖（宽度 λ₊/3），硬折点多项式不可逼近
    width = lam / 3
    dx = sig[1] - sig[0]
    kw = max(int(3 * width / dx), 1)
    ker = np.exp(-0.5 * ((np.arange(-kw, kw + 1) * dx) / width) ** 2)
    ker /= ker.sum()
    gs = np.convolve(np.pad(g, kw, mode="edge"), ker, mode="same")[kw:-kw]
    h = gs / np.maximum(sig, 1e-9)
    t = sig**2
    print(f"Part D: 平滑 t*（w={w}, λ₊={lam:.4f}）的 Chebyshev-on-Gram 拟合")
    print(
        f"{'deg(h)':>7} {'主GEMM':>7} {'max|Δ|':>9} {'泄漏(σ<λ₊)':>11} {'max|coef|':>10}"
    )
    for kdeg in (4, 6, 8, 10):
        coef = C.chebfit(2 * t - 1, h, kdeg)
        real = C.chebval(2 * t - 1, coef) * sig
        err = float(np.abs(real - gs).max())
        leak = float(real[sig < lam].max())
        print(
            f"{kdeg:>7} {kdeg + 2:>7} {err:>9.3e} {leak:>11.3f} "
            f"{float(np.abs(coef).max()):>10.2f}"
        )
    print()


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
