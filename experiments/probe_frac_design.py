"""谱指数 Muon 定稿设计：Chebyshev-on-Gram 单发实现正则化幂律谱变换。

目标谱变换（自研的两区族，非 Muon-p 的纯幂律）：
    t(σ) = σ·(σ²+s)^{(p-1)/2}
  - σ ≪ √s：t ≈ σ·s^{(p-1)/2}（线性地板，不硬拉极小 σ——实测小 σ 抬升
    有害：PE 0.86→+0.14 nat，classic 0.47→基准，cubic5 0.12→−0.08 nat）
  - σ ≫ √s：t ≈ σ^p（幂律主体，保大 σ 序信息；p=0 退化为 soft-polar）
  单调、C∞、尺度协变（幂律区 (cσ)^p=c^pσ^p，F-范数归一不改变形状）。

实现：h(t)=(t+s)^q 在 t∈[0,1] 做 Chebyshev 拟合（系数小且衰减，bf16
安全），矩阵评估用 T_j 三项递推（每度 1 个 Gram 空间 GEMM）：
    Ã=2G−I；T₀=I, T₁=Ã, T_{j+1}=2ÃT_j−T_{j-1}；H=Σc_jT_j(Ã)；D=X·H(G)
deg-k 总成本 k+2 个主 GEMM（cubic5 是 10，classic 是 15）。

用法: .venv/bin/python experiments/probe_frac_design.py
"""

import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

L0 = 7e-3


def design_cheb(p, s, k, n_grid=20000):
    """h(t)=(t+s)^q 的 deg-k Chebyshev 拟合，t∈[0,1]。返回 T_j(2t−1) 系数。"""
    q = (p - 1.0) / 2.0
    u = np.cos(np.pi * (np.arange(n_grid) + 0.5) / n_grid)
    t = 0.5 + 0.5 * u
    g = (t + s) ** q
    coef = C.chebfit(2 * t - 1, g, k)  # 以 T_j(2t−1) 为基
    # x 空间绝对误差：max |x·h(x²) − x(x²+s)^q|
    err = np.abs(C.chebval(2 * t - 1, coef) - g) * np.sqrt(t)
    return coef, float(err.max())


def frac_apply_cheb(M, coef, bf16=True):
    """D = X·H(G)，H = Σ c_j T_j(2G−I)，三项递推，bf16 安全。"""
    coef = [float(c) for c in coef]
    X = M.astype(mx.bfloat16) if bf16 else M.astype(mx.float32)
    dt0 = X.dtype
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.swapaxes(-1, -2)
    nrm = mx.linalg.norm(X.astype(mx.float32))
    X = X / (nrm + 1e-7).astype(dt0)
    G = X @ X.swapaxes(-1, -2)
    n = G.shape[-1]
    eye = mx.eye(n, dtype=dt0)
    Ah = 2 * G - eye
    T_prev, T_cur = eye, Ah
    H = coef[0] * T_prev + (coef[1] * T_cur if len(coef) > 1 else 0)
    for j in range(1, len(coef) - 1):
        T_next = 2 * (Ah @ T_cur) - T_prev
        H = H + coef[j + 1] * T_next
        T_prev, T_cur = T_cur, T_next
    H @ X if not tr else None
    # tr 时 X 是 (c,r)，Dᵀ = H@X，转回即可
    out = H @ X
    if tr:
        out = out.swapaxes(-1, -2)
    return out.astype(mx.float32)


def realized_check(p, s, k):
    r, c = 640, 384
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((r, c)))
    V, _ = np.linalg.qr(rng.standard_normal((c, c)))
    sg = np.logspace(np.log10(L0), 0, c)
    M = U @ np.diag(sg) @ V.T
    M = M / np.linalg.norm(M)
    coef, ferr = design_cheb(p, s, k)
    D = np.array(frac_apply_cheb(mx.array(M.astype(np.float32)), coef, bf16=True))
    sout = np.linalg.svd(D, compute_uv=False)
    sin = np.linalg.svd(M, compute_uv=False)
    tgt = sin * (sin**2 + s) ** ((p - 1.0) / 2.0)
    ae = np.abs(sout - tgt)
    bulk = sin >= 0.05
    rb = ae[bulk] / np.maximum(tgt[bulk], 1e-12)
    mono = bool(np.all(np.diff(sout[::-1]) >= -1e-3))
    return ferr, ae.max(), np.median(rb), mono, coef


def main():
    print("Chebyshev 设计（x 空间绝对误差 / 最大系数 / GEMM=k+2）:")
    cands = []
    for p, s in ((0.5, 1e-4), (0.25, 1e-4), (0.5, 1e-2), (0.0, 1e-4)):
        for k in (4, 6, 8):
            coef, ferr = design_cheb(p, s, k)
            print(
                f"  p={p:.2f} s={s:g} deg={k}: max|Δ| {ferr:.2e}  "
                f"max|c| {np.max(np.abs(coef)):.2f}  GEMM {k + 2}"
            )
            cands.append((p, s, k, coef, ferr))
    print("\nbf16 矩阵级实现验证:")
    for p, s, k in ((0.5, 1e-4, 8), (0.25, 1e-4, 8), (0.5, 1e-2, 6)):
        ferr, amax, rmed, mono, coef = realized_check(p, s, k)
        print(
            f"  p={p:.2f} s={s:g} deg={k}: 拟合 {ferr:.2e} | bf16 max|Δ| "
            f"{amax:.2e} | bulk rel 中位 {rmed:.2e} | 单调 {mono}"
        )
        print(f"    coefs = {[f'{c:.6f}' for c in coef]}")


if __name__ == "__main__":
    main()
