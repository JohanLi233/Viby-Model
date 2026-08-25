"""攻关项③收尾：量化 Newton 细化 warm-start 失败的机制（κ² 放大）。

（本文件由初版盆地/回退实验改写；初版结论：修正种子公式 bug 后
仍 100% 回退。）

已证：数学本身正确（精确种子 0 迭代自洽，残差 1.2e-4），但跨步 warm-start
100% 回退。本脚本验证机制：warm 残差 ||I − G₁G₀^{-1}|| ≈ ||ΔG·G₀^{-1}||
按 κ(G) = κ(X)² 放大——逆平方根恰恰把每步旋转最快的小 σ 方向放大，
动量 EMA 每步 ~1e-3 的相对旋转 × κ(G)~1e5 ⇒ warm 残差 ≫ Newton 盆地
（特征值须落在 (0,3)）。这与 r082 stale-D 失败同源：任何「逆因子复用」
都对条件数敏感。

用法: .venv/bin/python experiments/probe_ns_refine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

BETA = 0.95
R, C = 640, 384
T = 120
BURNIN = 40  # 跳过 EMA 冷启动段，测稳态


def fro(x):
    return mx.linalg.norm(x.astype(mx.float32))


def main():
    mx.random.seed(0)
    A = mx.random.normal((R, 64))
    B = mx.random.normal((C, 64))
    M = mx.zeros((R, C))
    rows = []
    X_prev, _Y_prev = None, None
    for t in range(T):
        if t % 20 == 19:
            A = 0.97 * A + 0.03 * mx.random.normal((R, 64))
            B = 0.97 * B + 0.03 * mx.random.normal((C, 64))
        Gt = A @ B.T / 8 + 0.3 * mx.random.normal((R, C))
        Mn = BETA * M + (1 - BETA) * Gt
        mx.eval(Mn)
        M = Mn
        X = (M / (fro(M) + 1e-7)).astype(mx.float32)
        if t < BURNIN:
            X_prev = X
            continue
        # 上一步精确 G^{-1/2}（eigh，CPU）
        G0 = X_prev.T @ X_prev
        w, V = mx.linalg.eigh(G0, stream=mx.cpu)
        w = mx.maximum(w, 1e-12)
        Y0 = (V / mx.sqrt(w)) @ V.T
        G1 = X.T @ X
        warm_res = float(fro(mx.eye(C) - G1 @ (Y0 @ Y0))) / (C**0.5)
        # 谱与旋转
        _, S, _ = mx.linalg.svd(X, stream=mx.cpu)
        mx.eval(S)
        kappa = float(S[0] / mx.maximum(S[-1], 1e-12))
        dX = float(fro(X - X_prev) / (fro(X_prev) + 1e-12))
        rows.append((warm_res, kappa, dX))
        X_prev = X

    import statistics as st

    wr = [r[0] for r in rows]
    kp = [r[1] for r in rows]
    dx = [r[2] for r in rows]
    print(f"稳态 {len(rows)} 步（BURNIN={BURNIN}）:")
    print(
        f"  warm 残差 ||I−G₁Y₀²||/√n: 中位 {st.median(wr):.2e}, "
        f"min {min(wr):.2e}, max {max(wr):.2e}"
    )
    print("  （Newton 盆地要求特征值∈(0,3)，warm 残差须 ≲1 才有戏）")
    print(f"  κ(X): 中位 {st.median(kp):.0f}, max {max(kp):.0f}")
    print(f"  每步相对旋转 ||ΔX||/||X||: 中位 {st.median(dx):.2e}")
    print(
        f"  机制核对: 中位旋转 × κ² = {st.median(dx) * st.median(kp) ** 2:.1e} "
        f"vs warm 残差中位 {st.median(wr):.1e}"
    )


if __name__ == "__main__":
    main()
