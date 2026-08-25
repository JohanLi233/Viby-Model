"""① Polar Express 最优系数 vs 经典 NS5：同 5 步、同 GEMM 数的精度对比。

Polar Express（arXiv 2505.16932, Amsel/Persson/Musco/Gower）把 NS 的固定
五次数系数换成逐迭代 minimax 最优系数：在 [l,u] 上对常数 1 的最优奇次
五次逼近（简化 Remez 解出），每迭代后按 p(l) 收缩区间再解下一步。
论文口径：同 5 步下正交化误差显著低于 Keller-Jordan 固定系数；bf16 下
靠 safety_factor=1.01 与 cushion 保稳定。系数生成代码逐行移植自
github.com/NoahAmsel/PolarExpress 的 polar_express.py（MIT）。

本脚本：生成系数 → MLX 复现迭代 → 在类动量谱矩阵上对比
NS5-f32 / NS5-bf16 / PE5-f32 / PE5-bf16 / PE6-bf16 的
||DᵀD−I||_F/√n 残差与奇异值范围。同步数同 GEMM 数，纯精度对比。

用法: .venv/bin/python experiments/probe_polar_express.py
"""

import os
import sys
from math import inf, sqrt

import numpy as np
from numpy.polynomial import Polynomial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx


# ---- 系数生成（逐行移植 polar_express.py，只依赖 numpy） ----


def optimal_quintic(l, u):
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u**3), (3 / 8) / (u**5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array(
            [
                [l, l**3, l**5, 1],
                [q, q**3, q**5, -1],
                [r, r**3, r**5, 1],
                [u, u**3, u**5, -1],
            ]
        )
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c)) / (10 * c)
        )
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, degree=5, safety_factor_eps=0, cushion=0):
    assert degree == 5
    u = 1.0
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for it in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        p = Polynomial.identity() * Polynomial((a, b, c))(Polynomial.identity() ** 2)
        if cushion * u > l:
            p *= 2 / (p(l) + p(u))
        if it < num_iters - 1:
            p = p(Polynomial.identity() / safety_factor)
        coefficients.append(tuple(float(x) for x in p.coef[1::2]))
        l = float(p(l))
        u = 2 - l
    return coefficients


COEFFS = optimal_composition(
    l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02
)
NS_CLASSIC = (3.4445, -4.7750, 2.0315)


# ---- MLX 迭代实现（与 trainer/muon.py 的 std 形式同构） ----


def iterate(M, coeffs_seq, bf16, norm_scale):
    X = M.astype(mx.bfloat16) if bf16 else M.astype(mx.float32)
    dt0 = X.dtype
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.swapaxes(-1, -2)
    nrm = mx.linalg.norm(X.astype(mx.float32))
    X = X / (nrm * norm_scale + 1e-7).astype(dt0)
    for a, b, c in coeffs_seq:
        A = X @ X.swapaxes(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.swapaxes(-1, -2)
    return X.astype(mx.float32)


def ns5(M, bf16):
    return iterate(M, [NS_CLASSIC] * 5, bf16, norm_scale=1.0)


def pe(M, steps, bf16):
    return iterate(M, COEFFS[:steps], bf16, norm_scale=1.01)


def metrics(D):
    r, c = D.shape
    n = min(r, c)
    G = D.T @ D if r >= c else D @ D.T
    res = float(mx.linalg.norm(mx.eye(n) - G)) / sqrt(n)
    _, S, _ = mx.linalg.svd(D, stream=mx.cpu)
    mx.eval(S)
    return res, float(S[0]), float(S[-1])


def make_momentum(seed, r, c, kind):
    rng = np.random.default_rng(seed)
    if kind == "gauss":
        M = rng.standard_normal((r, c))
    elif kind == "lowrank":  # 类动量 EMA：低秩骨架 + 噪声，条件数 ~300
        k = 64
        A = rng.standard_normal((r, k))
        B = rng.standard_normal((c, k))
        M = A @ B.T / sqrt(k) + 0.3 * rng.standard_normal((r, c))
    else:  # 近死神经元：几个超大奇异值 + 长尾小谱
        M = rng.standard_normal((r, c)) * 0.02
        U = rng.standard_normal((r, 4))
        V = rng.standard_normal((c, 4))
        M += U @ np.diag([3.0, 2.0, 1.5, 1.0]) @ V.T
    return mx.array(M.astype(np.float32))


def main():
    print("Polar Express 系数（前 6 组 a, b, c）:")
    for i, abc in enumerate(COEFFS[:6]):
        print(f"  t={i}: ({abc[0]:.6f}, {abc[1]:.6f}, {abc[2]:.6f})")
    print(f"经典 NS 系数: {NS_CLASSIC}\n")

    shapes = [(640, 384), (384, 384), (768, 768)]
    kinds = ["gauss", "lowrank", "spiky"]
    variants = [
        ("NS5-f32 ", lambda M: ns5(M, False)),
        ("NS5-bf16", lambda M: ns5(M, True)),
        ("PE5-f32 ", lambda M: pe(M, 5, False)),
        ("PE5-bf16", lambda M: pe(M, 5, True)),
        ("PE6-bf16", lambda M: pe(M, 6, True)),
    ]
    header = f"{'矩阵':<22}" + "".join(f"{name:>14}" for name, _ in variants)
    print(header + "   (残差 ||DᵀD−I||_F/√n，越小越好)")
    for r, c in shapes:
        for kind in kinds:
            M = make_momentum(hash((r, c, kind)) % 10000, r, c, kind)
            row = f"{kind:>8} {r}x{c}:"
            for _, fn in variants:
                res, smax, smin = metrics(fn(M))
                row += f"{res:>14.3e}"
            print(row)
    print("\n奇异值范围抽查（lowrank 640x384，σ_max/σ_min，理想都→1）:")
    M = make_momentum(7, 640, 384, "lowrank")
    for name, fn in variants:
        res, smax, smin = metrics(fn(M))
        print(f"  {name}: σ∈[{smin:.4f}, {smax:.4f}]  res={res:.3e}")


if __name__ == "__main__":
    main()
