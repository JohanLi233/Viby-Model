#!/usr/bin/env python3
"""
SPEC v1.1 数值复算脚本。

用法:
    python research/compute_isoquant.py          # 输出全部参考数值
    python research/compute_isoquant.py --json   # JSON 输出
"""
from __future__ import annotations

import argparse
import itertools
import json
import math


# ---- 拟合常数 -------------------------------------------------------------
HOFFMANN = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
BESIROGLU = dict(E=1.82, A=482.0, B=2085.4, alpha=0.348, beta=0.366)

N_T = 600e6          # 参考参数量
D_T = 12e9           # 参考数据量（20 tokens/param）
N_S = 60e6           # 候选参数量
C_T = 6 * N_T * D_T  # 参考训练 FLOPs（6ND 口径）


def loss(fit: dict, N: float, D: float) -> float:
    return fit["E"] + fit["A"] / N ** fit["alpha"] + fit["B"] / D ** fit["beta"]


def excess(fit: dict, N: float, D: float) -> float:
    return loss(fit, N, D) - fit["E"]


def delta_L(fit: dict, gamma: float) -> float:
    """60M @ gamma*C_t 相对 600M@12B 的 loss 差。"""
    D_s = 10 * gamma * D_T
    return loss(fit, N_S, D_s) - loss(fit, N_T, D_T)


def gamma_breakeven(fit: dict) -> tuple[float, float]:
    """解 ΔL<=0 的 gamma；返回 (gamma, 60M 所需 tokens)。"""
    nterm = fit["A"] * (N_S ** -fit["alpha"] - N_T ** -fit["alpha"])
    dterm_unit = fit["B"] * D_T ** -fit["beta"]
    y = 1 - nterm / dterm_unit          # y = (10 gamma)^-beta
    gamma = (1 / 10) * y ** (-1 / fit["beta"])
    return gamma, 10 * gamma * D_T


def mu_D_needed(fit: dict, mu_N: float) -> float:
    """达到参考 excess 所需的最小 mu_D。"""
    target = excess(fit, N_T, D_T)
    rem = target - fit["A"] / (mu_N * N_S) ** fit["alpha"]
    if rem <= 0:
        return 0.0
    return (fit["B"] / (rem * D_T ** fit["beta"])) ** (1 / fit["beta"])


def min_mu_C(fit: dict) -> tuple[float, float, float]:
    """最小化 mu_N*mu_D，拉格朗日闭式解。"""
    a, b = fit["alpha"], fit["beta"]
    target = excess(fit, N_T, D_T)
    # 最优时 alpha*term_N = beta*term_D
    tD = target / (1 + b / a)
    tN = target - tD
    mu_N = (fit["A"] / (tN * N_S ** a)) ** (1 / a)
    mu_D = (fit["B"] / (tD * D_T ** b)) ** (1 / b)
    return mu_N, mu_D, mu_N * mu_D


def loop_table(fit: dict, ks: list[int], gamma: float = 0.1) -> list[dict]:
    """固定 C=gamma*C_t、N=60M、循环 k 次：D = gamma*C_t/(6*N*k)。"""
    rows = []
    for k in ks:
        D = gamma * C_T / (6 * N_S * k)
        rows.append(dict(k=k, N_eff=k * N_S, D=D, L=loss(fit, k * N_S, D)))
    return rows


def loop_kstar(fit: dict, gamma: float = 0.1) -> float:
    a, b = fit["alpha"], fit["beta"]
    D0 = gamma * C_T / (6 * N_S)          # k=1 时的数据量
    c1 = fit["A"] / N_S ** a
    c2 = fit["B"] / D0 ** b
    return (a * c1 / (b * c2)) ** (1 / (a + b))


def besiroglu_check() -> dict:
    floor60 = BESIROGLU["E"] + BESIROGLU["A"] / N_S ** BESIROGLU["alpha"]
    ref = loss(BESIROGLU, N_T, D_T)
    mu_N_inf = (
        BESIROGLU["A"]
        / (excess(BESIROGLU, N_T, D_T) * N_S ** BESIROGLU["alpha"])
    ) ** (1 / BESIROGLU["alpha"])
    return dict(floor60=floor60, ref=ref, gap=floor60 - ref, mu_N_needed_inf=mu_N_inf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    f = HOFFMANN
    out: dict = {}

    out["hoffmann"] = {
        "reference": loss(f, N_T, D_T),
        "reference_excess": excess(f, N_T, D_T),
        "delta_L_gamma_0.1": delta_L(f, 0.1),
        "delta_L_gamma_1.0": delta_L(f, 1.0),
        "n_term_fixed": f["A"] * (N_S ** -f["alpha"] - N_T ** -f["alpha"]),
    }
    g, tokens = gamma_breakeven(f)
    out["hoffmann"]["gamma_breakeven"] = g
    out["hoffmann"]["tokens_breakeven"] = tokens

    mu_N_opt, mu_D_opt, mu_C_opt = min_mu_C(f)
    out["min_mu_C"] = {"mu_N": mu_N_opt, "mu_D": mu_D_opt, "mu_C": mu_C_opt}

    isoquant = []
    for mu_N in [1.0, 1.6, 2.0, 2.7, 3.0, 5.0, 7.26, 10.0]:
        mu_D = mu_D_needed(f, mu_N)
        isoquant.append({"mu_N": mu_N, "mu_D": mu_D, "mu_C": mu_N * mu_D})
    out["isoquant"] = isoquant

    ks = [1, 2, 3, 4, 6, 12]
    out["loop_table"] = loop_table(f, ks)
    kstar = loop_kstar(f)
    out["loop_kstar"] = kstar
    out["loop_gain"] = loop_table(f, [1])[0]["L"] - loss(f, kstar * N_S, (0.1 * C_T / (6 * N_S)) / kstar)

    out["besiroglu"] = besiroglu_check()

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print("=== Hoffmann 参考值 ===")
    print(f"600M@12B        L = {out['hoffmann']['reference']:.4f}")
    print(f"ΔL(γ=0.1)       = {out['hoffmann']['delta_L_gamma_0.1']:.4f}")
    print(f"ΔL(γ=1.0)       = {out['hoffmann']['delta_L_gamma_1.0']:.4f}")
    print(f"γ* breakeven    = {g:.1f}  -> 60M 需 {tokens/1e12:.2f}T tokens")
    print(f"min μ_C         = {mu_C_opt:.2f} @ (μ_N={mu_N_opt:.2f}, μ_D={mu_D_opt:.2f})")
    print("\n=== 等值线 ===")
    for r in isoquant:
        print(f"  μ_N={r['mu_N']:5.2f}  μ_D>={r['mu_D']:7.2f}  μ_C={r['mu_C']:7.2f}")
    print("\n=== 循环表 (0.1C_t, N=60M) ===")
    for r in out["loop_table"]:
        print(f"  k={r['k']:2d}  D={r['D']/1e9:5.1f}B  L={r['L']:.4f}")
    print(f"  k*={kstar:.2f}, 收益={out['loop_gain']:.4f} nats")
    print("\n=== Besiroglu ===")
    b = out["besiroglu"]
    print(f"  floor(60M)={b['floor60']:.4f}  600M@12B={b['ref']:.4f}  gap={b['gap']:.4f}")
    print(f"  即使 D→∞ 也需 μ_N>={b['mu_N_needed_inf']:.2f}")


if __name__ == "__main__":
    main()
