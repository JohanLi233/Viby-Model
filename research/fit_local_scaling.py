#!/usr/bin/env python3
"""
本地 (N,D)->CE 缩放律拟合（M4 子集用）。

模型：
  M5 加性:   L = E + A/N^a + B/D^b
  M6 交互:   L = E + A/N^a + B/D^b + C/N^{a/2}D^{b/2}

输入: TSV，列 N D holdout_ce [seed notes]（制表符分隔，N/D 为数字即可）
输出: 参数估计 + bootstrap CI + 关键预测点。

用法:
  uv run research/fit_local_scaling.py --tsv research/points.tsv
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch


def param_to_theta(p: dict) -> torch.Tensor:
    return torch.tensor(
        [
            p["logA"],
            p["logB"],
            p["logit_a"],
            p["logit_b"],
            p["E"],
            p.get("logC", -20.0),
        ],
        dtype=torch.float64,
    )


def theta_to_param(theta: torch.Tensor, interactive: bool) -> dict:
    logA, logB, logit_a, logit_b, E, logC = theta.tolist()
    return {
        "logA": logA,
        "logB": logB,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "E": E,
        "logC": logC if interactive else None,
        "A": math.exp(logA),
        "B": math.exp(logB),
        "a": 1.0 / (1.0 + math.exp(-logit_a)),
        "b": 1.0 / (1.0 + math.exp(-logit_b)),
        "C": math.exp(logC) if interactive else 0.0,
    }


def predict(p: dict, N: float, D: float, interactive: bool = False) -> float:
    base = p["E"] + p["A"] / N ** p["a"] + p["B"] / D ** p["b"]
    if interactive and p.get("C", 0.0):
        base += p["C"] / (N ** (p["a"] / 2) * D ** (p["b"] / 2))
    return base


def loss_fn(theta: torch.Tensor, N: torch.Tensor, D: torch.Tensor, y: torch.Tensor,
            interactive: bool) -> torch.Tensor:
    a = torch.sigmoid(theta[2])
    b = torch.sigmoid(theta[3])
    Nt = torch.as_tensor(N, dtype=torch.float64)
    Dt = torch.as_tensor(D, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    pred = theta[4] + torch.exp(theta[0]) / Nt ** a + torch.exp(theta[1]) / Dt ** b
    if interactive:
        pred = pred + torch.exp(theta[5]) / (Nt ** (a / 2) * Dt ** (b / 2))
    return torch.mean((pred - yt) ** 2)


def fit_once(N, D, y, interactive, seed):
    torch.manual_seed(seed)
    theta = torch.tensor(
        [math.log(400.0), math.log(400.0), 0.0, 0.0, 1.7, -20.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=2000, tolerance_grad=1e-12)
    best = (float("inf"), None)
    for _ in range(8):
        def closure():
            opt.zero_grad()
            l = loss_fn(theta, N, D, y, interactive)
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            val = float(loss_fn(theta, N, D, y, interactive))
            if val < best[0]:
                best = (val, theta.detach().clone())
    return theta_to_param(best[1], interactive), best[0]


def bootstrap(N, D, y, interactive, n_boot=400):
    rng = np.random.default_rng(0)
    base, _ = fit_once(N, D, y, interactive, 0)
    resid = y - np.array([predict(base, n, d, interactive) for n, d in zip(N, D)])
    params = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = np.array([predict(base, n, d, interactive) for n, d in zip(N, D)]) + resid[idx]
        try:
            p, _ = fit_once(N, D, yb, interactive, 1 + len(params))
        except Exception:
            continue
        params.append(p)
    return base, params


def ci(params, key):
    vals = sorted(p[key] for p in params)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--predict", action="store_true", help="打印关键预测点")
    args = ap.parse_args()

    rows = []
    with open(args.tsv) as f:
        header = f.readline().strip("\n").split("\t")
        assert {"N", "D"} <= set(header), f"TSV 需要 N,D 列: {header}"
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            rows.append((float(c[idx["N"]]), float(c[idx["D"]]), float(c[idx["holdout_ce"]])))
    N = np.array([r[0] for r in rows])
    D = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    print(f"records={len(rows)}  N∈[{N.min():.1e},{N.max():.1e}]  D∈[{D.min():.1e},{D.max():.1e}]")

    for interactive in [False, True]:
        p, mse = fit_once(N, D, y, interactive, 0)
        r2 = 1 - mse / np.var(y)
        print(f"\n{'M6 交互' if interactive else 'M5 加性'}: MSE={mse:.6f} R2={r2:.4f}")
        print(f"  E={p['E']:.4f} A={p['A']:.2f} a={p['a']:.4f} B={p['B']:.2f} b={p['b']:.4f}"
              + (f" C={p['C']:.2f}" if interactive else ""))
        if args.bootstrap and len(rows) >= 5:
            base, params = bootstrap(N, D, y, interactive, args.bootstrap)
            for key, label in [("E", "E"), ("A", "A"), ("a", "a"), ("B", "B"), ("b", "b")]:
                lo, hi = ci(params, key)
                print(f"  {label}: {base[key]:.3f}  CI[{lo:.3f},{hi:.3f}]")
        if args.predict:
            print("  预测:")
            for n, d, tag in [
                (60e6, 0.1e9, "60M@0.1B"),
                (60e6, 0.343e9, "60M@0.343B"),
                (600e6, 12e9, "600M@12B (外推)"),
            ]:
                print(f"    {tag}: {predict(p, n, d, interactive):.4f}")


if __name__ == "__main__":
    main()
