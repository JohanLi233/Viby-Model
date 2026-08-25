"""从 mp_fit 的 ρ(σ) 文本表（mpfit_all.txt / mp_fit_v2.txt）解析出
逐组逐时点的 excess persistence（中位ρ − 中位地板），并按操作性定义
给 σ*_upper：首个 excess ≥ 0.9 的 log-σ 分箱下沿。

用于 p30 这类 npz 快照已清理、只剩分析文本的情况。口径与
experiments/sigma_upper.py 完全一致（同样的箱沿、同样的阈值）。

用法: .venv/bin/python experiments/sigma_upper_from_txt.py <mpfit_txt>
"""

import re
import sys
from collections import defaultdict

import numpy as np

BIN_EDGES = np.array([1e-5, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
THRESH = 0.9

RE_STEP = re.compile(r"^优化器步 (\d+)")
RE_GROUP = re.compile(r"^\[(g2d|gst) (\d+)x(\d+)\]")
RE_BIN = re.compile(
    r"σ∈\[\s*([0-9.e-]+),\s*([0-9.e-]+)\)\s*ρ=\s*([-\d.]+)\s+地板\s+([-\d.]+)"
)


def parse(path):
    """{(step, kind, r, c): {bin_lo: excess}}"""
    out = defaultdict(dict)
    step = None
    group = None
    for line in open(path):
        m = RE_STEP.match(line)
        if m:
            step = int(m.group(1))
            group = None
            continue
        m = RE_GROUP.match(line.strip())
        if m:
            group = (step, m.group(1), int(m.group(2)), int(m.group(3)))
            continue
        m = RE_BIN.search(line)
        if m and group is not None:
            lo, _hi, rho, floor = (float(x) for x in m.groups())
            # 同组同箱可能出现多次（逐矩阵打印分开的组段落），保留首个
            out[group].setdefault(lo, rho - floor)
    return out


def main():
    path = sys.argv[1]
    data = parse(path)
    steps = sorted({k[0] for k in data})
    print(f"{path}  阈值: 箱 excess(中位ρ−中位地板) ≥ {THRESH}")
    for step in steps:
        print(f"\n=== 优化器步 {step} ===")
        print(
            f"{'组':<18}{'σ*_upper':>10}   "
            + " ".join(f"[{BIN_EDGES[i]:.0e},{BIN_EDGES[i + 1]:.0e})" for i in range(7))
        )
        groups = sorted({k[1:] for k in data if k[0] == step})
        for kind, r, c in groups:
            ex = data[(step, kind, r, c)]
            vals = [
                ex.get(float(BIN_EDGES[b]), float("nan"))
                for b in range(len(BIN_EDGES) - 1)
            ]
            first = next(
                (b for b, v in enumerate(vals) if np.isfinite(v) and v >= THRESH), None
            )
            fwd = f"{BIN_EDGES[first]:.1e}" if first is not None else "无越阈"
            vstr = " ".join(
                f"{v:>15.2f}" if np.isfinite(v) else f"{'—':>15}" for v in vals
            )
            print(f"{kind} {r}x{c:<8}{fwd:>10}   {vstr}")


if __name__ == "__main__":
    main()
