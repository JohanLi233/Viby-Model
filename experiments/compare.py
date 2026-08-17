#!/usr/bin/env python3
"""汇总 results.tsv 及各轮 loss 曲线，便于横向对比。

用法: uv run experiments/compare.py [round ...]
"""
import re
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
TSV = Path(__file__).parent.parent / "results.tsv"


def loss_curve(round_name):
    log = LOG_DIR / f"{round_name}.log"
    if not log.exists():
        return []
    curve = []
    for line in log.read_text().splitlines():
        m = re.match(r"Epoch:\[\d+/\d+\]\((\d+)/\d+\) loss:([\d.]+)", line)
        if m:
            curve.append((int(m.group(1)), float(m.group(2))))
    return curve


def main():
    rounds = sys.argv[1:]
    if not rounds:
        if not TSV.exists():
            print("results.tsv 不存在")
            return
        rows = [l.split("\t") for l in TSV.read_text().splitlines()]
        header, data = rows[0], rows[1:]
        widths = [max(len(str(r[i])) for r in [header] + data) for i in range(len(header))]
        for r in [header] + data:
            print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
        rounds = [r[0] for r in data]

    print("\nloss 曲线抽点对比:")
    curves = {r: loss_curve(r) for r in rounds}
    curves = {r: c for r, c in curves.items() if c}
    if not curves:
        return
    checkpoints = [100, 200, 400, 600, 780, 1000, 1500, 2000]
    header = "step".rjust(6) + "".join(r[:22].rjust(24) for r in curves)
    print(header)
    for s in checkpoints:
        row = str(s).rjust(6)
        for r, curve in curves.items():
            val = next((l for st, l in reversed(curve) if st <= s), None)
            row += (f"{val:.3f}" if val is not None else "-").rjust(24)
        print(row)


if __name__ == "__main__":
    main()
