"""σ*_upper 的操作性定义与重算（论文 gap 4）。

定义（预注册口径，替代目测）：
  按 log σ 分箱（×√10 步进），箱内取 **中位** ρ 与中位 ρ_floor（与
  experiments/mp_fit.py 的 ρ(σ) 表口径一致；中位对地板在小 σ 处的
  爆发稳健），excess = 中位ρ − 中位地板；σ*_upper := 首个满足
  excess ≥ 0.9 的箱的**下沿**。同时打印「从高端往回第一个跌破阈值的
  箱的上沿」作对照（单调性不佳时两者会不同，论文用前者）。

输入：trainer/snapshot.py 快照目录（probe_p18_snap / probe_p30_latesnap）。
用法: .venv/bin/python experiments/sigma_upper.py <snap_dir> [step ...]
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_fit import analyze_matrix, load_grads, load_group  # noqa: E402

# 论文 §4.2 的语义分组：shape -> 组名
GROUP_NAMES = {
    (768, 768): "attn q/k/v 768x768",
    (768, 384): "o_proj/lat 768x384",
    (384, 768): "o_proj/lat 384x768",
    (96, 768): "KDA gate 96x768",
    (768, 96): "KDA gate 768x96",
    (128, 768): "kv_up 128x768",
    (768, 128): "gate_down 768x128",
    (768, 1536): "shared FFN 768x1536",
    (1536, 768): "shared FFN 1536x768",
    (768, 2304): "MTP 768x2304",
    (2304, 768): "MTP 2304x768",
}
# 堆叠组（路由专家）按 (r,c) 同名合并展示
STACK_NAMES = {
    (768, 384): "routed gu 768x384",
    (384, 384): "routed dw 384x384",
    (384, 768): "lat up/down 384x768",
}

BIN_EDGES = np.array([1e-5, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
THRESH = 0.9


def collect(snap_dir, step):
    """返回 {(kind, r, c): (σ 列表, ρ 列表, floor 列表)}，逐方向 pooled。"""
    grads, eidx = load_grads(snap_dir, step)
    out = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, f"mom_*_step{step}.npz"))):
        U, meta = load_group(f)
        is_stack = meta["kind"] != "g2d"
        jobs = []
        if not is_stack:
            for i, p in enumerate(meta["paths"]):
                if grads and p not in grads[0]:
                    continue
                gi = [g[p] for g in grads] if grads else []
                jobs.append((U[i], gi))
        else:
            for si, p in enumerate(meta["paths"]):
                for ej, j in enumerate(meta["expert_idx"]):
                    if grads:
                        if p not in grads[0] or p not in eidx:
                            continue
                        gcol = eidx[p].index(j) if j in eidx[p] else None
                        if gcol is None:
                            continue
                        gi = [g[p][gcol] for g in grads]
                    else:
                        gi = []
                    jobs.append((U[si, ej], gi))
        r, c = meta["r"], meta["c"]
        key = ("stack" if is_stack else "g2d", r, c)
        sig_all, rho_all, fl_all = out.get(key, ([], [], []))
        for u, gi in jobs:
            if not gi:
                continue
            res = analyze_matrix(u, gi)
            sig_all.append(res["S"])
            rho_all.append(res["rho"])
            fl_all.append(res["rho_floor"])
        out[key] = (sig_all, rho_all, fl_all)
    return out


def upper_edge(sig, rho, floor):
    """返回 (σ*_upper 单次越阈, 对照值, 每箱 excess 列表)。箱统计量 =
    中位 ρ − 中位地板（mp_fit ρ(σ) 表口径），箱最少 10 个方向。"""
    sig = np.concatenate(sig)
    rho = np.concatenate(rho)
    floor = np.concatenate(floor)
    ibin = np.searchsorted(BIN_EDGES, sig) - 1
    exs = []
    for b in range(len(BIN_EDGES) - 1):
        m = ibin == b
        exs.append(
            float(np.median(rho[m]) - np.median(floor[m]))
            if m.sum() >= 10
            else float("nan")
        )
    first = next((b for b, v in enumerate(exs) if np.isfinite(v) and v >= THRESH), None)
    fwd = BIN_EDGES[first] if first is not None else float("nan")
    # 对照：从最高非 nan 箱往回，第一个 < THRESH 的箱的上沿
    last_bad = None
    for b in range(len(exs) - 1, -1, -1):
        v = exs[b]
        if not np.isfinite(v):
            continue
        if v < THRESH:
            last_bad = b
            break
    bwd = BIN_EDGES[last_bad + 1] if last_bad is not None else BIN_EDGES[0]
    return fwd, bwd, exs


def main():
    snap_dir = sys.argv[1]
    steps = [int(s) for s in sys.argv[2:]] or sorted(
        {
            int(m.group(1))
            for f in glob.glob(os.path.join(snap_dir, "mom_*"))
            for m in [re.search(r"_step(\d+)\.npz$", f)]
            if m
        }
    )
    for step in steps:
        print(
            f"\n{'=' * 84}\n优化器步 {step}（{snap_dir}）  阈值: 箱均值(excess ρ) ≥ {THRESH}\n{'=' * 84}"
        )
        print(
            f"{'组':<26}{'σ*_upper':>10}{'对照':>10}   箱 excess "
            + " ".join(f"[{BIN_EDGES[i]:.0e},{BIN_EDGES[i + 1]:.0e})" for i in range(5))
        )
        data = collect(snap_dir, step)
        rows = []
        for (kind, r, c), (sig, rho, floor) in sorted(data.items()):
            names = STACK_NAMES if kind == "stack" else GROUP_NAMES
            label = names.get((r, c), f"{kind} {r}x{c}")
            if kind == "stack" and (r, c) in GROUP_NAMES and (r, c) not in STACK_NAMES:
                label = f"stack {r}x{c}"
            fwd, bwd, exs = upper_edge(sig, rho, floor)
            rows.append((label, fwd, bwd, exs))
        for label, fwd, bwd, exs in rows:
            mstr = " ".join(
                f"{v:>16.2f}" if np.isfinite(v) else f"{'—':>16}" for v in exs[:5]
            )
            print(f"{label:<26}{fwd:>10.2e}{bwd:>10.2e}   {mstr}")


if __name__ == "__main__":
    main()
