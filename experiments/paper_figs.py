"""论文图 Fig.1-4 生成（research/PAPER.md 配套）。

数据源（全部为本仓库实测落盘）：
  Fig.1  probe_p18_snap/mp_fit_v2.txt（优化器步 150/300）
         + probe_p30_latesnap/mpfit_all.txt（优化器步 500/750/950）
  Fig.2  P-2/P-4 实测 loss@500（见 PAPER.md §5.1/§5.3 表）
  Fig.3  σ* 表（§4.2）、P22、P21/P21b（§6）
  Fig.4  probe_p28a/b、probe_p29a/b console.log 轨迹 + P19/P23/P24 step/s

运行：uv run --no-project --with matplotlib python experiments/paper_figs.py
输出：research/figures/fig{1..4}_{...}.png/.pdf
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "research" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# ---------- 解析 mp_fit 输出：{(step, group): {band: median signal fraction}} ----------
def parse_mpfit(path):
    out = defaultdict(lambda: defaultdict(list))
    step = group = None
    for line in open(path):
        m = re.search(r"优化器步 (\d+)", line)
        if m:
            step = int(m.group(1))
            continue
        h = re.match(r"\[(g2d|gst) ([\dx]+)\]", line.strip())
        if h:
            group = h.group(2)
            continue
        r = re.search(
            r"σ∈\[\s*([\de.-]+),\s*([\de.-]+)\) ρ=\s*([-\d.]+)\s*地板\s*([\d.]+)", line
        )
        if r and step is not None and group is not None:
            lo, hi, rho, fl = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            out[(step, group)][(lo, hi)].append(max(0.0, rho - fl))
    med = {}
    for k, bands in out.items():
        med[k] = {b: sorted(v)[len(v) // 2] for b, v in bands.items()}
    return med


P18 = parse_mpfit(ROOT / "research_runs/probe_p18_snap/mp_fit_v2.txt")
P30 = parse_mpfit(ROOT / "research_runs/probe_p30_latesnap/mpfit_all.txt")

# ---------- Fig.1：ρ(σ) 坍缩 + 平稳性 ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
panels = [
    ("768x2304", "MTP block 768×2304", 3e-4, 3e-3),
    ("768x768", "attention 768×768", 1e-3, 1e-2),
]
styles = {
    150: ("#999999", "o", 0.8),
    300: ("#bbbbbb", "s", 0.8),
    500: ("#1f77b4", "o", 1.2),
    750: ("#2ca02c", "s", 1.2),
    950: ("#d62728", "^", 1.2),
}
for ax, (grp, title, slo, shi) in zip(axes, panels):
    for step in (150, 300, 500, 750, 950):
        src = P18 if step in (150, 300) else P30
        bands = src.get((step, grp))
        if not bands:
            continue
        xs = sorted(bands, key=lambda b: b[0])
        xm = [((lo * hi) ** 0.5) for lo, hi in xs]
        ym = [bands[b] for b in xs]
        c, mk, lw = styles[step]
        lbl = f"opt step {step}" + (" (warmup)" if step in (150, 300) else "")
        ax.plot(xm, ym, marker=mk, ms=3, lw=lw, color=c, label=lbl)
    ax.axvspan(slo, shi, color="orange", alpha=0.15, lw=0)
    ax.axvline(slo, color="orange", ls="--", lw=0.8)
    ax.axvline(shi, color="orange", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("singular value σ (F-normalized)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.6)
axes[0].set_ylabel("persistent signal fraction  ρ − floor")
axes[0].text(4e-4, 1.45, "transition band", color="darkorange", fontsize=7.5)
axes[1].legend(loc="lower right", frameon=False)
fig.suptitle(
    "Fig.1  Direction persistence collapses at σ* — and σ* is stationary (MoE, micro-steps 300→1900)",
    y=1.04,
)
fig.savefig(FIG / "fig1_persistence.png")
fig.savefig(FIG / "fig1_persistence.pdf")
plt.close(fig)

# ---------- Fig.2：U 型质量曲线 × 两架构 ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
# MoE：五点全部 post-fix 同帧（P31/P20/P32/P19/P33，seed 1337，配对 vs b05=5.264）。
# 两臂带三种子 Δ 误差棒（P36–P39：逐种子 Δ 的 min–max）。
# pre-fix 极端点（PE5/no-NS/frac）跨帧不可减，只进 §5.2 表不进图。
moe = [
    (0.001, 0.094, "classic NS5", None),
    (5e-4, 0.054, "b002", (0.014, 0.054)),
    (0.004, 0.010, "cubic5", None),
    (0.010, 0.0, "b05 (default)", None),
    (0.02, 0.085, "b10", (0.038, 0.085)),
]
ax = axes[0]
ax.axvspan(3e-4, 1e-2, color="orange", alpha=0.15, lw=0)
for x, y, lbl, seeds in moe:
    ax.plot(x, y, marker="o", ms=5, color="#1f77b4", ls="none")
    if seeds:
        ax.plot([x, x], [seeds[0], seeds[1]], color="#1f77b4", lw=1.2, marker="_", ms=6)
    if lbl == "classic NS5":
        ax.annotate(
            lbl,
            (x, y),
            textcoords="offset points",
            xytext=(9, -3),
            ha="left",
            fontsize=7.5,
        )
    else:
        ax.annotate(
            lbl,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7.5,
        )
ax.axhline(0.05, color="grey", ls=":", lw=0.8)
ax.axhline(-0.05, color="grey", ls=":", lw=0.8)
ax.text(
    0.30,
    0.05,
    "paired noise ±0.05; bars = 3-seed Δ range",
    fontsize=7,
    color="grey",
    transform=ax.transAxes,
)
ax.set_xscale("log")
ax.set_xlabel("knee50 of the spectral response (log)")
ax.set_ylabel("Δ loss@500 vs b05 (nat)")
ax.set_title("MoE 1080M: U-shape, arms = band width", fontsize=9)
# Dense 132M：五点网格（P26a/b/c + P34/P35，全 post-fix，seed 1337）
# 带 = 实测 [σ*≈3e-4, 上沿≈3e-2]（FFN excess 0.9 阈在 3e-2；attn 视野内不达 0.9）
dense = [
    (5e-4, 3.943, "b002"),
    (2e-3, 3.950, "b005"),
    (5e-3, 3.963, "b01"),
    (0.011, 3.997, "b05"),
    (0.02, 3.974, "b10"),
]
ax = axes[1]
ax.axvspan(3e-4, 3e-2, color="orange", alpha=0.15, lw=0)
for x, y, lbl in dense:
    ax.plot(x, y, "o", ms=5, color="#9467bd")
    ax.annotate(
        f"{lbl}\n{y:.3f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, -22),
        ha="center",
        fontsize=7.5,
    )
ax.set_ylim(3.925, 4.015)
ax.set_xscale("log")
ax.set_xlabel("knee50 (log)")
ax.set_ylabel("loss@500")
ax.set_title(
    "Dense 132M @500: warmup optimum\n(reverses by 2000 steps, §5.4)", fontsize=9
)
fig.suptitle(
    "Fig.2  Quality vs. knee position; shaded band = measured transition band [σ*, upper edge]",
    y=1.08,
    fontsize=9.5,
)
fig.savefig(FIG / "fig2_ucurve.png")
fig.savefig(FIG / "fig2_ucurve.pdf")
plt.close(fig)

# ---------- Fig.3：标度三联 ----------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
# (i) 逐组实测 σ*（log 条）
groups = [
    ("sharedFFN", 5.5e-4, ""),
    ("MTP", 5.5e-4, ""),
    ("attn", 1.7e-3, ""),
    ("KDA", 1.7e-3, ""),
    ("o_proj", 3e-3, ""),
    ("exp gu", 5.5e-3, ""),
    ("exp dw", 7e-3, ""),
    ("kv_up", 2.4e-2, ">no\ncollapse"),
]
ax = axes[0]
xs = range(len(groups))
cols = ["#d62728" if groups[i][0] in ("sharedFFN", "MTP") else "#1f77b4" for i in xs]
ax.bar(xs, [g[1] for g in groups], color=cols)
for i, g in enumerate(groups):
    if g[2]:
        ax.text(i, g[1] * 1.3, g[2], ha="center", fontsize=6.5)
ax.set_yscale("log")
ax.set_xticks(list(xs))
ax.set_xticklabels([g[0] for g in groups], fontsize=7.5, rotation=35, ha="right")
ax.set_ylabel("measured σ* (geom. mid)")
ax.set_title("(i) per-group σ*: 30× spread,\nnot an MP ratio", fontsize=9)
# (ii) 批大小减半：b05 vs b07（bs6/accum2，@950）
ax = axes[1]
ax.bar([0, 1], [5.120, 5.143], color=["#1f77b4", "#ff7f0e"], width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["b05 (knee 0.010)", "b07 (0.010·√2)"], fontsize=8)
ax.set_ylim(5.05, 5.20)
ax.set_ylabel("loss@950 (bs6)")
ax.set_title("(ii) batch halved: √B scaling\nnot confirmed (Δ=+0.02, n.s.)", fontsize=9)
for x, y in zip([0, 1], [5.120, 5.143]):
    ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=8)
# (iii) per-head ± 膝盖补偿（@500）
ax = axes[2]
vals = [5.264, 5.187, 5.202]
lbls = ["b05\n(matrix)", "per-head\n+b05", "per-head\n+b10 (comp.)"]
ax.bar([0, 1, 2], vals, color=["#1f77b4", "#2ca02c", "#2ca02c"], width=0.55)
ax.axhspan(5.264 - 0.05, 5.264 + 0.05, color="grey", alpha=0.15, lw=0)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(lbls, fontsize=8)
ax.set_ylim(5.13, 5.32)
ax.set_title(
    "(iii) per-head NS harmless post-fix\n(compensation not needed)", fontsize=9
)
for x, y in zip([0, 1, 2], vals):
    ax.text(x, y + 0.006, f"{y:.3f}", ha="center", fontsize=8)
fig.suptitle(
    "Fig.3  Scaling checks: shape / batch / per-head compensation", y=1.10, fontsize=9.5
)
fig.savefig(FIG / "fig3_scaling.png")
fig.savefig(FIG / "fig3_scaling.pdf")
plt.close(fig)


# ---------- Fig.4：Pareto + 2000 步反转 ----------
def traj(p):
    out = {}
    for line in open(p):
        m = re.search(r"\((\d+)/\d+\) loss:([\d.]+)\(main:([\d.]+),mtp:([\d.]+)", line)
        if m:
            out[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out


base = traj(ROOT / "research_runs/probe_p28a_b05_2k/console.log")
runs = {
    "ph+edge (P28b)": (
        "#d62728",
        traj(ROOT / "research_runs/probe_p28b_ph_edge_2k/console.log"),
    ),
    "edge only (P29b)": (
        "#ff7f0e",
        traj(ROOT / "research_runs/probe_p29b_edge_2k/console.log"),
    ),
    "per-head only (P29a)": (
        "#2ca02c",
        traj(ROOT / "research_runs/probe_p29a_ph_2k/console.log"),
    ),
}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
ax = axes[0]
pareto = [
    ("b05 (default)", 0.74, 5.264, "#1f77b4", (6, 4)),
    ("EdgeCubic", 0.78, 5.271, "#ff7f0e", (6, 4)),
    ("per-head+b05", 0.76, 5.187, "#2ca02c", (-10, 8)),
    ("per-head+edge", 0.79, 5.192, "#d62728", (2, -13)),
]
for lbl, x, y, c, off in pareto:
    ax.plot(x, y, "o", ms=6, color=c)
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, fontsize=7.5)
ax.set_xlabel("throughput (optimizer steps/s)")
ax.set_ylabel("loss@500 (paired)")
ax.set_title("500-step tier looks promotable")
ax.set_xlim(0.72, 0.86)
ax.set_ylim(5.15, 5.30)
ax = axes[1]
for lbl, (c, tr) in runs.items():
    steps = sorted(set(tr) & set(base))
    steps = [s for s in steps if s > 0]
    ax.plot(steps, [tr[s][0] - base[s][0] for s in steps], color=c, lw=1.4, label=lbl)
    ax.plot(steps, [tr[s][2] - base[s][2] for s in steps], color=c, lw=1.0, ls="--")
ax.axhline(0, color="grey", lw=0.8)
ax.axvline(800, color="grey", ls=":", lw=0.8)
ax.text(810, -0.13, "crossover ≈800", fontsize=7, color="grey")
ax.set_xlabel("micro-step")
ax.set_ylabel("Δ vs b05 baseline (nat)")
ax.set_title("2000-step tier: monotone reversal (edge)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
axes[1].text(
    0.02,
    0.06,
    "solid: total loss   dashed: MTP component",
    transform=axes[1].transAxes,
    fontsize=7,
    color="grey",
)
fig.suptitle(
    "Fig.4  EdgeCubic: cheap at 500 steps, harmful by 2000 — localized in MTP as the corrected rule predicts",
    y=1.05,
)
fig.savefig(FIG / "fig4_pareto_reversal.png")
fig.savefig(FIG / "fig4_pareto_reversal.pdf")
plt.close(fig)

print("written:", *[p.name for p in sorted(FIG.glob("fig*"))], sep="\n  ")
