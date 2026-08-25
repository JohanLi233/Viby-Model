"""P-1 离线分析：真实训练动量的 MP bulk 拟合、λ̂₊ 估计、跨 microbatch
方向相关坍缩、行/列方差各向异性（论文 Fig.1 数据包，理论口径见
research/SPECTRAL_THEORY.md）。

输入：trainer/snapshot.py 产出的快照目录（mom_*_step{k}.npz +
grad_mb{j}_step{k}.npz）。

口径（与 NS 内部一致）：每个动量矩阵 F-归一（Σσ²=1）。噪声 iid 部分
E 的奇异值服从 MP 律：σ = √(y·w/n)，y ~ MP_c（c=max/min≥1，
支撑 (1±1/√c)²），bulk 上边缘 λ₊ = √w(1/√m+1/√n)，w = 噪声 F-能量
占比（spike peeling 估计：w ← 1 − Σ_{σ>λ₊(w)} σ² 迭代至不动点）。

跨 microbatch 方向相关：动量奇异方向 (u_k, v_k) 上，两个独立微批梯度
的投影乘积对 6 对取平均 ŝ²_k = mean_{a<b} d_{a,k}·d_{b,k}——独立噪声
零均值消去，留下该方向的持久信号能量。理论（BBP）：σ<λ₊ 的方向与
信号渐近正交 ⟹ ρ_k = ŝ²_k/σ²_k 应在 σ/λ̂₊=1 处坍缩到 0。
理论曲线（无自由参数）：ρ(θ) = θ²/σ_obs(θ)²，x = σ_obs(θ)/(1+√c)。

用法: .venv/bin/python experiments/mp_fit.py <snapshot_dir> [step ...]
"""

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_rmt_shrinker import bgn_overlaps, bgn_sigma_obs  # noqa: E402


# ---------- MP 律（σ 空间，F-归一口径）----------


def lam_plus(m, n, w):
    return np.sqrt(w) * (1.0 / np.sqrt(max(m, n)) + 1.0 / np.sqrt(min(m, n)))


def peel_w_c(sig, c):
    """c = max/min。返回 (w, n_spike, ok)。"""
    n_short = len(sig)
    m_long = int(round(n_short * c))
    w = 1.0
    for _ in range(10):
        edge = lam_plus(m_long, n_short, w)
        k = int(np.searchsorted(-sig, -edge))  # sig 降序：> edge 的个数
        if k >= 0.9 * n_short:
            return 0.0, k, False  # 几乎全谱都在「边缘上」→ 非 MP+spikes 形态
        w_new = 1.0 - float(np.sum(sig[:k] ** 2))
        if abs(w_new - w) < 1e-6:
            w = w_new
            break
        w = max(w_new, 1e-6)
    return w, k, w > 1e-4


# ---------- 快照装载 ----------


def load_group(path):
    d = np.load(path)
    meta = json.loads(d["__meta__"].tobytes())
    return d["U"], meta


def load_grads(snap_dir, step, n_mb=4):
    """返回 {path: array} 的列表（每个微批一个 dict）与 expert_idx 映射。"""
    out = []
    eidx = {}
    for j in range(n_mb):
        fs = glob.glob(os.path.join(snap_dir, f"grad_mb{j}_step{step}.npz"))
        if not fs:
            break
        d = np.load(fs[0])
        meta = json.loads(d["__meta__"].tobytes())
        eidx.update(meta.get("expert_idx", {}))
        out.append({k: d[k] for k in d.files if k != "__meta__"})
    return out, eidx


# ---------- 核心分析 ----------


def analyze_matrix(U_raw, grads_raw):
    """单个矩阵：谱 + MP peel + 跨微批信号能量。

    U_raw (m,n) 原始动量；grads_raw: 4 个同形微批梯度。
    返回 dict（σ 归一谱、w、λ₊、逐方向 ρ 与 σ/λ₊、梯度行/列方差）。
    """
    U_raw = np.asarray(U_raw, dtype=np.float64)
    fro = np.linalg.norm(U_raw)
    if not np.isfinite(fro) or fro < 1e-20:
        return {
            "ok": False,
            "S": np.array([0.0]),
            "w": 0.0,
            "k_spike": 0,
            "lam": float("nan"),
            "rho": np.array([0.0]),
            "rho_floor": np.array([0.0]),
            "s_over_lam": np.array([0.0]),
            "rowv": np.array([1.0]),
            "colv": np.array([1.0]),
            "m": U_raw.shape[0],
            "n": U_raw.shape[1],
            "zipf": float("nan"),
        }
    Un = U_raw / fro
    Ul, S, Vrt = np.linalg.svd(Un, full_matrices=False)
    m, n = Un.shape
    c = max(m, n) / min(m, n)
    w, k_spike, ok = peel_w_c(S, c)
    lam = lam_plus(m, n, w)

    # 跨 microbatch 投影（raw 单位）：d[a, k] = u_kᵀ G_a v_k
    ds = []
    for G in grads_raw:
        G = np.asarray(G, dtype=np.float64)
        ds.append(np.einsum("mk,mn,kn->k", Ul, G, Vrt) / fro)  # 归一化单位
    ds = np.array(ds)  # (n_mb, min(m,n))
    s2 = np.zeros(len(S))
    npairs = 0
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            s2 += ds[a] * ds[b]
            npairs += 1
    s2 /= max(npairs, 1)
    rho = s2 / np.maximum(S**2, 1e-30)
    # s2 估计量的噪声地板：d_a = 信号 + 独立噪声 τ，Var_a(d)=信号²+τ²，
    # ⟹ τ̂² = max(Var(d)−ŝ², 0)，6 对平均的地板 = τ̂²/√npairs
    tau2 = np.maximum(ds.var(axis=0, ddof=1) - s2, 0)
    rho_floor = tau2 / np.sqrt(max(npairs, 1)) / np.maximum(S**2, 1e-30)

    # 噪声各向异性（关键：先去信号再量噪声——4 个独立微批共享同一信号
    # S，对微批均值做中心化的残差 = 纯噪声（×4/3 修正有限样本偏置）。
    # 直接量原始梯度的行方差会把信号的行结构误报成噪声各向异性。）
    Gcat = np.concatenate([np.asarray(g, dtype=np.float64)[None] for g in grads_raw])
    Gres = (Gcat - Gcat.mean(axis=0, keepdims=True)) * np.sqrt(4 / 3)
    rowv = (Gres**2).mean(axis=2).ravel()  # 每矩阵每行的噪声二阶矩
    colv = (Gres**2).mean(axis=1).ravel()
    # Zipf 斜率：log σ vs log rank 中段（重尾信号谱签名；MP bulk 则很平）
    k_mid = len(S) // 4
    idx = np.arange(k_mid, max(k_mid + 2, 3 * len(S) // 4))
    zipf = (
        float(np.polyfit(np.log(idx + 1), np.log(S[idx]), 1)[0])
        if len(idx) >= 2 and np.all(S[idx] > 0)
        else float("nan")
    )
    return {
        "S": S,
        "w": w,
        "k_spike": k_spike,
        "ok": ok,
        "lam": lam,
        "rho": rho,
        "rho_floor": rho_floor,
        "s_over_lam": S / max(lam, 1e-12),
        "rowv": rowv,
        "colv": colv,
        "m": m,
        "n": n,
        "zipf": zipf,
    }


def ks_shape_check(spectra, ws, c, n_short):
    """pooled bulk 形状检验：y=σ²n/w_i  pooled 后与 MP_c 比 KS。"""
    ys = []
    for S, w in zip(spectra, ws):
        m_long = int(round(n_short * c))
        edge = lam_plus(m_long, n_short, w)
        bulk = S[S <= edge]
        ys.append(bulk**2 * n_short / w)
    y = np.sort(np.concatenate(ys))
    grid = np.linspace(1e-8, (1 + 1 / np.sqrt(c)) ** 2 * 1.001, 4000)
    a = (1 - 1 / np.sqrt(c)) ** 2
    # MP_c pdf（y 空间，ratio 1/c）
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf = np.sqrt(
            np.maximum(grid - a, 0) * np.maximum((1 + 1 / np.sqrt(c)) ** 2 - grid, 0)
        ) / (2 * np.pi * grid / c)
    cdf_t = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    cdf_t /= max(cdf_t[-1], 1e-12)
    # 标准 KS：经验 CDF vs 理论在样本点取值
    t_at_y = np.interp(y, grid, cdf_t)
    e_at_y = (np.arange(len(y)) + 0.5) / len(y)
    D = float(np.max(np.abs(t_at_y - e_at_y)))
    return D, len(y)


def theory_curve(c, n_pts=60):
    """无自由参数的 BGN 坍缩曲线：x=σ_obs/(1+√c)，ρ=θ²·cos²θ_u·cos²θ_v/σ_obs²。

    实测 ρ_k = (u_kᵀSv_k)²/σ_k²，BGN 下 u_k 与真方向重合 cosθ_u/v，
    故信号投影要乘两个重合度（不是裸的 θ²/σ_obs²）。
    """
    thc = c**0.25
    th = np.logspace(np.log10(thc * 1.02), np.log10(thc * 30), n_pts)
    so = bgn_sigma_obs(th, c)
    pu, pv = bgn_overlaps(th, c)
    return so / (1 + np.sqrt(c)), th**2 * pu * pv / so**2


BIN_EDGES = np.array([0.2, 0.4, 0.6, 0.8, 0.95, 1.05, 1.25, 1.6, 2.2, 3.5, 8.0])


def main():
    snap_dir = sys.argv[1]
    steps = [int(s) for s in sys.argv[2:]] or sorted(
        int(re.search(r"_step(\d+)\.npz$", f).group(1))
        for f in glob.glob(os.path.join(snap_dir, "mom_g2d_*"))
    )
    for step in steps:
        print(f"\n{'=' * 78}\n优化器步 {step}\n{'=' * 78}")
        grads, eidx = load_grads(snap_dir, step)
        if len(grads) < 2:
            print("  （无微批梯度快照，跳过坍缩分析）")
        mom_files = sorted(glob.glob(os.path.join(snap_dir, f"mom_*_step{step}.npz")))
        for f in mom_files:
            U, meta = load_group(f)
            mats = []  # (label, U_i, [G_i per mb])
            if meta["kind"] == "g2d":
                for i, p in enumerate(meta["paths"]):
                    if grads and p not in grads[0]:
                        continue
                    gi = [g[p] for g in grads] if grads else []
                    mats.append((p, U[i], gi))
                r, c2 = meta["r"], meta["c"]
            else:  # gst：U (slots, experts, r, c)
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
                        mats.append((f"{p}#e{j}", U[si, ej], gi))
                r, c2 = meta["r"], meta["c"]
            if not mats:
                continue
            cshape = max(r, c2) / min(r, c2)
            labeled = []
            for label, u, g in mats:
                if grads and not g:
                    continue
                x = analyze_matrix(u, g if g else [np.zeros_like(u)] * 2)
                labeled.append((label, x))
            if not labeled:
                continue
            res = [x for _, x in labeled]
            oks = [x for x in res if x["ok"]]
            print(
                f"\n[{meta['kind']} {r}x{c2}] {len(res)} 矩阵（peeling 可拟合"
                f" {len(oks)}/{len(res)}）  c={cshape:.2f}"
            )
            # 谱概要（所有矩阵，不管 peeling 是否收敛）
            qs = []
            for x in res:
                S = x["S"]
                n_s = len(S)
                qs.append(
                    [
                        S[0],
                        S[max(0, int(0.05 * n_s))],
                        S[int(0.25 * n_s)],
                        S[int(0.5 * n_s)],
                        S[int(0.75 * n_s)],
                        S[-1],
                    ]
                )
            qs = np.array(qs)
            print(
                "  谱分位（F-归一，中位 across 矩阵）：σ1={:.3f} σ5%={:.3f} "
                "σ25%={:.4f} σ50%={:.4f} σ75%={:.4f} σmin={:.2e}".format(
                    *np.median(qs, axis=0)
                )
            )
            print(
                f"  Zipf 斜率（logσ~logrank 中段）中位 "
                f"{np.median([x['zipf'] for x in res]):.3f}"
                f"（MP bulk ≈ 平；≤ −0.3 ≈ 重尾信号谱）"
            )
            if oks:
                lams = np.array([x["lam"] for x in oks])
                print(
                    f"  λ̂₊ 中位 {np.median(lams):.4f}  IQR "
                    f"[{np.percentile(lams, 25):.4f}, {np.percentile(lams, 75):.4f}]"
                    f"  ŵ 中位 {np.median([x['w'] for x in oks]):.4f}  "
                    f"spike 数中位 {int(np.median([x['k_spike'] for x in oks]))}"
                )
                D, nbulk = ks_shape_check(
                    [x["S"] for x in oks], [x["w"] for x in oks], cshape, min(r, c2)
                )
                if nbulk > 0:
                    ks_crit = 1.36 / np.sqrt(nbulk)
                    print(
                        f"  MP bulk 形状 KS D={D:.4f}（N={nbulk}，5% 临界 "
                        f"~{ks_crit:.4f}）→ {'符合 MP' if D < 2 * ks_crit else '偏离 MP'}"
                    )
            else:
                print("  ⚠ 无矩阵可 MP 拟合（w→0 发散）：谱非 MP+spikes 形态")
            if grads:
                # ρ vs 绝对 σ（log 分箱）：不依赖 peeling，所有矩阵可用
                xs = np.concatenate([x["S"] for x in res])
                rs = np.concatenate([x["rho"] for x in res])
                fs = np.concatenate([x["rho_floor"] for x in res])
                print("  ρ(σ)（持久信号能量占比；± 后为估计量噪声地板）：")
                edges = np.array([1e-5, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 2.0])
                for lo, hi in zip(edges[:-1], edges[1:]):
                    msk = (xs >= lo) & (xs < hi)
                    if msk.sum() < 10:
                        continue
                    print(
                        f"    σ∈[{lo:7g},{hi:6g}) ρ={np.median(rs[msk]):6.3f}"
                        f"  地板 {np.median(fs[msk]):6.3f} (n={msk.sum():5d})"
                    )
                rv = np.concatenate([x["rowv"] for x in res])
                cv = np.concatenate([x["colv"] for x in res])
                r_row = (rv.var() / rv.mean() ** 2) / (2 / res[0]["n"])
                r_col = (cv.var() / cv.mean() ** 2) / (2 / res[0]["m"])
                print(
                    f"  噪声各向异性比（中心化残差）：行 {r_row:7.2f}  "
                    f"列 {r_col:7.2f}  （1.0=各向同性）"
                )
            # 逐张量 λ̂₊（仅 peeling 收敛的）
            by_path = {}
            for label, x in labeled:
                if not x["ok"]:
                    continue
                p = label.split("#e")[0]
                by_path.setdefault(p, []).append(x["lam"])
            if by_path and len(by_path) > 1:
                print("  逐张量 λ̂₊：")
                for p in sorted(by_path):
                    v = np.array(by_path[p])
                    short = p.replace("model.stack.", "").replace(".weight", "")
                    print(f"    {short:<58} {np.median(v):.4f} (n={len(v)})")
    print(
        "\n参考膝盖（F-归一口径，knee50 / l0）：cubic5 l0=0.007，"
        "cubic5b05 l0=0.05（现默认），cubic5b10 l0=0.10"
    )


if __name__ == "__main__":
    main()
