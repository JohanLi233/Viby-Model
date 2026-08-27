"""M1 合成裁决：EigenGate 谱门 vs KDA 基线（免训练，状态动力学级）。

三个实验，对应方案的 P4（容量）、P3（旋转判别）与 needle 反例检验：
  Exp1 MQAR 容量：m 个联想（25% 热键重复出现），负载 m∈{D,2D,4D}，
        扫描膝盖 l0 × 衰减。hotmode=rewrite：热键复现带 β>0 复写（LM 语
        义，自愈闭环的承重条件）；hotmode=query：复现为纯查询（β=0）对照，
        检验自愈是否真的是机制来源。
  Exp2 needle：t=0 写一次性 needle（β=1），随后 m 个一次性干扰写，
        测末尾 needle 检索误差——谱门预期失败的负载（方案回避的风险点）。
  Exp3 旋转 key：key 统计轴对齐各向异性 + 逐通道衰减适配该统计；固定
        酉旋转后逐通道衰减失配，谱门酉等变应不受影响（P3 机制）。

  Exp4 λ 扫描：排除「λ=1 混太狠」对 Exp1 负面结果的替代解释。
  Exp5 SPR（自预条件读出，修正设计）：状态不动，读出换成
        o = Sᵀ(SSᵀ+λI)^{-1} q——占据归一化检索。合成尺度下精确求逆
        （float64），把「机制对不对」与「NS 近似好不好」分开。λ 相对
        tr(SSᵀ)/D 扫描；读出带每 cell 单全局标量尺度拟合（LayerNorm
        增益语义；逐键拟合=作弊，不做）。λ→∞ 在尺度拟合下退化为
        baseline 读出（_parity_check 含此断言）。预注册预测：
        P-A 热键改善或中性（符号必须翻转）；P-B 冷键增益 ≥ 状态版 5–7%；
        P-C needle 中滞后损害消失；P-D 增益随 m/D 递增。
        【结果：P-A/P-D 失败，SPR 死——delta 残差与占据反相关，
        SSᵀ 不是可用的自预条件子（diag 0.8–1.25）。】
  Exp6 touch（读取门控保留）：转移加 rank-1 保护项
        [D_t + ρ q̂q̂ᵀ(I−D_t)]——被读到的地址本步不老化（LRU touch）。
        非扩张（凸组合），零新增状态/投影。锚点：Exp1 里
        query-hot 0.87 vs rewrite-hot 0.26 的缺口。
        【结果：恒等于零（≤0.003）——逐步衰减豁免 ≠ 年龄归零，
        效果上限 = touch 步数/总步数。死。】
  Exp7 巩固回放（Consolidation Replay）：小型精确缓冲（R=32 条 (k,v)
        + surprise 准入 + LRU/RR 调度）把真实内容作为伪写入重新喂给
        delta 规则。核心恒等式：回放剂量 = 保留度赤字（残差自定量，
        deficit 列直接验证）。iso-D124 = 同等内存扩容判别消融。

递推是 _recurrent_kda 的单头镜像（main 里做 parity 核对），门控直接调
model.eigengate.apply（显式 l0/λ，不走 env）。f32，B 轴 = 配置轴。

用法: .venv/bin/python experiments/synth_eigengate.py [--exp 1|2|3|4|5|all] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model import eigengate

D = 96  # key/state 维（MoE 平台头维）
DV = 96
PERIOD = 256  # 门控周期（token），= K16 × chunk16


# ---------------------------------------------------------------- 核心递推


def run_stream(k, v, log_g, beta, gates, q=None, rhos=None):
    """k/v/log_g (B,T,D)，beta (B,T)；gates: 长度 B 的 l0 列表（None=关）。

    与 _recurrent_kda 的三行递推逐行一致；每 PERIOD 个 token 对每个 cell
    独立插入谱门（l0 不同无法合批，逐 cell apply）。

    touch（Exp6）：rhos 非 None 时转移变为
        S ← [D_t + ρ q̂_t q̂_tᵀ (I − D_t)] S
    即读出方向 q̂ 上的衰减按 ρ 撤销（ρ∈[0,1]，D 与 I 沿 q̂ 的凸组合，
    非扩张）。q (B,T,D) 为本步查询方向（合成负载取 q=k）。
    """
    B, T, _ = k.shape
    S = mx.zeros((B, D, DV), dtype=mx.float32)
    gates = [
        (g, 1.0) if g is not None and not isinstance(g, tuple) else g for g in gates
    ]
    coeffs = [
        eigengate.cubic_coeffs(l0=g[0]) if g is not None else None for g in gates
    ]
    if rhos is not None and any(r > 0 for r in rhos):
        rho_vec = mx.array(rhos, dtype=mx.float32)[:, None, None]  # (B,1,1)
    else:
        rho_vec = None
    for t in range(T):
        gexp = mx.exp(log_g[:, t])
        Sd = S * gexp[..., None]
        if rho_vec is not None:
            qh = q[:, t]
            qh = qh / mx.sqrt((qh * qh).sum(axis=-1, keepdims=True) + 1e-12)
            proj = ((S - Sd) * qh[..., None]).sum(axis=-2)  # q̂ᵀ(I−D)S, (B,Dv)
            S = Sd + rho_vec * qh[..., None] * proj[:, None, :]
        else:
            S = Sd
        kv = (S * k[:, t][..., None]).sum(axis=-2)
        delta = beta[:, t][..., None] * (v[:, t] - kv)
        S = S + k[:, t][..., None] * delta[..., None, :]
        if (t + 1) % PERIOD == 0:
            for b in range(B):
                if coeffs[b] is not None:
                    Sb = eigengate.apply(
                        S[b : b + 1], lam=gates[b][1], coeffs=coeffs[b],
                        target="keep",
                    )
                    S = mx.concatenate([S[:b], Sb, S[b + 1 :]], axis=0) if B > 1 else Sb
    mx.eval(S)
    return S


def rel_rmse(S, keys, vals):
    """S (B,D,Dv)，keys (B,m,D)，vals (B,m,Dv) → 每键相对检索误差 (B,m)。"""
    r = (S[:, None, :, :] * keys[:, :, :, None]).sum(axis=-2)  # (B,m,Dv)
    e2 = ((r - vals) ** 2).sum(-1) / ((vals**2).sum(-1) + 1e-12)
    mx.eval(e2)
    return np.sqrt(np.array(e2))


def _parity_check():
    """run_stream（门关）与 _recurrent_kda 逐位核对。"""
    from model.kda import _recurrent_kda

    rng = np.random.default_rng(0)
    B, T, d = 2, 40, 16
    global D, DV
    D0, DV0 = D, DV
    D = DV = d
    try:
        k = rng.normal(size=(B, T, d)).astype(np.float32)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.normal(size=(B, T, d)).astype(np.float32) / d**0.5
        lg = (-0.01 * rng.random(size=(B, T, d))).astype(np.float32)
        bt = (0.8 * rng.random(size=(B, T))).astype(np.float32)
        S_mine = run_stream(mx.array(k), mx.array(v), mx.array(lg), mx.array(bt), [None] * B)
        _, S_ref = _recurrent_kda(
            mx.array(k[:, None]), mx.array(k[:, None]), mx.array(v[:, None]),
            mx.array(lg[:, None]), mx.array(bt[:, None]),
        )
        mx.eval(S_ref)
        dmax = np.abs(np.array(S_mine) - np.array(S_ref[:, 0])).max()
        assert dmax < 1e-5, f"parity 失败: {dmax}"
        print(f"parity vs _recurrent_kda: max|ΔS|={dmax:.2e} OK")
    finally:
        D, DV = D0, DV0

    # SPR parity：λ→∞ 时读出 ∝ Sᵀq，尺度拟合后与 baseline 逐位一致
    rng = np.random.default_rng(1)
    Sn = rng.normal(size=(D0, DV0))
    kk = rng.normal(size=(64, D0))
    kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    vv = rng.normal(size=(64, DV0))
    base = _scaled_rmse((Sn.T @ kk.T).T, vv, axis=-1)
    A = Sn @ Sn.T
    mu = np.trace(A) / D0
    Rk = np.linalg.solve(A + 1e6 * mu * np.eye(D0), kk.T)
    spr = _scaled_rmse((Sn.T @ Rk).T, vv, axis=-1)
    d = np.abs(base - spr).max()
    assert d < 1e-3, f"SPR λ→∞ parity 失败: {d}"
    print(f"SPR λ→∞ parity: max|ΔRMSE|={d:.2e} OK")


def _scaled_rmse(r, v, axis):
    """scale-fitted 相对 RMSE：每 cell 拟合一个全局标量 α（LayerNorm 增益
    语义），e_i = ||α r_i − v_i||/||v_i||。r/v: (m, DV)，axis=1 逐键。"""
    alpha = (r * v).sum() / ((r**2).sum() + 1e-30)
    e2 = ((alpha * r - v) ** 2).sum(axis) / ((v**2).sum(axis) + 1e-30)
    return np.sqrt(e2)


# ---------------------------------------------------------------- 数据生成


def _unit(rng, *shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _make_keys(rng, m, keytype):
    if keytype == "iid":
        return _unit(rng, m, D)
    # corr：低有效秩谱（奇异值几何衰减 1→0.05）下的相关 key
    a = rng.normal(size=(D, D)).astype(np.float32)
    q1, _ = np.linalg.qr(a)
    q2, _ = np.linalg.qr(rng.normal(size=(D, D)).astype(np.float32))
    s = np.geomspace(1.0, 0.05, D).astype(np.float32)
    W = (q1 * s[None, :]) @ q2.T
    z = rng.normal(size=(m, D)).astype(np.float32)
    k = z @ W.T
    return (k / np.linalg.norm(k, axis=-1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- Exp 1


def exp1(seed):
    rng = np.random.default_rng(seed)
    cells = [(None, 0.0), (None, 0.001)] + [
        (l0, dec) for l0 in (0.01, 0.05, 0.15) for dec in (0.0, 0.001)
    ]
    labels = [
        ("off" if l0 is None else f"l0={l0}") + f",g={dec}" for l0, dec in cells
    ]
    gates = [l0 for l0, _ in cells]
    print(f"\n=== Exp1 MQAR 容量（热键 25%×8 次，β=0.8，period={PERIOD}）===")
    for keytype in ("iid", "corr"):
        for hotmode in ("rewrite", "query"):
            for m in (D, 2 * D, 4 * D):
                n_hot = m // 4
                reps, T = 8, m + (m // 4) * 7
                ks = np.zeros((len(cells), T, D), np.float32)
                vs = np.zeros((len(cells), T, DV), np.float32)
                bts = np.zeros((len(cells), T), np.float32)
                keys_q = np.zeros((len(cells), m, D), np.float32)
                vals_q = np.zeros((len(cells), m, DV), np.float32)
                for ci in range(len(cells)):
                    keys = _make_keys(rng, m, keytype)
                    vals = rng.normal(size=(m, DV)).astype(np.float32) / DV**0.5
                    idx = np.concatenate(
                        [np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)]
                    )
                    rng.shuffle(idx)
                    seen = set()
                    for t, i in enumerate(idx):
                        first = i not in seen
                        seen.add(i)
                        bts[ci, t] = 0.8 if (first or hotmode == "rewrite") else 0.0
                    ks[ci], vs[ci] = keys[idx], vals[idx]
                    keys_q[ci], vals_q[ci] = keys, vals
                lg = np.zeros((len(cells), T, D), np.float32)
                for ci, (_, dec) in enumerate(cells):
                    lg[ci] = -dec
                S = run_stream(
                    mx.array(ks), mx.array(vs), mx.array(lg), mx.array(bts), gates
                )
                e = rel_rmse(S, mx.array(keys_q), mx.array(vals_q))
                print(f"\n[{keytype}/{hotmode}] m={m} (T={T}, {T // PERIOD} 次门控)")
                print(f"{'cell':<18}{'hot':>8}{'cold':>8}{'all':>8}")
                for ci, lab in enumerate(labels):
                    eh, ec = np.median(e[ci, :n_hot]), np.median(e[ci, n_hot:])
                    print(f"{lab:<18}{eh:8.3f}{ec:8.3f}{np.median(e[ci]):8.3f}")


# ---------------------------------------------------------------- Exp 2


def exp2(seed):
    rng = np.random.default_rng(seed)
    cells = [(None, 0.0), (None, 0.001)] + [
        (l0, dec) for l0 in (0.01, 0.05, 0.15) for dec in (0.0, 0.001)
    ]
    labels = [
        ("off" if l0 is None else f"l0={l0}") + f",g={dec}" for l0, dec in cells
    ]
    gates = [l0 for l0, _ in cells]
    print(f"\n=== Exp2 needle（t=0 一次写入 β=1，随后 m 个一次性干扰 β=0.8）===")
    for m in (D, 2 * D, 4 * D, 8 * D):
        T = 1 + m
        ks = np.zeros((len(cells), T, D), np.float32)
        vs = np.zeros((len(cells), T, DV), np.float32)
        bts = np.full((len(cells), T), 0.8, np.float32)
        bts[:, 0] = 1.0
        kn = np.zeros((len(cells), 1, D), np.float32)
        vn = np.zeros((len(cells), 1, DV), np.float32)
        for ci in range(len(cells)):
            keys = _unit(rng, m + 1, D)
            vals = rng.normal(size=(m + 1, DV)).astype(np.float32) / DV**0.5
            ks[ci], vs[ci] = keys, vals
            kn[ci, 0], vn[ci, 0] = keys[0], vals[0]
        lg = np.zeros((len(cells), T, D), np.float32)
        for ci, (_, dec) in enumerate(cells):
            lg[ci] = -dec
        S = run_stream(mx.array(ks), mx.array(vs), mx.array(lg), mx.array(bts), gates)
        e = rel_rmse(S, mx.array(kn), mx.array(vn))[:, 0]
        print(f"\nm={m} (滞后 {m} token, {T // PERIOD} 次门控)")
        for ci, lab in enumerate(labels):
            print(f"  {lab:<18}needle RMSE = {e[ci]:.3f}")


# ---------------------------------------------------------------- Exp 3


def exp3(seed):
    rng = np.random.default_rng(seed)
    m = 4 * D
    # 轴对齐各向异性 key 统计：通道标准差几何衰减 1→0.05
    s_c = np.geomspace(1.0, 0.05, D).astype(np.float32)
    # 适配该统计的逐通道衰减：信息通道几乎不衰，尾部通道快衰
    log_g_c = (-0.3 * (1.0 - s_c)).astype(np.float32)
    R, _ = np.linalg.qr(rng.normal(size=(D, D)).astype(np.float32))
    cells = [
        ("aligned  off", False, None),
        ("aligned  l0=.05", False, 0.05),
        ("rotated  off", True, None),
        ("rotated  l0=.05", True, 0.05),
    ]
    T = m
    ks = np.zeros((len(cells), T, D), np.float32)
    vs = np.zeros((len(cells), T, DV), np.float32)
    keys_q = np.zeros((len(cells), m, D), np.float32)
    vals_q = np.zeros((len(cells), m, DV), np.float32)
    for ci, (_, rot, _) in enumerate(cells):
        keys = _unit(rng, m, D) * s_c[None, :]
        keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
        if rot:
            keys = keys @ R.T
        vals = rng.normal(size=(m, DV)).astype(np.float32) / DV**0.5
        ks[ci], vs[ci] = keys, vals
        keys_q[ci], vals_q[ci] = keys, vals
    lg = np.broadcast_to(log_g_c[None, None, :], (len(cells), T, D)).copy()
    bts = np.full((len(cells), T), 0.8, np.float32)
    S = run_stream(
        mx.array(ks), mx.array(vs), mx.array(lg), mx.array(bts),
        [g for _, _, g in cells],
    )
    e = rel_rmse(S, mx.array(keys_q), mx.array(vals_q))
    print(f"\n=== Exp3 旋转 key（m={m}，各向异性衰减适配旧基，{T // PERIOD} 次门控）===")
    for ci, (lab, _, _) in enumerate(cells):
        print(f"  {lab:<18}median RMSE = {np.median(e[ci]):.3f}  "
              f"p90 = {np.percentile(e[ci], 90):.3f}")


# ---------------------------------------------------------------- Exp 4


def exp4(seed):
    """λ 混合强度扫描：排除「λ=1 混太狠」对 Exp1 负面结果的替代解释。

    取 Exp1 两个 headline block，膝盖 l0∈{0.01,0.05} × λ∈{0.25,0.5,1.0}，
    衰减 g=0.001（此前最优 cell 所在的衰减档）。
    """
    rng = np.random.default_rng(seed)
    cells = [(None, 0.0)] + [
        ((l0, lam), 0.001)
        for l0 in (0.01, 0.05)
        for lam in (0.25, 0.5, 1.0)
    ]
    labels = [
        ("off,g=0" if g is None else f"l0={g[0]},lam={g[1]},g=.001")
        for g, _ in cells
    ]
    gates = [g for g, _ in cells]
    print(f"\n=== Exp4 λ 扫描（period={PERIOD}）===")
    for keytype, hotmode, m in (("iid", "rewrite", 4 * D), ("corr", "rewrite", 2 * D)):
        n_hot = m // 4
        reps, T = 8, m + (m // 4) * 7
        ks = np.zeros((len(cells), T, D), np.float32)
        vs = np.zeros((len(cells), T, DV), np.float32)
        bts = np.full((len(cells), T), 0.8, np.float32)
        keys_q = np.zeros((len(cells), m, D), np.float32)
        vals_q = np.zeros((len(cells), m, DV), np.float32)
        for ci in range(len(cells)):
            keys = _make_keys(rng, m, keytype)
            vals = rng.normal(size=(m, DV)).astype(np.float32) / DV**0.5
            idx = np.concatenate([np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)])
            rng.shuffle(idx)
            ks[ci], vs[ci] = keys[idx], vals[idx]
            keys_q[ci], vals_q[ci] = keys, vals
        lg = np.zeros((len(cells), T, D), np.float32)
        for ci, (_, dec) in enumerate(cells):
            lg[ci] = -dec
        S = run_stream(mx.array(ks), mx.array(vs), mx.array(lg), mx.array(bts), gates)
        e = rel_rmse(S, mx.array(keys_q), mx.array(vals_q))
        print(f"\n[{keytype}/{hotmode}] m={m} (T={T}, {T // PERIOD} 次门控)")
        print(f"{'cell':<22}{'hot':>8}{'cold':>8}{'all':>8}")
        for ci, lab in enumerate(labels):
            eh, ec = np.median(e[ci, :n_hot]), np.median(e[ci, n_hot:])
            print(f"{lab:<22}{eh:8.3f}{ec:8.3f}{np.median(e[ci]):8.3f}")


# ---------------------------------------------------------------- Exp 5 (SPR)

LAM_RELS = (0.01, 0.1, 0.3, 1.0, 3.0)


def _spr_rows(S, keys, vals):
    """同一状态上 baseline + λ 扫描的读出。S (D,Dv)，keys (m,D)，vals (m,Dv)，
    全 float64 精确解。返回 [(label, 每键 scale-fitted RMSE (m,))]。"""
    A = S @ S.T
    mu = float(np.trace(A)) / D
    Kt = keys.T  # (D,m)
    rows = [("base", (S.T @ Kt).T)]
    for lr in LAM_RELS:
        Rk = np.linalg.solve(A + (lr * mu) * np.eye(D), Kt)
        rows.append((f"λ={lr}", (S.T @ Rk).T))
    return [(lab, _scaled_rmse(r, vals, axis=-1)) for lab, r in rows]


def exp5(seed):
    rng = np.random.default_rng(seed)
    print("\n=== Exp5a SPR × MQAR（scale-fitted RMSE；状态无损，仅改读出）===")
    for keytype in ("iid", "corr"):
        for hotmode in ("rewrite", "query"):
            for m in (D, 2 * D, 4 * D, 8 * D):
                n_hot = m // 4
                reps, T = 8, m + n_hot * 7
                keys = _make_keys(rng, m, keytype)
                vals = (rng.normal(size=(m, DV)) / DV**0.5).astype(np.float32)
                idx = np.concatenate(
                    [np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)]
                )
                rng.shuffle(idx)
                seen, bts = set(), np.zeros(T, np.float32)
                for t, i in enumerate(idx):
                    first = i not in seen
                    seen.add(i)
                    bts[t] = 0.8 if (first or hotmode == "rewrite") else 0.0
                ks, vs = keys[idx], vals[idx]
                H = keys.T @ keys  # 等权理想 key Gram（自估计诊断参照）
                print(f"\n[{keytype}/{hotmode}] m={m} (T={T})")
                for dec in (0.0, 0.001):
                    lg = np.full((1, T, D), -dec, np.float32)
                    S = run_stream(
                        mx.array(ks[None]), mx.array(vs[None]), mx.array(lg),
                        mx.array(bts[None]), [None],
                    )
                    Sn = np.array(S[0]).astype(np.float64)
                    A = Sn @ Sn.T
                    dist = np.linalg.norm(A / np.linalg.norm(A) - H / np.linalg.norm(H))
                    print(f" g={dec}  diag||SSᵀ−H||_F/||·||_F = {dist:.3f}")
                    for lab, e in _spr_rows(Sn, keys.astype(np.float64), vals.astype(np.float64)):
                        eh, ec = np.median(e[:n_hot]), np.median(e[n_hot:])
                        print(
                            f"  g={dec:<7}{lab:<8} hot {eh:.3f}  "
                            f"cold {ec:.3f}  all {np.median(e):.3f}"
                        )
    print("\n=== Exp5b SPR × needle（t=0 一次写入 β=1，m 个一次性干扰）===")
    for m in (D, 2 * D, 4 * D, 8 * D):
        T = 1 + m
        keys = _unit(rng, m + 1, D)
        vals = (rng.normal(size=(m + 1, DV)) / DV**0.5).astype(np.float32)
        bts = np.full(T, 0.8, np.float32)
        bts[0] = 1.0
        print(f"\nm={m} (滞后 {m} token)")
        for dec in (0.0, 0.001):
            lg = np.full((1, T, D), -dec, np.float32)
            S = run_stream(
                mx.array(keys[None]), mx.array(vals[None]), mx.array(lg),
                mx.array(bts[None]), [None],
            )
            Sn = np.array(S[0]).astype(np.float64)
            out = []
            for lab, e in _spr_rows(Sn, keys[:1].astype(np.float64), vals[:1].astype(np.float64)):
                out.append(f"{lab} {e[0]:.3f}")
            print(f"  g={dec:<7}needle: " + "  ".join(out))


# ---------------------------------------------------------------- Exp 6 (touch)


def exp6(seed):
    """E1：touch 规则（读取门控保留）。ρ∈{0,0.5,1}，q=k，raw rel-RMSE
    （与 Exp1 同口径，锚点：g≈0 时 query-hot 0.87 / rewrite-hot 0.26）。

    预注册预测：
      query 模式强衰减下 hot 显著改善（≥0.1），向 rewrite 水平靠拢；
      rewrite 模式不变或改善；cold 与 once（无复用负载）中性；
      任何负载出现系统性损害 → kill。
    注意 touch 只在有衰减时非平凡（g=0 时 I−D=0），故衰减档取
    {0.001, 0.003, 0.01} 制造真实老化压力。
    """
    rng = np.random.default_rng(seed)
    rhos = [0.0, 0.5, 1.0]
    print("\n=== Exp6 touch（读取门控保留，raw rel-RMSE）===")
    for keytype in ("iid", "corr"):
        for hotmode in ("query", "rewrite", "once"):
            for m in (D, 2 * D, 4 * D):
                n_hot = m // 4
                reps = 1 if hotmode == "once" else 8
                T = m + n_hot * (reps - 1)
                keys = _make_keys(rng, m, keytype)
                vals = (rng.normal(size=(m, DV)) / DV**0.5).astype(np.float32)
                if hotmode == "once":
                    idx = np.arange(m)
                    rng.shuffle(idx)
                    bts = np.full(T, 0.8, np.float32)
                else:
                    idx = np.concatenate(
                        [np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)]
                    )
                    rng.shuffle(idx)
                    seen, bts = set(), np.zeros(T, np.float32)
                    for t, i in enumerate(idx):
                        first = i not in seen
                        seen.add(i)
                        bts[t] = 0.8 if (first or hotmode == "rewrite") else 0.0
                ks, vs = keys[idx], vals[idx]
                print(f"\n[{keytype}/{hotmode}] m={m} (T={T})")
                for dec in (0.001, 0.003, 0.01):
                    lg = np.full((1, T, D), -dec, np.float32)
                    S = run_stream(
                        mx.array(np.repeat(ks[None], 3, axis=0)),
                        mx.array(np.repeat(vs[None], 3, axis=0)),
                        mx.array(np.repeat(lg, 3, axis=0)),
                        mx.array(np.repeat(bts[None], 3, axis=0)),
                        [None] * 3,
                        q=mx.array(np.repeat(ks[None], 3, axis=0)),
                        rhos=rhos,
                    )
                    e = rel_rmse(
                        S,
                        mx.array(np.repeat(keys[None], 3, axis=0)),
                        mx.array(np.repeat(vals[None], 3, axis=0)),
                    )
                    for ci, r in enumerate(rhos):
                        if hotmode == "once":
                            print(f"  g={dec:<7}ρ={r:<4} all {np.median(e[ci]):.3f}")
                        else:
                            eh, ec = np.median(e[ci, :n_hot]), np.median(e[ci, n_hot:])
                            print(
                                f"  g={dec:<7}ρ={r:<4} hot {eh:.3f}  "
                                f"cold {ec:.3f}  all {np.median(e[ci]):.3f}"
                            )


# ---------------------------------------------------------------- Exp 7 (replay)


def _run_np(k, v, log_g, beta, key_ids=None, replay=None):
    """numpy 单 cell delta 递推（与 run_stream 同三行，f64），可选巩固回放。

    replay dict: R(缓冲容量), B(每 chunk 回放条数), C(chunk), beta_r, lru。
    准入按写入 surprise ‖β(v−Sᵀk)‖（HOLA 口径，scan 里已物化的 vt 行范数）；
    调度 LRU=查询计数优先（合成负载 q=k，用 key id 精确匹配——生产实现
    是 Q·Kᵀ 小 GEMM 的近似，这里取理想上限）/ RR=轮转对照；回放写 g=0、
    剂量由 delta 残差自定量。返回 (S, n_replays, 平均回放赤字)。
    """
    T = k.shape[0]
    S = np.zeros((D, DV))
    if replay is None:
        for t in range(T):
            S *= np.exp(log_g[t])[:, None]
            S += beta[t] * np.outer(k[t], v[t] - k[t] @ S)
        return S, 0, 0.0
    R, B, C, br, lru = (replay[x] for x in ("R", "B", "C", "beta_r", "lru"))
    buf, by_id, rr_ptr, n_rep, deficit = [], {}, 0, 0, []
    for t in range(T):
        S *= np.exp(log_g[t])[:, None]
        delta = beta[t] * (v[t] - k[t] @ S)
        S += np.outer(k[t], delta)
        kid = int(key_ids[t])
        e = by_id.get(kid)
        if e is not None:
            e["q"] += 1
        elif beta[t] > 0:
            e = {"k": k[t].copy(), "v": v[t].copy(),
                 "s": float(np.linalg.norm(delta)), "q": 0, "last": t, "id": kid}
            if len(buf) < R:
                buf.append(e)
                by_id[kid] = e
            else:
                i_min = min(range(len(buf)), key=lambda i: buf[i]["s"])
                if e["s"] > buf[i_min]["s"]:
                    del by_id[buf[i_min]["id"]]
                    buf[i_min] = e
                    by_id[kid] = e
        if (t + 1) % C == 0 and buf:
            if lru:
                sel = sorted(buf, key=lambda e: (-e["q"], e["last"]))[:B]
            else:
                sel = [buf[(rr_ptr + j) % len(buf)] for j in range(min(B, len(buf)))]
                rr_ptr = (rr_ptr + B) % len(buf)
            for e in sel:
                res = e["v"] - e["k"] @ S
                deficit.append(float(np.linalg.norm(res)))
                S += br * np.outer(e["k"], res)
                e["q"] = 0
                e["last"] = t
                n_rep += 1
    return S, n_rep, float(np.mean(deficit)) if deficit else 0.0


def _rel_rmse_np(S, keys, vals):
    r = keys @ S
    return np.sqrt(((r - vals) ** 2).sum(-1) / ((vals**2).sum(-1) + 1e-12))


def exp7(seed):
    """E1′：巩固回放。预测（touch 未达门槛的同一把尺子）：
      query-hot 0.87 → ≤0.35（R≥热键数时），改善 <0.1 → kill；
      rewrite 中性（复写清零赤字，回放近无操作——由 deficit 列直接验证）；
      once 无复用负载 = 污染上界（剂量-反应）；
      iso-D=124 对照：同等额外内存扩容必须输，否则回放只是变相扩容。
    """
    global D, DV
    rng = np.random.default_rng(seed)
    conds = [
        ("base", None),
        ("rep-RR", dict(R=32, B=4, C=16, beta_r=1.0, lru=False)),
        ("rep-LRU", dict(R=32, B=4, C=16, beta_r=1.0, lru=True)),
        ("rep-LRU.5", dict(R=32, B=4, C=16, beta_r=0.5, lru=True)),
    ]
    print("\n=== Exp7 巩固回放（R=32, B=4/chunk, raw rel-RMSE, f64）===")
    for keytype in ("iid", "corr"):
        for hotmode in ("query", "rewrite", "once"):
            for m in (D, 2 * D, 4 * D):
                n_hot = m // 4
                reps = 1 if hotmode == "once" else 8
                T = m + n_hot * (reps - 1)
                keys = _make_keys(rng, m, keytype)
                vals = (rng.normal(size=(m, DV)) / DV**0.5).astype(np.float64)
                if hotmode == "once":
                    idx = np.arange(m)
                    rng.shuffle(idx)
                    bts = np.full(T, 0.8)
                else:
                    idx = np.concatenate(
                        [np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)]
                    )
                    rng.shuffle(idx)
                    seen, bts = set(), np.zeros(T)
                    for t, i in enumerate(idx):
                        first = i not in seen
                        seen.add(i)
                        bts[t] = 0.8 if (first or hotmode == "rewrite") else 0.0
                ks, vs = keys[idx].astype(np.float64), vals[idx]
                print(f"\n[{keytype}/{hotmode}] m={m} (T={T})")
                for dec in (0.0, 0.001, 0.01):
                    lg = np.full((T, D), -dec)
                    print(f" g={dec}")
                    for lab, rp in conds:
                        S, nrep, dfc = _run_np(ks, vs, lg, bts, idx, rp)
                        e = _rel_rmse_np(S, keys, vals)
                        if hotmode == "once":
                            print(
                                f"  {lab:<11} all {np.median(e):.3f}"
                                f"   [rep={nrep} deficit={dfc:.3f}]"
                            )
                        else:
                            eh, ec = np.median(e[:n_hot]), np.median(e[n_hot:])
                            print(
                                f"  {lab:<11} hot {eh:.3f}  cold {ec:.3f}  "
                                f"all {np.median(e):.3f}"
                                f"   [rep={nrep} deficit={dfc:.3f}]"
                            )
                    if hotmode == "query":
                        # iso-memory：R(D+Dv)=6144 floats ≈ D² 9216→124²=15376
                        D0, DV0 = D, DV
                        D = DV = 124
                        try:
                            keys_i = _make_keys(rng, m, keytype)
                            vals_i = (
                                rng.normal(size=(m, DV)) / DV**0.5
                            ).astype(np.float64)
                            lg_i = np.full((T, D), -dec)
                            S, _, _ = _run_np(
                                keys_i[idx].astype(np.float64), vals_i[idx],
                                lg_i, bts, idx, None,
                            )
                            e = _rel_rmse_np(S, keys_i, vals_i)
                            eh, ec = np.median(e[:n_hot]), np.median(e[n_hot:])
                            print(
                                f"  {'iso-D124':<11} hot {eh:.3f}  cold {ec:.3f}  "
                                f"all {np.median(e):.3f}"
                            )
                        finally:
                            D, DV = D0, DV0


# ---------------------------------------------------------------- Exp 8 (replay 带宽探针)


def _run_np2(k, v, log_g, beta, key_ids=None, replay=None):
    """_run_np 的扩展：调度模式 lru/rr/deficit（每 chunk 对全部缓冲条目
    测赤字 K_B·S，回放赤字最大的 B 条——自定量恒等式的字面实现，
    每 chunk 额外一次 R×D×Dv 元数据读出）。
    """
    T = k.shape[0]
    S = np.zeros((D, DV))
    if replay is None:
        for t in range(T):
            S *= np.exp(log_g[t])[:, None]
            S += beta[t] * np.outer(k[t], v[t] - k[t] @ S)
        return S, 0, 0.0
    R, B, C, br, mode = (replay[x] for x in ("R", "B", "C", "beta_r", "mode"))
    buf, by_id, rr_ptr, n_rep, deficit = [], {}, 0, 0, []
    for t in range(T):
        S *= np.exp(log_g[t])[:, None]
        delta = beta[t] * (v[t] - k[t] @ S)
        S += np.outer(k[t], delta)
        kid = int(key_ids[t])
        e = by_id.get(kid)
        if e is not None:
            e["q"] += 1
        elif beta[t] > 0:
            e = {"k": k[t].copy(), "v": v[t].copy(),
                 "s": float(np.linalg.norm(delta)), "q": 0, "last": t, "id": kid}
            if len(buf) < R:
                buf.append(e)
                by_id[kid] = e
            else:
                i_min = min(range(len(buf)), key=lambda i: buf[i]["s"])
                if e["s"] > buf[i_min]["s"]:
                    del by_id[buf[i_min]["id"]]
                    buf[i_min] = e
                    by_id[kid] = e
        if (t + 1) % C == 0 and buf:
            if mode == "lru":
                sel = sorted(buf, key=lambda e: (-e["q"], e["last"]))[:B]
            elif mode == "deficit":
                sel = sorted(
                    buf, key=lambda e: -np.linalg.norm(e["v"] - e["k"] @ S)
                )[:B]
            else:
                sel = [buf[(rr_ptr + j) % len(buf)] for j in range(min(B, len(buf)))]
                rr_ptr = (rr_ptr + B) % len(buf)
            for e in sel:
                res = e["v"] - e["k"] @ S
                deficit.append(float(np.linalg.norm(res)))
                S += br * np.outer(e["k"], res)
                e["q"] = 0
                e["last"] = t
                n_rep += 1
    return S, n_rep, float(np.mean(deficit)) if deficit else 0.0


def exp8(seed):
    """回放带宽探针：E1′ 在 B=4/C=16 恰处盈亏平衡（热键回放周期 ≈ 96
    token ≈ 干涉累积时间尺度 D）。扫描 B∈{4,8,16}（B/C = 25%/50%/100%
    写入开销）× 调度（lru/deficit），裁决是否存在 affordable 带宽使
    query-hot ≤ 0.35。R=32，βr=1。
    """
    rng = np.random.default_rng(seed)
    conds = [("base", None)] + [
        (f"{mode}-B{B}", dict(R=32, B=B, C=16, beta_r=1.0, mode=mode))
        for mode in ("lru", "deficit")
        for B in (4, 8, 16)
    ]
    print("\n=== Exp8 回放带宽探针（iid/query, R=32, βr=1）===")
    for m, dec in ((D, 0.001), (D, 0.01), (4 * D, 0.001)):
        n_hot = m // 4
        reps, T = 8, m + n_hot * 7
        keys = _make_keys(rng, m, "iid")
        vals = (rng.normal(size=(m, DV)) / DV**0.5).astype(np.float64)
        idx = np.concatenate([np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)])
        rng.shuffle(idx)
        seen, bts = set(), np.zeros(T)
        for t, i in enumerate(idx):
            first = i not in seen
            seen.add(i)
            bts[t] = 0.8 if first else 0.0
        ks, vs = keys[idx].astype(np.float64), vals[idx]
        lg = np.full((T, D), -dec)
        print(f"\n[iid/query] m={m} (T={T}) g={dec}")
        for lab, rp in conds:
            S, nrep, dfc = _run_np2(ks, vs, lg, bts, idx, rp)
            e = _rel_rmse_np(S, keys, vals)
            eh, ec = np.median(e[:n_hot]), np.median(e[n_hot:])
            print(
                f"  {lab:<12} hot {eh:.3f}  cold {ec:.3f}  all {np.median(e):.3f}"
                f"   [rep={nrep} deficit={dfc:.3f}]"
            )


# ---------------------------------------------------------------- Exp 9 (β 噪声球)


def exp9(seed):
    """β 扫描：检验「rewrite-hot 地板 = 恒步长 Kaczmarz/LMS 噪声球，
    半径 ∝ β^{1/2}」的定量预测。iid/rewrite，m∈{D,2D}，g∈{0,0.001}。
    报告 log-log 斜率（预测 ≈ 0.5）。"""
    rng = np.random.default_rng(seed)
    betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    print("\n=== Exp9 β 扫描（iid/rewrite，raw rel-RMSE）===")
    for m in (D, 2 * D):
        n_hot = m // 4
        reps, T = 8, m + n_hot * 7
        keys = _make_keys(rng, m, "iid")
        vals = (rng.normal(size=(m, DV)) / DV**0.5).astype(np.float32)
        idx = np.concatenate([np.tile(np.arange(n_hot), reps), np.arange(n_hot, m)])
        rng.shuffle(idx)
        ks = np.repeat(keys[idx][None], len(betas), axis=0)
        vs = np.repeat(vals[idx][None], len(betas), axis=0)
        kq = np.repeat(keys[None], len(betas), axis=0)
        vq = np.repeat(vals[None], len(betas), axis=0)
        for dec in (0.0, 0.001):
            lg = np.full((len(betas), T, D), -dec, np.float32)
            bts = np.array(betas, np.float32)[:, None] * np.ones((1, T), np.float32)
            S = run_stream(
                mx.array(ks), mx.array(vs), mx.array(lg), mx.array(bts),
                [None] * len(betas),
            )
            e = rel_rmse(S, mx.array(kq), mx.array(vq))
            print(f"\nm={m} g={dec}")
            hots = []
            for ci, bw in enumerate(betas):
                eh, ec = np.median(e[ci, :n_hot]), np.median(e[ci, n_hot:])
                hots.append(eh)
                print(f"  β={bw:<5} hot {eh:.3f}  cold {ec:.3f}  all {np.median(e[ci]):.3f}")
            lb, lh = np.log(np.array(betas)), np.log(np.array(hots))
            slope = np.polyfit(lb, lh, 1)[0]
            print(f"  log-log 斜率（hot ~ β^x）: x = {slope:.2f}  [预测 0.5]")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp", default="all",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "all"],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _parity_check()
    if args.exp in ("1", "all"):
        exp1(args.seed)
    if args.exp in ("2", "all"):
        exp2(args.seed)
    if args.exp in ("3", "all"):
        exp3(args.seed)
    if args.exp in ("4", "all"):
        exp4(args.seed)
    if args.exp in ("5", "all"):
        exp5(args.seed)
    if args.exp in ("6", "all"):
        exp6(args.seed)
    if args.exp in ("7", "all"):
        exp7(args.seed)
    if args.exp in ("8", "all"):
        exp8(args.seed)
    if args.exp in ("9", "all"):
        exp9(args.seed)
