"""隔离调试：fused vjp 的数学（padded GEMM 反向）vs autodiff 穿过旧前向算子。
不涉及 custom_function——直接给定同一 dy，比较 dxb/dgu/ddw。
f32 小形状：E=2, EG=2, K=1, D=64, I=13, B=1, T=8。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn

DT = mx.float32
B, T, D, I, E, K, EG = 1, 8, 64, 13, 2, 1, 2
M = B * T
G = M * K

mx.random.seed(0)
xf = mx.random.normal((M, D), dtype=DT) * 0.5
gu = mx.random.normal((E, 2 * I, D), dtype=DT) * 0.1
dw = mx.random.normal((E, D, I), dtype=DT) * 0.3
idx = mx.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=mx.int32)  # (M,) K=1
w = mx.random.uniform(0.2, 1.0, (M, K)).astype(DT)
C = mx.random.normal((M, D))

# ---- 复刻 _sparse_forward 的共享前段（argsort/scatter → xb/row/counts/caps/starts）----
exps = idx.reshape(G)
order = mx.argsort(exps)
exps_s = exps[order]
counts = mx.zeros((E,), dtype=mx.int32).at[exps].add(1)
offsets = mx.cumsum(counts) - counts
tok_s = (order // K).astype(mx.int32)
w_s = w.reshape(-1)[order]
n_groups = (E + EG - 1) // EG
AL = 128
caps_l = counts.tolist()
caps = []
for gi in range(n_groups):
    base = max(caps_l[gi * EG : gi * EG + EG]) * 5 // 4 + 64
    caps.append(((base + AL - 1) // AL) * AL)
starts = [0]
for gi in range(n_groups):
    starts.append(starts[-1] + EG * caps[gi])
total_rows = starts[-1]
caps_dev = mx.array(caps, dtype=mx.int32)
start_dev = mx.array(starts[:-1], dtype=mx.int32)
grp = exps_s // EG
rank = mx.arange(G, dtype=mx.int32) - offsets[exps_s]
cap_g = caps_dev[grp]
row = start_dev[grp] + (exps_s - grp * EG) * cap_g + rank
row = mx.where(rank >= cap_g, total_rows, row)
xb = mx.zeros((total_rows + 1, D), dtype=DT).at[row].add(xf[tok_s])
mx.eval(xb, counts, row, tok_s, w_s)
print("counts:", counts.tolist(), "caps:", caps, "starts:", starts)


def downstream(y_flat):
    yw = y_flat[row] * w_s[:, None].astype(y_flat.dtype)
    out = mx.zeros((M, D), dtype=mx.float32).at[tok_s].add(yw.astype(mx.float32))
    return (out * C).sum()


def old_y(xb_, gu_, dw_):
    gu_t = gu_.swapaxes(-1, -2)
    dw_t = dw_.swapaxes(-1, -2)
    ys = []
    for gi in range(n_groups):
        e0, e1 = gi * EG, min(gi * EG + EG, E)
        eg = e1 - e0
        Cg = caps[gi]
        xg = xb_[starts[gi] : starts[gi] + eg * Cg].reshape(eg, Cg, D)
        g_, u_ = mx.split(xg @ gu_t[e0:e1], 2, axis=-1)
        ys.append(((nn.silu(g_) * u_) @ dw_t[e0:e1]).reshape(eg * Cg, D))
    ys.append(mx.zeros((1, D), dtype=DT))
    return mx.concatenate(ys, axis=0)


# ---- 参考：autodiff 同时穿过 old_y 与 downstream ----
def loss_ref(xb_, gu_, dw_):
    return downstream(old_y(xb_, gu_, dw_))


vg3 = mx.value_and_grad(loss_ref, argnums=(0, 1, 2))
val_r, (dx_ref, dgu_ref, ddw_ref) = vg3(xb, gu, dw)
mx.eval(val_r, dx_ref, dgu_ref, ddw_ref)

# ---- 我的 vjp：先取 dy（downstream 对 y_flat 的梯度），再跑 Stage A 算子 ----
y_flat = old_y(xb, gu, dw)
vg_dy = mx.value_and_grad(downstream)
_, dy_full = vg_dy(y_flat)
mx.eval(dy_full)

gu_t = gu.swapaxes(-1, -2)
dxb_parts, dgu_parts, ddw_parts = [], [], []
for gi in range(n_groups):
    e0, e1 = gi * EG, min(gi * EG + EG, E)
    eg = e1 - e0
    Cg = caps[gi]
    sl = slice(starts[gi], starts[gi] + eg * Cg)
    xg = xb[sl].reshape(eg, Cg, D)
    dyg = dy_full[sl].reshape(eg, Cg, D)
    gu_ = xg @ gu_t[e0:e1]
    g, u = mx.split(gu_, 2, axis=-1)
    h_g = nn.silu(g) * u
    dh = dyg @ dw[e0:e1]
    ddw_parts.append(mx.matmul(dyg.swapaxes(-1, -2), h_g))
    sg = mx.sigmoid(g)
    dg = dh * u * (sg * (1.0 + g * (1.0 - sg)))
    du = dh * (g * sg)
    dgu = mx.concatenate([dg, du], axis=-1)
    dgu_parts.append(mx.matmul(dgu.swapaxes(-1, -2), xg))
    dxb_parts.append((dgu @ gu[e0:e1]).reshape(eg * Cg, D))
dx_mine = mx.concatenate(dxb_parts + [mx.zeros((1, D), dtype=DT)], axis=0)
dgu_mine = mx.concatenate(dgu_parts, axis=0)
ddw_mine = mx.concatenate(ddw_parts, axis=0)
mx.eval(dx_mine, dgu_mine, ddw_mine)


def rep(name, a, b):
    d = (a - b).abs()
    print(
        f"{name}: max|diff|={d.max().item():.3e}  max|ref|={b.abs().max().item():.3e}"
    )
    return d


rep("dxb ", dx_mine, dx_ref)
rep("dgu ", dgu_mine, dgu_ref)
rep("ddw ", ddw_mine, ddw_ref)

# 定位 dxb 差异行
d = (dx_mine - dx_ref).abs().max(axis=-1)
bad = [i for i, v in enumerate(d.tolist()) if v > 1e-5]
print("dxb 差异行:", bad[:40], "（共", len(bad), "行）")
print("row 布局: counts=", counts.tolist(), " caps=", caps, " starts=", starts)

# ---- 逐中间量探针：直接对 (g,u) 求导 vs 我的 dg/du 公式 ----
dw_t = dw.swapaxes(-1, -2)
Cg0 = caps[0]
xg0 = xb[0 : EG * Cg0].reshape(EG, Cg0, D)
dyg0 = dy_full[0 : EG * Cg0].reshape(EG, Cg0, D)
g0, u0 = mx.split(xg0 @ gu_t[0:EG], 2, axis=-1)


def half_loss(g_, u_):
    return (dyg0 * ((nn.silu(g_) * u_) @ dw_t[0:EG])).sum()


_, (dg_ref, du_ref) = mx.value_and_grad(half_loss, argnums=(0, 1))(g0, u0)
dh0 = dyg0 @ dw[0:EG]
sg0 = mx.sigmoid(g0)
dg_m = dh0 * u0 * (sg0 * (1.0 + g0 * (1.0 - sg0)))
du_m = dh0 * (g0 * sg0)
mx.eval(dg_ref, du_ref, dg_m, du_m)
rep("dg  ", dg_m, dg_ref)
rep("du  ", du_m, du_ref)

# dh 本身：对 h 求导
h0 = nn.silu(g0) * u0


def h_loss(h_):
    return (dyg0 * (h_ @ dw_t[0:EG])).sum()


_, dh_ref = mx.value_and_grad(h_loss)(h0)
rep("dh  ", dh0, dh_ref)
