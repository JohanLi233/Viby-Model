"""KDA chunk 分解段内部归因（累积前缀差分，compile 口径）。

分解段 f+b 约 17.6ms/层（×7 层 ≈ 123ms/步），访存下界仅 ~2.3ms。里面
混着 elementwise 链（cumsum/exp/乘）和一堆 (16,96)@(96,16)、16³ 的极小
batched GEMM（batch = B·H·NC = 6144）。先确认是谁主导，再决定融合
kernel 的设计重点。

用法: uv run python experiments/probe_prep_parts.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H, D = 8, 96
C = KDA_CHUNK
NC = T // C
BW = 400e9
NB = B * H * NC

print(f"B={B} T={T} H={H} D={D} C={C} NC={NC}  batch=B·H·NC={NB}")
print(f"每个 chunk 的 GEMM: (C,D)@(D,C)={C}x{D}x{C}, {C}³ 倍增, (C,C)@(C,D)\n")

q = (mx.random.normal((B, H, NC, C, D)) * 0.1).astype(mx.float32)
k = (mx.random.normal((B, H, NC, C, D)) * 0.1).astype(mx.float32)
v = (mx.random.normal((B, H, NC, C, D)) * 0.5).astype(mx.float32)
lg = mx.maximum(-mx.random.uniform(0.001, 2.0, (B, H, NC, C, D)), -4.0)
bt = mx.random.uniform(0, 1, (B, H, NC, C)).astype(mx.float32)
mx.eval(q, k, v, lg, bt)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


SL = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
LO = mx.tril(mx.ones((C, C), dtype=mx.bool_))
EYE = mx.eye(C, dtype=mx.float32)
ROUNDS = int(math.log2(C)) - 1


def p1(q_, k_, v_, lg_, bt_):
    """cumsum + 三个 exp 乘（elementwise 主体）"""
    gc = mx.cumsum(lg_, axis=-2)
    eg = mx.exp(gc)
    return q_ * eg, k_ * eg, k_ * mx.exp(-gc), gc


def p2(q_, k_, v_, lg_, bt_):
    """+ L = (β·ke)@kiᵀ 与掩码"""
    qe, ke, ki, gc = p1(q_, k_, v_, lg_, bt_)
    L = (bt_[..., None] * ke) @ mx.swapaxes(ki, -1, -2)
    L = mx.where(SL, L, mx.zeros_like(L))
    return qe, ke, ki, gc, L


def p3(q_, k_, v_, lg_, bt_):
    """+ Neumann 倍增求 (I+L)⁻¹（3 轮，6 个 C³ GEMM）"""
    qe, ke, ki, gc, L = p2(q_, k_, v_, lg_, bt_)
    P = -L
    X = EYE + P
    pp = P
    for _ in range(ROUNDS):
        pp = pp @ pp
        X = X + pp @ X
    return qe, ke, ki, gc, X


def p4(q_, k_, v_, lg_, bt_):
    """+ Afb 与 w=Afb@ke、u=Afb@v"""
    qe, ke, ki, gc, X = p3(q_, k_, v_, lg_, bt_)
    Afb = X * bt_[..., None, :]
    return qe, ki, gc, Afb @ ke, Afb @ v_


def p5(q_, k_, v_, lg_, bt_):
    """+ Aqk = qe@kiᵀ 与掩码"""
    qe, ki, gc, w_, u_ = p4(q_, k_, v_, lg_, bt_)
    A = qe @ mx.swapaxes(ki, -1, -2)
    return qe, gc, w_, u_, mx.where(LO, A, mx.zeros_like(A))


def p6(q_, k_, v_, lg_, bt_):
    """+ kd = k·e^{gl−gc} 与 egl（完整分解段）"""
    qe, gc, w_, u_, A = p5(q_, k_, v_, lg_, bt_)
    gl = gc[:, :, :, -1, :]
    return qe, w_, u_, A, k_ * mx.exp(gl[:, :, :, None, :] - gc), mx.exp(gl)


STAGES = [
    ("cumsum+exp 三路 (qe/ke/ki)", p1),
    ("+ L=(β·ke)@kiᵀ + 掩码", p2),
    ("+ Neumann 倍增 (6× C³)", p3),
    ("+ w=Afb@ke, u=Afb@v", p4),
    ("+ Aqk=qe@kiᵀ + 掩码", p5),
    ("+ kd, egl（完整分解段）", p6),
]

a = (q, k, v, lg, bt)
prev_f = prev_b = 0.0
res = []
print(f"{'累积前缀':<30}{'fwd':>8}{'Δfwd':>8}{'bwd':>8}{'Δbwd':>8}{'Δf+b':>8}")
for label, fn in STAGES:

    def loss(q_, k_, v_, lg_, bt_, _fn=fn):
        return sum((o**2).sum() for o in _fn(q_, k_, v_, lg_, bt_))

    cf = mx.compile(loss)
    cg = mx.compile(mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))
    f = timed(lambda: cf(*a))
    fb = timed(lambda: cg(*a))
    b = fb - f
    res.append((label, f - prev_f, b - prev_b))
    print(
        f"{label:<30}{f:>8.2f}{f - prev_f:>8.2f}{b:>8.2f}{b - prev_b:>8.2f}"
        f"{(f - prev_f) + (b - prev_b):>8.2f}"
    )
    prev_f, prev_b = f, b
    mx.clear_cache()

print("\n各部分 f+b 降序（×7 层外推）")
for label, df, db in sorted(res, key=lambda r: -(r[1] + r[2])):
    print(f"  {label:<30}{df + db:>7.2f}ms → {(df + db) * 7:>6.1f}ms/步")

# 纯 GEMM 上界对照：同形状的裸 batched matmul
print("\n同形状裸 batched matmul（无 autodiff、无掩码）")
A1 = mx.random.normal((NB, C, D)).astype(mx.float32)
B1 = mx.random.normal((NB, D, C)).astype(mx.float32)
A2 = mx.random.normal((NB, C, C)).astype(mx.float32)
mx.eval(A1, B1, A2)
t = timed(lambda: (A1 @ B1,))
fl = NB * 2 * C * D * C / 1e9
print(f"  (C,D)@(D,C) ×{NB}: {t:.3f}ms  {fl / t:.2f} TFLOPS  （{fl:.2f} GFLOP）")
t = timed(lambda: (A2 @ A2,))
fl = NB * 2 * C * C * C / 1e9
print(f"  (C,C)@(C,C) ×{NB}: {t:.3f}ms  {fl / t:.2f} TFLOPS  （{fl:.2f} GFLOP）")
t = timed(lambda: (A2 @ A1,))
fl = NB * 2 * C * C * D / 1e9
print(f"  (C,C)@(C,D) ×{NB}: {t:.3f}ms  {fl / t:.2f} TFLOPS  （{fl:.2f} GFLOP）")
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
