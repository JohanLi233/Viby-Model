"""量化 KDA scan 的 Sall cotangent 浪费。

_chunk_kda 只消费 Sall[:, :, NC]（最后一片，3.5MB），但 _scan_dispatch
返回整个 Sall (B,H,NC+1,D,Dv) f32 = 230MB。反向时 autodiff 要为这个输出
构造完整 cotangent，其中除最后一片外全是零——226MB 的零分配 + 写 + 逐
chunk 读。

对照 scan 的反向在三种 cotangent 结构下的耗时：
  (a) 只对 o 求梯度（Sall 不进 loss）
  (b) o + Sall 最后一片（真实用法）
  (c) o + 整个 Sall（上界）

用法: uv run python experiments/probe_sall_cot.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK, _chunk_kda
from model.kernels.kda_scan import kda_scan_metal

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H, D = 8, 96
C = KDA_CHUNK
NC = T // C
BW = 400e9

sall_mb = B * H * (NC + 1) * D * D * 4 / 2**20
print(f"B={B} T={T} H={H} D={D} C={C} NC={NC}")
print(
    f"Sall f32 = {sall_mb:.1f}MB，其中被消费的最后一片 "
    f"{B * H * D * D * 4 / 2**20:.1f}MB "
    f"（{1 / (NC + 1) * 100:.1f}%）\n"
)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


qe = (mx.random.normal((B, H, NC, C, D)) * 0.1).astype(mx.float32)
w_ = (mx.random.normal((B, H, NC, C, D)) * 0.1).astype(mx.float32)
u_ = (mx.random.normal((B, H, NC, C, D)) * 0.3).astype(mx.float32)
Aqk = mx.tril(mx.random.normal((B, H, NC, C, C)) * 0.1)
kd = (mx.random.normal((B, H, NC, C, D)) * 0.1).astype(mx.float32)
egl = mx.exp(-mx.random.uniform(0, 1, (B, H, NC, D))).astype(mx.float32)
S0 = mx.zeros((B, H, D, D), dtype=mx.float32)
cot_o = mx.random.normal((B, H, NC, C, D))
mx.eval(qe, w_, u_, Aqk, kd, egl, S0, cot_o)
args = (qe, w_, u_, Aqk, kd, egl)


def L_o(*a):
    o, _ = kda_scan_metal(*a, S0)
    return (o * cot_o).sum()


def L_o_last(*a):
    o, Sall = kda_scan_metal(*a, S0)
    return (o * cot_o).sum() + (Sall[:, :, NC] ** 2).sum()


def L_o_all(*a):
    o, Sall = kda_scan_metal(*a, S0)
    return (o * cot_o).sum() + (Sall**2).sum()


idx = (0, 1, 2, 3, 4, 5)
print(f"{'scan 反向的 cotangent 结构':<34}{'fwd':>8}{'f+b':>9}{'bwd':>9}")
for label, fn in (
    ("(a) 只对 o", L_o),
    ("(b) o + Sall 末片（现状）", L_o_last),
    ("(c) o + 整个 Sall", L_o_all),
):
    f = timed(lambda _f=fn: _f(*args))
    fb = timed(lambda _f=fn: mx.value_and_grad(_f, argnums=idx)(*args))
    print(f"{label:<34}{f:>8.2f}{fb:>9.2f}{fb - f:>9.2f}")
    mx.clear_cache()

print(
    f"\n零 cotangent 分配+写+读的理论量 "
    f"{2 * (sall_mb - sall_mb / (NC + 1)):.0f}MB → "
    f"{2 * (sall_mb - sall_mb / (NC + 1)) * 2**20 / BW * 1e3:.2f}ms"
)

# ---- 完整 chunk 段对照 ----
print()
q = (mx.random.normal((B, H, T, D)) * 0.1).astype(mx.float32)
k = (mx.random.normal((B, H, T, D)) * 0.1).astype(mx.float32)
v = (mx.random.normal((B, H, T, D)) * 0.5).astype(mx.float32)
lg = mx.maximum(-mx.random.uniform(0.001, 2.0, (B, H, T, D)), -4.0)
bt = mx.random.uniform(0, 1, (B, H, T)).astype(mx.float32)
co = mx.random.normal((B, H, T, D))
mx.eval(q, k, v, lg, bt, co)


def ck_both(*a):
    o, S = _chunk_kda(*a)
    return (o * co).sum() + (S**2).sum()


def ck_o(*a):
    o, _ = _chunk_kda(*a)
    return (o * co).sum()


ca = (q, k, v, lg, bt)
for label, fn in (
    ("_chunk_kda: o + 末态 S（现状）", ck_both),
    ("_chunk_kda: 只对 o", ck_o),
):
    cf = mx.compile(fn)
    cg = mx.compile(mx.value_and_grad(fn, argnums=(0, 1, 2, 3, 4)))
    f = timed(lambda _f=cf: _f(*ca))
    fb = timed(lambda _f=cg: _f(*ca))
    print(f"{label:<34}{f:>8.2f}{fb:>9.2f}{fb - f:>9.2f}")
    mx.clear_cache()

print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
