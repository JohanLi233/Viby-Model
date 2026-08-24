"""_chunk_kda 三个融合 kernel（prep / inner / scan）的 f+b 与访存下界。

prof_kda_stages（已预热）显示 _chunk_kda 段 f+b 18.42ms/层 → 128.9ms/步，
是 KDA 里第二大项。本脚本逐 kernel 计时并给出 compulsory 访存下界，判断
哪一段还有空间。scan 的 cot_Sall 在训练里恒为全零（最终状态在无 cache
时不被消费），单独列出其流量占比。

用法: uv run python experiments/probe_chunk_parts.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK, _chunk_kda, _scan_prewarm
from model.kernels import kda_inner as inner_mod
from model.kernels import kda_prep as prep_mod
from model.kernels.kda_inner import kda_inner
from model.kernels.kda_prep import kda_prep
from model.kernels.kda_scan import kda_scan_metal

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H = 8
Dh = 96
C = KDA_CHUNK
NC = T // C
BW = 400e9

assert prep_mod.prewarm(C, Dh, NC), "kda_prep 预热失败"
assert inner_mod.prewarm(C, Dh), "kda_inner 预热失败"
assert _scan_prewarm(NC, C, Dh, Dh), "kda_scan 预热失败"

mx.random.seed(0)
q = (mx.random.normal((B, H, NC, C, Dh)) * 0.1).astype(mx.float32)
k = (mx.random.normal((B, H, NC, C, Dh)) * 0.1).astype(mx.float32)
v = (mx.random.normal((B, H, NC, C, Dh)) * 0.5).astype(mx.float32)
log_g = (-mx.random.uniform(0.01, 1.0, (B, H, NC, C, Dh))).astype(mx.float32)
beta = mx.random.uniform(0.1, 0.9, (B, H, NC, C)).astype(mx.float32)
mx.eval(q, k, v, log_g, beta)

qe, ke, ki, kd, egl = kda_prep(q, k, log_g)
w, u, Aqk = kda_inner(qe, ke, ki, v, beta)
S0 = mx.zeros((B, H, Dh, Dh), dtype=mx.float32)
mx.eval(qe, ke, ki, kd, egl, w, u, Aqk, S0)

EL = B * H * NC * C * Dh  # 一份 (B,H,NC,C,D) f32 的元素数
MB = EL * 4 / 2**20
SALL_MB = B * H * (NC + 1) * Dh * Dh * 4 / 2**20
print(f"B={B} T={T} H={H} D={Dh} C={C} NC={NC}")
print(f"单份 (B,H,NC,C,D) f32 = {MB:.1f}MB   Sall = {SALL_MB:.1f}MB\n")


def timed(fn, it=8, w_=3):
    for _ in range(w_):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def bench(label, fn, argnums, args, floor_mb):
    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=argnums))
    f = timed(lambda: cf(*args))
    fb = timed(lambda: cvg(*args))
    floor = floor_mb * 2**20 / BW * 1e3
    print(
        f"{label:<14}{f:>8.2f}{fb:>8.2f}{fb - f:>8.2f}{floor:>9.2f}{fb / floor:>8.1f}×"
    )
    mx.clear_cache()
    return f, fb


print(f"{'kernel':<14}{'fwd':>8}{'f+b':>8}{'bwd':>8}{'访存下界':>9}{'倍数':>8}")

# prep: 读 q/k/log_g，写 qe/ke/ki/kd/egl；bwd 再读一遍 + 5 个 cot，写 3 个
bench(
    "prep",
    lambda a, b_, c_: sum((o**2).sum() for o in kda_prep(a, b_, c_)),
    (0, 1, 2),
    (q, k, log_g),
    MB * (3 + 4) + MB * (3 + 4 + 3),
)
# inner: 读 qe/ke/ki/v，写 w/u + 3 个 C×C；bwd 读回 + 3 个 cot，写 4 个
bench(
    "inner",
    lambda a, b_, c_, d_, e_: sum((o**2).sum() for o in kda_inner(a, b_, c_, d_, e_)),
    (0, 1, 2, 3, 4),
    (qe, ke, ki, v, beta),
    MB * (4 + 2) + MB * (4 + 2 + 4),
)
# scan: 读 qe/w/u/Aqk/kd/egl，写 o + Sall；bwd 读回 + Sall + cot_o + cot_Sall
bench(
    "scan",
    lambda *a: sum((o**2).sum() for o in kda_scan_metal(*a)),
    tuple(range(7)),
    (qe, w, u, Aqk, kd, egl, S0),
    MB * (4 + 1) + SALL_MB + MB * (4 + 1 + 1 + 5) + 2 * SALL_MB,
)
# 整段
bench(
    "_chunk_kda",
    lambda a, b_, c_, d_, e_: sum(
        (o**2).sum() for o in _chunk_kda(a, b_, c_, d_, e_, None)
    ),
    (0, 1, 2, 3, 4),
    (
        q.reshape(B, H, T, Dh),
        k.reshape(B, H, T, Dh),
        v.reshape(B, H, T, Dh),
        log_g.reshape(B, H, T, Dh),
        beta.reshape(B, H, T),
    ),
    0.001,
)

print(f"\nscan 里 cot_Sall（训练恒零）单独占 {SALL_MB:.0f}MB 分配+读")
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
