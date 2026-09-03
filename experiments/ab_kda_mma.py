"""kda_scan MMA vs 标量：同进程交替 A/B（抵消热漂移）。

背景：probe_chunk_parts 背靠背显示 MMA scan_zsc f+b 1.8~2.7×，但
bench_train_step 跨进程对比出现反向 6% 差——彼时 optimizer 段（不碰
KDA）也慢 8%，疑为热节流污染。本脚本在同一进程内 trace 两份 compile
图（模块级 _MMA_DISABLED 在 trace 时读取），逐对交替计时，报告成对
比值的中位数，慢漂移在成对差分下抵消。

用法: uv run python experiments/ab_kda_mma.py
环境: VIBY_BENCH_B / VIBY_BENCH_T / VIBY_BENCH_H（默认 6/2048/16，整步真实形状）
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK, _chunk_kda, _scan_prewarm
from model.kernels import kda_scan as ks
from model.kernels.kda_scan import kda_scan_metal

B = int(os.environ.get("VIBY_BENCH_B", 6))
T = int(os.environ.get("VIBY_BENCH_T", 2048))
H = int(os.environ.get("VIBY_BENCH_H", 16))
Dh = 96
C = KDA_CHUNK
NC = T // C

assert _scan_prewarm(NC, C, Dh, Dh), "kda_scan 预热失败"

mx.random.seed(0)
q = (mx.random.normal((B, H, T, Dh)) * 0.1).astype(mx.float32)
k = (mx.random.normal((B, H, T, Dh)) * 0.1).astype(mx.float32)
v = (mx.random.normal((B, H, T, Dh)) * 0.5).astype(mx.float32)
log_g = (-mx.random.uniform(0.01, 1.0, (B, H, T, Dh))).astype(mx.float32)
beta = mx.random.uniform(0.1, 0.9, (B, H, T)).astype(mx.float32)
args = (q, k, v, log_g, beta)
mx.eval(*args)
print(f"B={B} T={T} H={H} D={Dh} C={C} NC={NC}")


def _loss_a(a, b_, c_, d_, e_):
    return (_chunk_kda(a, b_, c_, d_, e_, None, zero_state_cot=True)[0] ** 2).sum()


def _loss_b(a, b_, c_, d_, e_):
    return (_chunk_kda(a, b_, c_, d_, e_, None, zero_state_cot=True)[0] ** 2).sum()


def _scan_args():
    from model.kernels.kda_inner import kda_inner
    from model.kernels.kda_prep import kda_prep

    r = lambda x: x.reshape(B, H, NC, C, Dh)
    qe, ke, ki, kd, egl = kda_prep(r(q), r(k), r(log_g))
    w, u, Aqk = kda_inner(qe, ke, ki, r(v), beta.reshape(B, H, NC, C))
    S0 = mx.zeros((B, H, Dh, Dh), dtype=mx.float32)
    mx.eval(qe, w, u, Aqk, kd, egl, S0)
    return (qe, w, u, Aqk, kd, egl, S0)


def _pairs(fa, fb, aa, ab, it=16, warm=3):
    """交替计时：返回 (中位 A, 中位 B, 成对比值 A/B 的中位数)。"""
    for _ in range(warm):
        mx.eval(fa(*aa))
        mx.eval(fb(*ab))
    ratios, tas, tbs = [], [], []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fa(*aa))
        ta = time.perf_counter() - t0
        t0 = time.perf_counter()
        mx.eval(fb(*ab))
        tb = time.perf_counter() - t0
        tas.append(ta * 1e3)
        tbs.append(tb * 1e3)
        ratios.append(ta / tb)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    return med(tas), med(tbs), med(ratios)


# ---- scan 单核（zsc，训练真实形态）：显式几何=标量 vs 默认=MMA ----
sa = _scan_args()
gs = tuple(range(7))
scal = mx.compile(
    mx.value_and_grad(
        lambda *a: (
            kda_scan_metal(*a, dv_split=2, nt_fwd=256, nt_bwd=256, zsc=True)[0] ** 2
        ).sum(),
        argnums=gs,
    )
)
mma = mx.compile(
    mx.value_and_grad(
        lambda *a: (kda_scan_metal(*a, zsc=True)[0] ** 2).sum(), argnums=gs
    )
)
assert not ks._MMA_DISABLED, "需在 VIBY_KDA_SCAN_MMA=1 下运行"
ta, tb, r = _pairs(scal, mma, sa, sa)
print(f"scan_zsc f+b   标量 {ta:8.2f}ms   MMA {tb:8.2f}ms   成对比 {r:.2f}×")

# ---- chunk 整段（zsc）：两份 compile 图分别在不同标志下 trace ----
ga = (0, 1, 2, 3, 4)
ca = mx.compile(mx.value_and_grad(_loss_a, argnums=ga))
cb = mx.compile(mx.value_and_grad(_loss_b, argnums=ga))
ks._MMA_DISABLED = True
mx.eval(ca(*args))  # trace：标量
ks._MMA_DISABLED = False
mx.eval(cb(*args))  # trace：MMA
ta, tb, r = _pairs(ca, cb, args, args)
print(f"chunk_zsc f+b  标量 {ta:8.2f}ms   MMA {tb:8.2f}ms   成对比 {r:.2f}×")
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
