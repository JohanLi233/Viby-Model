"""compile 下 KDA 层内部 fwd/bwd 分段剖析（真实训练形状）。

训练侧已开 mx.compile，elementwise 链会被融合，所以 eager 口径的分段
计时会高估「分解段」。本脚本在 mx.compile 下分别测：
  整层 / chunk 全段 / chunk 分解段 / scan 段
并给出各段的访存下界，判断剩余空间。

用法: uv run python experiments/prof_kda_compile.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.kda import KDA_CHUNK, KDAAttention, _chunk_kda, _kda_scan
from model.kernels.kda_scan import kda_scan_metal

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = int(os.environ.get("VIBY_BENCH_D", 768))
H = int(os.environ.get("VIBY_BENCH_H", 8))
BW = 400e9


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


cfg = VibyConfig(
    hidden_size=D,
    num_hidden_layers=8,
    num_attention_heads=H,
    vocab_size=6400,
    max_position_embeddings=T,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=2,
    moe_intermediate_size=384,
)
HD = cfg.head_dim
C = KDA_CHUNK
NC = T // C
print(f"B={B} T={T} D={D} H={H} head_dim={HD} C={C} NC={NC}")
el5 = B * H * NC * C * HD  # (B,H,NC,C,D) 的元素数
print(
    f"(B,H,NC,C,Dh) f32 = {el5 * 4 / 2**20:.1f}MB  Sall f32 = "
    f"{B * H * (NC + 1) * HD * HD * 4 / 2**20:.1f}MB"
)

kda = KDAAttention(cfg, layer_idx=0)
kda.update(
    tree_map(
        lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a,
        kda.parameters(),
    )
)
mx.eval(kda.parameters())
kda.train()

x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
cot = mx.random.normal((B, T, D))
mx.eval(x, seg, cot)

# ---- 整层 ----
p = kda.trainable_parameters()


def layer_loss(x_, p_):
    kda.update(p_)
    return (kda(x_, segment_ids=seg)[0].astype(mx.float32) * cot).sum()


layer_vg = mx.value_and_grad(layer_loss, argnums=(0, 1))
c_layer_f = mx.compile(layer_loss)
c_layer_vg = mx.compile(layer_vg)
print(
    f"\n{'段':<24}{'eager fwd':>11}{'compile fwd':>12}{'eager f+b':>11}{'compile f+b':>12}"
)
lf_e = timed(lambda: layer_loss(x, p))
lf_c = timed(lambda: c_layer_f(x, p))
lb_e = timed(lambda: layer_vg(x, p))
lb_c = timed(lambda: c_layer_vg(x, p))
print(f"{'KDA 整层':<24}{lf_e:>11.2f}{lf_c:>12.2f}{lb_e:>11.2f}{lb_c:>12.2f}")

# ---- chunk 全段 ----
q = (mx.random.normal((B, H, T, HD)) * 0.1).astype(mx.float32)
k = (mx.random.normal((B, H, T, HD)) * 0.1).astype(mx.float32)
v = (mx.random.normal((B, H, T, HD)) * 0.5).astype(mx.float32)
log_g = mx.maximum(-mx.random.uniform(0.001, 2.0, (B, H, T, HD)), -4.0).astype(
    mx.float32
)
beta = mx.random.uniform(0, 1, (B, H, T)).astype(mx.float32)
cot_o = mx.random.normal((B, H, T, HD))
mx.eval(q, k, v, log_g, beta, cot_o)


def chunk_loss(q_, k_, v_, g_, b_):
    o, S = _chunk_kda(q_, k_, v_, g_, b_)
    return (o * cot_o).sum() + (S**2).sum()


chunk_vg = mx.value_and_grad(chunk_loss, argnums=(0, 1, 2, 3, 4))
c_chunk_f = mx.compile(chunk_loss)
c_chunk_vg = mx.compile(chunk_vg)
cf_e = timed(lambda: chunk_loss(q, k, v, log_g, beta))
cf_c = timed(lambda: c_chunk_f(q, k, v, log_g, beta))
cb_e = timed(lambda: chunk_vg(q, k, v, log_g, beta))
cb_c = timed(lambda: c_chunk_vg(q, k, v, log_g, beta))
print(f"{'chunk 全段':<24}{cf_e:>11.2f}{cf_c:>12.2f}{cb_e:>11.2f}{cb_c:>12.2f}")


# ---- 只做分解段（不含 scan）----
def prep(q_, k_, v_, g_, b_):
    def ch(a):
        return a.reshape(B, H, NC, C, *a.shape[3:])

    qc, kc, vc, lgc, bc = ch(q_), ch(k_), ch(v_), ch(g_), ch(b_)
    gc = mx.cumsum(lgc, axis=-2)
    eg = mx.exp(gc)
    qe = qc * eg
    ke = kc * eg
    ki = kc * mx.exp(-gc)
    sl = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    Lm = (bc[..., None] * ke) @ mx.swapaxes(ki, -1, -2)
    Lm = mx.where(sl, Lm, mx.zeros_like(Lm))
    P = -Lm
    X = mx.eye(C, dtype=mx.float32) + P
    pp = P
    for _ in range(int(math.log2(C)) - 1):
        pp = pp @ pp
        X = X + pp @ X
    Afb = X * bc[..., None, :]
    w_ = Afb @ ke
    u_ = Afb @ vc
    lower = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    Aqk = qe @ mx.swapaxes(ki, -1, -2)
    Aqk = mx.where(lower, Aqk, mx.zeros_like(Aqk))
    gl = gc[:, :, :, -1, :]
    kd = kc * mx.exp(gl[:, :, :, None, :] - gc)
    return qe, w_, u_, Aqk, kd, mx.exp(gl)


def prep_loss(q_, k_, v_, g_, b_):
    outs = prep(q_, k_, v_, g_, b_)
    return sum((o.astype(mx.float32) ** 2).sum() for o in outs)


prep_vg = mx.value_and_grad(prep_loss, argnums=(0, 1, 2, 3, 4))
c_prep_f = mx.compile(prep_loss)
c_prep_vg = mx.compile(prep_vg)
pf_e = timed(lambda: prep_loss(q, k, v, log_g, beta))
pf_c = timed(lambda: c_prep_f(q, k, v, log_g, beta))
pb_e = timed(lambda: prep_vg(q, k, v, log_g, beta))
pb_c = timed(lambda: c_prep_vg(q, k, v, log_g, beta))
print(f"{'  └分解段':<24}{pf_e:>11.2f}{pf_c:>12.2f}{pb_e:>11.2f}{pb_c:>12.2f}")

# ---- 只做 scan ----
qe, w_, u_, Aqk, kd, egl = prep(q, k, v, log_g, beta)
S0 = mx.zeros((B, H, HD, HD), dtype=mx.float32)
mx.eval(qe, w_, u_, Aqk, kd, egl, S0)
cot_s = mx.random.normal((B, H, NC, C, HD))
cot_S = mx.random.normal((B, H, NC + 1, HD, HD))
mx.eval(cot_s, cot_S)


def scan_loss_metal(a, b, c_, d, e, f):
    o, Sall = kda_scan_metal(a, b, c_, d, e, f, S0)
    return (o * cot_s).sum() + (Sall * cot_S).sum()


def scan_loss_eager(a, b, c_, d, e, f):
    o, Sall = _kda_scan(a, b, c_, d, e, f, S0)
    return (o * cot_s).sum() + (Sall * cot_S).sum()


sm_vg = mx.value_and_grad(scan_loss_metal, argnums=(0, 1, 2, 3, 4, 5))
se_vg = mx.value_and_grad(scan_loss_eager, argnums=(0, 1, 2, 3, 4, 5))
args = (qe, w_, u_, Aqk, kd, egl)
print(
    f"{'  └scan (Metal)':<24}{timed(lambda: scan_loss_metal(*args)):>11.2f}"
    f"{'-':>12}{timed(lambda: sm_vg(*args)):>11.2f}{'-':>12}"
)
print(
    f"{'  └scan (eager 参考)':<24}{timed(lambda: scan_loss_eager(*args)):>11.2f}"
    f"{'-':>12}{timed(lambda: se_vg(*args)):>11.2f}{'-':>12}"
)

# 分解段访存下界：读 q/k/v/log_g/beta，写 qe/w/u/kd(4×el5) + Aqk + egl
rd = 4 * el5 * 4 + B * H * T * 4
wr = 4 * el5 * 4 + B * H * NC * C * C * 4 + B * H * NC * HD * 4
print(
    f"\n分解段访存下界 @{BW / 1e9:.0f}GB/s: 读{rd / 2**20:.0f}MB + 写{wr / 2**20:.0f}MB "
    f"= {(rd + wr) / BW * 1e3:.2f}ms"
)
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
