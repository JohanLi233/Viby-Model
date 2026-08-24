"""KDA 门控段（rms_unit / f_b_proj / log_g / beta）逐项计时。

prof_kda_stages：该段 f+b 7.21ms/层 = 50.5ms/步，其中 fwd 只有 0.75ms、
bwd 6.47ms——反向是前向的 8.6 倍，远超「反向约 2× 前向」的常态，说明
不是算术量而是中间量物化/未融合。本脚本把它拆成 q/k 的 rms_unit、
f_b_proj 小 GEMM、f32 的 log_g 链、beta，各自单独编译计时。

用法: uv run python experiments/probe_kda_gates.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_G_MIN
from model.norms import _rms_unit

B, T, H, Dh = 12, 1024, 8, 96
proj = H * Dh
N = B * T * H * Dh

mx.random.seed(0)
q_in = (mx.random.normal((B, T, proj)) * 0.5).astype(mx.bfloat16)
k_in = (mx.random.normal((B, T, proj)) * 0.5).astype(mx.bfloat16)
fa = (mx.random.normal((B, T, Dh)) * 0.5).astype(mx.bfloat16)
bl = (mx.random.normal((B, T, H)) * 0.5).astype(mx.bfloat16)
w_fb = (mx.random.normal((proj, Dh)) * 0.1).astype(mx.bfloat16)
dt_bias = mx.zeros((H, Dh), dtype=mx.float32)
A_log = mx.zeros((H,), dtype=mx.float32)
scale = Dh**-0.5
mx.eval(q_in, k_in, fa, bl, w_fb, dt_bias, A_log)


def timed(fn, it=10, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def bench(label, fn, args, argnums):
    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=argnums))
    f = timed(lambda: cf(*args))
    fb = timed(lambda: cvg(*args))
    print(f"{label:<30}{f:>8.2f}{fb:>8.2f}{fb - f:>8.2f}")
    mx.clear_cache()


def logg_chain(a_, dt_, al_):
    z = a_.reshape(B, T, H, Dh).astype(mx.float32) + dt_
    return KDA_G_MIN * mx.sigmoid(mx.exp(al_)[None, None, :, None] * z)


mb32 = N * 4 / 2**20
print(
    f"B={B} T={T} H={H} head_dim={Dh}  (B,T,H,D) 元素 {N / 1e6:.2f}M"
    f"  f32 {mb32:.0f}MB / bf16 {mb32 / 2:.0f}MB\n"
)
print(f"{'段':<30}{'fwd':>8}{'f+b':>8}{'bwd':>8}")

bench(
    "rms_unit(q) 单个",
    lambda a: (scale**2 * _rms_unit(a.reshape(B, T, H, Dh))).sum(),
    (q_in,),
    (0,),
)
bench(
    "rms_unit(q)+rms_unit(k)",
    lambda a, b: (
        (scale**2 * _rms_unit(a.reshape(B, T, H, Dh))).sum()
        + (scale * _rms_unit(b.reshape(B, T, H, Dh))).sum()
    ),
    (q_in, k_in),
    (0, 1),
)
bench(
    "f_b_proj GEMM (B,T,96)@(96,768)",
    lambda a, w: (a @ w.T).astype(mx.float32).sum(),
    (fa, w_fb),
    (0, 1),
)

a_full = (fa @ w_fb.T).astype(mx.bfloat16)
mx.eval(a_full)
bench(
    "log_g 链 (f32 sigmoid)",
    lambda a, dt_, al_: logg_chain(a, dt_, al_).sum(),
    (a_full, dt_bias, A_log),
    (0, 1, 2),
)
bench(
    "log_g 链（输出转 bf16）",
    lambda a, dt_, al_: logg_chain(a, dt_, al_)
    .astype(mx.bfloat16)
    .astype(mx.float32)
    .sum(),
    (a_full, dt_bias, A_log),
    (0, 1, 2),
)
bench(
    "beta = sigmoid(bl.f32)",
    lambda b: mx.sigmoid(b.astype(mx.float32)).sum(),
    (bl,),
    (0,),
)
bench(
    "log_g 链 + transpose",
    lambda a, dt_, al_: logg_chain(a, dt_, al_).transpose(0, 2, 1, 3).sum(),
    (a_full, dt_bias, A_log),
    (0, 1, 2),
)


print("\nlog_g 链按 argnums 拆分（0=a 大张量, 1=dt_bias (H,D), 2=A_log (H,)）")
for tag, an in (
    ("仅 a", (0,)),
    ("a+dt_bias", (0, 1)),
    ("a+A_log", (0, 2)),
    ("仅 dt_bias", (1,)),
    ("仅 A_log", (2,)),
    ("全部", (0, 1, 2)),
):
    bench(
        f"  grad {tag}",
        lambda a, dt_, al_: logg_chain(a, dt_, al_).sum(),
        (a_full, dt_bias, A_log),
        an,
    )

print("\n改写候选：exp(A_log) 先 broadcast 到 (H,D) 再参与广播乘")


def logg_bcast(a_, dt_, al_):
    # (H,) 直接广播到 (B,T,H,D)，其 VJP 要沿 (0,1,3) 归约、保留中间轴；
    # 先显式扩到 (H,D)，VJP 变成「沿 (0,1) 归约」+ 一次 768 元素的小归约。
    e = mx.broadcast_to(mx.exp(al_)[:, None], (H, Dh))
    z = a_.reshape(B, T, H, Dh).astype(mx.float32) + dt_
    return KDA_G_MIN * mx.sigmoid(e * z)


bench(
    "  broadcast_to 版 全部 grad",
    lambda a, dt_, al_: logg_bcast(a, dt_, al_).sum(),
    (a_full, dt_bias, A_log),
    (0, 1, 2),
)
r0 = mx.value_and_grad(
    lambda a, dt_, al_: logg_chain(a, dt_, al_).sum(), argnums=(0, 1, 2)
)(a_full, dt_bias, A_log)
r1 = mx.value_and_grad(
    lambda a, dt_, al_: logg_bcast(a, dt_, al_).sum(), argnums=(0, 1, 2)
)(a_full, dt_bias, A_log)
mx.eval(r0, r1)
for tag, x0, x1 in (
    ("loss", r0[0], r1[0]),
    ("da", r0[1][0], r1[1][0]),
    ("d_dt", r0[1][1], r1[1][1]),
    ("d_A", r0[1][2], r1[1][2]),
):
    d = (x0.astype(mx.float32) - x1.astype(mx.float32)).abs().max().item()
    s = x0.astype(mx.float32).abs().max().item() + 1e-12
    print(f"    broadcast_to 版 {tag}: rel={d / s:.2e}")


def whole(q_, k_, a_, b_, dt_, al_):
    q = scale**2 * _rms_unit(q_.reshape(B, T, H, Dh))
    k = scale * _rms_unit(k_.reshape(B, T, H, Dh))
    lg = logg_chain(a_, dt_, al_)
    be = mx.sigmoid(b_.astype(mx.float32))
    return q.astype(mx.float32).sum() + k.astype(mx.float32).sum() + lg.sum() + be.sum()


bench("整段合计", whole, (q_in, k_in, a_full, bl, dt_bias, A_log), (0, 1, 2, 3, 4, 5))
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
