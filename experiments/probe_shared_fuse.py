"""共享专家融合：2×FeedForward(I) vs 单个 2I 宽等价体。

SiTU-GLU 逐元素 ⇒ 两个共享专家之和可以写成一个双宽专家：
  Σ_i D_i·situ(G_i x, U_i x) = [D_1 D_2] · situ([G_1;G_2]x, [U_1;U_2]x)
FLOPs 完全相同，但 GEMM 形状翻倍、kernel 发射减半。

probe_moe_parts（已预热）：共享专家 ×2 = 11.61ms f+b @ 11.2 TF/s，
稠密 bf16 峰值约 14 TF/s。本脚本量化融合能拿回多少，并核对数值。

用法: uv run python experiments/probe_shared_fuse.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.acts import situ_glu_gu
from model.kernels.situ import prewarm, prewarm_packed

B, T, D, I, N = 12, 1024, 768, 384, 2  # noqa: E741
M = B * T
assert prewarm(mx.bfloat16) and prewarm_packed(mx.bfloat16, [I, N * I])

mx.random.seed(0)
x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
Gs = [(mx.random.normal((I, D)) * D**-0.5).astype(mx.bfloat16) for _ in range(N)]
Us = [(mx.random.normal((I, D)) * D**-0.5).astype(mx.bfloat16) for _ in range(N)]
Ds = [(mx.random.normal((D, I)) * I**-0.5).astype(mx.bfloat16) for _ in range(N)]
mx.eval(x, Gs, Us, Ds)
params = (Gs, Us, Ds)


def f_per_expert(x_, p_):
    gs, us, ds = p_
    out = mx.zeros_like(x_)
    for g, u, dw in zip(gs, us, ds):
        h = x_ @ mx.concatenate([g, u], axis=0).T
        out = out + situ_glu_gu(h) @ dw.T
    return out


def f_fused(x_, p_):
    gs, us, ds = p_
    w = mx.concatenate(list(gs) + list(us), axis=0)  # (2NI, D)
    dw = mx.concatenate(list(ds), axis=1)  # (D, NI)
    return situ_glu_gu(x_ @ w.T) @ dw.T


def timed(fn, it=10, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


flops = N * 2 * M * (2 * D * I + I * D)
print(f"B={B} T={T} D={D} I={I} N={N}  M={M}  {flops / 1e9:.1f} GFLOP\n")
print(f"{'变体':<18}{'fwd':>9}{'f+b':>9}{'bwd':>9}{'TF/s_fb':>9}")
outs = {}
for name, fn in (("逐专家 ×2", f_per_expert), ("融合 2I 宽", f_fused)):
    loss = lambda a, p_: fn(a, p_).astype(mx.float32).sum()  # noqa: E731
    cf = mx.compile(loss)
    cvg = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
    f = timed(lambda: cf(x, params))
    fb = timed(lambda: cvg(x, params))
    print(
        f"{name:<18}{f:>9.2f}{fb:>9.2f}{fb - f:>9.2f}{3 * flops / (fb / 1e3) / 1e12:>9.1f}"
    )
    outs[name] = (fn(x, params), mx.grad(loss, argnums=0)(x, params))
    mx.clear_cache()

a, b = outs["融合 2I 宽"], outs["逐专家 ×2"]
mx.eval(a, b)
for tag, u, v in (("out", a[0], b[0]), ("dx", a[1], b[1])):
    u32, v32 = u.astype(mx.float32), v.astype(mx.float32)
    rel = (u32 - v32).abs().max().item() / (v32.abs().max().item() + 1e-12)
    print(f"  数值对齐 {tag}: rel={rel:.2e}")
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
