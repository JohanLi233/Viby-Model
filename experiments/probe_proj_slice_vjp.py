"""KDA 融合投影：六路切片 vs mx.split 的反向代价。

prof_kda_stages（已预热）：`fused = x @ w_in.T` 加六路切片 f+b 20.90ms/层
= 146ms/步，是 KDA 最大单项。纯 GEMM 应为 179.7 GFLOP，按实测 dense bf16
峰值 14 TF/s 只要 ~12.8ms ⇒ 8ms/层是非 GEMM 开销。

假设：`fused[..., a:b]` 的 VJP 是「scatter 进一个全宽零张量」，六路各产出
一份 (B*T, W) 再相加，反向白写约 6 份、白读白写约 5 份全宽张量。
`mx.split` 是单个 primitive，VJP 就是一次 concatenate，只写一份。

用法: uv run python experiments/probe_proj_slice_vjp.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B, T, DIM = 12, 1024, 768
H, HD = 8, 96
proj = H * HD
W = 4 * proj + HD + H
CUTS = [proj, 2 * proj, 3 * proj, 3 * proj + HD, 4 * proj + HD]

mx.random.seed(0)
x = (mx.random.normal((B, T, DIM)) * 0.5).astype(mx.bfloat16)
w = (mx.random.normal((W, DIM)) * 0.03).astype(mx.bfloat16)
mx.eval(x, w)


def timed(fn, it=10, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def consume(parts):
    """让六个分片各自参与一个非平凡的逐元素运算，模拟真实下游。"""
    q, k, v, fa, g, bl = parts
    return (
        (q * 1.5).sum()
        + (k * 1.25).sum()
        + (v * 0.75).sum()
        + mx.tanh(fa.astype(mx.float32)).sum()
        + mx.sigmoid(g.astype(mx.float32)).sum()
        + (bl * 2.0).sum()
    )


def f_slice(x_, w_):
    f = x_ @ w_.T
    return consume(
        (
            f[..., :proj],
            f[..., proj : 2 * proj],
            f[..., 2 * proj : 3 * proj],
            f[..., 3 * proj : 3 * proj + HD],
            f[..., 3 * proj + HD : 4 * proj + HD],
            f[..., -H:],
        )
    )


def f_split(x_, w_):
    return consume(mx.split(x_ @ w_.T, CUTS, axis=-1))


def f_gemm_only(x_, w_):
    return (x_ @ w_.T).astype(mx.float32).sum()


print(f"B={B} T={T} D={DIM} H={H} head_dim={HD}  fused 宽度 W={W}")
flops = 2 * B * T * DIM * W
print(f"GEMM {flops / 1e9:.1f} GFLOP  fused {B * T * W * 2 / 2**20:.0f}MB\n")
print(f"{'变体':<22}{'fwd':>9}{'f+b':>9}{'bwd':>9}{'TF/s_fb':>9}")
ref = None
for name, fn in (
    ("仅 GEMM", f_gemm_only),
    ("六路切片", f_slice),
    ("mx.split", f_split),
):
    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=(0, 1)))
    f = timed(lambda: cf(x, w))
    fb = timed(lambda: cvg(x, w))
    print(
        f"{name:<22}{f:>9.2f}{fb:>9.2f}{fb - f:>9.2f}{3 * flops / (fb / 1e3) / 1e12:>9.1f}"
    )
    if name == "六路切片":
        ref = mx.value_and_grad(fn, argnums=(0, 1))(x, w)
    elif name == "mx.split":
        got = mx.value_and_grad(fn, argnums=(0, 1))(x, w)
        mx.eval(ref, got)
        for tag, a, b in (
            ("loss", got[0], ref[0]),
            ("dx", got[1][0], ref[1][0]),
            ("dw", got[1][1], ref[1][1]),
        ):
            d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
            s = b.astype(mx.float32).abs().max().item() + 1e-12
            print(f"    数值对齐 {tag}: rel={d / s:.2e}")
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
