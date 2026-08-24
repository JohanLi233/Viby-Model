"""定位 KDA「+3× causal_conv」段的 12ms 缺口。

prof_kda_stages 里该段 f+b 16.85ms，而 bench_conv 单 conv 只有 1.57ms
（×3 = 4.7ms）。差额不在 conv kernel 本身，本脚本按变体二分：切片物化、
切片来源（融合 GEMM vs 独立 GEMM）、单核宽 conv。

用法: uv run python experiments/probe_conv_stage.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels.conv import causal_conv

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768
H, Dh = 8, 96
P = H * Dh  # 768
OUT = 3 * P + Dh + P + H  # 融合投影输出宽度 3176

mx.random.seed(0)
x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
w_in = (mx.random.normal((OUT, D)) * 0.02).astype(mx.bfloat16)
wq = (mx.random.normal((4, P)) * 0.3).astype(mx.bfloat16)
wk = (mx.random.normal((4, P)) * 0.3).astype(mx.bfloat16)
wv = (mx.random.normal((4, P)) * 0.3).astype(mx.bfloat16)
w3 = mx.concatenate([wq, wk, wv], axis=1)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
mx.eval(x, w_in, wq, wk, wv, w3, seg)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def sq(*outs):
    return sum((o.astype(mx.float32) ** 2).sum() for o in outs)


def split6(f):
    return (
        f[..., :P],
        f[..., P : 2 * P],
        f[..., 2 * P : 3 * P],
        f[..., 3 * P : 3 * P + Dh],
        f[..., 3 * P + Dh : 4 * P + Dh],
        f[..., -H:],
    )


# ---- 变体 ----
def v_proj_only(x_, p_):
    return sq(*split6(x_ @ p_["w_in"].T))


def v_slice_contig(x_, p_):
    q, k, v, fa, g, bl = split6(x_ @ p_["w_in"].T)
    return sq(mx.contiguous(q), mx.contiguous(k), mx.contiguous(v), fa, g, bl)


def v_conv_cur(x_, p_):
    q, k, v, fa, g, bl = split6(x_ @ p_["w_in"].T)
    return sq(
        causal_conv(q, p_["wq"], seg=seg),
        causal_conv(k, p_["wk"], seg=seg),
        causal_conv(v, p_["wv"], seg=seg),
        fa,
        g,
        bl,
    )


def v_conv_contig(x_, p_):
    q, k, v, fa, g, bl = split6(x_ @ p_["w_in"].T)
    return sq(
        causal_conv(mx.contiguous(q), p_["wq"], seg=seg),
        causal_conv(mx.contiguous(k), p_["wk"], seg=seg),
        causal_conv(mx.contiguous(v), p_["wv"], seg=seg),
        fa,
        g,
        bl,
    )


def v_conv_wide(x_, p_):
    """一次宽 conv：fused[..., :3P] 一个 2304 通道 depthwise。"""
    f = x_ @ p_["w_in"].T
    qkv = causal_conv(f[..., : 3 * P], p_["w3"], seg=seg)
    fa = f[..., 3 * P : 3 * P + Dh]
    g = f[..., 3 * P + Dh : 4 * P + Dh]
    bl = f[..., -H:]
    return sq(qkv[..., :P], qkv[..., P : 2 * P], qkv[..., 2 * P :], fa, g, bl)


def v_conv_sep_gemm(x_, p_):
    """独立 GEMM 出 q/k/v（无切片），其余仍融合。"""
    q = x_ @ p_["w_in"][:P].T
    k = x_ @ p_["w_in"][P : 2 * P].T
    v = x_ @ p_["w_in"][2 * P : 3 * P].T
    rest = x_ @ p_["w_in"][3 * P :].T
    return sq(
        causal_conv(q, p_["wq"], seg=seg),
        causal_conv(k, p_["wk"], seg=seg),
        causal_conv(v, p_["wv"], seg=seg),
        rest[..., :Dh],
        rest[..., Dh : Dh + P],
        rest[..., -H:],
    )


def v_conv_only(x_, p_):
    """conv 直接吃连续输入（无 GEMM），×3，作为纯 kernel 下界。"""
    return sq(
        causal_conv(x_, p_["wq"], seg=seg),
        causal_conv(x_, p_["wk"], seg=seg),
        causal_conv(x_, p_["wv"], seg=seg),
    )


p0 = {"w_in": w_in, "wq": wq, "wk": wk, "wv": wv, "w3": w3}
cx = (mx.random.normal((B, T, P)) * 0.5).astype(mx.bfloat16)
mx.eval(cx)

VARIANTS = [
    ("投影 only", v_proj_only),
    ("投影+切片物化", v_slice_contig),
    ("投影+3conv(现状)", v_conv_cur),
    ("投影+3conv(先物化)", v_conv_contig),
    ("投影+1宽conv(2304)", v_conv_wide),
    ("独立GEMM+3conv", v_conv_sep_gemm),
    ("纯3conv(连续输入)", v_conv_only),
]

print(f"B={B} T={T} D={D} P={P} 融合输出宽 {OUT}\n")
print(f"{'变体':<24}{'fwd':>9}{'f+b':>9}{'bwd':>9}")
base_f = base_fb = None
for label, fn in VARIANTS:
    xin = cx if fn is v_conv_only else x
    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=(0, 1)))
    f = timed(lambda: cf(xin, p0))
    fb = timed(lambda: cvg(xin, p0))
    print(f"{label:<24}{f:>9.2f}{fb:>9.2f}{fb - f:>9.2f}")
    if base_f is None:
        base_f, base_fb = f, fb
    else:
        print(f"{'  Δ vs 投影 only':<24}{f - base_f:>9.2f}{fb - base_fb:>9.2f}")
    mx.clear_cache()
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
