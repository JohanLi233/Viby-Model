"""KDA 六路融合投影的分组变体对照。

prof_kda_stages（已预热口径）显示该段 fwd 5.13ms / bwd 15.99ms，而两个
GEMM 的裸下界是 4.23 + 4.63 = 8.86ms——反向有 ~7ms/层不在 GEMM 上。
候选来源：权重 concatenate 的 VJP（切成 6 份）、cotangent 的 6 路
concatenate、以及 (M,3176) 中间张量的切片物化。

变体只改 GEMM 分组，数学等价（每个输出列仍是同一个点积）。

用法: uv run python experiments/probe_proj_split.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768
H, Dh = 8, 96
P = H * Dh
WIDTHS = (P, P, P, Dh, P, H)  # q k v f_a g b
OUT = sum(WIDTHS)

mx.random.seed(0)
x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
ws = {
    f"w{i}": (mx.random.normal((wd, D)) * 0.02).astype(mx.bfloat16)
    for i, wd in enumerate(WIDTHS)
}
mx.eval(x, ws)
KEYS = [f"w{i}" for i in range(len(WIDTHS))]


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def sq(outs):
    return sum((o.astype(mx.float32) ** 2).sum() for o in outs)


def grouped(x_, p_, groups):
    """groups 是 KEYS 的划分；每组一次 GEMM 后切回各自输出。"""
    outs = []
    for grp in groups:
        if len(grp) == 1:
            outs.append(x_ @ p_[grp[0]].T)
            continue
        w = mx.concatenate([p_[k] for k in grp], axis=0)
        f = x_ @ w.T
        o = 0
        for k in grp:
            n = p_[k].shape[0]
            outs.append(f[..., o : o + n])
            o += n
    return outs


GROUPINGS = [
    ("6 路全融合(现状)", [KEYS]),
    ("6 路各自独立", [[k] for k in KEYS]),
    ("qkv 融合 + 其余独立", [KEYS[:3], [KEYS[3]], [KEYS[4]], [KEYS[5]]]),
    ("qkvg 融合 + fa/b 独立", [[*KEYS[:3], KEYS[4]], [KEYS[3]], [KEYS[5]]]),
    ("qkv + fa/g/b 两组", [KEYS[:3], KEYS[3:]]),
    ("qkvg + fa/b 两组", [[*KEYS[:3], KEYS[4]], [KEYS[3], KEYS[5]]]),
]

print(f"B={B} T={T} D={D} 输出宽 {OUT}（{WIDTHS}）")
print(f"裸 GEMM 参考: fwd {12288 * OUT * D * 2 / 14.18e12 * 1e3:.2f}ms\n")
print(f"{'分组':<26}{'fwd':>9}{'f+b':>9}{'bwd':>9}{'Δf+b':>9}")
base = None
for label, groups in GROUPINGS:

    def fn(x_, p_, _g=groups):
        return sq(grouped(x_, p_, _g))

    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=(0, 1)))
    f = timed(lambda: cf(x, ws))
    fb = timed(lambda: cvg(x, ws))
    if base is None:
        base = fb
    print(f"{label:<26}{f:>9.2f}{fb:>9.2f}{fb - f:>9.2f}{fb - base:>9.2f}")
    mx.clear_cache()

# 数值对照：分组只改 GEMM 形状，输出应逐列一致
ref = mx.concatenate(grouped(x, ws, [KEYS]), axis=-1)
for label, groups in GROUPINGS[1:]:
    got = mx.concatenate(grouped(x, ws, groups), axis=-1)
    mx.eval(ref, got)
    d = (got.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
    rel = d / (ref.astype(mx.float32).abs().max().item() + 1e-12)
    print(f"  数值 {label:<26} rel={rel:.2e}")
