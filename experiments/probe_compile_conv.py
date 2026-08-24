"""compile 与否对 causal_conv（custom_function + metal_kernel）的影响。

bench_conv（不 compile）单 conv f+b 1.57ms；sweep_conv_tile（compile）
同形状 5.7ms。本脚本用完全相同的 loss 对照 compile / eager，并加入
纯 elementwise 对照排除测量口径问题。

用法: uv run python experiments/probe_compile_conv.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels.conv import _conv_eager, causal_conv

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
C = int(os.environ.get("VIBY_BENCH_C", 768))

mx.random.seed(0)
x = (mx.random.normal((B, T, C)) * 0.5).astype(mx.bfloat16)
w = (mx.random.normal((4, C)) * 0.3).astype(mx.bfloat16)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
cot = mx.random.normal((B, T, C)).astype(mx.bfloat16)
mx.eval(x, w, seg, cot)


def timed(fn, it=10, w_=3):
    for _ in range(w_):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def loss_fused(x_, w_):
    return (causal_conv(x_, w_, seg=seg).astype(mx.float32) * cot).sum()


def loss_eager(x_, w_):
    return (_conv_eager(x_, w_, seg, True).astype(mx.float32) * cot).sum()


def loss_elem(x_, w_):
    """纯 elementwise 对照：同样的输入输出流量，无 custom_function。"""
    y = x_ * w_[0] + x_ * w_[1]
    return (y.astype(mx.float32) * cot).sum()


print(f"B={B} T={T} C={C}\n")
print(f"{'loss':<20}{'模式':<10}{'fwd':>9}{'f+b':>9}{'bwd':>9}")
for name, fn in (
    ("fused conv", loss_fused),
    ("eager conv", loss_eager),
    ("纯 elementwise", loss_elem),
):
    vg = mx.value_and_grad(fn, argnums=(0, 1))
    cf, cvg = mx.compile(fn), mx.compile(vg)
    for mode, f_, b_ in (("eager", fn, vg), ("compile", cf, cvg)):
        tf = timed(lambda: f_(x, w))
        tfb = timed(lambda: b_(x, w))
        print(f"{name:<20}{mode:<10}{tf:>9.3f}{tfb:>9.3f}{tfb - tf:>9.3f}")
    mx.clear_cache()

# 多次调用同一 kernel（KDA 每层 3 个 conv 共享输入）
print()


def loss3(x_, ws_):
    return sum(
        (causal_conv(x_, ws_[i], seg=seg).astype(mx.float32) * cot).sum()
        for i in range(3)
    )


ws = [w, w * 1.1, w * 0.9]
mx.eval(ws)
vg3 = mx.value_and_grad(loss3, argnums=(0, 1))
c3, cvg3 = mx.compile(loss3), mx.compile(vg3)
for mode, f_, b_ in (("eager", loss3, vg3), ("compile", c3, cvg3)):
    tf = timed(lambda: f_(x, ws))
    tfb = timed(lambda: b_(x, ws))
    print(f"{'3× conv 共享输入':<20}{mode:<10}{tf:>9.3f}{tfb:>9.3f}{tfb - tf:>9.3f}")
