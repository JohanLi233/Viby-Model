"""causal_conv 分块/线程组参数在真实上下文里的扫描。

bench_conv 单 conv f+b 1.57ms，但 compile 下 3 个共享输入的 conv 是
15.78ms（probe_conv_stage）。这里在「3 conv 共享输入 + compile」口径下
扫 _T_BLOCK、_DW_T_CHUNKS 与线程组宽度，找真实最优。

用法: uv run python experiments/sweep_conv_tile.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import conv as convmod

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
C = int(os.environ.get("VIBY_BENCH_C", 768))

mx.random.seed(0)
x = (mx.random.normal((B, T, C)) * 0.5).astype(mx.bfloat16)
ws = [(mx.random.normal((4, C)) * 0.3).astype(mx.bfloat16) for _ in range(3)]
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
cot = [mx.random.normal((B, T, C)).astype(mx.bfloat16) for _ in range(3)]
mx.eval(x, ws, seg, cot)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def reset():
    convmod._KERNELS.clear()
    convmod._OPS.clear()
    convmod._VERIFIED.clear()
    convmod._DISABLED = False


def loss3(x_, ws_):
    """3 conv 共享输入（KDA q/k/v 口径），cotangent 外部给定。"""
    return sum(
        (convmod.causal_conv(x_, w, seg=seg).astype(mx.float32) * c).sum()
        for w, c in zip(ws_, cot)
    )


def loss1(x_, ws_):
    return (convmod.causal_conv(x_, ws_[0], seg=seg).astype(mx.float32) * cot[0]).sum()


el = B * T * C
print(f"B={B} T={T} C={C}  元素 {el / 1e6:.2f}M")
print(f"fwd 下界(读x+写y) {el * 4 / 400e9 * 1e3:.2f}ms")
print(f"bwd 下界(x+dy+dx+dz往返) {el * 14 / 400e9 * 1e3:.2f}ms\n")

orig_tb, orig_dw, orig_tg = convmod._T_BLOCK, convmod._DW_T_CHUNKS, convmod._TG_W
print(
    f"{'tblk':>6}{'dwchunk':>9}{'tgw':>6}{'1conv f+b':>12}{'3conv f+b':>12}{'3conv fwd':>11}"
)
best = None
for tb in (32, 64, 128, 256):
    for dwc in (32, 64, 128):
        for tgw in (64, 128, 256):
            convmod._T_BLOCK, convmod._DW_T_CHUNKS, convmod._TG_W = tb, dwc, tgw
            reset()
            c1 = mx.compile(mx.value_and_grad(loss1, argnums=(0, 1)))
            c3 = mx.compile(mx.value_and_grad(loss3, argnums=(0, 1)))
            c3f = mx.compile(loss3)
            t1 = timed(lambda: c1(x, ws))
            t3 = timed(lambda: c3(x, ws))
            t3f = timed(lambda: c3f(x, ws))
            print(f"{tb:>6}{dwc:>9}{tgw:>6}{t1:>12.2f}{t3:>12.2f}{t3f:>11.2f}")
            if best is None or t3 < best[0]:
                best = (t3, tb, dwc, tgw)
            mx.clear_cache()
convmod._T_BLOCK, convmod._DW_T_CHUNKS, convmod._TG_W = orig_tb, orig_dw, orig_tg
print(
    f"\n最优 3conv f+b {best[0]:.2f}ms @ tblk={best[1]} dwchunk={best[2]} tgw={best[3]}"
)
