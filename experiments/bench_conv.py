"""causal_conv（K=4 depthwise + 可选 SiLU）的 fwd / bwd 专项基准。

KDA 每层 3 个（q/k/v，带 SiLU），block 级 ShortConv 3 个（无 SiLU）。
实测 bwd/fwd ≈ 7×，怀疑是 dx/dw 两个 kernel 的 grid 并行度不足
（dw 只有 C×8 线程、dx 只有 C×B 线程，且都串行扫整个 T 轴）。

同时打印访存下界，判断还有多少空间。

用法: uv run python experiments/bench_conv.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import conv as convmod
from model.kernels.conv import _conv_eager, causal_conv

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
C = int(os.environ.get("VIBY_BENCH_C", 768))


def timed(fn, it=10, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


mx.random.seed(0)
x = (mx.random.normal((B, T, C)) * 0.5).astype(mx.bfloat16)
w = (mx.random.normal((4, C)) * 0.3).astype(mx.bfloat16)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
cot = mx.random.normal((B, T, C)).astype(mx.bfloat16)
mx.eval(x, w, seg, cot)

el = B * T * C
print(f"B={B} T={T} C={C}  元素 {el / 1e6:.2f}M")
print(f"x/y bf16 各 {el * 2 / 2**20:.1f}MB, dz f32 {el * 4 / 2**20:.1f}MB")
# fwd 下界：读 x + 写 y；bwd 下界：读 x + 读 dy + 写 dx + 写/读 dz
bw = 400e9  # M4 Max 实测可达带宽（理论 546）
print(
    f"访存下界 @{bw / 1e9:.0f}GB/s: fwd {el * 4 / bw * 1e3:.2f}ms  "
    f"bwd {el * 14 / bw * 1e3:.2f}ms"
)


def run_fwd(sg, silu):
    return causal_conv(x, w, seg=sg, silu=silu)


def make_vg(sg, silu):
    def loss(x_, w_):
        return (causal_conv(x_, w_, seg=sg, silu=silu).astype(mx.float32) * cot).sum()

    return mx.value_and_grad(loss, argnums=(0, 1))


print(f"\n{'配置':<22}{'fwd(ms)':>9}{'fwd+bwd':>9}{'bwd':>8}{'bwd/fwd':>9}")
for label, sg, silu in [
    ("seg+silu (KDA q/k/v)", seg, True),
    ("seg 无silu (ShortConv)", seg, False),
    ("无seg+silu", None, True),
]:
    tf = timed(lambda: run_fwd(sg, silu))
    vg = make_vg(sg, silu)
    tfb = timed(lambda: vg(x, w))
    print(
        f"{label:<22}{tf * 1e3:>9.3f}{tfb * 1e3:>9.3f}"
        f"{(tfb - tf) * 1e3:>8.3f}{(tfb - tf) / tf:>9.2f}"
    )


# eager 参考对照
def eager_vg(x_, w_):
    return (_conv_eager(x_, w_, seg, True).astype(mx.float32) * cot).sum()


vg_e = mx.value_and_grad(eager_vg, argnums=(0, 1))
tfe = timed(lambda: _conv_eager(x, w, seg, True))
tfbe = timed(lambda: vg_e(x, w))
print(
    f"{'eager 参考 (seg+silu)':<22}{tfe * 1e3:>9.3f}{tfbe * 1e3:>9.3f}"
    f"{(tfbe - tfe) * 1e3:>8.3f}{(tfbe - tfe) / tfe:>9.2f}"
)

# dw 切片数扫描（当前默认 8）
print(f"\ndw 归约切片数扫描（当前 _DW_T_CHUNKS={convmod._DW_T_CHUNKS}）")
orig = convmod._DW_T_CHUNKS
for tc in (8, 16, 32, 64, 128, 256):
    convmod._DW_T_CHUNKS = tc
    convmod._KERNELS.clear()
    convmod._OPS.clear()
    convmod._VERIFIED.clear()
    convmod._DISABLED = False
    vg = make_vg(seg, True)
    tfb = timed(lambda: vg(x, w))
    tf = timed(lambda: run_fwd(seg, True))
    print(
        f"  chunks={tc:>4}  fwd+bwd {tfb * 1e3:>7.3f}ms  bwd {(tfb - tf) * 1e3:>7.3f}ms"
        f"  线程数 {C * tc:>7}"
    )
convmod._DW_T_CHUNKS = orig
