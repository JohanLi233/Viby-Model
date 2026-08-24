"""KDA 分解段融合 kernel 的数值校验 + 微基准。

对照 eager 参考验 fwd 五个输出与 bwd 三个梯度，再测真实形状下的
fwd / fwd+bwd（eager 走 mx.compile，给它最有利的口径）。

用法: uv run python experiments/verify_kda_prep.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK
from model.kernels.kda_prep import _prep_eager, kda_prep, prewarm

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H, D = 8, 96
C = KDA_CHUNK
NC = T // C
BW = 400e9

print(f"B={B} T={T} H={H} D={D} C={C} NC={NC}")
ok = prewarm(C, D, NC)
print(f"prewarm(在线校验) = {ok}\n")
if not ok:
    print("校验失败，kernel 被禁用")
    sys.exit(1)

mx.random.seed(1)
q = (mx.random.normal((B, H, NC, C, D)) * 0.3).astype(mx.float32)
k = (mx.random.normal((B, H, NC, C, D)) * 0.3).astype(mx.float32)
lg = mx.maximum(-mx.random.uniform(0.001, 2.0, (B, H, NC, C, D)), -4.0)
mx.eval(q, k, lg)

NAMES = ("qe", "ke", "ki", "kd", "egl")
print("=== fwd 五路输出 vs eager ===")
worst = 0.0
for name, got, ref in zip(NAMES, kda_prep(q, k, lg), _prep_eager(q, k, lg)):
    mx.eval(got, ref)
    amax = mx.abs(ref).max().item()
    ad = mx.abs(got - ref).max().item()
    rel = ad / max(amax, 1e-30)
    worst = max(worst, rel)
    print(f"  {name:<4} max|Δ|={ad:.3e}  |ref|max={amax:.3e}  rel={rel:.3e}")

cots = [mx.random.normal((B, H, NC, C, D)) for _ in range(4)]
cots.append(mx.random.normal((B, H, NC, D)))
mx.eval(cots)


def mk(fn):
    def f(a, b, c):
        return sum((o * ct).sum() for o, ct in zip(fn(a, b, c), cots))

    return f


print("\n=== bwd 三路梯度 vs eager ===")
g_got = mx.grad(mk(kda_prep), argnums=(0, 1, 2))(q, k, lg)
g_ref = mx.grad(mk(_prep_eager), argnums=(0, 1, 2))(q, k, lg)
mx.eval(g_got, g_ref)
for name, got, ref in zip(("dq", "dk", "dlog_g"), g_got, g_ref):
    amax = mx.abs(ref).max().item()
    ad = mx.abs(got - ref).max().item()
    rel = ad / max(amax, 1e-30)
    worst = max(worst, rel)
    print(f"  {name:<7} max|Δ|={ad:.3e}  |ref|max={amax:.3e}  rel={rel:.3e}")

print(f"\n最大相对偏差 {worst:.3e}  {'OK' if worst < 1e-5 else '✗ 超差'}")


def timed(fn, it=10, w=4):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


print("\n=== 微基准（真实形状）===")
el = B * H * NC * C * D
low = (3 * el + 4 * el + B * H * NC * D) * 4 / BW * 1e3
print(
    f"单份 (B,H,NC,C,D) f32 = {el * 4 / 2**20:.1f}MB   fwd compulsory 下界 {low:.2f}ms"
)
print(f"{'实现':<24}{'fwd':>9}{'f+b':>9}{'bwd':>9}")
for label, fn in (("eager (mx.compile)", _prep_eager), ("融合 kernel", kda_prep)):
    cf = mx.compile(fn) if fn is _prep_eager else fn
    cg = mx.compile(mx.grad(mk(fn), argnums=(0, 1, 2)))
    f = timed(lambda: cf(q, k, lg))
    fb = timed(lambda: cg(q, k, lg))
    print(f"{label:<24}{f:>9.3f}{fb:>9.3f}{fb - f:>9.3f}")
    mx.clear_cache()

print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
sys.exit(0 if worst < 1e-5 else 1)
