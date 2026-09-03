"""iHC 融合核 vs eager：16 步残差链 compile 口径。

用法: .venv/bin/python experiments/ab_ihc_fused.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten

from model.ihc import IHCGate
from model.kernels import ihc_fused

B, T, D, M = 12, 1024, 768, 4
DT = mx.bfloat16
N_SUB = 16


def _time(fn, reps=7, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def chain(gate, r0, scale):
    r = r0
    for _ in range(N_SUB):
        x, hp = gate.mix(r)
        r = gate.write(r, x * scale, hp)
    return r.astype(mx.float32).square().sum()


def main():
    mx.random.seed(0)
    print("prewarm", ihc_fused.prewarm(M, D, DT, 1e-6))
    gate = IHCGate(D, M)
    gate.eval()
    r0 = (mx.random.normal((B, T, M, D)) * 0.5).astype(DT)
    scale = mx.array(0.1, dtype=DT)
    mx.eval(r0, scale, *dict(tree_flatten(gate.parameters())).values())

    vg = mx.value_and_grad(lambda r: chain(gate, r, scale))
    mx.eval(vg(r0))
    print(f"fused eager  f+b {_time(lambda: vg(r0)):7.2f}ms")
    cvg = mx.compile(mx.value_and_grad(lambda r: chain(gate, r, scale)))
    mx.eval(cvg(r0))
    print(f"fused compile f+b {_time(lambda: cvg(r0), reps=5, warmup=2):7.2f}ms")

    ihc_fused._DISABLED = True
    ihc_fused._MIX_FAILED.clear()
    ihc_fused._WR_FAILED.clear()
    vg2 = mx.value_and_grad(lambda r: chain(gate, r, scale))
    mx.eval(vg2(r0))
    print(f"eager eager  f+b {_time(lambda: vg2(r0)):7.2f}ms")
    cvg2 = mx.compile(mx.value_and_grad(lambda r: chain(gate, r, scale)))
    mx.eval(cvg2(r0))
    print(f"eager compile f+b {_time(lambda: cvg2(r0), reps=5, warmup=2):7.2f}ms")


if __name__ == "__main__":
    main()
