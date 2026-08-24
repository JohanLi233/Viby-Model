"""_chunk_kda 微基准：chunk 尺寸 / 精度对 fwd+bwd 的影响。

用法: .venv/bin/python experiments/bench_kda_chunk.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model import kda as kda_mod
from model.kda import _chunk_kda

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
B, H, T, D = 12, 8, 1024, 96


def make():
    q = mx.random.normal((B, H, T, D))
    k = mx.random.normal((B, H, T, D)) / D**0.5
    v = mx.random.normal((B, H, T, D))
    log_g = -mx.random.uniform(0.05, 1.2, (B, H, T, D))
    beta = mx.random.uniform(0.1, 0.9, (B, H, T))
    return q, k, v, log_g, beta


def timed(fn, iters=ITERS, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def fwd_fn():
    args = make()

    def f():
        return _chunk_kda(*args, None)[0]

    def fb():
        return mx.value_and_grad(
            lambda q, k, v, lg, b: _chunk_kda(q, k, v, lg, b, None)[0].sum(),
            argnums=(0, 1, 2, 3, 4),
        )(*args)

    return f, fb


print(f"== _chunk_kda micro B{B} H{H} T{T} D{D} (min of {ITERS}) ==")
for C in (16, 32, 64):
    kda_mod._CHUNK = C
    f, fb = fwd_fn()
    print(f"C={C:<3} eager   fwd {timed(f):8.1f}ms   fwd+bwd {timed(fb):8.1f}ms")
    cf = mx.compile(f)
    cfb = mx.compile(fb)
    print(f"C={C:<3} compile fwd {timed(cf):8.1f}ms   fwd+bwd {timed(cfb):8.1f}ms")
