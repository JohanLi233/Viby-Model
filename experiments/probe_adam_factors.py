"""Measure scalar bias-correction hoisting outside a large fused AdamW graph."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from trainer.muon import _adamw_body
from experiments.kernel_bench_utils import abba_blocks, append_jsonl

b1, b2, eps, wd = 0.9, 0.9999, 4.679e-15, 0.0
mx.random.seed(12)
shape = (96, 512, 1024)
p = mx.random.normal(shape).astype(mx.bfloat16)
g = (mx.random.normal(shape) * 0.01).astype(mx.bfloat16)
m = mx.zeros_like(p)
v = mx.ones_like(p) * 0.0001
lr = mx.array(0.0004, mx.bfloat16)
step = mx.array(100, mx.uint64)
c1 = (lr / (1 - b1**step)).astype(p.dtype)
c2 = mx.rsqrt(1 - b2**step).astype(p.dtype)
decay = 1 - lr * wd
mx.eval(p, g, m, v, lr, step, c1, c2, decay)
original = mx.compile(_adamw_body(b1, b2, eps, wd, True))


@mx.compile
def hoisted(p, g, m, v, c1, c2, decay):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * mx.square(g)
    p = p * decay
    return p - (c1 * m) / (mx.sqrt(v) * c2 + eps), m, v


a = original(p, g, m, v, lr, step)
b = hoisted(p, g, m, v, c1, c2, decay)
mx.eval(a, b)
delta = [
    float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
    for x, y in zip(a, b)
]
print("parity", delta, flush=True)
del a, b
result = abba_blocks(
    lambda: mx.eval(original(p, g, m, v, lr, step)),
    lambda: mx.eval(hoisted(p, g, m, v, c1, c2, decay)),
    warmup=4,
    block_iters=5,
    n_blocks=3,
)
append_jsonl(
    "research_runs/kernel_goal_20260912/adam_factors.jsonl", dict(delta=delta, **result)
)
print(
    json.dumps(
        {
            k: result[k]
            for k in ("paired_block_median_B_over_A", "aa_end_over_start_abs_rel")
        }
    ),
    flush=True,
)
print(result["A"]["median_s"], result["B"]["median_s"], flush=True)
