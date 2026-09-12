"""Same-state optimizer Sinkhorn comparison at the actual Engram table shape."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from trainer.muon import _sinkhorn_body
from trainer.sinkhorn_rows import prewarm_row_normalize
from trainer.sinkhorn_pairs import prewarm_column_then_row
from experiments.kernel_bench_utils import abba_blocks, append_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", action="store_true")
args = ap.parse_args()
mx.random.seed(143)
mx.set_cache_limit(4 << 30)
shape = (1966704, 64)
p = (mx.random.normal(shape) * 0.01).astype(mx.bfloat16)
g = (mx.random.normal(shape) * 0.001).astype(mx.bfloat16)
m = mx.zeros_like(p)
lr = mx.array(0.0004, mx.bfloat16)
mx.eval(p, g, m, lr)
prewarm_row_normalize((4, shape[1]))
prewarm_column_then_row((4, shape[1]))
a = mx.compile(_sinkhorn_body(0.95, 0.18, 11, 1e-20, 0.001, shape[1], True, False))
b = mx.compile(
    _sinkhorn_body(
        0.95, 0.18, 11, 1e-20, 0.001, shape[1], True, not args.pairs, args.pairs
    )
)
ra, rb = a(p, g, m, lr), b(p, g, m, lr)
mx.eval(ra, rb)
error = [
    float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
    for x, y in zip(ra, rb)
]
print("max_abs", error, flush=True)
del ra, rb
result = abba_blocks(
    lambda: mx.eval(a(p, g, m, lr)),
    lambda: mx.eval(b(p, g, m, lr)),
    warmup=5,
    block_iters=5,
    n_blocks=3,
)
append_jsonl(
    "research_runs/kernel_goal_20260912/sinkhorn_"
    + ("pairs" if args.pairs else "rows")
    + ".jsonl",
    dict(shape=shape, error=error, **result),
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
