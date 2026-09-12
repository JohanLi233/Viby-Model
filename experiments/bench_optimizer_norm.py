"""Same-input Sinkhorn optimizer ABBA; full windows remain the acceptance gate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from experiments.kernel_bench_utils import abba_blocks, append_jsonl
from trainer.muon import _sinkhorn_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=491873)
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    mx.random.seed(1234)
    mx.set_cache_limit(8 << 30)
    shape = (args.rows, args.cols)
    p = mx.random.normal(shape).astype(mx.bfloat16)
    g = (mx.random.normal(shape) * 0.01).astype(mx.bfloat16)
    m = mx.zeros_like(p)
    lr = mx.array(0.002, mx.bfloat16)
    mx.eval(p, g, m, lr)
    native = mx.compile(_sinkhorn_body(0.95, 0.18, 11, 1e-20, 0.001, args.cols))
    fast = mx.compile(_sinkhorn_body(0.95, 0.18, 11, 1e-20, 0.001, args.cols, True))
    a, am = native(p, g, m, lr)
    b, bm = fast(p, g, m, lr)
    delta = b.astype(mx.float32) - a.astype(mx.float32)
    numerical = dict(
        kind="numerical",
        shape=shape,
        max_abs=float(mx.max(mx.abs(delta))),
        rms=float(mx.sqrt(mx.mean(delta * delta))),
        momentum_max_abs=float(mx.max(mx.abs(am - bm))),
    )
    print(json.dumps(numerical), flush=True)
    append_jsonl(Path(args.run_dir) / "results.jsonl", numerical)
    del a, b, am, bm, delta

    def arm(fn):
        return lambda: mx.eval(fn(p, g, m, lr))

    result = abba_blocks(arm(native), arm(fast), warmup=5)
    append_jsonl(Path(args.run_dir) / "results.jsonl", dict(kind="timing", **result))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("paired_block_median_B_over_A", "aa_end_over_start_abs_rel")
            }
        ),
        flush=True,
    )
    print(
        "median seconds", result["A"]["median_s"], result["B"]["median_s"], flush=True
    )


if __name__ == "__main__":
    main()
