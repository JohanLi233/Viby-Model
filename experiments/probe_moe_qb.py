"""Synthetic controller recovery and router-only timing, not a loss benchmark.

Run: .venv/bin/python experiments/probe_moe_qb.py --output /tmp/moe_qb.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np

from model.config import VibyConfig
from model.moe import MoEGate, update_expert_bias, update_quantile_bias


def recovery():
    rng = np.random.default_rng(1337)
    base = rng.normal(0, 0.1, (8192, 96)).astype(np.float32)
    base[:, :6] += 1.0
    records = {}
    for scale in (0.02, 1.0, 20.0):
        raw = mx.array(base * scale)
        for method in ("noaux_tc", "qb"):
            bias = mx.zeros((96,))
            trace = []
            for _ in range(48):
                biased = raw + bias
                ids = mx.argpartition(-biased, kth=5, axis=1)[:, :6]
                counts = np.bincount(np.asarray(ids).reshape(-1), minlength=96)
                trace.append(float(counts.max() / counts.mean()))
                if method == "qb":
                    alpha = -mx.partition(-biased, kth=6, axis=1)[:, 6:7]
                    bias = update_quantile_bias(bias, raw - alpha, 6)
                else:
                    bias = update_expert_bias(bias, mx.array(counts), 1e-3)
                mx.eval(bias)
            records[f"{method}_scale_{scale}"] = {
                "max_load_ratios": trace,
                "first_below_1.2": next(
                    (i for i, r in enumerate(trace) if r < 1.2), None
                ),
            }
    return records


def timing():
    mx.random.seed(1337)
    cfg = VibyConfig(n_mtp_layers=0)
    gate = MoEGate(cfg)
    x = mx.random.normal((16384, cfg.dim)).astype(mx.bfloat16)
    weight = gate.weight
    bias = gate.bias
    cot = mx.random.normal((16384, 6))
    gate.qb_stats_rows = 4096  # accumulation=2, window budget=8192
    functions = {}
    for name, method, fp32 in (
        ("bf16_sign", "noaux_tc", False),
        ("fp32_sign", "noaux_tc", True),
        ("fp32_qb", "qb", True),
    ):
        gate.balance_method, gate.router_fp32 = method, fp32

        def f(w, b, inputs):
            gate.weight, gate.bias = w, b
            weights, ids, scores = gate(inputs)
            return (weights * cot).sum(), gate._last_load, gate._last_qb_margins

        compiled = mx.compile(mx.value_and_grad(f))
        w = weight if fp32 else weight.astype(mx.bfloat16)
        # Trace each static mode before constructing the next variant.
        for _ in range(3):
            mx.eval(compiled(w, bias, x))
        functions[name] = (compiled, w)
    values = {name: [] for name in functions}
    # Alternate modes in forward/reverse order, same inputs/process.
    for _ in range(5):
        for name in [*functions, *reversed(functions)]:
            fn, w = functions[name]
            start = time.perf_counter()
            mx.eval(fn(w, bias, x))
            values[name].append((time.perf_counter() - start) * 1000)
    margins = mx.random.normal((8192, 96))
    mx.eval(margins)
    update = mx.compile(lambda b, m: update_quantile_bias(b, m, 6))
    for _ in range(3):
        mx.eval(update(bias, margins))
    updates = []
    for _ in range(10):
        start = time.perf_counter()
        mx.eval(update(bias, margins))
        updates.append((time.perf_counter() - start) * 1000)
    return {
        "gate_fwd_bwd_ms": {name: float(np.median(v)) for name, v in values.items()},
        "single_layer_window_update_ms": float(np.median(updates)),
        "sample_buffer_mib_12_layers": 12 * 8192 * 96 * 4 / 2**20,
        "scope": "router only; excludes experts, attention, full optimizer and full-model throughput",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "mlx": mx.__version__,
        "seed": 1337,
        "recovery": recovery(),
        "timing": timing(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "recovery": {
                    k: {
                        "first_below_1.2": v["first_below_1.2"],
                        "last": v["max_load_ratios"][-1],
                    }
                    for k, v in result["recovery"].items()
                },
                "timing": result["timing"],
            },
            indent=2,
        )
    )
