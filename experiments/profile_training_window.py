"""Focused, state-restored training-window attribution on the current recipe.

Component barriers are explicit: component numbers guide optimization and are
not a substitute for the uninterrupted whole-window ABBA acceptance benchmark.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx.utils import tree_map
from experiments.bench_kernel_optimizations import (
    build_trainer,
    make_batch,
    run_fb,
    run_window,
)
from experiments.kernel_bench_utils import (
    snapshot_train_state,
    restore_train_state,
    summarize_times,
)
from trainer.flops import training_flops_per_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seg", type=float, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=8)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    mx.random.seed(1234)
    _, cfg, model, trainer = build_trainer(args)
    batches = [
        make_batch(cfg, args.batch, args.seq, 20260911 + i, args.seg)
        for i in range(args.accum)
    ]
    print("initialize full optimizer window", flush=True)
    run_window(trainer, batches)
    snap = snapshot_train_state(model, trainer.optimizer, trainer)
    samples = {}

    def measure(name, fn):
        start = time.perf_counter()
        result = fn()
        samples.setdefault(name, []).append(time.perf_counter() - start)
        return result

    # These are the actual optimizer objects chosen by current configuration.
    components = getattr(trainer.optimizer, "optimizers", None)
    if components is None:
        components = getattr(trainer.optimizer, "_optimizers", [trainer.optimizer])
    for index, opt in enumerate(components):
        original = opt.apply_gradients

        def apply(gradients, parameters, original=original, opt=opt, index=index):
            def call():
                result = original(gradients, parameters)
                mx.eval(result, opt.state)
                return result

            return measure(f"optimizer/{index}/{type(opt).__name__}", call)

        opt.apply_gradients = apply
    original_bias = model.update_moe_biases

    def bias_update(*a, **kw):
        def call():
            result = original_bias(*a, **kw)
            mx.eval(model.moe_bias_stack())
            return result

        return measure("router_bias", call)

    model.update_moe_biases = bias_update

    for rep in range(args.iters + 1):
        restore_train_state(model, trainer.optimizer, snap, trainer)
        if rep == 1:
            samples.clear()
        start = time.perf_counter()
        gradients, loads, margins = None, None, []
        for batch in batches:
            output, grad = measure("fwd_bwd", lambda: run_fb(trainer, batch))
            if output[6].size:
                margins.append(output[6])

            def accumulate():
                result = (
                    grad if gradients is None else tree_map(mx.add, gradients, grad)
                )
                mx.eval(result)
                return result

            gradients = measure("gradient_accumulation", accumulate)
            loads = output[2] if loads is None else loads + output[2]
            mx.eval(loads)
        measure(
            "optimizer_step_total",
            lambda: trainer._optimizer_step(
                gradients,
                args.accum,
                loads,
                mx.concatenate(margins, axis=1) if margins else None,
            ),
        )
        elapsed = time.perf_counter() - start
        samples.setdefault("window", []).append(elapsed)
        print("window", rep, elapsed, flush=True)
    result = {
        "args": vars(args),
        "config": cfg.to_dict(),
        "nominal_flops_per_token": training_flops_per_token(model, args.seq),
        "components": {k: summarize_times(v) for k, v in samples.items()},
        "peak_allocated_gb": mx.get_peak_memory() / 1e9,
    }
    path = Path(args.run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v["median_s"] for k, v in result["components"].items()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
