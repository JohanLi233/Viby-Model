"""Screen microbatch schedules with identical effective token batch and weights.

This measures execution geometry, not bitwise training-trajectory equivalence:
QB sample rounding depends on microbatch partitioning.
Kernel A/B acceptance must subsequently hold the chosen geometry fixed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from experiments.bench_kernel_optimizations import build_trainer, make_batch, run_window
from experiments.kernel_bench_utils import (
    snapshot_train_state,
    restore_train_state,
    summarize_times,
    append_jsonl,
)
from experiments.audit_training_flops import audit_training_flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--microbatches", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--effective-batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seg", type=float, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=4)
    ap.add_argument("--peak-tflops", type=float, default=13.5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    if any(b <= 0 or args.effective_batch % b for b in args.microbatches):
        ap.error("each microbatch must divide the effective batch")
    args.batch = args.microbatches[0]
    args.accum = args.effective_batch // args.batch
    mx.random.seed(1234)
    _, cfg, model, trainer = build_trainer(args)
    whole = make_batch(cfg, args.effective_batch, args.seq, 20260912, args.seg)
    path = Path(args.run_dir) / "results.jsonl"

    def batches_for(b):
        return [
            tuple(None if x is None else x[start : start + b] for x in whole)
            for start in range(0, args.effective_batch, b)
        ]

    batches = batches_for(args.batch)
    print("initializing optimizers", flush=True)
    run_window(trainer, batches)
    frozen = snapshot_train_state(model, trainer.optimizer, trainer)
    append_jsonl(
        path,
        dict(
            kind="protocol",
            args=vars(args),
            config=cfg.to_dict(),
            learning_rate=str(trainer.args.learning_rate),
            peak_tflops=args.peak_tflops,
        ),
    )
    for b in args.microbatches:
        restore_train_state(model, trainer.optimizer, frozen, trainer)
        trainer.args.batch_size = b
        trainer.args.accumulation_steps = args.effective_batch // b
        for gate in trainer._moe_gates:
            gate.qb_stats_rows = max(
                1, cfg.qb_stats_rows // trainer.args.accumulation_steps
            )
        trainer._loss_and_grad = trainer._build_loss_and_grad()
        batches = batches_for(b)
        # Visibility is counted on the actual rows, just as in the
        # fixed-geometry acceptance benchmark.
        audit = [audit_training_flops(model, batch) for batch in batches]
        fpt = sum(r["flops_per_token"] for r in audit) / len(audit)
        for _ in range(args.warmup):
            restore_train_state(model, trainer.optimizer, frozen, trainer)
            run_window(trainer, batches)
        times = []
        mx.reset_peak_memory()
        for _ in range(args.iters):
            restore_train_state(model, trainer.optimizer, frozen, trainer)
            start = time.perf_counter()
            run_window(trainer, batches)
            times.append(time.perf_counter() - start)
        summary = summarize_times(times)
        tps = args.effective_batch * args.seq / summary["median_s"]
        row = dict(
            kind="geometry",
            microbatch=b,
            accum=trainer.args.accumulation_steps,
            times=summary,
            tokens_per_second=tps,
            estimated_mfu=tps * fpt / (args.peak_tflops * 1e12),
            measured_attention_flops=fpt,
            peak_allocated_gb=mx.get_peak_memory() / 1e9,
            qb_rows_per_microbatch=trainer._moe_gates[0].qb_stats_rows,
        )
        append_jsonl(path, row)
        print(json.dumps(row), flush=True)
        if (
            mx.get_peak_memory()
            > 0.88 * mx.device_info()["max_recommended_working_set_size"]
        ):
            print("stopping geometry expansion at working-set limit", flush=True)
            break


if __name__ == "__main__":
    main()
