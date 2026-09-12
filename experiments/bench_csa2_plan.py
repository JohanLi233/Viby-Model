"""Corrected-baseline CSA2 ABBA: identical weights, inputs, dtype and optimizer.

Example: .venv/bin/python experiments/bench_csa2_plan.py --run-dir research_runs/csa2_plan
The reference includes correctness repairs; the buggy mHC/Indexer paths are
never used as a performance baseline. Outputs include all raw timed samples.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx.utils import tree_flatten

from experiments.bench_kernel_optimizations import (
    build_trainer,
    make_batch,
    run_fb,
    run_window,
)
from experiments.kernel_bench_utils import (
    abba_blocks,
    append_jsonl,
    input_identity,
    param_identity,
    restore_train_state,
    snapshot_train_state,
)
from model.kernels import (
    sparse_attention as sa,
    indexer_select as fs,
    moe_dispatch as moe,
    hc_pre_norm as hc,
)
from model.kernels import moe_counts, moe_decode, moe_gather
from trainer import muon as optimizer_impl
from trainer.flops import DEFAULT_PEAK_TFLOPS
from experiments.audit_training_flops import audit_training_flops


VARIANTS = {
    "corrected": dict(
        fused=False,
        bq=False,
        radix=False,
        combine=False,
        key=False,
        grouped=True,
        split=True,
    ),
    "fused_indexer": dict(
        fused=True,
        bq=True,
        radix=True,
        combine=False,
        key=False,
        grouped=True,
        split=True,
    ),
    "fused_combine": dict(
        fused=True,
        bq=True,
        radix=True,
        combine=True,
        key=False,
        grouped=True,
        split=True,
    ),
    "key_owned": dict(
        fused=True,
        bq=True,
        radix=True,
        combine=True,
        key=True,
        grouped=True,
        split=True,
    ),
    "legacy_bwd": dict(
        fused=False,
        bq=False,
        radix=False,
        combine=False,
        key=False,
        grouped=True,
        split=False,
    ),
    "ungrouped_dw": dict(
        fused=False,
        bq=False,
        radix=False,
        combine=False,
        key=False,
        grouped=False,
        split=True,
    ),
}


# Compare against the already-optimized stack, not the pre-CSA2 baseline.
VARIANTS.update(
    {
        "dataflow_gather": dict(VARIANTS["fused_combine"], gather_vjp=True),
        "dataflow_counts": dict(VARIANTS["fused_combine"], compact_aux=True),
        "dataflow_combined": dict(
            VARIANTS["fused_combine"], gather_vjp=True, compact_aux=True
        ),
        "optimizer_norm": dict(VARIANTS["fused_combine"], fast_opt_norm=True),
        "dataflow_optimizer": dict(
            VARIANTS["fused_combine"],
            gather_vjp=True,
            compact_aux=True,
            fast_opt_norm=True,
        ),
    }
)
for variant in VARIANTS.values():
    variant.setdefault("gather_vjp", False)
    variant.setdefault("compact_aux", False)
    variant.setdefault("fast_opt_norm", False)

# September 12 baseline: preserve every currently enabled optimization.
VARIANTS["current"] = dict(VARIANTS["fused_combine"], fast_opt_norm=True)
VARIANTS["parallel_bwd"] = dict(VARIANTS["current"], parallel_bwd=True)
VARIANTS["fused_bwd"] = dict(VARIANTS["current"], fused_bwd=True)
VARIANTS["contiguous_adam"] = dict(VARIANTS["fused_bwd"], contiguous_adam=True)
VARIANTS["fused_dataflow"] = dict(
    VARIANTS["fused_bwd"], gather_vjp=True, compact_aux=True
)
VARIANTS["sinkhorn_rows"] = dict(VARIANTS["fused_bwd"], sinkhorn_rows=True)
VARIANTS["sinkhorn_pairs"] = dict(VARIANTS["fused_bwd"], sinkhorn_pairs=True)
VARIANTS["window_owned"] = dict(VARIANTS["fused_bwd"], window_owned=True)


def configure(name):
    v = VARIANTS[name]
    fs._ENABLED, fs._BQ_ENABLED = v["fused"], v["bq"]
    sa._TOPK_ENABLED, sa._KEY_OWNED_BWD, sa._SPLIT_BWD = (
        v["radix"],
        v["key"],
        v["split"],
    )
    moe._COMBINE_ENABLED, hc._GROUPED_DW = v["combine"], v["grouped"]
    moe_gather._ENABLED, moe_counts._ENABLED = v["gather_vjp"], v["compact_aux"]
    moe_decode._ENABLED = False
    optimizer_impl._SINKHORN_FAST_NORM = v["fast_opt_norm"]
    sa._PARALLEL_BWD = v.get("parallel_bwd", False)
    sa._FUSED_BWD = v.get("fused_bwd", False)
    optimizer_impl._ADAM_CONTIG_GRADS = v.get("contiguous_adam", False)
    optimizer_impl._SINKHORN_FUSED_ROWS = v.get("sinkhorn_rows", False)
    optimizer_impl._SINKHORN_FUSED_PAIRS = v.get("sinkhorn_pairs", False)
    sa._WINDOW_OWNED_BWD = v.get("window_owned", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seg", type=float, default=200)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--cache-limit-gb", type=float, default=8)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-iters", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--reference", choices=VARIANTS, default="corrected")
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=["fused_indexer", "fused_combine", "key_owned"],
    )
    ap.add_argument("--mode", choices=["fb", "window"], default="fb")
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    args = ap.parse_args()
    if not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0:
        ap.error("--peak-tflops must be finite and positive")
    if (
        min(args.batch, args.seq, args.accum, args.block_iters, args.blocks) <= 0
        or args.warmup < 0
    ):
        ap.error("shape/count arguments must be positive; warmup must be nonnegative")
    mx.set_default_device(mx.gpu)
    mx.random.seed(1234)
    sa._parallel_backward_kernels()  # create callables before compile tracing
    sa._fused_backward_kernel()
    sa._fused_backward_kernel(False, True)
    from model.kernels.window_attention_backward import _kernels as window_kernels

    window_kernels()
    sa._fused_delta_kernel()
    configure(args.reference)
    _, cfg, model, trainer = build_trainer(args)
    batches = [
        make_batch(cfg, args.batch, args.seq, 20260911 + i, args.seg)
        for i in range(args.accum)
    ]
    # Materialize optimizer buffers before freezing the comparison state.
    print("initializing resident optimizer state", flush=True)
    run_window(trainer, batches)
    snap = snapshot_train_state(model, trainer.optimizer, trainer)
    en_delta = trainer._en_delta
    audits = [
        audit_training_flops(model, batch)
        for batch in (batches if args.mode == "window" else batches[:1])
    ]
    fpt = sum(a["flops_per_token"] for a in audits) / len(audits)
    tokens = args.batch * args.seq * (args.accum if args.mode == "window" else 1)
    names = list(dict.fromkeys([args.reference, *args.variants]))
    graphs = {}
    path = Path(args.run_dir) / "results.jsonl"
    meta = dict(
        kind="protocol",
        args=vars(args),
        params=param_identity(model),
        mlx=mx.__version__,
        flags={n: VARIANTS[n] for n in names},
        runtime_env={
            key: os.environ.get(key)
            for key in (
                "MLX_MAX_MB_PER_BUFFER",
                "MLX_MAX_OPS_PER_BUFFER",
                "MLX_METAL_FAST_SYNCH",
                "MLX_BFS_MAX_WIDTH",
            )
        },
        input_hash=input_identity(
            *[v for batch in batches for v in batch if v is not None]
        ),
        flops_audit=audits,
        peak_tflops=args.peak_tflops,
        peak_basis="declared empirical dense-GEMM ceiling; not an official hardware rating",
        resolved_config=cfg.to_dict(),
        target_80pct_seconds=tokens * fpt / (0.80 * args.peak_tflops * 1e12),
        target_70pct_seconds=tokens * fpt / (0.70 * args.peak_tflops * 1e12),
        model=dict(
            dim=cfg.dim,
            layers=cfg.n_layers,
            heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            window=cfg.window_size,
            index_heads=cfg.index_n_heads,
            index_dim=cfg.index_head_dim,
            candidate_block_size=cfg.candidate_block_size,
            candidate_topk_blocks=cfg.candidate_topk_blocks,
            ratios=list(cfg.compress_ratios),
        ),
    )
    append_jsonl(path, meta)
    print(json.dumps(meta), flush=True)
    reference_updated = None
    for name in names:
        restore_train_state(model, trainer.optimizer, snap, trainer)
        trainer._en_delta = en_delta
        configure(name)
        trainer._loss_and_grad = trainer._build_loss_and_grad()
        print("compile/warmup", name, flush=True)
        outputs, gradients = run_fb(trainer, batches[0])
        graphs[name] = trainer._loss_and_grad
        norm = mx.sqrt(
            sum(
                mx.sum(mx.square(v.astype(mx.float32)))
                for _, v in tree_flatten(gradients)
            )
        )
        mx.eval(norm)
        record = dict(
            kind="numerical",
            variant=name,
            loss=float(outputs[0]),
            gradient_norm=float(norm),
        )
        append_jsonl(path, record)
        print(json.dumps(record), flush=True)
        del outputs, gradients
        if args.mode == "window":
            run_window(trainer, batches)
            updated = dict(tree_flatten(model.trainable_parameters()))
            biases = model.moe_bias_stack()
            if reference_updated is None:
                reference_updated = (updated, biases)
            checks = []
            for key, value in updated.items():
                delta = value.astype(mx.float32) - reference_updated[0][key].astype(
                    mx.float32
                )
                checks.append(
                    mx.stack(
                        [
                            mx.max(mx.abs(delta)),
                            mx.sum(delta * delta),
                            mx.sum(
                                mx.square(reference_updated[0][key].astype(mx.float32))
                            ),
                        ]
                    )
                )
            metrics = mx.stack(checks)
            max_abs = float(mx.max(metrics[:, 0]))
            relative_l2 = float(
                mx.sqrt(
                    mx.sum(metrics[:, 1]) / mx.maximum(mx.sum(metrics[:, 2]), 1e-30)
                )
            )
            bias_max_abs = float(mx.max(mx.abs(biases - reference_updated[1])))
            record = dict(
                kind="window_numerical",
                variant=name,
                parameter_max_abs=max_abs,
                parameter_relative_l2=relative_l2,
                router_bias_max_abs=bias_max_abs,
            )
            append_jsonl(path, record)
            print(json.dumps(record), flush=True)
            if (
                not math.isfinite(max_abs)
                or not math.isfinite(relative_l2)
                or bias_max_abs != 0
            ):
                raise RuntimeError(
                    "nonfinite updated parameters or changed router bias update"
                )
            del updated, biases, checks, metrics, value, delta
    del reference_updated
    restore_train_state(model, trainer.optimizer, snap, trainer)
    trainer._en_delta = en_delta

    def arm(name):
        def run():
            configure(name)
            trainer._loss_and_grad = graphs[name]
            if args.mode == "fb":
                if snap["psr_step"] is not None:
                    trainer._psr_step = snap["psr_step"]
                run_fb(trainer, batches[0])
            else:
                run_window(trainer, batches)

        return run

    def reset():
        restore_train_state(model, trainer.optimizer, snap, trainer)
        trainer._en_delta = en_delta

    for name in args.variants:
        print("ABBA", args.reference, "vs", name, args.mode, flush=True)
        result = abba_blocks(
            arm(args.reference),
            arm(name),
            args.warmup,
            args.block_iters,
            args.blocks,
            before_a=reset if args.mode == "window" else None,
            before_b=reset if args.mode == "window" else None,
        )
        record = dict(
            kind="timing",
            reference=args.reference,
            variant=name,
            mode=args.mode,
            **result,
        )
        record["tokens_per_second"] = {
            arm: tokens / result[arm]["median_s"] for arm in ("A", "B")
        }
        record["mfu_estimate"] = {
            arm: rate * fpt / (args.peak_tflops * 1e12)
            for arm, rate in record["tokens_per_second"].items()
        }
        append_jsonl(path, record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "variant",
                        "mode",
                        "paired_block_median_B_over_A",
                        "aa_end_over_start_abs_rel",
                        "inconclusive_aa_drift",
                        "tokens_per_second",
                        "mfu_estimate",
                    )
                }
            ),
            flush=True,
        )
    reset()


if __name__ == "__main__":
    main()
