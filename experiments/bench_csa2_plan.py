"""Corrected-baseline CSA2 ABBA: identical weights, inputs, dtype and optimizer.

Example: .venv/bin/python experiments/bench_csa2_plan.py --run-dir research_runs/csa2_plan
The reference includes correctness repairs; the buggy mHC/Indexer paths are
never used as a performance baseline. Outputs include all raw timed samples.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx.utils import tree_flatten

from experiments.bench_kernel_optimizations import build_trainer, make_batch, run_fb, run_window
from experiments.kernel_bench_utils import (
    abba_blocks, append_jsonl, input_identity, param_identity, restore_train_state, snapshot_train_state,
)
from model.kernels import sparse_attention as sa, indexer_select as fs, moe_dispatch as moe, hc_pre_norm as hc
from model.kernels import moe_counts, moe_decode, moe_gather


VARIANTS = {
    "corrected": dict(fused=False, bq=False, radix=False, combine=False, key=False, grouped=True, split=True),
    "fused_indexer": dict(fused=True, bq=True, radix=True, combine=False, key=False, grouped=True, split=True),
    "fused_combine": dict(fused=True, bq=True, radix=True, combine=True, key=False, grouped=True, split=True),
    "key_owned": dict(fused=True, bq=True, radix=True, combine=True, key=True, grouped=True, split=True),
    "legacy_bwd": dict(fused=False, bq=False, radix=False, combine=False, key=False, grouped=True, split=False),
    "ungrouped_dw": dict(fused=False, bq=False, radix=False, combine=False, key=False, grouped=False, split=True),
}


# Compare against the already-optimized stack, not the pre-CSA2 baseline.
VARIANTS.update({
    "dataflow_gather": dict(VARIANTS["fused_combine"], gather_vjp=True),
    "dataflow_counts": dict(VARIANTS["fused_combine"], compact_aux=True),
    "dataflow_combined": dict(VARIANTS["fused_combine"], gather_vjp=True, compact_aux=True),
})
for variant in VARIANTS.values():
    variant.setdefault("gather_vjp", False)
    variant.setdefault("compact_aux", False)


def configure(name):
    v = VARIANTS[name]
    fs._ENABLED, fs._BQ_ENABLED = v["fused"], v["bq"]
    sa._TOPK_ENABLED, sa._KEY_OWNED_BWD, sa._SPLIT_BWD = v["radix"], v["key"], v["split"]
    moe._COMBINE_ENABLED, hc._GROUPED_DW = v["combine"], v["grouped"]
    moe_gather._ENABLED, moe_counts._ENABLED = v["gather_vjp"], v["compact_aux"]
    moe_decode._ENABLED = False


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
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=["fused_indexer", "fused_combine", "key_owned"])
    ap.add_argument("--mode", choices=["fb", "window"], default="fb")
    args = ap.parse_args()
    mx.set_default_device(mx.gpu)
    mx.random.seed(1234)
    configure(args.reference)
    _, cfg, model, trainer = build_trainer(args)
    batches = [make_batch(cfg, args.batch, args.seq, 20260911 + i, args.seg) for i in range(args.accum)]
    # Materialize optimizer buffers before freezing the comparison state.
    print("initializing resident optimizer state", flush=True)
    _, grads = run_fb(trainer, batches[0])
    trainer.optimizer.update(model, grads)
    mx.eval(model.parameters(), trainer.optimizer.state)
    del grads
    snap = snapshot_train_state(model, trainer.optimizer)
    en_delta = trainer._en_delta
    names = list(dict.fromkeys([args.reference, *args.variants]))
    graphs = {}
    path = Path(args.run_dir) / "results.jsonl"
    meta = dict(kind="protocol", args=vars(args), params=param_identity(model),
                mlx=mx.__version__, flags={n: VARIANTS[n] for n in names},
                input_hash=input_identity(*[v for batch in batches for v in batch if v is not None]),
                model=dict(dim=cfg.dim, layers=cfg.n_layers, heads=cfg.n_heads, head_dim=cfg.head_dim,
                           window=cfg.window_size, index_heads=cfg.index_n_heads, index_dim=cfg.index_head_dim,
                           candidate_block_size=cfg.candidate_block_size, candidate_topk_blocks=cfg.candidate_topk_blocks,
                           ratios=list(cfg.compress_ratios)))
    append_jsonl(path, meta)
    print(json.dumps(meta), flush=True)
    for name in names:
        configure(name)
        trainer._loss_and_grad = trainer._build_loss_and_grad()
        print("compile/warmup", name, flush=True)
        outputs, gradients = run_fb(trainer, batches[0])
        graphs[name] = trainer._loss_and_grad
        norm = mx.sqrt(sum(mx.sum(mx.square(v.astype(mx.float32))) for _, v in tree_flatten(gradients)))
        mx.eval(norm)
        record = dict(kind="numerical", variant=name, loss=float(outputs[0]), gradient_norm=float(norm))
        append_jsonl(path, record)
        print(json.dumps(record), flush=True)
        del outputs, gradients

    def arm(name):
        def run():
            configure(name)
            trainer._loss_and_grad = graphs[name]
            if args.mode == "fb":
                run_fb(trainer, batches[0])
            else:
                run_window(trainer, batches)
        return run

    def reset():
        restore_train_state(model, trainer.optimizer, snap)
        trainer._en_delta = en_delta

    for name in args.variants:
        print("ABBA", args.reference, "vs", name, args.mode, flush=True)
        result = abba_blocks(arm(args.reference), arm(name), args.warmup, args.block_iters, args.blocks,
                             before_a=reset if args.mode == "window" else None,
                             before_b=reset if args.mode == "window" else None)
        record = dict(kind="timing", reference=args.reference, variant=name, mode=args.mode, **result)
        append_jsonl(path, record)
        print(json.dumps({k: record[k] for k in ("variant", "mode", "paired_block_median_B_over_A",
                                               "aa_end_over_start_abs_rel", "inconclusive_aa_drift")}), flush=True)
    reset()


if __name__ == "__main__":
    main()
