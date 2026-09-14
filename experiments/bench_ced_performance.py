"""Recurrent CED reference/optimized ABBA, including real optimizer windows.

Run only when no other MLX GPU job is active. Default shapes are B1/B4 x T1024
with default model width, BF16, plain/packed/padded inputs. No quality claim or
speedup assertion is made. Each measured optimizer window restores one shared
snapshot before its two microbatches, outside the timer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

import model.attention as attention
import model.moe as moe
from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser
from trainer.utils import convert_model_dtype
from experiments.bench_kernel_optimizations import run_fb, run_window
from experiments.kernel_bench_utils import (
    abba_blocks,
    active_viby_flags,
    append_jsonl,
    memory_snapshot,
    param_identity,
    restore_train_state,
    snapshot_train_state,
)


def source_hashes():
    paths = [
        *sorted((ROOT / "model").rglob("*.py")),
        *sorted((ROOT / "trainer").rglob("*.py")),
        ROOT / "experiments/kernel_bench_utils.py",
        ROOT / "experiments/bench_kernel_optimizations.py",
        Path(__file__).resolve(),
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def make_config(args):
    settings = dict(
        n_mtp_layers=0,
        engram_layer_ids=(),
        ced_recurrent_enabled=True,
        ced_recurrent_stride=args.stride,
        ced_recurrent_rounds=args.rounds,
        max_seq_len=max(args.lengths) + 16,
    )
    if args.preset == "tiny":
        settings.update(
            preset="tiny",
            n_layers=12,
            dim=64,
            n_heads=2,
            o_groups=1,
            head_dim=32,
            rope_head_dim=16,
            q_lora_rank=32,
            o_lora_rank=32,
            moe_inter_dim=32,
            n_routed_experts=4,
            n_activated_experts=2,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            vocab_size=48,
            window_size=8,
        )
    return VibyConfig(**settings)


def make_trainer(args, cfg, model):
    argv = [
        "--ced-recurrent",
        "--mtp_depth",
        "0",
        "--no_save",
        "--out_dir",
        str(args.run_dir / "unused_checkpoints"),
        "--optimizer",
        args.optimizer,
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batches[0]),
        "--accumulation_steps",
        "2",
        "--max_seq_len",
        str(max(args.lengths)),
        "--cache_limit_gb",
        str(args.cache_limit_gb),
        "--grad_clip",
        str(args.grad_clip),
    ]
    if args.no_compile:
        argv.append("--no_compile")
    trainer_args = get_pretrain_parser().parse_args(argv)
    # Avoid LR schedule or compute-scaled LR inference. Native Muon still uses
    # its declared fixed group multiplier, recorded in the optimizer metadata.
    trainer_args.adam_lr = args.learning_rate
    trainer_args.muon_lr = args.muon_lr
    trainer_args.save_dir = str(args.run_dir / "unused_checkpoints")
    return BaseTrainer(trainer_args, model, None, cfg, "pretrain")


def make_batches(cfg, batch_size, length, layout, seed, mean_doc_length, pad_fraction):
    rng = np.random.default_rng(seed)
    result, metadata = [], []
    for microbatch in range(2):
        x = rng.integers(1, cfg.vocab_size, size=(batch_size, length), dtype=np.int32)
        y = rng.integers(1, cfg.vocab_size, size=x.shape, dtype=np.int32)
        valid = np.ones_like(x, dtype=bool)
        docs = np.zeros_like(x)
        if layout == "packed":
            boundaries = rng.random(x.shape) < 1.0 / mean_doc_length
            boundaries[:, 0] = True
            docs = np.cumsum(boundaries, axis=1, dtype=np.int32)
        if layout == "padded":
            for row in range(batch_size):
                # Ragged tails exercise differing anchor counts within a batch.
                fraction = pad_fraction * (0.75 + 0.5 * (row + 1) / batch_size)
                valid[row, max(1, int(length * (1 - fraction))) :] = False
        loss_mask = valid.copy()
        if layout == "packed":
            loss_mask[:, :-1] &= docs[:, :-1] == docs[:, 1:]
        attn = None if layout == "plain" else mx.array(valid.astype(np.int32))
        seg = mx.array(docs) if layout == "packed" else None
        arrays = (mx.array(x), mx.array(y), mx.array(loss_mask, mx.float32), attn, seg)
        mx.eval(*[a for a in arrays if a is not None])
        digest = hashlib.sha256()
        for array in (x, y, loss_mask, valid, docs):
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        anchors = 0
        for row in range(batch_size):
            count = 0
            previous = None
            for token in range(length):
                if not valid[row, token]:
                    count, previous = 0, None
                    continue
                if previous != int(docs[row, token]):
                    count = 0
                count += 1
                anchors += count % cfg.ced_recurrent_stride == 0
                previous = int(docs[row, token])
        metadata.append(
            dict(
                microbatch=microbatch,
                full_input_sha256=digest.hexdigest(),
                valid_tokens=int(valid.sum()),
                supervised_tokens=int(loss_mask.sum()),
                valid_anchors=int(anchors),
                allocated_anchor_capacity=batch_size
                * (length // cfg.ced_recurrent_stride),
            )
        )
        result.append(arrays)
    return result, metadata


def configure(variant, cfg, trainer):
    if not hasattr(attention, "_RECURRENT_OPTIMIZED"):
        raise RuntimeError(
            "model.attention._RECURRENT_OPTIMIZED reference switch is required"
        )
    attention._RECURRENT_OPTIMIZED = variant == "optimized"
    if not hasattr(moe, "_RECURRENT_MASKED_COUNTS"):
        raise RuntimeError(
            "model.moe._RECURRENT_MASKED_COUNTS reference switch is required"
        )
    moe._RECURRENT_MASKED_COUNTS = variant == "optimized"
    cfg.ced_recurrent_enabled = variant != "token_baseline"
    for gate in trainer._moe_gates:
        rounds = (
            cfg.ced_recurrent_rounds
            if (
                cfg.ced_recurrent_enabled
                and cfg.n_encoder_layers < gate.layer_idx < cfg.n_layers - 1
            )
            else 1
        )
        gate.qb_stats_rows = max(1, cfg.qb_stats_rows // 2 // rounds)


def optimizer_metadata(optimizer):
    children = getattr(optimizer, "optimizers", None)
    if children is None:
        children = [optimizer]
    return [
        dict(kind=type(child).__name__, learning_rate=float(child.learning_rate))
        for child in children
    ]


def difference(left, right):
    a, b = dict(tree_flatten(left)), dict(tree_flatten(right))
    if set(a) != set(b):
        return {"same_tree": False}
    checks = []
    for name in a:
        av = mx.array(a[name]).astype(mx.float32)
        bv = mx.array(b[name]).astype(mx.float32)
        delta = av - bv
        checks.append(
            mx.stack([mx.max(mx.abs(delta)), mx.sum(delta * delta), mx.sum(av * av)])
        )
    values = mx.stack(checks)
    mx.eval(values)
    return {
        "same_tree": True,
        "max_abs": float(mx.max(values[:, 0])),
        "relative_l2": float(
            mx.sqrt(mx.sum(values[:, 1]) / mx.maximum(mx.sum(values[:, 2]), 1e-30))
        ),
    }


def tree_is_finite(tree):
    checks = [mx.all(mx.isfinite(mx.array(value))) for _, value in tree_flatten(tree)]
    return bool(mx.all(mx.stack(checks))) if checks else True


def aa_median(record):
    return statistics.median(record["A"]["raw_s"] + record["B"]["raw_s"])


def measure_case(
    args, cfg, model, trainer, base_snapshot, batch_size, length, layout, seed
):
    trainer.args.batch_size = batch_size
    batches, inputs = make_batches(
        cfg, batch_size, length, layout, seed, args.mean_doc_length, args.pad_fraction
    )
    variants = ["reference", "optimized"] + (
        ["token_baseline"] if args.token_baseline else []
    )
    graphs, numerics = {}, {}
    reference_gradients = reference_outputs = None

    def restore():
        restore_train_state(model, trainer.optimizer, base_snapshot, trainer)

    # Separate mx.compile callables freeze the chosen implementation at trace.
    # The wrapper also sets the switch before every call, including any retrace.
    for variant in variants:
        restore()
        configure(variant, cfg, trainer)
        trainer._loss_and_grad = trainer._build_loss_and_grad()
        outputs, gradients = run_fb(trainer, batches[0])
        graphs[variant] = trainer._loss_and_grad
        numerics[variant] = {
            "scaled_loss": float(outputs[0]),
            "moe_loads_shape": list(outputs[2].shape),
            "qb_margins_shape": list(outputs[6].shape),
            "finite_loss_and_gradients": tree_is_finite((outputs[0], gradients)),
            "finite_loads": tree_is_finite(outputs[2]),
            "qb_has_no_infinities": bool(mx.all(~mx.isinf(outputs[6]))),
        }
        if variant == "reference":
            reference_gradients, reference_outputs = gradients, outputs
        elif variant == "optimized":
            numerics[variant]["gradient_difference_from_reference"] = difference(
                reference_gradients, gradients
            )
            numerics[variant]["loss_abs_difference_from_reference"] = abs(
                float(outputs[0]) - float(reference_outputs[0])
            )
            numerics[variant]["moe_loads_exact"] = bool(
                mx.array_equal(outputs[2], reference_outputs[2])
            )
            qa, qb = outputs[6], reference_outputs[6]
            numerics[variant]["qb_margins_exact_nan_equal"] = bool(
                mx.all((qa == qb) | (mx.isnan(qa) & mx.isnan(qb)))
            )
        del gradients, outputs
    del reference_gradients, reference_outputs
    restore()
    if "window" in args.modes:
        reference_updated = None
        for variant in ("reference", "optimized"):
            restore()
            configure(variant, cfg, trainer)
            trainer._loss_and_grad = graphs[variant]
            run_window(trainer, batches)
            mx.eval(model.parameters(), trainer.optimizer.state, model.moe_bias_stack())
            updated = snapshot_train_state(model, trainer.optimizer, trainer)
            record = {
                "finite_parameters": tree_is_finite(updated["params"]),
                "finite_optimizer": tree_is_finite(updated["optimizer"]),
                "finite_router_biases": tree_is_finite(updated["biases"]),
            }
            if variant == "reference":
                reference_updated = updated
            else:
                record.update(
                    parameters_difference=difference(
                        reference_updated["params"], updated["params"]
                    ),
                    optimizer_difference=difference(
                        reference_updated["optimizer"], updated["optimizer"]
                    ),
                    router_bias_difference=difference(
                        reference_updated["biases"], updated["biases"]
                    ),
                )
            numerics[variant]["restored_two_microbatch_window"] = record
        del updated, reference_updated
        restore()
    append_jsonl(
        args.run_dir / "measurements.jsonl",
        dict(
            kind="numerical",
            batch_size=batch_size,
            length=length,
            layout=layout,
            numerics=numerics,
        ),
    )
    # NaN padding markers in QB samples are expected; weights, gradients,
    # optimizer arrays and the actual next-window router biases must be finite.
    for variant, record in numerics.items():
        finite = (
            record["finite_loss_and_gradients"]
            and record["finite_loads"]
            and record["qb_has_no_infinities"]
        )
        window = record.get("restored_two_microbatch_window", {})
        finite = finite and all(
            window.get(key, True)
            for key in ("finite_parameters", "finite_optimizer", "finite_router_biases")
        )
        if not finite:
            raise RuntimeError(
                f"nonfinite numerical gate for {variant}; see measurements.jsonl"
            )
    result = dict(
        batch_size=batch_size,
        length=length,
        layout=layout,
        inputs=inputs,
        numerics=numerics,
        modes={},
    )
    for mode in args.modes:

        def arm(variant):
            def run():
                configure(variant, cfg, trainer)
                trainer._loss_and_grad = graphs[variant]
                if mode == "fb":
                    run_fb(trainer, batches[0])
                else:
                    run_window(trainer, batches)
                    mx.eval(
                        model.parameters(),
                        trainer.optimizer.state,
                        model.moe_bias_stack(),
                    )

            return run

        reference = arm("reference")
        initial_aa = abba_blocks(
            reference,
            reference,
            warmup=args.warmup,
            block_iters=args.aa_iters,
            n_blocks=args.aa_blocks,
            label_a="reference_initial_1",
            label_b="reference_initial_2",
            before_a=restore,
            before_b=restore,
        )
        comparisons = {}
        for variant in variants[1:]:
            comparisons[variant] = abba_blocks(
                reference,
                arm(variant),
                warmup=args.warmup,
                block_iters=args.block_iters,
                n_blocks=args.blocks,
                label_a="reference",
                label_b=variant,
                before_a=restore,
                before_b=restore,
            )
        final_aa = abba_blocks(
            reference,
            reference,
            warmup=0,
            block_iters=args.aa_iters,
            n_blocks=args.aa_blocks,
            label_a="reference_final_1",
            label_b="reference_final_2",
            before_a=restore,
            before_b=restore,
        )
        drift = abs(aa_median(final_aa) / aa_median(initial_aa) - 1)
        valid_tokens = sum(
            x["valid_tokens"] for x in inputs[: 1 if mode == "fb" else 2]
        )
        for comparison in comparisons.values():
            comparison["valid_tokens_per_second"] = {
                name: valid_tokens / comparison[name]["median_s"] for name in ("A", "B")
            }
            ratio = comparison["paired_block_median_B_over_A"]
            comparison["paired_time_reduction_fraction"] = 1 - ratio
            comparison["paired_throughput_increase_fraction"] = 1 / ratio - 1
        record = dict(
            initial_aa=initial_aa,
            comparisons=comparisons,
            final_aa=final_aa,
            initial_to_final_aa_abs_relative_drift=drift,
            aa_drift_threshold=0.03,
            inconclusive_due_to_drift=(
                drift > 0.03
                or initial_aa["inconclusive_aa_drift"]
                or final_aa["inconclusive_aa_drift"]
                or any(v["inconclusive_aa_drift"] for v in comparisons.values())
            ),
        )
        result["modes"][mode] = record
        append_jsonl(
            args.run_dir / "measurements.jsonl",
            dict(
                kind="timing",
                batch_size=batch_size,
                length=length,
                layout=layout,
                mode=mode,
                **record,
            ),
        )
        print(
            json.dumps(
                dict(
                    batch=batch_size,
                    length=length,
                    layout=layout,
                    mode=mode,
                    ratios={
                        k: v["paired_block_median_B_over_A"]
                        for k, v in comparisons.items()
                    },
                    aa_drift=drift,
                    inconclusive=record["inconclusive_due_to_drift"],
                )
            ),
            flush=True,
        )
    restore()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("tiny", "default"), default="default")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("plain", "packed", "padded"),
        default=["plain", "packed", "padded"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("fb", "window"), default=["fb", "window"]
    )
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="muon")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--muon-lr", type=float, default=4.333333333333333e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--token-baseline", action="store_true")
    parser.add_argument("--mean-doc-length", type=float, default=200)
    parser.add_argument("--pad-fraction", type=float, default=0.25)
    parser.add_argument("--cache-limit-gb", type=float, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--block-iters", type=int, default=5)
    parser.add_argument("--aa-blocks", type=int, default=1)
    parser.add_argument("--aa-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if (
        min(
            *args.batches,
            *args.lengths,
            args.warmup,
            args.blocks,
            args.block_iters,
            args.aa_blocks,
            args.aa_iters,
        )
        < 1
        or not 0 <= args.pad_fraction < 0.75
    ):
        parser.error("positive shapes/counts and 0 <= pad_fraction < 0.75 are required")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.muon_lr)
        or args.muon_lr <= 0
        or not math.isfinite(args.mean_doc_length)
        or args.mean_doc_length <= 0
    ):
        parser.error(
            "learning rate and mean document length must be finite and positive"
        )
    args.run_dir.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    metadata = dict(
        kind="protocol",
        command=sys.argv,
        mlx_version=mx.__version__,
        platform=platform.platform(),
        device=mx.device_info(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256=before,
        flags=active_viby_flags(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        variant_switches={
            "reference": {
                "attention._RECURRENT_OPTIMIZED": False,
                "moe._RECURRENT_MASKED_COUNTS": False,
                "ced_recurrent_enabled": True,
            },
            "optimized": {
                "attention._RECURRENT_OPTIMIZED": True,
                "moe._RECURRENT_MASKED_COUNTS": True,
                "ced_recurrent_enabled": True,
            },
            "token_baseline": {
                "attention._RECURRENT_OPTIMIZED": False,
                "moe._RECURRENT_MASKED_COUNTS": False,
                "ced_recurrent_enabled": False,
            },
            "attention._RECURRENT_SPARSE": getattr(
                attention, "_RECURRENT_SPARSE", None
            ),
        },
        scope="Same process and weights; explicit current reference/optimized recurrent switch. MTP, Engram disabled in both. f+b returns/evaluates MoE loads and QB margins. Window includes two real BaseTrainer microbatches, gradient accumulation, norm/clip, selected optimizer and next-window MoE bias update. Snapshot restore and data generation excluded; no data loader, checkpoint I/O, generation cache, efficacy or measured FLOPs claim.",
    )
    append_jsonl(args.run_dir / "measurements.jsonl", metadata)
    (args.run_dir / "working_tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    )
    mx.random.seed(args.seed)
    cfg = make_config(args)
    model = VibyForCausalLM(cfg)
    convert_model_dtype(model, args.dtype)
    model.train()
    mx.eval(model.parameters())
    trainer = make_trainer(args, cfg, model)
    configure("reference", cfg, trainer)
    # One untimed reference window materializes optimizer buffers. All later
    # arms use its exact resulting state, with fixed LR and no schedule calls.
    warm_batches, _ = make_batches(
        cfg,
        args.batches[0],
        args.lengths[0],
        args.layouts[0],
        args.seed,
        args.mean_doc_length,
        args.pad_fraction,
    )
    print(
        "materializing shared optimizer snapshot with one untimed reference window",
        flush=True,
    )
    run_window(trainer, warm_batches)
    mx.eval(model.parameters(), trainer.optimizer.state, model.moe_bias_stack())
    snapshot = snapshot_train_state(model, trainer.optimizer, trainer)
    metadata.update(
        resolved_config=cfg.to_dict(),
        resolved_trainer_args={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(trainer.args).items()
        },
        params=param_identity(model),
        optimizer_groups=optimizer_metadata(trainer.optimizer),
        snapshot_origin="one untimed two-microbatch reference optimizer window from seeded initialization",
        snapshot_memory=memory_snapshot(),
    )
    results = []
    for batch in args.batches:
        for length in args.lengths:
            for layout in args.layouts:
                print(f"case B={batch} T={length} layout={layout}", flush=True)
                results.append(
                    measure_case(
                        args,
                        cfg,
                        model,
                        trainer,
                        snapshot,
                        batch,
                        length,
                        layout,
                        args.seed + len(results) * 11,
                    )
                )
                report = dict(
                    protocol=metadata,
                    results=results,
                    source_unchanged_during_measurement=before == source_hashes(),
                )
                (args.run_dir / "results.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
                )
    print(str(args.run_dir / "results.json"), flush=True)


if __name__ == "__main__":
    main()
