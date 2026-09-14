"""Same-weight MLX timing and structural pointer-pool diagnostic.

This does not train a language model or establish reasoning quality. Timing
includes the actual forward graph / weight gradients, never estimated FLOPs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.utils import convert_model_dtype


def pointer_pool_diagnostic(seed, nodes=4096, tasks=512, budget=64, hops=(1, 2, 4, 8)):
    """Oracle retrieves random pointers only when the key is in a frozen pool.

    Fresh permutations and independent random terminal labels preclude solving
    tasks by memorizing labels. The pool always contains the initial key, but is
    otherwise sampled independently of later pointers. This measures a hard
    structural limitation, NOT Viby's trained candidate selector or task score.
    """
    if not 1 <= budget <= nodes or max(hops) >= nodes:
        raise ValueError("require 1 <= budget <= nodes and hops < nodes")
    rng = np.random.default_rng(seed)
    success = {h: 0 for h in hops}
    for _ in range(tasks):
        order = rng.permutation(nodes)
        pointers = np.empty(nodes, dtype=np.int64)
        pointers[order] = np.roll(order, -1)
        labels = rng.integers(0, 2**31, size=nodes)
        first = int(order[0])
        other = np.delete(np.arange(nodes), first)
        pool = {first, *rng.choice(other, budget - 1, replace=False).tolist()}
        for hops_count in hops:
            current, reachable = first, True
            for _ in range(hops_count):
                reachable &= current in pool
                current = int(pointers[current])
            # A terminal label also has to be read, rather than guessed.
            reachable &= current in pool
            expected = int(labels[current])
            observed = int(labels[current]) if reachable else None
            success[hops_count] += observed == expected
    return {
        "kind": "structural_oracle_with_random_fixed_pool_not_trained_model",
        "nodes": nodes,
        "tasks": tasks,
        "fixed_pool_size": budget,
        "full_memory_oracle_accuracy": 1.0,
        "fixed_pool_exact_chain_coverage": {
            str(h): n / tasks for h, n in success.items()
        },
        "limitation": "Does not evaluate learned routing or reasoning. Missing pointer keys cannot be recovered by additional rounds restricted to this pool.",
    }


def make_config(args):
    settings = dict(
        n_mtp_layers=0,
        ced_recurrent_enabled=True,
        ced_recurrent_stride=args.stride,
        ced_recurrent_rounds=args.rounds,
        engram_layer_ids=(),
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


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark(args):
    mx.random.seed(args.seed)
    c = make_config(args)
    model = VibyForCausalLM(c)
    convert_model_dtype(model, args.dtype)
    model.train()
    mx.eval(model.parameters())
    records = []
    for length in args.lengths:
        x = mx.random.randint(1, c.vocab_size, (args.batch_size, length))
        y = mx.random.randint(1, c.vocab_size, x.shape)
        mx.eval(x, y)
        params = model.trainable_parameters()
        biases = model.moe_bias_stack()
        calls = {}
        for recurrent in (False, True):

            def loss(weights, router_biases, ids, labels, enabled=recurrent):
                model.update(weights)
                model.apply_moe_biases(router_biases)
                result = model(
                    ids, labels=labels, use_mtp=False, use_ced_recurrent=enabled
                )
                # Returning MoE side outputs prevents compiled training telemetry
                # from being dead-code eliminated. No optimizer step is taken.
                auxiliary = (result.moe_loads, result.moe_qb_margins)
                return result.loss, auxiliary

            fwd = loss
            bwd = mx.value_and_grad(loss)
            if args.compile:
                fwd = mx.compile(fwd)
                bwd = mx.compile(bwd)
            prefix = "recurrent" if recurrent else "baseline"
            calls[prefix + "_forward"] = fwd
            calls[prefix + "_forward_backward"] = bwd

        def evaluate(fn):
            outputs = fn(params, biases, x, y)
            model.update(params)
            model.apply_moe_biases(biases)
            mx.eval(outputs)

        for fn in calls.values():
            for _ in range(args.warmup):
                evaluate(fn)
        samples = {name: [] for name in calls}
        # Alternate baseline/recurrent in ABBA order, for each measured graph.
        for _ in range(args.steps):
            for phase in ("forward", "forward_backward"):
                for variant in ("baseline", "recurrent", "recurrent", "baseline"):
                    key = variant + "_" + phase
                    start = time.perf_counter()
                    evaluate(calls[key])
                    samples[key].append(1000 * (time.perf_counter() - start))
        values = {name: summarize(value) for name, value in samples.items()}
        middle = c.n_layers - c.n_encoder_layers - 2
        original = middle * args.batch_size * length
        proposed = middle * args.rounds * args.batch_size * (length // args.stride)
        for phase in ("forward", "forward_backward"):
            values[phase + "_recurrent_over_baseline"] = (
                values["recurrent_" + phase]["median_ms"]
                / values["baseline_" + phase]["median_ms"]
            )
        records.append(
            {
                "length": length,
                "batch_size": args.batch_size,
                "logical_middle_query_block_updates": {
                    "baseline": original,
                    "recurrent": proposed,
                    "ratio": proposed / original,
                },
                "timing": values,
            }
        )
    return {
        "config": c.to_dict(),
        "measurements": records,
        "scope": "Same weights, same token/label arrays, same process; MTP and Engram disabled in both. Includes evaluated forward loss / all weight gradients, router telemetry, masks and lifting. Excludes optimizer update, data loading, distributed reduction, generation cache and compilation warmup. This is not a training-throughput or efficacy result.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("tiny", "default"), default="tiny")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--pointer-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.lengths) < 1
        or args.steps < 1
        or args.warmup < 1
        or args.batch_size < 1
    ):
        parser.error("lengths, batch size, steps and warmup must be positive")
    root = Path(__file__).resolve().parents[1]

    def source_hashes():
        files = [
            *sorted((root / "model").rglob("*.py")),
            root / "trainer/utils.py",
            root / "trainer/flops.py",
            Path(__file__).resolve(),
        ]
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        }

    before = source_hashes()
    result = {
        "seed": args.seed,
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "dtype": args.dtype,
        "device": mx.device_info(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": before,
        "compiled": args.compile,
        "pointer_pool": pointer_pool_diagnostic(args.seed),
    }
    if not args.pointer_only:
        result["benchmark"] = benchmark(args)
    result["source_unchanged_during_measurement"] = before == source_hashes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pointer_pool": result["pointer_pool"],
                "measurements": result.get("benchmark", {}).get("measurements"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
