"""Same-model inference ABBA, fixed prompt/tokens and restored decode caches."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx.utils import tree_map
from experiments.kernel_bench_utils import (
    abba_blocks,
    append_jsonl,
    input_identity,
    param_identity,
)
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model import moe
from model.kernels import decode_metadata as dm, indexer_select as fs, moe_dispatch
from model.kernels import hc_decode, moe_decode, rope_decode
from model.kernels import sinkhorn_fused as sinkhorn
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import build_model_kwargs, convert_model_dtype


def configure(name):
    complete_decode = name in (
        "simd_decode",
        "hc_decode",
        "rope_decode",
        "fused_decode",
    )
    combined = (
        name in ("combined", "compiled_moe", "simd_metadata", "simd_sinkhorn")
        or complete_decode
    )
    dm._ENABLED = name == "metadata" or combined
    moe._DECODE_GATHER = name == "experts" or combined
    dm._PREFILL_SELECT = fs._ENABLED = fs._BQ_ENABLED = combined
    moe_decode._ENABLED = name == "compiled_moe" or complete_decode
    dm._SIMD_POST = name == "simd_metadata" or complete_decode
    sinkhorn._SIMD_DECODE = name == "simd_sinkhorn" or complete_decode
    hc_decode._ENABLED = name in ("hc_decode", "fused_decode")
    rope_decode._ENABLED = name in ("rope_decode", "fused_decode")
    # Use the validated deterministic combine in both arms; this prevents
    # unrelated native bf16 scatter arrival-order noise in prefix activations.
    moe_dispatch._COMBINE_ENABLED = True


_FIELDS = ("window", "compress_kv", "index_k", "kv_state", "score_state", "filled")
_CACHE_FIELDS = (
    "start_pos",
    "decode_max_pos",
    "engram_prev",
    "thinking_state",
    "psr_mode",
    "psr_next_anchor",
    "psr_phases",
    "psr_options",
    "psr_gate",
)


def copy_value(x):
    if dataclasses.is_dataclass(x):
        return dataclasses.replace(
            x, **{f.name: copy_value(getattr(x, f.name)) for f in dataclasses.fields(x)}
        )
    return tree_map(lambda a: mx.array(a) if isinstance(a, mx.array) else a, x)


def snapshot(cache):
    return (
        [
            {key: copy_value(getattr(layer, key)) for key in _FIELDS}
            for layer in cache.layers
        ],
        {key: copy_value(getattr(cache, key)) for key in _CACHE_FIELDS},
    )


def restore(cache, saved):
    for layer, state in zip(cache.layers, saved[0]):
        for key, value in state.items():
            setattr(layer, key, copy_value(value))
    for key, value in saved[1].items():
        setattr(cache, key, copy_value(value))


def eval_cache(cache):
    arrays = []
    for layer in cache.layers:
        for key in _FIELDS:
            value = getattr(layer, key)
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, tuple):
                arrays.extend(a for a in value if isinstance(a, mx.array))
    if cache.engram_prev is not None:
        arrays.append(cache.engram_prev)
    if cache.thinking_state is not None:
        arrays.extend(
            getattr(cache.thinking_state, key)
            for key in ("slots", "anchor", "segment", "valid")
            if getattr(cache.thinking_state, key) is not None
        )
    mx.eval(arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mode", choices=["both", "prefill", "decode"], default="both")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-iters", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument(
        "--reference",
        choices=["reference", "experts", "combined", "simd_decode", "hc_decode"],
        default="reference",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "metadata",
            "experts",
            "combined",
            "compiled_moe",
            "simd_metadata",
            "simd_sinkhorn",
            "simd_decode",
            "hc_decode",
            "rope_decode",
            "fused_decode",
        ],
        default=["metadata", "experts", "combined"],
    )
    args = ap.parse_args()
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(8 * 1024**3)
    mx.random.seed(1234)
    configure(args.reference)
    ta = setup_training_args(
        get_pretrain_parser().parse_args(
            [
                "--out_dir",
                "research_runs/_bench",
                "--no_save",
                "--dtype",
                "bfloat16",
                "--max_seq_len",
                str(max(2048, args.context + args.steps)),
            ]
        ),
        "pretrain",
    )
    cfg = VibyConfig(**build_model_kwargs(ta))
    model = VibyForCausalLM(cfg)
    convert_model_dtype(model, "bfloat16")
    model.eval()
    mx.eval(model.parameters())
    mx.random.seed(20260911)
    tokens = mx.random.randint(
        0, cfg.vocab_size, (args.batch, args.context + args.steps)
    )
    mx.eval(tokens)
    path = Path(args.run_dir) / "results.jsonl"
    protocol = dict(
        kind="protocol",
        args=vars(args),
        params=param_identity(model),
        mlx=mx.__version__,
        input_hash=input_identity(tokens),
        dtype="bfloat16",
        execution="eager public prefill/decode APIs",
        native_expert_gemm=True,
        deterministic_prefill_combine_both_arms=True,
        psr_enabled=cfg.psr_enabled,
        psr_horizon=cfg.psr_horizon,
        resolved_config=cfg.to_dict(),
        compiled_moe_variants=[
            n
            for n in args.variants
            if n
            in (
                "compiled_moe",
                "simd_decode",
                "hc_decode",
                "rope_decode",
                "fused_decode",
            )
        ],
    )
    append_jsonl(path, protocol)
    print(json.dumps(protocol), flush=True)

    def prefill(name):
        configure(name)
        logits, cache = model.prefill(tokens[:, : args.context])
        mx.eval(logits)
        eval_cache(cache)
        return logits, cache

    ref_logits, cache = prefill(args.reference)
    frozen = snapshot(cache)
    opt_logits, _ = prefill("combined")
    error = mx.max(
        mx.abs(opt_logits.astype(mx.float32) - ref_logits.astype(mx.float32))
    )
    print("prefill max logit difference", float(error), flush=True)
    append_jsonl(path, dict(kind="numerical", phase="prefill", max_abs=float(error)))

    if args.mode in ("both", "prefill"):
        print("ABBA prefill", flush=True)
        result = abba_blocks(
            lambda: prefill(args.reference),
            lambda: prefill("combined"),
            args.warmup,
            args.block_iters,
            args.blocks,
        )
        append_jsonl(
            path, dict(kind="timing", phase="prefill", variant="combined", **result)
        )
        print(
            "prefill",
            result["paired_block_median_B_over_A"],
            "drift",
            result["aa_end_over_start_abs_rel"],
            flush=True,
        )

    def decode(name):
        configure(name)
        outputs = []
        for i in range(args.steps):
            logits, _ = model.decode_step(tokens[:, args.context + i], cache)
            mx.eval(logits)
            outputs.append(logits)
        eval_cache(cache)
        return mx.stack(outputs)

    if args.mode in ("both", "decode"):
        restore(cache, frozen)
        reference = decode(args.reference)
        mx.eval(reference)
        for name in args.variants:
            restore(cache, frozen)
            output = decode(name)
            delta = output.astype(mx.float32) - reference.astype(mx.float32)
            mx.eval(delta)
            record = dict(
                kind="numerical",
                phase="decode",
                variant=name,
                max_abs=float(mx.max(mx.abs(delta))),
                relative_l2=float(
                    mx.sqrt(
                        mx.sum(delta * delta)
                        / mx.sum(reference.astype(mx.float32) ** 2)
                    )
                ),
            )
            append_jsonl(path, record)
            print(json.dumps(record), flush=True)
            if (
                name
                in (
                    "compiled_moe",
                    "simd_metadata",
                    "simd_sinkhorn",
                    "simd_decode",
                    "hc_decode",
                    "rope_decode",
                    "fused_decode",
                )
                and record["max_abs"] != 0.0
            ):
                raise RuntimeError(
                    "decode optimization changed logits; refusing a speedup measurement"
                )
            print("ABBA decode", name, flush=True)
            result = abba_blocks(
                lambda: decode(args.reference),
                lambda: decode(name),
                args.warmup,
                args.block_iters,
                args.blocks,
                before_a=lambda: restore(cache, frozen),
                before_b=lambda: restore(cache, frozen),
            )
            append_jsonl(
                path, dict(kind="timing", phase="decode", variant=name, **result)
            )
            print(
                name,
                result["paired_block_median_B_over_A"],
                "drift",
                result["aa_end_over_start_abs_rel"],
                flush=True,
            )


if __name__ == "__main__":
    main()
