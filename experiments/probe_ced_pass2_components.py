"""Small compiled component screens for the second CED performance pass.

Metadata: integer validator/compactor only. HC: coefficient split + Sinkhorn,
with FP32 mixes and actual BF16 scale/base leaves, forward and VJP. QB: FP32
router projection + selected weights/indices/scores and returned load/margins.
These are not model-throughput or quality measurements. No GPU work on import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bench_ced_pass2 import GPUExclusivityGuard


def run(args, guard):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from model.config import VibyConfig
    import model.attention as attention
    import model.moe as moe
    from model import hc
    from model.kernels import hc_train, recurrent_metadata
    from experiments.kernel_bench_utils import (
        abba_blocks,
        append_jsonl,
        active_viby_flags,
    )

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(args.seed)

    def hashes():
        files = [
            *sorted((ROOT / "model").rglob("*.py")),
            Path(__file__).resolve(),
            ROOT / "experiments/bench_ced_pass2.py",
            ROOT / "experiments/kernel_bench_utils.py",
        ]
        return {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        }

    before = hashes()
    protocol = dict(
        command=sys.argv,
        args=vars(args) | {"run_dir": str(args.run_dir)},
        mlx_version=mx.__version__,
        device=mx.device_info(),
        platform=platform.platform(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256=before,
        environment=active_viby_flags(),
        compiled=True,
        scope="Component screens with explicit inputs, fixed weights/no optimizer, evaluated returned outputs. Timing excludes process scans before samples; background CPU process monitoring runs in both arms. Not end-to-end speed, MFU, efficacy, or a training-window result.",
    )
    append_jsonl(args.run_dir / "measurements.jsonl", dict(kind="protocol", **protocol))
    results = []

    def random(shape, dtype=mx.float32):
        return mx.array(rng.normal(size=shape).astype(np.float32), dtype=dtype)

    def array_hash(arrays):
        digest = hashlib.sha256()
        for _, array in tree_flatten(arrays):
            digest.update(str((array.shape, array.dtype)).encode())
            # Conversion describes the exact representable values, including
            # BF16 leaves; all copies happen outside the measured functions.
            host = np.asarray(array.astype(mx.float32))
            digest.update(host.tobytes())
        return digest.hexdigest()

    def compare(left, right, exact, rtol, atol, low_precision_tolerance=None):
        la, lb = tree_flatten(left), tree_flatten(right)
        if [k for k, _ in la] != [k for k, _ in lb]:
            return dict(passed=False, same_tree=False)
        leaves = []
        for (name, a), (_, b) in zip(la, lb):
            if a.shape != b.shape or a.dtype != b.dtype:
                return dict(
                    passed=False,
                    same_tree=True,
                    mismatched_leaf=name,
                    shapes=[list(a.shape), list(b.shape)],
                    dtypes=[str(a.dtype), str(b.dtype)],
                )
            leaf_rtol, leaf_atol = rtol, atol
            if low_precision_tolerance is not None and a.dtype in (
                mx.bfloat16,
                mx.float16,
            ):
                leaf_rtol, leaf_atol = low_precision_tolerance
            a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
            difference = a32 - b32
            finite = bool(mx.all(mx.isfinite(a32)) & mx.all(mx.isfinite(b32)))
            identical = bool(mx.array_equal(a, b))
            close = bool(
                mx.all(mx.abs(difference) <= leaf_atol + leaf_rtol * mx.abs(a32))
            )
            leaves.append(
                dict(
                    path=name,
                    shape=list(a.shape),
                    dtype=str(a.dtype),
                    finite=finite,
                    exact=identical,
                    close=close,
                    rtol=leaf_rtol,
                    atol=leaf_atol,
                    max_abs=float(mx.max(mx.abs(difference))) if a.size else 0.0,
                    relative_l2=float(
                        mx.sqrt(
                            mx.sum(difference * difference)
                            / mx.maximum(mx.sum(a32 * a32), 1e-30)
                        )
                    )
                    if a.size
                    else 0.0,
                )
            )
        return dict(
            passed=all(
                x["finite"] and (x["exact"] if exact else x["close"]) for x in leaves
            ),
            exact_required=exact,
            rtol=rtol,
            atol=atol,
            leaves=leaves,
        )

    def benchmark(
        label,
        functions,
        inputs,
        *,
        shape,
        exact=False,
        rtol=2e-6,
        atol=2e-7,
        low_precision_tolerance=None,
        restore=None,
        flags=None,
    ):
        mx.eval(inputs)
        guard.scan(label + ":numerical")
        calls = {arm: mx.compile(fn) for arm, fn in functions.items()}

        def invoke(arm):
            if flags is not None:
                flags(arm == "candidate")
            result = calls[arm](*inputs)
            if restore is not None:
                restore()
            mx.eval(result)
            guard.raise_if_conflicted()
            return result

        reference, candidate = invoke("reference"), invoke("candidate")
        numeric = compare(
            reference, candidate, exact, rtol, atol, low_precision_tolerance
        )
        record = dict(
            label=label, shape=shape, input_sha256=array_hash(inputs), numerical=numeric
        )
        append_jsonl(
            args.run_dir / "measurements.jsonl", dict(kind="numerical", **record)
        )
        del reference, candidate
        if not numeric["passed"]:
            raise RuntimeError(
                f"component numerical comparison failed for {label}; see measurements.jsonl"
            )

        def measured(arm):
            def f():
                guard.scope = label + ":sample_" + arm
                try:
                    invoke(arm)
                finally:
                    guard.scope = label + ":between_samples"

            return f

        def prepare():
            guard.scan(label + ":prepare")

        def abba(a, b, *, initial=False, final=False):
            return abba_blocks(
                measured(a),
                measured(b),
                warmup=0 if final else args.warmup,
                block_iters=args.aa_iters if initial or final else args.block_iters,
                n_blocks=args.aa_blocks if initial or final else args.blocks,
                label_a="reference_A1" if initial or final else "reference",
                label_b="reference_A2" if initial or final else "candidate",
                before_a=prepare,
                before_b=prepare,
            )

        initial_aa = abba("reference", "reference", initial=True)
        timing = abba("reference", "candidate")
        final_aa = abba("reference", "reference", final=True)
        aa_first = np.median(initial_aa["A"]["raw_s"] + initial_aa["B"]["raw_s"])
        aa_last = np.median(final_aa["A"]["raw_s"] + final_aa["B"]["raw_s"])
        drift = float(abs(aa_last / aa_first - 1))
        ratio = timing["paired_block_median_B_over_A"]
        record.update(
            initial_aa=initial_aa,
            timing=timing,
            final_aa=final_aa,
            initial_to_final_aa_abs_relative_drift=drift,
            aa_drift_threshold=0.03,
            inconclusive_due_to_drift=(
                drift > 0.03
                or any(
                    x["inconclusive_aa_drift"] for x in (initial_aa, timing, final_aa)
                )
            ),
            paired_time_reduction_fraction=1 - ratio,
            paired_throughput_increase_fraction=1 / ratio - 1,
        )
        results.append(record)
        guard.scan(label + ":complete")
        append_jsonl(args.run_dir / "measurements.jsonl", dict(kind="timing", **record))
        report = dict(
            protocol=protocol,
            results=results,
            gpu_exclusivity=guard.report(),
            source_unchanged_during_measurement=before == hashes(),
        )
        (args.run_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    label=label,
                    B_over_A=ratio,
                    aa_drift=drift,
                    inconclusive=record["inconclusive_due_to_drift"],
                )
            ),
            flush=True,
        )

    for query_count in args.queries:
        if "metadata" in args.components:
            n, k, q = args.memory_length, args.topk, query_count
            # The model normally carries already causal sorted Top-K routes.
            # Use that occupancy for timing instead of mostly invalid random
            # keys, which could make compaction look artificially inexpensive.
            qp = ((np.arange(q, dtype=np.int32) + 1) * n // q - 1)[None]
            mp = np.arange(n, dtype=np.int32)[None]
            qs, ms = np.zeros_like(qp), np.zeros_like(mp)
            qpad, mpad = np.ones((1, q), bool), np.ones((1, n), bool)
            qpad[:, -3:] = False
            selected = np.full((1, q, k), -1, dtype=np.int32)
            for row in range(q):
                available = int(qp[0, row]) + 1
                count = min(k, available)
                if qpad[0, row] and count:
                    selected[0, row, :count] = np.sort(
                        rng.choice(available, count, replace=False)
                    )
            arrays = tuple(mx.array(a) for a in (selected, qp, mp, qs, ms, qpad, mpad))

            def metadata_reference(selected, qp, mp, qs, ms, qpad, mpad):
                safe = mx.clip(selected, 0, mp.shape[1] - 1)
                kp = mx.take_along_axis(mp[:, None, :], safe, axis=2)
                ks = mx.take_along_axis(ms[:, None, :], safe, axis=2)
                km = mx.take_along_axis(mpad[:, None, :], safe, axis=2)
                good = (
                    (selected >= 0)
                    & (selected < mp.shape[1])
                    & (kp <= qp[..., None])
                    & (ks == qs[..., None])
                    & km
                    & qpad[..., None]
                )
                clean = mx.where(good, selected, -1).astype(mx.int32)
                ordered = mx.sort(mx.where(clean >= 0, clean, mp.shape[1]), axis=-1)
                lengths = (
                    mx.sum(ordered < mp.shape[1], axis=-1).astype(mx.int32).reshape(-1)
                )
                compact = mx.where(ordered < mp.shape[1], ordered, -1).reshape(
                    -1, selected.shape[-1]
                )
                return clean, (compact, lengths)

            benchmark(
                f"metadata_Q{q}",
                dict(
                    reference=metadata_reference,
                    candidate=recurrent_metadata.validate_compact,
                ),
                arrays,
                shape=dict(
                    B=1,
                    Q=q,
                    N=n,
                    K=k,
                    layout="single_document_sorted_causal_routes_with_tail_padding",
                ),
                exact=True,
                flags=lambda enabled: setattr(
                    attention, "_RECURRENT_METADATA_FUSED", enabled
                ),
            )
        if "hc" in args.components:
            arrays = (
                random((1, query_count, 24)),
                mx.array([0.31, -0.77, 1.41], mx.bfloat16),
                random((24,), mx.bfloat16),
            )
            cotangents = (
                random((1, query_count, 4)),
                random((1, query_count, 4)),
                random((1, query_count, 4, 4)),
            )

            def hc_reference(mixes, scale, bias):
                pre, post, comb = hc.hc_split(mixes, scale, bias, 4, args.hc_eps)
                return (
                    pre,
                    post,
                    hc.sinkhorn(
                        comb.reshape(*mixes.shape[:-1], 4, 4),
                        args.hc_iters,
                        args.hc_eps,
                    ),
                )

            def hc_candidate(mixes, scale, bias):
                return hc_train.split_sinkhorn_train(
                    mixes, scale, bias, args.hc_iters, args.hc_eps
                )

            def differentiated(fn):
                def f(mixes, scale, bias, cpre, cpost, ccomb):
                    return mx.vjp(fn, [mixes, scale, bias], [cpre, cpost, ccomb])

                return f

            def flags(enabled):
                return setattr(hc_train, "_TRAIN_FUSION", enabled)
            shape = dict(
                B=1,
                T=query_count,
                mixes_dtype="float32",
                scale_base_dtype="bfloat16",
                hc_mult=4,
                iters=args.hc_iters,
                eps=args.hc_eps,
            )
            benchmark(
                f"hc_forward_N{query_count}",
                dict(reference=hc_reference, candidate=hc_candidate),
                arrays,
                shape=shape,
                flags=flags,
            )
            benchmark(
                f"hc_forward_backward_N{query_count}",
                dict(
                    reference=differentiated(hc_reference),
                    candidate=differentiated(hc_candidate),
                ),
                (*arrays, *cotangents),
                shape=shape,
                rtol=5e-5,
                atol=1e-5,
                low_precision_tolerance=(0.009, 0.001),
                flags=flags,
            )
        if "qb" in args.components:
            cfg = VibyConfig(
                dim=1024,
                n_routed_experts=96,
                n_activated_experts=6,
                n_mtp_layers=0,
                engram_layer_ids=(),
                router_fp32=True,
                moe_balance_method="qb",
            )
            mx.random.seed(args.seed)
            gate = moe.MoEGate(cfg)
            gate.train()
            gate.qb_stats_rows = 1365  # floor(8192 / 2 microbatches / 3 rounds)
            inputs = random((query_count, 1024), mx.bfloat16)
            weights, biases = gate.weight.astype(mx.float32), random((96,)) * 0.01

            def qb_graph(enabled):
                def f(x, w, b):
                    moe._QB_THRESHOLD_REUSE = enabled
                    gate.weight, gate.bias = w, b
                    scores, indices, dense = gate(x)
                    return (
                        scores,
                        indices,
                        dense,
                        gate._last_load,
                        gate._last_qb_margins,
                    )

                return f

            def restore():
                gate.weight, gate.bias = weights, biases

            benchmark(
                f"qb_forward_Q{query_count}",
                dict(reference=qb_graph(False), candidate=qb_graph(True)),
                (inputs, weights, biases),
                shape=dict(
                    Q=query_count,
                    D=1024,
                    E=96,
                    K=6,
                    router_fp32=True,
                    input_dtype="bfloat16",
                    weight_dtype="float32",
                    sampled_rows=min(query_count, gate.qb_stats_rows),
                ),
                exact=True,
                restore=restore,
                flags=lambda enabled: setattr(moe, "_QB_THRESHOLD_REUSE", enabled),
            )
    print(str(args.run_dir / "results.json"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--components",
        choices=("metadata", "hc", "qb"),
        nargs="+",
        default=["metadata", "hc", "qb"],
    )
    p.add_argument("--queries", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--memory-length", type=int, default=1024)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--hc-iters", type=int, default=20)
    p.add_argument("--hc-eps", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--block-iters", type=int, default=20)
    p.add_argument("--aa-blocks", type=int, default=1)
    p.add_argument("--aa-iters", type=int, default=5)
    p.add_argument("--poll-seconds", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260912)
    args = p.parse_args()
    if min(
        *args.queries,
        args.memory_length,
        args.topk,
        args.warmup,
        args.blocks,
        args.block_iters,
        args.aa_blocks,
        args.aa_iters,
    ) < 1 or args.topk > min(256, args.memory_length):
        p.error("positive shapes/counts and topk <= min(256, memory_length) required")
    if (
        not math.isfinite(args.hc_eps)
        or args.hc_eps <= 0
        or not 0.1 <= args.poll_seconds <= 5
    ):
        p.error("finite positive hc_eps and 0.1 <= poll_seconds <= 5 required")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    guard = GPUExclusivityGuard(args.run_dir, args.poll_seconds)
    try:
        with guard.hold():
            run(args, guard)
    except BaseException as error:
        (args.run_dir / "failure.json").write_text(
            json.dumps(
                dict(
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc(),
                    gpu_exclusivity=guard.report(),
                ),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
