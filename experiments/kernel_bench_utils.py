"""Kernel 优化共用计时 / 快照 / JSONL 口径。

现行测量协议见 research/EXPERIMENT_PROTOCOL.md「正确性与性能」：
warmup ≥3；A,B,B,A block × ≥3 组 × 每槽 5 次；主指标用配对 block 中位数；
A/A 首尾漂移 >3% 本轮无结论；f+b 不更新权重；整窗口从同一快照恢复。
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

DEFAULT_VIBY_FLAGS = {
    "VIBY_SPARSE_ATTN_KERNEL": "1",
    "VIBY_SPARSE_ATTN_KEY_TILE": "16",
    "VIBY_SPARSE_TOPK_KERNEL": "1",
    "VIBY_SPARSE_ATTN_BWD_SPLIT": "1",
    "VIBY_SPARSE_ATTN_KEY_BWD": "0",
    "VIBY_SPARSE_ATTN_FUSED_BWD": "1",
    "VIBY_SPARSE_ATTN_FUSED_BWD_TILE": "32",
    "VIBY_INDEXER_KERNEL": "1",
    "VIBY_INDEXER_SELECT": "1",
    "VIBY_INDEXER_BQ": "1",
    "VIBY_MOE_KERNEL": "0",
    "VIBY_MOE_COMBINE_KERNEL": "1",
    "VIBY_HC_GROUPED_DW": "1",
    "VIBY_DECODE_METADATA": "1",
    "VIBY_PREFILL_SELECT": "1",
    "VIBY_MOE_DECODE_GATHER": "1",
    "VIBY_MOE_DECODE_COMPILE": "1",
    "VIBY_DECODE_SIMD_POST": "1",
    "VIBY_SINKHORN_SIMD_DECODE": "1",
    "VIBY_HC_KERNEL": "1",
    "VIBY_HC_PRE_NORM_KERNEL": "1",
    "VIBY_SINKHORN_KERNEL": "1",
    "VIBY_OPT_SINKHORN_FAST_NORM": "1",
}


def apply_default_flags(extra=None):
    for key, value in DEFAULT_VIBY_FLAGS.items():
        os.environ.setdefault(key, value)
    if extra:
        os.environ.update(extra)
    return active_viby_flags()


def active_viby_flags():
    return {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("VIBY_")}


def summarize_times(samples):
    samples = [float(x) for x in samples]
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    n = len(ordered)
    p90_i = min(n - 1, max(0, int(round(0.9 * (n - 1)))))
    return {
        "n": n,
        "raw_s": samples,
        "min_s": ordered[0],
        "median_s": statistics.median(ordered),
        "p90_s": ordered[p90_i],
        "mean_s": statistics.fmean(ordered),
    }


def memory_snapshot():
    peak = int(mx.get_peak_memory())
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    return {
        "peak_bytes": peak,
        "active_bytes": active,
        "cache_bytes": cache,
        "peak_gb": peak / 1e9,
        "peak_gib": peak / 1024**3,
        "active_gb": active / 1e9,
        "active_gib": active / 1024**3,
        "cache_gb": cache / 1e9,
        "cache_gib": cache / 1024**3,
    }


def input_identity(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        if arr is None:
            h.update(b"none")
            continue
        h.update(str(tuple(arr.shape)).encode())
        h.update(str(arr.dtype).encode())
        sl = arr.reshape(-1)[:64].astype(mx.float32)
        mx.eval(sl)
        h.update(str(sl.tolist()).encode())
    return h.hexdigest()[:16]


def clone_tree(tree):
    """冻结一份数组树。MLX 数组不可变，optimizer.update 是换叶子，引用即可。"""
    return tree_map(lambda x: x, tree)


def snapshot_train_state(model, optimizer, trainer=None):
    biases = None
    if hasattr(model, "moe_bias_stack"):
        biases = model.moe_bias_stack()
    return {
        "params": clone_tree(model.parameters()),
        "trainable": clone_tree(model.trainable_parameters()),
        "optimizer": clone_tree(optimizer.state),
        "biases": None if biases is None else biases,
        "side_optimizer": clone_tree(optimizer.psr_optimizer.state)
        if getattr(optimizer, "psr_optimizer", None) is not None
        else None,
        "psr_step": getattr(trainer, "_psr_step", None),
        "en_delta": clone_tree(getattr(trainer, "_en_delta", None)),
    }


def restore_train_state(model, optimizer, snap, trainer=None):
    model.update(snap["params"])
    # Optimizers mutate state dictionaries in place, even though array leaves
    # are immutable. Hand back fresh containers on EVERY restore.
    optimizer.state = clone_tree(snap["optimizer"])
    side = getattr(optimizer, "psr_optimizer", None)
    if side is not None and snap.get("side_optimizer") is not None:
        side.state = clone_tree(snap["side_optimizer"])
    if trainer is not None:
        if snap.get("psr_step") is not None:
            trainer._psr_step = snap["psr_step"]
            trainer.args.psr_microstep = snap["psr_step"]
        trainer._en_delta = clone_tree(snap.get("en_delta"))
    if snap["biases"] is not None and hasattr(model, "apply_moe_biases"):
        model.apply_moe_biases(snap["biases"])
    mx.eval(model.parameters(), optimizer.state)
    if side is not None:
        mx.eval(side.state)
    if snap["biases"] is not None:
        mx.eval(snap["biases"])


def measure_fn(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def check_exclusive_gpu():
    """Opt-in measurement guard; never stops another user's experiment.

    Probe outside timed regions. On a competing workspace benchmark, abort
    this measurement instead of reporting contended throughput as a result.
    """
    if os.environ.get("VIBY_BENCH_EXCLUSIVE", "0") != "1":
        return
    rows = subprocess.run(
        ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    for line in rows:
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        pid, executable, arguments = fields
        if Path(executable).name.startswith("python") and (
            "experiments/" in arguments or "trainer/train_" in arguments
        ):
            raise RuntimeError(
                f"GPU benchmark interrupted by competing Python process {pid}: {arguments}"
            )


def abba_blocks(
    run_a,
    run_b,
    warmup=3,
    block_iters=5,
    n_blocks=3,
    label_a="A",
    label_b="B",
    before_a=None,
    before_b=None,
):
    """预热两臂后按 A,B,B,A 测。fn 自己负责 mx.eval；before_* 在计时外调用。"""
    if label_a == label_b:
        raise ValueError("ABBA slot labels must differ, including A/A comparisons")
    if warmup < 0 or block_iters <= 0 or n_blocks <= 0:
        raise ValueError("expected warmup >= 0 and positive block_iters/n_blocks")

    def _prep(name):
        check_exclusive_gpu()
        fn = before_a if name == label_a else before_b
        if fn is not None:
            fn()

    for _ in range(warmup):
        _prep(label_a)
        run_a()
    for _ in range(warmup):
        _prep(label_b)
        run_b()
    blocks = []
    all_a, all_b = [], []
    first_a_median = None
    last_a_median = None
    for bi in range(n_blocks):
        slots = []
        for name, run in (
            (label_a, run_a),
            (label_b, run_b),
            (label_b, run_b),
            (label_a, run_a),
        ):
            mx.reset_peak_memory()
            times = []
            for _ in range(block_iters):
                _prep(name)
                t0 = time.perf_counter()
                run()
                times.append(time.perf_counter() - t0)
                check_exclusive_gpu()
            mem = memory_snapshot()
            rec = {
                "arm": name,
                "times_s": times,
                "median_s": statistics.median(times),
                "memory": mem,
            }
            slots.append(rec)
            if name == label_a:
                all_a.extend(times)
                if first_a_median is None:
                    first_a_median = rec["median_s"]
                last_a_median = rec["median_s"]
            else:
                all_b.extend(times)
        a_meds = [s["median_s"] for s in slots if s["arm"] == label_a]
        b_meds = [s["median_s"] for s in slots if s["arm"] == label_b]
        paired = statistics.median(b_meds) / statistics.median(a_meds)
        blocks.append(
            {
                "index": bi,
                "slots": slots,
                "A_median_s": statistics.median(a_meds),
                "B_median_s": statistics.median(b_meds),
                "B_over_A": paired,
                "delta_s": statistics.median(b_meds) - statistics.median(a_meds),
            }
        )
    block_speedups = [b["B_over_A"] for b in blocks]
    aa_drift = None
    if first_a_median and last_a_median:
        aa_drift = abs(last_a_median - first_a_median) / first_a_median
    inconclusive = aa_drift is not None and aa_drift > 0.03
    return {
        "warmup": warmup,
        "block_iters": block_iters,
        "n_blocks": n_blocks,
        "order": "ABBA",
        "blocks": blocks,
        "A": summarize_times(all_a),
        "B": summarize_times(all_b),
        "paired_block_median_B_over_A": statistics.median(block_speedups),
        "paired_block_medians_B_over_A": block_speedups,
        "aa_end_over_start_abs_rel": aa_drift,
        "inconclusive_aa_drift": inconclusive,
        "label_a": label_a,
        "label_b": label_b,
    }


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def param_identity(model):
    n_total = int(model.num_parameters())
    n_train = sum(int(a.size) for _, a in tree_flatten(model.trainable_parameters()))
    dtypes = sorted({str(a.dtype) for _, a in tree_flatten(model.parameters())})
    return {
        "num_parameters": n_total,
        "num_trainable": n_train,
        "parameter_dtypes": dtypes,
    }
