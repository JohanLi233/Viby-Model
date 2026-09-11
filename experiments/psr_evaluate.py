"""Token-weighted protected-PSR evaluation on an explicitly supplied NPZ corpus.

NPZ: input_ids, labels, loss_mask [N,T]; optional attention_mask, segment_ids,
 document_ids [N,T]. document_ids are required for document-level bootstrap.
Never evaluates a calibration set as final validation implicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.psr_pretrain import compact_anchor_plan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(checkpoint, config_path=None, enable=None):
    path = Path(config_path) if config_path else Path(checkpoint).with_suffix(".json")
    raw = json.loads(path.read_text())
    cfg = raw.get("config", raw)
    if enable is not None:
        cfg = {**cfg, "psr_enabled": enable}
    model = VibyForCausalLM(VibyConfig.from_dict(cfg), skip_init=True)
    weights = mx.load(str(checkpoint))
    if any(
        k.startswith(("model.reasoner.", "model.workspace_bridges.")) for k in weights
    ):
        raise ValueError(
            "Legacy PSR checkpoint must be audited using pinned 52d17a6, not silently remapped"
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def evaluate(model, arrays, mode, *, rounds=1, gate=1.0, batch_size=1):
    documents = {}
    position_groups = np.zeros((8, 3), dtype=np.float64)
    elapsed = 0.0
    scans = 0
    anchor_count = 0
    n = arrays["input_ids"].shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = mx.array(arrays["input_ids"][start:stop], mx.int32)
        y = mx.array(arrays["labels"][start:stop], mx.int32)
        mask = mx.array(arrays["loss_mask"][start:stop], mx.float32)
        pad = mx.array(
            arrays.get("attention_mask", np.ones_like(arrays["input_ids"]))[start:stop]
        )
        seg = (
            mx.array(arrays["segment_ids"][start:stop])
            if "segment_ids" in arrays
            else None
        )
        anchors = compact_anchor_plan(x, pad, seg, model.config.psr_horizon)
        mx.eval(x, y, mask, pad, anchors)
        options = (
            {}
            if mode == "off"
            else dict(psr_anchors=anchors, thinking_options={"rounds": rounds})
        )
        # Always fresh, uncached: no condition shares a contaminated cache.
        began = time.perf_counter()
        out = model(
            x,
            labels=y,
            loss_mask=mask,
            attention_mask=pad,
            segment_ids=seg,
            psr_mode=mode,
            psr_gate=mx.array(gate),
            return_metrics=True,
            return_thinking=True,
            **options,
        )
        mx.eval(out.metrics, out.base_token_nll, out.corrected_token_nll)
        if out.thinking_trace is not None:
            mx.eval(out.thinking_trace.states)
            scans += out.thinking_trace.full_scans
            anchor_count += int(mx.sum(anchors >= 0))
        elapsed += time.perf_counter() - began
        position_groups += np.asarray(out.metrics).sum(axis=0)
        bn, cn, cov = (
            np.asarray(out.base_token_nll),
            np.asarray(out.corrected_token_nll),
            np.asarray(out.bridge_mask),
        )
        valid = np.asarray(mask)
        ids = arrays.get("document_ids")
        for row in range(stop - start):
            docids = (
                ids[start + row]
                if ids is not None
                else np.full(x.shape[1], start + row)
            )
            for doc in np.unique(docids):
                chosen = (docids == doc) * valid[row]
                key = str(int(doc))
                record = documents.setdefault(
                    key,
                    dict(
                        base_nll_sum=0.0,
                        nll_sum=0.0,
                        valid_label_count=0.0,
                        covered_base_nll_sum=0.0,
                        covered_nll_sum=0.0,
                        covered_count=0.0,
                    ),
                )
                record["base_nll_sum"] += float((bn[row] * chosen).sum())
                record["nll_sum"] += float((cn[row] * chosen).sum())
                record["valid_label_count"] += float(chosen.sum())
                record["covered_base_nll_sum"] += float(
                    (bn[row] * chosen * cov[row]).sum()
                )
                record["covered_nll_sum"] += float((cn[row] * chosen * cov[row]).sum())
                record["covered_count"] += float((chosen * cov[row]).sum())
    sums = {
        k: sum(r[k] for r in documents.values()) for k in next(iter(documents.values()))
    }
    count = sums["valid_label_count"]
    return dict(
        mode=mode,
        rounds=rounds,
        gate=gate,
        **sums,
        mean_nll=sums["nll_sum"] / max(count, 1),
        base_mean_nll=sums["base_nll_sum"] / max(count, 1),
        bridge_coverage=sums["covered_count"] / max(count, 1),
        documents=documents,
        position_groups={
            name: dict(
                base_nll_sum=float(row[0]),
                nll_sum=float(row[1]),
                valid_label_count=float(row[2]),
            )
            for name, row in zip(
                (
                    "all",
                    "covered",
                    "complement",
                    "prefix",
                    "offset_0_3",
                    "offset_4_7",
                    "offset_8_15",
                    "offset_16_31",
                ),
                position_groups,
            )
        },
        resampling_unit="document" if "document_ids" in arrays else "row",
        seconds_including_first_compile=elapsed,
        supervised_labels_per_second=count / max(elapsed, 1e-12),
        logical_dense_scans=scans,
        valid_anchor_count=anchor_count,
        peak_memory_bytes=mx.get_peak_memory(),
        profile_note="logical scans with trace outputs forced; not a GPU profiler count",
    )


def paired_bootstrap(first, second, seed=1337, repetitions=1000):
    if first.keys() != second.keys():
        raise ValueError("document identities must be paired")
    if any(
        first[k]["valid_label_count"] != second[k]["valid_label_count"] for k in first
    ):
        raise ValueError("paired conditions must have identical label counts")
    keys = sorted(first)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        chosen = rng.choice(keys, len(keys), replace=True)
        count = sum(first[k]["valid_label_count"] for k in chosen)
        if count:
            deltas.append(
                sum(first[k]["nll_sum"] - second[k]["nll_sum"] for k in chosen) / count
            )
    return np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else [None, None]


def calibrate_gate(base, delta, labels, mask):
    """Fixed-logit convex calibration, never a test-set improvement guarantee."""

    def derivative(g):
        z = base.astype(np.float64) + g * delta
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        dy = np.take_along_axis(delta, labels[..., None], axis=-1)[..., 0]
        return float((((p * delta).sum(-1) - dy) * mask).sum() / max(mask.sum(), 1))

    if derivative(0) >= 0:
        return 0.0
    if derivative(1) <= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def profile_components(model, arrays):
    """Synchronized eager component timings; not a GPU kernel trace."""
    x = mx.array(arrays["input_ids"][:1], mx.int32)
    pad = mx.array(arrays.get("attention_mask", np.ones_like(arrays["input_ids"]))[:1])
    seg = mx.array(arrays["segment_ids"][:1]) if "segment_ids" in arrays else None
    anchors = compact_anchor_plan(x, pad, seg, model.config.psr_horizon)
    mx.eval(x, pad, anchors)
    times = []
    for _ in range(3):
        began = time.perf_counter()
        hidden, _, _, memory = model.model(
            x, pad_mask=pad, segment_ids=seg, collect_main=False, return_memory=True
        )
        mx.eval(hidden, memory.values, memory.features)
        base_end = time.perf_counter()
        boundary = model.model.layers[model.config.n_encoder_layers].attn
        state, trace = model.psr(
            memory, anchors, boundary.freq_cos, boundary.freq_sin, record_trace=True
        )
        mx.eval(trace.states)
        reasoner_end = time.perf_counter()
        positions = mx.broadcast_to(mx.arange(x.shape[1])[None], x.shape)
        vector, _, _ = model.psr.read_workspace(hidden, state, positions, seg, pad)
        mx.eval(vector)
        bridge_end = time.perf_counter()
        logits = mx.stop_gradient(
            model.logits(hidden)
        ) + model.psr.calibration_gate * model.psr.output(vector)
        mx.eval(logits)
        end = time.perf_counter()
        times.append(
            dict(
                baseline=base_end - began,
                reasoner_dense_read_and_update=reasoner_end - base_end,
                workspace_readout=bridge_end - reasoner_end,
                vocabulary_heads=end - bridge_end,
                psr_indexer=0.0,
                total=end - began,
            )
        )
    return dict(
        warmup=times[0],
        samples=times[1:],
        logical_scans_per_sample=trace.full_scans,
        mode="synchronized eager; forces intermediate outputs; not fused throughput or a Metal capture",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--baseline-config")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    arrays = dict(np.load(args.data))
    model = load_model(args.checkpoint, args.config)
    results = {
        mode: evaluate(model, arrays, mode, rounds=args.rounds, gate=args.gate)
        for mode in ("off", "state_only", "recurrent")
    }
    if args.baseline_checkpoint:
        baseline = load_model(args.baseline_checkpoint, args.baseline_config)
        results["baseline"] = evaluate(baseline, arrays, "off", rounds=args.rounds)
        on, off, base = (
            results[k]["mean_nll"] for k in ("recurrent", "off", "baseline")
        )
        results["decomposition"] = dict(
            total=on - base, injection=on - off, history=off - base
        )
    results["paired_recurrent_vs_off_95pct"] = paired_bootstrap(
        results["recurrent"]["documents"], results["off"]["documents"]
    )
    if args.profile:
        results["component_profile"] = profile_components(model, arrays)
    results["data_sha256"] = sha256(args.data)
    results["checkpoint_sha256"] = sha256(args.checkpoint)
    output = Path(args.output)
    with output.open("x") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
