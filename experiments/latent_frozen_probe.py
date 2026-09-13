#!/usr/bin/env python3
"""One preregistered frozen-CED A/B falsification, no hyperparameter search.

prepare -> cache validation -> paired one-pass training on cached train shards
-> document-paired evaluation / fixed-second-query intervention. Ordinary CED
is evaluated with DPR/PSR/recurrent/MTP disabled. Training never updates it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def emit(run, stage, **values):
    state = dict(pid=os.getpid(), stage=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values)
    write_json(run / "status.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def prepare(args, run):
    from transformers import AutoTokenizer

    destination = run / "data"
    destination.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"), local_files_only=True)
    if any(v is None for v in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)):
        raise ValueError("explicit BOS/EOS/PAD ids required")
    seen = set()
    exclude_hash = hashlib.sha256()
    emit(run, "exclude_checkpoint_training_documents", source=args.exclude_data)
    with open(args.exclude_data, "rb") as source:
        for line_no, raw in enumerate(source, 1):
            exclude_hash.update(raw)
            if raw.strip():
                seen.add(doc_hash(json.loads(raw)["text"]))
    excluded = len(seen)
    emit(run, "select_new_documents", excluded_unique_documents=excluded)
    budgets = {"train": args.train_tokens, "validation": args.validation_tokens}
    counts = dict.fromkeys(budgets, 0)
    tokens, documents = {k: [] for k in budgets}, {k: [] for k in budgets}
    records = []
    scanned, offset = 0, 0
    with open(args.data, "rb") as source:
        while any(counts[k] < budgets[k] for k in budgets):
            byte_start = source.tell()
            raw = source.readline()
            if not raw:
                raise ValueError("source exhausted before independent token budgets")
            scanned += 1
            offset = source.tell()
            if not raw.strip():
                continue
            text = json.loads(raw)["text"]
            digest = doc_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            split = "validation" if int(digest[:16], 16) % 11 == 0 else "train"
            if counts[split] >= budgets[split]:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Literal special ids in content would create untracked EOS/PAD
            # boundaries. Skip them explicitly and record that selection rule.
            if not ids or any(i in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) for i in ids):
                continue
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            take = min(len(ids) - 1, budgets[split] - counts[split])
            ids = ids[:take + 1]
            doc = len(records)
            tokens[split].extend(ids)
            documents[split].extend([doc] * len(ids))
            counts[split] += take
            records.append(dict(document_id=doc, split=split, line=scanned, byte_start=byte_start, byte_end=offset, text_sha256=digest, raw_sha256=hashlib.sha256(raw).hexdigest(), valid_tokens=take, truncated=take < len(tokenizer.encode(text, add_special_tokens=False)) + 1))
    manifest = dict(source=str(Path(args.data).resolve()), source_stat=dict(size=Path(args.data).stat().st_size, mtime_ns=Path(args.data).stat().st_mtime_ns), source_cursor=dict(line=scanned, byte_offset=offset), exclude_source=str(Path(args.exclude_data).resolve()), exclude_sha256=exclude_hash.hexdigest(), excluded_documents=excluded, tokenizer_sha256=sha256(ROOT / "model/tokenizer.json"), tokenizer_config_sha256=sha256(ROOT / "model/tokenizer_config.json"), normalization="collapse Unicode whitespace, exact document SHA256 exclusion", split_rule="SHA256 first 64 bits mod 11 == 0 is validation", special_token_policy="skip source text encoding literal BOS/EOS/PAD", counts=counts, bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, records=records)
    write_json(destination / "manifest.json", manifest)
    for split in budgets:
        stream = np.array(tokens[split], np.int32)
        seg = np.array(documents[split], np.int32)
        xs, ys, masks, segments = [], [], [], []
        for start in range(0, len(stream) - 1, args.sequence_length):
            end = min(start + args.sequence_length + 1, len(stream))
            seq, doc = stream[start:end], seg[start:end]
            padding = args.sequence_length + 1 - len(seq)
            seq = np.pad(seq, (0, padding), constant_values=tokenizer.pad_token_id)
            doc = np.pad(doc, (0, padding), constant_values=-1)
            mask = (doc[:-1] == doc[1:]) & (doc[1:] >= 0)
            xs.append(seq[:-1]); ys.append(seq[1:]); masks.append(mask); segments.append(doc[:-1])
        arrays = dict(input_ids=np.array(xs), labels=np.array(ys), loss_mask=np.array(masks), segments=np.array(segments))
        assert int(arrays["loss_mask"].sum()) == budgets[split]
        np.savez(destination / f"{split}.npz", **arrays)
    emit(run, "data_ready", counts=counts, documents=len(records), source_cursor=manifest["source_cursor"])
    return manifest


def load_base(args):
    import mlx.core as mx
    from model.config import VibyConfig
    from model.model import VibyForCausalLM

    meta = json.loads(Path(args.checkpoint).with_suffix(".json").read_text())
    config = VibyConfig.from_dict(meta["config"])
    if config.max_seq_len < args.sequence_length:
        raise ValueError("probe context exceeds checkpoint maximum")
    model = VibyForCausalLM(config, skip_init=True)
    model.load_weights(list(mx.load(args.checkpoint).items()), strict=True)
    model.eval()
    model.freeze()
    mx.eval(model.parameters())
    return model, meta


def cache_one(model, data, index, config, manifest, path):
    import mlx.core as mx
    from model.latent_inference import prepare_features

    x, y, mask, seg = (data[k][index] for k in ("input_ids", "labels", "loss_mask", "segments"))
    # Engram's rolling hash does not consume segment_ids. Put each document
    # fragment in its own batch row so its history, attention and n-gram lookup
    # all start independently. Stitch only detached features back into a shard.
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    spans = [(int(s), int(e)) for s, e in zip(starts, np.r_[starts[1:], len(x)]) if seg[s] >= 0]
    length = max(e-s for s,e in spans)
    padded = np.full((len(spans), length), manifest["pad_id"], np.int32)
    for row, (s, e) in enumerate(spans):
        padded[row, :e-s] = x[s:e]
    result = model.model(mx.array(padded), pad_mask=mx.array(padded != manifest["pad_id"]), collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
    d, (a, e, indices) = result[0], result[-1]
    def stitch(value):
        pieces = [value[row, :end-start] for row, (start, end) in enumerate(spans)]
        padding = len(x) - sum(end-start for start,end in spans)
        if padding:
            pieces.append(mx.zeros((padding, value.shape[-1]), value.dtype))
        return mx.concatenate(pieces, 0)
    selected = [mx.where(indices[row, :end-start] >= 0, indices[row, :end-start] + start, -1) for row, (start,end) in enumerate(spans)]
    if spans[-1][1] < len(x):
        selected.append(mx.full((len(x)-spans[-1][1], 128), -1, mx.int32))
    features = prepare_features(stitch(a), stitch(e), stitch(d), mx.concatenate(selected), x, y, mask, seg, config, eos_id=manifest["eos_id"], pad_id=manifest["pad_id"])
    features = {k: mx.stop_gradient(v) for k, v in features.items()}
    features["document_ids"] = mx.array(seg)
    mx.eval(features)
    mx.save_safetensors(path, features)
    return features


def paired_bootstrap(documents, seed=20260913, repeats=2000):
    # columns A, B, baseline, fixed-query A, valid tokens
    values = np.array(list(documents.values()), np.float64)
    totals = values.sum(0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        v = values[rng.integers(0, len(values), len(values))].sum(0)
        samples.append([(v[0] - v[2]) / v[4], (v[0] - v[1]) / v[4], (v[3] - v[0]) / v[4]])
    intervals = np.quantile(np.array(samples), [.025, .975], axis=0).T.tolist()
    means = totals[:4] / totals[4]
    return dict(ce=dict(zip(("A_filter", "B_static", "CED", "A_fixed_second_query"), means.tolist())), delta_A_minus_CED=float(means[0]-means[2]), delta_A_minus_B=float(means[0]-means[1]), delta_fixed_query_minus_A=float(means[3]-means[0]), paired_document_bootstrap_95ci=dict(zip(("A_minus_CED", "A_minus_B", "fixed_query_minus_A"), intervals)), documents=len(values), valid_tokens=int(totals[4]), statistical_gate_pass=intervals[0][1] < 0 and intervals[1][1] < 0)


def evaluate(run, branches, head, paths):
    import mlx.core as mx

    documents, timings = {}, dict(A_filter=0.0, B_static=0.0, A_fixed_second_query=0.0)
    started = time.perf_counter()
    for i, path in enumerate(paths):
        f = mx.load(path)
        mx.eval(f)
        outputs = []
        # Alternate order to avoid always measuring A first.
        names = ["A_filter", "B_static", "A_fixed_second_query"]
        if i % 2:
            names.reverse()
        by_name = {}
        for name in names:
            m = branches[1] if name == "B_static" else branches[0]
            before = time.perf_counter()
            nll, base = m.token_losses(f, head, static=name == "B_static", fixed_second_query=name == "A_fixed_second_query", rematerialize=False)
            mx.eval(nll, base)
            timings[name] += time.perf_counter() - before
            by_name[name] = np.array(nll)
        outputs = [by_name["A_filter"], by_name["B_static"], np.array(base), by_name["A_fixed_second_query"]]
        rows, doc = np.array(f["rows"]), np.array(f["document_ids"])
        valid = rows >= 0
        ids = doc[np.maximum(rows, 0)]
        for document in np.unique(ids[valid]):
            selected = valid & (ids == document)
            sums = np.array([float(v[selected].sum(dtype=np.float64)) for v in outputs] + [int(selected.sum())])
            documents[int(document)] = documents.get(int(document), np.zeros(5)) + sums
        if (i + 1) % 20 == 0:
            emit(run, "evaluate", shards=i+1, total_shards=len(paths))
    result = paired_bootstrap(documents)
    result.update(validation_wall_seconds=time.perf_counter()-started, component_evaluation_seconds=timings)
    with (run / "validation_documents.jsonl").open("w") as stream:
        for doc, row in sorted(documents.items()):
            stream.write(json.dumps(dict(document_id=doc, A_nll=row[0], B_nll=row[1], CED_nll=row[2], fixed_query_A_nll=row[3], tokens=int(row[4])))+"\n")
    return result


def run_probe(args, run, manifest):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from model.latent_inference import LatentConfig, ShortBlockLatent

    mx.set_cache_limit(1 << 30)
    emit(run, "load_frozen_checkpoint")
    model, metadata = load_base(args)
    config = LatentConfig(dim=model.config.dim, vocab=model.config.vocab_size)
    head = mx.stop_gradient(model._head_weight())
    mx.eval(head)
    for name, value in (("eos_id", model.config.eos_token_id), ("pad_id", model.config.pad_token_id)):
        if manifest[name] != value:
            raise ValueError("checkpoint/tokenizer special ids differ")
    branches = []
    for _ in range(2):
        mx.random.seed(args.seed)
        branches.append(ShortBlockLatent(config))
    for (ka, va), (kb, vb) in zip(tree_flatten(branches[0].parameters()), tree_flatten(branches[1].parameters())):
        assert ka == kb and bool(mx.array_equal(va, vb))
    branches[0].save_weights(str(run / "initial_branch.safetensors"))
    optimizers = [optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0) for _ in branches]
    functions = [nn.value_and_grad(m, lambda branch, f, static=bool(i): branch(f, head, static=static)) for i, m in enumerate(branches)]
    write_json(run / "resolved_config.json", dict(latent=config.to_dict(), base=model.config.to_dict(), checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint), checkpoint_epoch=metadata["epoch"], checkpoint_microstep=metadata["step"], base_execution=dict(use_dpr=False, use_ced_recurrent=False, psr=False, mtp=False, independent_document_rows=True), branch_dtype="float32", feature_dtype="checkpoint dtype (BF16 expected)", head_projection_dtype="checkpoint dtype in A/B/CED, then FP32 softmax", optimizer=dict(kind="AdamW", learning_rate=args.learning_rate, betas=[.9,.999], weight_decay=0, schedule="constant one pass", clipping=None), compile=False, vocabulary_rematerialization=True, branch_parameters=sum(v.size for _, v in tree_flatten(branches[0].parameters())), args=vars(args), python=platform.python_version(), mlx=mx.__version__, hardware=mx.device_info()))
    validation = np.load(run / "data/validation.npz")
    training = np.load(run / "data/train.npz")
    cache_dir = run / "features"
    cache_dir.mkdir()
    val_paths = []
    cache_seconds = 0.0
    for i in range(len(validation["input_ids"])):
        path = cache_dir / f"validation_{i:05d}.safetensors"
        before = time.perf_counter()
        cache_one(model, validation, i, config, manifest, str(path))
        cache_seconds += time.perf_counter() - before
        val_paths.append(str(path))
        if i < 2 or (i+1) % 20 == 0:
            emit(run, "cache_validation", shards=i+1, total_shards=len(validation["input_ids"]), elapsed_seconds=cache_seconds)
    token_count, train_seconds = 0, [0.0, 0.0]
    gradient_first = None
    with (run / "training_metrics.jsonl").open("w", buffering=1) as log:
        for i in range(len(training["input_ids"])):
            path = cache_dir / f"train_{i:05d}.safetensors"
            before = time.perf_counter()
            cache_one(model, training, i, config, manifest, str(path))
            # Both branches consume the actual on-disk, identical frozen shard.
            f = mx.load(str(path))
            mx.eval(f)
            cache_seconds += time.perf_counter() - before
            count = int((f["rows"] >= 0).sum())
            losses, elapsed = [0.0, 0.0], [0.0, 0.0]
            for j in ([0, 1] if i % 2 == 0 else [1, 0]):
                before = time.perf_counter()
                loss, grads = functions[j](branches[j], f)
                norm = mx.sqrt(sum(mx.sum(v*v) for _, v in tree_flatten(grads)))
                mx.eval(loss, grads, norm)
                if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
                    raise FloatingPointError(f"nonfinite branch {j}, shard {i}")
                if i == 0 and j == 0:
                    gradient_first = {k: float(mx.linalg.norm(v)) for k, v in tree_flatten(grads)}
                    write_json(run / "first_gradient_norms.json", gradient_first)
                optimizers[j].update(branches[j], grads)
                mx.eval(branches[j].parameters(), optimizers[j].state)
                losses[j] = float(loss)
                elapsed[j] = time.perf_counter() - before
                train_seconds[j] += elapsed[j]
                del grads
            token_count += count
            row = dict(step=i+1, valid_tokens=token_count, shard_tokens=count, A_filter_nll=losses[0], B_static_nll=losses[1], A_seconds=elapsed[0], B_seconds=elapsed[1], peak_memory_gib=mx.get_peak_memory()/2**30, cache_seconds=cache_seconds)
            log.write(json.dumps(row)+"\n")
            if i < 2 or (i+1) % 10 == 0:
                emit(run, "training", **row, total_shards=len(training["input_ids"]))
            if (i+1) % 100 == 0 or i+1 == len(training["input_ids"]):
                for j, name in enumerate(("A_filter", "B_static")):
                    branches[j].save_weights(str(run / f"{name}.safetensors"))
                    mx.save_safetensors(str(run / f"{name}.optimizer.safetensors"), dict(tree_flatten(optimizers[j].state)))
                write_json(run / "training_checkpoint.json", dict(step=i+1, valid_tokens=token_count, latent=config.to_dict(), train_seconds=train_seconds, cache_seconds=cache_seconds, completed=i+1 == len(training["input_ids"])))
    assert token_count == args.train_tokens
    # No optimizer has ever held the backbone. Drop it before validation.
    del model
    mx.clear_cache()
    emit(run, "evaluate", training_valid_tokens=token_count)
    result = evaluate(run, branches, head, val_paths)
    result.update(training_valid_tokens=token_count, branch_training_seconds=dict(A_filter=train_seconds[0], B_static=train_seconds[1]), feature_cache_seconds=cache_seconds, peak_memory_gib=mx.get_peak_memory()/2**30, full_training_authorized_by_gate=False, limitation="frozen interface test only; no backbone data/compute efficiency conclusion; wall-clock gate requires measured total continuation cost", first_gradient_norms=gradient_first)
    result["decision"] = "stop_latent_route" if not result["statistical_gate_pass"] else "frozen_gate_passed_only_review_wall_cost_before_matched_continuation"
    if (args.train_tokens, args.validation_tokens) != (1_000_000, 100_000):
        result["decision"] = "diagnostic_only_not_the_preregistered_falsification_budget"
    write_json(run / "results.json", result)
    emit(run, "completed", **result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--exclude-data", required=True)
    p.add_argument("--train-tokens", type=int, default=1_000_000)
    p.add_argument("--validation-tokens", type=int, default=100_000)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    if (run / "resolved_config.json").exists():
        p.error("run already started; use a new run directory (no silent restart)")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "command.json", dict(argv=sys.argv, cwd=str(ROOT), pid=os.getpid(), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    try:
        if (run / "data/manifest.json").exists():
            manifest = json.loads((run / "data/manifest.json").read_text())
            if manifest["counts"] != dict(train=args.train_tokens, validation=args.validation_tokens):
                raise ValueError("prepared token budgets differ")
        else:
            manifest = prepare(args, run)
        if not args.prepare_only:
            run_probe(args, run, manifest)
    except BaseException as error:
        emit(run, "failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
