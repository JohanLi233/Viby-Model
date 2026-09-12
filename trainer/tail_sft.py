"""TailSFT data identity, immutable initial-policy losses, and resume contract."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import mlx.core as mx
import numpy as np

TAIL_METRICS = (
    "sequences",
    "retained_sequences",
    "retained_tokens",
    "unfiltered_ce",
    "mean_offset",
)


def enabled(args):
    return getattr(args, "sft_algorithm", "standard") == "tail"


def validate_args(args):
    f = args.tail_sft_filter_fraction
    if not math.isfinite(f) or not 0 <= f < 1:
        raise ValueError("tail_sft_filter_fraction must be finite and in [0, 1)")
    if enabled(args):
        if args.pack_sequences or args.doc_mask:
            raise ValueError(
                "TailSFT requires individual sequences; stream --pack_sequences/--doc_mask splits responses. Use --sft_algorithm standard for legacy packing."
            )
        if args.freeze_backbone or args.psr_freeze_base:
            raise ValueError("TailSFT requires an unfrozen SFT backbone")
        if args.batch_size == 1 and f > 0:
            from .utils import Logger

            Logger(
                "TailSFT: batch_size=1 retains its only sequence; use batch_size >= 2 for filtering"
            )


def filter_fraction(args, step, total_steps):
    f = args.tail_sft_filter_fraction
    if args.tail_sft_schedule == "ramp":
        f *= min(max(step, 0) / max(total_steps - 1, 1), 1.0)
    return f


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def reference_identity(dataset, config, args, checkpoint):
    root = Path(__file__).resolve().parents[1]
    tok = dataset.tokenizer
    backend = getattr(tok, "backend_tokenizer", None)
    from dataset.lm_dataset import _tokenizer_cache_fingerprint

    return {
        "version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_sha256": file_sha256(dataset.data_path),
        "tokenizer": _digest(
            {
                "vocab": _tokenizer_cache_fingerprint(tok),
                "backend": backend.to_str() if backend is not None else None,
                "template": getattr(tok, "chat_template", None),
                "special": getattr(tok, "special_tokens_map", None),
                "padding_side": getattr(tok, "padding_side", None),
            }
        ),
        "config": config.to_dict(),
        "dtype": args.dtype,
        "seed": dataset.deterministic_seed,
        "max_length": dataset.max_length,
        "empty_think_ratio": dataset.empty_think_ratio,
        "sources": {
            str(p.relative_to(root)): file_sha256(p)
            for p in [
                *sorted((root / "model").rglob("*.py")),
                root / "dataset/lm_dataset.py",
                Path(__file__),
            ]
        },
        "environment": {k: v for k, v in os.environ.items() if k.startswith("VIBY_")},
    }


class TailSFTDataset:
    """Stable original row identity survives shuffling; invalid rows are excluded."""

    def __init__(self, dataset, losses, counts):
        self.dataset = dataset
        self.losses = np.asarray(losses, dtype=np.float32)
        self.indices = np.flatnonzero(counts > 0)
        if not len(self.indices):
            raise ValueError(
                "TailSFT dataset has no supervised assistant tokens after truncation"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        x, y, mask = self.dataset[row]
        return {"X": x, "Y": y, "loss_mask": mask, "tail_reference": self.losses[row]}


def validate_resume(path, args):
    """Called before loading optimizer/weights; legacy SFT is explicitly standard."""
    if getattr(args, "reset_optimizer", False):
        return
    sidecar = Path(path).with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if meta.get("training_type") not in (None, "sft", "full_sft"):
        return
    previous = meta.get("args", {})
    if previous.get("sft_algorithm", "standard") != getattr(
        args, "sft_algorithm", "standard"
    ):
        raise ValueError(
            "SFT algorithm changed: start a new run with --resume CHECKPOINT --reset_optimizer"
        )
    if enabled(args):
        if not previous.get("tail_sft_state") or previous["tail_sft_state"] != getattr(
            args, "tail_sft_state", None
        ):
            raise ValueError(
                "TailSFT reference/data/filter contract changed or is missing; restore the original settings/cache or use --reset_optimizer for a new run"
            )


def prepare_dataset(dataset, model, config, args, checkpoint):
    """Score pi_0 before BaseTrainer can restore a later SFT checkpoint.

    O(dataset rows) CPU storage; one eval forward per batch on a cache miss.
    Cache writes are atomic and never include weights. The original checkpoint
    is required to validate/rebuild the reference, including on resume.
    """
    from .utils import Logger, find_latest_checkpoint

    identity = reference_identity(dataset, config, args, checkpoint)
    key = _digest(identity)
    cache = (
        Path(args.tail_sft_cache)
        if args.tail_sft_cache
        else Path(__file__).resolve().parents[1] / ".cache" / f"tailsft_{key}.npz"
    )
    n = len(dataset)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["identity"].item()) != key:
                raise ValueError(f"TailSFT cache identity mismatch: {cache}")
            losses, counts = saved["losses"].copy(), saved["counts"].copy()
        Logger(f"TailSFT: reusing initial losses {cache}")
    else:
        losses, counts = np.zeros(n, np.float32), np.zeros(n, np.float32)
        was_training = model.training
        model.eval()
        started = last_log = time.monotonic()
        Logger(
            f"TailSFT: scoring {n} sequences with the initial policy (one-time pass)"
        )
        try:
            for start in range(0, n, args.batch_size):
                samples = [
                    dataset[i] for i in range(start, min(start + args.batch_size, n))
                ]
                x, y, mask = (mx.array(np.stack(items)) for items in zip(*samples))
                result = model(
                    x,
                    labels=y,
                    loss_mask=mask,
                    attention_mask=(x != dataset.tokenizer.pad_token_id),
                    psr_mode="off",
                    use_mtp=False,
                    return_sequence_losses=True,
                )
                mx.eval(result.sequence_losses, result.sequence_token_counts)
                end = start + len(samples)
                losses[start:end] = np.asarray(result.sequence_losses)
                counts[start:end] = np.asarray(result.sequence_token_counts)
                if time.monotonic() - last_log >= 15 or end == n:
                    Logger(
                        f"TailSFT reference: {end}/{n} sequences, {time.monotonic() - started:.1f}s"
                    )
                    last_log = time.monotonic()
        finally:
            model.train(was_training)
        if not np.isfinite(losses).all():
            raise ValueError("TailSFT initial-policy losses are non-finite")
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache.parent, suffix=".npz", delete=False
            ) as file:
                tmp = Path(file.name)
                np.savez(file, identity=key, losses=losses, counts=counts)
            os.replace(tmp, cache)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    if (
        losses.shape != (n,)
        or counts.shape != (n,)
        or not np.isfinite(losses).all()
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise ValueError(f"Invalid TailSFT reference cache: {cache}")
    wrapped = TailSFTDataset(dataset, losses, counts)
    args.tail_sft_cache = str(cache.resolve())
    args.tail_sft_reference = identity
    args.tail_sft_state = {
        "identity": key,
        "losses_sha256": hashlib.sha256(
            losses.tobytes() + counts.tobytes()
        ).hexdigest(),
        "filter_fraction": args.tail_sft_filter_fraction,
        "schedule": args.tail_sft_schedule,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "selection": "microbatch_round_half_up_keep_one",
        "valid_sequences": len(wrapped),
    }
    resume = args.resume or (
        find_latest_checkpoint(args.save_dir) if args.auto_resume else None
    )
    if resume:
        validate_resume(resume, args)
    Logger(
        f"TailSFT: {len(wrapped)} valid sequences, {n - len(wrapped)} empty-target rows excluded; "
        f"filter={args.tail_sft_filter_fraction}, schedule={args.tail_sft_schedule}"
    )
    return wrapped
