"""Paired, token-weighted CE interventions on a fixed held-out NPZ.

Input: input_ids, labels [N,T], optional loss_mask [N,T]. Every row is an
unpadded single document; batch_size >= 2. No training or cache reuse occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.utils import load_model_weights


def evaluate(model, arrays, batch_size):
    x, y, mask = arrays
    if batch_size < 2 or x.ndim != 2 or x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("Require matching [N,T] arrays and batch_size >= 2")
    if (
        len(x) < 2
        or not np.issubdtype(x.dtype, np.integer)
        or not np.issubdtype(y.dtype, np.integer)
    ):
        raise ValueError("Require at least two integer token rows")
    if not np.isfinite(mask).all() or (mask < 0).any() or mask.sum() <= 0:
        raise ValueError("loss_mask must be finite, nonnegative and have positive mass")
    if (
        (x < 0).any()
        or (y < 0).any()
        or (x >= model.config.vocab_size).any()
        or (y >= model.config.vocab_size).any()
    ):
        raise ValueError("Token IDs must fit checkpoint vocabulary")
    if x.shape[1] > model.config.max_seq_len:
        raise ValueError("Validation rows exceed checkpoint max_seq_len")
    model.eval()
    sums = dict.fromkeys(("normal", "off", "swap"), 0.0)
    total = 0.0
    records = []
    start = 0
    while start < len(x):
        end = min(start + batch_size, len(x))
        if len(x) - end == 1:
            end += 1  # retain the final row without a singleton self-swap
        ids, labels, weights = map(
            mx.array, (x[start:end], y[start:end], mask[start:end])
        )
        mass = float(mask[start:end].sum())
        record = {
            "start": start,
            "end": end,
            "token_weight": mass,
            "swap_donor_rows": list(range(start + 1, end)) + [start],
        }
        for mode in sums:
            result = model(
                ids,
                labels=labels,
                loss_mask=weights,
                use_mtp=False,
                ncp_intervention=mode,
            )
            ce = float(result.lm_loss.item())
            if not np.isfinite(ce):
                raise ValueError(f"Non-finite CE for {mode} at row {start}")
            sums[mode] += ce * mass
            record[mode + "_ce"] = ce
        records.append(record)
        total += mass
        start = end
    ce = {key: value / total for key, value in sums.items()}
    return {
        "ce": ce,
        "off_minus_normal": ce["off"] - ce["normal"],
        "swap_minus_normal": ce["swap"] - ce["normal"],
        "token_weight": total,
        "batches": records,
        "interpretation": "Positive CE deltas support useful conditioning within this checkpoint, not superiority to separately trained CED. Swap is an out-of-distribution intervention.",
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        required=True,
        help="Fixed held-out NPZ: input_ids, labels, optional loss_mask",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = Path(args.checkpoint)
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    cfg = VibyConfig.from_dict(meta["config"])
    if not cfg.ncp_enabled:
        raise ValueError("Checkpoint does not contain NCP")
    checkpoint_hash = sha256(checkpoint)
    data_hash = sha256(args.data)
    with np.load(args.data, allow_pickle=False) as archive:
        ids, labels = archive["input_ids"], archive["labels"]
        mask = (
            archive["loss_mask"]
            if "loss_mask" in archive
            else np.ones(ids.shape, np.float32)
        )
    model = VibyForCausalLM(cfg, skip_init=True)
    load_model_weights(model, str(checkpoint), strict=True)
    result = evaluate(model, (ids, labels, mask), args.batch_size)
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(
            "Checkpoint changed during evaluation; use an immutable checkpoint copy"
        )
    if sha256(args.data) != data_hash:
        raise RuntimeError("Validation data changed during evaluation")
    result.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        data=str(Path(args.data).resolve()),
        data_sha256=data_hash,
        config=cfg.to_dict(),
        batch_size=args.batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("ce", "off_minus_normal", "swap_minus_normal", "token_weight")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
