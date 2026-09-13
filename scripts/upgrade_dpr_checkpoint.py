"""Convert a DPR v1 checkpoint to contextual v2 without resetting progress.

Copies tensor bytes in bounded chunks. Original files are never modified.
Only the obsolete target network/state is removed; the fixed target projection
is added. The token/predictor/output weights and other optimizer states survive.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(path):
    with Path(path).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def rewrite(source, target, drop, additions=None):
    entries, start = header(source)
    new, copies, offset = {}, [], 0
    if "__metadata__" in entries:
        new["__metadata__"] = entries["__metadata__"]
    for key, value in sorted(
        ((k, v) for k, v in entries.items() if k != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        if drop(key):
            continue
        lo, hi = value["data_offsets"]
        new[key] = {**value, "data_offsets": [offset, offset + hi - lo]}
        copies.append((lo, hi - lo))
        offset += hi - lo
    blobs = []
    for key, array in (additions or {}).items():
        blob = array.tobytes(order="C")
        new[key] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(new, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with Path(source).open("rb") as inp, Path(target).open("xb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for lo, size in copies:
            inp.seek(start + lo)
            while size:
                chunk = inp.read(min(size, 1 << 20))
                if not chunk:
                    raise ValueError("Truncated safetensors input")
                out.write(chunk)
                size -= len(chunk)
        for blob in blobs:
            out.write(blob)


def upgrade(source, destination, dry_run=False):
    from model.config import VibyConfig
    from model.dpr import context_projection
    from trainer.utils import checkpoint_execution

    source, destination = Path(source).resolve(), Path(destination).resolve()
    sidecar = source.with_suffix(".json")
    optimizer = source.with_suffix(".optimizer.safetensors")
    meta = json.loads(sidecar.read_text())
    cfg = VibyConfig.from_dict(meta["config"])
    if not cfg.dpr_enabled or cfg.dpr_variant != "legacy_v1":
        raise ValueError("Source must be a legacy_v1 DPR checkpoint")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not optimizer.is_file():
        raise FileNotFoundError("Optimizer checkpoint is required to preserve progress")
    weights, _ = header(source)
    if not any(k.startswith("model.dpr.target.") for k in weights):
        raise ValueError("Source has no legacy target tensors")
    cfg.dpr_variant = "contextual_v2"
    cfg._validate()
    result = {
        "source": str(source),
        "destination": str(destination),
        "epoch": meta["epoch"],
        "step": meta["step"],
        "execution": checkpoint_execution(cfg),
    }
    if dry_run:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dpr-upgrade-", dir=destination.parent))
    try:
        rewrite(
            source,
            temporary / source.name,
            lambda k: k.startswith("model.dpr.target."),
            {
                "model.dpr.context_projection": context_projection(
                    cfg.dim, cfg.dpr_dim, cfg.dpr_seed
                )
            },
        )
        rewrite(
            optimizer, temporary / optimizer.name, lambda k: ".model.dpr.target." in k
        )
        # Old provenance describes the input artifact, not this conversion.
        (temporary / "source_metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        for key in (
            "code_sha",
            "dirty_diff_sha256",
            "source_sha256",
            "common_weights_sha256",
            "optimizer_sha256",
        ):
            meta.pop(key, None)
        meta["config"] = cfg.to_dict()
        meta["execution"] = checkpoint_execution(cfg)
        meta["args"].update(
            dpr_variant="contextual_v2",
            out_dir=str(destination),
            save_dir=str(destination),
        )
        if "optimizer_parameter_groups" in meta:
            meta["optimizer_parameter_groups"] = [
                [k for k in group if not k.startswith("model.dpr.target.")]
                for group in meta["optimizer_parameter_groups"]
            ]
        meta["migration"] = {
            **result,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Target objective changed; all retained tensor bytes and progress preserved. Legacy Muon radii are adopted at first update.",
        }
        (temporary / sidecar.name).write_text(json.dumps(meta, indent=2) + "\n")
        (temporary / "latest_checkpoint.txt").write_text(
            str(destination / source.name) + "\n"
        )
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, help="New output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.destination, args.dry_run), indent=2))
