"""Host-only validation for the current microbatch-index resume protocol."""

from pathlib import Path


def validate_resume_layout(metadata, args):
    """Do not silently replay/skip samples when resuming with a new layout.

    A microstep is not a sample cursor. Changing batch/sequence layout needs
    explicit cursor and LR migration; preserving the optimizer step alone is
    insufficient. Missing legacy metadata remains readable, without claiming
    that its data order has been verified.
    """
    if getattr(args, "reset_optimizer", False) or getattr(
        args, "freeze_backbone", False
    ):
        return
    saved = metadata.get("args", {})
    mismatches = []
    for key in (
        "batch_size",
        "max_seq_len",
        "seed",
        "pack_sequences",
        "doc_mask",
        "doc_align",
        "max_doc_len",
        "data_path",
    ):
        if key not in saved or not hasattr(args, key):
            continue
        old, new = saved[key], getattr(args, key)
        if key == "data_path" and old and new:
            equal = Path(old).expanduser().resolve() == Path(new).expanduser().resolve()
        else:
            equal = old == new
        if not equal:
            mismatches.append(f"{key}: checkpoint={old!r}, requested={new!r}")
    if mismatches:
        raise ValueError(
            "Cannot resume by microbatch index after a data-layout change ("
            + "; ".join(mismatches)
            + "). Keep the saved layout; changing it requires an explicit sample/token-cursor migration. "
            "--reset_optimizer is a new warm start, not continuous resume."
        )
