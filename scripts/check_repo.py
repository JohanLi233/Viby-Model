#!/usr/bin/env python3
"""Repository entry checks and explicit pytest groups; default never imports MLX."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = (
    "AGENTS.md",
    "README.md",
    "research/README.md",
    "research/EXPERIMENT_PROTOCOL.md",
)
TEST_GROUPS = {
    "tools": ("test_repo_tools.py",),
    "host": ("test_moe_dataflow_host.py",),
    "config": ("test_v41_config.py",),
    "attention": (
        "test_v41_attention.py",
        "test_v41_consistency.py",
        "test_v41_xsa.py",
    ),
    "kernels": (
        "test_sparse_attention_kernel.py",
        "test_sparse_attention_key_owned.py",
        "test_indexer_kernel.py",
        "test_indexer_select.py",
        "test_sinkhorn_kernel.py",
        "test_kernel_plan_regressions.py",
        "test_inference_kernel_optimizations.py",
        "test_deep_kernel_optimizations.py",
        "test_v41_hc.py",
        "test_v41_mhc_single_pass.py",
        "test_hc_decode_kernel.py",
        "test_rope_decode_kernel.py",
        "test_window_attention_backward.py",
    ),
    "moe": ("test_v41_moe.py", "test_moe_qb.py", "test_moe_dataflow_metal.py"),
    "psr": ("test_psr.py", "test_psr_pretrain.py", "test_psr_engine.py"),
    "recurrent": (
        "test_ced_recurrent.py",
        "test_ced_recurrent_trainer.py",
        "test_ced_recurrent_runtime.py",
        "test_ced_recurrent_fused.py",
        "test_ced_position_kernel.py",
        "test_ced_masked_moe_counts.py",
        "test_ced_optimized.py",
    ),
    "engine": (
        "test_engine_speculative.py",
        "test_psr_engine.py",
        "test_v41_consistency.py",
    ),
    "data": ("test_pack_dataset.py", "test_pack_sft.py"),
    "sft": ("test_tail_sft.py", "test_pack_sft.py"),
    "training": (
        "test_v41_train.py",
        "test_muonh.py",
        "test_optimizer_fast_norm.py",
        "test_kernel_bench_state.py",
        "test_training_flops.py",
        "test_checkpoint_save.py",
        "test_optimizer_step_logging.py",
        "test_sinkhorn_rows_kernel.py",
        "test_sinkhorn_pairs_kernel.py",
    ),
    "all": (),
}


def project_python() -> str:
    candidate = ROOT / ".venv/bin/python"
    return str(candidate) if candidate.is_file() else sys.executable


def markdown_errors(path: Path) -> list[str]:
    """Check fenced blocks and inline link paths, leaving URLs and anchors alone."""
    errors = []
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = (run[0], len(run), number)
            elif run[0] == fence[0] and len(run) >= fence[1] and not tail.strip():
                fence = None
            continue
        if fence is not None:
            continue
        # Ignore Markdown examples inside inline code spans.
        line = re.sub(r"(`+).*?\1", "", line)
        for match in re.finditer(
            r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\s*\)", line
        ):
            target = match.group(1).removeprefix("<").removesuffix(">")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f"{path}:{number}: missing link target: {target}")
    if fence is not None:
        errors.append(f"{path}:{fence[2]}: unclosed code fence")
    return errors


def check_docs() -> int:
    paths = [ROOT / name for name in ENTRY_DOCS]
    paths += sorted((ROOT / "docs").glob("*.md"))
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"{path}: missing entry document")
        else:
            errors.extend(markdown_errors(path))
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    for group, files in TEST_GROUPS.items():
        for name in files:
            if not (ROOT / "tests" / name).is_file():
                errors.append(f"test group {group}: missing tests/{name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"OK: {len(paths)} entry documents, tool syntax, and test group paths (no MLX/GPU)."
    )
    return 0


def doctor() -> int:
    python = project_python()
    print(
        f"Repository: {ROOT}\nRunner: {sys.executable}\nProject Python: {python}",
        flush=True,
    )
    # Query the target interpreter's package metadata in a separate process.
    # In particular, importing mlx.core is not needed to read its version.
    probe = """import sys
from importlib import metadata
print('Python:', sys.version.split()[0])
for name in ('mlx', 'pytest', 'numpy', 'transformers'):
    try:
        print(f'{name}: {metadata.version(name)}')
    except metadata.PackageNotFoundError:
        print(f'{name}: missing')
"""
    result = subprocess.run([python, "-c", probe], cwd=ROOT, check=False)
    if result.returncode:
        return result.returncode
    for args in (("rev-parse", "--short", "HEAD"), ("status", "--short")):
        result = subprocess.run(["git", *args], cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


def run_tests(args: argparse.Namespace, extra: list[str]) -> int:
    selected = [f"tests/{name}" for name in TEST_GROUPS[args.group]] or ["tests/"]
    missing = [path for path in selected if not (ROOT / path).exists()]
    if missing:
        print("Missing test paths: " + ", ".join(missing), file=sys.stderr)
        return 2
    python = args.python or project_python()
    command = [python, "-m", "pytest", "-q", *selected, *extra]
    print(f"cwd: {ROOT}\n{shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    probe = subprocess.run(
        [
            python,
            "-c",
            "import importlib.util; raise SystemExit(importlib.util.find_spec('pytest') is None)",
        ],
        cwd=ROOT,
        check=False,
    )
    if probe.returncode:
        print(
            "pytest unavailable in this interpreter. See docs/DEVELOPMENT.md; nothing was installed.",
            file=sys.stderr,
        )
        return probe.returncode
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        index = argv.index("--")
        argv, extra = argv[:index], argv[index + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "docs", help="Check entry documents and group paths; no MLX/GPU (default)"
    )
    sub.add_parser("doctor", help="Read environment metadata and Git state; no MLX/GPU")
    sub.add_parser("list", help="List explicit pytest groups")
    test = sub.add_parser("test", help="Run one group; put pytest arguments after --")
    test.add_argument("group", choices=TEST_GROUPS)
    test.add_argument(
        "--dry-run", action="store_true", help="Print command without importing tests"
    )
    test.add_argument("--python", help="Override the test interpreter")
    args = parser.parse_args(argv)
    if extra and args.command != "test":
        parser.error("arguments after -- are supported only for test")
    if args.command in (None, "docs"):
        return check_docs()
    if args.command == "doctor":
        return doctor()
    if args.command == "list":
        for group, files in TEST_GROUPS.items():
            print(
                f"{group}: "
                + (
                    ", ".join(f"tests/{name}" for name in files)
                    or "tests/ (full suite)"
                )
            )
        return 0
    return run_tests(args, extra)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"check_repo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
