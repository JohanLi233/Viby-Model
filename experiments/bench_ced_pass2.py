"""Second CED optimization pass, compared with the already optimized CED.

The reference keeps recurrent sparse attention and masked MoE counts enabled.
Only metadata validation / QB threshold reuse / training mHC fusion differ. Uses the existing restored
BaseTrainer FB/window numerical and ABBA protocol. No GPU work occurs on import.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_inventory():
    """Host-only scan. Command text is used for classification, never logged."""
    raw = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,stat=,command="], text=True
    )
    processes = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        try:
            processes.append(
                dict(
                    pid=int(parts[0]),
                    ppid=int(parts[1]),
                    state=parts[2],
                    command=parts[3],
                )
            )
        except ValueError:
            continue
    return processes


def gpu_capable_process(process, excluded=()):
    """Conservative workload detection, not a GPU-utilization measurement.

    Python stdin/-c jobs are opaque and are treated as potentially using GPU.
    Only clearly identified host inspection/compile helpers are exempted.
    """
    if process["pid"] in excluded or "Z" in process["state"]:
        return False
    try:
        argv = shlex.split(process["command"])
    except ValueError:
        argv = process["command"].split()
    if not argv:
        return False
    executable = Path(argv[0]).name.lower()
    python = bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable))
    if python:
        if len(argv) >= 3 and argv[1:3] in (["-m", "py_compile"], ["-m", "compileall"]):
            return False
        if (
            len(argv) >= 3
            and Path(argv[1]).name == "check_repo.py"
            and argv[2] in ("doctor", "list", "--help")
        ):
            return False
        return True
    return executable in {
        "pytest",
        "mlx_lm",
        "ollama",
        "llama-server",
        "llama-cli",
        "vllm",
    } or executable.startswith("train_pretrain")


def safe_process_record(process):
    try:
        argv = shlex.split(process["command"])
    except ValueError:
        argv = process["command"].split()
    # Preserve identity without recording arbitrary CLI credentials/payloads.
    entrypoint = next((Path(a).name for a in argv[1:] if a.endswith(".py")), None)
    return {k: process[k] for k in ("pid", "ppid", "state")} | {
        "executable": Path(argv[0]).name if argv else "unknown",
        "entrypoint": entrypoint,
        "command_sha256": hashlib.sha256(process["command"].encode()).hexdigest(),
    }


class GPUExclusivityGuard:
    """Advisory lock plus before-sample and periodic external-workload checks."""

    def __init__(self, run_dir, interval=0.5):
        self.run_dir = run_dir
        self.interval = interval
        self.started = time.monotonic()
        self.events = []
        self.scope = "startup"
        self.conflicted = False
        self.stop = threading.Event()
        self.mutex = threading.Lock()
        processes = process_inventory()
        parents = {p["pid"]: p["ppid"] for p in processes}
        self.excluded = {os.getpid()}
        ancestor = os.getppid()
        while ancestor > 0 and ancestor not in self.excluded:
            self.excluded.add(ancestor)
            ancestor = parents.get(ancestor, 0)

    def _save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "gpu_exclusivity.json").write_text(
            json.dumps(self.report(), indent=2) + "\n"
        )

    def report(self):
        return {
            "pid": os.getpid(),
            "lock": "/tmp/viby-mlx-gpu.lock",
            "poll_seconds": self.interval,
            "conflicted": self.conflicted,
            "events": list(self.events),
            "limitation": "Conservative ps workload scan plus cooperative file lock, not global per-process GPU telemetry. Unrecognized non-Python jobs or jobs shorter than the poll interval can escape detection. Polling runs on CPU during both arms and its overhead remains in measured wall time.",
        }

    def scan(self, scope=None, fail=True):
        external = [
            safe_process_record(p)
            for p in process_inventory()
            if gpu_capable_process(p, self.excluded)
        ]
        with self.mutex:
            if scope is not None:
                self.scope = scope
            event = dict(
                elapsed_s=time.monotonic() - self.started,
                scope=self.scope,
                external_gpu_capable_processes=external,
            )
            self.events.append(event)
            self.conflicted |= bool(external)
        if external:
            self._save()
        if fail and self.conflicted:
            raise RuntimeError(
                "external GPU-capable process detected; no processes were killed; see gpu_exclusivity.json"
            )

    def raise_if_conflicted(self):
        if self.conflicted:
            raise RuntimeError(
                "GPU overlap detected during measurement; samples are confounded; see gpu_exclusivity.json"
            )

    def _monitor(self):
        while not self.stop.wait(self.interval):
            try:
                self.scan(fail=False)
            except Exception as error:
                with self.mutex:
                    self.conflicted = True
                    self.events.append(
                        dict(
                            elapsed_s=time.monotonic() - self.started,
                            scope=self.scope,
                            monitor_error=type(error).__name__,
                        )
                    )
                self._save()

    @contextmanager
    def hold(self):
        with open("/tmp/viby-mlx-gpu.lock", "a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self.events.append(
                    dict(
                        elapsed_s=time.monotonic() - self.started,
                        scope="startup",
                        lock_busy=True,
                    )
                )
                self.conflicted = True
                self._save()
                raise RuntimeError(
                    "another cooperating GPU task holds /tmp/viby-mlx-gpu.lock"
                ) from error
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid()) + "\n")
            lock.flush()
            self.scan("before_MLX_import")
            self._save()
            monitor = threading.Thread(target=self._monitor, daemon=True)
            monitor.start()
            try:
                yield self
            finally:
                self.stop.set()
                monitor.join(timeout=max(2.0, self.interval * 2))
                self._save()
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--variant",
        choices=("metadata", "qb_threshold", "hc_train", "combined"),
        required=True,
    )
    p.add_argument("--preset", choices=("tiny", "default"), default="default")
    p.add_argument("--batches", type=int, nargs="+", default=[1])
    p.add_argument("--lengths", type=int, nargs="+", default=[1024])
    p.add_argument(
        "--layouts", nargs="+", choices=("plain", "packed", "padded"), default=["plain"]
    )
    p.add_argument(
        "--modes", nargs="+", choices=("fb", "window"), default=["fb", "window"]
    )
    p.add_argument("--optimizer", choices=("adamw", "muon"), default="muon")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--muon-lr", type=float, default=4.333333333333333e-4)
    p.add_argument("--grad-clip", type=float, default=0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--mean-doc-length", type=float, default=200)
    p.add_argument("--pad-fraction", type=float, default=0.25)
    p.add_argument("--cache-limit-gb", type=float, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--block-iters", type=int, default=5)
    p.add_argument("--aa-blocks", type=int, default=1)
    p.add_argument("--aa-iters", type=int, default=3)
    p.add_argument("--poll-seconds", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260912)
    return p


def execute(args, guard):
    from experiments import bench_ced_performance as base
    from experiments.kernel_bench_utils import append_jsonl

    mx = base.mx
    try:
        hc_train = importlib.import_module("model.kernels.hc_train")
    except ModuleNotFoundError:
        hc_train = None
    if args.variant in ("hc_train", "combined") and hc_train is None:
        raise RuntimeError("HC candidate model.kernels.hc_train is unavailable")
    if args.variant in ("metadata", "combined") and not hasattr(
        base.attention, "_RECURRENT_METADATA_FUSED"
    ):
        raise RuntimeError(
            "metadata candidate model.attention._RECURRENT_METADATA_FUSED is unavailable"
        )
    for module, name in (
        (base.attention, "_RECURRENT_OPTIMIZED"),
        (base.moe, "_RECURRENT_MASKED_COUNTS"),
        (base.moe, "_QB_THRESHOLD_REUSE"),
    ):
        if not hasattr(module, name):
            raise RuntimeError(
                f"required candidate/reference flag is unavailable: {module.__name__}.{name}"
            )

    def configure(arm, cfg, trainer):
        base.attention._RECURRENT_OPTIMIZED = True
        base.attention._RECURRENT_SPARSE = True
        base.moe._RECURRENT_MASKED_COUNTS = True
        base.moe._QB_THRESHOLD_REUSE = arm == "optimized" and args.variant in (
            "qb_threshold",
            "combined",
        )
        if hasattr(base.attention, "_RECURRENT_METADATA_FUSED"):
            base.attention._RECURRENT_METADATA_FUSED = (
                arm == "optimized" and args.variant in ("metadata", "combined")
            )
        enabled = arm == "optimized" and args.variant in ("hc_train", "combined")
        if hc_train is not None:
            hc_train._TRAIN_FUSION = enabled
        cfg.ced_recurrent_enabled = True
        for gate in trainer._moe_gates:
            rounds = (
                cfg.ced_recurrent_rounds
                if cfg.n_encoder_layers < gate.layer_idx < cfg.n_layers - 1
                else 1
            )
            gate.qb_stats_rows = max(1, cfg.qb_stats_rows // 2 // rounds)

    original_abba = base.abba_blocks

    def guarded_abba(run_a, run_b, *positional, **kwargs):
        def prepared(before, label):
            def f():
                guard.scan("prepare_" + label)
                if before is not None:
                    before()
                guard.raise_if_conflicted()

            return f

        def measured(run, label):
            def f():
                guard.scope = "sample_" + label
                guard.raise_if_conflicted()
                try:
                    return run()
                finally:
                    guard.scope = "between_samples"
                    guard.raise_if_conflicted()

            return f

        label_a, label_b = kwargs.get("label_a", "A"), kwargs.get("label_b", "B")
        kwargs["before_a"] = prepared(kwargs.get("before_a"), label_a)
        kwargs["before_b"] = prepared(kwargs.get("before_b"), label_b)
        return original_abba(
            measured(run_a, label_a), measured(run_b, label_b), *positional, **kwargs
        )

    base.configure, base.abba_blocks = configure, guarded_abba
    args.token_baseline = False

    def hashes():
        source = Path(__file__).resolve()
        return base.source_hashes() | {
            str(source.relative_to(ROOT)): hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        }

    before = hashes()
    metadata = dict(
        kind="pass2_protocol",
        candidate_variant=args.variant,
        command=sys.argv,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        mlx_version=mx.__version__,
        device=mx.device_info(),
        platform=platform.platform(),
        source_sha256=before,
        flags=base.active_viby_flags(),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        reference="current optimized CED: optimized sparse attention and masked counts enabled; metadata fusion, QB threshold reuse and HC training fusion disabled",
        candidate="same reference plus requested candidate flags; no token-baseline comparison",
        scope="Imported same-snapshot BaseTrainer fb and two-microbatch optimizer-window ABBA/AA, numerical gradients/parameters/optimizer/router checks. No data loader, checkpoint I/O, generation cache, actual FLOPs or efficacy claim. Monitoring overhead runs during both arms.",
    )
    append_jsonl(args.run_dir / "measurements.jsonl", metadata)
    (args.run_dir / "working_tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    )
    guard.scan("before_model_initialization")
    mx.random.seed(args.seed)
    cfg = base.make_config(args)
    model = base.VibyForCausalLM(cfg)
    base.convert_model_dtype(model, args.dtype)
    model.train()
    mx.eval(model.parameters())
    trainer = base.make_trainer(args, cfg, model)
    configure("reference", cfg, trainer)
    batches, _ = base.make_batches(
        cfg,
        args.batches[0],
        args.lengths[0],
        args.layouts[0],
        args.seed,
        args.mean_doc_length,
        args.pad_fraction,
    )
    guard.scan("materializing_reference_snapshot")
    base.run_window(trainer, batches)
    mx.eval(model.parameters(), trainer.optimizer.state, model.moe_bias_stack())
    snapshot = base.snapshot_train_state(model, trainer.optimizer, trainer)
    metadata.update(
        resolved_config=cfg.to_dict(),
        optimizer_groups=base.optimizer_metadata(trainer.optimizer),
        params=base.param_identity(model),
        snapshot_memory=base.memory_snapshot(),
        resolved_trainer_args=vars(trainer.args),
        snapshot_origin="one untimed reference two-microbatch optimizer window",
    )
    results = []
    for batch in args.batches:
        for length in args.lengths:
            for layout in args.layouts:
                guard.scan(f"case_B{batch}_T{length}_{layout}")
                results.append(
                    base.measure_case(
                        args,
                        cfg,
                        model,
                        trainer,
                        snapshot,
                        batch,
                        length,
                        layout,
                        args.seed + len(results) * 11,
                    )
                )
                guard.scan("case_completed")
                report = dict(
                    protocol=metadata,
                    results=results,
                    gpu_exclusivity=guard.report(),
                    source_unchanged_during_measurement=before == hashes(),
                )
                (args.run_dir / "results.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
                )
    print(str(args.run_dir / "results.json"), flush=True)


def main():
    p = parser()
    args = p.parse_args()
    if (
        min(
            *args.batches,
            *args.lengths,
            args.warmup,
            args.blocks,
            args.block_iters,
            args.aa_blocks,
            args.aa_iters,
        )
        < 1
        or not 0 <= args.pad_fraction < 0.75
        or not 0.1 <= args.poll_seconds <= 5
    ):
        p.error(
            "positive shapes/counts, 0 <= pad_fraction < 0.75 and 0.1 <= poll_seconds <= 5 required"
        )
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.learning_rate, args.muon_lr, args.mean_doc_length)
    ):
        p.error("learning rates and mean document length must be finite and positive")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    guard = GPUExclusivityGuard(args.run_dir, args.poll_seconds)
    try:
        with guard.hold():
            execute(args, guard)
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
