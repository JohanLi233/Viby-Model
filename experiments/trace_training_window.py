"""Short real-window workload for Metal System Trace, with steady-state logs."""

import json
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from experiments.bench_kernel_optimizations import build_trainer, make_batch, run_window
from experiments.kernel_bench_utils import snapshot_train_state, restore_train_state

ap = argparse.ArgumentParser()
ap.add_argument("--ready-file")
ap.add_argument("--start-file")
ap.add_argument("--loops", type=int, default=12)
control = ap.parse_args()

args = SimpleNamespace(
    batch=4, seq=1024, accum=2, seg=200, cache_limit_gb=4, no_compile=False
)
mx.random.seed(1234)
_, cfg, model, trainer = build_trainer(args)
batches = [
    make_batch(cfg, args.batch, args.seq, 20260912 + i, args.seg)
    for i in range(args.accum)
]
run_window(trainer, batches)
snap = snapshot_train_state(model, trainer.optimizer, trainer)
if control.ready_file:
    Path(control.ready_file).write_text(
        json.dumps(dict(pid=os.getpid(), ready_ns=time.time_ns()))
    )
    print("READY", os.getpid(), flush=True)
if control.start_file:
    deadline = time.monotonic() + 180
    while not Path(control.start_file).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("profiler did not start within 180 seconds")
        time.sleep(0.05)
for i in range(control.loops):
    restore_train_state(model, trainer.optimizer, snap, trainer)
    start = time.time_ns()
    run_window(trainer, batches)
    print(json.dumps(dict(window=i, start_ns=start, end_ns=time.time_ns())), flush=True)
