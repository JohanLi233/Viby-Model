# DPR memory lifetime repair and resumed training

2026-09-13. Based on `ae1ab42` plus the current uncommitted stability/v2 changes.
No precision reduction, optimizer recipe change, activation approximation, or
additional training sweep. User explicitly requested `grad_clip=0`.

## Measurement and change

The corrected Adam body is already compiled into a minimal allocation path.
For one `[96,256,1024]` expert tensor (BF16 parameters/gradients, FP32 moments),
warmed isolated evaluation measured 301,989,924 input-active bytes and a peak
of 553,648,164 bytes. The increment is exactly the three necessary outputs:
251,658,240 bytes. A replacement Metal kernel cannot remove those outputs
while retaining immutable-array semantics and the same state precision.
No redundant custom Adam kernel was introduced.

The epoch loop retained its last microbatch gradient tree after accumulation,
through the optimizer and into the following forward. It now releases that
Python reference once the accumulator owns the result. Eager accumulation has
already materialized its sum; lazy accumulation retains dependencies through
the sum graph. Accumulation=1 still retains the gradient through the accumulator.
No in-place mutation or gradient truncation is used.

The diagnostic reference switch is `VIBY_RELEASE_MICROBATCH_GRADS=0`; corrected
behavior defaults to 1. Existing FP32 Adam states and fixed Muon radii remain.
DPR JSONL now includes the existing memory metrics in addition to SwanLab.

## Focused validation

The real epoch loop and real Adam optimizer were driven by fixed contiguous
synthetic gradients (16,777,216 BF16 elements) for two accumulation windows.
This is an allocation/optimizer check, not a language-model benchmark.

| Mode | Peak active allocation |
| --- | --- |
| Retain the unused gradient | 384.00007 MiB |
| Release the unused gradient | 352.00007 MiB |

The reduction is exactly one 32 MiB gradient array. Final parameter, m and v
values agree. Tests also cover lazy accumulation. Metal may defer buffer
reclamation until the next dispatch, so an immediate entry-time memory counter
does not necessarily fall at the instant the Python reference is dropped.

The saved full-model trainable gradient tree occupies 2.298132 GiB by tensor
header accounting. This is the removable tree's size, not a measured 2.30 GiB
reduction in the full training peak; the peak may occur in a different phase.

43 focused tests passed in 3.27 seconds:

```sh
.venv/bin/python -m pytest -q tests/test_gradient_lifetime.py tests/test_dpr.py tests/test_dpr_contextual.py tests/test_moe_qb.py tests/test_optimizer_step_logging.py
```

Static checks, diff whitespace and changed-file lint passed. Initial allocation
test stubs lacked a tokenizer and used broadcast fills; the final fixture has
the required tokenizer and contiguous arrays. These fixture corrections do not
represent production fixes. Raw allocation samples, kernel probe numbers and
the working patch are in `research_runs/dpr_memory_20260913/evidence.json`.

## Resume

An existing v2 checkpoint was found at microstep 10261, with B=14/T=1024 and
accumulation=2. It was used directly; the older v1 checkpoint was not migrated
again. The launch preserves that saved batch configuration and the existing
17533-microstep LR horizon. Gradient clipping remains disabled.

The allocator's idle cache is capped at 4 GiB (`--cache_limit_gb 4`), rather
than unrestricted retention. This does not cap live tensors or guarantee a
4 GiB process size; it may trade some allocation reuse for lower resident memory.

Launch artifacts, full command, source checkpoint metadata, source hashes,
source ZIP, stdout and process status:
`research_runs/viby_dpr_jepa_v2/launches/20260913_122756/`.
The supervisor updates `launch.json` when the child exits. Startup/running
observations are time-specific, not a guarantee of successful completion.

Observed at 2026-09-13 12:29:34 +08:00: Python training PID 7714, uv PID 7713,
supervisor PID 7712. Reached microstep 10305 (44 resumed microbatches), CE 1.777,
pre-clip gradient norm 1.386, approximately 8,064 tokens/s. Log-point active
MLX allocation alternates around 9.68/12.0 GiB with accumulation; peak since
startup is 36.43 GiB and cached free allocation is approximately 4 GiB.
No NaN/traceback appeared in this startup interval. This is startup evidence,
not annealing stability or a matched full-model peak-memory comparison.
Structured observation: `launches/20260913_122756/observation.json` under the v2 run.
