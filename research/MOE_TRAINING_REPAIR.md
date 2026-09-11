# MoE correctness repairs and training-window optimization

2026-09-11, Apple M4 Max 48 GiB, MLX 0.32.2.

## Repairs

- Empty route gathers now use an empty reshape, whose adjoint is valid.
  MLX 0.32.2's native empty take adjoint raises a scatter shape error.
- Compiled small-M decode falls back to eager for FP32: fused arithmetic was
  changing output bits. bf16/fp16 retain the compiled experiment. The Metal
  tests also now unpack the list returned by `mx.vjp` correctly; several of
  the original failures were in the test harness, not the Metal arithmetic.
- Optimizer restore clones dictionary/list containers on every restore.
  Previously `optimizer.state = snap["optimizer"]` let optimizer updates mutate
  the frozen snapshot. Prior restored-window results, including the user's
  33.8%-drift run and historical CSA2 windows, cannot establish same-state A/B.
  A regression replays three Adam updates and checks the frozen state and the
  updated parameters. A/A benchmarks also use distinct slot labels; previously
  identical labels left the B sample group empty. The generic benchmark now
  seeds model initialization as requested by its CLI.
- Gradient norms use FP32 square/reduction, including across parameter leaves.
  The old bf16 square/sum rounded the norm and FP16 could overflow. Large
  tensors now use a one-read Metal reduction without full-size FP32 temporaries.
  The no-clipping path still leaves gradients unchanged and skips nonfinite
  updates. The existing clipping path is unchanged.

## Optimization

`trainer/fast_norm.py` adds fused square reductions for Sinkhorn optimizer row
and column norms. Row reductions use one SIMD group per row; column reductions
read coalesced columns in 256-row tiles and reduce compact FP32 partials.
This removes repeated full-size cast/square arrays for the large Engram tables.

All 11 normalization iterations, momentum, near-zero-row masking, epsilon,
division dtype, update scaling and optimizer state structure remain in place.
Floating-point reduction grouping can differ; this is not bitwise training
equivalence or evidence of long-run convergence. `VIBY_OPT_SINKHORN_FAST_NORM=0`
retains the native optimizer reductions. The new optimizer reduction defaults
to **1** after the standalone full-window repeat passed. The MoE dataflow and
compiled-decode switches remain opt-in and independent.

## Measurements

Same model/process, bf16, B=4, T=1024, mean packed-document length 200,
accumulation=2, resident optimizer state, 8 GiB allocator cache. Each arm gets
five warmups, then three ABBA blocks with five repetitions per slot. Windows
restore parameters, optimizer state, router biases and explicit-Nesterov state
outside every timed invocation. Report paired-block ratios, not a ratio of
separately selected minima. A/A absolute drift above 3% rejects the timing.

| Comparison | A median | B median | Paired time change | A/A drift |
| --- | ---: | ---: | ---: | ---: |
| MoE dataflow alone, fwd+bwd | 460.55 ms | 454.03 ms | -0.97% | 0.43% |
| Large table Sinkhorn update, isolated | 50.36 ms | 25.12 ms | -49.90% | 2.67% |
| MoE dataflow + optimizer norms, complete window, initial | 1363.11 ms | 1262.34 ms | -7.88% | 2.79% |
| Same combination, with FP32 gradient norm repair and FLOP audit | 1360.53 ms | 1244.90 ms | -9.94% | 1.77% |
| Optimizer norms alone, audited repeat; promoted default | 1325.53 ms | 1223.29 ms | -8.31% | 0.24% |

The MoE-only block ratios were 0.9903, 0.9831, 1.0101: small and mixed, so this
does not justify enabling those switches by default. The audited combined
window improved in all three blocks (0.8879, 0.9006, 0.9024). Observed peak active
allocation fell from 21.71 to 19.97 GB (allocator cache is recorded separately).
These peaks include the retained frozen A/B snapshot.

The first optimizer-only full-window run had 23.10% A/A drift. Its apparent
2.26% paired improvement is rejected; raw results are retained. A repeat is
recorded separately, without overwriting that failure. The repeat passed with
0.24% drift and 8.31% paired improvement. The optimizer-only repeat reached
6180 -> 6697 tokens/s, estimated MFU 35.93% -> 38.93%. Its post-window parameter
relative L2 difference was 0.00031328, with unchanged router bias updates.

After one full window, combined-arm parameter max difference was 0.0009765625,
relative L2 was 0.00034445, and router-bias max difference was zero. Initial
forward loss was identical. The isolated 491873-by-256 table update had exactly
equal momentum, parameter RMS difference 8.70e-8 and max difference 0.0009765625.
These are fixed initialized-model/short-update checks, not a training-quality
claim. Old and new runs have nondeterministic low-precision gradient reductions;
their initial losses must not be compared as if initialization updates were
bitwise reproducible across processes.

## MFU and the 70% target

The previous FLOP estimator charged the whole compressed pool as dense
attention. It now uses nominal sparse lengths for ordinary logs and accepts
measured per-layer lengths. The audit in `experiments/audit_training_flops.py`
counts actual compact-selection occurrences (including threshold ties) and
valid document/padding-masked window positions outside timing. It currently
supports the backbone-only indexed training protocol. Generic DSpark estimates
also account for block expansion and exclude its Markov lookup embedding.

The remaining GEMM term is explicitly a **6N estimate**, not an exact operator
FLOP audit. The denominator remains the declared **13.5 TFLOPS** empirical GEMM
reference; no new hardware-peak calibration is claimed. Optimizer/recompute
FLOPs are excluded from the numerator; complete-window time includes overhead.

For the audited two microbatches, the estimate averages 0.784867284 GFLOPs/token.
The combined window delivers 6021 -> 6580 tokens/s and estimated MFU
**35.01% -> 38.26%**. Reaching 70% under this same convention requires a window
at or below **0.6804 s**, another **45.3%** latency reduction from 1.2449 s.
This round has **not reached 70%**. An optimizer-only improvement cannot close
that gap: even two current ~0.46-s forward/backward passes exceed the entire
70% window budget before accumulation or optimizer work.

`bench_train_step.py` now labels its fwd+bwd MFU as excluding optimizer time
and its separately summed window time as a proxy. Use `bench_csa2_plan.py` for
the measured complete-window metric.

## Reproduction and records

**132 focused tests passed** across the command below (130 plus the two FLOP
accounting checks). This is not a full-repository test run.

```sh
.venv/bin/python -m pytest -q \
  tests/test_moe_dataflow_host.py tests/test_moe_dataflow_metal.py \
  tests/test_kernel_bench_state.py tests/test_optimizer_fast_norm.py \
  tests/test_training_flops.py tests/test_campaign_optim.py tests/test_v41_train.py

.venv/bin/python experiments/bench_csa2_plan.py \
  --run-dir research_runs/optimizer_norm_repro --mode window \
  --reference fused_combine --variants optimizer_norm --warmup 5

.venv/bin/python experiments/bench_csa2_plan.py \
  --run-dir research_runs/combined_norm_repro --mode window \
  --reference fused_combine --variants dataflow_optimizer --warmup 5
```

- [MoE-only fwd+bwd](../research_runs/moe_repair_20260911_fb/results.jsonl)
- [Isolated optimizer](../research_runs/optimizer_norm_20260911/results.jsonl)
- [Initial fixed-snapshot window](../research_runs/moe_repair_20260911_window/results.jsonl)
- [Audited combined window](../research_runs/moe_repair_20260911_audited_window/results.jsonl)
- [Optimizer-only window](../research_runs/moe_repair_20260911_optimizer_only/results.jsonl)
- [Optimizer-only repeat](../research_runs/moe_repair_20260911_optimizer_only_repeat/results.jsonl)

No checkpoint was saved and no long training run was performed.
