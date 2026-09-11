# MoE dataflow optimizations (experimental)

Base: `b7c5019537f570de12cef555f987633c9d0ba95c`, 2026-09-11.
This change implements three independently switchable paths. It does **not**
claim 80% MFU or a measured training/decode speedup. The existing accepted
numbers in [CSA2_KERNEL_ACCEPTANCE.md](CSA2_KERNEL_ACCEPTANCE.md) belong to the
base commit, not to this change.

## Implemented changes

### Token-owned route-gather VJP

`model/kernels/moe_gather.py` retains the original forward gather and native,
sorted, single-index `gather_mm`. It does not replace expert GEMMs and does not
pass both lhs/rhs indices to the sorted training path.

For `order[sorted_row] = token*K + choice`, the inverse satisfies
`inverse[token*K + choice] = sorted_row`. Each backward thread owns one
`dx[token, feature]`, sums its K occurrences in FP32, and casts once to the
input dtype. No floating-point atomic, output initialization, or threadgroup
storage/barrier is used. The inverse is shared with the existing route combine;
with the flag off, the original late inverse construction remains in place.
Empty arrays and unsupported dtype/device paths use native gather. Every
custom-VJP array primal has a matching array cotangent, including metadata.

This removes scatter-style accumulation from this adjoint, **not** the forward
route tensor. At B=4, T=1024, K=6, D=1024, bf16, that forward tensor remains
48 MiB. FP32 accumulation order changes are expected; bitwise training-gradient
equality and convergence are not asserted.

### Compact sequence counts, shared with noaux_tc

`model/kernels/moe_counts.py` counts route occurrences directly into sequence
histograms. On Metal, 128 threads process 256 routes with an integer TG
histogram; all partial outputs are uniquely written. At E<=512 TG storage is
at most 2 KiB. Larger E and CPU use a compact native scatter.

At B=4, T=1024, E=96, K=6, the old FP32 incidence matrix is 1.5 MiB. The new
integer partials are 36 KiB, followed by a 1.5 KiB [B,E] result. The existing
sequence objective uses the same mean scores and `/T` then `/K` normalization.
The [B,E] result also supplies the router's global load side channel, allowing
the lazy graph to discard the superseded global scatter. Routing, probabilities,
loss weighting and the bias update equation do not change. Repeated assignments
are counted, not deduplicated. Above 2**24 routes, integration retains the old
path rather than changing FP32 increment semantics. FP32 loss/gradient rounding
may differ slightly after moving the count reduction.

### Cached pure decode MoE region

`model/kernels/moe_decode.py` compiles a shape-aware region containing selected
expert GEMMs, the original expert-axis contribution sum, the shared expert and
the final add. Mutable model arrays are explicit arguments, not closure state.
Routing remains outside the region. The growing attention KV pool is not an
input, so changing context length alone does not create another MoE graph.
Shared and routed clamping limits remain independently represented.

The original low-precision expert contribution view and reduction order remain;
this does not substitute a different K-way sum. Only eval mode, matching dtypes,
Metal and 1..8 tokens (the module's existing small-M threshold) are eligible.
All other cases retain the eager path. Exact decode parity is an acceptance
gate, not a result demonstrated on the implementation host.

## Switches and validation status

All three default to **0** until Apple-hardware acceptance:

| Flag | Effect |
| --- | --- |
| `VIBY_MOE_GATHER_VJP=1` | Token-owned input-gradient reduction |
| `VIBY_MOE_COMPACT_AUX=1` | Compact auxiliary counts and shared global load |
| `VIBY_MOE_DECODE_COMPILE=1` | Pure compiled small-M eval region |

Setting them to zero retains the old arithmetic paths. Sparse Attention stays
sharded by default. Selection, CED cache ownership, Sinkhorn iteration/cast
semantics, model structure and checkpoints are unchanged.

On the Linux x86_64 implementation host, syntax checks and **29 host tests**
passed. The Metal regression module was **skipped** because MLX/Apple Metal is
unavailable. Host tests exercise the actual Python fallback/dispatch code and
an independent NumPy adjoint identity; they do not emulate Metal arithmetic,
prove GPU compilation, or measure performance. The baseline's 172 tests were
not re-run. No Apple numerical/performance result is claimed for this patch.

Metal gates include gather forward/VJP and compiled-VJP checks, metadata leaf
alignment, tails/empty inputs, hot experts and repeated occurrences, sequence
loss/score gradients, compiled load side channels, changed weights/clamping
through a decode cache hit, and full-model prefill plus eight decode steps.

## Incremental acceptance commands

```sh
python -m pytest -q tests/test_moe_dataflow_host.py tests/test_moe_dataflow_metal.py

# Compare against the current optimized stack, not the pre-CSA2 baseline.
python experiments/bench_csa2_plan.py --run-dir research_runs/moe_dataflow_fb \
  --reference fused_combine --variants dataflow_gather dataflow_counts dataflow_combined
python experiments/bench_csa2_plan.py --run-dir research_runs/moe_dataflow_window \
  --mode window --reference fused_combine --variants dataflow_combined
python experiments/bench_csa2_inference.py --run-dir research_runs/moe_dataflow_decode \
  --mode decode --reference combined --variants compiled_moe --warmup 5
```

The existing ABBA/restored-state protocol is retained. The decode benchmark
refuses to time the compiled variant if its checked logits differ. Promotion
requires the Metal gates, existing regression suite, stable full-window/decode
wins and memory checks; host success alone is insufficient.

## MFU is not an acceptance shortcut

`trainer/flops.py` still assumes the older dense compressed-attention workload,
while the default training stack now uses indexed sparse attention. Its default
13.5 TFLOPS denominator is an empirically measured bf16 GEMM reference, not a
new measurement in this PR. The FLOP estimator and denominator are left alone;
logging 80% with stale workload accounting would not establish the target.

For an audited useful-FLOP count F per window and a declared peak P, 80% MFU
requires window latency <= F/(0.8*P). Count actual sparse occurrences (including
boundary ties), rather than silently replacing them by k or the old dense pool.
Optimizer/recomputation work is overhead for model-FLOP utilization, not extra
useful FLOPs to add merely to raise the metric.

The base acceptance medians give 1358.17 - 2*481.41 = 395.35 ms outside two
isolated forward/backward passes. This subtraction is only a rough window
budget, **not** a separate optimizer profile; it includes accumulation and
other window work. These MoE changes do not remove that entire budget.
