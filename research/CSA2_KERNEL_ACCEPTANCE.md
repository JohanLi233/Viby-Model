# CSA2 / CED kernel implementation and acceptance

**2026-09-11 follow-up:** the historical restored-window timings below used a
snapshot utility that aliased mutable optimizer-state dictionaries on restore.
Those window measurements do not establish same-state A/B performance. The
snapshot repair and fresh measurements are recorded in
[MOE_TRAINING_REPAIR.md](MOE_TRAINING_REPAIR.md); original records remain intact.

2026-09-11. Apple M4 Max (48 GiB), MLX 0.32.2. This records the completed
implementation and measured defaults after the seven-section algorithm audit.
The baseline for performance includes correctness repairs; the broken
Indexer tile predicate and truncated mHC gradient are never used as a baseline.

## Current implementation

| Section | Implementation |
| --- | --- |
| 1. Attention ownership | Query-owned dQ/Delta/dSinkRow in one kernel; Q/G storage actually aliases the final dQ scratch. QK and dOK sequentially reuse bounded DotPart. NP=D/64, backward BK=16. KV stages Q/G, combines score/value contributions and both head groups in Grad8 before one atomic per occurrence/dimension. Window shard uses t%4; compressed uses bit folding. Empty tiles skip uniformly; final sink reduction is parallel. Forward BK=32 remains optional. |
| 2. Key-owned dKV | Implicit window adjacency, count/recursive GPU scan/fill occurrence CSR, no deduplication, four fixed compressed partials and non-atomic reduction. No neighbor-degree cap or floating-point atomic. Direct comparison did not beat sharded dKV, so opt-in remains. CSR is currently rebuilt per layer; metadata reuse across layers is not implemented. |
| 3. Indexer | Candidate eligibility moves before scoring. Per-key-tile flags have the correct physical stride, independently of decode split count. BQ=4 dense/full-domain and BQ=2 candidate-domain scoring share K without changing per-query D partitions or head reduction order. Decode keeps BQ=1 with static key splits. |
| 4. Fused score/selection | N<=1024 stopped-gradient entry keeps scores in TG storage, derives source block maxima before overwriting scores, pins the original latest block only when length>0, and returns ordered block ids plus token selection. Candidate lists use a whole-TG k-way merge. Training avoids global dense FP32 scores and keep masks. Reuse shares final selection. A separate differentiable compact-score arena has a query-owned VJP that visits compact slots and scatters directly to original global K ids. |
| 5. Exact radix/compact | FP32 adjustment prevents multiplication/subtraction contraction and canonicalizes signed zero. All threshold ties survive. The generic dense entry retains the complete rank domain; the fused Indexer entry ranks eligible slots. Packed output uses GPU offsets and a static-capacity arena, with no host total read or [rows,width,total] broadcast. Attention still consumes row-major metadata. |
| 6. MoE | Original sorting/routing and native gather_mm remain. Inverse + token-owned forward/backward use FP32 accumulation, unique route-gradient writes, and shared weight-gradient reduction. K=1/6, arbitrary feature tails and empty input are covered. Experimental custom expert GEMMs remain disabled. |
| 7. mHC | M>=1024 uses a TG processing four tokens and emits ceil(M/4) FP32 partial rows. The original low-precision partial value is cast before accumulation. Two non-atomic reductions use 128-thread groups, including F=4096. dx/dpre retain the original token algorithm. |

The audit's three primary bugs are fixed: the Attention DotPart out-of-bounds
write, Indexer cross-query/tile validity indexing, and mHC dropping roughly
three quarters of weight-gradient contributions at M>=1024. The additional
empty-latest-block bug came from MLX integer division truncating -1/CB to zero.

Logical lengths, offsets and CSR totals stay on GPU. Allocations use static
shape/capacity bounds, including worst-case ties; no claim of exact dynamic
allocation by nnz is made. Every custom VJP preserves array metadata leaves.
Sinkhorn iterations, hc_post, architecture, routing probabilities and
noaux_tc updates are unchanged.

## Inference additions

- Decode still calls the original fixed-k argpartition. A Metal postprocessor
  fuses only position sorting, eligibility checking and offset encoding. It
  does not substitute training's variable-length threshold semantics.
- Window and compressed KV gathers, zeroing invalid entries and slot-mask
  output are combined in one kernel; SDPA itself retains its original dtype.
- Small-M inference in eval mode uses native gather_mm for selected experts
  rather than multiplying all experts. A small [E,M,F] zero-filled contribution
  view preserves the original expert-axis low-precision reduction order;
  selected positions are uniquely stored without floating-point atomics.
  This extra view deliberately avoids changing the summation grouping, which
  otherwise changed downstream logits despite identical expert GEMM results.
- Prefill uses fused Indexer selection and materializes only the boolean mask
  needed by native SDPA. It preserves the existing SDPA/output dtype path;
  substituting the training Attention output cast before inverse RoPE was
  rejected during parity checks. No such experimental prefill path remains.

## Defaults and fallbacks

| Environment variable | Default |
| --- | --- |
| VIBY_SPARSE_ATTN_BWD_SPLIT | 1 |
| VIBY_SPARSE_ATTN_KEY_BWD | 0 |
| VIBY_SPARSE_ATTN_KEY_TILE | 16 |
| VIBY_SPARSE_TOPK_KERNEL | 1 |
| VIBY_INDEXER_SELECT | 1 |
| VIBY_INDEXER_BQ | 1 |
| VIBY_MOE_COMBINE_KERNEL | 1 |
| VIBY_HC_GROUPED_DW | 1 |
| VIBY_DECODE_METADATA | 1 |
| VIBY_PREFILL_SELECT | 1 |
| VIBY_MOE_DECODE_GATHER | 1 |
| VIBY_MOE_KERNEL (experimental expert GEMM) | 0 |

Each new optimization can be disabled independently with its flag set to 0.
Fused Indexer selection requires supported GPU shapes, T>1 and N<=1024;
unsupported shapes retain the dense path. Decode selected-expert GEMMs require
eval mode, matching model/activation dtypes and the existing small-M branch.

## Numerical validation

**172 tests passed** across the following two focused commands:

```sh
.venv/bin/python -m pytest -q \
  tests/test_sparse_attention_kernel.py tests/test_sparse_attention_key_owned.py \
  tests/test_kernel_plan_regressions.py tests/test_indexer_kernel.py \
  tests/test_indexer_select.py tests/test_inference_kernel_optimizations.py \
  tests/test_v41_attention.py tests/test_v41_hc.py tests/test_v41_moe.py tests/test_v41_train.py
# 147 passed

.venv/bin/python -m pytest -q \
  tests/test_v41_consistency.py tests/test_v41_mhc_single_pass.py tests/test_v41_xsa.py
# 25 passed
```

The tests include original-order Metal score equality and exact keep/index
sets, disjoint candidate blocks, non-aligned tails, zero lengths, all boundary
ties, unreachable high scores in the generic dense rank domain, signed zero
and infinities, compiled packed output with changing GPU lengths, compact-score
VJPs, Full/Reuse/Reindex composition, high-degree CSR, empty gradients, mHC
M=1023/1024/1027 and F=4096, and forward BK=32 with bounded BK=16 backward.

Inference module and full-model multi-step checks require exact equality for
the tested outputs. In the base-model inference benchmark below, prefill and
all eight teacher-forced decode logits also had max absolute difference 0.
Both arms use the same deterministic prefill route combine to avoid unrelated
native bf16 scatter-arrival noise. Attention/dKV training reductions retain
the specified operand casts but do not promise general bitwise gradient equality.

## Performance protocol and results

All comparisons use one process/model, fixed input tensors, fixed weights,
materialized outputs, and ABBA blocks. Each slot has five timed repetitions;
there are three ABBA blocks. Training warmup is three per arm; final inference
decode warmup is five. The primary ratio is the median of paired block B/A
ratios. Absolute first/last A drift above 3% rejects a run. Raw samples,
per-block ratios, memory observations and numerical summaries are retained.

Training: B=4, T=1024, mean document length 200, bf16, compiled loss/VJP,
8 GiB allocator cache, resident optimizer state, accumulation=2. The baseline
already includes all correctness repairs and grouped mHC. Fwd+bwd comparisons
do not update weights. Whole windows restore identical parameters, optimizer,
router biases and explicit-Nesterov state before every measurement, then
execute two microbatches plus the actual optimizer/bias update.

| Comparison | A median | B median | Paired elapsed change | A drift |
| --- | ---: | ---: | ---: | ---: |
| Corrected baseline -> fused Indexer + MoE combine, fwd+bwd | 499.34 ms | 481.41 ms | -3.68% | 2.01% |
| Same comparison, complete accumulation/optimizer window | 1386.92 ms | 1358.17 ms | -1.92% | 1.89% |
| Sharded -> key-owned dKV, otherwise identical optimized stack | See raw direct record | See raw direct record | +1.43% (slower) | 0.18% |

Fused Indexer alone had a paired estimate near -0.7% with mixed block results;
it does not establish an independent reliable latency win. The combined stack
improved in all three blocks for both training comparisons above.

Inference: base model in eval mode, bf16, B=1, context=1024, eight fixed
teacher-forced tokens, eager public prefill/decode APIs. Each decode sample
restores the same fully materialized prefix cache. The model is approximately
1.23B parameters; trainable parameter count matches the training benchmark
(additional inference maximum-position buffers account for the total difference).

| Comparison | A median | B median | Paired elapsed change | A drift |
| --- | ---: | ---: | ---: | ---: |
| Decode: original all-expert branch -> selected expert GEMMs + metadata fusion, eight tokens | 83.41 ms | 55.27 ms | -33.51% | 0.24% |
| Decode: selected expert GEMMs -> add metadata fusion, direct comparison | 59.94 ms | 55.71 ms | -6.79% | 0.005% |
| Prefill: native score/selection -> fused score/selection, otherwise same stack | 64.34 ms | 64.51 ms | approximately 0% | 1.56% |

Final decode medians correspond to about **10.43 -> 6.91 ms/token**, or
**95.9 -> 144.8 tokens/s**. These are this protocol's results, not a universal
throughput promise. Prefill removes global FP32 Indexer scores, but no stable
prefill latency improvement was measured. An earlier metadata-only run and an
earlier combined decode run exceeded the 3% drift limit; their results are
retained but excluded from the accepted performance claim.

### Raw records and reproduction

- [Training fwd+bwd variants](../research_runs/csa2_plan_20260911_fb/results.jsonl)
- [Complete training window](../research_runs/csa2_plan_20260911_window/results.jsonl)
- [Direct dKV backend comparison](../research_runs/csa2_plan_20260911_key_direct/results.jsonl)
- [Initial inference comparisons, including rejected drift runs](../research_runs/csa2_plan_20260911_inference/results.jsonl)
- [Direct metadata increment](../research_runs/csa2_plan_20260911_inference_direct/results.jsonl)
- [Accepted final decode comparison](../research_runs/csa2_plan_20260911_inference_final/results.jsonl)

```sh
.venv/bin/python experiments/bench_csa2_plan.py --run-dir research_runs/csa2_fb_repro
.venv/bin/python experiments/bench_csa2_plan.py --run-dir research_runs/csa2_window_repro --mode window --variants fused_combine
.venv/bin/python experiments/bench_csa2_plan.py --run-dir research_runs/csa2_key_repro --reference fused_combine --variants key_owned
.venv/bin/python experiments/bench_csa2_inference.py --run-dir research_runs/csa2_inference_repro --mode decode --variants combined --warmup 5
```

Weights are fixed initialized weights (training materializes optimizer state
with one update), not a production trained checkpoint. This acceptance does
not establish long-run convergence, holdout quality, compiled serving latency,
other hardware/context lengths, or a full-repository test result. No training
checkpoints, model architecture, or external services were changed.
