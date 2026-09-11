# Sinkhorn Metal fusion — 2026-09-10

Implemented `model/kernels/sinkhorn_fused.py`, integrated through `model/hc.py`.
Default enabled for FP32 4×4 matrices on GPU. Other sizes/dtypes/devices use the
original MLX formula. `VIBY_SINKHORN_KERNEL=0` disables it before launch.
20 iterations, row/column order, eps and row-max stabilization are unchanged.
The VJP includes the epsilon denominator and the derivative of row max (ties included).

Forward: one thread per matrix, 40 normalizations inside one dispatch.
Backward: recompute a coalesced normalization tape, then reverse all 40 steps in
one dispatch. No tape is written during inference. Model construction prewarms
both directions eagerly, before `mx.compile` traces.

## Validation

- `.venv/bin/python -m pytest tests/ -q`: **152 passed** (133 existing + 19 new).
- New tests compare compiled forward and VJP to the original graph: 1/5/20
  iterations, eps 1e-6/0.1, uniform/random/extreme logits, transposed inputs.
- `.venv/bin/python experiments/check_sinkhorn_parity.py`: same model and input,
  seed 1234 for model / 7 for data, B2×T256, segment_ids, no MTP.
  Reference loss **8.959578**, gradient norm **6.075557**;
  fused loss **8.959578**, gradient norm **6.079125**.
- The separate original `/tmp/parity.py` run gave fused loss **8.959578** and norm
  **6.082394**, slightly above the requested 6.08 ceiling. Do not interpret that
  ceiling as a deterministic bound: the paired run passed, but atomic scatter
  noise remains a limitation. No tolerance was loosened in kernel tests.
- `git diff --check`: passed.

## Same-process serial compiled A/B

Commands (no concurrent GPU jobs):

```sh
.venv/bin/python experiments/bench_sinkhorn_kernel.py
.venv/bin/python experiments/bench_sinkhorn_kernel.py --model --iters 5
```

A/B uses reference → fused → fused → reference, 2 warmups per block, `mx.eval`
inside timing. Model mode uses one model and identical weights/data, separate
compiled graphs, B4×T1024, segment_ids (mean document length 128), accumulation 2,
optimizer state allocated once and resident, cache limit 8 GiB. No optimizer
updates occur between A/B blocks.

Component value-and-grad, seconds:

| Block | Each round | Min |
|---|---|---|
| Reference 1 | .003248 .003278 .002159 .001506 .001480 | .001480 |
| Fused 1 | .000538 .000556 .000574 .000546 .000542 | .000538 |
| Fused 2 | .000594 .000616 .000597 .000668 .000623 | .000594 |
| Reference 2 | .001373 .001432 .001727 .001489 .001426 | .001373 |

Whole-model forward+backward, seconds:

| Block | Each round | Min | Peak GB |
|---|---|---|---|
| Reference 1 | .867816 .870977 .870718 .875286 .872424 | .867816 | 30.09 |
| Fused 1 | .850867 .856988 .853918 .861854 .854055 | .850867 | 29.85 |
| Fused 2 | .857146 .856456 .855536 .862043 .861218 | .855536 | 29.85 |
| Reference 2 | .871579 .879303 .877402 .879400 .879877 | .871579 | 30.09 |

Best minima: **4720 → 4814 tokens/s (+1.99%)**, step time −1.95%.
This is forward+backward throughput with optimizer state resident; optimizer
update time is excluded. No new end-to-end accumulation-window throughput or
MFU claim is made. Attention and MoE kernels were not modified in this change.
