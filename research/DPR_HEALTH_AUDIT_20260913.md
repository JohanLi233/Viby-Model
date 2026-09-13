# DPR v2 full-run health and throughput audit

2026-09-13. This audit uses the actual uninterrupted continuation launched at
12:27:56, beginning at microstep 10262, with batch 14, accumulation 2 and no
gradient clipping. Raw audit data: `research_runs/dpr_health_audit_20260913/audit.json`.
Source logs: `research_runs/viby_dpr_jepa_v2/launches/20260913_122756/train.log`
and the rows containing `mem/active_gb` in the run's `dpr_metrics.jsonl`.

## Gradient spikes and numerical stability

The largest observed update-window norm is 2.626624 at step 14173. It returns
to 0.941 at the next update, step 14175. Another isolated high point is
2.278502 at 11273. The value is repeated on the following microstep because
the optimizer updates every two microbatches; that is not another spike.
No NaN/Inf metrics or newly skipped optimizer updates were found in this
continuation. The six optimizer clocks agree. They inherit an offset of one
update from the incoming checkpoint; no additional offset develops here.

At 14173, component norms are: other backbone parameters 2.267113, attention
1.321356, router 0.067308, FFN 0.047678, DPR predictor 0.043347 and DPR output
0.068504. DPR-only parameters contribute approximately 0.1% of squared global
norm. This does not isolate the loss source: auxiliary gradients also reach
the backbone. Old logs did not store the largest tensor name below norm 10.

The gradient median falls from approximately 0.78 at 14000–14999 to 0.48 at
16000–16999 and 0.44 at 17000–17999. The late tail is approximately 0.46.
This run has not repeated v1's sustained late increase. Isolated spikes still
exist; finite norms alone do not establish semantic quality.

## Throughput events

The regular low points at 11000/11001, 12000/12001, ... 19000/19001 follow
synchronous checkpoint writes at steps 10999, 11999, ... 18999 exactly. This
run saves every 1000 microbatches; v1 saved every 10000. Saving writes model and
optimizer tensors and hashes the artifacts. The wall-clock throughput metric
includes this time, and its EMA smears the stall into subsequent points.
These are real I/O interruptions, not evidence of gradient corruption.

The isolated slow interval 19575–19577 is not adjacent to a save. Its gradient
norm stays approximately 0.49, and active/cached/peak allocation stays near
its normal range. The run recovers afterward. The existing logs do not separate
data wait, GPU work, allocator activity and OS scheduling, so its cause is
unresolved. System swap usage at audit time is not a time-local swap trace and
cannot explain this earlier event by itself.

Yellow is batch 14; blue v1 was batch 16. Nominal tokens/sec equals microsteps/sec
times B*T. The smaller-batch run can have faster steps while lower token
throughput. Optimizer precision, JEPA target, saving frequency and cache policy
also changed, so these curves do not isolate the gradient-lifetime switch.

## Confirmed resume/data-layout defect

The original migrated checkpoint completed 10000 batches of 16, or 160000
packed samples. The first v2 continuation used batch 14 but still skipped
10000 microbatches, resuming at sample 140000. The saved epoch shuffle RNG
states match, and the loader always shuffles then slices by batch size.
For this same dataset/order, approximately 20000 packed samples (20,480,000
nominal token positions, not all necessarily valid labels) are replayed.

The predicted boundary is 160000/14 = 11428.57 microbatches. CE is approximately
1.776 at 11200–11399 and 1.804 at 11400–11427, then 2.169 at 11429–11599.
The alignment supports replay as the explanation of the early low-loss region;
it cannot be advertised as JEPA token-efficiency improvement. The later
12:27 continuation kept the incoming batch 14; the defect originated during
the earlier 16-to-14 change. The prior startup audit should have caught it.

Also, B14 makes the epoch 20037 microsteps, but the preserved LR horizon is
17533. This leaves 2504 microsteps at the minimum LR. That configuration is
legal but is not an equal-token annealing comparison with v1.

## JEPA representation diagnostics are not all healthy

| Window | NTP CE | Target variance | Covariance/mean diagnostic | Kernel score |
| --- | --- | --- | --- | --- |
| 15000–15999 | 1.9614 | 0.4994 | 1.1005 | 0.2557 |
| 16000–16999 | 1.8768 | 0.4993 | 1.0955 | 0.2305 |
| 18000–18999 | 1.8379 | 0.2967 | 1.5397 | 0.1536 |
| 19000–late tail | ~1.833 | ~0.213 | ~1.750 | ~0.125 |

Coverage stays near 98.23%, lambda is 0.05, particle entropy stays near 0.99
nats (maximum log(4)=1.386), and particle spread remains nonzero. Target count
is ample; the old `cov_rank_deficient=0` only checks N>r, not actual matrix rank.

Target variance decreasing while the covariance/mean diagnostic rises is a
representation concern. v2's target is detached and covariance is not a loss,
so the old target-side optimization feedback is absent. Encoder drift can
still change targets. Existing aggregate logs cannot distinguish a shared mean
shift from increasingly concentrated covariance; low kernel loss is not
evidence of more useful future prediction. CE currently remains finite/stable.

At logging points, active allocation alternates around 9.68/12.0 GiB, idle cache
is near 4 GiB, and peak allocation stays at 36.435 GiB. No memory-growth trend
was observed. Rounded max-load counters imply a typical worst-expert ratio of
approximately 2.3x mean, with isolated larger values; no sustained tail routing
collapse is visible. Exact load CV and empty-expert counts are uploaded to
SwanLab but were not retained in the local JSONL examined here.

## Changes from this audit

- Added a host-only same-stage resume guard against changing batch, sequence,
  seed, packing/document layout or data path while preserving a microstep
  cursor. It rejects unsafe continuation; it does not invent missing sample
  cursor history or undo already replayed training.
- Added target mean-square, second moment and covariance effective-rank
  diagnostics, reusing the existing moments and avoiding eigen decomposition.
  The training objective is unchanged.
- Added the largest gradient tensor name/norm to local DPR records even below
  the console warning threshold, and checkpoint duration/step metrics.
  Wall-clock `tokens_per_sec` retains real save overhead rather than hiding it.

The running process was not hot-reloaded or restarted. Its loaded code is
preserved in the launch `source.zip`; newly edited files on disk are not proof
that the running process used these additional diagnostics. The new checks and
diagnostics apply on the next process launch. The existing run was allowed to
finish without modifying clipping, the target objective or the late LR schedule.

The process exited successfully at 2026-09-13 17:25:22 +08:00, saving microstep
20036. Its last logged pre-clip norm was 0.486 and CE was 1.826. Inspection
also found that the old pretraining loop discarded the final incomplete
accumulation window (20037 microbatches with accumulation=2). The loop now
flushes and rescales that short window for all training types, not only TailSFT.
This fixes future runs; the completed checkpoint was not overwritten or given
an unrequested extra update. The inherited one-update clock offset is separate
from this final incomplete window and was already present at resume.

Validation after the run ended: 60 tests and 3 unittest subtests passed in
4.77s, covering DPR v1/v2, the new moment diagnostics, resume layout rejection,
partial final windows, gradient lifetimes, QB, checkpoint and PSR logging paths.

## Interpreting the absence of a demonstrated quality gain

Current DPR is a deterministic per-token transformation of the existing CED
state, with a single linear particle head and a 132-feature residual readout
at the default M=4/r=32. Its output lies in a subspace of rank at most 132.
It does not add a separate concept-attention stack or additional context reads.
Such a residual can help through optimization bias, but there is no structural
guarantee that it contributes computation the existing decoder lacks.

The fixed 32-dimensional target projection and short future averaging are not
trained to preserve token-prediction sufficiency. Predictable shared components
can dominate this target while carrying little additional useful supervision.
Stop-gradient prevents the direct target-side update within a backward pass;
it does not freeze the shared encoder across training steps. Thus encoder drift
can still make the target easier without improving next-token prediction.

The observed target-variance reduction supports concern about this proxy, but
does not prove it caused the absent gain. The residual remains directly
trainable by NTP; actual use or bypass has not been measured by an intervention.
The extra objective's gradient alignment with CE has not been measured either.
These are mechanism-based explanations, not a proof that JEPA generally fails.

The run has no demonstrated token-efficiency benefit, and its batch/layout
confounds prevent a precise paired effect estimate. Numerical correctness and
reduced auxiliary compute should not be promoted to language-quality gains.
No further training budget was launched during this audit.
