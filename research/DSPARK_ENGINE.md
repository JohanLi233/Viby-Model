# DSpark speculative decoding in VibyEngine

`SamplingParams(use_mtp_speculative=True)` and the corresponding eval CLI flag
now activate drafting and target verification. A model without MTP modules
fails explicitly; the engine does not instantiate a random draft head as a
replacement for missing checkpoint weights. Module presence can be validated;
whether those weights have been trained remains a checkpoint requirement.

```sh
.venv/bin/python eval_model.py --use_mtp_speculative --num_speculative_tokens 3
# Optional: stop drafting when the confidence probability drops below 0.5.
.venv/bin/python eval_model.py --use_mtp_speculative --num_speculative_tokens 3 --mtp_confidence_threshold 0.5
```

Each round proposes at most `min(num_speculative_tokens, dspark_block_size-1,
remaining_output_budget-1)` new tokens. The default proposal count is one.
The final slot in the output budget is reserved for a target bonus/correction.
Confidence threshold defaults to zero (disabled). A low-confidence round falls
back to normal target decoding, without changing its distribution.

## Draft conditioning

The engine retains the configured backbone target-layer inputs at the last
processed token. Prefix snapshots optionally include those anchor features.
A full prefix hit without features is recomputed when speculation is requested;
partial hits acquire fresh features during continuation.

The sequence always has one emitted token not yet processed by the target.
The draft input starts with `[processed_anchor, pending_token]`, then appends
proposals. Slot 1 predicts the first new proposal, leaving slot 0's already
sampled target prediction unused. `VibyForCausalLM.dspark_draft` uses the same
main projection, stage stack, pre_mix chain, final norm, shared vocabulary head,
and Markov alignment as `_mtp_loss`: previous ids are `[anchor, ids[:-1]]`.
Only the known causal prefix needs to be evaluated; later noise slots cannot
affect its outputs. Current drafting recomputes these short stage prefixes.

## Acceptance and state

Greedy accepts only a proposal equal to the target argmax and otherwise emits
the target argmax. Stochastic sampling applies the request's repetition penalty,
temperature, top-k and top-p to both distributions at the same token history.
Proposal y is accepted with `min(1,p(y)/q(y))`; rejection samples normalized
`max(p-q,0)`. Full acceptance emits a bonus token from the final target row.
This preserves the target sampling law, not identical RNG consumption or
identical stochastic output for a fixed seed.

Verification uses an isolated cache row and feeds `[pending, proposals...]`
through the actual decode path. Each intermediate state retains the window
ring, compressed/index pools, partial compressor state, Engram history and
backbone anchor features. Only the accepted prefix state is restored to the
live row. Truncating a logical length alone would not undo overwritten ring
slots or partial compression groups, so the implementation uses real snapshots.
Python `filled` counts are restored for dense prefix continuation as well.

Only accepted/corrected/bonus tokens enter the output stream, stopping logic
and logprob list. Logprobs retain the existing engine convention: raw target
log-softmax, not proposal scores. Stop-string truncation also truncates logprobs.
EOS, output/context limits, n>1, queue admission and row compaction are respected.
Mixed speculative/ordinary requests are processed through isolated rows during
a speculative scheduler step; purely ordinary batches keep the original path.

Stats now contain real `mtp_drafted`, `mtp_accepted`, `mtp_rejected`,
`mtp_rounds`, and `mtp_confidence_stops` counts. `decode_tokens` and
`batch_steps` count actual target decode work, including discarded verification
positions. Drafts after a rejection are discarded, not counted as accepted.

## Verification boundary

The verifier builds one lazy MLX graph out of sequential target decode calls
and evaluates the target logits together. It does **not** replace decode with
dense prefill: those paths have different fixed-k/tie and compressor-boundary
semantics. It is also **not** a parallel multi-token backbone verifier. Draft
recomputation, snapshots and sequential target work may outweigh acceptance
benefits; no throughput improvement is claimed by this implementation.

Targeted tests cover real draft inference, greedy parity for every rejection
position, complete acceptance and bonus tokens, rolling-window/compressor
rollback, prefixes and mixed queues, EOS/limits/streaming, confidence fallback,
and the positive-residual sampling law. Run:

```sh
.venv/bin/python -m pytest -q tests/test_engine_speculative.py tests/test_v41_train.py tests/test_v41_consistency.py
```

Local validation: **43 passed** on 2026-09-11. CLI help and `git diff --check`
also passed. Tests use small initialized models (including bf16), not a trained
DSpark checkpoint, so they establish functional correctness rather than a
production acceptance rate or throughput gain.
