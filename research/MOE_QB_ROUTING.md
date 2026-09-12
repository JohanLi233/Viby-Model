# Causal QB routing repair

2026-09-12. This changes the training recipe; it does not establish a lower
validation loss or prove the cause of the late-training instability.

## Evidence from the interrupted run

The local `swanlog/run-20260911_210819-7ymtirum/files/config.yaml` matches the
screenshots: 12 backbone gates, 96 experts, top-6, B=16, T=1024, accumulation=2,
bf16, fixed bias update 0.001, WSD schedule, no gradient clipping. The expected
mean is 1024 assignments per expert per microbatch. `max_load_k=10` therefore
means maximum/mean load about 10, rather than a normalized imbalance of 10%.
The saved router weights are BF16; the correction biases are F32.

Local NCP metrics confirm mean NTP loss 1.99946 at steps 16500–16999 versus
2.14913 at 17100–17219. Over those same windows, NCP loss fell from 0.53731 to
0.51767 and VQ loss fell from 0.53848 to 0.51882. This supports investigating
routing and shared training dynamics; it does not identify a unique cause.
The console records interruption at step 17215 and a saved checkpoint. No
training process was running when this repair began; that checkpoint was not
modified or resumed by this work.

## Implementation

- Default `moe_balance_method=qb`. Let `s` be the unbiased affinity and `b` the
  previous window's frozen bias. For each sampled token, find `alpha`, the
  (K+1)-th largest element of `s+b`. The new target for expert e is the negative
  upper K/E column quantile of `s_e-alpha`. Blend half of the centered old bias
  with half of the centered target, then recenter. Integer quantiles use an
  order statistic with ceil(samples*K/E) target assignments.
- This is a simultaneous, one-step approximation with EMA, not exact constrained
  assignment. A fixed sample does not guarantee balance on the next batch,
  especially under ties, homogeneous tokens or a changing data distribution.
- Selection remains top-K of `s+b`; combination weights use the original `s`.
  No tokens are dropped and experts still use native sorted `gather_mm`.
- Export detached margins through the compiled loss outputs. Update biases
  only after a completed optimizer window. Current-sequence statistics never
  affect its own outputs; inference never updates the controller.
- Split the 8192-row per-layer budget over accumulation microbatches. Sample
  uniformly spaced bin centers, retain arrays in a list, concatenate once at
  the window boundary. At 12 layers and 96 experts the retained sample payload
  is at most 36 MiB (concatenation and quantile temporaries are additional).
  At least one sampled row per microbatch is required. This sampling is an
  approximation; periodic or unrepresentative samples can bias the estimate.
- Preserve router weights/biases as fp32 in training conversion and checkpoint
  loading; perform the routing GEMM in fp32. Promoting an old BF16 weight cannot
  recover past rounding, but allows future small updates to accumulate.
- Floor softplus at 1e-20 before sqrt to prevent an infinite derivative at
  underflowed zero. Normalize affinities over experts per token before the
  sequence auxiliary loss, removing the incentive to shrink every score.
- Retain the small sequence auxiliary coefficient 1e-4. This local QB + sequence
  constraint recipe differs from the aux-free Marin example; set
  `--aux_balance_loss_weight 0` for a pure bias-balancing comparison.
- `noaux_tc` retains a fixed negative-feedback sign controller for comparison,
  with centered biases and no update for an empty observation.

The statistics/update interface covers backbone gates in the shared pretrain/SFT
loop. It does not add a new balancing loop for DPO or DSpark-only training.

## Running and monitoring

The existing pretraining command now defaults to QB and fp32 routing. Explicit
equivalent options, with optional gradient clipping for the unstable run:

```sh
--moe_balance_method qb --qb_update_rate 0.5 --qb_stats_rows 8192 \
--router_fp32 --grad_clip 1.0
```

Gradient clipping is still opt-in; its repository default remains 0. Use a
separate output/run directory when comparing against an old checkpoint. Merely
changing source files does not alter an already-running process. Prefer a
checkpoint before the rise for a controlled continuation if one is available;
the local latest checkpoint is already in the unstable tail.

Watch `moe/gateN_max_load_ratio` (1 is uniform), `load_cv`, `empty_experts`,
`grad_norm`, NTP loss and held-out loss. The load metrics describe the current
microbatch, while the controller uses the completed accumulation window.
Grad norm is logged before clipping. `--moe_balance_method noaux_tc` provides
a controller comparison; it still includes the numerical and auxiliary-loss
repairs, so it is not a bitwise replay of the old training code.

## Focused validation

The tests cover quantile order statistics, common-shift invariance, top-K
selection versus weights, compiled dynamic bias inputs, detached statistics,
finite gradients at extreme negative logits, auxiliary score-scale invariance,
old checkpoint dtype promotion, accumulation boundaries and nonfinite skips.
Related MoE dispatch/decode, configuration, training, PSR and checkpoint
tests verify the changed output interface.

**152 focused tests passed** (150 in the combined run, then the two epoch-loop
checks after correcting their test-call argument order). This is not a full
repository suite. The epoch-loop checks run in both eager and compiled modes.
This is a historical validation count; the command below lists the current retained tests.

```sh
.venv/bin/python -m pytest -q \
  tests/test_moe_qb.py tests/test_v41_moe.py tests/test_v41_train.py \
  tests/test_moe_dataflow_metal.py tests/test_v41_config.py \
  tests/test_checkpoint_save.py tests/test_psr.py \
  tests/test_psr_pretrain.py tests/test_campaign_optim.py

.venv/bin/python experiments/probe_moe_qb.py \
  --output research_runs/moe_qb_20260912/probe.json
```

Synthetic controller probe: E=96, K=6, 8192 fixed score vectors, all traffic
initially concentrated in six experts, seed 1337, scales 0.02/1/20. QB brought
maximum/mean load from 16 to below 1.2 after eight updates at all three scales;
the final ratio was 1.00586. Fixed sign 0.001 briefly crossed 1.2 at the smallest
scale but ended at 1.26953; at scales 1 and 20 it remained at 16 after 48
observations. This deliberately tests controller response under severe fixed
skew, not language-model convergence or typical training distributions.

Router-only same-process timing on MLX 0.32.2, B*T=16384, D=1024, E=96, K=6,
compiled forward/backward, warmed and alternating mode order:

| Mode | Median milliseconds per gate |
| --- | ---: |
| bf16 + sign | 1.150 |
| fp32 + sign | 1.424 |
| fp32 + QB sample export | 1.497 |
| QB quantile update, 8192 rows, one layer/window | 0.370 |

These timings exclude experts, attention and the full optimizer. They do not
establish end-to-end throughput, MFU or long-run loss improvement. Raw traces
and timings are in the probe JSON above. No long pretraining was launched.

## Sources and transfer boundary

- [Marin team's QB training report](https://openathena.ai/blog/quantile-balancing/)
  motivates quantile-based correction, but its model/scale differ from Viby.
- [Pinned Marin implementation](https://github.com/marin-community/marin/blob/c4ce3ae9e427e57d625ece10248911c5310e5991/experiments/grug/moe/model.py)
  uses the (K+1)-th threshold and column order statistic. Viby applies the idea
  to its existing sqrt-softplus affinity space, retaining checkpoint bias units.
- [Loss-Free Balancing](https://arxiv.org/abs/2408.15664) motivates historical
  bias updates and keeping the bias out of combination weights.

The sampled, EMA-smoothed Viby variant is a local engineering choice. It is not
a claim that QB universally beats fixed-step balancing or removes the need to
check training/validation loss.
