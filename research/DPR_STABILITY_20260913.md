# DPR objective audit and Adam moment precision repair

2026-09-13, base commit `ae1ab4244a4bda8ad7f81a6a11246b6807ecfc63`.
Scope: source audit, analytic derivation, scalar arithmetic and small regression
tests. No training experiment, hyperparameter sweep, or performance benchmark.
User reports gradient norm reaching 40 on further continuation; local logs end
at microstep 16878 and do not contain that continuation. The source of that
particular spike has not been isolated.

## Follow-up: persistent Muon radii and contextual JEPA implemented

The architectural direction below was subsequently implemented as
`contextual_v2`; the original derivation and earlier evidence remain historical.
See [the current contract](DPR_JEPA.md) for configuration and migration.
The follow-up passed 95 focused checks in 5.01s, then the migration check again
after adding dimension validation. Evidence, scalar samples and source snapshots
are in `research_runs/dpr_contextual_v2_20260913/`.

A second numerical contract violation was confirmed from the actual saved
checkpoint and same-seed DPR initialization:

| Matrix | Initial Frobenius norm | Checkpoint step 9999 |
| --- | --- | --- |
| DPR predict.weight | 5.039885 | 15.177878 |
| DPR target.proj.weight | 2.476149 | 4.140509 |

Both matrices belong to the norm-freezing MuonH group. The old projection used
the current parameter norm as each step's radius and cast its scale to BF16.
Rounding deviations could become the next radius, compounding over updates.
That violates frozen-norm semantics; it is not just a large gradient warning.

The repaired optimizer stores a persistent FP32 radius per matrix/head/expert.
It projects the FP32 update before rounding the final weight. For normal BF16
values, a single rounding has relative norm error at most u=2^-8, giving
`(1-u)R <= ||W_t|| <= (1+u)R` each step, instead of a product of per-step
errors. Zero matrices establish a radius at their first nonzero update.
Restored old states adopt the checkpoint weight norm once and retain clocks.
The existing trained function is not forcibly rescaled; previous drift and
lost Adam history are not undone by upgrading.

The combination of distorted Adam preconditioning and broken Muon norm
constraints provides concrete amplification mechanisms. It does not identify
the unique largest-gradient tensor at the unrecorded 40-norm continuation.
Training now exports component norms from the existing gradient reductions
and prints the largest tensor during spikes, without another backward pass.

The broader small regression run initially exposed a stale test stub lacking
the trainer's `lm_config`; the stub was updated to the current constructor
contract. An initial new compiled test also left trace placeholders in its
model before calling eager inference; restoring the original parameters,
as the real trainer already does, fixed that test harness. Neither is reported
as a repaired production-model defect.

## Confirmed numerical defect and implemented repair

The actual run uses Adam beta2=0.9997499061952749. Both Adam moments were stored
in BF16, including approximately 906M expert parameters. For constant g=1,
v0=0, the exact second moment is `v_t=1-beta2**t`.

| Updates | Old fused BF16 v | Exact v |
| --- | --- | --- |
| 100 | 0.0245361328125 | 0.024702287402948 |
| 1000 | 0.125 | 0.221296625882967 |
| 5000 | 0.125 | 0.713674332083392 |

With v=1 and g=0, the old kernel returned exactly 1 instead of beta2. These
are actual calls to `_adamw_kernel`, not a language-model experiment. Near
0.125, BF16 spacing above the value is 0.0009765625, and the EMA increment
is only about 0.0002188. It rounds away. With the same bias correction,
underestimating v by this amount inflates the inverse-square-root factor by
sqrt(0.713674/0.125)=2.389; old first-moment rounding makes the full constant-g
step factor approximately 2.35. This example proves an optimizer error, not
the unique cause of the reported gradient norm 40.

`trainer/muon.py` now computes and stores m/v in FP32, including g squared,
bias corrections and the parameter update. LR remains FP32 until calculation.
Only the returned weights are rounded to their original dtype. All dispatches
share the corrected body; FP32 moment size also bounds stacked dispatch.
Legacy moments are promoted on their next update, without resetting clocks or
inventing lost history. At 5000 updates the repaired kernel returns
v=0.713683009147644 and m=0.999999761581421 (FP32 rounding error remains).

There are no master weights: small updates to BF16 weights can still round
away. The saved run's Adam moment payload grows by 3,624,624,304 bytes
(3.375694 GiB), computed from its optimizer safetensors header. Temporary
allocation and runtime cost have not been measured. This is a numerical
training change, not a bitwise-equivalent speed optimization.

## Why a proper kernel score does not secure this trainable target

For a prefix x, let P be the predicted particle distribution and Q_phi the
distribution of the future encoding y. The implemented score satisfies

```text
E_{Y~Q_phi} S(P,Y)
 = MMD_K(P,Q_phi)^2 + 1 - E_{Y,Y'~Q_phi} K(Y,Y').
```

The last two terms are constant with respect to P only when Q is fixed.
Here the target network receives gradients through S. Concentrating Q
increases its kernel self-similarity and lowers that supposedly constant
term. Even an ideal predictor P=Q retains this incentive. The covariance
penalty competes with it; it does not remove the target-side prediction
gradient. A zero-variance stationary point is NOT thereby proved to be a
global optimum or stable attractor of the complete regularized objective.

Illustrative population calculation (Gaussian Q, ideal unrestricted P;
not an exact model of the four bounded particles): set Q=N(0,s I_r), r=32.
Then at P=Q:

```text
J(s) = 1 - (1+2s/r)^(-r/2) + (s-1)^2
J'(s) = (1+2s/r)^(-r/2-1) + 2(s-1).
```

J'(1)>0, so even with perfect distribution prediction, covariance I is not
stationary: reducing variance lowers the joint objective. This explains a
structural pressure, not the time of the observed tail deterioration.

Local means in `research_runs/viby_dpr_jepa_v1/dpr_metrics.jsonl`:

| Metric | Steps 13000–13999 | Steps 16779–16878 |
| --- | --- | --- |
| NTP CE | 2.019381 | 1.884661 |
| Kernel score | 0.383989 | 0.136304 |
| Covariance regularizer | 0.116513 | 0.645539 |
| Target variance | 0.692010 | 0.201894 |
| Valid-target coverage | 0.982309 | 0.982608 |

Score plus regularizer worsens from about 0.501 to 0.782. Shrinking targets
do not by themselves explain the global gradient spike. For the Gaussian
kernel used here,

```text
||grad_y S|| <= 2 / sqrt(e*r) = 0.21444... (r=32).
```

This is a per-target bound before lambda and valid-count averaging. Parameter
gradients also contain target/predictor/backbone Jacobians and the covariance
regularizer. Their magnitudes are not bounded by the kernel bound alone.

## Token efficiency and NCP

The [NCP report](https://arxiv.org/abs/2609.10715) reports reaching the
OLMo-3-7B final pretraining loss using 51.3% of its tokens; the NCP model is
8.9B, and the report separately includes a parameter-matched 8.9B comparison.
The token figure should not be restated as an equal-parameter or wall-clock
speed claim. See the [paper](https://arxiv.org/pdf/2609.10715), sections 2–4.

The released NCP construction uses contextual encoder states, a dedicated
concept module on a four-times shorter sequence, a product-quantized
vocabulary, and detached future targets. NTP trains its concept feedback
path. Current DPR instead learns a future-only projection of raw embeddings
under kernel and whitening objectives, and predicts it with a single linear
particle head. More particles do not establish more useful semantic labels.

Whitening constrains moments, not task relevance. A representation of an
independent nuisance variable can have zero mean and covariance I while
having zero information about the desired token. For a frozen local SPD
preconditioner A, writing g=grad L_NTP and a=grad L_aux gives the first-order
approximation

```text
Delta L_NTP = -eta * g^T A g - eta * lambda * g^T A a + O(eta^2).
```

Low auxiliary loss does not establish a positive second inner product. This
is a local explanatory approximation, not an exact description of stateful
Muon or a theorem that a revised architecture will improve token efficiency.

## Minimal architectural direction (not implemented)

Replace the independently optimized future encoder with detached contextual
targets from the existing CED encoder BEFORE DPR injection. Form document-local
future groups; use a fixed projection if a smaller target dimension is required.
Apply target normalization without a learned scale and detach the whole target.
The encoder still learns from NTP and the predictor's historical input path.
Reuse its existing forward; do not add an EMA copy of the full model.

Keep future states entirely in the auxiliary target path. The decoder must
receive only predictions from the available prefix. This removes the direct
target-shrink feedback and ties the representation source to language modeling,
but a small projection may discard useful information and CE still may bypass
the residual. Neither point is repaired merely by adding stop-gradient to
the existing ungrounded future encoder. Matching NCP's architecture requires
more than this minimal revision; NCP's published efficiency does not transfer
automatically to DPR or Viby's old partial NCP port.

## Stability actions and limits

Use the precision repair before more long training. `--grad_clip 1.0` bounds
the gradient supplied to the optimizer; logged grad_norm is PRE-clipping and
may still be 40. Adam and Muon largely normalize gradient scale, so clipping
is not a mathematical bound on the final parameter displacement. In particular,
lowering a common lambda on a target-only Muon group need not proportionally
reduce its update: ideal polar(cG)=polar(G) for positive c.

The LR schedule itself has no singularity at decay start. A smaller LR does
not imply a smaller gradient at the current weights. The NCP paper also
reports shared Muon/QK instabilities and per-head QK normalization as a
stabilization intervention (not part of its main efficiency configuration).
Viby normalizes the low-rank query BEFORE `wq_b`; per-head Muon is not
post-projection per-head Q normalization. That is an unresolved additional
source, not diagnosed by this audit. Changing attention normalization during
checkpoint continuation would change model behavior and has not been done.

Do not abruptly remove the trained residual when resuming. Do not silently
reset all optimizer history: current `--reset_optimizer` also resets training
progress. Changing DPR objective/weight currently triggers the execution
compatibility guard, so a new objective needs an explicit migration design.
The precision fix itself preserves parameter groups and can load old states.

## Checks and evidence

```sh
.venv/bin/python -m pytest -q tests/test_adamw_precision.py tests/test_muonh.py tests/test_dpr.py tests/test_checkpoint_save.py
python3 scripts/check_repo.py
git diff --check
```

35 focused tests passed in 5.46s, including constant-gradient closed forms,
zero-gradient decay, single/stacked/shapeful dispatch, FP16 square underflow,
legacy BF16 promotion, optimizer save/reload, and actual small DPR trainer
accumulation/checkpoint paths. Scalar samples and patch provenance are in
`research_runs/dpr_stability_20260913/`. No repaired long-run convergence,
throughput, memory peak, or token-efficiency improvement has been established.
