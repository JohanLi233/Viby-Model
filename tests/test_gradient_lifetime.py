"""Allocation regression through the real epoch loop; no language training."""

import gc
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import trainer.base_trainer as base
from trainer.muon import FusedAdamW


def run_lifetime_case(release, elements=1 << 20, eager=True, microbatches=4):
    """Same fixed gradients and Adam state, alternate only source-tree lifetime."""
    gc.collect()
    mx.clear_cache()
    tr = object.__new__(base.BaseTrainer)
    tr.args = SimpleNamespace(
        accumulation_steps=2,
        grad_clip=0.0,
        warmup_iters=0,
        log_interval=100,
        epochs=1,
        max_steps=None,
        no_save=True,
    )
    tr.lm_config = SimpleNamespace(psr_enabled=False, dpr_enabled=False, n_mtp_layers=0)
    tr.tokenizer = SimpleNamespace(pad_token_id=0)
    tr.model = nn.Module()
    tr.model.w = mx.full((elements,), 0.125, mx.bfloat16)
    tr.optimizer = FusedAdamW(0.001, betas=(0.9, 0.9997499061952749), weight_decay=0)
    tr._expl_nest_mu, tr._en_delta = 0, None
    tr._time_limit_exceeded = lambda: False
    tr._save_if_needed = lambda *a: None
    tr.optimizer.init(tr.model.trainable_parameters())
    mx.eval(tr.model.parameters(), tr.optimizer.state)
    mx.reset_peak_memory()
    stages = []
    calls = 0

    def gradients(*args):
        nonlocal calls
        stages.append(
            {
                "phase": "before_forward",
                "microstep": calls,
                "active": mx.get_active_memory(),
            }
        )
        calls += 1
        scalar = mx.array(1.0)
        empty = mx.zeros((0,))
        return (scalar, scalar, empty, scalar, scalar, empty, empty), {
            "w": mx.contiguous(mx.full((elements,), calls * 0.0001, mx.bfloat16))
        }

    step = tr._optimizer_step

    def update(*args, **kwargs):
        stages.append(
            {
                "phase": "before_optimizer",
                "microstep": calls,
                "active": mx.get_active_memory(),
            }
        )
        return step(*args, **kwargs)

    tr._compute_loss_and_grad = gradients
    tr._optimizer_step = update
    old_release, old_eager, old_log = (
        base._RELEASE_MICROBATCH_GRADS,
        base._ACCUM_EAGER,
        base.log_training_progress,
    )
    base._RELEASE_MICROBATCH_GRADS, base._ACCUM_EAGER = release, eager
    base.log_training_progress = lambda *a, **kw: None
    x = mx.ones((1, 4), mx.int32)
    try:
        tr._run_epoch_steps(
            iter([(x, x, x)] * microbatches),
            0,
            microbatches,
            microbatches,
            None,
            0,
            time.time(),
            0,
        )
        mx.eval(tr.model.parameters(), tr.optimizer.state)
        return {
            "release": release,
            "eager": eager,
            "elements": elements,
            "gradient_bytes": elements * 2,
            "stages": stages,
            "peak": mx.get_peak_memory(),
            "optimizer_steps": int(tr.optimizer.step),
            "parameter": float(tr.model.w[0]),
            "m": float(tr.optimizer.state["w"]["m"][0]),
            "v": float(tr.optimizer.state["w"]["v"][0]),
        }
    finally:
        (
            base._RELEASE_MICROBATCH_GRADS,
            base._ACCUM_EAGER,
            base.log_training_progress,
        ) = old_release, old_eager, old_log


@pytest.mark.parametrize("eager", [True, False])
def test_release_preserves_updates_and_drops_unused_gradient(eager):
    a = run_lifetime_case(False, eager=eager)
    b = run_lifetime_case(True, eager=eager)
    for key in ("parameter", "m", "v"):
        assert a[key] == b[key]
    before = [s["active"] for s in a["stages"] if s["phase"] == "before_optimizer"]
    after = [s["active"] for s in b["stages"] if s["phase"] == "before_optimizer"]
    if eager:
        # Metal can defer reclamation until the next dispatch, so entry-time
        # counters need not drop immediately after the Python reference dies.
        assert np.all(np.array(after) <= np.array(before) + 4096)
        assert a["peak"] - b["peak"] >= a["gradient_bytes"] - 4096
    # At the next forward, the completed window's final gradient is unneeded
    # even when accumulation itself was lazy.
    a_next = [s["active"] for s in a["stages"] if s["phase"] == "before_forward"][2]
    b_next = [s["active"] for s in b["stages"] if s["phase"] == "before_forward"][2]
    assert a_next - b_next >= a["gradient_bytes"] - 4096


def test_pretrain_final_short_window_is_updated_at_actual_average():
    result = run_lifetime_case(True, elements=16, microbatches=3)
    assert result["optimizer_steps"] == 2
    g = [mx.array(i * 0.0001, mx.bfloat16) for i in (1, 2, 3)]
    first = float(g[0] + g[1])
    last = float(g[2] * 2)
    expected_m = 0.9 * (0.1 * first) + 0.1 * last
    expected_v = (
        0.9997499061952749 * ((1 - 0.9997499061952749) * first**2)
        + (1 - 0.9997499061952749) * last**2
    )
    assert result["m"] == pytest.approx(expected_m, rel=2e-6)
    assert result["v"] == pytest.approx(expected_v, rel=2e-6)
