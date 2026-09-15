"""Analytic EMA regression at the actual high-beta2 training setting."""

import mlx.core as mx
import numpy as np
import pytest

from trainer.muon import FusedAdamW


@pytest.mark.parametrize("stack", [False, True])
def test_high_beta_ema_matches_closed_form_and_preserves_weight_dtype(stack):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        b2 = 0.9997499061952749
        opt = FusedAdamW(learning_rate=0.001, betas=[0.9, b2], eps=1e-8, weight_decay=0)
        params = {
            k: mx.ones((4,), mx.bfloat16) for k in (("a", "b") if stack else ("a",))
        }
        grads = {k: mx.ones_like(v) for k, v in params.items()}
        for _ in range(5000):
            params = opt.apply_gradients(grads, params)
            mx.eval(params, opt.state)
        for k, p in params.items():
            assert p.dtype == mx.bfloat16
            assert opt.state[k]["m"].dtype == opt.state[k]["v"].dtype == mx.float32
            np.testing.assert_allclose(
                np.array(opt.state[k]["v"]), 1 - b2**5000, rtol=5e-5
            )
    finally:
        mx.set_default_device(old)
