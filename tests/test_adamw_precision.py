"""Closed-form checks for the large-beta2 optimizer used by pretraining."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import trainer.muon as muon


BETA2 = 0.9997499061952749


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("dispatch", ["single", "stacked", "shapeful"])
def test_constant_gradient_moments_follow_closed_form(dtype, dispatch, monkeypatch):
    if dispatch == "shapeful":
        monkeypatch.setattr(muon, "_SHAPEFUL_MIN_BYTES", 1)
    keys = ("a", "b") if dispatch == "stacked" else ("a",)
    params = {k: mx.ones((2,), dtype) for k in keys}
    grads = {k: mx.ones((2,), dtype) for k in keys}
    opt = muon.FusedAdamW(0.0, betas=(0.9, BETA2), weight_decay=0.0)
    for t in range(1, 1001):
        params = opt.apply_gradients(grads, params)
        if t % 100 == 0:
            mx.eval(params, opt.state)
    for k in keys:
        assert params[k].dtype == dtype
        assert opt.state[k]["m"].dtype == mx.float32
        assert opt.state[k]["v"].dtype == mx.float32
        np.testing.assert_allclose(
            np.array(opt.state[k]["m"]), 1 - 0.9**1000, rtol=2e-6
        )
        np.testing.assert_allclose(
            np.array(opt.state[k]["v"]), 1 - BETA2**1000, rtol=3e-5
        )


def test_legacy_bf16_moments_decay_and_roundtrip(tmp_path):
    params = {"w": mx.ones((2,), mx.bfloat16)}
    opt = muon.FusedAdamW(0.0, betas=(0.9, BETA2))
    opt.init(params)
    opt.state["step"] = mx.array(5000, mx.uint64)
    opt.state["w"] = {"m": mx.ones((2,), mx.bfloat16), "v": mx.ones((2,), mx.bfloat16)}
    grads = {"w": mx.zeros((2,), mx.bfloat16)}
    params = opt.apply_gradients(grads, params)
    mx.eval(params, opt.state)
    assert int(opt.step) == 5001
    np.testing.assert_allclose(np.array(opt.state["w"]["v"]), BETA2, rtol=1e-7)
    path = str(tmp_path / "optimizer.safetensors")
    mx.save_safetensors(path, dict(tree_flatten(opt.state)))
    restored = muon.FusedAdamW(0.0, betas=(0.9, BETA2))
    restored.state = tree_unflatten(list(mx.load(path).items()))
    expected = opt.apply_gradients(grads, params)
    actual = restored.apply_gradients(grads, params)
    mx.eval(expected, actual, opt.state, restored.state)
    assert int(restored.step) == 5002
    for name, value in tree_flatten(opt.state):
        np.testing.assert_array_equal(
            np.array(value), np.array(dict(tree_flatten(restored.state))[name])
        )
    assert actual["w"].dtype == mx.bfloat16


@pytest.mark.parametrize("bias_correction", [True, False])
@pytest.mark.parametrize("cautious", [True, False])
def test_update_matches_fp64_formula_then_parameter_rounding(bias_correction, cautious):
    # FP16 small gradients must be squared in FP32 before second-moment EMA.
    p = mx.array([1.0, -1.0], mx.float16)
    g = mx.array([1e-4, 1e-4], mx.float16)
    gn = np.array(g).astype(np.float64)
    lr, wd = 1e-3, 0.1
    opt = muon.FusedAdamW(
        lr,
        betas=(0.9, BETA2),
        eps=1e-8,
        weight_decay=wd,
        bias_correction=bias_correction,
        cautious=cautious,
    )
    actual = opt.apply_gradients({"w": g}, {"w": p})["w"]
    m, v = 0.1 * gn, (1 - BETA2) * gn**2
    mask = np.array([1.0, 0.0]) if cautious else np.ones(2)
    expected = np.array([1.0, -1.0]) * (1 - lr * wd * mask)
    if bias_correction:
        expected -= lr * (m / 0.1) / (np.sqrt(v / (1 - BETA2)) + 1e-8)
    else:
        expected -= lr * m / (np.sqrt(v) + 1e-8)
    np.testing.assert_allclose(
        np.array(actual), expected.astype(np.float16), atol=1e-3, rtol=0
    )
    np.testing.assert_allclose(np.array(opt.state["w"]["v"]), v, rtol=1e-6, atol=0)
