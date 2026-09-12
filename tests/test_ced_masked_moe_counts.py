"""Exact masked MoE bookkeeping; forward/aux/grad/QB A/B gates."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from _v41_common import cfg_tiny
import model.moe as moe
from model.kernels import moe_counts


def _legacy_counts(ids, valid, b, t, e):
    loads = mx.zeros((b, e), mx.float32)
    rows = mx.repeat(mx.arange(b), t)
    mask = valid.reshape(-1).astype(mx.float32)
    for j in range(ids.shape[-1]):
        loads = loads.at[rows, ids[:, j]].add(mask)
    return loads


def _assert_native_reduction_close(expected, repeated, actual, dtype, name):
    """Calibrate only expert output/gradient reductions against a native A/A.

    The GPU's expert scatter/weight-gradient reductions can change last bits
    without changing routes. Allow four times measured A/A variation or two
    output-dtype ulps at tensor scale. Router bookkeeping is checked separately
    with exact equality below and never receives this tolerance.
    """
    reference, control, candidate = [
        np.asarray(value.astype(mx.float32)) for value in (expected, repeated, actual)
    ]
    scale = max(float(np.abs(reference).max(initial=0)), 1e-20)
    ulp = np.finfo(np.float32).eps if dtype == "float32" else 2**-7
    aa = float(np.abs(reference - control).max(initial=0))
    tolerance = max(4 * aa, 2 * ulp * scale)
    np.testing.assert_allclose(
        candidate,
        reference,
        rtol=0,
        atol=tolerance,
        err_msg=f"{name}: native A/A max={aa:g}, tolerance={tolerance:g}",
    )


@pytest.mark.parametrize(
    "b,t,k,e", [(2, 173, 6, 192), (3, 7, 4, 16), (1, 0, 2, 8), (2, 11, 3, 513)]
)
@pytest.mark.parametrize("all_pad", [False, True])
def test_exact_occurrences_cpu_fallback_and_empty_rows(b, t, k, e, all_pad):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(171)
        # Repeated expert IDs also count occurrences rather than unique tokens.
        ids = mx.array(rng.integers(0, e, size=(b * t, k), dtype=np.int32))
        valid = mx.array(
            np.zeros((b, t), bool) if all_pad else rng.random((b, t)) > 0.35
        )
        actual = moe_counts.masked_sequence_route_counts(ids, valid, b, t, e)
        expected = _legacy_counts(ids, valid, b, t, e)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert actual.dtype == mx.float32
    finally:
        mx.set_default_device(previous)


def test_metal_dispatch_and_integer_counts(monkeypatch):
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.skip("requires the Metal dispatch")
    calls = []
    original = moe_counts._masked_count_kernel

    def traced_kernel():
        kernel = original()

        def invoke(**kwargs):
            calls.append(kwargs["output_dtypes"])
            return kernel(**kwargs)

        return invoke

    monkeypatch.setattr(moe_counts, "_masked_count_kernel", traced_kernel)
    b, t, k, e = 2, 173, 6, 192
    ids = (mx.arange(b * t * k, dtype=mx.int32).reshape(b * t, k) * 17) % e
    valid = mx.arange(b * t).reshape(b, t) % 5 != 0
    actual = moe_counts.masked_sequence_route_counts(ids, valid, b, t, e)
    expected = _legacy_counts(ids, valid, b, t, e)
    mx.eval(actual, expected)
    assert calls == [[mx.uint32]]
    assert bool(mx.array_equal(actual, expected))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("all_pad", [False, True])
def test_masked_moe_counts_preserve_output_aux_gradients_and_bias(
    monkeypatch, compiled, dtype, all_pad
):
    mx.random.seed(190)
    cfg = cfg_tiny(
        dim=64, moe_inter_dim=32, qb_stats_rows=32, aux_balance_loss_weight=1e-4
    )
    ffn = moe.MoEFeedForward(cfg, 0)
    dt = getattr(mx, dtype)
    ffn.update(
        tree_unflatten(
            [
                (name, value.astype(mx.float32 if name.startswith("router.") else dt))
                for name, value in tree_flatten(ffn.parameters())
            ]
        )
    )
    x = mx.random.normal((2, 11, cfg.dim)).astype(dt)
    valid = (
        mx.zeros((2, 11), mx.bool_)
        if all_pad
        else mx.array(
            [
                [True, True, False, True, True, False, True, False, True, True, False],
                [False] * 11,
            ]
        )
    )
    params = ffn.trainable_parameters()
    bias = ffn.router.bias
    outputs = []
    for enabled in (False, False, True):
        monkeypatch.setattr(moe, "_RECURRENT_MASKED_COUNTS", enabled)

        def objective(weights):
            ffn.update(weights)
            y = ffn(x, pad_mask=valid)
            loss = (
                mx.mean(y.astype(mx.float32) ** 2 * valid[..., None])
                + 0.1 * ffn._last_aux
            )
            return (
                loss,
                y,
                ffn._last_aux,
                ffn.router._last_load,
                ffn.router._last_qb_margins,
            )

        call = mx.value_and_grad(objective)
        if compiled:
            call = mx.compile(call)
        result, grads = call(params)
        mx.eval(result, grads)
        ffn.update(params)
        updated = moe.update_quantile_bias(
            bias, result[4], cfg.n_activated_experts, ignore_padding=True
        )
        mx.eval(updated)

        # Isolate the balancing gradient from native expert reductions; these
        # gradients depend only on identical router probabilities/counts.
        def auxiliary(weights):
            ffn.update(weights)
            ffn(x, pad_mask=valid)
            return ffn._last_aux

        aux_call = mx.grad(auxiliary)
        if compiled:
            aux_call = mx.compile(aux_call)
        aux_grads = aux_call(params)
        mx.eval(aux_grads)
        ffn.update(params)
        outputs.append(
            (result, dict(tree_flatten(grads)), updated, dict(tree_flatten(aux_grads)))
        )
    for index, name in ((0, "loss"), (1, "expert output")):
        _assert_native_reduction_close(
            outputs[0][0][index],
            outputs[1][0][index],
            outputs[2][0][index],
            dtype,
            name,
        )
    for expected, actual in zip(outputs[0][0][2:], outputs[2][0][2:]):
        # QB missing-row NaNs are intentional and must remain at identical rows.
        np.testing.assert_array_equal(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
        )
    assert outputs[0][1].keys() == outputs[2][1].keys()
    for key in outputs[0][1]:
        # Router parameters remain FP32 even with BF16 experts.
        grad_dtype = "float32" if outputs[0][1][key].dtype == mx.float32 else dtype
        _assert_native_reduction_close(
            outputs[0][1][key], outputs[1][1][key], outputs[2][1][key], grad_dtype, key
        )
    for key in outputs[0][3]:
        assert bool(mx.array_equal(outputs[0][3][key], outputs[2][3][key])), (
            f"aux: {key}"
        )
    assert bool(mx.array_equal(outputs[0][2], outputs[2][2]))
    if all_pad:
        assert float(outputs[2][0][2]) == 0.0
        assert float(mx.sum(outputs[2][0][3])) == 0.0
        assert bool(mx.array_equal(outputs[2][2], bias))


def test_nonboolean_masks_keep_legacy_weighted_counts(monkeypatch):
    cfg = cfg_tiny(dim=64, moe_inter_dim=32)
    ffn = moe.MoEFeedForward(cfg, 0)
    monkeypatch.setattr(moe, "_RECURRENT_MASKED_COUNTS", True)

    def forbidden(*args, **kwargs):
        raise AssertionError("fractional masks must use the original FP32 scatter")

    monkeypatch.setattr(moe_counts, "masked_sequence_route_counts", forbidden)
    x = mx.ones((1, 3, cfg.dim))
    y = ffn(x, pad_mask=mx.array([[1.0, 0.5, 0.0]]))
    mx.eval(y, ffn.router._last_load)
    assert float(mx.sum(ffn.router._last_load)) == 1.5 * cfg.n_activated_experts
