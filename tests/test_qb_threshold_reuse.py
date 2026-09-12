"""Exact QB threshold reuse without changing native expert selection/order."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from _v41_common import cfg_tiny
import model.moe as moe


def _exact(left, right):
    if left.dtype in (mx.float32, mx.float16, mx.bfloat16):
        left, right = left.astype(mx.float32), right.astype(mx.float32)
    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


@pytest.mark.parametrize("k,cap", [(1, 1), (4, 7), (4, 64), (16, 7)])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("force_generic", [False, True])
def test_gate_threshold_weights_indices_gradients_load_and_bias_exact(
    monkeypatch, k, cap, tied, dtype, compiled, force_generic
):
    if force_generic:
        monkeypatch.setattr(mx, "__version__", "untested-version")
    mx.random.seed(741)
    cfg = cfg_tiny(
        dim=32, n_routed_experts=16, n_activated_experts=k, qb_stats_rows=cap
    )
    gate = moe.MoEGate(cfg)
    x = mx.random.normal((23, cfg.dim)).astype(getattr(mx, dtype))
    weight = mx.zeros_like(gate.weight) if tied else gate.weight
    # Nonzero tied bias groups also exercise equal admission thresholds.
    bias = ((mx.arange(16) // 4) * 0.25).astype(mx.float32)
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(moe, "_QB_THRESHOLD_REUSE", enabled)

        def forward(w, inputs):
            gate.weight, gate.bias = w, bias
            weights, ids, scores = gate(inputs)
            value = mx.sum(weights * (mx.arange(k) + 1)) + 0.01 * mx.sum(scores**2)
            return value, weights, ids, scores, gate._last_load, gate._last_qb_margins

        call = mx.value_and_grad(forward, argnums=(0, 1))
        if compiled:
            call = mx.compile(call)
        result, grads = call(weight, x)
        mx.eval(result, grads)
        new_bias = moe.update_quantile_bias(bias, result[-1], k)
        mx.eval(new_bias)
        outputs.append((result, grads, new_bias))
    for left, right in zip(outputs[0][0], outputs[1][0]):
        _exact(left, right)
    for left, right in zip(outputs[0][1], outputs[1][1]):
        _exact(left, right)
    _exact(outputs[0][2], outputs[1][2])
    assert outputs[1][0][-1].shape == ((min(23, cap), 16) if k < 16 else (0, 16))
    if k < 16:
        scores = np.asarray(outputs[1][0][3])
        sample = ((np.arange(min(23, cap)) + 0.5) * (23 / min(23, cap))).astype(
            np.int32
        )
        biased = scores[sample] + np.asarray(bias)
        alpha = np.sort(biased, axis=-1)[:, -k - 1 : -k]
        np.testing.assert_array_equal(
            np.asarray(outputs[1][0][-1]), scores[sample] - alpha
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_dispatch_removes_only_second_partition(monkeypatch, enabled):
    cfg = cfg_tiny(dim=32, qb_stats_rows=7)
    gate = moe.MoEGate(cfg)
    x = mx.ones((11, cfg.dim))
    calls = []
    original = mx.partition
    original_arg = mx.argpartition

    def traced(value, *args, **kwargs):
        calls.append(("partition", kwargs["kth"]))
        return original(value, *args, **kwargs)

    def traced_arg(value, *args, **kwargs):
        calls.append(("argpartition", kwargs["kth"]))
        return original_arg(value, *args, **kwargs)

    monkeypatch.setattr(mx, "partition", traced)
    monkeypatch.setattr(mx, "argpartition", traced_arg)
    monkeypatch.setattr(moe, "_QB_THRESHOLD_REUSE", enabled)
    weights, ids, scores = gate(x)
    mx.eval(weights, ids, scores, gate._last_qb_margins)
    expected = [("argpartition", gate.top_k - 1)]
    if not enabled:
        expected.append(("partition", gate.top_k))
    assert calls == expected


@pytest.mark.parametrize("force_generic", [False, True])
def test_version_guard_selects_scalar_or_generic_tail_gather(
    monkeypatch, force_generic
):
    if mx.default_device() != mx.gpu or getattr(mx, "__version__", None) != "0.32.2":
        pytest.skip("requires the audited MLX 0.32.2 Metal backend")
    gate = moe.MoEGate(cfg_tiny(dim=32, qb_stats_rows=7))
    x = mx.ones((11, 32))
    widths = []
    original = mx.take_along_axis

    def traced(array, indices, *args, **kwargs):
        # The threshold uses seven sampled rows; final routing weights have 11.
        if array.shape == (7, 16):
            widths.append(indices.shape[-1])
        return original(array, indices, *args, **kwargs)

    monkeypatch.setattr(mx, "take_along_axis", traced)
    monkeypatch.setattr(moe, "_QB_THRESHOLD_REUSE", True)
    if force_generic:
        monkeypatch.setattr(mx, "__version__", "untested-version")
    weights, ids, scores = gate(x)
    mx.eval(weights, ids, scores, gate._last_qb_margins)
    assert widths == [16 - gate.top_k if force_generic else 1]


@pytest.mark.parametrize("all_pad", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("compiled", [False, True])
def test_ffn_padding_aux_and_balancing_gradients_exact(
    monkeypatch, all_pad, dtype, compiled
):
    mx.random.seed(842)
    cfg = cfg_tiny(
        dim=32, moe_inter_dim=32, qb_stats_rows=7, aux_balance_loss_weight=1e-4
    )
    ffn = moe.MoEFeedForward(cfg)
    dt = getattr(mx, dtype)
    ffn.update(
        tree_unflatten(
            [
                (name, value.astype(mx.float32 if name.startswith("router.") else dt))
                for name, value in tree_flatten(ffn.parameters())
            ]
        )
    )
    params, bias = ffn.trainable_parameters(), ffn.router.bias
    x = mx.random.normal((2, 11, cfg.dim)).astype(dt)
    valid = (
        mx.zeros((2, 11), mx.bool_)
        if all_pad
        else mx.arange(22).reshape(2, 11) % 3 != 0
    )
    results = []
    for enabled in (False, False, True):
        monkeypatch.setattr(moe, "_QB_THRESHOLD_REUSE", enabled)

        def forward(weights):
            ffn.update(weights)
            out = ffn(x, pad_mask=valid)
            return (
                ffn._last_aux,
                out,
                ffn.router._last_load,
                ffn.router._last_qb_margins,
            )

        call = mx.value_and_grad(forward)
        if compiled:
            call = mx.compile(call)
        output, grads = call(params)
        mx.eval(output, grads)
        ffn.update(params)
        updated = moe.update_quantile_bias(
            bias, output[-1], cfg.n_activated_experts, ignore_padding=True
        )
        mx.eval(updated)
        results.append((output, dict(tree_flatten(grads)), updated))
    for index in (0, 2, 3):
        _exact(results[0][0][index], results[2][0][index])
    for name in results[0][1]:
        _exact(results[0][1][name], results[2][1][name])
    _exact(results[0][2], results[2][2])
    # The auxiliary gradient above is exact. Native expert scatter is unrelated
    # to QB threshold computation; isolate its measured A/A rounding envelope.
    reference, control, actual = [
        np.asarray(result[0][1].astype(mx.float32)) for result in results
    ]
    aa = float(np.abs(reference - control).max(initial=0))
    ulp = np.finfo(np.float32).eps if dtype == "float32" else 2**-7
    scale = max(float(np.abs(reference).max(initial=0)), 1e-20)
    np.testing.assert_allclose(
        actual, reference, rtol=0, atol=max(4 * aa, 2 * ulp * scale)
    )
    if all_pad:
        assert float(results[2][0][0]) == 0.0
        assert float(mx.sum(results[2][0][2])) == 0.0
        _exact(results[2][2], bias)
