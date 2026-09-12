"""Training coefficient fusion versus the current composed MLX graph."""

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from model import hc
from model.kernels import hc_train


pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _inputs(shape, seed=1324):
    mx.set_default_device(mx.gpu)
    mx.random.seed(seed)
    mixes = mx.random.normal((*shape, 24)).astype(mx.float32)
    scale = mx.array([0.31, -0.77, 1.41], mx.float32)
    base = mx.random.normal((24,)).astype(mx.float32)
    cot = (
        mx.random.normal((*shape, 4)),
        mx.random.normal((*shape, 4)),
        mx.random.normal((*shape, 4, 4)),
    )
    return [mixes, scale, base], cot


def _reference(mixes, scale, base, iters, eps):
    pre, post, comb = hc.hc_split(mixes, scale, base, 4, eps)
    return pre, post, hc.sinkhorn(comb.reshape(*mixes.shape[:-1], 4, 4), iters, eps)


def _close(actual, expected, *, rtol=2e-5, atol=2e-6):
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape,iters,eps",
    [
        ((1, 1), 0, 1e-6),
        ((2, 3), 1, 1e-6),
        ((2, 17), 7, 0.03),
        ((4, 128), 20, 1e-6),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_forward_and_all_parameter_gradients(shape, iters, eps, compiled):
    values, cot = _inputs(shape)

    def ref(m, s, b):
        return _reference(m, s, b, iters, eps)

    def fused(m, s, b):
        return hc_train.split_sinkhorn_train(m, s, b, iters, eps)

    def run(fn):
        def differentiate(m, s, b):
            return mx.vjp(fn, [m, s, b], list(cot))

        return (mx.compile(differentiate) if compiled else differentiate)(*values)

    out, grads = run(fused)
    expected, expected_grads = run(ref)
    mx.eval(out, grads, expected, expected_grads)
    assert len(out) == len(grads) == 3
    for got, want in zip(out, expected):
        assert got.dtype == mx.float32
        _close(got, want, rtol=2e-6, atol=2e-7)
    for got, want, primal in zip(grads, expected_grads, values):
        assert got.dtype == primal.dtype
        _close(got, want, rtol=5e-5, atol=1e-5)


def test_saturated_sigmoid_zero_scales_and_sinkhorn_ties():
    values, cot = _inputs((1, 11))
    mixes = mx.zeros_like(values[0])
    scale = mx.array([0.0, -0.2, 0.0])
    base = mx.array([-100, -30, 30, 100, -100, -30, 30, 100] + [0.0] * 16)
    functions = (
        lambda m, s, b: hc_train.split_sinkhorn_train(m, s, b, 3, 0.01),
        lambda m, s, b: _reference(m, s, b, 3, 0.01),
    )
    runs = [mx.vjp(fn, [mixes, scale, base], list(cot)) for fn in functions]
    mx.eval(runs)
    for got, want in zip(runs[0][0], runs[1][0]):
        _close(got, want)
    for got, want in zip(runs[0][1], runs[1][1]):
        _close(got, want)


def test_compiled_weights_remain_dynamic():
    values, _ = _inputs((2, 13))
    fused = mx.compile(lambda m, s, b: hc_train.split_sinkhorn_train(m, s, b, 4, 1e-5))
    first = fused(*values)
    changed = [values[0] * 0.7, values[1] + mx.array([0.4, 0.8, -0.2]), values[2] - 0.3]
    second = fused(*changed)
    expected = _reference(*changed, 4, 1e-5)
    mx.eval(first, second, expected)
    assert any(not bool(mx.array_equal(a, b)) for a, b in zip(first, second))
    for got, want in zip(second, expected):
        _close(got, want)


def test_hyperconnection_dispatch_keeps_projection_and_training_gradients(monkeypatch):
    mx.random.seed(151)
    module = hc.HyperConnection(32, 4, sinkhorn_iters=4)
    x = mx.random.normal((2, 17, 4, 32)).astype(mx.bfloat16)
    module.fn.weight = module.fn.weight.astype(mx.bfloat16)
    weights = module.trainable_parameters()
    cot = [
        mx.random.normal((2, 17, 4)),
        mx.random.normal((2, 17, 4)),
        mx.random.normal((2, 17, 4, 4)),
    ]
    runs = []
    for enabled in (False, True):
        monkeypatch.setattr(hc_train, "_TRAIN_FUSION", enabled)

        def loss(params):
            module.update(params)
            outputs = module.mixes(x)
            return sum(mx.sum(a * c) for a, c in zip(outputs, cot))

        result = mx.value_and_grad(loss)(weights)
        mx.eval(result)
        runs.append(result)
    _close(runs[0][0], runs[1][0], atol=1e-5)
    old, new = dict(tree_flatten(runs[0][1])), dict(tree_flatten(runs[1][1]))
    assert old.keys() == new.keys() == {"fn.weight", "scale", "base"}
    for key in old:
        _close(
            old[key].astype(mx.float32),
            new[key].astype(mx.float32),
            rtol=0.02,
            atol=0.004,
        )
    module.eval()
    assert not hc_train.enabled_for(
        mx.zeros((2, 17, 24)), module.scale, module.base, 4, module.training
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("compiled", [False, True])
def test_low_precision_coefficient_leaves_keep_native_promotion_and_vjp(
    dtype, compiled
):
    values, cot = _inputs((2, 37), seed=641)
    values[1:] = [v.astype(dtype) for v in values[1:]]

    def fused(m, s, b):
        return hc_train.split_sinkhorn_train(m, s, b, 7, 0.003)

    def reference(m, s, b):
        return _reference(m, s, b, 7, 0.003)

    def run(fn):
        def call(m, s, b):
            return mx.vjp(fn, [m, s, b], list(cot))
        return (mx.compile(call) if compiled else call)(*values)

    output, gradients = run(fused)
    expected, expected_gradients = run(reference)
    mx.eval(output, gradients, expected, expected_gradients)
    for actual, want in zip(output, expected):
        assert actual.dtype == want.dtype == mx.float32
        _close(actual, want, rtol=2e-6, atol=2e-7)
    for actual, want, primal in zip(gradients, expected_gradients, values):
        assert actual.dtype == want.dtype == primal.dtype
        # Parameter leaves are rounded only after the original FP32 reduction.
        _close(
            actual.astype(mx.float32), want.astype(mx.float32), rtol=0.009, atol=0.001
        )


def test_actual_bf16_model_conversion_enters_training_fusion(monkeypatch):
    from _v41_common import build
    from test_ced_recurrent import config, inputs
    from trainer.utils import convert_model_dtype

    model = build(config(hc_sinkhorn_iters=4))
    convert_model_dtype(model, "bfloat16")
    model.train()
    selected = model.model.layers[7].attn_hc
    assert selected.scale.dtype == selected.base.dtype == mx.bfloat16
    assert selected.fn.weight.dtype == mx.bfloat16
    seen = []
    operation = hc._train_mixes

    def spy(mixes, scale, base, iters, eps):
        seen.append((mixes.dtype, scale.dtype, base.dtype))
        return operation(mixes, scale, base, iters, eps)

    monkeypatch.setattr(hc, "_train_mixes", spy)
    tokens = inputs(12)
    runs = []
    for enabled in (False, True):
        monkeypatch.setattr(hc_train, "_TRAIN_FUSION", enabled)
        before = len(seen)
        result = model(tokens, labels=tokens, need_logits=True)
        mx.eval(result.loss, result.logits)
        assert (len(seen) > before) == enabled
        runs.append(result)
    assert seen and all(
        dtypes == (mx.float32, mx.bfloat16, mx.bfloat16) for dtypes in seen
    )
    _close(runs[1].loss, runs[0].loss, rtol=0.001, atol=0.001)
    _close(
        runs[1].logits.astype(mx.float32),
        runs[0].logits.astype(mx.float32),
        rtol=0.01,
        atol=0.008,
    )
