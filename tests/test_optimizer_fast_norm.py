import mlx.core as mx
import numpy as np
import pytest

from trainer.fast_norm import square_sum, gradient_square_sum
from trainer.muon import _sinkhorn_body


@pytest.mark.parametrize("shape", [(1, 1), (19, 129), (513, 64), (2049, 256)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_square_sum(shape, dtype):
    mx.random.seed(19)
    x = mx.random.normal(shape).astype(dtype)
    for axis in (0, 1):
        actual = square_sum(x, axis)
        expected = mx.sum(mx.square(x.astype(mx.float32)), axis=axis, keepdims=True)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sinkhorn_all_iterations_and_state(dtype):
    mx.random.seed(23)
    p = mx.random.normal((513, 128)).astype(dtype)
    g = (mx.random.normal(p.shape) * 0.01).astype(dtype)
    g = mx.where(mx.arange(513)[:, None] % 7 == 0, 0, g)
    m = mx.zeros_like(p)
    lr = mx.array(0.002, dtype)
    refs = (p, m)
    opts = (p, m)
    native = mx.compile(_sinkhorn_body(.95, .18, 11, 1e-20, .001, 128))
    fast = mx.compile(_sinkhorn_body(.95, .18, 11, 1e-20, .001, 128, True))
    for _ in range(3):
        refs = native(refs[0], g, refs[1], lr)
        opts = fast(opts[0], g, opts[1], lr)
        mx.eval(refs, opts)
        np.testing.assert_array_equal(np.array(refs[1].astype(mx.float32)),
                                      np.array(opts[1].astype(mx.float32)))
        # FP32 reduction regrouping can cross bf16 rounding boundaries.
        delta = opts[0].astype(mx.float32) - refs[0].astype(mx.float32)
        assert float(mx.max(mx.abs(delta))) <= (0.008 if dtype == mx.bfloat16 else 2e-6)
        assert float(mx.sqrt(mx.mean(delta * delta))) < (2e-4 if dtype == mx.bfloat16 else 2e-7)


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 8193])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_gradient_norm_tails_and_precision(size, dtype):
    mx.random.seed(91)
    x = mx.random.normal((size,)).astype(dtype)
    expected = mx.sum(mx.square(x.astype(mx.float32)))
    np.testing.assert_allclose(float(gradient_square_sum(x)), float(expected), rtol=2e-6)
    if size:
        # FP16 square would overflow; FP32 accumulation stays finite.
        large = mx.full((size,), 300, dtype)
        assert np.isfinite(float(gradient_square_sum(large)))
        bad = large.at[0].add(float("nan"))
        assert np.isnan(float(gradient_square_sum(bad)))
