"""Fused Sinkhorn against the original graph, including the epsilon-aware VJP."""
import mlx.core as mx
import numpy as np
import pytest

from model.kernels.sinkhorn_fused import (
    prewarm_sinkhorn, sinkhorn_fused, sinkhorn_ref,
)


@pytest.mark.parametrize('iters', [1, 5, 20])
@pytest.mark.parametrize('eps', [1e-6, 0.1])
@pytest.mark.parametrize('scale', [0.0, 3.0, 80.0])
def test_sinkhorn_forward_and_vjp(iters, eps, scale):
    mx.random.seed(823)
    # Transposed input checks Metal's contiguous-input handling as well.
    x = (mx.random.normal((2, 7, 4, 4)) * scale).swapaxes(-1, -2)
    g = mx.random.normal(x.shape)
    prewarm_sinkhorn(iters, eps)

    def run(fn):
        def loss(a):
            return mx.sum(fn(a, iters, eps) * g)
        return mx.compile(lambda a: (fn(a, iters, eps), mx.grad(loss)(a)))(x)

    ref, dr = run(sinkhorn_ref)
    out, dx = run(sinkhorn_fused)
    mx.eval(ref, dr, out, dx)
    np.testing.assert_allclose(out, ref, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(dx, dr, atol=3e-6, rtol=3e-4)


def test_sinkhorn_reference_fallback():
    x = mx.ones((2, 3, 3), mx.float32)
    np.testing.assert_array_equal(sinkhorn_fused(x, 20, 1e-6), sinkhorn_ref(x, 20, 1e-6))
