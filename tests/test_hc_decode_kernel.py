"""Exact eager parity gates for decode mHC coefficient fusion."""

import mlx.core as mx
import numpy as np
import pytest

from model.hc import HyperConnection
from model.kernels import hc_decode as hd

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize(
    "n,iters,eps,magnitude",
    [
        (1, 1, 1e-6, 0),
        (1, 20, 1e-6, 3),
        (8, 20, 0.1, 80),
        (3, 5, 1e-6, 3),
    ],
)
def test_split_sinkhorn_matches_eager_and_vjp(n, iters, eps, magnitude):
    mx.random.seed(27182)
    # Transposition exercises input packing in the custom Metal primitive.
    mixes = (mx.random.normal((24, 1, n)) * magnitude).transpose(2, 1, 0)
    scale = mx.array([0.7183923, -1.120739, 0.5920031], mx.float32)
    base = mx.random.normal((24,))
    cot = [mx.random.normal(shape) for shape in ((n, 1, 4), (n, 1, 4), (n, 1, 4, 4))]
    args = [mixes, scale, base]
    reference = mx.vjp(lambda m, s, b: hd._reference(m, s, b, iters, eps), args, cot)
    fused = mx.vjp(
        lambda m, s, b: hd.split_sinkhorn_decode(m, s, b, iters, eps), args, cot
    )
    mx.eval(reference, fused)
    for ref_part, got_part in zip(reference, fused):
        for ref, got in zip(ref_part, got_part):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_hyperconnection_dispatch_and_dynamic_parameters(monkeypatch, dtype):
    mx.random.seed(27183)
    hc = HyperConnection(128, 4)
    hc.fn.weight = hc.fn.weight.astype(dtype)
    hc.eval()
    x = mx.random.normal((2, 1, 4, 128)).astype(dtype)
    for _ in range(2):
        values = []
        for enabled in (False, True):
            monkeypatch.setattr(hd, "_ENABLED", enabled)
            values.append(hc.mixes(x))
        mx.eval(values)
        for ref, got in zip(*values):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        hc.scale = mx.array([0.1, 3, -0.7], mx.float32)
        hc.base = mx.random.normal((24,)) * 3
    assert hd.enabled_for(mx.zeros((2, 1, 24)), hc.scale, hc.base, 4, False)
    assert not hd.enabled_for(mx.zeros((2, 1, 24)), hc.scale, hc.base, 4, True)
    assert not hd.enabled_for(mx.zeros((9, 1, 24)), hc.scale, hc.base, 4, False)
