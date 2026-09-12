"""Exact paired Sinkhorn iteration and reused initial-norm gates."""

import mlx.core as mx
import numpy as np
import pytest

from trainer.fast_norm import square_sum
from trainer import sinkhorn_pairs as sp

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (19, 129), (513, 64), (33, 1024)])
def test_pair_preserves_both_typed_iterations(dtype, shape):
    mx.random.seed(159)
    x = mx.random.normal((shape[0], shape[1] * 2)).astype(dtype)[:, ::2]
    for eps in (1e-20, 0.01):
        reference = sp._reference(x, eps)
        candidate = sp.column_then_row(x, eps)
        compiled_reference = mx.compile(lambda a: sp._reference(a, eps))(x)
        compiled_candidate = mx.compile(lambda a: sp.column_then_row(a, eps))(x)
        mx.eval(reference, candidate, compiled_reference, compiled_candidate)
        for got, ref in (
            (candidate, reference),
            (compiled_candidate, compiled_reference),
        ):
            assert got.dtype == ref.dtype == dtype
            np.testing.assert_array_equal(
                np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32))
            )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_first_row_norm_reuse_keeps_mask_and_zero_division(dtype):
    mx.random.seed(160)
    x = mx.random.normal((513, 64)).astype(dtype)
    x = mx.where(mx.arange(x.shape[0])[:, None] % 7 == 0, 0, x)
    rho = mx.sqrt(square_sum(x, 1))
    mask = rho <= 0.8 * mx.mean(rho)
    for eps in (1e-20, 0.01):

        def reference(a, norm, zero):
            value = mx.where(zero, mx.zeros_like(a), a)
            return value / (mx.sqrt(square_sum(value, 1)) + eps).astype(a.dtype)

        candidate = mx.compile(lambda a, r, z: sp.first_row_from_norm(a, r, z, eps))(
            x, rho, mask
        )
        original = mx.compile(reference)(x, rho, mask)
        mx.eval(candidate, original)
        np.testing.assert_array_equal(
            np.asarray(candidate.astype(mx.float32)),
            np.asarray(original.astype(mx.float32)),
        )
