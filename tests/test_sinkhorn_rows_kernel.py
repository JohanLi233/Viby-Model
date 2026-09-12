"""Exact gates against the existing optimizer row square-sum and typed division."""

import mlx.core as mx
import numpy as np
import pytest

from trainer import sinkhorn_rows as sr

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (19, 129), (513, 64), (33, 1024)])
def test_row_normalize_matches_existing_reduction_and_division(dtype, shape):
    mx.random.seed(143)
    # Keep a strided case so MLX's input packing participates in the check.
    x = mx.random.normal((shape[0], shape[1] * 2)).astype(dtype)[:, ::2]
    x = mx.where(mx.arange(shape[0])[:, None] % 7 == 0, 0, x)
    for eps in (1e-20, 0.01):
        reference = sr._reference(x, eps)
        candidate = sr.row_normalize(x, eps)
        compiled_reference = mx.compile(lambda a: sr._reference(a, eps))(x)
        compiled_candidate = mx.compile(lambda a: sr.row_normalize(a, eps))(x)
        mx.eval(reference, candidate, compiled_reference, compiled_candidate)
        assert sr.enabled_for(x)
        assert candidate.dtype == reference.dtype == dtype
        np.testing.assert_array_equal(
            np.asarray(candidate.astype(mx.float32)),
            np.asarray(reference.astype(mx.float32)),
        )
        np.testing.assert_array_equal(
            np.asarray(compiled_candidate.astype(mx.float32)),
            np.asarray(compiled_reference.astype(mx.float32)),
        )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_row_normalize_retains_extreme_denominator_casts(dtype):
    # The current FP16 denominator may overflow or underflow after its cast.
    # Fusion must preserve those outcomes, including NaN at zero / zero.
    x = mx.array(
        [[0, 0, 0], [300, -300, 300], [float("inf"), 1, -1], [float("nan"), 2, 3]],
        dtype,
    )
    reference = sr._reference(x, 1e-20)
    candidate = sr.row_normalize(x, 1e-20)
    mx.eval(reference, candidate)
    np.testing.assert_array_equal(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(reference.astype(mx.float32)),
    )
