"""Numerical gates for the September 12 sparse VJP and SIMD decode kernels."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from model.kernels import (
    decode_metadata as dm,
    sinkhorn_fused as sf,
    sparse_attention as sa,
)
from test_sparse_attention_kernel import _inputs, _run_vjp
from test_sparse_attention_key_owned import _assert_pool_close

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize(
    "n,iters,eps,scale",
    [
        (1, 1, 1e-6, 0),
        (1, 20, 1e-6, 80),
        (8, 20, 0.1, 3),
        (9, 5, 1e-6, 3),
    ],
)
def test_simd_sinkhorn_preserves_values_and_vjp(monkeypatch, n, iters, eps, scale):
    mx.random.seed(1234)
    x = (mx.random.normal((n, 4, 4)) * scale).swapaxes(-1, -2)
    cot = mx.random.normal(x.shape)
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(sf, "_SIMD_DECODE", enabled)
        result = mx.vjp(lambda a: sf.sinkhorn_fused(a, iters, eps), [x], [cot])
        mx.eval(result)
        results.append(result)
    for got, ref in zip(tree_flatten(results[1]), tree_flatten(results[0])):
        np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(ref[1]))


@pytest.mark.parametrize(
    "n,k", [(1, 1), (79, 6), (129, 64), (259, 129), (300, 256), (301, 260)]
)
def test_register_sort_retains_fixed_k_and_sentinels(monkeypatch, n, k):
    mx.random.seed(27)
    for scores in (
        mx.random.normal((3, 1, n)),
        mx.full((3, 1, n), 1e10),
        mx.full((3, 1, n), -1e30),
    ):
        reach = mx.random.uniform(shape=scores.shape) > 0.3
        results = []
        for enabled in (False, True):
            monkeypatch.setattr(dm, "_SIMD_POST", enabled)
            out = dm.topk_indices(scores, reach, k, 128)
            mx.eval(out)
            results.append(np.asarray(out))
        np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "dim,seq,keys,window", [(64, 35, 0, 35), (128, 65, 39, 35), (128, 1, 1, 128)]
)
def test_fused_sparse_vjp_matches_split(monkeypatch, dtype, dim, seq, keys, window):
    q, w, c, visible, seg, pad, sinks, W, scale = _inputs(
        2, seq, keys, dim, window, 21, dtype
    )
    cot = mx.random.normal((2, 16, seq, dim)).astype(dtype)
    monkeypatch.setattr(sa, "_SPLIT_BWD", True)
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", False)
    monkeypatch.setattr(sa, "_FUSED_BWD_TILE", 32)
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(sa, "_FUSED_BWD", enabled)
        result = _run_vjp(
            sa.indexed_attention, q, w, c, sinks, visible, seg, pad, W, scale, cot
        )
        mx.eval(result)
        results.append(result)
    np.testing.assert_array_equal(
        np.asarray(results[0][0].astype(mx.float32)),
        np.asarray(results[1][0].astype(mx.float32)),
    )
    for name, got, ref in zip(("dq", "dw", "dc", "ds"), results[1][1], results[0][1]):
        if got.size:
            _assert_pool_close(
                np.asarray(got.astype(mx.float32)),
                np.asarray(ref.astype(mx.float32)),
                dtype,
                name,
            )
