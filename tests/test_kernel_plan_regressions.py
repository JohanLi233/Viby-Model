"""Ownership, capacity, and discrete-selection regressions for the CSA2 plan."""
import mlx.core as mx
import numpy as np
import pytest

from model.kernels import hc_pre_norm as hc, indexer_score as ix, moe_dispatch as moe
from model.kernels import sparse_attention as sa

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="need Metal")


@pytest.mark.parametrize("rows,dim", [(1023, 64), (1024, 64), (1027, 160), (1024, 4096)])
def test_grouped_hc_preserves_all_token_gradients(monkeypatch, rows, dim):
    mx.set_default_device(mx.gpu)
    mx.random.seed(6)
    x = mx.random.normal((rows, 4, dim)).astype(mx.bfloat16)
    p = mx.full((rows, 4), 0.25, mx.float32)
    w = mx.ones((dim,), mx.bfloat16)
    g = mx.ones((rows, dim), mx.bfloat16)
    results = []
    for grouped in (False, True):
        monkeypatch.setattr(hc, "_GROUPED_DW", grouped)
        _, grads = mx.vjp(lambda a, b, c: hc.hc_pre_norm(a, b, c, 1e-6), [x, p, w], [g])
        mx.eval(grads)
        results.append([np.asarray(v.astype(mx.float32)) for v in grads])
    for a, b in zip(results[0][:2], results[1][:2]):
        np.testing.assert_array_equal(a, b)
    # Original partial values, including their bf16 write cast, then an
    # independent FP32 reduction (the old mx.sum on bf16 rounds internally).
    fwd, bwd = hc._kernels()
    _, h, r = fwd(inputs=[x, p, w, mx.array([1e-6])], template=[("T", x.dtype), ("D", dim)],
                  grid=(128, rows, 1), threadgroup=(128, 1, 1),
                  output_shapes=[(rows, dim), (rows, dim), (rows,)],
                  output_dtypes=[x.dtype, x.dtype, mx.float32])
    _, _, partial = bwd(inputs=[x, p, w, g, h, r], template=[("T", x.dtype), ("D", dim)],
                        grid=(128, rows, 1), threadgroup=(128, 1, 1),
                        output_shapes=[x.shape, p.shape, (rows, dim)],
                        output_dtypes=[x.dtype, mx.float32, w.dtype])
    expected = mx.sum(partial.astype(mx.float32), axis=0).astype(w.dtype)
    np.testing.assert_array_equal(results[1][2], np.asarray(expected.astype(mx.float32)))


@pytest.mark.parametrize("rows,keys", [(256, 32), (259, 47), (3, 1025)])
def test_indexer_empty_tiles_keep_reachable_scores_and_gradients(rows, keys):
    mx.set_default_device(mx.gpu)
    q = mx.ones((1, rows, 4, 64), mx.float16)
    k = mx.ones((1, keys, 64), mx.float16)
    w = mx.ones((1, rows, 4), mx.float16)
    reach = mx.broadcast_to((mx.arange(keys) // 16 % 2 == 1)[None, None], (1, rows, keys))
    out, grads = mx.vjp(ix.indexer_score, [q, k, w, reach], [mx.ones(reach.shape)])
    mx.eval(out, grads)
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(mx.where(reach, 256.0, -1e30)))
    valid_count = int(mx.sum(reach[0, 0]).item())
    np.testing.assert_array_equal(np.asarray(grads[0].astype(mx.float32)), valid_count)
    expected_k = mx.where(reach[0, 0, :, None], rows * 4, 0)
    np.testing.assert_array_equal(np.asarray(grads[1][0].astype(mx.float32)),
                                  np.asarray(mx.broadcast_to(expected_k, (keys, 64)).astype(mx.float16).astype(mx.float32)))


@pytest.mark.parametrize("routes,dim,rows", [(1, 129, 5), (6, 257, 7), (6, 64, 0)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_token_owned_combine_fp32_reference(routes, dim, rows, dtype):
    mx.set_default_device(mx.gpu)
    mx.random.seed(4)
    order = mx.argsort(mx.random.uniform(shape=(rows * routes,))).astype(mx.int32)
    inv = moe.route_inverse(order) if rows else mx.zeros((0,), mx.int32)
    y = mx.random.normal((rows * routes, dim)).astype(dtype)
    weights = mx.random.normal((rows * routes,)).astype(mx.float32)
    cot = mx.random.normal((rows, dim)).astype(dtype)

    def kernel(a, w):
        return moe.combine_routes(a, w, order, inv, routes)

    def ref(a, w):
        return mx.sum(a[inv].astype(mx.float32).reshape(rows, routes, dim) *
                      w.reshape(rows, routes, 1), axis=1).astype(dtype)

    ya, ga = mx.vjp(kernel, [y, weights], [cot])
    if rows == 0:
        mx.eval(ya, ga)
        assert ya[0].shape == (0, dim)
        assert ga[0].shape == y.shape and ga[1].shape == weights.shape
        return
    yr, gr = mx.vjp(ref, [y, weights], [cot])
    mx.eval(ya, ga, yr, gr)
    for a, b in zip([ya[0], *ga], [yr[0], *gr]):
        np.testing.assert_allclose(np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32)),
                                   rtol=1e-2, atol=1e-3)


def test_packed_selection_compiles_with_dynamic_lengths():
    fn = mx.compile(lambda sc, reach: sa.select_topk_packed(sc, reach, 1))
    for score in ([1.0, 2.0, 3.0, 4.0], [1e10] * 4):
        offsets, _, (flat, lengths) = fn(mx.array([[score]], mx.float32), mx.ones((1, 1, 4), mx.bool_))
        mx.eval(offsets, flat, lengths)
        total = offsets[-1].item()
        expected = [3] if total == 1 else [0, 1, 2, 3]
        assert flat.size == 4
        assert np.asarray(flat[:total]).tolist() == expected


@pytest.mark.parametrize("dim", [64, 128])
def test_forward_tile32_uses_bounded_tile16_backward(dim):
    from test_sparse_attention_kernel import _inputs, _run_vjp, eager_indexed_attention
    q, w, c, visible, seg, pad, sink, W, scale = _inputs(2, 17, 7, dim, 8, seed=3)
    cot = mx.ones((2, 16, 17, dim), q.dtype)

    def tile32(*args):
        return sa.indexed_attention(*args, key_tile=32)

    y, grads = _run_vjp(tile32, q, w, c, sink, visible, seg, pad, W, scale, cot)
    yr, gr = _run_vjp(eager_indexed_attention, q, w, c, sink, visible, seg, pad, W, scale, cot)
    mx.eval(y, grads, yr, gr)
    for a, b in zip([y, *grads], [yr, *gr]):
        np.testing.assert_allclose(np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32)),
                                   rtol=8e-2, atol=2e-1)


@pytest.mark.parametrize("scores,reach,k", [
    ([100.0, 1.0], [False, True], 1),
    ([-0.0, 1e-7], [True, True], 1),
    ([float("inf"), float("inf"), 1.0], [True, True, True], 1),
    ([-1e30, -float("inf")], [True, True], 1),
    ([0.0, 0.0, 0.0], [True, True, True], 8),
])
def test_radix_dense_rank_domain_and_float_boundaries(scores, reach, k):
    from model.attention import _topk_masks
    scores, reach = mx.array([[scores]], mx.float32), mx.array([[reach]], mx.bool_)
    expected, _ = _topk_masks(scores, reach, k, 0, need_idx=False)
    got, _ = sa.select_topk(scores, reach, k)
    mx.eval(got, expected)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
