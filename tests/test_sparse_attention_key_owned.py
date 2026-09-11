"""Targeted acceptance for the occurrence-preserving, non-atomic dKV backend."""
import mlx.core as mx
import numpy as np
import pytest

from model.kernels import sparse_attention as sa
from test_sparse_attention_kernel import _inputs, _run_vjp, eager_indexed_attention


pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="need Metal")


@pytest.mark.parametrize("heads", [1, 16])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("score_path", [False, True])
def test_one_key_value_path_writes_every_dimension(heads, dim, score_path):
    """Binary-exact operands isolate both dKV paths and all dimension stripes."""
    mx.set_default_device(mx.gpu)
    q = mx.full((1, 1, heads, dim), 2 if score_path else 0, mx.float16)
    g = (mx.arange(heads * dim).reshape(q.shape) % 13 + 1).astype(q.dtype)
    window = mx.zeros((1, 1, dim), q.dtype)
    delta = mx.arange(heads).reshape(1, 1, heads).astype(mx.float32)
    (got,) = sa._key_owned_kernels()[-2](
        inputs=[q, window, mx.zeros((1, 0, dim), q.dtype),
                mx.full((1, 1, heads), np.log(4), mx.float32), g,
                delta, mx.zeros((1,), mx.int32),
                mx.zeros((0,), mx.int32), mx.zeros((0,), mx.int32),
                mx.zeros((1, 1), mx.int32), mx.ones((1, 1), mx.bool_),
                mx.array([1, 1, 0], mx.uint32), mx.array([1.0], mx.float32)],
        template=[("T", q.dtype), ("D", dim), ("H", heads),
                  ("WIN", 1), ("COMPRESSED", False), ("OutT", q.dtype)],
        grid=(128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, dim)], output_dtypes=[q.dtype],
    )
    expected = mx.sum(g.astype(mx.float32) - delta[..., None] * q.astype(mx.float32), axis=(1, 2)) * 0.25
    mx.eval(got, expected)
    np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(expected))


@pytest.mark.parametrize("size", [0, 1, 255, 256, 257, 512, 65537])
def test_csr_exclusive_scan(size):
    counts = np.arange(size, dtype=np.int32) % 7
    result = sa._csr_exclusive_scan(mx.array(counts))
    expected = np.concatenate([np.zeros(1, np.int32), np.cumsum(counts)])
    np.testing.assert_array_equal(np.asarray(result), expected)
    compiled = mx.compile(sa._csr_exclusive_scan)
    for values in (counts, counts // 2):
        expected = np.concatenate([np.zeros(1, np.int32), np.cumsum(values)])
        np.testing.assert_array_equal(np.asarray(compiled(mx.array(values))), expected)


def test_csr_preserves_occurrences_and_ignores_unused_slots():
    b, t, n = 2, 3, 4
    indices = np.array([[2, 2, 0, -123], [3, -123, -123, -123],
                        [-123, -123, -123, -123], [1, 1, 1, 1],
                        [0, 2, -123, -123], [2, 0, 2, -123]], np.int32)
    lengths = np.array([3, 1, 0, 4, 2, 3], np.int32)
    row_ptr, edge_q, edge_slot = sa._compressed_occurrence_csr(
        mx.array(indices), mx.array(lengths), b, t, n)
    mx.eval(row_ptr, edge_q, edge_slot)
    rp, eq, es = map(np.asarray, (row_ptr, edge_q, edge_slot))
    assert rp[-1] == lengths.sum()
    assert eq.size == indices.size
    for key in range(b * n):
        expected = sorted((q, slot) for q in range(b * t) for slot in range(lengths[q])
                          if q // t * n + indices[q, slot] == key)
        actual = sorted(zip(eq[rp[key]:rp[key + 1]], es[rp[key]:rp[key + 1]]))
        assert actual == expected


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("batch,seq,keys,dim,window", [
    (1, 1, 1, 64, 1),
    (2, 17, 7, 64, 8),
    (3, 48, 24, 128, 16),
    (2, 33, 0, 128, 16),
])
def test_key_owned_vjp_matches_query_owned_and_eager(monkeypatch, dtype, batch, seq, keys, dim, window):
    mx.set_default_device(mx.gpu)
    q, w, c, vis, seg, pad, sinks, W, scale = _inputs(
        batch, seq, keys, dim, window, seed=21, dtype=dtype)
    cot = mx.random.normal((batch, 16, seq, dim)).astype(dtype)
    monkeypatch.setattr(sa, "_SPLIT_BWD", True)
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", True)
    y, grads = _run_vjp(sa.indexed_attention, q, w, c, sinks, vis, seg, pad, W, scale, cot)
    mx.eval(y, grads)
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", False)
    yref, gref = _run_vjp(sa.indexed_attention, q, w, c, sinks, vis, seg, pad, W, scale, cot)
    yeager, geager = _run_vjp(eager_indexed_attention, q, w, c, sinks, vis, seg, pad, W, scale, cot)
    mx.eval(yref, gref, yeager, geager)
    np.testing.assert_array_equal(np.asarray(y.astype(mx.float32)), np.asarray(yref.astype(mx.float32)))
    for name, got, ref, eager in zip(("dq", "dw", "dc", "ds"), grads, gref, geager):
        actual, expected, exact = [np.asarray(x.astype(mx.float32)) for x in (got, ref, eager)]
        if name in ("dq", "ds"):
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        else:
            _assert_pool_close(actual, expected, dtype, name)
        # The existing MMA operands round P and Ds to the input dtype; the
        # full FP32 eager derivative is a separate, less exact comparison.
        np.testing.assert_allclose(actual, exact, rtol=8e-2, atol=2e-1, err_msg=name)


def _assert_pool_close(actual, expected, dtype, name):
    # Different FP32 dot reduction trees can cross a P/Ds rounding midpoint.
    # Bound both local cancellation error and the global relative L2 error.
    bf16 = dtype == mx.bfloat16
    np.testing.assert_allclose(actual, expected, rtol=1e-2 if bf16 else 2e-3,
                               atol=1e-2 if bf16 else 2e-3, err_msg=name)
    relative_l2 = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
    assert relative_l2 < (2e-3 if bf16 else 3e-4), (name, relative_l2)


@pytest.mark.parametrize("keys", [0, 7])
def test_empty_adjacency_writes_zero_gradients(monkeypatch, keys):
    q, w, c, vis, seg, pad, sinks, W, scale = _inputs(2, 9, keys, 64, 16, seed=5)
    pad, vis = mx.zeros_like(pad), mx.zeros_like(vis)
    cot = mx.ones((2, 16, 9, 64), q.dtype)
    monkeypatch.setattr(sa, "_SPLIT_BWD", True)
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", True)
    y, grads = _run_vjp(sa.indexed_attention, q, w, c, sinks, vis, seg, pad, W, scale, cot)
    mx.eval(y, grads)
    for value in (y, *grads):
        np.testing.assert_array_equal(np.asarray(value.astype(mx.float32)), 0)


@pytest.mark.parametrize("heads,dim", [(1, 64), (16, 128)])
def test_compressed_hot_key_parts_cover_all_occurrences(heads, dim):
    degrees = [0, 1, 32, 33, 64, 65, 96, 97, 4101]
    t, n = sum(degrees), len(degrees)
    idx = np.full((t, n), -123, np.int32)
    idx[:, 0] = np.repeat(np.arange(n), degrees)
    rp, eq, es = sa._compressed_occurrence_csr(mx.array(idx), mx.ones((t,), mx.int32), 1, t, n)
    q = mx.zeros((1, t, heads, dim), mx.float16)
    (got,) = sa._key_owned_kernels()[-2](
        inputs=[q, mx.zeros((1, t, dim), q.dtype), mx.zeros((1, n, dim), q.dtype),
                mx.full((1, t, heads), np.log(4), mx.float32), mx.ones_like(q),
                mx.zeros((1, t, heads), mx.float32), rp, eq, es,
                mx.zeros((1, t), mx.int32), mx.zeros((1, t), mx.bool_),
                mx.array([1, t, n], mx.uint32), mx.array([1.0], mx.float32)],
        template=[("T", q.dtype), ("D", dim), ("H", heads), ("WIN", 1),
                  ("COMPRESSED", True), ("OutT", mx.float32)],
        grid=(128, n, 4), threadgroup=(128, 1, 1),
        output_shapes=[(4, n, dim)], output_dtypes=[mx.float32],
    )
    expected = np.zeros((4, n, dim), np.float32)
    for key, degree in enumerate(degrees):
        parts = min(4, max(1, (degree + 31) // 32))
        for part in range(parts):
            expected[part, key] = len(range(part, degree, parts)) * heads * 0.25
    np.testing.assert_array_equal(np.asarray(got), expected)


def _eager_occurrence_attention(q, w, c, indices, lengths, seg, pad, sinks, W, scale):
    """Small independent multiset oracle: gather each slot, including duplicates."""
    b, t, h, d = q.shape
    n = c.shape[1]
    pos = mx.arange(t)[:, None] - W + 1 + mx.arange(W)[None, :]
    safe_pos = mx.maximum(pos, 0)
    batch = mx.arange(b)[:, None, None]
    wk = w.astype(mx.float32)[batch, safe_pos[None]]
    valid_w = (pos[None] >= 0) & pad[batch, safe_pos[None]] & (seg[batch, safe_pos[None]] == seg[:, :, None])
    idx = indices.reshape(b, t, n)
    valid_c = mx.arange(n)[None, None, :] < lengths.reshape(b, t, 1)
    ck = c.astype(mx.float32)[batch, mx.where(valid_c, idx, 0)]
    keys = mx.concatenate([wk, ck], axis=2)
    valid = mx.concatenate([valid_w, valid_c], axis=2)
    scores = mx.sum(q.astype(mx.float32)[:, :, :, None, :] * keys[:, :, None, :, :], axis=-1) * scale
    scores = mx.where(valid[:, :, None, :], scores, -float("inf"))
    logits = mx.concatenate([scores, mx.broadcast_to(sinks.reshape(1, 1, h, 1), (b, t, h, 1))], axis=-1)
    p = mx.softmax(logits, axis=-1)[..., :-1]
    return mx.sum(p[..., None] * keys[:, :, None], axis=-2).astype(q.dtype).transpose(0, 2, 1, 3)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_duplicate_selection_vjp_and_shared_primal(monkeypatch, dtype):
    b, t, n, d, W = 2, 19, 5, 64, 8
    q, w, c, _, seg, pad, sinks, _, scale = _inputs(b, t, n, d, W, seed=9, dtype=dtype)
    # Row slots 0, 1, 3 all name key 0. Key 4 has degree zero. The first
    # document and mixed padding still affect only the window pool.
    indices = mx.broadcast_to(mx.array([0, 0, 2, 0, 3], mx.int32), (b * t, n))
    lengths = (mx.arange(b * t) % (n + 1)).astype(mx.int32)
    cot = mx.random.normal((b, 16, t, d)).astype(dtype)
    monkeypatch.setattr(sa, "_SPLIT_BWD", True)

    def run(key_owned, eager=False, compiled=False):
        monkeypatch.setattr(sa, "_KEY_OWNED_BWD", key_owned)

        def fn(a, window, compressed, s):
            def layer(qq):
                if eager:
                    return _eager_occurrence_attention(qq, window, compressed, indices, lengths, seg, pad, s, W, scale)
                return sa.indexed_attention(qq, window, compressed, None, seg, pad, s,
                                            W, scale, selection=(indices, lengths))
            # Two layers depend on the same compressed primal and metadata.
            return layer(a) + layer(a * 0.5)

        def vjp(a, window, compressed, s, g):
            return mx.vjp(fn, [a, window, compressed, s], [g])
        result = (mx.compile(vjp) if compiled else vjp)(q, w, c, sinks, cot)
        mx.eval(result)
        return result

    y, grads = run(True)
    yc, gc = run(True, compiled=True)
    yr, gr = run(False)
    ye, ge = run(False, eager=True)
    for got, expected in zip(grads, gc):
        np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))
    for name, got, ref, eager in zip(("dq", "dw", "dc", "ds"), grads, gr, ge):
        actual, expected, exact = [np.asarray(x.astype(mx.float32)) for x in (got, ref, eager)]
        _assert_pool_close(actual, expected, dtype, name)
        np.testing.assert_allclose(actual, exact, rtol=8e-2, atol=2e-1, err_msg=name)
    np.testing.assert_array_equal(np.asarray(grads[2][:, 4].astype(mx.float32)), 0)
    for output in (yc, yr, ye):
        np.testing.assert_allclose(np.asarray(y[0].astype(mx.float32)),
                                   np.asarray(output[0].astype(mx.float32)), rtol=5e-2, atol=2e-1)
