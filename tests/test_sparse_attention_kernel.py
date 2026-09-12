"""H=16 稀疏注意力 GPU kernel：前向与 dQ/dWindow/dCompressed/dSinks 对拍。

tiny 模型只有 4 头，通常打不中此 kernel，不能当 GPU 路径证明。
"""

import mlx.core as mx
import numpy as np
import pytest

from model.kernels.sparse_attention import enabled_for, indexed_attention


NEG_INF = -1e30


def eager_indexed_attention(
    q,
    window,
    compressed,
    visible,
    segment_ids,
    pad_mask,
    sinks,
    window_size,
    softmax_scale,
):
    """与 Metal 路径同公式的 MLX 参照：滑窗 + 压缩 + learned sink。"""
    b, t, h, d = q.shape
    n = int(compressed.shape[1])
    qf = q.astype(mx.float32)
    wf = window.astype(mx.float32)
    cf = compressed.astype(mx.float32)
    scale = mx.array(softmax_scale, mx.float32)
    segment = (
        mx.zeros((b, t), mx.int32)
        if segment_ids is None
        else segment_ids.astype(mx.int32)
    )
    pad = mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_)
    slot = mx.arange(window_size)
    pos = mx.arange(t)[:, None] - window_size + 1 + slot[None, :]
    in_range = pos >= 0
    clipped = mx.maximum(pos, 0).astype(mx.int32)
    batch = mx.arange(b)[:, None, None]
    keys_w = wf[batch, clipped[None, :, :]]
    valid_w = (
        in_range[None, :, :]
        & pad[batch, clipped[None, :, :]]
        & (segment[batch, clipped[None, :, :]] == segment[:, :, None])
    )
    scores_w = mx.sum(qf[:, :, :, None, :] * keys_w[:, :, None, :, :], axis=-1) * scale
    scores_w = mx.where(valid_w[:, :, None, :], scores_w, mx.array(NEG_INF, mx.float32))
    parts = [scores_w]
    if n > 0:
        vis = mx.broadcast_to(visible, (b, t, n))
        scores_c = (
            mx.sum(qf[:, :, :, None, :] * cf[:, None, None, :, :], axis=-1) * scale
        )
        scores_c = mx.where(vis[:, :, None, :], scores_c, mx.array(NEG_INF, mx.float32))
        parts.append(scores_c)
    sink = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, 1, h, 1), (b, t, h, 1))
    logits = mx.concatenate(parts + [sink], axis=-1)
    lse = mx.logsumexp(logits, axis=-1, keepdims=True)
    prob = mx.exp(logits - lse)
    out = mx.sum(prob[..., :window_size, None] * keys_w[:, :, None, :, :], axis=-2)
    if n > 0:
        out = out + mx.sum(
            prob[..., window_size : window_size + n, None] * cf[:, None, None, :, :],
            axis=-2,
        )
    return out.astype(q.dtype).transpose(0, 2, 1, 3)


def _require_gpu_kernel(head_dim, window):
    if not mx.metal.is_available():
        pytest.skip("need GPU")
    mx.set_default_device(mx.gpu)
    q = mx.zeros((1, 1, 16, head_dim), mx.bfloat16)
    if not enabled_for(q, window):
        pytest.skip("sparse attention kernel not enabled for this shape")


def _inputs(b, t, n, head_dim, window, seed, dtype=mx.bfloat16):
    mx.random.seed(seed)
    q = mx.random.normal((b, t, 16, head_dim)).astype(dtype)
    window_kv = mx.random.normal((b, t, head_dim)).astype(dtype)
    compressed = (
        mx.random.normal((b, n, head_dim)).astype(dtype)
        if n
        else mx.zeros((b, 0, head_dim), dtype)
    )
    if n:
        causal = mx.arange(n)[None, None, :] <= mx.arange(t)[None, :, None]
        keep = mx.random.uniform(shape=(b, t, n)) > 0.35
        visible = mx.broadcast_to(causal, (b, t, n)) & keep
    else:
        visible = mx.zeros((b, t, 0), mx.bool_)
    segment = mx.zeros((b, t), mx.int32)
    if t >= 8:
        segment = segment.at[:, t // 2 :].add(1)
    pad = mx.ones((b, t), mx.bool_)
    if t >= 4:
        pad = pad & (mx.arange(t)[None, :] < (t - 1))
    sinks = mx.random.normal((16,)).astype(mx.float32)
    scale = head_dim**-0.5
    return q, window_kv, compressed, visible, segment, pad, sinks, window, scale


def _arr(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return x


def _finite_close(got, ref, rtol, atol):
    g = np.asarray(_arr(got).astype(mx.float32))
    r = np.asarray(_arr(ref).astype(mx.float32))
    assert g.shape == r.shape
    assert np.isfinite(g).all()
    assert np.isfinite(r).all()
    assert np.allclose(g, r, rtol=rtol, atol=atol)


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU MMA")
@pytest.mark.parametrize(
    "seq,keys,dim,window",
    [
        (1, 1, 128, 32),
        (8, 4, 128, 8),
        (17, 17, 64, 16),
        (4, 0, 128, 4),
    ],
)
def test_indexed_attention_forward_matches_eager(seq, keys, dim, window):
    _require_gpu_kernel(dim, window)
    q, wkv, ckv, vis, seg, pad, sinks, W, scale = _inputs(
        1, seq, keys, dim, window, seed=3
    )
    got = indexed_attention(q, wkv, ckv, vis, seg, pad, sinks, W, scale)
    ref = eager_indexed_attention(q, wkv, ckv, vis, seg, pad, sinks, W, scale)
    mx.eval(got, ref)
    _finite_close(got, ref, rtol=5e-2, atol=2e-1)


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU MMA")
def test_indexed_attention_vjp_matches_eager():
    dim, window, t, n = 128, 8, 8, 4
    _require_gpu_kernel(dim, window)
    q, wkv, ckv, vis, seg, pad, sinks, W, scale = _inputs(1, t, n, dim, window, seed=5)
    cot = mx.random.normal((1, 16, t, dim)).astype(mx.bfloat16)

    def run(fn):
        def wrapped(a, w, c, s):
            return fn(a, w, c, vis, seg, pad, s, W, scale)

        return mx.vjp(wrapped, [q, wkv, ckv, sinks], [cot])

    y0, g0 = run(indexed_attention)
    y1, g1 = run(eager_indexed_attention)
    mx.eval(y0, y1, g0, g1)
    names = ("dQ", "dWindow", "dCompressed", "dSinks")
    _finite_close(y0, y1, rtol=5e-2, atol=2e-1)
    for name, a, b in zip(names, g0, g1):
        _finite_close(a, b, rtol=8e-2, atol=2e-1)


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU MMA")
def test_indexed_attention_independent_window_and_compressed_lengths():
    dim, window = 64, 16
    _require_gpu_kernel(dim, window)
    q, wkv, ckv, vis, seg, pad, sinks, W, scale = _inputs(2, 16, 7, dim, window, seed=9)
    assert wkv.shape[1] == q.shape[1]
    assert ckv.shape[1] == vis.shape[-1] != q.shape[1]
    out = indexed_attention(q, wkv, ckv, vis, seg, pad, sinks, W, scale)
    mx.eval(out)
    assert tuple(out.shape) == (2, 16, 16, dim)
    assert np.isfinite(np.asarray(out.astype(mx.float32))).all()


def _run_vjp(fn, q, wkv, ckv, sinks, vis, seg, pad, W, scale, cot):
    def wrapped(a, w, c, s):
        return fn(a, w, c, vis, seg, pad, s, W, scale)

    out, grads = mx.vjp(wrapped, [q, wkv, ckv, sinks], [cot])
    return out[0] if isinstance(out, list) else out, list(grads)


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU MMA")
@pytest.mark.parametrize(
    "batch,seq,keys,dim,window",
    [
        (2, 32, 16, 64, 16),
        (2, 17, 7, 64, 8),
        (3, 48, 24, 128, 16),
        (2, 64, 0, 64, 16),
    ],
)
def test_indexed_attention_split_backward_matches_eager(batch, seq, keys, dim, window):
    """`VIBY_SPARSE_ATTN_BWD_SPLIT=1`：query-owned dQ/dSink + 片上归约 dKV 路径。"""
    from model.kernels import sparse_attention as sa

    _require_gpu_kernel(dim, window)
    q, wkv, ckv, vis, seg, pad, sinks, W, scale = _inputs(
        batch, seq, keys, dim, window, seed=11
    )
    mx.random.seed(3)
    cot = mx.random.normal((batch, 16, seq, dim)).astype(mx.bfloat16)

    prev = sa._SPLIT_BWD
    sa._SPLIT_BWD = True
    sa._operation.cache_clear()
    try:
        y0, g0 = _run_vjp(
            sa.indexed_attention, q, wkv, ckv, sinks, vis, seg, pad, W, scale, cot
        )
        y1, g1 = _run_vjp(
            eager_indexed_attention, q, wkv, ckv, sinks, vis, seg, pad, W, scale, cot
        )
        mx.eval(y0, y1, g0, g1)
        assert np.isfinite(np.asarray(y0.astype(mx.float32))).all()
        names = ("dQ", "dWindow", "dCompressed", "dSinks")
        _finite_close(y0, y1, rtol=5e-2, atol=2e-1)
        for name, a, b in zip(names, g0, g1):
            arr = np.asarray(a.astype(mx.float32))
            assert np.isfinite(arr).all(), f"{name} produced non-finite values"
            _finite_close(a, b, rtol=8e-2, atol=2e-1)
    finally:
        sa._SPLIT_BWD = prev
        sa._operation.cache_clear()


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU")
def test_select_topk_packed_matches_row_major_compact():
    """§5.4 packed 输出：行内保序的一维紧凑 + exclusive row_offsets。"""
    from model.kernels.sparse_attention import select_topk, select_topk_packed

    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    # 用全可达的行，保证走的是 §5.1 的 Indexer 快路径（阈值只在有效域内取）。
    scores = mx.array(
        [
            [
                [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
                [1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            ]
        ],
        mx.float32,
    )
    reach = mx.ones((1, 3, 8), mx.bool_)
    keep, (indices, lengths) = select_topk(scores, reach, 2)
    row_offsets, keep2, (flat, lengths2) = select_topk_packed(scores, reach, 2)
    mx.eval(keep, indices, lengths, row_offsets, keep2, flat, lengths2)

    assert np.array_equal(np.asarray(keep), np.asarray(keep2))
    assert np.array_equal(np.asarray(lengths), np.asarray(lengths2))
    counts = np.asarray(lengths)
    assert np.array_equal(
        np.asarray(row_offsets), np.concatenate([[0], np.cumsum(counts)])
    )
    idx = np.asarray(indices)
    got = np.asarray(flat)
    expected = np.concatenate([idx[r, : int(counts[r])] for r in range(len(counts))])
    assert got.size == indices.size  # static arena, not a host-sized allocation
    assert np.array_equal(got[: expected.size], expected)
