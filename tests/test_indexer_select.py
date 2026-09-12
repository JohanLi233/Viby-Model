"""Bit-exact score/keep checks against the original-order dense Metal path."""

import mlx.core as mx
import numpy as np
import pytest

from model.attention import _topk_masks, select_candidate_blocks
from model.kernels import indexer_score as ix, indexer_select as fs

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="need Metal")


@pytest.fixture(autouse=True)
def original_score_reference(monkeypatch):
    # The reference must remain the original single-query Metal arithmetic,
    # even after BQ becomes the application default.
    monkeypatch.setattr(fs, "_BQ_ENABLED", False)


def inputs(b, t, n, h, d, dtype):
    mx.set_default_device(mx.gpu)
    mx.random.seed(37)
    q = mx.random.normal((b, t, h, d)).astype(dtype)
    k = mx.random.normal((b, n, d)).astype(dtype)
    w = mx.random.normal((b, t, h)).astype(dtype)
    reach = mx.random.uniform(shape=(b, t, n)) > 0.4
    reach = reach & (mx.arange(n)[None, None] < mx.arange(t)[None, :, None] * 9)
    return q, k, w, reach


def assert_selection(selection, keep):
    ids, lengths = selection
    mx.eval(ids, lengths, keep)
    ids, lengths = np.asarray(ids), np.asarray(lengths)
    expected = np.asarray(keep).reshape(ids.shape)
    for row in range(ids.shape[0]):
        target = np.flatnonzero(expected[row])
        assert lengths[row] == len(target)
        np.testing.assert_array_equal(ids[row, : lengths[row]], target)


@pytest.mark.parametrize("b,t,n,h,d", [(2, 19, 47, 4, 32), (1, 256, 1024, 8, 64)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bq", [2, 4])
def test_bq_scores_and_fused_selection_exact(b, t, n, h, d, dtype, bq):
    q, k, w, reach = inputs(b, t, n, h, d, dtype)
    dense = ix.indexer_score(q, k, w, reach)
    got = fs.score_bq(q, k, w, reach, bq)
    mx.eval(dense, got)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(dense))
    keep, _ = _topk_masks(dense, reach, 7, 0, need_idx=False)
    selection, _ = fs.fused_select(q, k, w, reach, 7, bq=bq)
    assert_selection(selection, keep)


@pytest.mark.parametrize("cb", [1, 7, 32])
def test_candidate_source_and_disjoint_compact_lists(cb):
    b, t, n = 2, 19, 65
    q, k, w, reach = inputs(b, t, n, 8, 64, mx.bfloat16)
    latest = mx.broadcast_to(mx.arange(t)[None, :] * 3, (b, t))
    dense = ix.indexer_score(q, k, w, reach)
    keep, _ = _topk_masks(dense, reach, 5, 0, need_idx=False)
    selected, candidates = fs.fused_select(
        q,
        k,
        w,
        reach,
        5,
        candidate_source=True,
        latest=latest,
        block_size=cb,
        block_topk=2,
    )
    assert_selection(selected, keep)
    mask = fs.candidate_mask(candidates, b, t, n, cb)
    ref_mask = select_candidate_blocks(dense, latest, 2, cb)
    mx.eval(mask, ref_mask)
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(ref_mask))
    # The next layer uses a different Q and must keep original global-id ties.
    q2 = q * 0.5
    dense2 = ix.indexer_score(q2, k, w, reach & mask)
    expected, _ = _topk_masks(dense2, reach, 5, 0, need_idx=False)
    selected, _ = fs.fused_select(
        q2, k, w, reach, 5, candidates=candidates, block_size=cb
    )
    assert_selection(selected, expected)


def test_all_boundary_ties_and_compile():
    b, t, n, h, d = 1, 7, 73, 4, 32
    q = mx.ones((b, t, h, d), mx.bfloat16)
    k = mx.ones((b, n, d), mx.bfloat16)
    w = mx.full((b, t, h), 1e6, mx.bfloat16)
    run = mx.compile(lambda q, k, w, r: fs.fused_select(q, k, w, r, 1)[0])
    for reach in (mx.ones((b, t, n), mx.bool_), mx.zeros((b, t, n), mx.bool_)):
        selected = run(q, k, w, reach)
        assert_selection(selected, reach)


def test_compact_scores_vjp_scatter_to_global_keys():
    b, t, n, cb = 2, 9, 47, 7
    q, k, w, reach = inputs(b, t, n, 4, 32, mx.float16)
    nb = (n + cb - 1) // cb
    ids = mx.broadcast_to(
        mx.array([0, 3, 6, -123, -123, -123, -123], mx.int32), (b * t, nb)
    )
    lengths = (mx.arange(b * t) % 4).astype(mx.int32)
    mask = fs.candidate_mask((ids, lengths), b, t, n, cb)
    values, actual_lengths = fs.score_candidate_blocks(
        q, k, w, reach, (ids, lengths), cb
    )
    dense = ix.indexer_score(q, k, w, reach & mask)
    mx.eval(values, actual_lengths, dense)
    for row in range(b * t):
        count = lengths[row].item()
        gids = (
            np.concatenate(
                [np.arange(j * cb, min(n, (j + 1) * cb)) for j in [0, 3, 6][:count]]
            )
            if count
            else np.array([], int)
        )
        assert actual_lengths[row].item() == len(gids)
        np.testing.assert_array_equal(
            np.asarray(values).reshape(b * t, n)[row, : len(gids)],
            np.asarray(dense).reshape(b * t, n)[row, gids],
        )

    def compact_loss(a, kk, ww):
        scores, lens = fs.score_candidate_blocks(a, kk, ww, reach, (ids, lengths), cb)
        valid = mx.arange(n)[None, None, :] < lens.reshape(b, t, 1)
        return mx.sum(mx.where(valid, scores, 0))

    def dense_loss(a, kk, ww):
        return mx.sum(mx.where(mask, ix.indexer_score(a, kk, ww, reach & mask), 0))

    gc = mx.grad(compact_loss, argnums=(0, 1, 2))(q, k, w)
    gd = mx.grad(dense_loss, argnums=(0, 1, 2))(q, k, w)
    mx.eval(gc, gd)
    for a, z in zip(gc, gd):
        np.testing.assert_allclose(
            np.asarray(a.astype(mx.float32)),
            np.asarray(z.astype(mx.float32)),
            rtol=3e-3,
            atol=3e-3,
        )


def test_full_reuse_reindex_attention_compiled(monkeypatch):
    from _v41_common import cfg_mix
    from model.attention import Attention
    from model.cache import SharedAttnState
    from model.kernels.sparse_attention import prewarm_sparse_attention, prewarm_topk
    from mlx.utils import tree_map

    mx.set_default_device(mx.gpu)
    mx.random.seed(3)
    cfg = cfg_mix(
        n_heads=16,
        head_dim=64,
        window_size=8,
        index_n_heads=4,
        index_head_dim=32,
        candidate_block_size=7,
        candidate_topk_blocks=2,
        index_topk=5,
    )
    layers = [Attention(cfg, i) for i in (3, 4, 5)]
    for layer in layers:
        layer.update(tree_map(lambda a: a.astype(mx.bfloat16), layer.parameters()))
    prewarm_sparse_attention(64, 8, 64**-0.5)
    prewarm_topk(32)
    fs._kernel()
    x = mx.random.normal((2, 32, cfg.dim)).astype(mx.bfloat16)
    seg = mx.broadcast_to((mx.arange(32) // 13)[None], (2, 32))
    pad = mx.broadcast_to((mx.arange(32) < 29)[None], (2, 32))

    def stack(x):
        shared = SharedAttnState()
        for layer in layers:
            x = layer(x, 0, shared, segment_ids=seg, pad_mask=pad)
        return x

    results = []
    for fused in (False, True):
        monkeypatch.setattr(fs, "_ENABLED", fused)
        run = mx.compile(lambda a: mx.vjp(stack, [a], [mx.ones_like(a)]))
        result = run(x)
        mx.eval(result)
        results.append(result)
    for a, z in zip(
        (results[0][0][0], results[0][1][0]), (results[1][0][0], results[1][1][0])
    ):
        np.testing.assert_allclose(
            np.asarray(a.astype(mx.float32)),
            np.asarray(z.astype(mx.float32)),
            rtol=1e-2,
            atol=1e-2,
        )
