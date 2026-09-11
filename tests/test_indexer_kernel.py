"""Indexer 融合打分 kernel：与两次 einsum 的前向 / VJP 对拍。"""
import mlx.core as mx
import numpy as np
import pytest

from model.kernels.indexer_score import eager_indexer_score, indexer_score


def _devices():
    yield pytest.param("cpu", id="cpu")
    if mx.metal.is_available():
        yield pytest.param("gpu", id="gpu")


@pytest.mark.parametrize("device", list(_devices()))
@pytest.mark.parametrize("heads,dim", [(4, 32), (8, 64)])
def test_indexer_score_matches_einsum(device, heads, dim):
    mx.set_default_device(getattr(mx, device))
    mx.random.seed(0)
    B, T, N = 2, 16, 16
    q = mx.random.normal((B, T, heads, dim)).astype(mx.bfloat16)
    k = mx.random.normal((B, N, dim)).astype(mx.bfloat16)
    w = mx.random.normal((B, T, heads)).astype(mx.bfloat16)
    reach = (mx.arange(N)[None, None, :] <= mx.arange(T)[None, :, None])
    reach = mx.broadcast_to(reach, (B, T, N))
    got = indexer_score(q, k, w, reach)
    ref = eager_indexer_score(q, k, w, reach)
    mx.eval(got, ref)
    g = np.asarray(got, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    finite = r > -1e20
    assert ((g > -1e20) == finite).all()
    assert np.allclose(g[finite], r[finite], rtol=5e-2, atol=2e-1)


@pytest.mark.parametrize("device", list(_devices()))
def test_indexer_score_vjp_matches_einsum(device):
    mx.set_default_device(getattr(mx, device))
    mx.random.seed(1)
    B, T, H, D, N = 1, 8, 4, 32, 8
    q = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, N, D)).astype(mx.bfloat16)
    w = mx.random.normal((B, T, H)).astype(mx.bfloat16)
    reach = mx.ones((B, T, N), dtype=mx.bool_)
    cot = mx.random.normal((B, T, N)).astype(mx.float32)

    def run(fn):
        return mx.vjp(fn, [q, k, w, reach], [cot])

    (y0, g0) = run(indexer_score)
    (y1, g1) = run(eager_indexer_score)
    mx.eval(y0, y1, g0, g1)
    assert np.allclose(np.asarray(y0), np.asarray(y1), rtol=5e-2, atol=2e-1)
    for a, b in zip(g0[:3], g1[:3]):
        aa = np.asarray(a.astype(mx.float32))
        bb = np.asarray(b.astype(mx.float32))
        assert np.allclose(aa, bb, rtol=8e-2, atol=2e-1)


def test_indexer_score_empty_keys():
    q = mx.ones((1, 2, 4, 32), mx.bfloat16)
    k = mx.zeros((1, 0, 32), mx.bfloat16)
    w = mx.ones((1, 2, 4), mx.bfloat16)
    r = mx.zeros((1, 2, 0), mx.bool_)
    y = indexer_score(q, k, w, r)
    mx.eval(y)
    assert y.shape == (1, 2, 0)


def _compare(q, k, w, reach, rtol=5e-2, atol=2e-1, vjp=False):
    if not vjp:
        got = indexer_score(q, k, w, reach)
        ref = eager_indexer_score(q, k, w, reach)
        mx.eval(got, ref)
        g = np.asarray(got, dtype=np.float32)
        r = np.asarray(ref, dtype=np.float32)
        finite = r > -1e20
        assert ((g > -1e20) == finite).all()
        assert np.allclose(g[finite], r[finite], rtol=rtol, atol=atol)
        return
    cot = mx.random.normal(reach.shape).astype(mx.float32)
    y0, g0 = mx.vjp(indexer_score, [q, k, w, reach], [cot])
    y1, g1 = mx.vjp(eager_indexer_score, [q, k, w, reach], [cot])
    mx.eval(y0, y1, g0, g1)
    assert np.allclose(np.asarray(y0), np.asarray(y1), rtol=rtol, atol=atol)
    for a, b in zip(g0[:3], g1[:3]):
        assert np.allclose(
            np.asarray(a.astype(mx.float32)),
            np.asarray(b.astype(mx.float32)),
            rtol=8e-2,
            atol=2e-1,
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="need GPU MMA")
@pytest.mark.parametrize("seq,keys,heads,dim", [
    (1, 64, 8, 64),   # decode：T=1，key 维切开
    (64, 64, 8, 64),  # 训练：多块 BK=16
    (17, 17, 4, 32),  # N 不整除 BK
])
def test_indexer_score_mma_shapes(seq, keys, heads, dim):
    mx.set_default_device(mx.gpu)
    mx.random.seed(2)
    q = mx.random.normal((2, seq, heads, dim)).astype(mx.bfloat16)
    k = mx.random.normal((2, keys, dim)).astype(mx.bfloat16)
    w = mx.random.normal((2, seq, heads)).astype(mx.bfloat16)
    reach = (mx.arange(keys)[None, None, :] <= mx.arange(seq)[None, :, None])
    reach = mx.broadcast_to(reach, (2, seq, keys))
    _compare(q, k, w, reach)
    _compare(q, k, w, reach, vjp=True)
