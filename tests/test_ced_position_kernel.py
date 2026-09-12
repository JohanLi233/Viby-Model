"""Explicit original-token spans, compact routes, and every custom-VJP leaf."""

import mlx.core as mx
import numpy as np
import pytest

from model.kernels import sparse_attention as sa


pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="requires Metal MMA"
)


def _case(dim, dtype=mx.bfloat16):
    mx.set_default_device(mx.gpu)
    mx.random.seed(12091)
    b, t, n = 2, 7, 57
    positions = mx.array([[3, 7, 11, 31, 35, 50, 54], [3, 8, 12, 16, 20, 39, 43]])
    docs = mx.array([[0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 1, 1]])
    pad = mx.array([[1, 1, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], mx.bool_)
    q = (mx.random.normal((b, t, 16, dim)) * 0.5).astype(dtype)
    window = mx.random.normal((b, t, dim)).astype(dtype)
    memory = mx.random.normal((b, n, dim)).astype(dtype)
    sinks = mx.random.normal((16,)).astype(mx.float32)
    # Already-validated evidence routes: varying counts and unused padded slots.
    ids = mx.stack(
        [positions - 1, positions, mx.zeros_like(positions)], axis=-1
    ).reshape(b * t, 3)
    lengths = mx.where(pad, 2, 0).reshape(b * t).astype(mx.int32)
    return (q, window, memory, sinks), positions, docs, pad, (ids, lengths)


def _oracle(q, window, memory, sinks, positions, docs, pad, selection, token_window=13):
    """Independent full-matrix formula, including learned zero-value sinks."""
    b, t, h, _ = q.shape
    n = memory.shape[1]
    ids, lengths = selection
    ids = ids.reshape(b, t, -1)
    valid = mx.arange(ids.shape[-1])[None, None, :] < lengths.reshape(b, t, 1)
    visible_memory = mx.any(
        (mx.arange(n)[None, None, None, :] == ids[..., None]) & valid[..., None],
        axis=-2,
    )
    visible_window = (
        (positions[:, None, :] <= positions[:, :, None])
        & (positions[:, None, :] > positions[:, :, None] - token_window)
        & (docs[:, None, :] == docs[:, :, None])
        & pad[:, None, :]
        & pad[:, :, None]
    )
    keys = mx.concatenate([window, memory], axis=1).astype(mx.float32)
    scores = (
        mx.einsum("bthd,bsd->bhts", q.astype(mx.float32), keys) * q.shape[-1] ** -0.5
    )
    visible = mx.concatenate([visible_window, visible_memory], axis=-1)
    scores = mx.where(visible[:, None], scores, -1e30)
    sink = mx.broadcast_to(sinks[None, :, None, None], (b, h, t, 1))
    probabilities = mx.softmax(mx.concatenate([scores, sink], axis=-1), axis=-1)
    return mx.einsum("bhts,bsd->bhtd", probabilities[..., :-1], keys).astype(q.dtype)


def _error(actual, expected):
    a, e = (
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )
    assert np.isfinite(a).all()
    return np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-8)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("backend", ["fused", "split", "legacy", "key_owned_requested"])
def test_positions_compact_rectangular_forward_and_vjp(monkeypatch, dim, backend):
    monkeypatch.setattr(sa, "_SPLIT_BWD", backend != "legacy")
    monkeypatch.setattr(sa, "_FUSED_BWD", backend in ("fused", "key_owned_requested"))
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", backend == "key_owned_requested")
    sa._position_operation.cache_clear()
    values, positions, docs, pad, selection = _case(dim)
    if backend == "key_owned_requested":
        # This experimental backend uses implicit adjacency; position mode must
        # take its documented sharded fallback rather than silently mis-mask.
        def forbidden(*args, **kwargs):
            raise AssertionError("position mode entered implicit key-owned adjacency")

        monkeypatch.setattr(sa, "_key_owned_pool_grad", forbidden)

    def fused(q, w, c, s):
        return sa.indexed_attention(
            q,
            w,
            c,
            None,
            docs,
            pad,
            s,
            4,
            dim**-0.5,
            selection=selection,
            query_positions=positions,
            token_window_size=13,
        )

    def oracle(q, w, c, s):
        return _oracle(q, w, c, s, positions, docs, pad, selection)

    cot = mx.random.normal((2, 16, 7, dim)).astype(values[0].dtype)
    out, grads = mx.vjp(fused, list(values), [cot])
    reference, expected_grads = mx.vjp(oracle, list(values), [cot])
    mx.eval(out, grads, reference, expected_grads)
    actual = out[0] if isinstance(out, list) else out
    expected = reference[0] if isinstance(reference, list) else reference
    assert _error(actual, expected) < 0.015
    for name, g, ref in zip(
        ("q", "self-kv", "evidence-kv", "sink"), grads, expected_grads
    ):
        assert _error(g, ref) < 0.045, name
    assert bool(mx.all(actual[0, :, [2, 6]] == 0))
    sa._position_operation.cache_clear()


def test_actual_token_gap_removes_hidden_and_gradient_path():
    values, _, _, _, _ = _case(64)
    q, w, c, sinks = [v[:1] if v.ndim > 1 else v for v in values]
    positions = mx.array([[3, 7, 31, 35, 39, 43, 47]])
    ids = mx.zeros((7, 0), mx.int32)
    lengths = mx.zeros((7,), mx.int32)

    def run(window):
        return sa.indexed_attention(
            q,
            window,
            c,
            None,
            None,
            None,
            sinks,
            4,
            64**-0.5,
            selection=(ids, lengths),
            query_positions=positions,
            token_window_size=13,
        )

    original = run(w)
    changed = run(w.at[:, 0].add(8.0))
    gradient = mx.grad(lambda window: run(window)[:, :, 2].astype(mx.float32).sum())(w)
    mx.eval(original, changed, gradient)
    assert bool(mx.array_equal(original[:, :, 2:], changed[:, :, 2:]))
    assert bool(mx.all(gradient[:, :2] == 0))
    assert bool(mx.any(gradient[:, 2] != 0))


def test_compiled_position_vjp_preserves_all_nine_array_leaves():
    values, positions, docs, pad, selection = _case(64, mx.float16)
    q, w, c, sinks = values
    ids, lengths = selection
    arrays = [q, w, c, ids, lengths, docs, pad, sinks, positions]
    operation = sa._position_operation(4, 13, 64**-0.5, 16)

    def objective(*args):
        out, _ = operation(*args)
        return out.sum()

    call = mx.compile(mx.value_and_grad(objective, argnums=tuple(range(9))))
    loss, gradients = call(*arrays)
    mx.eval(loss, gradients)
    assert bool(mx.isfinite(loss))
    assert len(gradients) == len(arrays) == 9
    for i, (gradient, primal) in enumerate(zip(gradients, arrays)):
        assert gradient.shape == primal.shape
        assert gradient.dtype == primal.dtype
        assert bool(mx.all(mx.isfinite(gradient)))
        if i in (3, 4, 5, 6, 8):
            assert bool(mx.all(gradient == 0))


def test_regular_positions_keep_legacy_kernel_math(monkeypatch):
    monkeypatch.setattr(sa, "_KEY_OWNED_BWD", False)
    values, _, docs, _, selection = _case(64)
    q, w, c, sinks = values
    b, t = q.shape[:2]
    pos = mx.broadcast_to(mx.arange(t)[None], (b, t))
    compact, lengths = selection
    # The old kernel has NC as its metadata row stride.
    full = mx.concatenate(
        [compact, mx.zeros((b * t, c.shape[1] - compact.shape[-1]), mx.int32)], axis=-1
    )
    legacy = sa.indexed_attention(
        q, w, c, None, docs, None, sinks, 4, 64**-0.5, selection=(full, lengths)
    )
    positioned = sa.indexed_attention(
        q,
        w,
        c,
        None,
        docs,
        None,
        sinks,
        4,
        64**-0.5,
        selection=(compact, lengths),
        query_positions=pos,
        token_window_size=4,
    )
    mx.eval(legacy, positioned)
    assert bool(mx.array_equal(legacy, positioned))
