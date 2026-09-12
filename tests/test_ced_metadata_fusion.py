"""Exact integer selection checks, including noncanonical direct callers."""

import mlx.core as mx
import numpy as np
import pytest

from model.kernels.recurrent_metadata import validate_compact


@pytest.mark.parametrize("width", [1, 3, 32, 64, 129, 256])
@pytest.mark.parametrize("ordered", [False, True])
def test_exact_validation_and_compaction(width, ordered):
    rng = np.random.default_rng(29 + width)
    b, q, n = 2, 7, 137
    ids = rng.integers(-3, n + 3, size=(b, q, width), dtype=np.int32)
    if ordered:
        ids.sort(axis=-1)
    qp = np.broadcast_to(np.array([3, 8, 17, 32, 64, 128, 136], np.int32), (b, q))
    mp = np.broadcast_to(np.arange(n, dtype=np.int32), (b, n))
    ms = np.broadcast_to(np.arange(n, dtype=np.int32) // 17, (b, n))
    qs = qp // 17
    qpad = np.array([[1, 1, 0, 1, 1, 1, 1], [0] * 7], bool)
    mpad = rng.random((b, n)) > 0.2
    safe = np.clip(ids, 0, n - 1)
    rows = np.arange(b)[:, None, None]
    valid = (
        (ids >= 0)
        & (ids < n)
        & qpad[..., None]
        & mpad[rows, safe]
        & (mp[rows, safe] <= qp[..., None])
        & (ms[rows, safe] == qs[..., None])
    )
    expected_clean = np.where(valid, ids, -1)
    ordered_ids = np.sort(np.where(valid, ids, n), axis=-1)
    expected_compact = np.where(ordered_ids < n, ordered_ids, -1).reshape(b * q, width)
    arrays = [mx.array(a) for a in (ids, qp, mp, qs, ms, qpad, mpad)]
    clean, (compact, lengths) = mx.compile(validate_compact)(*arrays)
    mx.eval(clean, compact, lengths)
    np.testing.assert_array_equal(np.asarray(clean), expected_clean)
    np.testing.assert_array_equal(np.asarray(compact), expected_compact)
    np.testing.assert_array_equal(np.asarray(lengths), valid.sum(-1).reshape(-1))


def test_empty_memory_and_large_document_ids():
    ids = mx.array([[[0, 1, -1], [1, 0, 5]]], mx.int32)
    qp = mx.array([[1, 3]], mx.int32)
    docs = mx.array([[2**34, 2**34 + 1]], mx.int64)
    clean, (compact, lengths) = validate_compact(
        ids,
        qp,
        mx.zeros((1, 0), mx.int32),
        docs,
        mx.zeros((1, 0), mx.int64),
        mx.ones((1, 2), mx.bool_),
        mx.ones((1, 0), mx.bool_),
    )
    assert (
        bool(mx.all(clean == -1))
        and bool(mx.all(compact == -1))
        and bool(mx.all(lengths == 0))
    )


def test_full_model_forward_and_gradient_contract(monkeypatch):
    import model.attention as attention
    from _v41_common import build
    from test_ced_recurrent import config, inputs
    from trainer.utils import convert_model_dtype
    from mlx.utils import tree_flatten

    model = build(config(n_heads=16, head_dim=64))
    convert_model_dtype(model, "bfloat16")
    x = inputs(16)
    seg = mx.array([[0] * 5 + [1] * 11])
    params = model.trainable_parameters()
    values = []
    for flag in (False, False, True):
        monkeypatch.setattr(attention, "_RECURRENT_METADATA_FUSED", flag)

        def objective(p):
            model.update(p)
            out = model(x, labels=x, segment_ids=seg, need_logits=True)
            return out.loss, (out.logits, out.moe_loads, out.moe_qb_margins)

        val, grad = mx.compile(mx.value_and_grad(objective))(params)
        model.update(params)
        mx.eval(val, grad)
        values.append((val, dict(tree_flatten(grad))))
    assert bool(mx.array_equal(values[0][0][0], values[2][0][0]))
    for a, b in zip(values[0][0][1], values[2][0][1]):
        assert bool(mx.all((a == b) | (mx.isnan(a) & mx.isnan(b))))
    norms = []
    for path, gradient in values[0][1].items():
        # Atomic floating-point VJPs can vary even across unchanged A/A runs;
        # control metadata and all forward outputs above remain exact.
        assert bool(mx.all(mx.isfinite(values[2][1][path])))
        a = gradient.astype(mx.float32)
        aa = values[1][1][path].astype(mx.float32)
        b = values[2][1][path].astype(mx.float32)
        norms.append(
            mx.stack([mx.sum(a * a), mx.sum((a - aa) ** 2), mx.sum((a - b) ** 2)])
        )
    total = mx.sum(mx.stack(norms), axis=0)
    relative = mx.sqrt(total[2] / mx.maximum(total[0], 1e-12))
    reference_noise = mx.sqrt(total[1] / mx.maximum(total[0], 1e-12))
    assert float(relative) <= max(4 * float(reference_noise), 0.015)
