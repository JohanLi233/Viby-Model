"""Exact promotion, product rounding, layout and VJP gates for decode RoPE."""

import mlx.core as mx
import numpy as np
import pytest

from model import rope
from model.kernels import rope_decode as rd

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize(
    "dtype,batch,heads,dim,rot,inverse,broadcast",
    [
        (mx.float32, 1, 0, 128, 32, False, False),
        (mx.bfloat16, 3, 4, 128, 32, True, False),
        (mx.float16, 8, 16, 64, 16, False, True),
        (mx.bfloat16, 1, 4, 64, 64, True, True),
        (mx.float32, 2, 0, 65, 32, True, False),
    ],
)
def test_partial_rope_matches_eager_and_vjp(
    monkeypatch, dtype, batch, heads, dim, rot, inverse, broadcast
):
    mx.random.seed(314159)
    shape = (batch, 1, heads, dim) if heads else (batch, 1, dim)
    x = mx.random.normal((*shape[:-1], dim * 2)).astype(dtype)[..., ::2]
    cb = 1 if broadcast else batch
    freq_shape = (cb, 1, 1, rot // 2) if heads else (cb, 1, rot // 2)
    cos = mx.random.normal((rot // 2, cb)).T.reshape(freq_shape)
    sin = mx.random.normal((rot // 2, cb)).T.reshape(freq_shape)
    cot = mx.random.normal(shape)
    values = []
    for enabled in (False, True):
        monkeypatch.setattr(rd, "_ENABLED", enabled)
        values.append(
            mx.vjp(
                lambda a, c, s: rope.rope_partial(a, c, s, rot, inverse),
                [x, cos, sin],
                [cot],
            )
        )
    mx.eval(values)
    assert rd.enabled_for(x, cos, sin, rot)
    assert values[1][0][0].dtype == mx.float32
    np.testing.assert_array_equal(np.asarray(values[1][0]), np.asarray(values[0][0]))
    for got, ref in zip(values[1][1], values[0][1]):
        np.testing.assert_array_equal(
            np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32))
        )


def test_partial_rope_guard_retains_other_paths(monkeypatch):
    monkeypatch.setattr(rd, "_ENABLED", True)
    x = mx.ones((1, 1, 4, 64), mx.bfloat16)
    freq = mx.ones((1, 1, 1, 16), mx.float32)
    assert rd.enabled_for(x, freq, freq, 32)
    assert not rd.enabled_for(mx.ones((1, 2, 4, 64), mx.bfloat16), freq, freq, 32)
    assert not rd.enabled_for(x, freq.astype(mx.bfloat16), freq, 32)
    assert not rd.enabled_for(x, freq, freq, 31)
