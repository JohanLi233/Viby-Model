"""Window-only gradient against the live attention VJP with compressed keys."""

import mlx.core as mx
import numpy as np
import pytest
from model.kernels import sparse_attention as sa
from model.kernels.window_attention_backward import window_attention_backward
from test_sparse_attention_kernel import _inputs
from test_sparse_attention_key_owned import _assert_pool_close

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "t,n,d,w", [(1, 1, 64, 128), (35, 17, 128, 31), (65, 0, 64, 16)]
)
def test_key_tile_window_backward(dtype, t, n, d, w):
    q, window, c, visible, seg, pad, sinks, _, scale = _inputs(2, t, n, d, w, 41, dtype)
    indices, lengths = sa.compact_visible(visible)
    out, lse = sa._operation(w, scale, 16)(
        q, window, c, indices, lengths, seg, pad, sinks
    )
    g = mx.random.normal(q.shape).astype(dtype)
    delta, _ = sa._fused_delta_kernel()(
        inputs=[g, out, sinks, lse],
        template=[("D", d), ("H", 16)],
        grid=(d, 2 * t, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(2, t, 16)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    got = window_attention_backward(q, window, g, lse, delta, seg, pad, w, scale)
    _, (ref,) = mx.vjp(
        lambda wk: sa.indexed_attention(q, wk, c, visible, seg, pad, sinks, w, scale),
        [window],
        [g.transpose(0, 2, 1, 3)],
    )
    mx.eval(got, ref)
    _assert_pool_close(
        np.asarray(got.astype(mx.float32)),
        np.asarray(ref.astype(mx.float32)),
        dtype,
        "dw",
    )
    empty = window_attention_backward(
        q, window, g, lse, delta, seg, mx.zeros_like(pad), w, scale
    )
    np.testing.assert_array_equal(np.asarray(empty.astype(mx.float32)), 0)
