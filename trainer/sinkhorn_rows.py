"""Fuse one row-normalization step of the optimizer's Sinkhorn iteration.

One 32-lane SIMD group owns each row. Every lane caches its original input
elements while accumulating the same FP32 square-sum chain as
``fast_norm.square_sum(x, 1)``. The row is read only once; the denominator is
sqrt(sum) + epsilon in FP32, cast to x.dtype before the original typed division.

This is optimizer bookkeeping, not a differentiable model operator. The caller
controls whether the fusion is selected; unsupported shapes retain the current
square_sum reference. No optimizer recipe, iteration count or state is changed.
"""

from functools import lru_cache

import mlx.core as mx


def enabled_for(x):
    return (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and x.ndim == 2
        and x.size > 0
        and 0 < x.shape[1] <= 4096
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
    )


def _reference(x, eps):
    from .fast_norm import square_sum

    return x / (mx.sqrt(square_sum(x, 1)) + eps).astype(x.dtype)


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="optimizer_sinkhorn_row_normalize",
        input_names=["x", "epsilon"],
        output_names=["out"],
        source=r"""
        uint row=thread_position_in_grid.x/32;
        uint lane=thread_position_in_threadgroup.x%32;
        if (row>=R) return;
        T values[(C+31)/32];
        float acc=0.0f;
        for (uint c=lane;c<C;c+=32) {
            T value=x[(size_t)row*C+c];
            values[c/32]=value;
            float v=float(value);
            // Same lane traversal and multiply/add expression as square_sum.
            acc+=v*v;
        }
        acc=simd_sum(acc);
        T denominator=T(metal::precise::sqrt(acc)+epsilon[0]);
        for (uint c=lane;c<C;c+=32)
            out[(size_t)row*C+c]=T(values[c/32]/denominator);
        """,
    )


def row_normalize(x, eps=1e-20):
    """Return x / cast_x(sqrt(square_sum_FP32(x, rows)) + eps)."""
    if not enabled_for(x):
        return _reference(x, eps)
    rows, cols = x.shape
    return _kernel()(
        inputs=[x, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("C", cols), ("R", rows)],
        grid=(rows * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@lru_cache(None)
def prewarm_row_normalize(shape, eps=1e-20, dtype=mx.bfloat16):
    """Eager JIT materialization before a surrounding optimizer graph is traced."""
    x = mx.ones(tuple(shape), dtype)
    if enabled_for(x):
        mx.eval(row_normalize(x, eps))
