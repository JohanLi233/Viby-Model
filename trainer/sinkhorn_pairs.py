"""Fuse adjacent column/row optimizer Sinkhorn normalization steps.

The column square-sum keeps its existing tiled FP32 reduction. Its typed
denominator is materialized as a small [1,C] array. A single full-table kernel
then rounds the column division to x.dtype, caches those values, computes the
next row norm with the existing 32-lane square-sum order, and emits the row
division. The intermediate column-normalized table is never written/read.

This only removes materialization boundaries: every iteration's denominator
cast, division rounding and FP32 reduction order is retained. These functions
are optimizer bookkeeping, not differentiable model operators.
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

    column = x / (mx.sqrt(square_sum(x, 0)) + eps).astype(x.dtype)
    return column / (mx.sqrt(square_sum(column, 1)) + eps).astype(x.dtype)


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="optimizer_sinkhorn_column_row_pair",
        input_names=["x", "column_denominator", "epsilon"],
        output_names=["out"],
        source=r"""
        uint row=thread_position_in_grid.x/32;
        uint lane=thread_position_in_threadgroup.x%32;
        if (row>=R) return;
        T values[(C+31)/32];
        float acc=0.0f;
        for (uint c=lane;c<C;c+=32) {
            // The completed even iteration is stored in T before its square.
            T value=T(x[(size_t)row*C+c]/column_denominator[c]);
            values[c/32]=value;
            float v=float(value);
            acc+=v*v;
        }
        acc=simd_sum(acc);
        T denominator=T(metal::precise::sqrt(acc)+epsilon[0]);
        for (uint c=lane;c<C;c+=32)
            out[(size_t)row*C+c]=T(values[c/32]/denominator);
        """,
    )


def column_then_row(x, eps=1e-20):
    """Perform one even column iteration and its following odd row iteration."""
    if not enabled_for(x):
        return _reference(x, eps)
    from .fast_norm import square_sum

    # This small typed input also prevents broadcasting sqrt across the table
    # in a surrounding compiled pointwise division graph.
    column_denominator = (mx.sqrt(square_sum(x, 0)) + eps).astype(x.dtype)
    rows, cols = x.shape
    return _kernel()(
        inputs=[x, column_denominator, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("C", cols), ("R", rows)],
        grid=(rows * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def first_row_from_norm(x, row_norm, zero_mask, eps=1e-20):
    """Reuse the exact pre-mask row norm for Sinkhorn's first normalization.

    ``row_norm`` must be sqrt(fast_norm.square_sum(x, 1)); ``zero_mask`` is the
    original rho <= tau * mean(rho) decision. Unmasked rows already have their norm;
    masked rows are the exact all-zero rows. This composed pointwise graph saves a
    redundant full-table square reduction without changing zero/epsilon behavior.
    """
    denominator = (mx.where(zero_mask, mx.zeros_like(row_norm), row_norm) + eps).astype(
        x.dtype
    )
    masked = mx.where(zero_mask, mx.zeros_like(x), x)
    return masked / denominator


@lru_cache(None)
def prewarm_column_then_row(shape, eps=1e-20, dtype=mx.bfloat16):
    """Materialize the pair kernel eagerly before optimizer compilation."""
    x = mx.ones(tuple(shape), dtype)
    if enabled_for(x):
        mx.eval(column_then_row(x, eps))
