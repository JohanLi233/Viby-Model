"""FP32 square reductions without full-size cast/square temporaries.

These kernels are optimizer bookkeeping, not differentiable model operators.
Reduction order may differ from native MLX; division/casts stay with callers.
"""
from functools import lru_cache

import mlx.core as mx


@lru_cache(None)
def _kernel(axis):
    if axis == 1:
        source = r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        float acc = 0.0f;
        for (uint c = lane; c < C; c += 32) {
            float v = float(x[(size_t)row * C + c]);
            acc += v * v;
        }
        acc = simd_sum(acc);
        if (lane == 0) out[row] = acc;
        """
    else:
        source = r"""
        uint c = thread_position_in_grid.x;
        uint tile = threadgroup_position_in_grid.y;
        if (c >= C) return;
        float acc = 0.0f;
        uint end = min((tile + 1) * 256u, uint(R));
        for (uint r = tile * 256; r < end; ++r) {
            float v = float(x[(size_t)r * C + c]);
            acc += v * v;
        }
        out[(size_t)tile * C + c] = acc;
        """
    return mx.fast.metal_kernel(
        name=f"optimizer_square_sum_axis{axis}", input_names=["x"],
        output_names=["out"], source=source,
    )


def square_sum(x, axis):
    """Return FP32 [R,1] or [1,C] sum of squares for a 2D array."""
    if (x.ndim != 2 or axis not in (0, 1) or x.size == 0
            or mx.default_device() != mx.gpu
            or x.dtype not in (mx.float32, mx.float16, mx.bfloat16)):
        y = x.astype(mx.float32)
        return mx.sum(y * y, axis=axis, keepdims=True)
    r, c = x.shape
    tiles = (r + 255) // 256
    shape = (r, 1) if axis == 1 else (tiles, c)
    grid = (r * 32, 1, 1) if axis == 1 else (((c + 127) // 128) * 128, tiles, 1)
    (out,) = _kernel(axis)(
        inputs=[x], template=[("R", r), ("C", c)], grid=grid,
        threadgroup=(32 if axis == 1 else 128, 1, 1),
        output_shapes=[shape], output_dtypes=[mx.float32],
    )
    return out if axis == 1 else mx.sum(out, axis=0, keepdims=True)


@lru_cache(None)
def _flat_kernel():
    return mx.fast.metal_kernel(
        name="optimizer_gradient_square_sum", input_names=["x"], output_names=["out"],
        source=r"""
        uint tile = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        threadgroup float partial[8];
        float acc = 0.0f;
        size_t end = min(size_t(tile + 1) * 4096, size_t(N));
        for (size_t i = size_t(tile) * 4096 + tid; i < end; i += 256) {
            float v = float(x[i]);
            acc += v * v;
        }
        acc = simd_sum(acc);
        if (tid % 32 == 0) partial[tid / 32] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float sum = simd_sum(tid < 8 ? partial[tid] : 0.0f);
            if (tid == 0) out[tile] = sum;
        }
        """,
    )


def gradient_square_sum(x):
    """One read of a gradient, FP32 accumulation, scalar FP32 result."""
    if x.size < 4096 or mx.default_device() != mx.gpu:
        return mx.sum(mx.square(x.astype(mx.float32)))
    tiles = (x.size + 4095) // 4096
    (partial,) = _flat_kernel()(
        inputs=[x], template=[("N", x.size)], grid=(tiles * 256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(tiles,)], output_dtypes=[mx.float32],
    )
    return mx.sum(partial)
