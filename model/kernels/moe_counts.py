"""Per-sequence route counts without a [B*T,E] incidence matrix.

Count occurrences, not distinct token/expert pairs. Both sequence auxiliary
loss and noaux_tc can consume the same [B,E] counts. Probabilities and the
argpartition-selected routes are inputs to this bookkeeping, never modified.
"""

import os
import operator
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_MOE_COMPACT_AUX", "0") != "0"
_THREADS = 128
_ROUTES_PER_TILE = 256
_MAX_EXACT_FLOAT_COUNT = 1 << 24


def enabled_for(indices):
    # Beyond this bound the legacy FP32 scatter need not count every +1.
    # Retain that path rather than silently changing noaux_tc statistics.
    return _ENABLED and indices.size <= _MAX_EXACT_FLOAT_COUNT


@lru_cache(None)
def _count_kernel():
    return mx.fast.metal_kernel(
        name="moe_sequence_route_counts",
        input_names=["indices"],
        output_names=["partial"],
        source=r"""
        uint tid = thread_position_in_threadgroup.x;
        uint tile = threadgroup_position_in_grid.x;
        uint batch = threadgroup_position_in_grid.y;
        threadgroup atomic_uint counts[E];
        for (uint e = tid; e < E; e += NT)
            atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RT; i += NT) {
            uint route = tile * RT + i;
            if (route < R) {
                uint e = uint(indices[(size_t)batch * R + route]);
                atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < E; e += NT)
            partial[((size_t)batch * TILES + tile) * E + e] =
                atomic_load_explicit(&counts[e], memory_order_relaxed);
        """,
    )


def sequence_route_counts(indices, batch_size, seq_len, n_experts):
    """Return FP32 counts[B,E]; indices is the existing router output [B*T,K].

    Integer TG histograms have at most 256 increments each, so no float atomic
    or [B,T,E] temporary is needed. GPU lengths are never read on the host.
    E>512 and CPU use a compact native scatter; no custom-kernel dependency.
    Expert ids must be in [0,E), as guaranteed by the router. The integration
    falls back to the legacy path above 2**24 routes (exact FP32 integer range).
    """
    b, t, e = map(operator.index, (batch_size, seq_len, n_experts))
    if b < 0 or t < 0 or e <= 0 or indices.ndim != 2:
        raise ValueError("expected nonnegative B/T, positive E, and indices[B*T,K]")
    if indices.shape[0] != b * t or indices.shape[1] <= 0:
        raise ValueError("indices shape does not match B*T and positive K")
    if indices.dtype not in (mx.int32, mx.uint32):
        raise ValueError("expert indices must be int32 or uint32")
    if b * e > 2**31 - 1 or t * indices.shape[1] > 2**32 - 1:
        raise ValueError("route count dimensions exceed supported index range")
    if indices.size > _MAX_EXACT_FLOAT_COUNT:
        raise ValueError("FP32 occurrence counts require at most 2**24 routes")
    k = indices.shape[1]
    if b == 0 or t == 0:
        return mx.zeros((b, e), dtype=mx.float32)
    ids = mx.stop_gradient(indices)
    if mx.default_device() == mx.gpu and mx.metal.is_available() and e <= 512:
        tiles = (t * k + _ROUTES_PER_TILE - 1) // _ROUTES_PER_TILE
        (partial,) = _count_kernel()(
            inputs=[ids],
            template=[
                ("E", e),
                ("NT", _THREADS),
                ("RT", _ROUTES_PER_TILE),
                ("R", t * k),
                ("TILES", tiles),
            ],
            grid=(tiles * _THREADS, b, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(b, tiles, e)],
            output_dtypes=[mx.uint32],
        )
        # Sum counts as integers before the single FP32 conversion.
        return mx.stop_gradient(mx.sum(partial, axis=1).astype(mx.float32))
    offsets = (mx.arange(b, dtype=mx.int32) * e)[:, None]
    flat = (ids.astype(mx.int32).reshape(b, t * k) + offsets).reshape(-1)
    counts = (
        mx.zeros((b * e,), dtype=mx.float32)
        .at[flat]
        .add(mx.ones((indices.size,), dtype=mx.float32))
    )
    return mx.stop_gradient(counts.reshape(b, e))


def masked_enabled_for(indices, pad_mask):
    """Boolean occurrence masks preserve exact legacy FP32 +1 counts."""
    return pad_mask.dtype == mx.bool_ and indices.size <= _MAX_EXACT_FLOAT_COUNT


@lru_cache(None)
def _masked_count_kernel():
    return mx.fast.metal_kernel(
        name="moe_masked_sequence_route_counts",
        input_names=["indices", "valid"],
        output_names=["partial"],
        source=r"""
        uint tid = thread_position_in_threadgroup.x;
        uint tile = threadgroup_position_in_grid.x;
        uint batch = threadgroup_position_in_grid.y;
        threadgroup atomic_uint counts[E];
        for (uint e = tid; e < E; e += NT)
            atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RT; i += NT) {
            uint route = tile * RT + i;
            if (route < R && valid[(size_t)batch * T + route / K]) {
                uint e = uint(indices[(size_t)batch * R + route]);
                atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < E; e += NT)
            partial[((size_t)batch * TILES + tile) * E + e] =
                atomic_load_explicit(&counts[e], memory_order_relaxed);
        """,
    )


def masked_sequence_route_counts(indices, pad_mask, batch_size, seq_len, n_experts):
    """Exact valid-token occurrences [B,E], without K scatter launches.

    ``pad_mask`` is Boolean [B,T]; masked tokens contribute zero to every
    selected expert. The result is detached bookkeeping, just like the legacy
    FP32 scatters. Nonboolean masks intentionally remain on the caller's legacy
    path because they may encode fractional occurrence weights.
    """
    b, t, e = map(operator.index, (batch_size, seq_len, n_experts))
    if b < 0 or t < 0 or e <= 0 or indices.ndim != 2:
        raise ValueError("expected nonnegative B/T, positive E, and indices[B*T,K]")
    if indices.shape[0] != b * t or indices.shape[1] <= 0:
        raise ValueError("indices shape does not match B*T and positive K")
    if indices.dtype not in (mx.int32, mx.uint32):
        raise ValueError("expert indices must be int32 or uint32")
    if pad_mask.shape != (b, t) or pad_mask.dtype != mx.bool_:
        raise ValueError("pad_mask must be Boolean [B,T]")
    if b * e > 2**31 - 1 or t * indices.shape[1] > 2**32 - 1:
        raise ValueError("route count dimensions exceed supported index range")
    if indices.size > _MAX_EXACT_FLOAT_COUNT:
        raise ValueError("FP32 occurrence counts require at most 2**24 routes")
    if b == 0 or t == 0:
        return mx.zeros((b, e), dtype=mx.float32)
    k = indices.shape[1]
    ids, valid = mx.stop_gradient(indices), mx.stop_gradient(pad_mask)
    if mx.default_device() == mx.gpu and mx.metal.is_available() and e <= 512:
        tiles = (t * k + _ROUTES_PER_TILE - 1) // _ROUTES_PER_TILE
        (partial,) = _masked_count_kernel()(
            inputs=[ids, valid],
            template=[
                ("E", e),
                ("NT", _THREADS),
                ("RT", _ROUTES_PER_TILE),
                ("R", t * k),
                ("T", t),
                ("K", k),
                ("TILES", tiles),
            ],
            grid=(tiles * _THREADS, b, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(b, tiles, e)],
            output_dtypes=[mx.uint32],
        )
        return mx.stop_gradient(mx.sum(partial, axis=1).astype(mx.float32))
    offsets = (mx.arange(b, dtype=mx.int32) * e)[:, None]
    flat = (ids.astype(mx.int32).reshape(b, t * k) + offsets).reshape(-1)
    weights = mx.repeat(valid.reshape(-1).astype(mx.float32), k)
    counts = mx.zeros((b * e,), dtype=mx.float32).at[flat].add(weights)
    return mx.stop_gradient(counts.reshape(b, e))
