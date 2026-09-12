"""Validate and compact fixed-budget evidence indices in one integer dispatch.

The selected set and its order are unchanged. All inputs/outputs are detached
control metadata; no gradients or floating-point attention math are replaced.
"""

from functools import lru_cache
import mlx.core as mx


def supported(
    selected,
    query_positions,
    memory_positions,
    query_docs,
    memory_docs,
    query_pad,
    memory_pad,
):
    return (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and selected.dtype == mx.int32
        and selected.ndim == 3
        and 0 < selected.shape[-1] <= 256
        and selected.size < 2**31
        and all(
            mx.issubdtype(x.dtype, mx.integer)
            for x in (query_positions, memory_positions, query_docs, memory_docs)
        )
        and query_positions.dtype == memory_positions.dtype
        and query_docs.dtype == memory_docs.dtype
        and query_pad.dtype == memory_pad.dtype == mx.bool_
    )


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="ced_validate_compact_routes",
        input_names=["selected", "qp", "mp", "qs", "ms", "qpad", "mpad", "dims"],
        output_names=["clean", "compact", "lengths"],
        source=r"""
        uint lane=thread_position_in_threadgroup.x;
        uint row=thread_position_in_grid.y;
        uint Q=dims[0], N=dims[1], K=dims[2], b=row/Q;
        threadgroup int values[CAP];
        for (uint col=lane;col<CAP;col+=32) values[col]=2147483647;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint used=0;
        for (uint start=0; start<K; start+=32) {
            uint col=start+lane;
            int key=col<K ? selected[(size_t)row*K+col] : -1;
            bool good=col<K && key>=0 && uint(key)<N && qpad[row];
            if (good) {
                size_t at=(size_t)b*N+uint(key);
                good=mpad[at] && mp[at]<=qp[row] && ms[at]==qs[row];
            }
            if (col<K) clean[(size_t)row*K+col]=good ? key : -1;
            uint prefix=good ? 1u : 0u;
            for (uint offset=1;offset<32;offset*=2) {
                uint preceding=simd_shuffle_up(prefix,ushort(offset));
                if (lane>=offset) prefix+=preceding;
            }
            uint count=simd_sum(good ? 1u : 0u);
            if (good) values[used+prefix-1]=key;
            used+=count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool unordered=false;
        for (uint col=lane;col+1<used;col+=32)
            unordered=unordered || values[col]>values[col+1];
        // Model-generated routes are already sorted. Preserve the reference's
        // sorted compact order for arbitrary direct callers too, without a
        // second kernel on the common canonical route.
        if (simd_sum(unordered ? 1u : 0u)>0) {
            for (uint width=2;width<=CAP;width*=2) {
                for (uint gap=width/2;gap>0;gap/=2) {
                    for (uint col=lane;col<CAP;col+=32) {
                        uint other=col^gap;
                        if (other>col) {
                            int a=values[col], bvalue=values[other];
                            bool ascending=(col&width)==0;
                            values[col]=ascending ? min(a,bvalue) : max(a,bvalue);
                            values[other]=ascending ? max(a,bvalue) : min(a,bvalue);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        for (uint col=lane;col<K;col+=32)
            compact[(size_t)row*K+col]=col<used ? values[col] : -1;
        if (lane==0) lengths[row]=int(used);
        """,
    )


def validate_compact(
    selected,
    query_positions,
    memory_positions,
    query_docs,
    memory_docs,
    query_pad,
    memory_pad,
):
    """Return (same-shaped cleaned ids, (compact ids[B*Q,K], lengths[B*Q]))."""
    if not supported(
        selected,
        query_positions,
        memory_positions,
        query_docs,
        memory_docs,
        query_pad,
        memory_pad,
    ):
        raise ValueError("unsupported recurrent metadata shape/device/dtype")
    b, q, k = selected.shape
    n = memory_positions.shape[1]
    if (
        query_positions.shape != (b, q)
        or query_docs.shape != (b, q)
        or query_pad.shape != (b, q)
        or memory_positions.shape[0] != b
        or memory_docs.shape != (b, n)
        or memory_pad.shape != (b, n)
    ):
        raise ValueError("recurrent metadata must match B/Q/N")
    if q == 0:
        return selected, (selected.reshape(b * q, k), mx.zeros((b * q,), mx.int32))
    arrays = (
        selected,
        query_positions,
        memory_positions,
        query_docs,
        memory_docs,
        query_pad,
        memory_pad,
    )
    clean, compact, lengths = _kernel()(
        inputs=[*[mx.stop_gradient(x) for x in arrays], mx.array([q, n, k], mx.uint32)],
        template=[("CAP", 1 << (k - 1).bit_length())],
        grid=(32, b * q, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[selected.shape, (b * q, k), (b * q,)],
        output_dtypes=[mx.int32] * 3,
    )
    return mx.stop_gradient(clean), (
        mx.stop_gradient(compact),
        mx.stop_gradient(lengths),
    )
