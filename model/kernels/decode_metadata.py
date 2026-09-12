"""Fuse decode index postprocessing and separate-pool KV gathering.

The original argpartition determines the fixed-k set. Only its position sort,
eligibility/offset encoding and subsequent pool gathers are fused here.
Up to 256 selected positions sort in one SIMD group's registers by default;
VIBY_DECODE_SIMD_POST=0 restores the original threadgroup bitonic sort.
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_DECODE_METADATA", "1") != "0"
_PREFILL_SELECT = os.environ.get("VIBY_PREFILL_SELECT", "1") != "0"
_SIMD_POST = os.environ.get("VIBY_DECODE_SIMD_POST", "1") != "0"


@lru_cache(None)
def _simd_post_kernel():
    """Sort up to 256 selected positions entirely in one SIMD group's registers.

    Each lane holds CAP/32 positions. Small bitonic exchanges use shuffle_xor;
    exchanges across 32-position blocks use registers in the same lane. All
    lanes participate, including sentinel lanes when fewer than 32 ids exist.
    """
    return mx.fast.metal_kernel(
        name="decode_fixed_k_simd_postprocess",
        input_names=["ids", "scores", "reach", "dims"],
        output_names=["out"],
        source=r"""
        uint lane=thread_position_in_threadgroup.x;
        uint q=thread_position_in_grid.y;
        uint n=dims[0], k=dims[1], offset=dims[2];
        uint values[ITEMS];
        for (uint r=0;r<ITEMS;++r) {
            uint i=r*32+lane;
            values[r]=i<k ? uint(ids[(size_t)q*k+i]) : 0xffffffffu;
        }
        for (uint size=2;size<=CAP;size*=2) {
            for (uint stride=size/2;stride>0;stride/=2) {
                uint next[ITEMS];
                for (uint r=0;r<ITEMS;++r) {
                    uint i=r*32+lane;
                    uint a=values[r];
                    uint b=stride<32 ? simd_shuffle_xor(a, ushort(stride))
                                     : values[r^(stride/32)];
                    bool take_min=((i&size)==0)==((i&stride)==0);
                    next[r]=take_min ? min(a,b) : max(a,b);
                }
                for (uint r=0;r<ITEMS;++r) values[r]=next[r];
            }
        }
        for (uint r=0;r<ITEMS;++r) {
            uint i=r*32+lane;
            if (i<k) {
                uint j=values[r];
                bool yes=reach[(size_t)q*n+j] && scores[(size_t)q*n+j]>-1e30f;
                out[(size_t)q*k+i]=yes ? int(j+offset) : -1;
            }
        }
        """,
    )


@lru_cache(None)
def _kernels():
    post = mx.fast.metal_kernel(
        name="decode_fixed_k_postprocess",
        input_names=["ids", "scores", "reach", "dims"],
        output_names=["out"],
        source=r"""
        uint tid=thread_position_in_grid.x, q=thread_position_in_grid.y;
        uint n=dims[0], k=dims[1], offset=dims[2];
        threadgroup uint sorted[CAP];
        for (uint i=tid;i<CAP;i+=128) sorted[i]=i<k ? uint(ids[(size_t)q*k+i]) : 0xffffffffu;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint size=2;size<=CAP;size*=2) for (uint stride=size/2;stride>0;stride/=2) {
            for (uint i=tid;i<CAP;i+=128) {
                uint j=i^stride;
                if (j>i) {
                    uint a=sorted[i], b=sorted[j];
                    bool up=(i&size)==0;
                    sorted[i]=up ? min(a,b) : max(a,b);
                    sorted[j]=up ? max(a,b) : min(a,b);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i=tid;i<k;i+=128) {
            uint j=sorted[i];
            bool yes=reach[(size_t)q*n+j] && scores[(size_t)q*n+j]>-1e30f;
            out[(size_t)q*k+i]=yes ? int(j+offset) : -1;
        }
        """,
    )
    gather = mx.fast.metal_kernel(
        name="decode_gather_kv_pools",
        input_names=["window", "compressed", "win_idx", "comp_idx", "dims"],
        output_names=["kv", "valid"],
        source=r"""
        uint tid=thread_position_in_grid.x, slot=thread_position_in_grid.y, b=thread_position_in_grid.z;
        uint nw=dims[0], nc=dims[1], pool_n=dims[2], offset=dims[3];
        int raw=slot<nw ? win_idx[(size_t)b*nw+slot] : comp_idx[(size_t)b*nc+slot-nw];
        bool yes=raw>=0;
        uint pos=yes ? (slot<nw ? uint(raw) : uint(raw)-offset) : 0u;
        if (tid==0) valid[(size_t)b*(nw+nc)+slot]=yes;
        for (uint d=tid;d<D;d+=128) {
            OutT v=OutT(0);
            if (yes) v=slot<nw ? OutT(window[((size_t)b*offset+pos)*D+d])
                                : OutT(compressed[((size_t)b*pool_n+pos)*D+d]);
            kv[((size_t)b*(nw+nc)+slot)*D+d]=v;
        }
        """,
    )
    return post, gather


def topk_indices(scores, reach, k, offset):
    b, t, n = scores.shape
    if n == 0 or k == 0:
        return mx.zeros((b, t, 0), mx.int32)
    k = min(k, n)
    adjusted = mx.stop_gradient(scores) - mx.arange(n, dtype=scores.dtype) * 1e-7
    # Intentionally retain the original fixed-k argpartition, including ties.
    ids = mx.argpartition(-adjusted, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
    return _postprocess_indices(ids, scores, reach, k, offset)


def _postprocess_indices(ids, scores, reach, k, offset):
    b, t, n = scores.shape
    capacity = 1 << (k - 1).bit_length()
    use_simd = _SIMD_POST and capacity <= 256
    kernel = _simd_post_kernel() if use_simd else _kernels()[0]
    template = [("CAP", capacity)]
    if use_simd:
        template.append(("ITEMS", max(1, capacity // 32)))
    threads = 32 if use_simd else 128
    (out,) = kernel(
        inputs=[
            ids,
            scores,
            mx.broadcast_to(reach, scores.shape),
            mx.array([n, k, offset], mx.uint32),
        ],
        template=template,
        grid=(threads, b * t, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(b, t, k)],
        output_dtypes=[mx.int32],
    )
    return out


def gather_pools(window, compressed, win_idx, comp_idx):
    b, W, d = window.shape
    nw, nc = win_idx.shape[-1], comp_idx.shape[-1]
    if compressed is None:
        compressed = mx.zeros((b, 0, d), window.dtype)
    dtype = mx.result_type(window, compressed)
    return _kernels()[1](
        inputs=[
            window,
            compressed,
            win_idx,
            comp_idx,
            mx.array([nw, nc, compressed.shape[1], W], mx.uint32),
        ],
        template=[("D", d), ("OutT", dtype)],
        grid=(128, nw + nc, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, 1, nw + nc, d), (b, 1, nw + nc)],
        output_dtypes=[dtype, mx.bool_],
    )


@lru_cache(None)
def _expert_view_kernel():
    return mx.fast.metal_kernel(
        name="decode_expert_contributions",
        input_names=["y", "weights", "experts", "dims"],
        output_names=["out"],
        source=r"""
        uint tid=thread_position_in_grid.x, route=thread_position_in_grid.y;
        uint token=route/K, expert=uint(experts[route]), M=dims[0];
        for (uint d=tid;d<D;d+=128)
            out[((size_t)expert*M+token)*D+d]=T(y[(size_t)route*D+d]*T(weights[route]));
        """,
    )


@lru_cache(None)
def _expert_view_op(n_experts, routes):
    kernel = _expert_view_kernel()

    @mx.custom_function
    def op(y, weights, experts):
        m, d = weights.size // routes, y.shape[-1]
        return kernel(
            inputs=[y, weights, experts, mx.array([m], mx.uint32)],
            template=[("T", y.dtype), ("K", routes), ("D", d)],
            grid=(128, m * routes, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_experts, m, d)],
            output_dtypes=[y.dtype],
            init_value=0,
        )[0]

    @op.vjp
    def vjp(primals, cotangent, output):
        y, weights, experts = primals
        token = mx.arange(weights.size) // routes
        g = cotangent[experts, token].astype(y.dtype)
        dy = (g * weights.astype(y.dtype)[:, None]).reshape(y.shape)
        dw = mx.sum((g * y.reshape(g.shape)).astype(weights.dtype), axis=-1)
        return dy, dw, mx.zeros_like(experts)

    return op


def combine_selected_experts(y, weights, experts, n_experts, routes):
    """Retain the original [E,M,D] reduction order for small-M inference.

    Unselected experts have zero contributions and their GEMMs are skipped.
    Selected expert/token positions are unique: stores need no float atomic.
    The small scratch preserves the original low-precision sum's grouping.
    """
    return mx.sum(_expert_view_op(n_experts, routes)(y, weights, experts), axis=0)
