"""Fuse decode index postprocessing and separate-pool KV gathering.

The original argpartition determines the fixed-k set. Only its position sort,
eligibility/offset encoding and subsequent pool gathers are fused here.
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_DECODE_METADATA", "1") != "0"
_PREFILL_SELECT = os.environ.get("VIBY_PREFILL_SELECT", "1") != "0"


@lru_cache(None)
def _kernels():
    post = mx.fast.metal_kernel(
        name="decode_fixed_k_postprocess", input_names=["ids", "scores", "reach", "dims"],
        output_names=["out"], source=r"""
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
        name="decode_gather_kv_pools", input_names=["window", "compressed", "win_idx", "comp_idx", "dims"],
        output_names=["kv", "valid"], source=r"""
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
    (out,) = _kernels()[0](
        inputs=[ids, scores, mx.broadcast_to(reach, scores.shape), mx.array([n, k, offset], mx.uint32)],
        template=[("CAP", 1 << (k - 1).bit_length())],
        grid=(128, b * t, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b, t, k)], output_dtypes=[mx.int32],
    )
    return out


def gather_pools(window, compressed, win_idx, comp_idx):
    b, W, d = window.shape
    nw, nc = win_idx.shape[-1], comp_idx.shape[-1]
    if compressed is None:
        compressed = mx.zeros((b, 0, d), window.dtype)
    dtype = mx.result_type(window, compressed)
    return _kernels()[1](
        inputs=[window, compressed, win_idx, comp_idx, mx.array([nw, nc, compressed.shape[1], W], mx.uint32)],
        template=[("D", d), ("OutT", dtype)],
        grid=(128, nw + nc, b), threadgroup=(128, 1, 1),
        output_shapes=[(b, 1, nw + nc, d), (b, 1, nw + nc)], output_dtypes=[dtype, mx.bool_],
    )


@lru_cache(None)
def _expert_view_kernel():
    return mx.fast.metal_kernel(
        name="decode_expert_contributions", input_names=["y", "weights", "experts", "dims"],
        output_names=["out"], source=r"""
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
            grid=(128, m * routes, 1), threadgroup=(128, 1, 1),
            output_shapes=[(n_experts, m, d)], output_dtypes=[y.dtype], init_value=0,
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
