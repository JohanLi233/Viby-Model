"""Indexed MQA flash attention for the 16-head training recipe.

KV stays in its original window/compressed pools. One threadgroup owns one query;
its 16 heads share each indirectly loaded key tile (16 keys by default, 32 with
VIBY_SPARSE_ATTN_KEY_TILE=32) and use SIMD-group MMA. Online softmax includes the
learned sink. Backward recomputes only visible scores,
uses MMA for dQ and head-reduced d(K=V), and shards the latter's FP32 accumulation.
No [B,T,visible,D] gathered KV tensor or [B,H,T,N] score tensor is materialized.

The existing Boolean mask is compacted verbatim, including ties that can select
more than index_topk positions. Full/Reindex/Reuse selection and document/padding
rules are not approximated. VIBY_SPARSE_ATTN_KERNEL=0 restores native SDPA.

2026-09-11 回滚：`VIBY_SPARSE_ATTN_KEY_TILE` 与 `VIBY_SPARSE_TOPK_KERNEL` 的
默认值都退回改动前（16 / 0），代码保留——`=32` / `=1` 可原样重开。
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_SPARSE_ATTN_KERNEL", "1") != "0"
_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
_TOPK_ENABLED = os.environ.get("VIBY_SPARSE_TOPK_KERNEL", "0") != "0"
_SHARDS = 4
_KEY_TILE = int(os.environ.get("VIBY_SPARSE_ATTN_KEY_TILE", "16"))
if _KEY_TILE not in (16, 32):
    raise ValueError("VIBY_SPARSE_ATTN_KEY_TILE must be 16 or 32")

_COMPACT = r"""
    uint lane = thread_position_in_grid.x;
    uint query = thread_position_in_grid.y;
    uint N = dims[0];
    uint count = 0;
    for (uint base=0;base<N;base+=32) {
        uint j=base+lane;
        uint yes=(j<N && visible[(size_t)query*N+j]) ? 1u : 0u;
        uint rank=simd_prefix_exclusive_sum(yes);
        if (yes) indices[(size_t)query*N+count+rank]=j;
        count+=simd_sum(yes);
    }
    if (lane==0) lengths[query]=count;
"""

# Exact radix selection over the IEEE-754 ordered representation. Only the
# threshold is selected: the final >= comparison retains ALL boundary ties,
# exactly as _topk_masks does. This fuses selection, mask and index compaction.
_SELECT = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32, lane=tid%32;
    uint query=thread_position_in_grid.y, N=dims[0], kth=dims[1];
    uint keys[ITEMS];
    for (uint i=0;i<ITEMS;++i) {
        uint j=i*128+tid;
        float value=j<N ? scores[(size_t)query*N+j]-float(j)*1e-7f : -INFINITY;
        uint bits=as_type<uint>(value);
        keys[i]=bits ^ ((bits&0x80000000u) ? 0xffffffffu : 0x80000000u);
    }
    threadgroup uint hist[4*16];
    threadgroup uint prefix, prefix_mask, rank, selected_count;
    if (tid==0) { prefix=0; prefix_mask=0; rank=kth; selected_count=N; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kth<N) for (int shift=28;shift>=0;shift-=4) {
        // Once only one candidate remains, its full bits give the threshold.
        if (selected_count==1) {
            uint winner=0;
            for (uint i=0;i<ITEMS;++i)
                if (i*128+tid<N && (keys[i]&prefix_mask)==prefix) winner=max(winner,keys[i]);
            winner=simd_max(winner);
            if (lane==0) hist[sg]=winner;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid==0) prefix=max(max(hist[0],hist[1]),max(hist[2],hist[3]));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }
        uint counts[16];
        for (uint bucket=0;bucket<16;++bucket) counts[bucket]=0;
        for (uint i=0;i<ITEMS;++i)
            if (i*128+tid<N && (keys[i]&prefix_mask)==prefix) ++counts[(keys[i]>>shift)&15u];
        for (uint bucket=0;bucket<16;++bucket) {
            uint count=simd_sum(counts[bucket]);
            if (lane==0) hist[sg*16+bucket]=count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid==0) {
            for (int bucket=15;bucket>=0;--bucket) {
                uint count=hist[bucket]+hist[16+bucket]+hist[32+bucket]+hist[48+bucket];
                if (rank>count) rank-=count;
                else {
                    prefix|=uint(bucket)<<shift;
                    prefix_mask|=15u<<shift;
                    selected_count=count;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reuse hist for a four-SIMD-group exclusive prefix, preserving key order.
    uint total=0;
    for (uint i=0;i<ITEMS;++i) {
        uint j=i*128+tid;
        bool yes=j<N && keys[i]>=prefix && reach[(size_t)query*N+j]
                     && scores[(size_t)query*N+j]>-1e30f;
        if (j<N) keep[(size_t)query*N+j]=yes;
        uint local=simd_prefix_exclusive_sum(uint(yes));
        uint count=simd_sum(uint(yes));
        if (lane==0) hist[sg]=count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset=0;
        for (uint g=0;g<sg;++g) offset+=hist[g];
        if (yes) indices[(size_t)query*N+total+offset+local]=j;
        total+=hist[0]+hist[1]+hist[2]+hist[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid==0) lengths[query]=total;
"""


@lru_cache(None)
def _selection_kernel():
    return mx.fast.metal_kernel(
        name="mqa_topk_compact", input_names=["scores", "reach", "dims"],
        output_names=["keep", "indices", "lengths"], source=_SELECT,
    )


def _selection_items(n):
    return 1 << max(0, ((n+127)//128-1).bit_length())


def topk_enabled():
    return _TOPK_ENABLED


def select_topk(scores, reach, k):
    """Drop-in threshold selection plus ready-to-use indexed-attention metadata."""
    b, t, n=scores.shape
    if n==0:
        return mx.zeros(scores.shape, mx.bool_), (mx.zeros((b*t, 0), mx.int32), mx.zeros((b*t,), mx.int32))
    keep, indices, lengths = _selection_kernel()(
        inputs=[mx.stop_gradient(scores), mx.broadcast_to(reach, scores.shape),
                mx.array([n, min(k, n)], mx.uint32)],
        template=[("ITEMS", _selection_items(n))],
        grid=(128, b*t, 1), threadgroup=(128, 1, 1),
        output_shapes=[scores.shape, (b*t, n), (b*t,)],
        output_dtypes=[mx.bool_, mx.int32, mx.int32],
    )
    return keep, (indices, lengths)


@lru_cache(None)
def prewarm_topk(max_keys):
    if not _TOPK_ENABLED or mx.default_device()!=mx.gpu:
        return
    items=1
    while items<=_selection_items(max_keys):
        n=items*128
        values=mx.zeros((1, 1, n), mx.float32)
        mx.eval(select_topk(values, mx.ones(values.shape, mx.bool_), min(64, n)))
        items*=2


# Every forward/backward key tile has exactly the same visibility and addresses.
# Invalid window slots load zeros and are assigned -inf before softmax.
_LOAD_TILE = r"""
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<BK) {
            uint slot=base+tid;
            int key=-1;
            if (slot<W) {
                int p=int(t)-int(W)+1+int(slot);
                if (p>=0 && pad[b*TQ+uint(p)] && segment[b*TQ+uint(p)]==segment[query])
                    key=p;
            } else if (slot<W+uint(lengths[query])) {
                key=int(TQ)+indices[(size_t)query*NC+slot-W];
            }
            ids[tid]=key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<BK*D;i+=NT) {
            uint row=i/D,d=i%D;
            int key=ids[row];
            T value=T(0);
            if (key>=0) {
                value=uint(key)<TQ ? window[((size_t)b*TQ+uint(key))*D+d]
                    : compressed[((size_t)b*NC+uint(key)-TQ)*D+d];
            }
            Ks[row*(D+8)+d]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_FWD = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    constexpr uint H=16;
    constexpr uint SS=BK+4, SP=BK+8;
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup float scores[NP*H*SS];
    threadgroup T Ps[H*SP];
    threadgroup float Os[H*(D+4)];
    threadgroup float ms[H], ls[H], rescale[H];
    if (tid<H) { ms[tid]=sinks[tid]; ls[tid]=1.0f; }
    simdgroup_matrix<T,8,8> Qf[DD/8];
    simdgroup_matrix<float,8,8> Of[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],q+((size_t)query*H+hg*8)*D+col+d*8,D);
        Of[d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    }
    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8];
        for (uint j=0;j<BK/8;++j) Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
        }
        for (uint j=0;j<BK/8;++j) simdgroup_store(Sf[j],scores+(dp*H+hg*8)*SS+j*8,SS);
        for (uint d=0;d<DD/8;++d) simdgroup_store(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) {
            float old=ms[tid], next=old;
            for (uint j=0;j<BK;++j) {
                float dot=scores[tid*SS+j];
                for (uint p=1;p<NP;++p) dot+=scores[(p*H+tid)*SS+j];
                float s=ids[j]>=0 ? dot*scale[0] : -INFINITY;
                scores[tid*SS+j]=s;
                next=max(next,s);
            }
            float alpha=exp(old-next), sum=0.0f;
            for (uint j=0;j<BK;++j) {
                float p=exp(scores[tid*SS+j]-next);
                Ps[tid*SP+j]=T(p);
                sum+=p;
            }
            ms[tid]=next; ls[tid]=ls[tid]*alpha+sum; rescale[tid]=alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<H*D;i+=NT) Os[(i/D)*(D+4)+i%D]*=rescale[i/D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=0;d<DD/8;++d) {
            simdgroup_load(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
            for (uint j=0;j<BK/8;++j) {
                simdgroup_matrix<T,8,8> Pf,Vf;
                simdgroup_load(Pf,Ps+hg*8*SP+j*8,SP);
                simdgroup_load(Vf,Ks+j*8*(D+8)+col+d*8,D+8);
                simdgroup_multiply_accumulate(Of[d],Pf,Vf,Of[d]);
            }
        }
    }
    for (uint d=0;d<DD/8;++d) simdgroup_store(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<H*D;i+=NT) out[(size_t)query*H*D+i]=Os[(i/D)*(D+4)+i%D]/ls[i/D];
    if (tid<H) lse[query*H+tid]=ms[tid]+log(ls[tid]);
"""

_BWD = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint B=dims[0], TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    uint shard=(t/16)%SHARDS;
    constexpr uint H=16;
    constexpr uint SS=BK+(NP==2 ? 2 : 4), SP=BK+8;
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup float scores[NP*H*SS], dprob[NP*H*SS];
    threadgroup T Ps[H*SP], Ds[H*SP];
    threadgroup float tile_grad[16*(D+4)];
    threadgroup float delta[H];
    if (tid<H) {
        float a=0.0f;
        for (uint d=0;d<D;++d) a+=float(g[((size_t)query*H+tid)*D+d])*out[((size_t)query*H+tid)*D+d];
        delta[tid]=a;
    }
    simdgroup_matrix<T,8,8> Qf[DD/8], Gf[DD/8];
    simdgroup_matrix<float,8,8> DQ[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],q+((size_t)query*H+hg*8)*D+col+d*8,D);
        simdgroup_load(Gf[d],g+((size_t)query*H+hg*8)*D+col+d*8,D);
        DQ[d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    }
    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8], DP[BK/8];
        for (uint j=0;j<BK/8;++j) {
            Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            DP[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        }
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
            simdgroup_multiply_accumulate(DP[j],Gf[d],Kf,DP[j]);
        }
        for (uint j=0;j<BK/8;++j) {
            simdgroup_store(Sf[j],scores+(dp*H+hg*8)*SS+j*8,SS);
            simdgroup_store(DP[j],dprob+(dp*H+hg*8)*SS+j*8,SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float dot=scores[tid*SS+j], dg=dprob[tid*SS+j];
            for (uint part=1;part<NP;++part) {
                dot+=scores[(part*H+tid)*SS+j];
                dg+=dprob[(part*H+tid)*SS+j];
            }
            float p=ids[j]>=0 ? exp(dot*scale[0]-lse[query*H+tid]) : 0.0f;
            Ps[tid*SP+j]=T(p);
            Ds[tid*SP+j]=T(p*(dg-delta[tid])*scale[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Df,Kf;
            simdgroup_load(Df,Ds+hg*8*SP+j*8,SP);
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8);
            simdgroup_multiply_accumulate(DQ[d],Df,Kf,DQ[d]);
        }
        // Each SIMD group handles eight key rows. A 32-key tile is drained in
        // two 16-row stripes: keeping only 16 rows of tile_grad avoids crossing
        // Metal's 32 KiB limit (D=128, BK=32, NP=2: 32,608 bytes).
        for (uint key_base=0;key_base<BK;key_base+=16) {
            for (uint d=0;d<DD/8;++d) {
                simdgroup_matrix<float,8,8> DK=make_filled_simdgroup_matrix<float,8,8>(0.0f);
                for (uint h=0;h<2;++h) {
                    simdgroup_matrix<T,8,8> Df,Pf,Q,G;
                    simdgroup_load(Df,Ds+h*8*SP+key_base+hg*8,SP,ulong2(0,0),true);
                    simdgroup_load(Pf,Ps+h*8*SP+key_base+hg*8,SP,ulong2(0,0),true);
                    simdgroup_load(Q,q+((size_t)query*H+h*8)*D+col+d*8,D);
                    simdgroup_load(G,g+((size_t)query*H+h*8)*D+col+d*8,D);
                    simdgroup_multiply_accumulate(DK,Df,Q,DK);
                    simdgroup_multiply_accumulate(DK,Pf,G,DK);
                }
                simdgroup_store(DK,tile_grad+hg*8*(D+4)+col+d*8,D+4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tid;i<16*D;i+=NT) {
                uint r=i/D,d=i%D;
                int key=ids[key_base+r];
                float value=tile_grad[r*(D+4)+d];
                if (key>=0 && uint(key)<TQ)
                    atomic_fetch_add_explicit(dwindow+(((size_t)shard*B+b)*TQ+uint(key))*D+d,value,memory_order_relaxed);
                else if (key>=0)
                    atomic_fetch_add_explicit(dcompressed+(((size_t)shard*B+b)*NC+uint(key)-TQ)*D+d,value,memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // Last tile's atomic readers must finish before its scratch is reused for dQ.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=0;d<DD/8;++d) simdgroup_store(DQ[d],tile_grad+hg*8*(D+4)+col+d*8,D+4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<H*D;i+=NT)
        atomic_store_explicit(dq+(size_t)query*H*D+i,tile_grad[(i/D)*(D+4)+i%D],memory_order_relaxed);
    if (tid<H)
        atomic_store_explicit(dsink+query*H+tid,-exp(sinks[tid]-lse[query*H+tid])*delta[tid],memory_order_relaxed);
"""


@lru_cache(None)
def _kernels():
    compact = mx.fast.metal_kernel(
        name="mqa_compact_visible", input_names=["visible", "dims"],
        output_names=["indices", "lengths"], source=_COMPACT,
    )
    inputs = ["q", "window", "compressed", "indices", "lengths", "segment", "pad", "sinks", "dims", "scale"]
    fwd = mx.fast.metal_kernel(
        name="mqa_indexed_flash_fwd", input_names=inputs, output_names=["out", "lse"],
        source=_FWD.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
    )
    bwd = mx.fast.metal_kernel(
        name="mqa_indexed_flash_bwd", input_names=inputs+["g", "out", "lse"],
        output_names=["dq", "dwindow", "dcompressed", "dsink"],
        source=_BWD.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
        atomic_outputs=True,
    )
    return compact, fwd, bwd


def enabled_for(q, window_size):
    return (_ENABLED and mx.default_device() == mx.gpu and q.shape[-2] == 16
            and q.shape[-1] in (64, 128) and window_size > 0
            and q.dtype in (mx.bfloat16, mx.float16))


def _compact_mask(mask):
    b, t, n = mask.shape
    return _kernels()[0](
        inputs=[mask, mx.array([n], mx.uint32)],
        grid=(32, b*t, 1), threadgroup=(32, 1, 1),
        output_shapes=[(b*t, n), (b*t,)], output_dtypes=[mx.int32, mx.int32],
    )


@lru_cache(None)
def _operation(window_size, softmax_scale, key_tile):
    _, forward, backward = _kernels()
    parts = 2 if key_tile == 32 else 1
    threads = 64 * parts

    def constants(q, compressed):
        b, t, _, d = q.shape
        return mx.array([b, t, compressed.shape[1]], mx.uint32), mx.array([softmax_scale], mx.float32)

    @mx.custom_function
    def op(q, window, compressed, indices, lengths, segment, pad, sinks):
        b, t, h, d = q.shape
        dims, scale = constants(q, compressed)
        out, lse = forward(
            inputs=[q, window, compressed, indices, lengths, segment, pad, sinks, dims, scale],
            template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", key_tile), ("NP", parts)],
            grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
            output_shapes=[q.shape, (b, t, h)], output_dtypes=[mx.float32, mx.float32],
        )
        return out, lse

    @op.vjp
    def vjp(primals, cotangent, output):
        q, window, compressed, indices, lengths, segment, pad, sinks = primals
        g, _ = cotangent  # LSE is internal state and is never exposed by the wrapper.
        out, lse = output
        b, t, h, d = q.shape
        dims, scale = constants(q, compressed)
        dq, dw, dc, ds = backward(
            inputs=[q, window, compressed, indices, lengths, segment, pad, sinks,
                    dims, scale, g.astype(q.dtype), out, lse],
            template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", key_tile), ("NP", parts), ("SHARDS", _SHARDS)],
            grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
            output_shapes=[q.shape, (_SHARDS,)+window.shape,
                           (_SHARDS,)+compressed.shape, (b, t, h)],
            output_dtypes=[mx.float32]*4, init_value=0,
        )
        # MLX 0.32.2 flattens a custom VJP with tree_flatten(..., strict=False):
        # None leaves disappear rather than reserving argument positions.
        # CustomTransforms::vjp then indexes that vector with the ORIGINAL
        # argnums. In particular sinks is argument 7; four None leaves here
        # shrink eight input VJPs to four and make its lookup out of bounds.
        # Return one array per primal, even for nondifferentiable metadata.
        # These zero leaves are discarded when MLX selects trainable argnums.
        return (
            dq.astype(q.dtype),
            mx.sum(dw, axis=0).astype(window.dtype),
            mx.sum(dc, axis=0).astype(compressed.dtype),
            mx.zeros_like(indices),
            mx.zeros_like(lengths),
            mx.zeros_like(segment),
            mx.zeros_like(pad),
            mx.sum(ds, axis=(0, 1)).astype(sinks.dtype),
        )

    return op


def indexed_attention(q, window, compressed, visible, segment_ids, pad_mask, sinks,
                      window_size, softmax_scale, selection=None, key_tile=None):
    """Returns [B,H,T,D]. Caller selects the supported no-cache training path."""
    b, t, h, d = q.shape
    if selection is None:
        visible = mx.broadcast_to(visible, (b, t, compressed.shape[1]))
        indices, lengths = _compact_mask(mx.stop_gradient(visible))
    else:
        indices, lengths = selection
    segment = (mx.zeros((b, t), mx.int32) if segment_ids is None
               else segment_ids.astype(mx.int32))
    pad = (mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_))
    out, _ = _operation(window_size, softmax_scale, _KEY_TILE if key_tile is None else key_tile)(
        q, window.astype(q.dtype), compressed.astype(q.dtype), indices, lengths,
        segment, pad, sinks.astype(mx.float32),
    )
    return out.astype(q.dtype).transpose(0, 2, 1, 3)


@lru_cache(None)
def prewarm_sparse_attention(head_dim, window_size, softmax_scale, dtype=mx.bfloat16, key_tile=None):
    """Materialize JIT libraries eagerly; no correctness or performance checks."""
    q = mx.zeros((1, 1, 16, head_dim), dtype)
    if not enabled_for(q, window_size):
        return
    kv = mx.zeros((1, 1, head_dim), dtype)
    mask = mx.ones((1, 1, 1), mx.bool_)
    sinks = mx.zeros((16,), mx.float32)

    def fn(a, b, c, s):
        return indexed_attention(a, b, c, mask, None, None, s, window_size, softmax_scale, key_tile=key_tile)

    out, grads = mx.vjp(fn, [q, kv, kv, sinks], [mx.ones((1, 16, 1, head_dim), dtype)])
    mx.eval(out, grads)
