"""Original-order Indexer MMA with BQ K reuse and on-chip exact selection.

Only the stopped-gradient entry fuses away global scores. Public differentiable
scores retain the original query-owned VJP. Candidate arenas have static N/ NB
capacity, GPU lengths, increasing global ids, and retain all threshold ties.
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_INDEXER_SELECT", "1") != "0"
_BQ_ENABLED = os.environ.get("VIBY_INDEXER_BQ", "1") != "0"

_HEADER = r"""
#include <metal_simdgroup_matrix>
using namespace metal;
inline uint adjusted_key(float raw, uint gid) {
    #pragma clang fp contract(off)
    float offset=float(gid)*1e-7f;
    float value=raw-offset;
    if (value==0.0f) value=0.0f;
    uint u=as_type<uint>(value);
    return (u&0x80000000u) ? ~u : (u^0x80000000u);
}
template <typename IdPtr>
inline uint global_id(uint c, bool candidate, uint cb, IdPtr ids) {
    return candidate ? uint(ids[c/cb])*cb+c%cb : c;
}
// All threads in the TG call this function for ONE query at a time.
template <typename IdPtr, typename ReachPtr>
inline void exact_select(
    threadgroup float* raw, uint count, uint kth,
    bool candidate, uint cb, IdPtr ids,
    ReachPtr reach, bool use_reach,
    threadgroup uint* bits, threadgroup uint* hist, threadgroup uint* state,
    uint tid, uint nt, device int* output, device int* length) {
    threadgroup uint* keys=reinterpret_cast<threadgroup uint*>(raw);
    uint lane=tid%32, sg=tid/32, ng=nt/32;
    uint local_count=0;
    for (uint word=tid;word<(count+31)/32;word+=nt) {
        uint flags=0;
        for (uint j=0;j<32;++j) {
            uint i=word*32+j;
            if (i<count) {
                uint gid=global_id(i,candidate,cb,ids);
                float v=raw[i];
                bool eligible=v>-1e30f && (!use_reach || reach[gid]);
                if (eligible) { flags|=1u<<j; ++local_count; }
                keys[i]=adjusted_key(v,gid);
            }
        }
        bits[word]=flags;
    }
    uint total=simd_sum(local_count);
    if (lane==0) hist[sg]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) {
        uint m=0; for (uint g=0;g<ng;++g) m+=hist[g];
        state[0]=0; state[1]=0; state[2]=min(kth,m); state[3]=m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kth>0 && kth<state[3]) {
        for (int shift=28;shift>=0;shift-=4) {
            uint buckets[16]; for (uint j=0;j<16;++j) buckets[j]=0;
            for (uint i=tid;i<count;i+=nt)
                if ((bits[i/32]&(1u<<(i%32))) && (keys[i]&state[1])==state[0])
                    ++buckets[(keys[i]>>shift)&15u];
            for (uint j=0;j<16;++j) {
                uint v=simd_sum(buckets[j]);
                if (lane==0) hist[sg*16+j]=v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid==0) {
                for (int j=15;j>=0;--j) {
                    uint v=0; for (uint g=0;g<ng;++g) v+=hist[g*16+j];
                    if (state[2]>v) state[2]-=v;
                    else { state[0]|=uint(j)<<shift; state[1]|=15u<<shift; break; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint running=0;
    for (uint base=0;base<count;base+=nt) {
        uint i=base+tid;
        bool yes=kth>0 && i<count && (bits[i/32]&(1u<<(i%32))) && keys[i]>=state[0];
        uint rank=simd_prefix_exclusive_sum(uint(yes));
        uint n=simd_sum(uint(yes));
        if (lane==0) hist[sg]=n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset=0, size=0;
        for (uint g=0;g<ng;++g) { if (g<sg) offset+=hist[g]; size+=hist[g]; }
        if (yes) output[running+offset+rank]=int(global_id(i,candidate,cb,ids));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running+=size;
    }
    if (tid==0) length[0]=int(running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

_SCORE = r"""
    uint tid=thread_position_in_threadgroup.x, sg=tid/32;
    uint b=thread_position_in_grid.z, first=thread_position_in_grid.y*BQ;
    uint TQ=dims[0], N=dims[1], NB=(N+CB-1)/CB;
    constexpr uint NP=ID/32, NT=32*BQ*NP, SS=20, SP=ID+8;
    uint a=sg/NP, dp=sg%NP, col=dp*32;
    threadgroup T Qs[BQ*8*SP], Ks[16*SP], Ws[BQ*8];
    union DotWorkspace { float dots[BQ*NP*8*SS]; float blocks[SOURCE ? (MAXN+CB-1)/CB : 1]; };
    threadgroup DotWorkspace work;
    threadgroup uint valid[BQ*16], ptr[BQ], member[BQ], next_block, any_valid;
    threadgroup float local_scores[FUSED ? BQ*MAXN : 1];
    threadgroup uint eligible[FUSED ? (MAXN+31)/32 : 1];
    threadgroup uint hist[FUSED ? BQ*NP*16 : 1], state[4];
    for (uint i=tid;i<BQ*8*ID;i+=NT) {
        uint aq=i/(8*ID), h=(i/ID)%8, d=i%ID, t=first+aq;
        Qs[(aq*8+h)*SP+d]=(t<TQ && h<IH) ? q[((size_t)b*TQ+t)*IH*ID+h*ID+d] : T(0);
    }
    for (uint i=tid;i<BQ*8;i+=NT) {
        uint t=first+i/8, h=i%8;
        Ws[i]=(t<TQ && h<IH) ? w[((size_t)b*TQ+t)*IH+h] : T(0);
    }
    if (tid<BQ) ptr[tid]=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<T,8,8> Qf[4];
    for (uint d=0;d<4;++d) simdgroup_load(Qf[d],Qs+a*8*SP+col+d*8,SP);
    // Candidate lists are ascending: merge their union without reading or
    // computing noncandidate blocks, and load each common K vector once.
    for (;;) {
        if (tid==0) {
            uint next=0xffffffffu;
            for (uint aq=0;aq<BQ;++aq) {
                uint t=first+aq, qr=b*TQ+t;
                if (t<TQ && (CANDIDATE ? ptr[aq]<uint(block_lengths[qr]) : ptr[aq]<1))
                    next=min(next,CANDIDATE ? uint(block_ids[(size_t)qr*NB+ptr[aq]]) : 0u);
            }
            next_block=next;
            for (uint aq=0;aq<BQ;++aq) {
                uint t=first+aq, qr=b*TQ+t;
                member[aq]=t<TQ && (CANDIDATE ? (ptr[aq]<uint(block_lengths[qr]) &&
                    uint(block_ids[(size_t)qr*NB+ptr[aq]])==next) : ptr[aq]<1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (next_block==0xffffffffu) break;
        uint start=CANDIDATE ? next_block*CB : 0, span=CANDIDATE ? min(uint(CB),N-start) : N;
        for (uint base=0;base<span;base+=16) {
            for (uint i=tid;i<BQ*16;i+=NT) {
                uint aq=i/16, j=i%16, t=first+aq, n=start+base+j;
                valid[i]=member[aq] && base+j<span && n<N && reach[((size_t)b*TQ+t)*N+n];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid==0) {
                uint any=0; for (uint i=0;i<BQ*16;++i) any|=valid[i]; any_valid=any;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (any_valid) {
                for (uint i=tid;i<16*ID;i+=NT) {
                    uint j=i/ID,d=i%ID,yes=0;
                    for (uint aq=0;aq<BQ;++aq) yes|=valid[aq*16+j];
                    Ks[j*SP+d]=yes ? k[((size_t)b*N+start+base+j)*ID+d] : T(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_matrix<float,8,8> Sf[2];
                for (uint j=0;j<2;++j) Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
                for (uint d=0;d<4;++d) for (uint j=0;j<2;++j) {
                    simdgroup_matrix<T,8,8> Kf;
                    simdgroup_load(Kf,Ks+j*8*SP+col+d*8,SP,ulong2(0,0),true);
                    simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
                }
                for (uint j=0;j<2;++j)
                    simdgroup_store(Sf[j],work.dots+((a*NP+dp)*8)*SS+j*8,SS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint i=tid;i<BQ*16;i+=NT) {
                uint aq=i/16, j=i%16, t=first+aq;
                if (member[aq] && base+j<span) {
                    float acc=0.0f;
                    if (valid[i]) for (uint h=0;h<IH;++h) {
                        float dot=work.dots[(aq*NP*8+h)*SS+j];
                        for (uint p=1;p<NP;++p) dot+=work.dots[((aq*NP+p)*8+h)*SS+j];
                        acc+=max(dot,0.0f)*float(Ws[aq*8+h]);
                    }
                    uint slot=CANDIDATE ? ptr[aq]*CB+base+j : base+j;
                    float value=valid[i] ? acc : -1e30f;
                    if (FUSED) local_scores[aq*MAXN+slot]=value;
                    else out_scores[((size_t)b*TQ+t)*N+slot]=value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid<BQ && member[tid]) ++ptr[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!FUSED && tid<BQ && first+tid<TQ) {
        uint qr=b*TQ+first+tid, count=N;
        if (CANDIDATE) {
            uint blocks=ptr[tid]; count=blocks*CB;
            if (blocks && uint(block_ids[(size_t)qr*NB+blocks-1])==NB-1) count-=NB*CB-N;
        }
        lengths[qr]=int(count);
    }
    if (FUSED) for (uint aq=0;aq<BQ;++aq) {
        uint t=first+aq, qr=b*TQ+t;
        if (t>=TQ) break;  // uniform across the entire TG
        auto bids=block_ids+(CANDIDATE ? (size_t)qr*NB : 0);
        uint count=N;
        if (CANDIDATE) {
            uint blocks=uint(block_lengths[qr]);
            count=blocks*CB;
            if (blocks && uint(bids[blocks-1])==NB-1) count-=NB*CB-N;
        }
        if (SOURCE) {
            for (uint block=tid;block<NB;block+=NT) {
                float best=-1e30f;
                for (uint j=block*CB;j<min(N,(block+1)*CB);++j)
                    best=max(best,local_scores[aq*MAXN+j]);
                if (latest[qr]>0 && block==uint((latest[qr]-1)/CB)) best=INFINITY;
                work.blocks[block]=best;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            exact_select(work.blocks,NB,dims[3],false,CB,bids,reach,false,
                eligible,hist,state,tid,NT,out_block_ids+(size_t)qr*NB,out_block_lengths+qr);
        }
        exact_select(local_scores+aq*MAXN,count,dims[2],CANDIDATE,CB,bids,reach+(size_t)qr*N,true,
            eligible,hist,state,tid,NT,indices+(size_t)qr*N,lengths+qr);
    }
"""


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="indexer_bq_score_select", input_names=["q", "k", "w", "reach", "block_ids", "block_lengths", "latest", "dims"],
        output_names=["out_scores", "indices", "lengths", "out_block_ids", "out_block_lengths"],
        source=_SCORE, header=_HEADER,
    )


def supported(q, k):
    return (mx.default_device() == mx.gpu and q.dtype in (mx.float16, mx.bfloat16)
            and q.shape[-2] in (4, 8) and q.shape[-1] in (32, 64)
            and q.shape[1] > 1 and 0 < k.shape[1] <= 1024)


def _run(q, k, w, reach, *, fused, topk=1, candidates=None, candidate_source=False,
         latest=None, block_size=32, block_topk=1, bq=None):
    b, t, h, d = q.shape
    n = k.shape[1]
    nb = (n + block_size - 1) // block_size
    candidate = candidates is not None
    bq = bq or (2 if candidate else 4)
    ids, lens = candidates if candidate else (mx.zeros((2,), mx.int32), mx.zeros((2,), mx.int32))
    if latest is None:
        latest = mx.zeros((b, t), mx.int32)
    else:
        latest = mx.broadcast_to(latest, (b, t)).astype(mx.int32)
    nt = 32 * bq * (d // 32)
    outputs = _kernel()(
        inputs=[q, k.astype(q.dtype), w.astype(q.dtype), mx.broadcast_to(reach, (b, t, n)), ids, lens, latest,
                mx.array([t, n, topk, block_topk], mx.uint32)],
        template=[("T", q.dtype), ("IH", h), ("ID", d), ("BQ", bq), ("CB", block_size),
                  ("MAXN", n if fused else 1), ("FUSED", fused), ("CANDIDATE", candidate), ("SOURCE", candidate_source)],
        grid=(nt, (t + bq - 1) // bq, b), threadgroup=(nt, 1, 1),
        output_shapes=[(0,) if fused else (b, t, n), (b * t, n) if fused else (0,),
                       (b * t,), (b * t, nb) if candidate_source else (0,),
                       (b * t,) if candidate_source else (0,)],
        output_dtypes=[mx.float32, mx.int32, mx.int32, mx.int32, mx.int32],
    )
    return outputs


def score_bq(q, k, w, reach, bq=4):
    """Dense FP32 score forward; caller supplies the original Indexer VJP."""
    return _run(q, k, w, reach, fused=False, bq=bq)[0]


def fused_select(q, k, w, reach, topk, *, candidates=None, candidate_source=False,
                 latest=None, block_size=32, block_topk=1, bq=None):
    """Stopped-gradient row-major selection, optionally with source block ids."""
    if not supported(q, k):
        raise ValueError("fused Indexer selection requires GPU, T>1, N<=1024, H=4/8, D=32/64")
    if candidate_source and candidates is not None:
        raise ValueError("a candidate source must scan the complete reachable domain")
    _, ids, lengths, blocks, block_lengths = _run(
        mx.stop_gradient(q), mx.stop_gradient(k), mx.stop_gradient(w), mx.stop_gradient(reach),
        fused=True, topk=topk, candidates=candidates, candidate_source=candidate_source,
        latest=latest, block_size=block_size, block_topk=block_topk, bq=bq)
    return (ids, lengths), ((blocks, block_lengths) if candidate_source else None)


def candidate_mask(candidates, b, t, n, block_size):
    """Materialize only for an unfused consumer (not on the fused training path)."""
    ids, lengths = candidates
    nb = ids.shape[-1]
    chosen = selection_mask((ids, lengths), b, t, nb)
    return mx.repeat(chosen.reshape(b, t, nb), block_size, axis=-1)[..., :n]


@lru_cache(None)
def _selection_mask_kernel():
    return mx.fast.metal_kernel(
        name="indexer_selection_mask", input_names=["indices", "lengths", "dims"],
        output_names=["mask"], source=r"""
        uint tid=thread_position_in_grid.x, row=thread_position_in_grid.y, n=dims[0];
        for (uint slot=tid;slot<uint(lengths[row]);slot+=128)
            mask[(size_t)row*n+indices[(size_t)row*n+slot]]=true;
        """,
    )


def selection_mask(selection, b, t, n):
    """Dense bool only when native prefill SDPA needs it; training omits it."""
    return _selection_mask_kernel()(
        inputs=[*selection, mx.array([n], mx.uint32)],
        grid=(128, b * t, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b, t, n)], output_dtypes=[mx.bool_], init_value=0,
    )[0]


@lru_cache(None)
def _compact_backward():
    from .indexer_score import _BWD, _HEADER as header
    source = _BWD.replace(
        "uint b = query / Tlen;",
        """uint b = query / Tlen;
        uint nb=(N+CB-1)/CB, blocks=uint(block_lengths[query]);
        uint count=blocks*CB;
        if (blocks && uint(block_ids[(size_t)query*nb+blocks-1])==nb-1) count-=nb*CB-N;
        threadgroup uint tile_ok;""",
    ).replace("base < N;", "base < count;")
    source = source.replace(
        "if (!tile_valid[(size_t)query * ((N+BK-1)/BK) + base/BK]) continue;",
        """if (tid==0) {
            uint any=0;
            for (uint c=base;c<min(count,base+BK);++c) {
                uint n=uint(block_ids[(size_t)query*nb+c/CB])*CB+c%CB;
                any|=uint(rb[n]);
            }
            tile_ok=any;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tile_ok) continue;""",
    )
    for offset in ("row", "j"):
        source = source.replace(f"uint n = base + {offset};", f"""uint slot=base+{offset};
            uint n=slot<count ? uint(block_ids[(size_t)query*nb+slot/CB])*CB+slot%CB : N;""")
    source = source.replace("float(gb[n])", "float(gb[slot])")
    return mx.fast.metal_kernel(
        name="indexer_compact_score_bwd", input_names=["q", "k", "w", "reach", "block_ids", "block_lengths", "g", "dims"],
        output_names=["dq", "dk", "dw"], source=source, header=header, atomic_outputs=True,
    )


@lru_cache(None)
def _compact_operation(block_size):
    backward = _compact_backward()

    @mx.custom_function
    def op(q, k, w, reach, ids, lengths):
        output = _run(q, k, w, reach, fused=False, candidates=(ids, lengths), block_size=block_size)
        return output[0], output[2]

    @op.vjp
    def vjp(primals, cotangent, output):
        q, k, w, reach, ids, lengths = primals
        g, _ = cotangent
        b, t, h, d = q.shape
        n = k.shape[1]
        dq, dk, dw = backward(
            inputs=[q, k, w, reach, ids, lengths, g.astype(q.dtype), mx.array([t, n, b, 1], mx.uint32)],
            template=[("T", q.dtype), ("IH", h), ("ID", d), ("CB", block_size), ("SHARDS", 8)],
            grid=(d, b * t, 1), threadgroup=(d, 1, 1),
            output_shapes=[q.shape, (8,) + k.shape, w.shape], output_dtypes=[mx.float32] * 3,
            init_value=0,
        )
        return (dq.astype(q.dtype), mx.sum(dk, axis=0).astype(k.dtype), dw.astype(w.dtype),
                mx.zeros_like(reach), mx.zeros_like(ids), mx.zeros_like(lengths))

    return op


def score_candidate_blocks(q, k, w, reach, candidates, block_size):
    """Differentiable compact FP32 arena [B,T,N] plus GPU per-row lengths.

    Only each valid prefix is initialized; slot c names the global key in
    block_ids[c//CB]*CB+c%CB. Gradients traverse those same slots and scatter
    directly to the original K primal without making dense score gradients.
    """
    return _compact_operation(block_size)(q, k.astype(q.dtype), w.astype(q.dtype),
                                         mx.broadcast_to(reach, (q.shape[0], q.shape[1], k.shape[1])), *candidates)
