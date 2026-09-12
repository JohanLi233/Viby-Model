"""Key-tile-owned backward for the contiguous-token MQA window pool.

This optional building block is deliberately not installed as an attention
backend here. The caller must keep positioned/recurrent windows on their
position-aware implementation and supply the forward LSE and matching Delta
from the same attention call. Compressed keys are handled by the caller.

Each threadgroup owns 32 consecutive keys and one of four disjoint query
ranges. QK and dOK use the same split-D MMA arithmetic as ``_BWD_KV`` in
``sparse_attention.py``. P and Ds retain its activation-dtype rounding points.
Persistent FP32 MMA accumulators replace per-query floating-point atomics;
four FP32 partials are summed before one cast to the window dtype. Accumulation
order changes, so bitwise parity is not promised.

Only H=16, D=64/128, matching BF16/FP16 operands, and standard consecutive
token positions are supported. The maximum static threadgroup allocation is
26,916 bytes at D=128; no Grad8 scratch or floating-point atomics are used.
GPU numerical and performance acceptance is required before enabling this
candidate in a training recipe.
"""

import math
from functools import lru_cache

import mlx.core as mx

_BK = 32
_SPLITS = 4
_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

_BACKWARD = r"""
    const uint tid=thread_position_in_threadgroup.x, sg=tid/32;
    constexpr uint H=16, BK=32, NP=D/64, DD=D/NP, NT=64*NP;
    constexpr uint SS=BK+4, QS=D+8, PS=BK+8;
    const uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    const uint TQ=dims[0], B=dims[1], TP=dims[2], tiles=TP/BK;
    const uint tile=thread_position_in_grid.y;
    const uint split=thread_position_in_grid.z;
    const uint b=tile/tiles, p0=(tile%tiles)*BK;

    threadgroup T Qs[H*QS], Gs[H*QS], Ks[BK*QS];
    threadgroup float DotPart[NP*H*SS], Sfull[H*SS];
    threadgroup T Ds[H*PS], Ps[H*PS];
    threadgroup bool KeyValid[BK];
    threadgroup uint AnyKey;

    // Load each physical key once. Padding beyond TQ is real output storage
    // but never an input read or an attention neighbor.
    for (uint i=tid;i<BK*D;i+=NT) {
        const uint j=i/D, d=i%D, key=p0+j;
        Ks[j*QS+d]=(key<TQ && pad[(size_t)b*TQ+key])
            ? window[((size_t)b*TQ+key)*D+d] : T(0);
    }

    // During QK/dOK, hg owns a head group. During dKV, the same SIMD group
    // owns key rows hg*8 and 16+hg*8, at dimension stripe dp.
    simdgroup_matrix<float,8,8> DK[2][DD/8];
    for (uint strip=0;strip<2;++strip) for (uint d=0;d<DD/8;++d)
        DK[strip][d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    // Union of the causal query neighborhoods of keys [p0,p0+BK).
    // size_t in the partition multiplication prevents intermediate overflow.
    const uint count=min(TQ-p0,uint(BK-1)+uint(W));
    const uint begin=p0+uint((size_t)count*split/SPLITS);
    const uint end=p0+uint((size_t)count*(split+1)/SPLITS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t=begin;t<end;++t) {
        const size_t query=(size_t)b*TQ+t;
        if (tid<BK) {
            const uint key=p0+tid;
            // Match _LOAD_TILE: key padding is tested, query padding is not.
            // This is a token-index distance, never an original-position mask.
            const bool valid=key<TQ && key<=t && t-key<uint(W)
                && pad[(size_t)b*TQ+key]
                && segment[(size_t)b*TQ+key]==segment[query];
            KeyValid[tid]=valid;
            const uint any=simd_sum(uint(valid));
            if (tid==0) AnyKey=any;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A uniform skip preserves zero output for completely empty tiles,
        // without multiplying masked probabilities by unused query operands.
        if (!AnyKey) {
            // All SIMD groups must read AnyKey before the first one may
            // overwrite it for the next query.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        for (uint i=tid;i<H*D;i+=NT) {
            const uint h=i/D, d=i%D;
            Qs[h*QS+d]=q[(query*H+h)*D+d];
            Gs[h*QS+d]=g[(query*H+h)*D+d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<T,8,8> Qf[DD/8], Gf[DD/8];
        for (uint d=0;d<DD/8;++d) {
            simdgroup_load(Qf[d],Qs+hg*8*QS+col+d*8,QS);
            simdgroup_load(Gf[d],Gs+hg*8*QS+col+d*8,QS);
        }
        simdgroup_matrix<float,8,8> Sf[BK/8], Uf[BK/8];
        for (uint j=0;j<BK/8;++j) {
            Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            Uf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        }
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*QS+col+d*8,QS,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
            simdgroup_multiply_accumulate(Uf[j],Gf[d],Kf,Uf[j]);
        }
        for (uint j=0;j<BK/8;++j)
            simdgroup_store(Sf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<H*BK;i+=NT) {
            const uint h=i/BK, j=i%BK;
            float dot=DotPart[h*SS+j];
            for (uint part=1;part<NP;++part)
                dot+=DotPart[(part*H+h)*SS+j];
            Sfull[h*SS+j]=dot*scale[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // QK readers have finished; reuse DotPart for dOK as in _BWD_KV.
        for (uint j=0;j<BK/8;++j)
            simdgroup_store(Uf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<H*BK;i+=NT) {
            const uint h=i/BK, j=i%BK;
            float dot=DotPart[h*SS+j];
            for (uint part=1;part<NP;++part)
                dot+=DotPart[(part*H+h)*SS+j];
            if (KeyValid[j]) {
                const float p=exp(Sfull[h*SS+j]-lse[query*H+h]);
                Ds[h*PS+j]=T(p*(dot-delta[query*H+h])*scale[0]);
                Ps[h*PS+j]=T(p);
            } else {
                Ds[h*PS+j]=T(0);
                Ps[h*PS+j]=T(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Every SIMD group reduces both 8-head chunks into its own two key
        // strips. Keep these FP32 fragments alive across all assigned queries.
        for (uint head8=0;head8<2;++head8) for (uint d=0;d<DD/8;++d) {
            simdgroup_matrix<T,8,8> Qk,Gk;
            simdgroup_load(Qk,Qs+head8*8*QS+col+d*8,QS);
            simdgroup_load(Gk,Gs+head8*8*QS+col+d*8,QS);
            for (uint strip=0;strip<2;++strip) {
                const uint key8=strip*16+hg*8;
                simdgroup_matrix<T,8,8> Df,Pf;
                simdgroup_load(Df,Ds+head8*8*PS+key8,PS,ulong2(0,0),true);
                simdgroup_load(Pf,Ps+head8*8*PS+key8,PS,ulong2(0,0),true);
                simdgroup_multiply_accumulate(DK[strip][d],Df,Qk,DK[strip][d]);
                simdgroup_multiply_accumulate(DK[strip][d],Pf,Gk,DK[strip][d]);
            }
        }
        // Qs/Gs/Ds/Ps remain live until every SIMD group finishes this query.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // TP pads the last tile to 32 keys, so every full 8x8 device store is safe.
    // Every [split,b,key,d] is owned by exactly one SIMD group, including zeros
    // for empty query splits; no output initialization or atomics are needed.
    for (uint strip=0;strip<2;++strip) for (uint d=0;d<DD/8;++d) {
        const uint key=p0+strip*16+hg*8;
        simdgroup_store(DK[strip][d],
            partials+(((size_t)split*B+b)*TP+key)*D+col+d*8,D);
    }
"""

_FINALIZE = r"""
    const uint d=thread_position_in_grid.x;
    const uint key=thread_position_in_grid.y;
    const uint b=thread_position_in_grid.z;
    const uint TQ=dims[0], B=dims[1], TP=dims[2];
    if (d>=D) return;
    float value=0.0f;
    for (uint split=0;split<SPLITS;++split)
        value+=partials[(((size_t)split*B+b)*TP+key)*D+d];
    dwindow[((size_t)b*TQ+key)*D+d]=T(value);
"""


def supported_for(q, window, g, lse, delta, segment, pad, window_size):
    """Check shapes/dtypes only; the caller also guards GPU and token positions."""
    if (
        q.ndim != 4
        or q.shape[-2] != 16
        or q.shape[-1] not in (64, 128)
        or q.dtype not in (mx.bfloat16, mx.float16)
    ):
        return False
    b, t, h, d = q.shape
    return (
        isinstance(window_size, int)
        and 0 < window_size < 2**31
        and 0 < b * t < 2**31
        and window.shape == (b, t, d)
        and g.shape == q.shape
        and window.dtype == g.dtype == q.dtype
        and lse.shape == delta.shape == (b, t, h)
        and lse.dtype == delta.dtype == mx.float32
        and segment.shape == pad.shape == (b, t)
        and segment.dtype == mx.int32
        and pad.dtype == mx.bool_
    )


@lru_cache(None)
def _kernels():
    backward = mx.fast.metal_kernel(
        name="mqa_window_bwd_key_tile32",
        input_names=[
            "q",
            "window",
            "g",
            "lse",
            "delta",
            "segment",
            "pad",
            "dims",
            "scale",
        ],
        output_names=["partials"],
        source=_BACKWARD,
        header=_HEADER,
    )
    finalize = mx.fast.metal_kernel(
        name="mqa_window_bwd_key_tile32_finalize",
        input_names=["partials", "dims"],
        output_names=["dwindow"],
        source=_FINALIZE,
    )
    return backward, finalize


def window_attention_backward(
    q, window, g, lse, delta, segment, pad, window_size, softmax_scale
):
    """Return dwindow using matching BTHD cotangent, FP32 LSE and Delta.

    ``delta[b,t,h]`` must be the existing attention backward's dot(dO, O),
    including the effect of compressed keys and the attention sink. Both LSE
    and Delta belong to the same forward call. The function intentionally has
    no custom VJP and implements first-order backward only. Unsupported calls
    raise before dispatch so integration can retain its existing fallback.
    Noncontiguous inputs are packed by the MLX custom Metal primitive.
    """
    if not supported_for(q, window, g, lse, delta, segment, pad, window_size):
        raise ValueError(
            "window backward requires H16/D64-or-128 BF16/FP16 BTHD operands"
        )
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")
    b, t, _, d = q.shape
    padded = ((t + _BK - 1) // _BK) * _BK
    threads = 64 * (d // 64)
    dims = mx.array([t, b, padded], mx.uint32)
    backward, finalize = _kernels()
    template = [("T", q.dtype), ("D", d), ("SPLITS", _SPLITS)]
    (partials,) = backward(
        inputs=[
            q,
            window,
            g,
            lse,
            delta,
            segment,
            pad,
            dims,
            mx.array([softmax_scale], mx.float32),
        ],
        template=template + [("W", window_size)],
        grid=(threads, b * (padded // _BK), _SPLITS),
        threadgroup=(threads, 1, 1),
        output_shapes=[(_SPLITS, b, padded, d)],
        output_dtypes=[mx.float32],
    )
    (dwindow,) = finalize(
        inputs=[partials, dims],
        template=template,
        grid=(d, t, b),
        threadgroup=(d, 1, 1),
        output_shapes=[window.shape],
        output_dtypes=[window.dtype],
    )
    return dwindow
