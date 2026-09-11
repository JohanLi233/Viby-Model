"""Sorted MoE routing without an expanded input or atomic token reduction.

The gate/up GEMM gathers token rows directly into its threadgroup tiles. Its
weight VJP also gathers directly, so [M*K,D] input copies are never needed.
The input VJP and route combine reduce choices using an inverse permutation.
The down GEMM weights and writes its tiles directly to token accumulators; its
VJP uses the smaller [M*K,I] intermediate. Expert ordering remains the existing
MLX argsort; no capacity or token drops. Accumulation order differs from MLX.
VIBY_MOE_KERNEL=1 re-enables this implementation.
VIBY_MOE_COMBINE_KERNEL=0 disables the native gather_mm route-combine P2 path.

2026-09-10 实测（M4 Max / MLX 0.32.2 / bf16 / B=4,T=1024,D=1024,E=96,K=6,I=256）：
这条路径比 MLX 原生 gather_mm 慢，**默认关闭**。
- 单层 fwd+bwd：本 kernel 22.85 ms vs MLX 原生 14.56 ms（慢 57%）
- 整步 fwd+bwd：0.669 s vs 0.500 s；吞吐 6,122 vs 8,195 tok/s（MFU 44.0% vs 58.9%）
原因：MLX 原生 gather_mm 的两段专家 GEMM 已跑到 11.7 / 10.8 TFLOPS（稠密上限 12~14），
本 kernel 只到 ~5 TFLOPS。重新启用前请先把这两段 MMA 效率追平。
"""
import os
from functools import lru_cache

import mlx.core as mx

# Keep the hand-written gather/down GEMMs independently opt-in.  Route
# combine is a separate P2 experiment because native ``mx.gather_mm`` remains
# the production GEMM path.
# 2026-09-11 回滚：route-combine 默认值退回改动前（关闭），代码保留——
# VIBY_MOE_COMBINE_KERNEL=1 可原样重开。
_ENABLED = os.environ.get("VIBY_MOE_KERNEL", "0") != "0"
_COMBINE_ENABLED = os.environ.get(
    "VIBY_MOE_COMBINE_KERNEL",
    os.environ.get("VIBY_MOE_COMBINE", "0"),
) != "0"
_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

_META_SOURCE = r"""
    uint gid = thread_position_in_grid.x;
    uint G = order_shape[0];
    uint E = dims[0];
    if (gid < G) inverse[order[gid]] = gid;
    if (gid < 128) {
        threadgroup uint tiles[128];
        uint first = 0, last = G;
        if (gid < E) {
            uint lo = 0, hi = G;
            while (lo < hi) {
                uint mid = (lo+hi)/2;
                if (experts[mid] < int(gid)) lo = mid+1; else hi = mid;
            }
            first = lo;
            hi = G;
            while (lo < hi) {
                uint mid = (lo+hi)/2;
                if (experts[mid] <= int(gid)) lo = mid+1; else hi = mid;
            }
            last = lo;
            offsets[gid] = first;
            tiles[gid] = (last-first+31)/32;
        } else {
            tiles[gid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint e = 0; e < gid; ++e) prefix += tiles[e];
        if (gid < E) tile_offsets[gid] = prefix;
        if (gid == 0) {
            uint total = 0;
            for (uint e = 0; e < E; ++e) total += tiles[e];
            tile_offsets[E] = total;
            offsets[E] = G;
        }
    }
"""

# 32 sorted rows x 64 output columns, reduction blocks of 32. Four SIMD groups
# each own a 16x32 output tile. Padding prevents threadgroup bank conflicts.
_GATHER_MM_SOURCE = r"""
    uint tid = thread_position_in_grid.x % 128;
    uint tile = thread_position_in_grid.x / 128;
    uint nc = thread_position_in_grid.y * 64;
    uint E = offsets_shape[0]-1;
    if (tile >= uint(tile_offsets[E])) return;
    uint lo = 0, hi = E;
    while (lo+1 < hi) {
        uint mid = (lo+hi)/2;
        if (uint(tile_offsets[mid]) <= tile) lo = mid; else hi = mid;
    }
    uint e = lo;
    uint r0 = offsets[e] + (tile-tile_offsets[e])*32;
    uint rend = offsets[e+1];
    uint sg = tid/32;
    uint mr = (sg/2)*16, mc = (sg%2)*32;
    threadgroup T A[32*40];
    threadgroup T B[32*72];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (uint i=0;i<2;++i) for (uint j=0;j<4;++j)
        acc[i][j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint kk=0;kk<D;kk+=32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<32*32;i+=128) {
            uint r=i/32, d=i%32;
            A[r*40+d] = r0+r < rend ? x[(size_t)(order[r0+r]/K)*D+kk+d] : T(0);
        }
        for (uint i=tid;i<32*64;i+=128) {
            uint d=i/64, c=i%64;
            B[d*72+c] = nc+c<N ? weight[((size_t)e*N+nc+c)*D+kk+d] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k=0;k<4;++k) {
            simdgroup_matrix<T,8,8> af[2], bf[4];
            for (uint i=0;i<2;++i) simdgroup_load(af[i],A+(mr+i*8)*40+k*8,40);
            for (uint j=0;j<4;++j) simdgroup_load(bf[j],B+k*8*72+mc+j*8,72);
            for (uint i=0;i<2;++i) for (uint j=0;j<4;++j)
                simdgroup_multiply_accumulate(acc[i][j],af[i],bf[j],acc[i][j]);
        }
    }
    // Reuse the A/B allocation only after all SIMD groups finish loading it.
    threadgroup float C[32*68];
    for (uint i=0;i<2;++i) for (uint j=0;j<4;++j)
        simdgroup_store(acc[i][j],C+(mr+i*8)*68+mc+j*8,68);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<32*64;i+=128) {
        uint r=i/64,c=i%64;
        if (r0+r<rend && nc+c<N) out[(size_t)(r0+r)*N+nc+c]=T(C[r*68+c]);
    }
"""

_DW_SOURCE = r"""
    uint tid = thread_position_in_grid.x % 128;
    uint nr = (thread_position_in_grid.x / 128)*32;
    uint dc = thread_position_in_grid.y*32;
    uint e = thread_position_in_grid.z;
    uint sg = tid/32, mr=(sg/2)*16, mc=(sg%2)*16;
    uint first=offsets[e], end=offsets[e+1];
    threadgroup T A[32*40];
    threadgroup T B[32*40];
    simdgroup_matrix<float,8,8> acc[2][2];
    for (uint i=0;i<2;++i) for (uint j=0;j<2;++j)
        acc[i][j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint rr=first;rr<end;rr+=32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<32*32;i+=128) {
            uint r=i/32,c=i%32;
            A[r*40+c]=(rr+c<end && nr+r<N) ? grad[(size_t)(rr+c)*N+nr+r] : T(0);
            B[r*40+c]=(rr+r<end) ? x[(size_t)(order[rr+r]/K)*D+dc+c] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k=0;k<4;++k) {
            simdgroup_matrix<T,8,8> af[2],bf[2];
            for (uint i=0;i<2;++i) simdgroup_load(af[i],A+(mr+i*8)*40+k*8,40);
            for (uint j=0;j<2;++j) simdgroup_load(bf[j],B+k*8*40+mc+j*8,40);
            for (uint i=0;i<2;++i) for (uint j=0;j<2;++j)
                simdgroup_multiply_accumulate(acc[i][j],af[i],bf[j],acc[i][j]);
        }
    }
    threadgroup float C[32*36];
    for (uint i=0;i<2;++i) for (uint j=0;j<2;++j)
        simdgroup_store(acc[i][j],C+(mr+i*8)*36+mc+j*8,36);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<32*32;i+=128) {
        uint r=i/32,c=i%32;
        if (nr+r<N) dw[((size_t)e*N+nr+r)*D+dc+c]=T(C[r*36+c]);
    }
"""

_REDUCE_SOURCE = r"""
    uint gid=thread_position_in_grid.x;
    uint M=inverse_shape[0]/K;
    if (gid>=M*D) return;
    uint t=gid/D,d=gid%D;
    float v=0.0f;
    for (uint k=0;k<K;++k) v+=float(g[(size_t)inverse[t*K+k]*D+d]);
    out[gid]=T(v);
"""

_COMBINE_SOURCE = r"""
    uint gid=thread_position_in_grid.x;
    uint M=inverse_shape[0]/K;
    if (gid>=M*D) return;
    uint t=gid/D,d=gid%D;
    // Match native BF16 scatter_add's per-update rounding.  The fixed choice
    // order is still different from the original atomic CAS arrival order.
    T v=T(0);
    for (uint k=0;k<K;++k) {
        uint route=t*K+k;
        T product=T(y[(size_t)inverse[route]*D+d]*T(weights[route]));
        v=T(v+product);
    }
    out[gid]=v;
"""

_COMBINE_VJP_SOURCE = r"""
    uint tid=thread_position_in_grid.x;
    uint s=thread_position_in_grid.y;
    uint route=order[s],t=route/K;
    T w=T(weights[route]);
    float acc=0.0f;
    for (uint d=tid;d<D;d+=128) {
        T v=g[(size_t)t*D+d];
        dy[(size_t)s*D+d]=T(v*w);
        acc+=float(T(v*y[(size_t)s*D+d]));
    }
    acc=simd_sum(acc);
    threadgroup float sums[4];
    if (tid%32==0) sums[tid/32]=acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) dw[route]=float(T((sums[0]+sums[1])+(sums[2]+sums[3])));
"""

# The down projection reads already sorted activations. Weighting and token
# reduction occur at its epilogue, so the large [G,D] expert output is not stored.
_SORTED_X = _GATHER_MM_SOURCE.replace(
    "x[(size_t)(order[r0+r]/K)*D+kk+d]", "x[(size_t)(r0+r)*D+kk+d]"
)
_ATOMIC_EPILOGUE = r"""
        if (r0+r<rend && nc+c<N) {
            uint route=order[r0+r];
            T value=T(T(C[r*68+c])*T(weights[route]));
            atomic_fetch_add_explicit(out+(size_t)(route/K)*N+nc+c,float(value),memory_order_relaxed);
        }
"""
_DOWN_SOURCE = _SORTED_X.replace(
    "if (r0+r<rend && nc+c<N) out[(size_t)(r0+r)*N+nc+c]=T(C[r*68+c]);",
    _ATOMIC_EPILOGUE,
)
# Reverse GEMMs consume weights in their original stored orientation.
_TRANSPOSED_WEIGHT = "weight[((size_t)e*D+kk+d)*N+nc+c]"
_UPSTREAM_SOURCE = _GATHER_MM_SOURCE.replace(
    "weight[((size_t)e*N+nc+c)*D+kk+d]", _TRANSPOSED_WEIGHT
)
_DX_SOURCE = _DOWN_SOURCE.replace(
    "weight[((size_t)e*N+nc+c)*D+kk+d]", _TRANSPOSED_WEIGHT
)
_DOWN_DW_SOURCE = _DW_SOURCE.replace(
    "grad[(size_t)(rr+c)*N+nr+r]",
    "T(grad[(size_t)(order[rr+c]/K)*N+nr+r]*T(weights[order[rr+c]]))",
).replace("x[(size_t)(order[rr+r]/K)*D+dc+c]", "x[(size_t)(rr+r)*D+dc+c]")

_DOWN_SCALE_VJP = r"""
    uint tid=thread_position_in_grid.x;
    uint s=thread_position_in_grid.y;
    uint route=order[s];
    float a=0.0f;
    T w=T(weights[route]);
    for (uint d=tid;d<D;d+=128) {
        T u=upstream[(size_t)s*D+d];
        dx[(size_t)s*D+d]=T(u*w);
        a+=float(T(u*x[(size_t)s*D+d]));
    }
    a=simd_sum(a);
    threadgroup float sums[4];
    if (tid%32==0) sums[tid/32]=a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) dw[route]=float(T((sums[0]+sums[1])+(sums[2]+sums[3])));
"""

# Native gather_mm already gives us sorted route rows.  The combine kernel
# consumes the inverse permutation (original route -> sorted row), so build
# only that vector instead of the full expert offsets/tile metadata.
_INVERSE_SOURCE = r"""
    uint gid=thread_position_in_grid.x;
    if (gid < order_shape[0]) inverse[order[gid]]=gid;
"""


@lru_cache(None)
def _kernels():
    def make(name, inputs, outputs, source, mma=False, atomic=False):
        return mx.fast.metal_kernel(name=name, input_names=inputs,
                                    output_names=outputs, source=source,
                                    header=_HEADER if mma else "", atomic_outputs=atomic)
    return (
        make("moe_route_meta", ["order", "experts", "dims"],
             ["offsets", "tile_offsets", "inverse"], _META_SOURCE),
        make("moe_gather_gate_up", ["x", "weight", "order", "offsets", "tile_offsets"],
             ["out"], _GATHER_MM_SOURCE, True),
        make("moe_gather_weight_vjp", ["x", "grad", "order", "offsets"],
             ["dw"], _DW_SOURCE, True),
        make("moe_inverse_reduce", ["g", "inverse"], ["out"], _REDUCE_SOURCE),
        make("moe_route_combine", ["y", "weights", "inverse"], ["out"], _COMBINE_SOURCE),
        make("moe_route_combine_vjp", ["y", "weights", "order", "g"],
             ["dy", "dw"], _COMBINE_VJP_SOURCE),
        make("moe_down_write", ["x", "weight", "order", "offsets", "tile_offsets", "weights"],
             ["out"], _DOWN_SOURCE, True, True),
        make("moe_down_upstream", ["x", "weight", "order", "offsets", "tile_offsets"],
             ["out"], _UPSTREAM_SOURCE, True),
        make("moe_down_weight_vjp", ["x", "grad", "order", "offsets", "weights"],
             ["dw"], _DOWN_DW_SOURCE, True),
        make("moe_gate_input_vjp", ["x", "weight", "order", "offsets", "tile_offsets", "weights"],
             ["out"], _DX_SOURCE, True, True),
        make("moe_down_scale_vjp", ["x", "upstream", "weights", "order"],
             ["dx", "dw"], _DOWN_SCALE_VJP),
        make("moe_route_inverse", ["order"], ["inverse"], _INVERSE_SOURCE),
    )


def enabled_for(x, n_experts, width):
    return (_ENABLED and mx.default_device() == mx.gpu
            and x.dtype in (mx.bfloat16, mx.float16)
            and x.shape[-1] % 32 == 0 and width % 32 == 0
            and 0 < n_experts <= 128)


def combine_enabled_for(x, k=None):
    """Whether native gather_mm output should use the fused route combine.

    This flag intentionally does not consult ``VIBY_MOE_KERNEL``: the native
    two-GEMM path and the route-combine epilogue are separate experiments.
    """
    return (_COMBINE_ENABLED and mx.default_device() == mx.gpu
            and x.dtype in (mx.bfloat16, mx.float16)
            and x.shape[-1] > 0 and (k is None or k > 0))


def route_metadata(order, experts, n_experts):
    return _kernels()[0](
        inputs=[order, experts, mx.array([n_experts], mx.uint32)],
        grid=(((order.size+127)//128)*128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n_experts+1,), (n_experts+1,), order.shape],
        output_dtypes=[mx.int32]*3,
    )


def route_inverse(order):
    """Return original-route -> sorted-row permutation for native gather_mm."""
    return _kernels()[-1](
        inputs=[order],
        grid=(((order.size + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[order.shape], output_dtypes=[mx.int32],
    )[0]


@lru_cache(None)
def _linear_op(k):
    forward, weight_vjp = _kernels()[1:3]
    input_vjp = _kernels()[9]

    @mx.custom_function
    def op(x, weight, order, experts, offsets, tile_offsets, inverse):
        g, n, d, e = order.size, weight.shape[1], x.shape[-1], weight.shape[0]
        return forward(
            inputs=[x, weight, order, offsets, tile_offsets],
            template=[("T", x.dtype), ("D", d), ("N", n), ("K", k)],
            grid=(((g+31)//32+e)*128, (n+63)//64, 1), threadgroup=(128, 1, 1),
            output_shapes=[(g, n)], output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def vjp(primals, cotangent, output):
        x, weight, order, experts, offsets, tile_offsets, inverse = primals
        g, n, d, e = order.size, weight.shape[1], x.shape[-1], weight.shape[0]
        cotangent = cotangent.astype(x.dtype)
        dx = input_vjp(
            inputs=[cotangent, weight, order, offsets, tile_offsets, mx.ones((g,), mx.float32)],
            template=[("T", x.dtype), ("D", n), ("N", d), ("K", k)],
            grid=(((g+31)//32+e)*128, (d+63)//64, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape], output_dtypes=[mx.float32], init_value=0,
        )[0].astype(x.dtype)
        dw = weight_vjp(
            inputs=[x, cotangent, order, offsets],
            template=[("T", x.dtype), ("D", d), ("N", n), ("K", k)],
            grid=(((n+31)//32)*128, d//32, e), threadgroup=(128, 1, 1),
            output_shapes=[weight.shape], output_dtypes=[weight.dtype],
        )[0]
        # MLX 0.32.2 drops None leaves while flattening custom-VJP results.
        # Keep one array placeholder for every primal, including metadata.
        return (
            dx, dw,
            mx.zeros_like(order), mx.zeros_like(experts),
            mx.zeros_like(offsets), mx.zeros_like(tile_offsets),
            mx.zeros_like(inverse),
        )

    return op


def gather_gate_up(x, weight, order, experts, metadata, k):
    offsets, tile_offsets, inverse = metadata
    return _linear_op(k)(x, weight, order, experts, offsets, tile_offsets, inverse)


@lru_cache(None)
def _combine_op(k):
    forward, backward = _kernels()[4:6]

    @mx.custom_function
    def op(y, weights, order, inverse):
        d, m = y.shape[-1], order.size//k
        return forward(
            inputs=[y, weights, inverse], template=[("T", y.dtype), ("D", d), ("K", k)],
            grid=(m*d, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(m, d)], output_dtypes=[y.dtype],
        )[0]

    @op.vjp
    def vjp(primals, cotangent, output):
        y, weights, order, inverse = primals
        dy, dw = backward(
            inputs=[y, weights, order, cotangent],
            template=[("T", y.dtype), ("D", y.shape[-1]), ("K", k)],
            grid=(128, order.size, 1), threadgroup=(128, 1, 1),
            output_shapes=[y.shape, weights.shape], output_dtypes=[y.dtype, mx.float32],
        )
        # Keep placeholders for integer metadata; a None before a trainable
        # input can shift argnums after tree flattening on MLX 0.32.2.
        return (
            dy, dw.astype(weights.dtype),
            mx.zeros_like(order), mx.zeros_like(inverse),
        )

    return op


def combine_routes(y, weights, order, inverse, k):
    return _combine_op(k)(y, weights, order, inverse)


@lru_cache(None)
def _down_op(k):
    forward, upstream, weight_vjp = _kernels()[6:9]
    scale_vjp = _kernels()[10]

    @mx.custom_function
    def op(x, weight, weights, order, offsets, tile_offsets):
        g, d, n, e = order.size, x.shape[-1], weight.shape[1], weight.shape[0]
        return forward(
            inputs=[x, weight, order, offsets, tile_offsets, weights],
            template=[("T", x.dtype), ("D", d), ("N", n), ("K", k)],
            grid=(((g+31)//32+e)*128, (n+63)//64, 1), threadgroup=(128, 1, 1),
            output_shapes=[(g//k, n)], output_dtypes=[mx.float32], init_value=0,
        )[0].astype(x.dtype)

    @op.vjp
    def vjp(primals, cotangent, output):
        x, weight, weights, order, offsets, tile_offsets = primals
        g, d, n, e = order.size, x.shape[-1], weight.shape[1], weight.shape[0]
        # For each route, u = g[token] @ W_down. Then dx=w*u and
        # d(route_weight)=dot(x,u). This moves the latter dot from D to I;
        # only the smaller [G,I] tensor is materialized.
        u = upstream(
            inputs=[cotangent, weight, order, offsets, tile_offsets],
            template=[("T", x.dtype), ("D", n), ("N", d), ("K", k)],
            grid=(((g+31)//32+e)*128, (d+63)//64, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape], output_dtypes=[x.dtype],
        )[0]
        dx, dweights = scale_vjp(
            inputs=[x, u, weights, order], template=[("T", x.dtype), ("D", d)],
            grid=(128, g, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape, weights.shape], output_dtypes=[x.dtype, mx.float32],
        )
        dw = weight_vjp(
            inputs=[x, cotangent, order, offsets, weights],
            template=[("T", x.dtype), ("D", d), ("N", n), ("K", k)],
            grid=(((n+31)//32)*128, d//32, e), threadgroup=(128, 1, 1),
            output_shapes=[weight.shape], output_dtypes=[weight.dtype],
        )[0]
        return (
            dx, dw, dweights.astype(weights.dtype),
            mx.zeros_like(order), mx.zeros_like(offsets),
            mx.zeros_like(tile_offsets),
        )

    return op


def down_project_routes(x, weight, weights, order, metadata, k):
    return _down_op(k)(x, weight, weights, order, metadata[0], metadata[1])


@lru_cache(None)
def prewarm_moe(dim, width, k, dtype=mx.bfloat16):
    """JIT materialization only, before model compile; no comparison/timing."""
    x = mx.zeros((1, dim), dtype)
    if not enabled_for(x, 1, width):
        return
    weight = mx.zeros((1, width, dim), dtype)
    order = mx.arange(k, dtype=mx.int32)
    experts = mx.zeros((k,), mx.int32)
    meta = route_metadata(order, experts, 1)
    linear = _linear_op(k)
    args = (x, weight, order, experts, *meta)
    out = linear(*args)
    _, grads = mx.vjp(lambda a, b: linear(a, b, *args[2:]), [x, weight],
                      [mx.ones((k, width), dtype)])
    y = mx.zeros((k, dim), dtype)
    weights = mx.ones((k,), mx.float32)
    combine = _combine_op(k)
    result = combine(y, weights, order, meta[2])
    _, dg = mx.vjp(lambda a, b: combine(a, b, order, meta[2]), [y, weights],
                   [mx.ones((1, dim), dtype)])
    hidden = mx.zeros((k, width//2), dtype)
    down_w = mx.zeros((1, dim, width//2), dtype)
    down = _down_op(k)
    down_out, down_grads = mx.vjp(
        lambda a, b, c: down(a, b, c, order, meta[0], meta[1]),
        [hidden, down_w, weights], [mx.ones((1, dim), dtype)],
    )
    mx.eval(out, grads, result, dg, down_out, down_grads)


@lru_cache(None)
def prewarm_route_combine(dim, k, dtype=mx.bfloat16):
    """Materialize native gather_mm route combine and its custom VJP.

    This is separate from :func:`prewarm_moe` because ``VIBY_MOE_KERNEL=0``
    intentionally leaves the hand-written GEMMs disabled while P2 combine is
    independently selectable.
    """
    y = mx.zeros((k, dim), dtype)
    if not combine_enabled_for(y, k):
        return
    order = mx.arange(k, dtype=mx.int32)
    inverse = route_inverse(order)
    weights = mx.ones((k,), mx.float32)
    combine = _combine_op(k)
    out, grads = mx.vjp(
        lambda a, b: combine(a, b, order, inverse), [y, weights],
        [mx.ones((1, dim), dtype)],
    )
    mx.eval(inverse, out, grads)
