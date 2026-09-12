"""Fuse four-stream hc_pre and its following RMSNorm, including the VJP.

Preserves the existing low-precision product/square/cast locations. Each token
uses one threadgroup; the RMS statistic is accumulated in FP32. The backward
also fuses stream expansion and the per-token coefficient reductions.
VIBY_HC_PRE_NORM_KERNEL=0 restores the composed MLX graph.
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_HC_PRE_NORM_KERNEL", "1") != "0"
# §7：权梯度 partial 的分组 + 两级非 atomic 归约。M>=1024 时每 TG 顺序处理
# 4 个 token（RT=4），partial 行数从 M 降到 ceil(M/4)；小 M 用 RT=1，避免白减
# 并行 TG 数。`VIBY_HC_GROUPED_DW=0` 回到原来的单级 `mx.sum(dw, axis=0)`。
_GROUPED_DW = os.environ.get("VIBY_HC_GROUPED_DW", "1") != "0"

_FWD = r"""
    uint tid=thread_position_in_grid.x, row=thread_position_in_grid.y;
    threadgroup T values[D];
    threadgroup float partial[4];
    float sum=0.0f;
    for (uint d=tid;d<D;d+=128) {
        float a=0.0f;
        for (uint c=0;c<4;++c) a+=float(T(T(pre[row*4+c])*x[((size_t)row*4+c)*D+d]));
        T v=T(a);
        values[d]=v;
        h[(size_t)row*D+d]=v;
        sum+=float(T(v*v));
    }
    sum=simd_sum(sum);
    if (tid%32==0) partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=((partial[0]+partial[1])+(partial[2]+partial[3]))/float(D);
    float r=rsqrt(var+epsilon[0]);
    if (tid==0) rstd[row]=r;
    for (uint d=tid;d<D;d+=128) y[(size_t)row*D+d]=T(T(values[d]*T(r))*weight[d]);
"""

_BWD = r"""
    uint tid=thread_position_in_grid.x, row=thread_position_in_grid.y;
    threadgroup float partial[4];
    threadgroup float coeff[4*4];
    float dr=0.0f;
    float r=rstd[row];
    for (uint d=tid;d<D;d+=128) {
        size_t pos=(size_t)row*D+d;
        T gh=T(g[pos]*weight[d]);
        dr+=float(T(gh*h[pos]));
        dweight[pos]=T(g[pos]*T(h[pos]*T(r)));
    }
    dr=simd_sum(dr);
    if (tid%32==0) partial[tid/32]=dr;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dr=float(T((partial[0]+partial[1])+(partial[2]+partial[3])));
    T dv=T((-0.5f*r*r*r*dr)/float(D));
    float dp[4]={0.0f,0.0f,0.0f,0.0f};
    for (uint d=tid;d<D;d+=128) {
        size_t pos=(size_t)row*D+d;
        T gh=T(g[pos]*weight[d]);
        T v=T(dv*h[pos]);
        T dh=T(T(gh*T(r))+T(v+v));
        for (uint c=0;c<4;++c) {
            size_t p=((size_t)row*4+c)*D+d;
            dx[p]=T(dh*T(pre[row*4+c]));
            dp[c]+=float(T(dh*x[p]));
        }
    }
    for (uint c=0;c<4;++c) {
        float a=simd_sum(dp[c]);
        if (tid%32==0) coeff[c*4+tid/32]=a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<4) dpre[row*4+tid]=float(T((coeff[tid*4]+coeff[tid*4+1])+(coeff[tid*4+2]+coeff[tid*4+3])));
"""

_BWD_GROUPED = r"""
    uint tid=thread_position_in_grid.x, group=thread_position_in_grid.y;
    threadgroup float partial[4];
    threadgroup float coeff[4*4];
    float dw_acc[(D+127)/128];
    for (uint r=0;r<(D+127)/128;++r) dw_acc[r]=0.0f;
    for (uint local=0;local<RT;++local) {
        uint row=group*RT+local;
        if (row>=dims[0]) break;
        /*TOKEN_BACKWARD*/
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d=tid;d<D;d+=128) dweight[(size_t)group*D+d]=dw_acc[d/128];
""".replace(
    "/*TOKEN_BACKWARD*/",
    _BWD[_BWD.index("    float dr=") :].replace(
        "dweight[pos]=T(g[pos]*T(h[pos]*T(r)));",
        "dw_acc[d/128]+=float(T(g[pos]*T(h[pos]*T(r))));",
    ),
)


@lru_cache(None)
def _grouped_backward():
    return mx.fast.metal_kernel(
        name="hc_pre_rms_bwd_grouped",
        input_names=["x", "pre", "weight", "g", "h", "rstd", "dims"],
        output_names=["dx", "dpre", "dweight"],
        source=_BWD_GROUPED,
    )


@lru_cache(None)
def _kernels():
    return (
        mx.fast.metal_kernel(
            name="hc_pre_rms_fwd",
            input_names=["x", "pre", "weight", "epsilon"],
            output_names=["y", "h", "rstd"],
            source=_FWD,
        ),
        mx.fast.metal_kernel(
            name="hc_pre_rms_bwd",
            input_names=["x", "pre", "weight", "g", "h", "rstd"],
            output_names=["dx", "dpre", "dweight"],
            source=_BWD,
        ),
    )


def enabled_for(x):
    return (
        _ENABLED
        and mx.default_device() == mx.gpu
        and x.dtype in (mx.bfloat16, mx.float16)
        and x.shape[-2] == 4
        and 0 < x.shape[-1] <= 4096
        and x.shape[-1] % 32 == 0
    )


@lru_cache(None)
def _reduce_kernels():
    """§7.3 两级非 atomic 权梯度归约。"""
    stage1 = mx.fast.metal_kernel(
        name="hc_pre_dw_stage1",
        input_names=["partial", "dims"],
        output_names=["second"],
        source=r"""
    uint tid=thread_position_in_grid.x;
    uint g=thread_position_in_grid.y;
    uint G=dims[0], D=dims[1];
    uint dd=tid;
    if (dd>=D || g*64>=G) return;
    float acc=0.0f;
    for (uint r=0;r<64 && g*64+r<G;++r) acc+=partial[(size_t)(g*64+r)*D+dd];
    second[(size_t)g*D+dd]=acc;
""",
    )
    stage2 = mx.fast.metal_kernel(
        name="hc_pre_dw_stage2",
        input_names=["second", "dims"],
        output_names=["dweight"],
        source=r"""
    uint dd=thread_position_in_grid.x;
    uint G2=dims[0], D=dims[1];
    if (dd>=D) return;
    float acc=0.0f;
    for (uint g=0;g<G2;++g) acc+=second[(size_t)g*D+dd];
    dweight[dd]=acc;
""",
    )
    return stage1, stage2


@lru_cache(None)
def _operation(eps):
    forward, backward = _kernels()
    stage1, stage2 = _reduce_kernels()

    @mx.custom_function
    def op(x, pre, weight):
        n, _, d = x.shape
        y, h, r = forward(
            inputs=[x, pre, weight, mx.array([eps], mx.float32)],
            template=[("T", x.dtype), ("D", d)],
            grid=(128, n, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n, d), (n, d), (n,)],
            output_dtypes=[x.dtype, x.dtype, mx.float32],
        )
        return y, h, r

    @op.vjp
    def vjp(primals, cotangent, output):
        x, pre, weight = primals
        g, _, _ = cotangent
        _, h, r = output
        n, _, d = x.shape
        if _GROUPED_DW:
            rt = 4 if n >= 1024 else 1
            groups = (n + rt - 1) // rt
            dx, dp, dw = _grouped_backward()(
                inputs=[x, pre, weight, g, h, r, mx.array([n], mx.uint32)],
                template=[("T", x.dtype), ("D", d), ("RT", rt)],
                grid=(128, groups, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[x.shape, pre.shape, (groups, d)],
                output_dtypes=[x.dtype, mx.float32, mx.float32],
            )
            rows = max(1, (groups + 63) // 64)
            second = stage1(
                inputs=[dw, mx.array([groups, d], mx.uint32)],
                grid=(d, rows, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(rows, d)],
                output_dtypes=[mx.float32],
            )[0]
            dweight = stage2(
                inputs=[second, mx.array([rows, d], mx.uint32)],
                grid=(d, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(d,)],
                output_dtypes=[mx.float32],
            )[0]
            return dx, dp.astype(pre.dtype), dweight.astype(weight.dtype)
        dx, dp, dw = backward(
            inputs=[x, pre, weight, g, h, r],
            template=[("T", x.dtype), ("D", d)],
            grid=(128, n, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[x.shape, pre.shape, (n, d)],
            output_dtypes=[x.dtype, mx.float32, weight.dtype],
        )
        return dx, dp.astype(pre.dtype), mx.sum(dw, axis=0).astype(weight.dtype)

    return op


def hc_pre_norm(x, pre, weight, eps):
    shape, d = x.shape[:-2], x.shape[-1]
    out, _, _ = _operation(eps)(
        x.reshape(-1, 4, d),
        pre.reshape(-1, 4).astype(mx.float32),
        weight.astype(x.dtype),
    )
    return out.reshape(*shape, d)


@lru_cache(None)
def prewarm_hc_pre_norm(dim, eps, dtype=mx.bfloat16):
    """Eager JIT materialization only; no numerical checks or timing."""
    x = mx.zeros((1, 4, dim), dtype)
    if not enabled_for(x):
        return
    p = mx.ones((1, 4), mx.float32)
    w = mx.ones((dim,), dtype)
    out, grads = mx.vjp(
        lambda a, b, c: hc_pre_norm(a, b, c, eps), [x, p, w], [mx.ones((1, dim), dtype)]
    )
    mx.eval(out, grads)
