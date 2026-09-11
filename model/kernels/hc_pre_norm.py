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


@lru_cache(None)
def _kernels():
    return (
        mx.fast.metal_kernel(name="hc_pre_rms_fwd", input_names=["x", "pre", "weight", "epsilon"],
                             output_names=["y", "h", "rstd"], source=_FWD),
        mx.fast.metal_kernel(name="hc_pre_rms_bwd", input_names=["x", "pre", "weight", "g", "h", "rstd"],
                             output_names=["dx", "dpre", "dweight"], source=_BWD),
    )


def enabled_for(x):
    return (_ENABLED and mx.default_device() == mx.gpu
            and x.dtype in (mx.bfloat16, mx.float16) and x.shape[-2] == 4
            and 0 < x.shape[-1] <= 4096 and x.shape[-1] % 32 == 0)


@lru_cache(None)
def _operation(eps):
    forward, backward = _kernels()

    @mx.custom_function
    def op(x, pre, weight):
        n, _, d = x.shape
        y, h, r = forward(
            inputs=[x, pre, weight, mx.array([eps], mx.float32)],
            template=[("T", x.dtype), ("D", d)], grid=(128, n, 1), threadgroup=(128, 1, 1),
            output_shapes=[(n, d), (n, d), (n,)], output_dtypes=[x.dtype, x.dtype, mx.float32],
        )
        return y, h, r

    @op.vjp
    def vjp(primals, cotangent, output):
        x, pre, weight = primals
        g, _, _ = cotangent
        _, h, r = output
        n, _, d = x.shape
        dx, dp, dw = backward(
            inputs=[x, pre, weight, g, h, r], template=[("T", x.dtype), ("D", d)],
            grid=(128, n, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape, pre.shape, (n, d)],
            output_dtypes=[x.dtype, mx.float32, weight.dtype],
        )
        return dx, dp.astype(pre.dtype), mx.sum(dw, axis=0).astype(weight.dtype)

    return op


def hc_pre_norm(x, pre, weight, eps):
    shape, d = x.shape[:-2], x.shape[-1]
    out, _, _ = _operation(eps)(x.reshape(-1, 4, d), pre.reshape(-1, 4).astype(mx.float32),
                               weight.astype(x.dtype))
    return out.reshape(*shape, d)


@lru_cache(None)
def prewarm_hc_pre_norm(dim, eps, dtype=mx.bfloat16):
    """Eager JIT materialization only; no numerical checks or timing."""
    x = mx.zeros((1, 4, dim), dtype)
    if not enabled_for(x):
        return
    p = mx.ones((1, 4), mx.float32)
    w = mx.ones((dim,), dtype)
    out, grads = mx.vjp(lambda a, b, c: hc_pre_norm(a, b, c, eps), [x, p, w],
                        [mx.ones((1, dim), dtype)])
    mx.eval(out, grads)
