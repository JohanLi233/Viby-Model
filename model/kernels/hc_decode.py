"""Decode-only fusion of mHC coefficient splitting and 4x4 Sinkhorn.

The RMS normalization and native projection remain unchanged. One SIMD group
per token handles the three FP32 affine transforms, pre/post sigmoids and all
Sinkhorn normalizations. Explicit product rounding retains the eager FP32
multiply/add boundary; the sums match ``sinkhorn_fused`` in entry order.

VIBY_HC_DECODE_FUSION=1 opts in while numerical/performance gates are pending.
Training and larger token counts retain the original composed implementation.
The custom VJP delegates to that same graph if an eval-mode call is differentiated.
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_HC_DECODE_FUSION", "0") != "0"


def enabled_for(mixes, scale, base, hc_mult, training):
    return (
        _ENABLED
        and not training
        and hc_mult == 4
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and mixes.ndim == 3
        and mixes.shape[-1] == 24
        and 0 < mixes.shape[0] * mixes.shape[1] <= 8
        and mixes.dtype == scale.dtype == base.dtype == mx.float32
        and scale.shape == (3,)
        and base.shape == (24,)
    )


def _reference(mixes, scale, base, iters, eps):
    from ..hc import hc_split
    from .sinkhorn_fused import sinkhorn_fused

    pre, post, comb = hc_split(mixes, scale, base, 4, eps)
    return pre, post, sinkhorn_fused(comb.reshape(*mixes.shape[:-1], 4, 4), iters, eps)


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="hc4_split_sinkhorn_decode",
        input_names=["mixes", "scale", "base", "epsilon"],
        output_names=["pre", "post", "comb"],
        source=r"""
        uint lane=thread_position_in_threadgroup.x;
        uint token=thread_position_in_grid.y;
        if (lane<8) {
            // A volatile FP32 temporary forbids multiply/add contraction across
            // the original eager materialization boundary.
            volatile float product=mixes[(size_t)token*24+lane]*scale[lane/4];
            volatile float z=product+base[lane];
            // Explicit precise exp matches the native MLX 0.32.2 unary result;
            // custom-kernel default exp differs by ULPs on the decode gate.
            float small=1.0f/(1.0f+metal::precise::exp(metal::abs(z)));
            volatile float sigmoid=z<0.0f ? small : 1.0f-small;
            if (lane<4) pre[(size_t)token*4+lane]=sigmoid+epsilon[0];
            else post[(size_t)token*4+lane-4]=2.0f*sigmoid;
        }
        uint row=lane/4, col=lane%4;
        volatile float product=lane<16 ? mixes[(size_t)token*24+8+lane]*scale[2] : 0.0f;
        float z=lane<16 ? product+base[8+lane] : 0.0f;
        float maximum=simd_shuffle(z,ushort(row*4));
        for (uint j=1;j<4;++j) maximum=max(maximum,simd_shuffle(z,ushort(row*4+j)));
        float value=exp(z-maximum);
        for (uint step=0;step<STEPS;++step) {
            float denominator=0.0f;
            for (uint j=0;j<4;++j)
                denominator+=simd_shuffle(value,ushort(step%2==0 ? row*4+j : j*4+col));
            denominator+=epsilon[0];
            value/=denominator;
        }
        if (lane<16) comb[(size_t)token*16+lane]=value;
        """,
    )


@lru_cache(None)
def _operation(iters, eps):
    @mx.custom_function
    def op(mixes, scale, base):
        leading = mixes.shape[:-1]
        return tuple(
            _kernel()(
                inputs=[mixes, scale, base, mx.array([eps], mx.float32)],
                template=[("STEPS", 2 * max(1, iters))],
                grid=(32, mixes.size // 24, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(*leading, 4), (*leading, 4), (*leading, 4, 4)],
                output_dtypes=[mx.float32] * 3,
            )
        )

    @op.vjp
    def vjp(primals, cotangent, output):
        # Return exactly one array gradient per primal, including scale/base.
        _, gradients = mx.vjp(
            lambda m, s, b: _reference(m, s, b, iters, eps),
            list(primals),
            list(cotangent),
        )
        return tuple(gradients)

    return op


def split_sinkhorn_decode(mixes, scale, base, iters, eps):
    """Run the guarded FP32 hc=4 region with dynamic coefficient parameters."""
    return _operation(iters, eps)(mixes, scale, base)
