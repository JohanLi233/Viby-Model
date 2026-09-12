"""Opt-in FP32 training fusion of mHC splitting and 4x4 Sinkhorn.

RMSNorm/projection stay native. The forward preserves multiply/add rounding,
sigmoid evaluation and Sinkhorn's four-term sum order. Backward reuses the
epsilon-aware Sinkhorn VJP and fuses sigmoid/affine derivatives, returning all
three parameter gradient leaves. Saved coefficients are tiny compared with the
hidden streams, but this is a performance candidate until full-window timing.
Coefficient parameter leaves may be FP32/BF16/FP16: native dtype promotion with
the FP32 projection output is preserved, and their gradients return that dtype.
"""

import os
from functools import lru_cache

import mlx.core as mx


_TRAIN_FUSION = os.environ.get("VIBY_HC_TRAIN_FUSION", "0") != "0"


def enabled_for(mixes, scale, base, hc_mult, training):
    return (
        _TRAIN_FUSION
        and training
        and hc_mult == 4
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and mixes.ndim == 3
        and mixes.shape[-1] == 24
        and mixes.size > 0
        and mixes.dtype == mx.float32
        and scale.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and base.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and scale.shape == (3,)
        and base.shape == (24,)
    )


_FORWARD = r"""
    uint token=thread_position_in_grid.x, n=dims[0];
    if (token>=n) return;
    for (uint j=0;j<8;++j) {
        volatile float product=mixes[(size_t)token*24+j]*float(scale[j/4]);
        volatile float z=product+float(base[j]);
        float small=1.0f/(1.0f+metal::precise::exp(metal::abs(z)));
        volatile float sig=z<0.0f ? small : 1.0f-small;
        sigmoid[(size_t)token*8+j]=sig;
        if (j<4) pre[(size_t)token*4+j]=sig+epsilon[0];
        else post[(size_t)token*4+j-4]=2.0f*sig;
    }
    float c[16], z[16];
    for (uint j=0;j<16;++j) {
        volatile float product=mixes[(size_t)token*24+8+j]*float(scale[2]);
        z[j]=product+float(base[8+j]);
        logits[(size_t)token*16+j]=z[j];
    }
    for (uint row=0;row<4;++row) {
        float maximum=z[row*4];
        for (uint col=1;col<4;++col) maximum=max(maximum,z[row*4+col]);
        for (uint col=0;col<4;++col) c[row*4+col]=exp(z[row*4+col]-maximum);
    }
    for (uint step=0;step<STEPS;++step) {
        for (uint a=0;a<4;++a) {
            float denominator=0.0f;
            for (uint b=0;b<4;++b) denominator+=c[step%2==0 ? a*4+b : b*4+a];
            denominator+=epsilon[0];
            for (uint b=0;b<4;++b) c[step%2==0 ? a*4+b : b*4+a]/=denominator;
        }
    }
    for (uint j=0;j<16;++j) comb[(size_t)token*16+j]=c[j];
"""


_BACKWARD = r"""
    uint token=thread_position_in_grid.x, n=dims[0];
    if (token>=n) return;
    float ds[3]={0.0f,0.0f,0.0f};
    for (uint j=0;j<24;++j) {
        float gradient;
        uint group=j<8 ? j/4 : 2;
        if (j<8) {
            float sig=sigmoid[(size_t)token*8+j];
            float upstream=j<4 ? gpre[(size_t)token*4+j] : 2.0f*gpost[(size_t)token*4+j-4];
            gradient=(sig*(1.0f-sig))*upstream;
        } else gradient=gcomb[(size_t)token*16+j-8];
        dmixes[(size_t)token*24+j]=gradient*float(scale[group]);
        dbase_rows[(size_t)token*24+j]=gradient;
        ds[group]+=gradient*mixes[(size_t)token*24+j];
    }
    for (uint j=0;j<3;++j) dscale_rows[(size_t)token*3+j]=ds[j];
"""


@lru_cache(None)
def _kernels():
    forward = mx.fast.metal_kernel(
        name="hc4_train_split_sinkhorn",
        input_names=["mixes", "scale", "base", "epsilon", "dims"],
        output_names=["pre", "post", "comb", "logits", "sigmoid"],
        source=_FORWARD,
    )
    backward = mx.fast.metal_kernel(
        name="hc4_train_split_affine_vjp",
        input_names=["mixes", "scale", "sigmoid", "gpre", "gpost", "gcomb", "dims"],
        output_names=["dmixes", "dscale_rows", "dbase_rows"],
        source=_BACKWARD,
    )
    return forward, backward


@lru_cache(None)
def _operation(iters, eps):
    from .sinkhorn_fused import sinkhorn_fused

    forward, backward = _kernels()

    @mx.custom_function
    def op(mixes, scale, base):
        leading = mixes.shape[:-1]
        n = mixes.size // 24
        # The last two arrays are private VJP state, omitted by the public API.
        return tuple(
            forward(
                inputs=[
                    mixes,
                    scale,
                    base,
                    mx.array([eps], mx.float32),
                    mx.array([n], mx.uint32),
                ],
                template=[("STEPS", 2 * max(1, iters))],
                grid=(n, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[
                    (*leading, 4),
                    (*leading, 4),
                    (*leading, 4, 4),
                    (*leading, 4, 4),
                    (*leading, 8),
                ],
                output_dtypes=[mx.float32] * 5,
            )
        )

    @op.vjp
    def vjp(primals, cotangent, output):
        mixes, scale, base = primals
        gpre, gpost, gcomb, _, _ = cotangent
        _, _, _, logits, sigmoid = output
        # This reuses the established epsilon/max/tie-aware comb derivative;
        # its private forward is dead, and only tape recomputation + VJP run.
        _, (grad_comb,) = mx.vjp(
            lambda c: sinkhorn_fused(c, iters, eps), [logits], [gcomb]
        )
        n = mixes.size // 24
        dm, ds_rows, db_rows = backward(
            inputs=[
                mixes,
                scale,
                sigmoid,
                gpre,
                gpost,
                grad_comb,
                mx.array([n], mx.uint32),
            ],
            grid=(n, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[mixes.shape, (n, 3), (n, 24)],
            output_dtypes=[mx.float32] * 3,
        )
        return (
            dm,
            mx.sum(ds_rows, axis=0).astype(scale.dtype),
            mx.sum(db_rows, axis=0).astype(base.dtype),
        )

    return op


def split_sinkhorn_train(mixes, scale, base, iters, eps):
    """Guarded hc=4 FP32 region; all coefficient parameters remain dynamic."""
    return _operation(iters, eps)(mixes, scale, base)[:3]
