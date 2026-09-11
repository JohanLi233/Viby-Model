"""FP32 4x4 Sinkhorn with an epsilon-aware VJP.

Forward uses one dispatch; backward uses recomputation plus one VJP dispatch.
Each GPU thread owns one matrix. Training intermediates use [step, entry, token]
layout for coalesced writes. Other sizes/dtypes retain the reference path.
Set VIBY_SINKHORN_KERNEL=0 before launch to disable.
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_SINKHORN_KERNEL", "1") != "0"


def sinkhorn_ref(x, iters, eps):
    c = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
    c = c / (mx.sum(c, axis=-1, keepdims=True) + eps)
    c = c / (mx.sum(c, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (mx.sum(c, axis=-1, keepdims=True) + eps)
        c = c / (mx.sum(c, axis=-2, keepdims=True) + eps)
    return c


_FWD = r"""
    uint t = thread_position_in_grid.x;
    uint n = dims[0];
    if (t >= n) return;
    float c[16];
    for (uint r = 0; r < 4; ++r) {
        float m = x[t*16+r*4];
        for (uint j = 1; j < 4; ++j) m = max(m, x[t*16+r*4+j]);
        for (uint j = 0; j < 4; ++j) c[r*4+j] = exp(x[t*16+r*4+j]-m);
    }
    for (uint s = 0; s < STEPS; ++s) {
        for (uint a = 0; a < 4; ++a) {
            float d = 0.0f;
            for (uint b = 0; b < 4; ++b) d += c[(s%2 == 0) ? a*4+b : b*4+a];
            d += epsilon[0];
            if (SAVE) den[(s*4+a)*n+t] = d;
            for (uint b = 0; b < 4; ++b) {
                uint k = (s%2 == 0) ? a*4+b : b*4+a;
                c[k] /= d;
                if (SAVE) hist[(s*16+k)*n+t] = c[k];
            }
        }
    }
    for (uint k = 0; k < 16; ++k) out[t*16+k] = c[k];
"""

_BWD = r"""
    uint t = thread_position_in_grid.x;
    uint n = dims[0];
    if (t >= n) return;
    float v[16];
    for (uint k = 0; k < 16; ++k) v[k] = g[t*16+k];
    for (int s = STEPS-1; s >= 0; --s) {
        for (uint a = 0; a < 4; ++a) {
            float dot = 0.0f;
            for (uint b = 0; b < 4; ++b) {
                uint k = (s%2 == 0) ? a*4+b : b*4+a;
                dot += v[k] * hist[(s*16+k)*n+t];
            }
            float d = den[(s*4+a)*n+t];
            for (uint b = 0; b < 4; ++b) {
                uint k = (s%2 == 0) ? a*4+b : b*4+a;
                v[k] = (v[k] - dot) / d;
            }
        }
    }
    // exp(x-max(x)): retain max's derivative, including ties. With epsilon
    // normalization this correction is small but not identically zero.
    for (uint r = 0; r < 4; ++r) {
        float m = x[t*16+r*4];
        for (uint j = 1; j < 4; ++j) m = max(m, x[t*16+r*4+j]);
        float sum = 0.0f;
        float count = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            uint k = r*4+j;
            v[k] *= exp(x[t*16+k]-m);
            sum += v[k];
            count += (x[t*16+k] == m) ? 1.0f : 0.0f;
        }
        for (uint j = 0; j < 4; ++j) {
            uint k = r*4+j;
            dx[t*16+k] = v[k] - ((x[t*16+k] == m) ? sum/count : 0.0f);
        }
    }
"""


@lru_cache(None)
def _operation(iters, eps):
    steps = 2 * max(1, iters)
    fwd = mx.fast.metal_kernel(
        name="sinkhorn4_fwd", input_names=["x", "epsilon", "dims"],
        output_names=["out", "hist", "den"], source=_FWD,
    )
    bwd = mx.fast.metal_kernel(
        name="sinkhorn4_bwd", input_names=["x", "g", "hist", "den", "dims"],
        output_names=["dx"], source=_BWD,
    )

    def forward(x, save):
        n = x.size // 16
        return fwd(
            inputs=[x, mx.array([eps], mx.float32), mx.array([n], mx.uint32)],
            template=[("STEPS", steps), ("SAVE", save)],
            grid=(n, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape, (steps, 16, n) if save else (1,),
                           (steps, 4, n) if save else (1,)],
            output_dtypes=[mx.float32]*3,
        )

    @mx.custom_function
    def op(x):
        return forward(x, False)[0]

    @op.vjp
    def vjp(primals, cotangent, output):
        x = primals
        # Recompute tiny matrices only for backward; inference writes no tape.
        _, hist, den = forward(x, True)
        (dx,) = bwd(
            inputs=[x, cotangent, hist, den, mx.array([x.size // 16], mx.uint32)], template=[("STEPS", steps)],
            grid=(x.size // 16, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[x.shape], output_dtypes=[mx.float32],
        )
        return dx

    return op


def sinkhorn_fused(x, iters, eps):
    if (not _ENABLED or x.dtype != mx.float32 or x.shape[-2:] != (4, 4)
            or x.size == 0 or mx.default_device() != mx.gpu):
        return sinkhorn_ref(x, iters, eps)
    return _operation(iters, eps)(x)


@lru_cache(None)
def prewarm_sinkhorn(iters=20, eps=1e-6):
    if not _ENABLED or mx.default_device() != mx.gpu:
        return
    x = mx.zeros((1, 4, 4), mx.float32)
    op = _operation(iters, eps)
    mx.eval(op(x), mx.grad(lambda a: mx.sum(op(a)))(x))
