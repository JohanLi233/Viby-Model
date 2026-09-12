"""One-dispatch partial RoPE for small single-token batches.

Retains the existing FP32 promotion from the frequency tables, interleaved tail
pairs, inverse sign, and rounding of each product before addition/subtraction.
It is intentionally not ``mx.fast.rope``, whose output dtype differs from this
model's composed graph. The VJP delegates to that original graph.

VIBY_ROPE_DECODE_FUSION=1 opts in while exact-output and timing gates are pending.
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_ROPE_DECODE_FUSION", "0") != "0"


def enabled_for(x, cos, sin, rope_dim):
    if (
        not _ENABLED
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or x.ndim not in (3, 4)
        or x.shape[1] != 1
        or not 0 < x.shape[0] <= 8
        or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or cos.dtype != mx.float32
        or sin.dtype != mx.float32
        or rope_dim <= 0
        or rope_dim > x.shape[-1]
        or rope_dim % 2
        or x.size == 0
    ):
        return False
    # These are the exact frequency broadcast layouts used by Attention.decode.
    suffix = (1, rope_dim // 2) if x.ndim == 3 else (1, 1, rope_dim // 2)
    return all(
        a.ndim == x.ndim and a.shape[0] in (1, x.shape[0]) and a.shape[1:] == suffix
        for a in (cos, sin)
    )


def _reference(x, cos, sin, rope_dim, inverse):
    from ..rope import apply_rope

    if rope_dim == x.shape[-1]:
        return apply_rope(x, cos, sin, inverse)
    head = x[..., : x.shape[-1] - rope_dim]
    tail = x[..., x.shape[-1] - rope_dim :]
    return mx.concatenate([head, apply_rope(tail, cos, sin, inverse)], axis=-1)


@lru_cache(None)
def _kernel():
    return mx.fast.metal_kernel(
        name="partial_rope_decode_fp32",
        input_names=["x", "cosine", "sine"],
        output_names=["out"],
        source=r"""
        uint i=thread_position_in_grid.x;
        if (i>=COUNT) return;
        uint channel=i%D;
        if (channel<D-R) {
            out[i]=float(x[i]);
            return;
        }
        uint row=i/D, batch=row/H;
        uint pair=(channel-(D-R))/2;
        uint offset=row*D+(D-R)+2*pair;
        float real=float(x[offset]), imag=float(x[offset+1]);
        float c=cosine[(CB ? 0 : batch)*(R/2)+pair];
        float s=sine[(SB ? 0 : batch)*(R/2)+pair];
        if (INVERSE) s=-s;
        // Separate eager multiply primitives round before the add/subtract.
        // Volatile retains those boundaries even in an FMA-capable shader.
        if ((channel-(D-R))%2==0) {
            volatile float a=real*c;
            volatile float b=imag*s;
            out[i]=a-b;
        } else {
            volatile float a=real*s;
            volatile float b=imag*c;
            out[i]=a+b;
        }
        """,
    )


@lru_cache(None)
def _operation(rope_dim, inverse):
    @mx.custom_function
    def op(x, cos, sin):
        return _kernel()(
            inputs=[x, cos, sin],
            template=[
                ("D", x.shape[-1]),
                ("R", rope_dim),
                ("H", x.shape[2] if x.ndim == 4 else 1),
                ("CB", cos.shape[0] == 1),
                ("SB", sin.shape[0] == 1),
                ("COUNT", x.size),
                ("INVERSE", inverse),
            ],
            grid=(x.size, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )[0]

    @op.vjp
    def vjp(primals, cotangent, output):
        _, gradients = mx.vjp(
            lambda x, c, s: _reference(x, c, s, rope_dim, inverse),
            list(primals),
            [cotangent],
        )
        return tuple(gradients)

    return op


def rope_partial_decode(x, cos, sin, rope_dim, inverse=False):
    """Run a guarded single-token partial rotation; output retains FP32 dtype."""
    return _operation(rope_dim, inverse)(x, cos, sin)
