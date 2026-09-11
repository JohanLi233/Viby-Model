"""Native route gather with a token-owned, non-atomic input VJP.

Keep the materialized sorted lhs: passing *both* lhs_indices and rhs_indices
would give up the single-index sorted gather_mm fast path. Only the gather's
backward changes. The inverse permutation can be shared with route combine.
"""
import os
import operator
from functools import lru_cache

import mlx.core as mx

# Promote only after Metal parity and full-window ABBA measurements.
_ENABLED = os.environ.get("VIBY_MOE_GATHER_VJP", "0") != "0"
_THREADS = 128


def enabled_for(x, routes):
    return (_ENABLED and mx.default_device() == mx.gpu and mx.metal.is_available() and x.ndim >= 2
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and x.size > 0 and routes > 0)


@lru_cache(None)
def _backward_kernel():
    return mx.fast.metal_kernel(
        name="moe_route_gather_token_vjp",
        input_names=["g", "inverse"], output_names=["dx"],
        source=r"""
        uint d = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        if (d >= D) return;
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            size_t row = size_t(inverse[(size_t)token * K + k]);
            acc += float(g[row * D + d]);
        }
        dx[(size_t)token * D + d] = T(acc);
        """,
    )


@lru_cache(maxsize=32)
def _op(routes):
    @mx.custom_function
    def gather(x, order, inverse):
        # This is deliberately the original gather, including its output dtype.
        return x[(mx.stop_gradient(order) // routes).astype(mx.int32)]

    @gather.vjp
    def vjp(primals, cotangent, output):
        x, order, inverse = primals
        m, d = x.shape
        (dx,) = _backward_kernel()(
            inputs=[cotangent.astype(x.dtype), inverse],
            template=[("T", x.dtype), ("D", d), ("K", routes)],
            grid=(((d + _THREADS - 1) // _THREADS) * _THREADS, m, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[x.shape], output_dtypes=[x.dtype],
        )
        # MLX custom VJPs need one array leaf for EVERY array primal.
        return dx, mx.zeros_like(order), mx.zeros_like(inverse)

    return gather


def gather_routes(x, order, inverse, routes):
    """Gather [M,D] -> [M*K,D] with a reusable inverse permutation.

    Preconditions: order is a permutation of range(M*K), and
    inverse[order[i]] == i. These values are produced by route_inverse, not
    checked on the host. Repeated token/expert assignments remain occurrences;
    no route is deduplicated. Empty/CPU/unsupported dtype calls use native take.
    """
    routes = operator.index(routes)
    if routes <= 0:
        raise ValueError("routes must be positive")
    if x.ndim != 2 or order.ndim != 1 or inverse.ndim != 1:
        raise ValueError("expected x[M,D], order[M*K], inverse[M*K]")
    if order.size != x.shape[0] * routes or inverse.shape != order.shape:
        raise ValueError("route permutation size does not match M*K")
    if order.dtype not in (mx.int32, mx.uint32) or inverse.dtype not in (mx.int32, mx.uint32):
        raise ValueError("route permutations must contain int32 or uint32 indices")
    order, inverse = mx.stop_gradient(order), mx.stop_gradient(inverse)
    if x.size == 0:
        # Native empty take has a broken scatter adjoint in MLX 0.32.2.
        # A reshape has the same empty value and a valid empty adjoint.
        return x.reshape(order.size, x.shape[1])
    if not enabled_for(x, routes):
        return x[(order // routes).astype(mx.int32)]
    return _op(routes)(x, order, inverse)
