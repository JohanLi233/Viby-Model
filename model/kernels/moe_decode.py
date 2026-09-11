"""Shape-cached pure decode MoE region; no model/cache arrays in closures.

The region includes the two selected-expert gather_mm calls, the original
expert-axis contribution sum, shared expert, and final addition. It deliberately
does not capture routing, model mutation, or the ever-growing attention pool.
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_MOE_DECODE_COMPILE", "0") != "0"


def enabled_for(x, training, decode_gather, max_tokens, weights):
    return (_ENABLED and decode_gather and not training
            and mx.default_device() == mx.gpu and mx.metal.is_available() and x.ndim == 3
            # FP32 fusion can contract arithmetic and change output bits.
            # Keep the exact-parity contract by retaining eager FP32 decode.
            and x.dtype in (mx.bfloat16, mx.float16)
            and 0 < x.shape[0] * x.shape[1] <= max_tokens
            and all(w.dtype == x.dtype for w in weights))


@lru_cache(maxsize=32)
def _compiled_region(n_experts, routes, limit, shared_limit):
    # Import only callables/constants. Every mutable weight is a function input.
    from ..moe import expert_act
    from .decode_metadata import combine_selected_experts

    def forward(x, idx, w, gate_up_w, down_w, shared_gate, shared_down, shared_up):
        b, t, d = x.shape
        m = b * t
        exps = mx.stop_gradient(idx).reshape(m * routes).astype(mx.int32)
        tokens = (mx.arange(m * routes) // routes).astype(mx.int32)
        h = mx.gather_mm(
            x.reshape(m, 1, d), gate_up_w.swapaxes(-1, -2),
            lhs_indices=tokens, rhs_indices=exps,
        )
        gate, up = mx.split(h, 2, axis=-1)
        act = expert_act(gate, up, limit)
        y = mx.gather_mm(act, down_w.swapaxes(-1, -2), rhs_indices=exps)
        mixed = combine_selected_experts(
            y, w.reshape(-1), exps, n_experts, routes,
        ).reshape(b, t, d)
        shared = expert_act(x @ shared_gate.T, x @ shared_up.T, shared_limit) @ shared_down.T
        return mixed + shared

    # Shape-aware compilation; stable small-M shapes reuse their graph even
    # when context length grows. Do not use shapeless=True for shape logic.
    return mx.compile(forward)


def compiled_decode(x, idx, w, gate_up_w, down_w, shared_gate, shared_down, shared_up,
                    n_experts, routes, limit, shared_limit):
    """Run the guarded region. Callers retain the original eager fallback."""
    return _compiled_region(n_experts, routes, float(limit), float(shared_limit))(
        x, idx, w, gate_up_w, down_w, shared_gate, shared_down, shared_up,
    )
