"""只跑 dq kernel 若干遍，供 xctrace Metal System Trace 抓时间线。

用法: xcrun xctrace record --template 'Metal System Trace' --time-limit 15s \
        --launch -- uv run experiments/trace_dq.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_fused as af

B, H, T, DK, DV = 12, 8, 1024, 128, 96


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(DK)
    vpad = mx.concatenate([v, mx.zeros((B, H, T, DK - DV), dtype=v.dtype)], axis=-1)
    o = mx.fast.scaled_dot_product_attention(q, k, vpad, scale=scale, mask="causal")[
        ..., :DV
    ]
    delta = (do.astype(mx.float32) * o.astype(mx.float32)).sum(-1)
    nt, strb, res = af.NTHREADS, af.STR, af._res(af.NTHREADS)
    mt = af._METAL_TYPE[q.dtype]
    key = (DK, DV, T, T, H, scale, False, mt, nt, strb)
    lse = af._get("lse", *key)(
        inputs=[q, k],
        output_shapes=[(B, H, T)],
        output_dtypes=[mx.float32],
        grid=(nt, T // res, B * H),
        threadgroup=(nt, 1, 1),
    )[0]
    mx.eval(q, k, v, do, o, delta, lse)
    base = [q, k, v, do, lse, delta]
    kern = af._get("dq", *key)
    time.sleep(2)
    for i in range(30):
        dq = kern(
            inputs=base,
            output_shapes=[(B, H, T, DK)],
            output_dtypes=[mx.float32],
            grid=(nt, T // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0]
        mx.eval(dq)
        time.sleep(0.05)
    print("done")


if __name__ == "__main__":
    main()
