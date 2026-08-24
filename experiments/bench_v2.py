"""attn_v2 vs attn_fused(v1)：逐 kernel A/B 基准（真实形状，交替计时取中位数）。

用法: uv run experiments/bench_v2.py
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from experiments import attn_v2 as v2
from model.kernels import attn_fused as af

B = int(os.environ.get("VIBY_BENCH_B", 12))
H = int(os.environ.get("VIBY_BENCH_H", 8))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
DK = int(os.environ.get("VIBY_BENCH_DK", 128))
DV = int(os.environ.get("VIBY_BENCH_DV", 96))

ARMS = {
    "v1 (af)": (af, None),
    "v2 default": (v2, {}),
    "v2 nofast": (v2, {"fast_exp": False}),
    "v2 nomax": (v2, {"lse_nomax": False}),
    "v2 nodelta": (v2, {"lse_delta": False}),
    "v2 nohoist": (v2, {"dq_hoist": False}),
    "v2 qcache": (v2, {"dq_qcache": True}),
    "v2 nokhoist": (v2, {"dkv_khoist": False}),
    "v2 nopad": (v2, {"dkv_pad": False}),
}


def getk(mod, kind, *key, cfg=None):
    if cfg is None or mod is af:
        return mod._get(kind, *key)
    return mod._get(kind, *key, cfg=cfg)


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    scale = 1.0 / math.sqrt(DK)
    vpad = mx.concatenate([v, mx.zeros((B, H, T, DK - DV), dtype=v.dtype)], axis=-1)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    dmask = mx.where((same & tril)[:, None], 0.0, -1e9).astype(mx.bfloat16)
    o_ = mx.fast.scaled_dot_product_attention(q, k, vpad, scale=scale, mask="causal")[
        ..., :DV
    ]
    mx.eval(q, k, v, do, dmask, o_)

    half = B * H * (T * T / 2)
    fl = {
        "lse": half * DK * 2,
        "dq": half * (DK * 2 * 2 + DV * 2),
        "dkv": half * (DK * 2 * 2 + DV * 2 * 2),
    }
    nt, strb, res = af.NTHREADS, af.STR, af._res(af.NTHREADS)
    mt = af._METAL_TYPE[q.dtype]

    # 全臂共享的 lse/delta（用 v2 默认算一遍，各 dq/dkv 臂共用）
    hm = True
    lse, delta = getk(v2, "lse", DK, DV, T, T, H, scale, hm, mt, nt, strb, cfg={})(
        inputs=[q, k, do, o_, dmask],
        output_shapes=[(B, H, T), (B, H, T)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(nt, T // res, B * H),
        threadgroup=(nt, 1, 1),
    )
    mx.eval(lse, delta)
    base = [q, k, v, do, lse, delta, dmask]

    for mname, (mod, cfg) in ARMS.items():
        print(f"\n== {mname} ==")
        rows = {}
        # lse（v2 含 delta；v1 不含）
        if mod is v2 and cfg.get("lse_delta", True):
            out = getk(mod, "lse", DK, DV, T, T, H, scale, hm, mt, nt, strb, cfg=cfg)(
                inputs=[q, k, do, o_, dmask],
                output_shapes=[(B, H, T), (B, H, T)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )
            mx.eval(out)
            rows["lse+Δ"] = lambda: getk(
                mod, "lse", DK, DV, T, T, H, scale, hm, mt, nt, strb, cfg=cfg
            )(
                inputs=[q, k, do, o_, dmask],
                output_shapes=[(B, H, T), (B, H, T)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )
        else:
            out = getk(mod, "lse", DK, DV, T, T, H, scale, hm, mt, nt, strb, cfg=cfg)(
                inputs=[q, k, dmask],
                output_shapes=[(B, H, T)],
                output_dtypes=[mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )
            mx.eval(out)
            rows["lse"] = lambda: getk(
                mod, "lse", DK, DV, T, T, H, scale, hm, mt, nt, strb, cfg=cfg
            )(
                inputs=[q, k, dmask],
                output_shapes=[(B, H, T)],
                output_dtypes=[mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )
            if mod is v2 and not cfg.get("lse_delta", True):
                rows["Δ(mlx)"] = lambda: (
                    do.astype(mx.float32) * o_.astype(mx.float32)
                ).sum(-1)

        kq = (DK, DV, T, T, H, scale, hm, mt, nt, strb)
        rows["dq"] = lambda: getk(mod, "dq", *kq, cfg=cfg)(
            inputs=base,
            output_shapes=[(B, H, T, DK)],
            output_dtypes=[mx.float32],
            grid=(nt, T // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0]
        rows["dkv"] = lambda: getk(mod, "dkv", *kq, cfg=cfg)(
            inputs=base,
            output_shapes=[(B, H, T, DK), (B, H, T, DV)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(nt, T // res, B * H),
            threadgroup=(nt, 1, 1),
        )
        for n, f in rows.items():
            mx.eval(f())

        ss = {n: [] for n in rows}
        for rnd in range(8):
            for n, f in rows.items():
                t0 = time.perf_counter()
                mx.eval(f())
                ss[n].append(time.perf_counter() - t0)
        for n in rows:
            t = statistics.median(ss[n][2:])
            tf = f"{fl[n.split('+')[0]] / t / 1e12:>7.2f}" if n in fl else "-"
            print(f"  {n:<10}{t * 1e3:>8.2f}ms  {tf} TFLOPS")


if __name__ == "__main__":
    main()
