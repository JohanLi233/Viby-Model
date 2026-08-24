"""手写 flash 反向 vs mlx autodiff 反向，同进程交替计时。

分三段报（lse / dq / dkv），便于定位哪一段拖后腿。机器状态在分钟尺度会漂，
两条臂逐次交替取中位数。

用法: uv run experiments/bench_flash_bwd.py
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_fused as af

B = int(os.environ.get("VIBY_BENCH_B", 12))
H = int(os.environ.get("VIBY_BENCH_H", 8))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
DK = int(os.environ.get("VIBY_BENCH_DK", 128))
DV = int(os.environ.get("VIBY_BENCH_DV", 96))


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
    o = mx.fast.scaled_dot_product_attention(q, k, vpad, scale=scale, mask="causal")[
        ..., :DV
    ]
    mx.eval(q, k, v, do, vpad, dmask, o)

    def mlx_bwd_causal():
        def f(q_, k_, v_):
            out = mx.fast.scaled_dot_product_attention(
                q_, k_, v_, scale=scale, mask="causal"
            )
            return (out[..., :DV].astype(mx.float32) * do).sum()

        return mx.value_and_grad(f, argnums=(0, 1, 2))(q, k, vpad)

    def mlx_bwd_mask():
        def f(q_, k_, v_):
            out = mx.fast.scaled_dot_product_attention(
                q_, k_, v_, scale=scale, mask=dmask
            )
            return (out[..., :DV].astype(mx.float32) * do).sum()

        return mx.value_and_grad(f, argnums=(0, 1, 2))(q, k, vpad)

    arms = {
        "mlx autodiff (causal)": mlx_bwd_causal,
        "mlx autodiff (doc_mask)": mlx_bwd_mask,
        "手写 flash (causal)": lambda: af.flash_backward(q, k, v, o, do, scale, None),
        "手写 flash (doc_mask)": lambda: af.flash_backward(
            q, k, v, o, do, scale, dmask
        ),
    }
    samples = {n: [] for n in arms}
    for rnd in range(8):
        for n, f in arms.items():
            t0 = time.perf_counter()
            mx.eval(f())
            samples[n].append(time.perf_counter() - t0)
    med = {n: statistics.median(s[2:]) for n, s in samples.items()}

    half = B * H * (T * T / 2)
    fl = half * (DK * 2) * 4 + half * (DV * 2) * 3  # S×2 dQ dK / dP×2 dV
    print(f"B={B} H={H} T={T} d_qk={DK} d_v={DV}   反向 FLOPs {fl / 1e9:.1f} GFLOP")
    print(f"{'':<26}{'fwd+bwd':>10}{'TFLOPS':>9}")
    for n, t in med.items():
        print(f"  {n:<24}{t * 1e3:>10.2f}{fl / t / 1e12:>9.2f}")
    print(
        f"\n  causal   加速 {med['mlx autodiff (causal)'] / med['手写 flash (causal)']:.2f}x"
        f"\n  doc_mask 加速 {med['mlx autodiff (doc_mask)'] / med['手写 flash (doc_mask)']:.2f}x"
        "\n  （mlx 那一栏含前向；手写那栏是纯反向，前向仍复用 mlx SDPA）"
    )

    print("\n分段（doc_mask）")
    delta = (do.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)
    mx.eval(delta)
    nt, strb = af.NTHREADS, af.STR
    mt = af._METAL_TYPE[q.dtype]
    key = (DK, DV, T, T, H, scale, True, mt, nt, strb)
    split = af._split_ok(DK, DV, nt)
    res = af._res_split(nt) if split else af._res(nt)

    if split:
        strl = af._lse_str(nt, strb)
        key_lse = (DK, DV, T, T, H, scale, True, mt, nt, strl)

        def run_lse():
            return af._get_lse_fused(*key_lse)(
                inputs=[q, k, do, o, dmask],
                output_shapes=[(B, H, T), (B, H, T)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )
    else:

        def run_lse():
            return af._get("lse", *key)(
                inputs=[q, k, dmask],
                output_shapes=[(B, H, T)],
                output_dtypes=[mx.float32],
                grid=(nt, T // res, B * H),
                threadgroup=(nt, 1, 1),
            )[0]

    lse_out = run_lse()
    mx.eval(lse_out)
    if split:
        lse, delta = lse_out
    else:
        lse = lse_out
    base = [q, k, v, do, lse, delta, dmask]
    segs = {
        "Δ=rowsum(dO∘O)": lambda: (do.astype(mx.float32) * o.astype(mx.float32)).sum(
            -1
        ),
        "lse+Δ kernel": run_lse,
        "dq kernel": lambda: af._get("dq", *key)(
            inputs=base,
            output_shapes=[(B, H, T, DK)],
            output_dtypes=[mx.float32],
            grid=(nt, T // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0],
        "dkv kernel": lambda: af._get("dkv", *key)(
            inputs=base,
            output_shapes=[(B, H, T, DK), (B, H, T, DV)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(nt, T // res, B * H),
            threadgroup=(nt, 1, 1),
        ),
    }
    sfl = {
        "lse+Δ kernel": half * DK * 2,
        "dq kernel": half * (DK * 2 * 2 + DV * 2),
        "dkv kernel": half * (DK * 2 * 2 + DV * 2 * 2),
    }
    ss = {n: [] for n in segs}
    for rnd in range(8):
        for n, f in segs.items():
            t0 = time.perf_counter()
            mx.eval(f())
            ss[n].append(time.perf_counter() - t0)
    for n in segs:
        t = statistics.median(ss[n][2:])
        tf = f"{sfl[n] / t / 1e12:>8.2f}" if n in sfl else "       -"
        print(f"  {n:<18}{t * 1e3:>8.2f}ms{tf} TFLOPS")


if __name__ == "__main__":
    main()
