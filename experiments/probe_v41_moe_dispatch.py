"""MoE 分发形态 probe：gather_mm（现状）vs 容量分桶批量 matmul。

现状 `_sparse_forward` 走 `mx.gather_mm(xs[:,None,:], gu_t, rhs_indices=exps_s,
sorted_indices=True)`：按专家排序后一次调用，但 MLX 内部要按专家切段，实测只有
~4.9 TFLOPS（等价稠密 GEMM 在同样形状上是 12~14）。

这里对比「容量分桶」写法：按专家把 (token,choice) 行 scatter 进
[E, C, D] 的定长桶，再用**批量 matmul** [E,C,D]×[E,D,2I]，全程只读一次权重。

形状口径：M=4096, D=1024, E=96, K=6, I=256（默认 ≈1B 配方）。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402

from model.config import VibyConfig  # noqa: E402
from model.moe import MoEFeedForward, expert_act  # noqa: E402


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0] * 1000, ts[len(ts) // 2] * 1000


def sparse_gather_mm(x, idx, w, gu_t, dw_t):
    M, D = x.shape
    K = idx.shape[1]
    G = M * K
    flat = idx.reshape(G)
    order = mx.argsort(mx.stop_gradient(flat))
    exps_s = flat[order].astype(mx.int32)
    tok_s = (order // K).astype(mx.int32)
    w_s = w.reshape(G)[order].astype(x.dtype)
    xs = x[tok_s]
    h = mx.gather_mm(xs[:, None, :], gu_t, rhs_indices=exps_s, sorted_indices=True)
    gate, up = mx.split(h, 2, axis=-1)
    act = expert_act(gate, up, 10.0)
    y = mx.gather_mm(act, dw_t, rhs_indices=exps_s, sorted_indices=True)[:, 0, :]
    return mx.zeros((M, D), dtype=x.dtype).at[tok_s].add(y * w_s[:, None])


def sparse_bucketed(x, idx, w, gu_t, dw_t, E):
    M, D = x.shape
    K = idx.shape[1]
    G = M * K
    flat = idx.reshape(G)
    order = mx.argsort(mx.stop_gradient(flat))
    exps_s = flat[order].astype(mx.int32)
    tok_s = (order // K).astype(mx.int32)
    w_s = w.reshape(G)[order].astype(x.dtype)
    # 每个 (token,choice) 行在本专家桶内的槽位 = 组内序号
    counts = (
        mx.zeros((E,), dtype=mx.int32).at[exps_s].add(mx.ones((G,), dtype=mx.int32))
    )
    starts = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)[:-1]])
    slot = mx.arange(G, dtype=mx.int32) - starts[exps_s]
    C = int(mx.max(counts).item())
    xb = mx.zeros((E, C, D), dtype=x.dtype).at[exps_s, slot].add(x[tok_s])
    hb = xb @ gu_t  # [E,C,2I]
    gate, up = mx.split(hb, 2, axis=-1)
    act = expert_act(gate, up, 10.0)
    yb = act @ dw_t  # [E,C,D]
    y = yb[exps_s, slot]
    return mx.zeros((M, D), dtype=x.dtype).at[tok_s].add(y * w_s[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()

    cfg = VibyConfig(n_mtp_layers=0)
    moe = MoEFeedForward(cfg, 6)
    moe.train()
    B, T, D = args.batch, args.seq, cfg.dim
    E, K = cfg.n_routed_experts, cfg.n_activated_experts
    x3 = mx.random.normal((B, T, D)).astype(mx.bfloat16)
    mx.eval(x3, moe.parameters())

    M = B * T
    x = x3.reshape(M, D)
    w, idx, _ = moe.router(x)
    # 让负载接近均匀（真实训练有 aux loss + bias 均衡）
    idx = mx.stop_gradient(
        (mx.arange(M)[:, None] + mx.arange(K)[None, :] * (E // K)) % E
    ).astype(mx.int32)
    mx.eval(x, w, idx)

    gu_t = moe.experts.gate_up_w.swapaxes(-1, -2)
    dw_t = moe.experts.down_w.swapaxes(-1, -2)
    gu_t = gu_t.astype(x.dtype)
    dw_t = dw_t.astype(x.dtype)
    mx.eval(gu_t, dw_t)

    gflops = (
        2 * M * K * D * 2 * cfg.moe_inter_dim + 2 * M * K * cfg.moe_inter_dim * D
    ) / 1e9
    print(
        "MoE 分发对比 M=%d D=%d E=%d K=%d I=%d  专家 GEMM 共 %.1f GFLOP(fwd)"
        % (M, D, E, K, cfg.moe_inter_dim, gflops)
    )
    for name, fn in (
        ("gather_mm(sorted)", lambda: sparse_gather_mm(x, idx, w, gu_t, dw_t)),
        ("容量分桶批量 matmul", lambda: sparse_bucketed(x, idx, w, gu_t, dw_t, E)),
    ):
        mn, med = bench(lambda: mx.eval(fn()), iters=4, warmup=2)
        print(
            "  %-20s %7.2f / %7.2f ms   %5.1f TFLOPS(fwd)"
            % (name, mn, med, gflops / mn)
        )
    # 数值对拍
    a = sparse_gather_mm(x, idx, w, gu_t, dw_t)
    b = sparse_bucketed(x, idx, w, gu_t, dw_t, E)
    mx.eval(a, b)
    print(
        "  max|Δ| = %.3e"
        % float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
    )


if __name__ == "__main__":
    main()
