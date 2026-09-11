"""MoE 路径 probe：gather_mm 分发形态 vs 变体，fwd / fwd+bwd 墙钟。

形状口径：B=4, T=1024, dim=1024, E=96, top_k=6, moe_inter=256。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402
from mlx import nn  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()
    cfg = VibyConfig()
    moe = MoEFeedForward(cfg, 6)
    moe.train()
    B, T, D = args.batch, args.seq, cfg.dim
    x = mx.random.normal((B, T, D)).astype(mx.bfloat16)
    mx.eval(x, moe.parameters())

    def fwd():
        mx.eval(moe(x))

    def fwdbwd():
        def loss(m):
            return mx.sum(m(x) ** 2)

        mx.eval(nn.value_and_grad(moe, loss)(moe))

    print("MoE E=%d k=%d inter=%d B=%d T=%d" % (cfg.n_routed_experts, cfg.n_activated_experts, cfg.moe_inter_dim, B, T))
    print("  fwd     %.2f / %.2f ms" % bench(fwd))
    print("  fwd+bwd %.2f / %.2f ms" % bench(fwdbwd, iters=3, warmup=1))

    # 参考上界：同样 FLOPs 的纯批量 GEMM（无路由分发）
    M = B * T
    E, k = cfg.n_routed_experts, cfg.n_activated_experts
    xf = x.reshape(M, D)
    w1 = mx.random.normal((E, D, 2 * cfg.moe_inter_dim)).astype(mx.bfloat16) * 0.02
    w2 = mx.random.normal((E, cfg.moe_inter_dim, D)).astype(mx.bfloat16) * 0.02
    # 每个 token 只挑 k 个专家：用 top-k 索引 gather 权重后 einsum，这里只做
    # "k 个专家的 GEMM 总量" 的上界估计，按 token 分块避免 O(M*E*D)
    idx = mx.random.randint(0, E, (M, k))
    mx.eval(idx, w1, w2)

    def upper():
        # [M,k,D] gather 权重再逐 k 做 matmul（等价 FLOPs，形状友好）
        acc = mx.zeros((M, D), dtype=mx.bfloat16)
        for j in range(k):
            wj = w1[idx[:, j]]                     # [M, D, 2I]
            h = mx.einsum("md,mdi->mi", xf, wj)
            g, u = mx.split(h, 2, axis=-1)
            a = expert_act(g, u, cfg.swiglu_limit)
            wj2 = w2[idx[:, j]]                    # [M, I, D]
            acc = acc + mx.einsum("mi,mid->md", a, wj2)
        return acc

    print("  k 次 [M,D]×[M,D,2I] einsum 上界  %.2f / %.2f ms" % bench(lambda: mx.eval(upper()), iters=3, warmup=1))


if __name__ == "__main__":
    main()
