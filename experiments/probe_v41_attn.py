"""注意力路径 probe：稠密 [T, T+N] sdpa vs 逐 query gather 后的短 sdpa。

验证 hypothesis：训练路径的滑动窗口分支把整个 block 的 T 个 key 都算进去，
只有 window 个可见 —— 换成"每个 query 只 gather 自己要看的 key"能省多少。

形状口径：B=4, T=1024, H=16, D=128, window=128, index_topk=64。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402

NEG_INF = float("-inf")


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
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    B, T, H, D = args.batch, args.seq, args.heads, args.head_dim
    W, K = args.window, args.topk
    mx.random.seed(0)
    q = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    kv = mx.random.normal((B, T, D)).astype(mx.bfloat16)
    mx.eval(q, kv)

    # ---- 现状：dense，S = T + N ----
    def dense(S):
        k = mx.random.normal((B, S, D)).astype(mx.bfloat16)
        mask = mx.random.uniform(0, 1, (B, 1, T, S)) > 0.5
        mx.eval(k, mask)

        def f():
            o = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k[:, None],
                k[:, None],
                scale=D**-0.5,
                mask=mask,
            )
            mx.eval(o)

        return f

    # ---- 方案：逐 query gather 出 [B,T,W+K,D]，q 折进 batch 维 ----
    def gathered():
        wi = mx.arange(T)[:, None] - (W - 1) + mx.arange(W)[None, :]  # [T,W]
        wi = mx.maximum(wi, 0)
        ci = mx.random.randint(0, T, (B, T, K))
        idx = mx.concatenate(
            [mx.broadcast_to(wi[None], (B, T, W)), ci], axis=-1
        ).astype(mx.int32)  # [B,T,W+K]
        mx.eval(idx)

        def f():
            kg = mx.take_along_axis(
                kv[:, None, :, :], idx[..., None], axis=2
            )  # [B,T,S,D]
            qf = q.reshape(B * T, H, 1, D)
            kf = kg.reshape(B * T, 1, kg.shape[2], D)
            o = mx.fast.scaled_dot_product_attention(qf, kf, kf, scale=D**-0.5)
            mx.eval(o.reshape(B, T, H, D))

        return f

    print(
        "形状 B=%d T=%d H=%d D=%d window=%d topk=%d  (min/med ms, 前向)"
        % (B, T, H, D, W, K)
    )
    for S in (T, T + T // 2, 2 * T, W + K):
        mn, med = bench(dense(S) if S != W + K else gathered())
        tag = "dense S=%d" % S if S != W + K else "gather S=%d" % S
        print("  %-14s %.2f / %.2f ms" % (tag, mn, med))


if __name__ == "__main__":
    main()
