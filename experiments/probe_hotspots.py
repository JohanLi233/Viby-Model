"""热点归因探针：
1) SDPA 在 MLA 形状（qk=128 / v=96 不等宽）下是否走 MLX 快路径；
2) MoE 稀疏桶容量表收敛后的真实 padding 比与其对 fwd/bwd 的影响。

用法: .venv/bin/python experiments/probe_hotspots.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B, T, H = 6, 2048, 8


def timed(fn, iters=10, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def probe_sdpa():
    print("=== SDPA 形状敏感性（B=6 H=8 T=2048, bf16）===")
    print(f"{'qk_dim':>7}{'v_dim':>7}{'mask':>10}{'fwd(ms)':>10}{'TFLOPS':>9}")
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(bias)

    for qk, vd in [(128, 96), (128, 128), (96, 96), (64, 64)]:
        q = (mx.random.normal((B, H, T, qk)) * 0.1).astype(mx.bfloat16)
        k = (mx.random.normal((B, H, T, qk)) * 0.1).astype(mx.bfloat16)
        v = (mx.random.normal((B, H, T, vd)) * 0.1).astype(mx.bfloat16)
        mx.eval(q, k, v)
        scale = qk**-0.5
        for mlabel, m in [("causal", "causal"), ("array", bias)]:
            t = timed(
                lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=m
                )
            )
            fl = 2 * B * H * T * T * (qk + vd)
            print(f"{qk:>7}{vd:>7}{mlabel:>10}{t * 1e3:>10.2f}{fl / t / 1e12:>9.2f}")


def probe_moe_padding():
    from model.config import VibyConfig
    from model.moe import MoEFeedForward

    print("\n=== MoE 桶 padding 收敛与耗时（bs6×2048, E=112 I=104 K=6）===")
    D, E, I, K = 768, 112, 104, 6  # noqa: E741
    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=1,
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=T,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=I,
    )
    from mlx.utils import tree_map

    moe = MoEFeedForward(cfg)
    moe.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            moe.parameters(),
        )
    )
    mx.eval(moe.parameters())
    moe.train()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5
    G = B * T * K

    def loss(x_, p):
        moe.update(p)
        return (moe(x_).astype(mx.float32) * C).sum()

    p = moe.trainable_parameters()
    # 迭代到容量表稳态
    for i in range(40):
        mx.eval(loss(x, p))
        moe.update_capacity_table()
        if i in (0, 4, 14, 39):
            rows = sum(moe._cap_table[0]) * min(moe._SPARSE_GROUP, E)
            print(f"  iter {i + 1:>3}: rows={rows} padding={rows / G:.2f}×")

    moe._pending_counts.get(0)
    idx, w = moe.router(x)
    cnt = mx.zeros((E,), dtype=mx.int32).at[idx.reshape(-1)].add(1)
    mx.eval(cnt)
    c = sorted(cnt.tolist(), reverse=True)
    print(
        f"  专家计数分布: max={c[0]} p90={c[11]} 中位={c[E // 2]} min={c[-1]} "
        f"均值={G / E:.0f}"
    )
    # 分组 max 之和（当前连续分组 vs 按计数排序分组）
    raw = cnt.tolist()
    EG = 8
    cur = sum(max(raw[g * EG : (g + 1) * EG]) for g in range(E // EG)) * EG
    srt = sum(max(c[g * EG : (g + 1) * EG]) for g in range(E // EG)) * EG
    print(
        f"  理想桶行数(无余量): 连续分组={cur} ({cur / G:.2f}×) | "
        f"排序分组={srt} ({srt / G:.2f}×) | 逐专家={sum(raw)} (1.00×)"
    )

    vg = mx.value_and_grad(loss, argnums=(0, 1))
    f = timed(lambda: loss(x, p), 6, 2)
    fb = timed(lambda: vg(x, p), 6, 2)
    rows = sum(moe._cap_table[0]) * EG
    print(
        f"  稳态 padding {rows / G:.2f}×: fwd {f * 1e3:.1f}ms | "
        f"fwd+bwd {fb * 1e3:.1f}ms ⇒ bwd {(fb - f) * 1e3:.1f}ms"
    )


if __name__ == "__main__":
    probe_sdpa()
    probe_moe_padding()
