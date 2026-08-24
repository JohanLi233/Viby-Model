"""A/B：对真实 Attention 模块的 value_and_grad 套 mx.compile 是否有收益。

probe_attn 的替换实验显示：注意力模块 fwd+bwd 47.6ms 里 SDPA 只占 26.8ms，
另外 20.8ms 中纯投影 GEMM 约 13ms，剩下 ~8ms 是 split/concat/transpose/
QK-norm/RoPE/attn-gate 这类访存型胶水（×9 层 ≈ 70ms/步）。这些正是
mx.compile 能融合掉中间物化的部分。

MoE 的桶形状每步都变，整步 compile 会反复重编译；但注意力模块形状是静态的，
可以单独编译。先验证 compile 能否穿过 mx.fast.scaled_dot_product_attention
并保持梯度正确，再看收益是否值得改造。

用法: uv run experiments/ab_attn_compile.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from model.attention import GQAAttention as Attention
from model.config import VibyConfig

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = int(os.environ.get("VIBY_BENCH_D", 768))
H = int(os.environ.get("VIBY_BENCH_H", 8))


def main():
    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=2,
        num_attention_heads=H,
        vocab_size=6400,
        max_position_embeddings=T,
        use_attn_gate=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    mx.random.seed(0)
    attn = Attention(cfg, layer_idx=0)  # local 层：partial RoPE + 滑窗
    attn.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            attn.parameters(),
        )
    )
    mx.eval(attn.parameters())

    rope = cfg.head_dim // 2  # partial RoPE：只旋转前半维
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5
    cos = mx.random.normal((T, rope)).astype(mx.bfloat16)
    sin = mx.random.normal((T, rope)).astype(mx.bfloat16)
    pos = (cos, sin)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(x, C, cos, sin, bias)

    p = attn.trainable_parameters()

    def loss(x_, p_):
        attn.update(p_)
        o, _ = attn(x_, pos, causal_bias=bias, mask_is_full=False)
        return (o.astype(mx.float32) * C).sum()

    vg = mx.value_and_grad(loss, argnums=(0, 1))
    vg_c = mx.compile(vg)

    l0, (gx0, gp0) = vg(x, p)
    mx.eval(l0, gx0, gp0)
    l1, (gx1, gp1) = vg_c(x, p)
    mx.eval(l1, gx1, gp1)

    def maxrel(a, b):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        d = mx.abs(a - b).max().item()
        s = mx.abs(a).max().item()
        return d / max(s, 1e-9)

    worst = maxrel(gx0, gx1)
    name = "dx"
    for (k, v0), (_, v1) in zip(tree_flatten(gp0), tree_flatten(gp1)):
        r = maxrel(v0, v1)
        if r > worst:
            worst, name = r, k
    print(
        f"数值一致性：loss {abs(l0.item() - l1.item()) / abs(l0.item()):.2e}，"
        f"梯度最大相对偏差 {worst:.2e}（{name}）"
    )

    arms = {"eager": vg, "compile": vg_c}
    samples = {k: [] for k in arms}
    for rnd in range(9):
        for n, f in arms.items():
            t0 = time.perf_counter()
            mx.eval(f(x, p))
            samples[n].append(time.perf_counter() - t0)
    med = {n: statistics.median(s[2:]) for n, s in samples.items()}
    print(f"\nB={B} T={T} D={D} H={H}  单层 Attention fwd+bwd")
    for n, t in med.items():
        print(f"  {n:<8}{t * 1e3:>8.2f}ms")
    print(
        f"  加速 {med['eager'] / med['compile']:.2f}x"
        f"（9 层外推 {(med['eager'] - med['compile']) * 9 * 1e3:.0f}ms/步）"
    )


if __name__ == "__main__":
    main()
