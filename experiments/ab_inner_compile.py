"""验证「内层 mx.compile + 外层 value_and_grad」是否生效且数值一致。

真实训练里 value_and_grad 套在整个模型 loss 外面，而整步 compile 会被 MoE
的动态桶形状反复触发重编译。所以想拿到 compile 收益，只能把形状静态的子图
（注意力 + 层内 norm/residual）在模块内部编译，让外层求导穿过它。

MLX 的 compile 号称对变换透明，但「透明」可能意味着外层求导时直接退回未编译
的原函数 —— 那就一分钱收益都没有。这里用真实 Attention 模块直接测。

用法: uv run experiments/ab_inner_compile.py
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
from model.norms import RMSNorm

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
    norm = RMSNorm(D, eps=cfg.rms_norm_eps)
    for m in (attn, norm):
        m.update(
            tree_map(
                lambda a: a.astype(mx.bfloat16)
                if mx.issubdtype(a.dtype, mx.floating)
                else a,
                m.parameters(),
            )
        )
    mx.eval(attn.parameters(), norm.parameters())

    rope = cfg.head_dim // 2  # partial RoPE：只旋转前半维
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5
    cos = mx.random.normal((T, rope)).astype(mx.bfloat16)
    sin = mx.random.normal((T, rope)).astype(mx.bfloat16)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(x, C, cos, sin, bias)

    # 纯函数化：权重全部走显式实参，compile 才能缓存计算图
    keys = [k for k, _ in tree_flatten(attn.trainable_parameters())]

    def body(x_, nw, *ws):
        attn.update(tree_map(lambda a: a, attn.trainable_parameters()))
        d = dict(zip(keys, ws))
        attn.update(_unflatten(d))
        norm.update({"weight": nw})
        o, _ = attn(norm(x_), (cos, sin), causal_bias=bias, mask_is_full=False)
        return x_ + o

    def _unflatten(d):
        out: dict = {}
        for k, v in d.items():
            cur = out
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        return out

    body_c = mx.compile(body)

    ws = [v for _, v in tree_flatten(attn.trainable_parameters())]
    nw = norm.weight

    def mk(f):
        def loss(nw_, *ws_):
            return (f(x, nw_, *ws_).astype(mx.float32) * C).sum()

        return mx.value_and_grad(loss, argnums=tuple(range(1 + len(ws))))

    vg_e, vg_c = mk(body), mk(body_c)
    l0, g0 = vg_e(nw, *ws)
    mx.eval(l0, g0)
    l1, g1 = vg_c(nw, *ws)
    mx.eval(l1, g1)

    worst = 0.0
    for a, b in zip(g0, g1):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        worst = max(
            worst,
            mx.abs(a - b).max().item() / max(mx.abs(a).max().item(), 1e-9),
        )
    print(
        f"数值一致性：loss {abs(l0.item() - l1.item()) / abs(l0.item()):.2e}，"
        f"梯度最大相对偏差 {worst:.2e}"
    )

    arms = {"内层 eager": vg_e, "内层 compile": vg_c}
    samples = {k: [] for k in arms}
    for rnd in range(9):
        for n, f in arms.items():
            t0 = time.perf_counter()
            mx.eval(f(nw, *ws))
            samples[n].append(time.perf_counter() - t0)
    med = {n: statistics.median(s[2:]) for n, s in samples.items()}
    print(f"\nB={B} T={T} D={D} H={H}  norm+Attention+residual 单层 fwd+bwd")
    for n, t in med.items():
        print(f"  {n:<14}{t * 1e3:>8.2f}ms")
    sp = med["内层 eager"] / med["内层 compile"]
    print(
        f"  加速 {sp:.2f}x"
        f"（9 层外推 {(med['内层 eager'] - med['内层 compile']) * 9 * 1e3:.0f}ms/步）"
    )


if __name__ == "__main__":
    main()
