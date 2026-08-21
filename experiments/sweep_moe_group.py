"""扫描 MoE 稀疏桶的分组大小 _SPARSE_GROUP 与前向路径选择。

容量表修好（衰减峰值口径）后，专家计数在均衡态下差异很小，组内取 max
的 padding 代价与组大小几乎无关，但更大的组意味着更少、更胖的 batched
GEMM。这里在真实形状下扫 EG × 前向路径，取端到端 fwd / fwd+bwd 最优点。

用法: .venv/bin/python experiments/sweep_moe_group.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.model import MoEFeedForward, VibyConfig

B, T, D, E, I, K = 6, 2048, 768, 112, 104, 6
G = B * T * K


def timed(fn, it=6, w=2):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def run(eg, fwd_mode, iters):
    MoEFeedForward._SPARSE_GROUP = eg
    MoEFeedForward._MLX_FORWARD = fwd_mode
    MoEFeedForward._FUSED_DISABLED = False
    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=1,
        num_attention_heads=8,
        kv_lora_rank=192,
        qk_rope_head_dim=32,
        vocab_size=6400,
        max_position_embeddings=T,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=I,
        hrm_H_cycles=2,
        hrm_L_cycles=3,
        hrm_cycle_router=1,
        hrm_cycle_router_rank=8,
    )
    mx.random.seed(0)
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

    def loss(x_, p):
        moe.update(p)
        return (moe(x_, step_idx=0).astype(mx.float32) * C).sum()

    p = moe.trainable_parameters()
    for _ in range(4):
        mx.eval(loss(x, p))
        moe.update_capacity_table()
    vg = mx.value_and_grad(loss, argnums=(0, 1))
    f = timed(lambda: loss(x, p), iters)
    fb = timed(lambda: vg(x, p), iters)
    rows = sum(moe._cap_table[0]) * min(eg, E)
    return f, fb, rows


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"形状 bs{B}×{T} E={E} I={I} K={K}，真实 pair={G}")
    print(
        f"{'EG':>5}{'前向路径':>10}{'桶行数':>10}{'pad':>7}{'fwd':>9}{'fwd+bwd':>10}{'bwd':>9}"
    )
    for eg in (4, 8, 14, 16, 28, 56, 112):
        for mode, label in (("0", "kernel"), ("1", "mlxgemm")):
            try:
                f, fb, rows = run(eg, mode, iters)
                print(
                    f"{eg:>5}{label:>10}{rows:>10}{rows / G:>7.2f}"
                    f"{f * 1e3:>9.1f}{fb * 1e3:>10.1f}{(fb - f) * 1e3:>9.1f}"
                )
            except Exception as exc:
                print(f"{eg:>5}{label:>10}  ERR {type(exc).__name__}: {str(exc)[:50]}")


if __name__ == "__main__":
    main()
