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

from model.config import VibyConfig
from model.moe import MoEFeedForward


def _env(name, default):
    return int(os.environ.get(name, default))


B = _env("VIBY_BENCH_B", 6)
T = _env("VIBY_BENCH_T", 2048)
D = _env("VIBY_BENCH_D", 768)
E = _env("VIBY_BENCH_E", 112)
I = _env("VIBY_BENCH_I", 104)  # noqa: E741
K = _env("VIBY_BENCH_K", 6)
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


def run(eg, fwd_mode, iters, align=None):
    MoEFeedForward._SPARSE_GROUP = eg
    MoEFeedForward._MLX_FORWARD = fwd_mode
    MoEFeedForward._FUSED_DISABLED = False
    if align is not None:
        MoEFeedForward._SPARSE_ALIGN = align
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
        return (moe(x_).astype(mx.float32) * C).sum()

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
    egs = [int(v) for v in os.environ.get("VIBY_SWEEP_EG", "").split(",") if v]
    als = [int(v) for v in os.environ.get("VIBY_SWEEP_AL", "128").split(",") if v]
    if not egs:
        egs = [e for e in (4, 8, 16, 32, 64, 144, E) if e <= E]
    print(f"形状 bs{B}×{T} D={D} E={E} I={I} K={K}，真实 pair={G}")
    print(
        f"{'EG':>5}{'AL':>5}{'前向路径':>10}{'桶行数':>10}{'pad':>7}"
        f"{'fwd':>9}{'fwd+bwd':>10}{'bwd':>9}"
    )
    for al in als:
        for eg in egs:
            for mode, label in (("0", "kernel"), ("1", "mlxgemm")):
                try:
                    f, fb, rows = run(eg, mode, iters, align=al)
                    print(
                        f"{eg:>5}{al:>5}{label:>10}{rows:>10}{rows / G:>7.2f}"
                        f"{f * 1e3:>9.1f}{fb * 1e3:>10.1f}{(fb - f) * 1e3:>9.1f}"
                    )
                except Exception as exc:
                    print(
                        f"{eg:>5}{al:>5}{label:>10}  ERR "
                        f"{type(exc).__name__}: {str(exc)[:50]}"
                    )


if __name__ == "__main__":
    main()
