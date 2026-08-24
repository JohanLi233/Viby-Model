"""同进程交替 A/B：MoE 稀疏桶的两条前向路径 × 不同 padding 水平。

_MLX_FORWARD="auto" 按 padding 比在「手写融合 kernel」和「MLX padded batched
GEMM」之间切换（阈值 _MLX_FWD_RATIO=2.0）。真实训练里路由塌缩会让 padding
在 1.5~3 之间来回，日志中 mlxfwd 在 100%/24%/0% 跳变 —— 但阈值是在「MLX 前向
还会在反向里重算 gate/up」的年代定的。现在 MLX 前向把预激活透给了 vjp 复用，
两条路的反向成本已不同，阈值需要重新标定。

跨进程计时在这台机器上漂移可达 ±30%，所以两条路在同一进程里逐次交替、取中位数。

用法: uv run experiments/ab_moe_fwd.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.moe import MoEFeedForward

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = int(os.environ.get("VIBY_BENCH_D", 768))
E = int(os.environ.get("VIBY_BENCH_E", 288))
I = int(os.environ.get("VIBY_BENCH_I", 104))  # noqa: E741
K = int(os.environ.get("VIBY_BENCH_K", 6))
G = B * T * K


def build():
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
        moe_router_noise=0.0,
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
    return moe


def set_padding(moe, target):
    """把容量表整体缩放到目标 padding（桶行数 / 真实 pair 数）。"""
    EG = moe._group_size(E)
    n_groups = E // EG
    # 每组容量相同时：rows = n_groups·EG·cap = E·cap ⇒ cap = target·G/E
    cap = max(64, int(round(target * G / E / 64)) * 64)
    moe._cap_table[0] = [cap] * n_groups
    moe._cap_G[0] = G
    moe._cap_peak.pop(0, None)
    moe._cap_perm.pop(0, None)
    moe._cap_order.pop(0, None)
    return cap * E / G


def main():
    moe = build()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5
    p = moe.trainable_parameters()
    mx.eval(x, C)

    def loss(p_):
        moe.update(p_)
        return (moe(x).astype(mx.float32) * C).sum()

    vg = mx.value_and_grad(loss)

    print(f"B={B} T={T} D={D} E={E} I={I} K={K}  真实 pair={G}")
    print(
        f"{'目标pad':>8}{'实际pad':>9}{'路径':>10}{'fwd':>9}{'fwd+bwd':>10}{'相对':>8}"
    )
    targets = [float(v) for v in os.environ.get("VIBY_AB_PAD", "").split(",") if v]
    for target in targets or (1.5, 1.8, 2.2, 3.0, 4.0):
        arms = {"kernel": "0", "mlxgemm": "1"}
        samples = {k: {"f": [], "fb": []} for k in arms}
        actual = None
        for rnd in range(7):
            for name, mode in arms.items():
                MoEFeedForward._MLX_FORWARD = mode
                actual = set_padding(moe, target)
                moe._pending_counts.clear()
                t0 = time.perf_counter()
                mx.eval(loss(p))
                t1 = time.perf_counter()
                moe._pending_counts.clear()
                set_padding(moe, target)
                mx.eval(vg(p))
                t2 = time.perf_counter()
                moe._pending_counts.clear()
                if rnd >= 2:
                    samples[name]["f"].append(t1 - t0)
                    samples[name]["fb"].append(t2 - t1)
        med = {
            n: (statistics.median(s["f"]), statistics.median(s["fb"]))
            for n, s in samples.items()
        }
        base = med["mlxgemm"][1]
        for name in arms:
            f, fb = med[name]
            print(
                f"{target:>8.1f}{actual:>9.2f}{name:>10}{f * 1e3:>9.2f}"
                f"{fb * 1e3:>10.2f}{base / fb:>8.2f}x"
            )
    MoEFeedForward._MLX_FORWARD = "auto"


if __name__ == "__main__":
    main()
