"""MoE 桶装槽策略的 A/B 计时：按专家 id 成组 vs 按负载排序装槽。

用 dump_moe_load.py 导出的真实逐专家计数作为路由，交替跑两种分组并取
中位数，抵消机器状态漂移（长跑之间墙钟能差 3 倍，跨轮次比较不可信）。

两种分组的算术完全相同（见 verify_moe_slotting.py 的逐位等价验证），
差别只在 padded 桶的总行数，所以时间差就是纯粹白算掉的部分。

用法:
    .venv/bin/python experiments/bench_moe_slotting.py [--loads /tmp/moe_load.npz]
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from model.model import MoEFeedForward, VibyConfig

B, T, D, E, I, K = 6, 2048, 768, 112, 104, 6


def build():
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


def set_routing(moe, counts, seed=0):
    """按给定的逐专家计数构造固定路由（同一 token 的 K 个专家互不重复）"""
    rng = np.random.default_rng(seed)
    p = counts.astype(np.float64) / counts.sum()
    idx = np.stack([rng.choice(E, K, replace=False, p=p) for _ in range(B * T)])
    idx_a = mx.array(idx.reshape(B, T, K).astype(np.int32))
    w_a = mx.array(rng.random((B, T, K)).astype(np.float32)).astype(mx.bfloat16)

    def stub(x, step_idx=None, collect_aux=False):
        return idx_a, w_a

    moe.router = stub  # type: ignore[assignment]


def force_id_grouping(moe):
    """把容量表改回按专家 id 成组（容量推导规则不变），用作对照"""
    EG = min(moe._SPARSE_GROUP, E)
    AL = moe._SPARSE_ALIGN
    n_groups = (E + EG - 1) // EG
    for k in list(moe._cap_peak):
        peaks = moe._cap_peak[k]
        moe._cap_order[k] = list(range(E))
        moe._cap_perm[k] = list(range(E))
        moe._cap_table[k] = [
            max(
                (int(max(peaks[gi * EG : (gi + 1) * EG])) * 5 // 4 + 64 + AL - 1)
                // AL
                * AL,
                AL,
            )
            for gi in range(n_groups)
        ]


def measure(moe, identity, iters=7):
    for d in (
        moe._cap_table,
        moe._cap_peak,
        moe._cap_perm,
        moe._cap_order,
        moe._cap_G,
        moe._pending_counts,
    ):
        d.clear()
    mx.random.seed(1)
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5
    p = moe.trainable_parameters()

    def loss(p_):
        moe.update(p_)
        return (moe(x, step_idx=0).astype(mx.float32) * C).sum()

    vg = mx.value_and_grad(loss)
    ts = []
    for it in range(iters + 3):
        t0 = time.perf_counter()
        v, g = vg(p)
        mx.eval(v, g)
        dt = time.perf_counter() - t0
        moe.update_capacity_table()
        if identity:
            force_id_grouping(moe)
        if it >= 3:  # 前 3 次让容量表收敛 + 预热
            ts.append(dt)
    rows = sum(moe._cap_table[0]) * min(moe._SPARSE_GROUP, E)
    return statistics.median(ts), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", type=str, default="/tmp/moe_load.npz")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    d = np.load(args.loads)
    L = d["loads"]
    n = len(L)
    cases = [
        ("早期 (路由健康)", L[: n // 3][len(L[: n // 3]) // 2]),
        ("中期", L[n // 3 : 2 * n // 3][len(L[n // 3 : 2 * n // 3]) // 2]),
        ("后期 (路由塌缩)", L[2 * n // 3 :][len(L[2 * n // 3 :]) // 2]),
    ]
    G = B * T * K
    moe = build()
    print(f"MoE fwd+bwd，bs{B}×seq{T}，E={E} top{K}，真实负载向量\n")
    print(f"{'负载':<18}{'max/mean':>9}{'按id成组':>20}{'按负载装槽':>20}{'提速':>8}")
    for name, counts in cases:
        counts = np.maximum(counts, 1)
        set_routing(moe, counts)
        ta, tb, ra, rb = [], [], None, None
        for _ in range(args.reps):  # 交替，抵消漂移
            t, ra = measure(moe, identity=True)
            ta.append(t)
            t, rb = measure(moe, identity=False)
            tb.append(t)
        a, b = statistics.median(ta), statistics.median(tb)
        print(
            f"{name:<16}{counts.max() / counts.mean():>9.1f}"
            f"{a * 1e3:>12.1f}ms {ra / G:>5.2f}x"
            f"{b * 1e3:>12.1f}ms {rb / G:>5.2f}x"
            f"{a / b:>7.2f}x"
        )


if __name__ == "__main__":
    main()
