"""验证 MoE 桶「按负载排序装槽」与「按专家 id 成组」逐位等价。

装槽只改变 pair 在 padded 桶缓冲里的落点（以及由此派生的每组容量），
不改变任何一个 (token, expert) 对参与的算术，因此前向 loss 与全部梯度
都应逐位相同——差异只应体现在桶总行数（padding 倍率）上。

用固定的倾斜路由（绕开 router，直接喂 idx/w）覆盖两种分组容量确实不同
的情形，并同时验证融合 kernel 前向与 MLX GEMM 前向两条路径。

用法:
    .venv/bin/python experiments/verify_moe_slotting.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from model.config import VibyConfig
from model.moe import MoEFeedForward

B, T, D, E, I, K = 4, 2048, 768, 112, 104, 6  # noqa: E741


def _force_id_grouping(moe):
    """把容量表改回「按专家 id 成组」的旧算法（容量同样由逐专家峰值派生），
    使两种装槽方式的对比是自洽的：只差分组方式，不差容量推导规则。"""
    EG = moe._group_size(E)
    n_groups = (E + EG - 1) // EG
    for k in list(moe._cap_peak):
        peaks = moe._cap_peak[k]
        moe._cap_order[k] = list(range(E))
        moe._cap_perm[k] = list(range(E))
        moe._cap_table[k] = [
            moe._cap_from_peak(max(peaks[gi * EG : (gi + 1) * EG]))
            for gi in range(n_groups)
        ]


def build(skew: float, seed: int = 0):
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
    mx.random.seed(seed)
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

    # 固定倾斜路由：expert 概率 ∝ i^-skew，热专家散布到随机 id
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, E + 1) ** skew
    p = p[rng.permutation(E)]
    p /= p.sum()
    idx = np.stack([rng.choice(E, K, replace=False, p=p) for _ in range(B * T)])
    idx = mx.array(idx.reshape(B, T, K).astype(np.int32))
    w = mx.array(rng.random((B, T, K)).astype(np.float32)).astype(mx.bfloat16)

    def stub(x):
        return idx, w

    moe.router = stub  # type: ignore[assignment]
    return moe


def run(moe, identity: bool, n: int = 8):
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
        return (moe(x).astype(mx.float32) * C).sum()

    v = g = None
    for _ in range(n):
        v, g = mx.value_and_grad(loss)(p)
        mx.eval(v, g)
        moe.update_capacity_table()
        if identity:
            _force_id_grouping(moe)
    grads = {k: np.array(a.astype(mx.float32)) for k, a in tree_flatten(g)}
    rows = sum(moe._cap_table[0]) * moe._group_size(E)
    return float(v), grads, rows


def main():
    ok = True
    for name, forward in (("融合 kernel", "0"), ("MLX GEMM", "1")):
        os.environ["VIBY_MOE_MLX_FWD"] = forward
        MoEFeedForward._MLX_FORWARD = forward
        print(f"\n=== 前向路径: {name} ===")
        for skew in (0.0, 0.6, 1.2):
            moe = build(skew)
            G = B * T * K
            v1, g1, r1 = run(moe, identity=True)
            v2, g2, r2 = run(moe, identity=False)
            dv = abs(v1 - v2) / max(abs(v1), 1e-9)
            dg = max(
                float(np.abs(g1[k] - g2[k]).max() / (np.abs(g1[k]).max() + 1e-9))
                for k in g1
            )
            good = dv == 0.0 and dg == 0.0
            ok &= good
            print(
                f"  skew={skew:<4} padding {r1 / G:.2f}x → {r2 / G:.2f}x "
                f"({r1 / r2:.2f}x 少算)   loss差 {dv:.1e} 梯度差 {dg:.1e} "
                f"{'[OK]' if good else '[FAIL]'}"
            )
    print("\n全部逐位一致" if ok else "\n存在数值差异，需排查")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
