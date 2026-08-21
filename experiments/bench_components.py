"""组件级 fwd / fwd+bwd 微基准（真实训练形状）。

按 r073 配置（bs6×seq2048、D=768、8 heads、E=112×I=104 top6、8 个 HRM
cycle）逐组件计时，再按调用次数外推到整步，定位真正的优化杠杆。

用法: .venv/bin/python experiments/bench_components.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.model import (
    Attention,
    Engram,
    MoEFeedForward,
    VibyConfig,
    cross_entropy,
)

B, T, D, E, I, K = 6, 2048, 768, 112, 104, 6
V = 6400
N_CYCLES = 8  # H2 × (L3+1)


def timed(fn, iters=6, warm=2):
    for _ in range(warm):
        out = fn()
        mx.eval(out)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def make_config(**kw):
    base = dict(
        hidden_size=D,
        num_hidden_layers=1,
        num_attention_heads=8,
        kv_lora_rank=192,
        qk_rope_head_dim=32,
        vocab_size=V,
        max_position_embeddings=T,
        use_value_res=True,
        use_attn_gate=True,
        hrm_H_cycles=2,
        hrm_L_cycles=3,
        hrm_cycle_router=1,
        hrm_cycle_router_rank=8,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=I,
        engram_orders=(2, 3),
        engram_slots=8192,
        engram_sub_dim=128,
    )
    base.update(kw)
    return VibyConfig(**base)


def to_bf16(mod):
    from mlx.utils import tree_map

    mod.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            mod.parameters(),
        )
    )
    mx.eval(mod.parameters())
    return mod


def bench_attention(iters):
    cfg = make_config()
    attn = to_bf16(Attention(cfg))
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    cos = mx.zeros((T, 32), dtype=mx.bfloat16)
    sin = mx.zeros((T, 32), dtype=mx.bfloat16)
    freqs = mx.ones((16,))
    pos = (cos, sin, (freqs, 0, 1.0))
    C = mx.random.normal((B, T, D)) * 0.5

    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(bias)

    def mk(mask_is_full, cb):
        def loss(x_, p):
            attn.update(p)
            o, _ = attn(x_, pos, causal_bias=cb, mask_is_full=mask_is_full)
            return (o.astype(mx.float32) * C).sum()

        return loss

    out = {}
    for label, mif, cb in [("纯causal", True, None), ("doc_mask", False, bias)]:
        loss = mk(mif, cb)
        p = attn.trainable_parameters()
        vg = mx.value_and_grad(loss, argnums=(0, 1))
        f = timed(lambda: loss(x, p), iters)
        fb = timed(lambda: vg(x, p), iters)
        out[label] = (f, fb)
    return out


def bench_moe(iters, conc=3.3):
    cfg = make_config()
    moe = to_bf16(MoEFeedForward(cfg))
    moe.train()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5

    def loss(x_, p):
        moe.update(p)
        return (moe(x_, step_idx=0).astype(mx.float32) * C).sum()

    p = moe.trainable_parameters()
    vg = mx.value_and_grad(loss, argnums=(0, 1))
    # 先跑一次让容量表按实测计数收敛
    for _ in range(3):
        mx.eval(loss(x, p))
        for m in [moe]:
            m.update_capacity_table()
    f = timed(lambda: loss(x, p), iters)
    fb = timed(lambda: vg(x, p), iters)
    rows = sum(moe._cap_table.get(0, [0])) * min(moe._SPARSE_GROUP, E)
    return f, fb, rows


def bench_engram(iters):
    cfg = make_config()
    eng = to_bf16(Engram(cfg))
    ids = mx.random.randint(1, V, (B, T))
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    h = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5

    def loss(h_, p):
        eng.update(p)
        read = eng(ids, None, seg)
        return (read.apply(h_).astype(mx.float32) * C).sum()

    p = eng.trainable_parameters()
    vg = mx.value_and_grad(loss, argnums=(0, 1))
    return timed(lambda: loss(h, p), iters), timed(lambda: vg(h, p), iters)


def bench_ce(iters):
    h = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    W = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
    labels = mx.random.randint(0, V, (B, T))
    mask = mx.ones((B, T), dtype=mx.int64)

    def loss(h_, W_):
        return cross_entropy(h_ @ W_.T, labels, mask=mask)

    vg = mx.value_and_grad(loss, argnums=(0, 1))
    return timed(lambda: loss(h, W), iters), timed(lambda: vg(h, W), iters)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"形状: B={B} T={T} D={D} | E={E} I={I} K={K} | cycles={N_CYCLES}\n")

    a = bench_attention(iters)
    print("MLA attention（单次调用）")
    for k, (f, fb) in a.items():
        print(f"  {k:<10} fwd {f * 1e3:7.1f}ms | fwd+bwd {fb * 1e3:7.1f}ms")

    mf, mfb, rows = bench_moe(iters)
    print(
        f"\nMoE 层（单次调用, 桶行数={rows}, 真实pair={B * T * K}, "
        f"padding {rows / (B * T * K):.2f}×）"
    )
    print(f"  fwd {mf * 1e3:7.1f}ms | fwd+bwd {mfb * 1e3:7.1f}ms")

    ef, efb = bench_engram(iters)
    print(f"\nEngram（单次）\n  fwd {ef * 1e3:7.1f}ms | fwd+bwd {efb * 1e3:7.1f}ms")

    cf, cfb = bench_ce(iters)
    print(f"\nlm_head+CE（单次）\n  fwd {cf * 1e3:7.1f}ms | fwd+bwd {cfb * 1e3:7.1f}ms")

    print("\n=== 外推到整步（8 cycle + 1 MTP block）===")
    attn_fb = a["doc_mask"][1]
    tot = attn_fb * 9 + mfb * 9 + efb + cfb * 2
    print(
        f"  attention ×9 : {attn_fb * 9 * 1e3:7.1f}ms  ({attn_fb * 9 / tot * 100:4.1f}%)"
    )
    print(f"  MoE       ×9 : {mfb * 9 * 1e3:7.1f}ms  ({mfb * 9 / tot * 100:4.1f}%)")
    print(f"  engram    ×1 : {efb * 1e3:7.1f}ms  ({efb / tot * 100:4.1f}%)")
    print(f"  lm+CE     ×2 : {cfb * 2 * 1e3:7.1f}ms  ({cfb * 2 / tot * 100:4.1f}%)")
    print(f"  合计         : {tot * 1e3:7.1f}ms")
    print(
        f"\n  doc_mask 相对纯 causal 的 attention 额外开销: "
        f"{(a['doc_mask'][1] - a['纯causal'][1]) * 9 * 1e3:.1f}ms/步"
    )


if __name__ == "__main__":
    main()
