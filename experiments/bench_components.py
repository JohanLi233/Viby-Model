"""组件级 fwd / fwd+bwd 微基准（真实训练形状，当前 KDA+GQA 架构）。

逐组件计时，再按调用次数外推到整步，定位优化杠杆。
形状默认 r081 重配置：B=12 T=1024 D=768 H=8 E=256×I=320 top8 V=6400，
可用 VIBY_BENCH_* 环境变量覆盖。

用法: .venv/bin/python experiments/bench_components.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.attention import GQAAttention
from model.block import VibyBlock
from model.config import VibyConfig
from model.kda import KDAAttention
from model.kernels.ce import cross_entropy
from model.moe import MoEFeedForward


def _env(name, default):
    return int(os.environ.get(name, default))


B = _env("VIBY_BENCH_B", 12)
T = _env("VIBY_BENCH_T", 1024)
D = _env("VIBY_BENCH_D", 768)
E = _env("VIBY_BENCH_E", 256)
I = _env("VIBY_BENCH_I", 320)  # noqa: E741
K = _env("VIBY_BENCH_K", 8)
V = _env("VIBY_BENCH_V", 6400)
L = _env("VIBY_BENCH_LAYERS", 8)


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
        num_hidden_layers=L,
        num_attention_heads=8,
        vocab_size=V,
        max_position_embeddings=T,
        use_attn_gate=True,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=2,
        moe_intermediate_size=I,
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


def _seg_ids():
    return mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )


def bench_module(mod, iters, label, **kw):
    """对 (B,T,D) 输入的模块测 fwd / fwd+bwd（随机 cotangent 标量化）。"""
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    C = mx.random.normal((B, T, D)) * 0.5

    def loss(x_, p):
        mod.update(p)
        out = mod(x_, **kw)
        if isinstance(out, tuple):
            out = out[0]
        return (out.astype(mx.float32) * C).sum()

    p = mod.trainable_parameters()
    vg = mx.value_and_grad(loss, argnums=(0, 1))
    f = timed(lambda: loss(x, p), iters)
    fb = timed(lambda: vg(x, p), iters)
    print(f"  {label:<14} fwd {f * 1e3:7.1f}ms | fwd+bwd {fb * 1e3:7.1f}ms")
    return f, fb


def bench_ce(iters):
    h = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    W = (mx.random.normal((V, D)) * 0.02).astype(mx.bfloat16)
    labels = mx.random.randint(0, V, (B, T))
    mask = mx.ones((B, T), dtype=mx.int64)

    def loss(h_, W_):
        return cross_entropy(h_ @ W_.T, labels, mask=mask)

    vg = mx.value_and_grad(loss, argnums=(0, 1))
    f = timed(lambda: loss(h, W), iters)
    fb = timed(lambda: vg(h, W), iters)
    print(f"  {'lm_head+CE':<14} fwd {f * 1e3:7.1f}ms | fwd+bwd {fb * 1e3:7.1f}ms")
    return f, fb


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cfg = make_config()
    seg = _seg_ids()
    print(f"形状: B={B} T={T} D={D} | E={E} I={I} K={K} | L={L}\n")

    kda = to_bf16(KDAAttention(cfg, layer_idx=0))
    kf, kfb = bench_module(kda, iters, "KDA attn", segment_ids=seg)

    gqa = to_bf16(GQAAttention(cfg, layer_idx=3))
    gf, gfb = bench_module(gqa, iters, "GQA attn(global)", segment_ids=seg)

    moe = to_bf16(MoEFeedForward(cfg))
    moe.train()
    mf, mfb = bench_module(moe, iters, "MoE")

    blk_k = to_bf16(VibyBlock(cfg, layer_idx=0))  # local KDA block
    bkf, bkb = bench_module(blk_k, iters, "Block(KDA)", segment_ids=seg)
    blk_g = to_bf16(VibyBlock(cfg, layer_idx=3))  # global GQA block
    bgf, bgb = bench_module(blk_g, iters, "Block(GQA)", segment_ids=seg)

    cf, cfb = bench_ce(iters)

    # 外推：L 层中 global 层数 = (L+1)//4 + 1（每 4 层 + 末层，去重）
    n_global = len({i for i in range(L) if (i + 1) % 4 == 0} | {L - 1})
    n_local = L - n_global
    n_moe = L + 1  # 主干 + 1 MTP block
    print(f"\n=== 外推整步（{n_local} KDA + {n_global} GQA block + MTP + CE×2）===")
    parts = {
        "KDA attn": kfb * (n_local + 1),
        "GQA attn": gfb * n_global,
        "MoE": mfb * n_moe,
        "lm+CE": cfb * 2,
    }
    tot = sum(parts.values())
    for k, v in parts.items():
        print(f"  {k:<10} : {v * 1e3:7.1f}ms  ({v / tot * 100:4.1f}%)")
    print(f"  {'组件和':<10} : {tot * 1e3:7.1f}ms")
    blk_sum = bkb * (n_local + 1) + bgb * n_global + cfb * 2
    print(f"  Block 口径 : {blk_sum * 1e3:7.1f}ms（含 norm/AttnRes/ShortConv）")


if __name__ == "__main__":
    main()
