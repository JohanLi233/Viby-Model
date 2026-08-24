"""新架构（GQA+AttnRes+LatentMoE+MTP）训练单步组件归因探针。

按推荐配置 bs12×seq1024、D=768、8 层、E=256/I=320/K=8、latent=384、
shared=2、vocab=6400、mtp=1，逐组件计时 fwd / fwd+bwd，把整步墙钟
归因到：注意力 / MoE / AttnRes 合并 / 主干其余 / lm_head+CE / MTP。

用法: .venv/bin/python experiments/probe_breakdown_v2.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.block import _attn_res_merge
from model.config import VibyConfig
from model.model import VibyForCausalLM

B, T, D, V = 12, 1024, 768, 6400
LAYERS, E, MI, K = 8, 256, 320, 8
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def cfg_of(**kw):
    base = dict(
        hidden_size=D,
        num_hidden_layers=LAYERS,
        num_attention_heads=8,
        vocab_size=V,
        max_position_embeddings=T,
        mtp_depth=1,
        mtp_loss_weight=0.3,
        use_attn_gate=True,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=2,
        moe_intermediate_size=MI,
        routed_scaling_factor=2.5,
    )
    base.update(kw)
    return VibyConfig(**base)


def to_bf16(mod):
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


def timed(fn, iters=ITERS, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3  # ms


def make_batch():
    X = mx.random.randint(1, V, (B, T))
    Y = mx.random.randint(1, V, (B, T))
    mask = mx.ones((B, T), dtype=mx.int32)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    return X, Y, mask, seg


X, Y, mask, seg = make_batch()


def bench_model(mtp_depth, label):
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=mtp_depth)))
    model.train()
    for g in model.moe_gates():
        g.collect_stats = True

    def fwd():
        return model(input_ids=X, labels=Y, loss_mask=mask, segment_ids=seg).loss

    params = model.trainable_parameters()

    def fb(params_):
        model.update(params_)
        return model(input_ids=X, labels=Y, loss_mask=mask, segment_ids=seg).loss

    vg = mx.value_and_grad(fb)
    t_f = timed(fwd)
    t_fb = timed(lambda: vg(params, X)[0] if False else vg(params))
    print(f"{label:<28} fwd {t_f:8.1f}ms   fwd+bwd {t_fb:8.1f}ms")
    return t_f, t_fb


def bench_stack():
    """主干（embed+8 层+final_norm），不含 lm_head/CE/MTP。"""
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=0)))
    model.train()
    for g in model.moe_gates():
        g.collect_stats = True

    def fwd():
        h, _, _ = model.model(input_ids=X, segment_ids=seg)
        return h.sum()

    params = model.trainable_parameters()
    vg = mx.value_and_grad(
        lambda p: (model.update(p), model.model(input_ids=X, segment_ids=seg)[0].sum())[
            1
        ]
    )
    t_f = timed(fwd)
    t_fb = timed(lambda: vg(params))
    print(f"{'stack(embed+8L+norm)':<28} fwd {t_f:8.1f}ms   fwd+bwd {t_fb:8.1f}ms")
    return t_f, t_fb


def bench_block(layer_idx, label):
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=0)))
    blk = model.model.stack.layers[layer_idx]
    blk.train()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)

    def fwd():
        h, _ = blk(x, segment_ids=seg)
        return h.sum()

    params = blk.trainable_parameters()
    vg = mx.value_and_grad(
        lambda p: (blk.update(p), blk(x, segment_ids=seg)[0].sum())[1]
    )
    t_f = timed(fwd)
    t_fb = timed(lambda: vg(params))
    print(f"{label:<28} fwd {t_f:8.1f}ms   fwd+bwd {t_fb:8.1f}ms")


def bench_parts():
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=0)))
    attn = model.model.stack.layers[0].self_attn  # local 层（KDA）
    gattn = model.model.stack.layers[3].self_attn  # global 层
    mlp = model.model.stack.layers[0].mlp
    for m in (attn, gattn, mlp):
        m.train()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)

    t = timed(lambda: attn(x, segment_ids=seg)[0].sum())
    p = attn.trainable_parameters()
    vg = mx.value_and_grad(
        lambda pp: (attn.update(pp), attn(x, segment_ids=seg)[0].sum())[1]
    )
    tb = timed(lambda: vg(p))
    print(f"{'attn local ×8 (外推)':<28} fwd {t * 8:8.1f}ms   fwd+bwd {tb * 8:8.1f}ms")

    t = timed(lambda: gattn(x, segment_ids=seg)[0].sum())
    p = gattn.trainable_parameters()
    vg = mx.value_and_grad(
        lambda pp: (gattn.update(pp), gattn(x, segment_ids=seg)[0].sum())[1]
    )
    tb = timed(lambda: vg(p))
    print(f"{'attn global ×1 (外推)':<28} fwd {t:8.1f}ms   fwd+bwd {tb:8.1f}ms")

    t = timed(lambda: mlp(x).sum())
    p = mlp.trainable_parameters()
    vg = mx.value_and_grad(lambda pp: (mlp.update(pp), mlp(x).sum())[1])
    tb = timed(lambda: vg(p))
    print(f"{'mlp(MoE) ×8 (外推)':<28} fwd {t * 8:8.1f}ms   fwd+bwd {tb * 8:8.1f}ms")


def bench_attn_res():
    """模拟整步 17 次 AttnRes 合并（block i: N=2i+2 / 2i+3）。"""
    ws = [mx.zeros((D,), dtype=mx.bfloat16) for _ in range(2 * LAYERS)]
    vs_pool = [
        (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
        for _ in range(2 * LAYERS + 1)
    ]

    def all_merges():
        out = 0
        for i in range(LAYERS):
            n1, n2 = 2 * i + 2, 2 * i + 3
            out = out + _attn_res_merge(ws[2 * i], vs_pool[:n1]).sum()
            out = out + _attn_res_merge(ws[2 * i + 1], vs_pool[:n2]).sum()
        return out

    t = timed(all_merges)
    vg = mx.value_and_grad(lambda w: _attn_res_merge(w, vs_pool[:17]).sum())
    tb = sum(timed(lambda: vg(w)) for w in ws[:4])  # 抽样 4 个大 N 合并
    print(f"{'AttnRes 17 次合并':<28} fwd {t:8.1f}ms   bwd(抽样×4) {tb:8.1f}ms")


def bench_lm_ce():
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=0)))
    h = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)

    from model.kernels.ce import cross_entropy

    def fwd():
        logits = model._lm_logits(h)
        loss, z = cross_entropy(logits, Y, mask=mask, return_z=True)
        return loss + 1e-4 * z

    params = model.lm_head.trainable_parameters()
    vg = mx.value_and_grad(lambda p: (model.lm_head.update(p), fwd())[1])
    t = timed(fwd)
    tb = timed(lambda: vg(params))
    print(f"{'lm_head+CE(主)':<28} fwd {t:8.1f}ms   fwd+bwd {tb:8.1f}ms")


def bench_mtp():
    model = to_bf16(VibyForCausalLM(cfg_of(mtp_depth=1)))
    model.train()
    feats = [(mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16) for _ in range(3)]

    def fwd():
        return model._mtp_loss(feats, X, Y, mask, None, True, seg)[0]

    t = timed(fwd)
    print(f"{'MTP(depth1, 含 lm_head+CE)':<28} fwd {t:8.1f}ms")


print(f"== 组件归因 bs{B}×{T} D{D} L{LAYERS} E{E} I{MI} K{K} V{V} (min of {ITERS}) ==")
t_full_f, t_full_fb = bench_model(1, "完整模型 (mtp=1)")
t_nm_f, t_nm_fb = bench_model(0, "完整模型 (mtp=0)")
bench_stack()
bench_block(0, "单层 block (local) ×1")
bench_parts()
bench_attn_res()
bench_lm_ce()
bench_mtp()
print(f"\n参考: 整步 fwd {t_full_f:.0f}ms / fwd+bwd {t_full_fb:.0f}ms")
print(
    f"MTP 开销 ≈ fwd +{t_full_f - t_nm_f:.0f}ms / fwd+bwd +{t_full_fb - t_nm_fb:.0f}ms"
)
