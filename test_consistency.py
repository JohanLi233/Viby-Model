"""架构正确性回归测试（不依赖训练数据，随机初始化的小模型即可运行）。

模型为顺序 MoE 主干：KDA local 层 + GQA global 层混合（Kimi Linear 3:1，
全模型无 RoPE）+ ShortConv + AttnRes + LatentMoE（QB 路由）。覆盖：
1. 因果性：prefill T 与 prefill T+K 的前 T 个 logits 必须一致；
2. prefill == 分段 prefill == 逐 token decode（use_cache 一致性）；
3. padding 等价性：左/右 padding 下有效位置的 logits 与无 padding 一致；
4. SFT / DPO loss mask：从首个内容 token 监督到 <|im_end|> 为止，
   不多掩码下一轮的 <|im_start|>；
5. MTP（EAGLE-3 口径）草稿 cache / TTT 草稿训练 / MoE 三路径 /
   LatentMoE / QB 偏置快照专项。

运行：python test_consistency.py
"""

import sys

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from model.cache import KVCache
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.moe import MoEFeedForward

ATOL = 2e-3  # float32 下不同分块/增量路径的浮点误差上限

# 前向类测试覆盖的架构变体（tiny_config 基座已是顺序 MoE 主干，这里只叠加开关）
ARCH_VARIANTS = {
    # 基线：KDA local + GQA global 混合注意力 + 全 MoE
    "moe": {},
    # 基线机制组合：attn_gate + MTP
    "moe+gate+mtp": {
        "use_attn_gate": True,  # 与默认相同，显式钉住
        "mtp_depth": 1,
    },
}


def tiny_config(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=512,
        mtp_depth=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        # moe_latent_dim 默认 None→hidden//2（开启）；既有测试的逐专家
        # 参考按全宽专家计算，显式关闭，latent 路径由 test_moe_latent 覆盖
        moe_latent_dim=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def make_model(**kw):
    mx.random.seed(42)
    model = VibyForCausalLM(tiny_config(**kw))
    model.eval()
    return model


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def rand_ids(T, B=1, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(3, 256, (B, T)).astype(np.int64))


def test_prefill_causality():
    """同一前缀，追加未来 token 后，前缀位置的 logits 不得改变。"""
    for name, kw in ARCH_VARIANTS.items():
        model = make_model(**kw)
        ids = rand_ids(40, seed=1)
        extra = rand_ids(24, seed=2)
        out_short = model(ids).logits
        out_long = model(mx.concatenate([ids, extra], axis=1)).logits[:, :40]
        d = maxdiff(out_short, out_long)
        assert d < ATOL, (
            f"[{name}] 因果性破坏：追加未来 token 后前缀 logits 最大偏差 {d:.4f}"
        )


def test_prefill_chunk_decode_consistency():
    """一次性 prefill == 分段 prefill == 逐 token decode。"""
    for name, kw in ARCH_VARIANTS.items():
        model = make_model(**kw)
        ids = rand_ids(40, seed=3)
        full = model(ids).logits

        # 分段 prefill（17 + 13 + 10）
        past = None
        outs = []
        for s, e in [(0, 17), (17, 30), (30, 40)]:
            o = model(ids[:, s:e], past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(o.logits)
        chunked = mx.concatenate(outs, axis=1)
        d_chunk = maxdiff(full, chunked)
        assert d_chunk < ATOL, f"[{name}] 分段 prefill 不一致：{d_chunk:.4f}"

        # 逐 token decode
        past = None
        outs = []
        for t in range(40):
            o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values
            outs.append(o.logits)
        decoded = mx.concatenate(outs, axis=1)
        d_dec = maxdiff(full, decoded)
        assert d_dec < ATOL, f"[{name}] 逐 token decode 不一致：{d_dec:.4f}"


def test_kvcache_rewind_clamp():
    """投机 EOS tail 会 rewind 到超过当前 offset；对齐旧 tuple 切片语义。"""
    cache = KVCache(chunk=8)
    cache.update(mx.ones((1, 5, 2, 4)), mx.ones((1, 5, 2, 3)))
    assert cache.offset == 5
    cache.rewind(3)
    assert cache.offset == 3
    cache.rewind(88)
    assert cache.offset == 3


def test_mtp_draft_cache_consistency():
    """MTP 草稿 KV cache：推理口径（prefill 建缓存 + 单位置草稿步 +
    批量追平）必须与训练口径（整流因果注意力，_mtp_loss 同约定）逐位一致。

    EAGLE-3 口径下流位置 t 消费 (feats_low/mid/high[t], emb(token t+1))；
    训练时 MTP block 对整流做因果注意力，推理草稿若丢弃上下文（无 cache
    的单位置调用），草稿分布显著偏离（r060 实测 argmax 命中率 45%→34%）。
    """
    variants = {
        "moe+mtp": dict(mtp_depth=1),
        "moe+gate+mtp": dict(use_attn_gate=True, mtp_depth=1),
    }
    for name, kw in variants.items():
        model = make_model(**kw)
        ids = rand_ids(33, seed=21)
        feats = model(ids, output_features=True).features  # 3 × (1, T, D)
        assert len(feats) == 3, f"[{name}] 多层特征应为 3 层: {len(feats)}"
        T = ids.shape[1]
        emb = model.model.embed_tokens(ids)
        mtp = model.mtp_modules[0]

        # 训练口径：整流一次前向（因果注意力）。流位置 t 消费
        # (feats[*][t], emb(token t+1))，logits_full[t] 预测 token t+2。
        sub = T - 1
        h_full, _ = mtp([f[:, :sub] for f in feats], emb[:, 1:])
        logits_full = model._lm_logits(h_full)[0]  # (sub, V)

        # 推理口径 1（草稿步）：prefill 建 cache 到 sub-2，再单位置草稿
        _, mtp_past = mtp(
            [f[:, : sub - 1] for f in feats],
            emb[:, 1:sub],
            use_cache=True,
        )
        h_step, _ = mtp(
            [f[:, sub - 1 : sub] for f in feats],
            emb[:, sub : sub + 1],
            past_key_value=mtp_past,
            use_cache=True,
        )
        d = maxdiff(model._lm_logits(h_step)[0, 0], logits_full[sub - 1])
        assert d < ATOL, f"[{name}] 带 cache 草稿步与训练口径不符: {d:.4f}"

        # 推理口径 2（验证后批量追平）：prefill 到 sub-3，再批量追加 2 个位置
        _, mtp_past2 = mtp(
            [f[:, : sub - 2] for f in feats],
            emb[:, 1 : sub - 1],
            use_cache=True,
        )
        h_cat, _ = mtp(
            [f[:, sub - 2 : sub] for f in feats],
            emb[:, sub - 1 : sub + 1],
            past_key_value=mtp_past2,
            use_cache=True,
        )
        d0 = maxdiff(model._lm_logits(h_cat)[0, 0], logits_full[sub - 2])
        d1 = maxdiff(model._lm_logits(h_cat)[0, 1], logits_full[sub - 1])
        assert max(d0, d1) < ATOL, (
            f"[{name}] 批量追平与训练口径不符: {d0:.4f}, {d1:.4f}"
        )


def test_draft_ttt():
    """EAGLE-3 TTT 草稿训练：冻结后仅 MTP 参数收梯度；TTT step 1 与
    _mtp_loss depth-1 口径一致；多步 rollout loss 有限。"""
    from trainer.train_draft import draft_ttt_loss

    model = make_model(mtp_depth=1)
    model.freeze()
    for m in model.mtp_modules:
        m.unfreeze()
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all(k.startswith("mtp_modules.") for k in trainable), (
        "冻结后应只剩 MTP 参数可训练"
    )

    ids = rand_ids(24, B=2, seed=5)
    Y = mx.concatenate([ids[:, 1:], mx.zeros((2, 1), dtype=ids.dtype)], axis=1)
    loss_mask = mx.ones_like(ids)
    attn_mask = mx.ones((2, 24), dtype=mx.int32)

    # TTT step 1 与预训练 _mtp_loss depth-1 完全同口径（特征输入 +
    # 真实下一 token 嵌入 → 预测 t+2）
    feats = model(ids, output_features=True).features
    ce_ref, _ = model._mtp_loss(feats, ids, Y, loss_mask, attn_mask, False)
    ce1 = draft_ttt_loss(model, ids, Y, loss_mask, attn_mask, False, ttt_steps=1)
    d = maxdiff(ce_ref, ce1)
    assert d < 1e-5, f"TTT step1 与 _mtp_loss depth-1 不符: {d}"

    # 多步 rollout：loss 有限，且梯度只落在 MTP 参数上。
    # make_model() 默认 eval()，会走 GatedNorm / SiTU-GLU / MoE 的推理
    # Metal kernel（无 VJP）。真实训练循环会 model.train()，那些
    # kernel 一律旁路到可微 eager / 稀疏路径。
    model.train()

    def loss_fn(p):
        model.update(p)
        return draft_ttt_loss(model, ids, Y, loss_mask, attn_mask, False, ttt_steps=4)

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    assert np.isfinite(float(val)) and float(val) > 0, f"TTT loss 非有限: {val}"
    gflat = dict(tree_flatten(grads))
    assert gflat and all(k.startswith("mtp_modules.") for k in gflat), (
        "冻结后仅 MTP 参数应收梯度"
    )
    for key in (
        "mtp_modules.0.fc_l.weight",
        "mtp_modules.0.proj.weight",
        "mtp_modules.0.block.mlp.router.weight",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"{key} 应收到非零梯度"


def test_pad_equivalence():
    """padding 不得影响有效位置的输出（左/右 padding 均需与无 padding 一致）。"""
    ids = rand_ids(24, seed=4)
    pad_tail = mx.zeros((1, 2), dtype=ids.dtype)
    right_ids = mx.concatenate([ids, pad_tail], axis=1)
    right_mask = mx.concatenate(
        [mx.ones((1, 24), dtype=mx.int32), mx.zeros((1, 2), dtype=mx.int32)], axis=1
    )
    left_ids = mx.concatenate([pad_tail, ids], axis=1)
    left_mask = mx.concatenate(
        [mx.zeros((1, 2), dtype=mx.int32), mx.ones((1, 24), dtype=mx.int32)], axis=1
    )

    for name, kw in ARCH_VARIANTS.items():
        model = make_model(**kw)
        out_ref = model(ids).logits

        out_pad = model(right_ids, attention_mask=right_mask).logits[:, :24]
        d = maxdiff(out_ref, out_pad)
        assert d < ATOL, f"[{name}] 右 padding 污染有效位置：最大偏差 {d:.4f}"

        out_pad = model(left_ids, attention_mask=left_mask).logits[:, 2:]
        d = maxdiff(out_ref, out_pad)
        assert d < ATOL, f"[{name}] 左 padding 污染有效位置：最大偏差 {d:.4f}"


def test_loss_masks():
    """SFT/DPO loss mask：监督范围为 [首个内容 token, <|im_end|>]。"""
    from dataset.lm_dataset import SFTDataset, DPODataset

    for cls in (SFTDataset, DPODataset):
        ds = cls.__new__(cls)
        ds.bos_id = [99, 98]  # "<|im_start|>assistant"
        ds.eos_id = [97]  # "<|im_end|>"
        ds.max_length = 64
        ids = [1, 99, 98, 11, 12, 13, 97, 2, 99, 98, 21, 97]
        # 第一轮内容: idx 3..6 (11,12,13,<|im_end|>)，第二轮: idx 10..11
        expect = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
        got = ds._generate_loss_mask(ids)
        assert got == expect, f"{cls.__name__} loss mask 错误: {got} != {expect}"


def test_decode_with_pad():
    """带 padding decode 的一致性：

    1) 相同内容不同左 pad 的 batch 行，prefill 有效位置 logits 一致；
    2) prefill 后逐 token decode 不再因 causal_bias 的 key 维仍指向全量
       cache 而广播失败（修复回归）。
    """
    model = make_model()
    content = rand_ids(30, seed=31)
    total_len = 40
    left_pads = [3, 7]
    ids = mx.concatenate(
        [
            mx.concatenate(
                [
                    mx.zeros((1, p), dtype=content.dtype),
                    content,
                    mx.zeros(
                        (1, total_len - p - content.shape[1]),
                        dtype=content.dtype,
                    ),
                ],
                axis=1,
            )
            for p in left_pads
        ],
        axis=0,
    )
    mask = mx.concatenate(
        [
            mx.concatenate(
                [
                    mx.zeros((1, p), dtype=mx.int32),
                    mx.ones((1, content.shape[1]), dtype=mx.int32),
                    mx.zeros(
                        (1, total_len - p - content.shape[1]),
                        dtype=mx.int32,
                    ),
                ],
                axis=1,
            )
            for p in left_pads
        ],
        axis=0,
    )

    out = model(ids, attention_mask=mask, use_cache=True)
    mx.eval(out.logits, out.past_key_values)
    d_prefill = maxdiff(
        out.logits[0, left_pads[0] : left_pads[0] + content.shape[1]],
        out.logits[1, left_pads[1] : left_pads[1] + content.shape[1]],
    )
    assert d_prefill < ATOL, f"prefill 受 padding 影响：{d_prefill:.4f}"

    next_ids = mx.concatenate([ids, mx.array([[7], [7]], dtype=ids.dtype)], axis=1)
    next_mask = mx.concatenate([mask, mx.ones((2, 1), dtype=mask.dtype)], axis=1)
    dec = model(
        next_ids[:, -1:],
        attention_mask=next_mask,
        past_key_values=out.past_key_values,
        use_cache=True,
    )
    mx.eval(dec.logits)
    assert bool(mx.all(mx.isfinite(dec.logits)).item()), "decode 输出包含 NaN/Inf"


def test_moe():
    """DeepSeekMoE 专项：结构审计、逐专家参考等价、路由权重性质、
    偏置冻结/无辅助损失更新、负载统计、梯度可达性、MTP 同构。"""

    E, K, moe_in = 8, 2, 48
    kw = dict(
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=moe_in,
    )
    model = make_model(**kw)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 结构审计：主干所有层均为 MoE；专家权重为 (E,out,in) 堆叠
    layers = model.model.stack.layers
    assert all(isinstance(layer.mlp, MoEFeedForward) for layer in layers), (
        "所有层的 FFN 应为 MoE"
    )
    gw = flat["model.stack.layers.1.mlp.experts.gate_up_w"]
    assert gw.shape == (E, 2 * moe_in, 128), f"专家堆叠形状错误: {gw.shape}"
    # expert_bias 冻结（不进梯度/优化器），但随 checkpoint 持久化
    assert "model.stack.layers.1.mlp.router.expert_bias" not in flat, (
        "expert_bias 不应可训练"
    )
    allp = dict(tree_flatten(model.parameters()))
    assert "model.stack.layers.1.mlp.router.expert_bias" in allp, "expert_bias 应持久化"

    # 2) 路由参考等价：逐 (token, expert) 朴素循环 vs 融合 kernel 实现
    # （G=20 <= _KERNEL_MAX_PAIRS，mlp(x) 走 decode Metal kernel 路径）
    mlp = model.model.stack.layers[1].mlp
    x = mx.random.normal((2, 5, 128))
    out = mlp(x)
    idx, w = mlp.router(x)
    B, T, D = x.shape
    gu_ = np.array(mlp.experts.gate_up_w)  # (E, 2I, D)
    gw_, uw_ = gu_[:, :moe_in], gu_[:, moe_in:]
    dw_ = np.array(mlp.experts.down_w)
    xn, idxn, wn = np.array(x), np.array(idx), np.array(w)

    def situ(g, u, b1=4.0, b2=25.0):
        return (b1 * np.tanh(g / b1) / (1.0 + np.exp(-g))) * (b2 * np.tanh(u / b2))

    ref = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            for j in range(K):
                e = idxn[b, t, j]
                h = situ(xn[b, t] @ gw_[e].T, xn[b, t] @ uw_[e].T)
                ref[b, t] += wn[b, t, j] * (h @ dw_[e].T)
    sh = 0
    for ff in mlp.shared:
        sh = sh + ff(x)
    ref = mx.array(ref) + sh
    d = maxdiff(out, ref)
    assert d < 1e-4, f"kernel 融合 MoE 与逐专家参考不符: {d}"

    # 2b) 三路径等价：kernel（decode，上面已验）/ 稠密（小 prefill）/ 稀疏（训练）
    mlp._KERNEL_MAX_PAIRS = 0  # 强制稠密
    d = maxdiff(ref, mlp(x))
    assert d < 1e-4, f"稠密批量 MoE 与逐专家参考不符: {d}"
    mlp._DENSE_MAX_PAIRS = 0  # 强制稀疏
    d = maxdiff(ref, mlp(x))
    mlp._KERNEL_MAX_PAIRS = 512  # 还原默认
    mlp._DENSE_MAX_PAIRS = 4096
    assert d < 1e-4, f"稀疏分段 MoE 与逐专家参考不符: {d}"

    # 2c) bf16 下 kernel 路径（含 router kernel）与原生路径整体一致
    # （decode 实际运行 dtype；随机输入 tie 概率 0，选择集合应精确一致）
    saved = (mlp.experts.gate_up_w, mlp.experts.down_w, mlp.router.weight)
    mlp.experts.gate_up_w = mlp.experts.gate_up_w.astype(mx.bfloat16)
    mlp.experts.down_w = mlp.experts.down_w.astype(mx.bfloat16)
    mlp.router.weight = mlp.router.weight.astype(mx.bfloat16)
    xb = x.astype(mx.bfloat16)
    idxb, wb = mlp.router(xb)
    k_out = mlp._kernel_forward(xb)
    d_out = mlp._dense_forward(xb, idxb, wb)
    rel = maxdiff(k_out, d_out) / maxdiff(d_out, mx.zeros_like(d_out))
    assert rel < 2e-2, f"bf16 kernel 与稠密路径不符: rel={rel}"
    (mlp.experts.gate_up_w, mlp.experts.down_w, mlp.router.weight) = saved

    # 3) 路由权重性质：norm_topk_prob 下每 token 权重和 == routed_scaling_factor
    sums = np.array(w.sum(axis=-1))
    assert np.allclose(sums, 2.5, atol=1e-5), f"路由权重和应为 scaling factor: {sums}"

    # 4) 负载统计与 QB 偏置快照更新：bias = −(margin 的 (1−K/E) 上分位数)，
    #    零均值化。margin = 选择分 − per-token 阈值 alpha（训练模式才收集）。
    #    顺序主干下每个 gate 每次前向调用一次。
    gates = model.moe_gates()
    for g in gates:
        g.collect_stats = True
    ids = rand_ids(16, B=2, seed=7)
    model.train()
    model(ids, labels=ids)
    model.eval()
    stats = model.moe_load_stats()
    mx.eval(stats)
    assert stats.shape == (len(gates) * E,), f"负载统计形状错误: {stats.shape}"
    per = stats.reshape(len(gates), E)
    sums = sorted(float(per[i].sum()) for i in range(len(gates)))
    assert sums == [2 * 16 * K] * len(gates), (
        f"负载统计总量不符（每 gate 每前向 1 次调用）: {sums}"
    )
    margins = model.qb_margin_stats()
    mx.eval(margins)
    assert margins.shape == (len(gates), 2 * 16, E), (
        f"margin 统计形状错误: {margins.shape}"
    )
    model.update_moe_biases(margins)
    bias1 = np.array(gates[0].expert_bias)
    m0 = np.array(margins[0])  # (N, E)
    beta = np.quantile(m0, 1.0 - K / E, axis=0)
    expected = -(beta - beta.mean())
    assert np.allclose(bias1, expected, atol=1e-5), (
        f"QB 偏置应等于 −beta 零均值化: {bias1} vs {expected}"
    )
    assert abs(float(bias1.mean())) < 1e-6, "QB 偏置应保持零均值"
    order = np.argsort(beta)
    assert np.all(np.diff(bias1[order]) <= 1e-6), (
        "margin 分位数越大的专家 bias 应越小（−beta）"
    )
    assert gates[0].expert_bias.dtype == mx.float32, "bias 应保持 fp32"

    # 5) 偏置影响选择：某专家 bias 拉满后必被选中
    g0 = gates[0]
    g0.expert_bias = mx.zeros((E,)).at[3].add(100.0)
    idx2, _ = g0(x)
    assert bool(mx.all(mx.any(idx2 == 3, axis=-1)).item()), "大 bias 专家应必中"

    # 6) 梯度可达：router 与专家堆叠均收非零梯度。
    # 第 4 节以 eval() 收尾；eval 会走无 VJP 的推理 kernel（GatedNorm /
    # SiTU-GLU / MoE decode）。train() 后与真实训练循环同路径。
    model.train()

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.stack.layers.1.mlp.router.weight",
        "model.stack.layers.1.mlp.experts.gate_up_w",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"{key} 应收到非零梯度"

    # 7) MTP 块同构为 MoE（V3/V4 的 MTP 与主干同构）
    model_mtp = make_model(mtp_depth=1, **kw)
    assert isinstance(model_mtp.mtp_modules[0].block.mlp, MoEFeedForward), (
        "MTP 块应为 MoE 层"
    )

    # 8) MTP 开启时 QB margin 统计仍可 stack：MTP gate 少一个流位置
    #    （T−1 vs T），qb_margin_stats 统一裁到最小 N（回归：曾直接 stack
    #    报 shape 不一致）
    for g in model_mtp.moe_gates():
        g.collect_stats = True
    ids = rand_ids(16, B=2, seed=9)
    model_mtp.train()
    model_mtp(ids, labels=ids)
    model_mtp.eval()
    margins = model_mtp.qb_margin_stats()
    mx.eval(margins)
    assert margins.shape == (len(model_mtp.moe_gates()), 2 * 15, E), (
        f"MTP 下 margin 统计形状错误: {margins.shape}"
    )
    model_mtp.update_moe_biases(margins)
    bias = np.array(model_mtp.moe_gates()[0].expert_bias)
    assert np.isfinite(bias).all() and abs(float(bias.mean())) < 1e-6


def test_moe_latent():
    """Latent MoE（moe_latent_dim>0）：结构审计、三路径与逐专家参考等价
    （latent 投影含在内）、梯度可达、sidecar 往返。"""

    E, K, moe_in, D, d = 8, 2, 48, 128, 64
    kw = dict(
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=moe_in,
        moe_latent_dim=d,
    )
    model = make_model(**kw)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 结构审计：专家 in/out 维 = latent_dim；lat_down/lat_up 可训练且形
    #    状正确；共享专家保持全宽
    mlp = model.model.stack.layers[1].mlp
    assert mlp.experts.gate_up_w.shape == (E, 2 * moe_in, d), (
        f"latent 专家堆叠形状错误: {mlp.experts.gate_up_w.shape}"
    )
    assert mlp.experts.down_w.shape == (E, d, moe_in)
    assert flat["model.stack.layers.1.mlp.lat_down.weight"].shape == (d, D)
    assert flat["model.stack.layers.1.mlp.lat_up.weight"].shape == (D, d)
    assert isinstance(mlp.shared, list) and len(mlp.shared) == 1
    assert mlp.shared[0].gate_proj.weight.shape[0] == moe_in, "共享专家应保持全宽"
    assert mlp.latent_norm.weight.shape == (d,), "latent_norm gain 形状应为 (d,)"
    assert flat["model.stack.layers.1.mlp.latent_norm.weight"].shape == (d,)

    # 2) 逐专家参考（含 latent 投影 + latent_norm）vs kernel / 稠密 / 稀疏三路径
    x = mx.random.normal((2, 5, D))
    out_k = mlp(x)  # G=20 走 decode kernel 路径
    idx, w = mlp.router(x)
    B, T, _ = x.shape
    ld = np.array(mlp.lat_down.weight)  # (d, D)
    lu = np.array(mlp.lat_up.weight)  # (D, d)
    gu_ = np.array(mlp.experts.gate_up_w)  # (E, 2I, d)
    gw_, uw_ = gu_[:, :moe_in], gu_[:, moe_in:]
    dw_ = np.array(mlp.experts.down_w)  # (E, d, I)
    xl = np.array(x) @ ld.T  # (B,T,d)
    # latent_norm（gain 初始为 1）：x / sqrt(mean(x²) + eps)
    xn = xl / np.sqrt((xl**2).mean(-1, keepdims=True) + model.config.rms_norm_eps)
    idxn, wn = np.array(idx), np.array(w)

    def situ(g, u, b1=4.0, b2=25.0):
        return (b1 * np.tanh(g / b1) / (1.0 + np.exp(-g))) * (b2 * np.tanh(u / b2))

    ref = np.zeros((B, T, d), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            for j in range(K):
                e = idxn[b, t, j]
                h = situ(xn[b, t] @ gw_[e].T, xn[b, t] @ uw_[e].T)
                ref[b, t] += wn[b, t, j] * (h @ dw_[e].T)
    sh = 0
    for ff in mlp.shared:
        sh = sh + ff(x)
    ref = mx.array(ref @ lu.T) + sh
    dd = maxdiff(out_k, ref)
    assert dd < 1e-4, f"latent kernel 路径与逐专家参考不符: {dd}"

    mlp._KERNEL_MAX_PAIRS = 0  # 强制稠密
    dd = maxdiff(ref, mlp(x))
    assert dd < 1e-4, f"latent 稠密路径与参考不符: {dd}"
    mlp._DENSE_MAX_PAIRS = 0  # 强制稀疏
    dd = maxdiff(ref, mlp(x))
    mlp._KERNEL_MAX_PAIRS = 512
    mlp._DENSE_MAX_PAIRS = 4096
    assert dd < 1e-4, f"latent 稀疏路径与参考不符: {dd}"

    # 3) 梯度可达：lat_down/lat_up/专家/router 均收非零梯度。
    # make_model() 默认 eval()，推理 kernel 无 VJP；train() 后走可微路径。
    model.train()
    ids = rand_ids(16, B=2, seed=7)

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.stack.layers.1.mlp.lat_down.weight",
        "model.stack.layers.1.mlp.lat_up.weight",
        "model.stack.layers.1.mlp.latent_norm.weight",
        "model.stack.layers.1.mlp.experts.gate_up_w",
        "model.stack.layers.1.mlp.router.weight",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"{key} 应收到非零梯度"

    # 4) sidecar 往返保留 latent 配置；缺键时套用 __init__ 默认（hidden//2）
    cfg_dict = model.config.to_dict()
    assert cfg_dict["moe_latent_dim"] == d
    assert VibyConfig.from_dict(cfg_dict).moe_latent_dim == d
    old = {k: v for k, v in cfg_dict.items() if k != "moe_latent_dim"}
    assert VibyConfig.from_dict(old).moe_latent_dim == D // 2, (
        "缺键应回退 __init__ 默认 hidden_size//2"
    )


def main():
    tests = [
        test_prefill_causality,
        test_prefill_chunk_decode_consistency,
        test_pad_equivalence,
        test_decode_with_pad,
        test_loss_masks,
        test_mtp_draft_cache_consistency,
        test_draft_ttt,
        test_kvcache_rewind_clamp,
        test_moe,
        test_moe_latent,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
