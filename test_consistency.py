"""架构正确性回归测试（不依赖训练数据，随机初始化的小模型即可运行）。

覆盖：
1. 因果性：prefill T 与 prefill T+K 的前 T 个 logits 必须一致；
2. prefill == 分段 prefill == 逐 token decode（use_cache 一致性）；
3. padding 等价性：左/右 padding 下有效位置的 logits 与无 padding 一致；
4. SFT / DPO loss mask：从首个内容 token 监督到 <|im_end|> 为止，
   不多掩码下一轮的 <|im_start|>。
5. ΔW-Loop（loop_k>1 + dw_rank>0）：参数审计、V=0 初始化等价、
   delta 通路接线与 LoRA 式启动动力学（V=0 时仅 dw_v 收梯度）。

前向类测试同时覆盖纯基础架构与当前基线机制（value_res / attn_gate / mtp /
loop_k + dw_rank）。

运行：python test_consistency.py
"""

import sys

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from model.model import VibyConfig, VibyForCausalLM, engram_indices_mx

ATOL = 2e-3  # float32 下不同分块/增量路径的浮点误差上限

# 前向类测试覆盖的架构变体
ARCH_VARIANTS = {
    "plain": {},
    "value_res+attn_gate+mtp": {
        "use_value_res": True,
        "use_attn_gate": True,
        "mtp_depth": 1,
    },
    "loop2+dw8": {
        "loop_k": 2,
        "dw_rank": 8,
    },
    "loop2+ws": {
        "loop_k": 2,
        "ws_loop": 1,
    },
    "loop2+dw8+ws": {
        "loop_k": 2,
        "dw_rank": 8,
        "ws_loop": 1,
    },
    "hrm1L2": {
        "hrm_H_cycles": 1,
        "hrm_L_cycles": 2,
    },
    "hrm2L2+had+sand+eng": {
        "hrm_H_cycles": 2,
        "hrm_L_cycles": 2,
        "ffn_type": "hadamard",
        "sandwich_norm": 1,
        "use_value_res": True,
        "use_attn_gate": True,
        "engram_layers": (1,),
        "engram_orders": (1, 2, 3),
        "engram_slots": 32,
        "engram_sub_dim": 16,
    },
    "san": {
        "ffn_type": "none",
        "zero_centered_norm": 1,
        "use_res_gate": 1,
        "sandwich_norm": 1,
        "san_res_init": 1,
    },
    "moe": {
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 48,
        "n_dense_layers": 1,
    },
    # 100M V4 风格配置的同款组合：MoE + MLA + engram + value_res + attn_gate + mtp
    "moe+eng+mtp+vr+gate": {
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 48,
        "n_dense_layers": 1,
        "engram_layers": (1,),
        "engram_orders": (2, 3),
        "engram_slots": 32,
        "engram_sub_dim": 16,
        "use_value_res": True,
        "use_attn_gate": True,
        "mtp_depth": 1,
    },
    # HRM×MoE（r070）：双状态循环 + MoE 双栈 + CycleRouter + CycleFiLM
    "hrm2L2+moe+cycle": {
        "hrm_H_cycles": 2,
        "hrm_L_cycles": 2,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 48,
        "n_dense_layers": 0,
        "hrm_cycle_router": 1,
        "hrm_cycle_film": 1,
        "engram_layers": (0,),
        "engram_orders": (2, 3),
        "engram_slots": 32,
        "engram_sub_dim": 16,
        "mtp_depth": 1,
    },
}


def tiny_config(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        kv_lora_rank=24,
        qk_rope_head_dim=8,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=512,
        mtp_depth=0,
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


def test_engram_decode_consistency():
    """Engram 缓存解码与训练口径一致：

    1) W_v 离地（engram 真实注入）后，携带 n-gram 窗口的逐 token decode
       与一次性 prefill 完全一致（标准 generate 路径）；
    2) 分块 decode（一次送多个 token + 窗口）与 prefill 一致——这就是投机
       验证步的窗口口径（每草稿位置携带自己的 n-gram 上下文）；
    3) greedy generate 端到端与全量前向的逐 token argmax 一致，证明
       generate 内部自动启用的窗口没有漏掉任何位置。

    零初始化的 engram 注入恒为 0，旧测试即使 decode 侧漏传窗口也会通过；
    这里先把表/W_v 打离地再比较，专门覆盖生成路径。
    """
    variants = {
        "engram": dict(
            engram_layers=(0, 1),
            engram_orders=(2, 3),
            engram_slots=32,
            engram_sub_dim=16,
        ),
        # r060 同款组合：MoE + MLA + engram + value_res + attn_gate + MTP
        "moe+eng+mtp+vr+gate": dict(
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=48,
            n_dense_layers=1,
            engram_layers=(1,),
            engram_orders=(2, 3),
            engram_slots=32,
            engram_sub_dim=16,
            use_value_res=True,
            use_attn_gate=True,
            mtp_depth=1,
        ),
        # r070 同款组合：HRM 双栈循环 + 全 MoE + CycleRouter/FiLM + engram
        # （engram 注入 L 栈、每循环重读；decode 窗口在每层每 cycle 收窄）
        "hrm2L2+moe+cycle+eng": dict(
            hrm_H_cycles=2,
            hrm_L_cycles=2,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=48,
            n_dense_layers=0,
            hrm_cycle_router=1,
            hrm_cycle_film=1,
            engram_layers=(0,),
            engram_orders=(2, 3),
            engram_slots=32,
            engram_sub_dim=16,
            mtp_depth=1,
        ),
    }
    for name, kw in variants.items():
        model = make_model(**kw)
        # 激活 engram：表离地 + W_v 离地，否则恒等注入测不出窗口口径；
        # taps 也离地，否则 4-tap 扩张卷积的历史项全为 0，测不出窗口
        # 是否覆盖了 t-3d/t-6d/t-9d 处 ev 的 n-gram 感受野。
        for eng in model.model.engrams:
            eng.table = mx.random.normal(eng.table.shape) * 0.05
            eng.value_proj.weight = mx.random.normal(eng.value_proj.weight.shape) * 0.02
            eng.taps = mx.random.normal(eng.taps.shape) * 0.02

        orders = tuple(model.config.engram_orders or ())
        max_order = max(orders) if orders else 0
        taps = model.model.engrams[0].taps.shape[0]
        win_len = max_order + (taps - 1) * max_order
        assert win_len > 0, "engram 窗口长度应为正"

        ids = rand_ids(40, seed=13)
        full = model(ids).logits  # 无 cache：engram 默认全位置注入

        # 1) 逐 token decode（generate 标准路径同口径）
        past = None
        outs = []
        for t in range(40):
            win = ids[:, max(0, t + 1 - win_len) : t + 1]
            o = model(
                ids[:, t : t + 1],
                past_key_values=past,
                use_cache=True,
                engram_input_ids=win,
                engram_decode=True,
            )
            past = o.past_key_values
            outs.append(o.logits)
        decoded = mx.concatenate(outs, axis=1)
        d = maxdiff(full, decoded)
        assert d < ATOL, f"[{name}] engram 逐 token decode 不一致：{d:.4f}"

        # 2) 分块 decode（投机验证步同口径：窗口 = 上下文尾部 + 整块）
        past = None
        outs = []
        for s, e in [(0, 17), (17, 30), (30, 40)]:
            # 窗口 = (win_len-1) 个前文 + 本块 T 个 token；VibyBlock 只取
            # 尾部 T 个位置注入（每位置自己的 n-gram + 卷积感受野）
            win = None if s == 0 else ids[:, max(0, s - (win_len - 1)) : e]
            o = model(
                ids[:, s:e],
                past_key_values=past,
                use_cache=True,
                engram_input_ids=win,
                engram_decode=True,
            )
            past = o.past_key_values
            outs.append(o.logits)
        chunked = mx.concatenate(outs, axis=1)
        d = maxdiff(full, chunked)
        assert d < ATOL, f"[{name}] engram 分块 decode 不一致：{d:.4f}"

        # 3) greedy generate 端到端：自动启用的窗口必须产出与全量前向
        #    一致的 token（任何窗口/注入错位都会让贪心链偏离）
        out_ids = model.generate(
            ids, do_sample=False, max_new_tokens=12, eos_token_id=None
        )
        gen_full = model(out_ids).logits
        for t in range(40, 52):
            got = int(out_ids[0, t].item())
            exp = int(mx.argmax(gen_full[0, t - 1]).item())
            assert got == exp, (
                f"[{name}] generate 第 {t} 个 token 偏离全量前向：{got} != {exp}"
            )

        # 4) MTP 投机路径冒烟：贪心下与标准 generate 一致（fp32 小模型，
        #    无 bf16 翻转问题），且每轮确实产出了草稿
        if model.config.mtp_depth > 0:
            spec = model.generate(
                ids,
                do_sample=False,
                max_new_tokens=12,
                eos_token_id=None,
                use_mtp_speculative=True,
            )
            assert spec.shape == out_ids.shape, (
                f"[{name}] 投机路径长度异常：{spec.shape} != {out_ids.shape}"
            )
            assert bool(mx.all(spec == out_ids).item()), (
                f"[{name}] 投机贪心与标准 generate 不一致"
            )
            stats = model._last_spec_stats
            assert stats["drafted"] > 0, f"[{name}] 投机路径未产出草稿"


def test_mtp_draft_cache_consistency():
    """MTP 草稿 KV cache：推理口径（prefill 建缓存 + 单位置草稿步 +
    批量追平）必须与训练口径（整流因果注意力，_mtp_loss 同约定）逐位一致。

    训练时 MTP block 对整流做因果注意力；推理草稿若丢弃上下文（无 cache
    的单位置调用），草稿分布显著偏离（r060 实测 argmax 命中率 45%→34%）。
    """
    variants = {
        "plain-mtp": dict(mtp_depth=1),
        # r060 同款组合：MoE + MLA + engram + value_res + attn_gate + MTP
        "moe+eng+mtp+vr+gate": dict(
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=48,
            n_dense_layers=1,
            engram_layers=(1,),
            engram_orders=(2, 3),
            engram_slots=32,
            engram_sub_dim=16,
            use_value_res=True,
            use_attn_gate=True,
            mtp_depth=1,
        ),
    }
    for name, kw in variants.items():
        model = make_model(**kw)
        ids = rand_ids(33, seed=21)
        H = model(ids).hidden_states  # (1, T, D)
        T = ids.shape[1]
        emb = model.model.embed_tokens(ids)
        mtp = model.mtp_modules[0]

        # 训练口径：整流一次前向（因果注意力）。流位置 t 消费
        # (H[t], emb(token t+1))，logits_full[t] 预测 token t+2。
        sub = T - 1
        rope = (model.model.freqs_cos[:sub], model.model.freqs_sin[:sub])
        h_full, _ = mtp(H[:, :sub], emb[:, 1:], rope)
        logits_full = model._lm_logits(h_full)[0]  # (sub, V)

        # 推理口径 1（草稿步）：prefill 建 cache 到 sub-2，再单位置草稿
        _, mtp_past = mtp(
            H[:, : sub - 1],
            emb[:, 1:sub],
            (model.model.freqs_cos[: sub - 1], model.model.freqs_sin[: sub - 1]),
            use_cache=True,
        )
        h_step, _ = mtp(
            H[:, sub - 1 : sub],
            emb[:, sub : sub + 1],
            (
                model.model.freqs_cos[sub - 1 : sub],
                model.model.freqs_sin[sub - 1 : sub],
            ),
            past_key_value=mtp_past,
            use_cache=True,
        )
        d = maxdiff(model._lm_logits(h_step)[0, 0], logits_full[sub - 1])
        assert d < ATOL, f"[{name}] 带 cache 草稿步与训练口径不符: {d:.4f}"

        # 推理口径 2（验证后批量追平）：prefill 到 sub-3，再批量追加 2 个位置
        _, mtp_past2 = mtp(
            H[:, : sub - 2],
            emb[:, 1 : sub - 1],
            (model.model.freqs_cos[: sub - 2], model.model.freqs_sin[: sub - 2]),
            use_cache=True,
        )
        h_cat, _ = mtp(
            H[:, sub - 2 : sub],
            emb[:, sub - 1 : sub + 1],
            (
                model.model.freqs_cos[sub - 2 : sub],
                model.model.freqs_sin[sub - 2 : sub],
            ),
            past_key_value=mtp_past2,
            use_cache=True,
        )
        d0 = maxdiff(model._lm_logits(h_cat)[0, 0], logits_full[sub - 2])
        d1 = maxdiff(model._lm_logits(h_cat)[0, 1], logits_full[sub - 1])
        assert max(d0, d1) < ATOL, (
            f"[{name}] 批量追平与训练口径不符: {d0:.4f}, {d1:.4f}"
        )


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


def test_dw_loop():
    """ΔW-Loop 专项：参数审计、V=0 初始化等价、delta 通路接线、梯度可达性。"""
    r, k = 8, 2
    model = make_model(loop_k=k, dw_rank=r)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 参数审计：每个被包装的 Linear 有 dw_u(out×r)+dw_v(r×in)+dw_g(k×r)；
    #    2 层 ×（6 个注意力投影 + 3 个 FFN 投影），MTP/embedding 不携带。
    dw_u_keys = [key for key in flat if key.endswith(".dw_u")]
    n_expected = 2 * (6 + 3)
    assert len(dw_u_keys) == n_expected, (
        f"ΔW 包装矩阵数不符: {len(dw_u_keys)} != {n_expected}"
    )
    expect = 0
    for key in dw_u_keys:
        out_f, in_f = flat[key[: -len("dw_u")] + "base.weight"].shape
        expect += r * (out_f + in_f) + k * r
    actual = sum(
        v.size for key, v in flat.items() if key.endswith((".dw_u", ".dw_v", ".dw_g"))
    )
    assert actual == expect, f"ΔW 参数量审计不符: {actual} != {expect}"

    # loop_k=1 时不创建任何 dw 参数（严格向后兼容）
    model_off = make_model(loop_k=1, dw_rank=r)
    assert not any(
        key.endswith((".dw_u", ".dw_v", ".dw_g"))
        for key in dict(tree_flatten(model_off.trainable_parameters()))
    ), "loop_k=1 时不应创建 dw 参数"

    # 2) 初始化等价：V=0 时任意 g 都不影响输出
    ids = rand_ids(32, seed=5)
    ref = model(ids).logits
    proj = model.model.layers[0].self_attn.q_proj
    proj.dw_g = mx.random.normal(proj.dw_g.shape)
    d = maxdiff(ref, model(ids).logits)
    assert d == 0.0, f"V=0 初始化下改动 dw_g 不应影响输出，偏差 {d}"

    # 3) 梯度可达性（LoRA 式启动动力学：V=0 时只有 dw_v 收梯度，
    #    dw_u/dw_g 的梯度恰为 0，待 V 离开原点后才开始学习）
    def loss_fn(m):
        return m(ids, labels=ids).loss

    _, grads = mx.value_and_grad(loss_fn)(model)
    gflat = dict(tree_flatten(grads))
    dv_max = max(
        float(mx.abs(v).max().item())
        for key, v in gflat.items()
        if key.endswith(".dw_v")
    )
    assert dv_max > 0, "dw_v 应收到非零梯度"
    for suffix in (".dw_u", ".dw_g"):
        g_max = max(
            float(mx.abs(v).max().item())
            for key, v in gflat.items()
            if key.endswith(suffix)
        )
        assert g_max == 0.0, f"V=0 时 {suffix} 梯度应恰为 0，实际 {g_max}"

    # 4) delta 通路接线确认：dw_v 非零后输出必须改变
    proj.dw_v = mx.random.normal(proj.dw_v.shape) * 0.02
    d = maxdiff(ref, model(ids).logits)
    assert d > 1e-4, f"dw_v 非零后输出未变（delta 通路未接线？），偏差 {d}"


def test_ws_loop():
    """W-Scale-Loop 专项：参数审计、s=1 初始化等价、梯度可达、通路接线。"""
    k = 2
    model = make_model(loop_k=k, ws_loop=1)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 参数审计：2 层 ×（6 个注意力投影 + 3 个 FFN 投影），每个包装
    #    有 ws_in(k×in) + ws_out(k×out)
    ws_in_keys = [key for key in flat if key.endswith(".ws_in")]
    n_expected = 2 * (6 + 3)
    assert len(ws_in_keys) == n_expected, (
        f"W-Scale 包装矩阵数不符: {len(ws_in_keys)} != {n_expected}"
    )
    expect = 0
    for key in ws_in_keys:
        in_f = flat[key].shape[1]
        out_f = flat[key[: -len("ws_in")] + "ws_out"].shape[1]
        expect += k * (in_f + out_f)
    actual = sum(
        v.size for key, v in flat.items() if key.endswith((".ws_in", ".ws_out"))
    )
    assert actual == expect, f"W-Scale 参数量审计不符: {actual} != {expect}"

    # loop_k=1 时不创建 ws 参数（严格向后兼容）
    model_off = make_model(loop_k=1, ws_loop=1)
    assert not any(
        key.endswith((".ws_in", ".ws_out"))
        for key in dict(tree_flatten(model_off.trainable_parameters()))
    ), "loop_k=1 时不应创建 ws 参数"

    # 2) 初始化等价：s=1 时输出严格等于 base 路径
    ids = rand_ids(32, seed=6)
    ref = model(ids).logits
    proj = model.model.layers[0].self_attn.q_proj
    proj.ws_in = mx.ones_like(proj.ws_in)
    proj.ws_out = mx.ones_like(proj.ws_out)
    d = maxdiff(ref, model(ids).logits)
    assert d == 0.0, f"s=1 初始化下输出应严格不变，偏差 {d}"

    # 3) 梯度可达：s 参数在初始化处就应收到梯度（与 V=0 的 ΔW 不同）
    def loss_fn(m):
        return m(ids, labels=ids).loss

    _, grads = mx.value_and_grad(loss_fn)(model)
    gflat = dict(tree_flatten(grads))
    for suffix in (".ws_in", ".ws_out"):
        g_max = max(
            float(mx.abs(v).max().item())
            for key, v in gflat.items()
            if key.endswith(suffix)
        )
        assert g_max > 0, f"{suffix} 应收到非零梯度"

    # 4) 通路接线确认：扰动 s 后输出必须改变
    proj.ws_out = mx.ones_like(proj.ws_out) * 1.05
    d = maxdiff(ref, model(ids).logits)
    assert d > 1e-4, f"ws_out 扰动后输出未变（通路未接线？），偏差 {d}"


def test_engram():
    """Engram n-gram 记忆：参数审计、零初始化启动动力学、因果性、padding、梯度与通路。"""
    kw = dict(
        engram_layers=(0, 1),
        engram_orders=(2, 3),
        engram_slots=32,
        engram_sub_dim=16,
    )
    model = make_model(**kw)
    flat = dict(tree_flatten(model.trainable_parameters()))
    table_keys = [k for k in flat if k.endswith(".table")]
    assert len(table_keys) == len(kw["engram_layers"]), (
        f"Engram 表数量 {len(table_keys)} != {len(kw['engram_layers'])}"
    )

    ids = rand_ids(24, seed=11)

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    # value_proj 零初始化（r064/r065 负收益修复：随机 ev 经 α≈σ(0)=0.5 的门
    # 向残差流注噪）。启动动力学与 ΔW V=0 同款：仅 value_proj.weight 收梯度，
    # table/key_proj 梯度恰为 0；W_v 离地后表梯度恢复。
    vproj_keys = [k for k in flat if k.endswith("value_proj.weight")]
    assert vproj_keys, "缺少 value_proj 参数"
    for k in vproj_keys:
        assert float(mx.abs(flat[k]).max().item()) == 0.0, f"{k} 应零初始化"
    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    for k in vproj_keys:
        assert float(mx.abs(gflat[k]).max().item()) > 0, f"{k} 应收到非零梯度"
    gmax = max(float(mx.abs(gflat[k]).max().item()) for k in table_keys)
    assert gmax == 0.0, f"value_proj=0 时 table 梯度应恰为 0，实得 {gmax}"

    # 激活 engram（模拟 W_v 离地后），后续测试在 engram 实际工作的状态下进行
    for eng in model.model.engrams:
        eng.table = mx.random.normal(eng.table.shape) * 0.05
        eng.value_proj.weight = mx.random.normal(eng.value_proj.weight.shape) * 0.02

    # 因果性：追加未来 token 不得改变前缀 logits（Engram 全程启用）
    extra = rand_ids(12, seed=12)
    ref = model(ids).logits
    long_out = model(mx.concatenate([ids, extra], axis=1)).logits[:, :24]
    d = maxdiff(ref, long_out)
    assert d < ATOL, f"Engram 因果性破坏：{d:.4f}"

    # padding 等价：右 padding 不得污染有效位置
    pad = mx.zeros((1, 2), dtype=ids.dtype)
    ids_pad = mx.concatenate([ids, pad], axis=1)
    mask = mx.concatenate(
        [mx.ones((1, 24), dtype=mx.int32), mx.zeros((1, 2), dtype=mx.int32)],
        axis=1,
    )
    out_pad = model(ids_pad, attention_mask=mask).logits[:, :24]
    d = maxdiff(ref, out_pad)
    assert d < ATOL, f"Engram padding 污染：{d:.4f}"

    # W_v≠0 后梯度可达表
    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    gmax = max(float(mx.abs(gflat[k]).max().item()) for k in table_keys)
    assert gmax > 0, "W_v≠0 后 Engram table 未收到梯度"

    # 表通路接线确认：扰动表后输出必须改变
    ref = model(ids).logits
    eng0 = model.model.engrams[0]
    eng0.table = eng0.table + 0.05
    d = maxdiff(ref, model(ids).logits)
    assert d > 1e-4, f"Engram 表扰动后输出未变（通路未接线？），偏差 {d}"

    # HRM 模式：engram 保留并注入 L-module（每个循环调用重读表）
    m2 = make_model(
        hrm_H_cycles=2,
        hrm_L_cycles=1,
        ffn_type="hadamard",
        sandwich_norm=1,
        engram_layers=(1,),
        engram_orders=(1, 2, 3),
        engram_slots=32,
        engram_sub_dim=16,
    )
    assert len(m2.model.engrams) == 1, "HRM 模式应保留 engram"
    e2 = m2.model.engrams[0]
    assert float(mx.abs(e2.value_proj.weight).max().item()) == 0.0, (
        "HRM engram value_proj 应零初始化"
    )
    e2.table = mx.random.normal(e2.table.shape) * 0.05
    e2.value_proj.weight = mx.random.normal(e2.value_proj.weight.shape) * 0.02
    ref = m2(ids).logits
    e2.table = e2.table + 0.05
    d = maxdiff(ref, m2(ids).logits)
    assert d > 1e-4, f"HRM engram 表扰动后输出未变（未接线？），偏差 {d}"


def test_engram_order_mask():
    """低阶 n-gram 的因果掩码必须按 order 独立计算。

    t=1 时 order=2 的窗口有效、order=3 无效；若实现先取所有 order 的
    交集再广播，会把 order=2 的查表结果也清零。这里把 order=2 的命中槽
    打离地，要求 t=1 的 logits 改变而 t=0 不变。
    """
    model = make_model(
        engram_layers=(0,),
        engram_orders=(2, 3),
        engram_heads=1,
        engram_slots=32,
        engram_sub_dim=16,
    )
    eng = model.model.engrams[0]
    eng.table = mx.random.normal(eng.table.shape) * 0.05
    eng.value_proj.weight = mx.random.normal(eng.value_proj.weight.shape) * 0.02

    ids = mx.array([[5, 7, 9, 10, 11]], dtype=mx.int64)
    idx = engram_indices_mx(ids, eng.orders, eng.heads, eng.slots)
    slot = int(idx[0, 1, 0].item())  # order=2, head=0 在 t=1 的命中槽

    ref = model(ids).logits
    eng.table = eng.table.at[0, slot, :].add(1.0)
    out = model(ids).logits
    mx.eval(ref, out)
    assert maxdiff(ref[:, 0], out[:, 0]) < ATOL, (
        "t=0 所有 order 都应无效，扰动 order=2 表不应改变输出"
    )
    d = maxdiff(ref[:, 1], out[:, 1])
    assert d > 1e-4, f"t=1 的 order=2 窗口被错误掩码（扰动无效果）：{d:.4f}"


def test_moe():
    """DeepSeekMoE 专项：结构审计、逐专家参考等价、路由权重性质、
    偏置冻结/无辅助损失更新、负载统计、梯度可达性、MTP 同构。"""
    from model.model import MoEFeedForward

    E, K, moe_in = 8, 2, 48
    kw = dict(
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        moe_intermediate_size=moe_in,
        n_dense_layers=1,
    )
    model = make_model(**kw)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 结构审计：layer0 dense（SwiGLU），layer1 MoE；专家权重为 (E,out,in) 堆叠
    l0, l1 = model.model.layers
    assert not isinstance(l0.mlp, MoEFeedForward), "dense 前缀层应为 SwiGLU"
    assert isinstance(l1.mlp, MoEFeedForward), "n_dense_layers 之后应为 MoE"
    gw = flat["model.layers.1.mlp.experts.gate_up_w"]
    assert gw.shape == (E, 2 * moe_in, 128), f"专家堆叠形状错误: {gw.shape}"
    # expert_bias 冻结（不进梯度/优化器），但随 checkpoint 持久化
    assert "model.layers.1.mlp.router.expert_bias" not in flat, "expert_bias 不应可训练"
    allp = dict(tree_flatten(model.parameters()))
    assert "model.layers.1.mlp.router.expert_bias" in allp, "expert_bias 应持久化"

    # 2) 路由参考等价：逐 (token, expert) 朴素循环 vs 融合 kernel 实现
    # （G=20 <= _KERNEL_MAX_PAIRS，l1.mlp(x) 走 decode Metal kernel 路径）
    x = mx.random.normal((2, 5, 128))
    out = l1.mlp(x)
    idx, w = l1.mlp.router(x)
    B, T, D = x.shape
    gu_ = np.array(l1.mlp.experts.gate_up_w)  # (E, 2I, D)
    gw_, uw_ = gu_[:, :moe_in], gu_[:, moe_in:]
    dw_ = np.array(l1.mlp.experts.down_w)
    xn, idxn, wn = np.array(x), np.array(idx), np.array(w)

    def silu(v):
        return v / (1.0 + np.exp(-v))

    ref = np.zeros((B, T, D), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            for j in range(K):
                e = idxn[b, t, j]
                h = silu(xn[b, t] @ gw_[e].T) * (xn[b, t] @ uw_[e].T)
                ref[b, t] += wn[b, t, j] * (h @ dw_[e].T)
    ref = mx.array(ref) + l1.mlp.shared(x)
    d = maxdiff(out, ref)
    assert d < 1e-4, f"kernel 融合 MoE 与逐专家参考不符: {d}"

    # 2b) 三路径等价：kernel（decode，上面已验）/ 稠密（小 prefill）/ 稀疏（训练）
    l1.mlp._KERNEL_MAX_PAIRS = 0  # 强制稠密
    d = maxdiff(ref, l1.mlp(x))
    assert d < 1e-4, f"稠密批量 MoE 与逐专家参考不符: {d}"
    l1.mlp._DENSE_MAX_PAIRS = 0  # 强制稀疏
    d = maxdiff(ref, l1.mlp(x))
    l1.mlp._KERNEL_MAX_PAIRS = 512  # 还原默认
    l1.mlp._DENSE_MAX_PAIRS = 4096
    assert d < 1e-4, f"稀疏分段 MoE 与逐专家参考不符: {d}"

    # 2c) bf16 下 kernel 路径（含 router kernel）与原生路径整体一致
    # （decode 实际运行 dtype；随机输入 tie 概率 0，选择集合应精确一致）
    mlp = l1.mlp
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

    # 4) 负载统计与无辅助损失偏置更新（V3 规则：超载减、欠载增）
    gates = model.moe_gates()
    for g in gates:
        g.collect_stats = True
    ids = rand_ids(16, B=2, seed=7)
    model(ids, labels=ids)
    stats = model.moe_load_stats()
    mx.eval(stats)
    assert stats.shape == (len(gates) * E,), f"负载统计形状错误: {stats.shape}"
    per = stats.reshape(len(gates), E)
    assert float(per[0].sum()) == 2 * 16 * K, f"计数总量不符: {float(per[0].sum())}"
    bias0 = np.array(gates[0].expert_bias)
    load = per[0]
    model.update_moe_biases(stats, 0.001)
    bias1 = np.array(gates[0].expert_bias)
    over, under = load > load.mean(), load < load.mean()
    assert np.all(bias1[over] < bias0[over]), "超载专家 bias 应减小"
    assert np.all(bias1[under] > bias0[under]), "欠载专家 bias 应增大"
    assert gates[0].expert_bias.dtype == mx.float32, "bias 应保持 fp32 累积"

    # 5) 偏置影响选择：某专家 bias 拉满后必被选中
    g0 = gates[0]
    g0.expert_bias = mx.zeros((E,)).at[3].add(100.0)
    idx2, _ = g0(x)
    assert bool(mx.all(mx.any(idx2 == 3, axis=-1)).item()), "大 bias 专家应必中"

    # 6) 梯度可达：router 与专家堆叠均收非零梯度
    # （kernel 路径不可微；训练前向走稠密/稀疏路径，这里强制稠密模拟）
    for m in (l1.mlp,):
        m._KERNEL_MAX_PAIRS = 0

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    l1.mlp._KERNEL_MAX_PAIRS = 512  # 还原默认
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.layers.1.mlp.router.weight",
        "model.layers.1.mlp.experts.gate_up_w",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"{key} 应收到非零梯度"

    # 7) MTP 块同构为 MoE（V3/V4 的 MTP 与主干同构）
    model_mtp = make_model(mtp_depth=1, **kw)
    assert isinstance(model_mtp.mtp_modules[0].block.mlp, MoEFeedForward), (
        "MTP 块应为 MoE 层"
    )


def test_hrm_moe_cycle():
    """HRM×MoE（CycleRouter/CycleFiLM）：参数审计、零初始化等价、
    梯度可达、通路接线、跨循环负载统计累加。"""
    kw = dict(
        hrm_H_cycles=2,
        hrm_L_cycles=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        n_dense_layers=0,
        hrm_cycle_router=1,
        hrm_cycle_film=1,
        mtp_depth=1,
    )
    H, L, D = 2, 2, 128
    n_cycles = H * (L + 1)  # 6
    model = make_model(**kw)
    flat = dict(tree_flatten(model.trainable_parameters()))

    # 1) 参数审计：每个 MoE gate（l/h 栈各 P=2 层 + MTP 块 = 5 个）携带
    #    cycle_emb(6×128)；VibyModel 携带 hrm_film_scale/shift(6×128)；
    #    关闭开关时不创建任何对应参数。
    cb_keys = [k for k in flat if k.endswith(".cycle_emb")]
    assert len(cb_keys) == 5, f"cycle_emb 数量不符: {len(cb_keys)} != 5"
    for k in cb_keys:
        assert flat[k].shape == (n_cycles, D), f"cycle_emb 形状不符: {flat[k].shape}"
    for name in ("hrm_film_scale", "hrm_film_shift"):
        key = f"model.{name}"
        assert key in flat and flat[key].shape == (n_cycles, D), f"{key} 缺失或形状不符"
    model_off = make_model(**{**kw, "hrm_cycle_router": 0, "hrm_cycle_film": 0})
    assert not any(
        k.endswith(".cycle_emb") or "hrm_film" in k
        for k in dict(tree_flatten(model_off.trainable_parameters()))
    ), "开关关闭时不应创建 cycle 参数"

    # 2) 零初始化等价：cycle 参数全零时输出与关闭开关一致。容差而非严格
    #    相等：cycle 模型在小批量 decode 会绕开融合 kernel 走稠密路径
    #    （kernel 不含 cycle_emb 项），与 off 模型的 kernel 路径存在
    #    ~1e-6 级数值差；训练侧两条路径选择完全一致，故对训练严格等价。
    ids = rand_ids(32, seed=11)
    ref = model_off(ids).logits
    d = maxdiff(ref, model(ids).logits)
    assert d < 1e-5, f"零初始化下 cycle 参数不应影响输出，偏差 {d}"

    # 3) 梯度可达：cycle_emb 经 sigmoid 分→w 回传（全 cycle 回传时非零），
    #    hrm_film 同理（显式参数形式：model 形式的 value_and_grad 会对
    #    freeze 的 expert_bias 返回 None 叶，tree_flatten 不接受）
    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    _, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    gflat = dict(tree_flatten(grads))
    ce_max = max(
        float(mx.abs(v).max().item())
        for k, v in gflat.items()
        if k.endswith(".cycle_emb")
    )
    assert ce_max > 0, "cycle_emb 应收到非零梯度"
    film_max = max(
        float(mx.abs(v).max().item()) for k, v in gflat.items() if "hrm_film" in k
    )
    assert film_max > 0, "hrm_film 应收到非零梯度"

    # 4) 通路接线：扰动 cycle_emb / hrm_film 后输出必须改变
    g0 = model.model.l_module.layers[0].mlp.router
    g0.cycle_emb = g0.cycle_emb.at[0].add(0.05)
    d = maxdiff(ref, model(ids).logits)
    assert d > 1e-4, f"cycle_emb 扰动后输出未变（通路未接线？），偏差 {d}"
    g0.cycle_emb = mx.zeros_like(g0.cycle_emb)
    model.model.hrm_film_scale = model.model.hrm_film_scale.at[3].add(0.1)
    d = maxdiff(ref, model(ids).logits)
    assert d > 1e-4, f"hrm_film 扰动后输出未变（通路未接线？），偏差 {d}"
    model.model.hrm_film_scale = mx.zeros_like(model.model.hrm_film_scale)

    # 5) 负载统计跨循环累加：L 栈 gate 每前向调 H*L=4 次、H 栈 gate 调
    #    H=2 次；计数总量 = 调用次数 × M×K
    model2 = make_model(**kw)
    for g in model2.moe_gates():
        g.collect_stats = True
    B, T, K = 2, 16, 2
    ids2 = rand_ids(T, B=B, seed=13)
    model2(ids2, labels=ids2)
    l_g = model2.model.l_module.layers[0].mlp.router
    h_g = model2.model.h_module.layers[0].mlp.router
    mx.eval(l_g.last_load, h_g.last_load)
    assert float(l_g.last_load.sum()) == 4 * B * T * K, (
        f"L 栈负载计数应为 4×M×K: {float(l_g.last_load.sum())}"
    )
    assert float(h_g.last_load.sum()) == 2 * B * T * K, (
        f"H 栈负载计数应为 2×M×K: {float(h_g.last_load.sum())}"
    )
    # 下次前向开始时应重置（不累加上次前向）
    model2(ids2, labels=ids2)
    mx.eval(l_g.last_load)
    assert float(l_g.last_load.sum()) == 4 * B * T * K, "负载统计未按前向重置"


def main():
    tests = [
        test_prefill_causality,
        test_prefill_chunk_decode_consistency,
        test_pad_equivalence,
        test_decode_with_pad,
        test_loss_masks,
        test_dw_loop,
        test_ws_loop,
        test_engram,
        test_engram_order_mask,
        test_engram_decode_consistency,
        test_mtp_draft_cache_consistency,
        test_moe,
        test_hrm_moe_cycle,
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
