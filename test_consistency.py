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

from model.model import VibyConfig, VibyForCausalLM

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
        v.size
        for key, v in flat.items()
        if key.endswith((".dw_u", ".dw_v", ".dw_g"))
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
        v.size
        for key, v in flat.items()
        if key.endswith((".ws_in", ".ws_out"))
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


def main():
    tests = [
        test_prefill_causality,
        test_prefill_chunk_decode_consistency,
        test_pad_equivalence,
        test_loss_masks,
        test_dw_loop,
        test_ws_loop,
        test_engram,
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
