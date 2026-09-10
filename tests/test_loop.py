"""SMELT 中间层 loop（arXiv:2609.01343）专项测试。

覆盖：config 往返/校验、exec_order 展开、退化等价（loop_span=0 /
loop_count=1 与无 loop 逐位一致）、权重共享（参数量不变）、残差写入
缩放数值、prefill/decode 一致性、反向梯度可达、训练步冒烟
（compile 开/关）。

运行：python test_loop.py
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from model.config import VibyConfig
from model.model import VibyForCausalLM

ATOL = 2e-3  # 与 test_consistency 同口径


def _cfg(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=256,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        mtp_depth=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        moe_latent_dim=0,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def make_model(**kw):
    mx.random.seed(42)
    model = VibyForCausalLM(_cfg(**kw))
    model.eval()
    return model


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def rand_ids(T, B=1, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(3, 256, (B, T)).astype(np.int64))


def test_config_roundtrip_and_validation():
    """sidecar 往返；非法 span / count / scale 报错；旧 sidecar 缺键退化为关。"""
    cfg = _cfg(loop_span=2, loop_count=2, loop_res_scale="r")
    assert cfg.loop_layer_range() == (1, 3)
    assert cfg.num_exec_layers == 6
    d = cfg.to_dict()
    assert (d["loop_span"], d["loop_count"], d["loop_res_scale"]) == (2, 2, "r")
    rt = VibyConfig.from_dict(d)
    assert rt.loop_layer_range() == (1, 3) and rt.num_exec_layers == 6
    # 旧 sidecar 无 loop 键 → 默认无 loop
    old = {k: v for k, v in d.items() if not k.startswith("loop_")}
    oc = VibyConfig.from_dict(old)
    assert oc.loop_span == 0 and oc.loop_layer_range() is None
    assert oc.num_exec_layers == oc.num_hidden_layers
    # 校验
    try:
        _cfg(loop_span=-1)
    except ValueError as e:
        assert "loop_span" in str(e)
    else:
        raise AssertionError("负 loop_span 应报错")
    try:
        _cfg(loop_span=3)  # L=4 时最大 2（首尾层不参与）
    except ValueError as e:
        assert "loop_span" in str(e)
    else:
        raise AssertionError("loop_span > L-2 应报错")
    try:
        _cfg(loop_span=2, loop_count=0)
    except ValueError as e:
        assert "loop_count" in str(e)
    else:
        raise AssertionError("loop_count<1 应报错")
    try:
        _cfg(loop_span=2, loop_res_scale="sqrt")
    except ValueError as e:
        assert "loop_res_scale" in str(e)
    else:
        raise AssertionError("非法 loop_res_scale 应报错")
    # 静默退化组合
    assert _cfg(loop_span=2, loop_count=1).loop_layer_range() is None
    assert _cfg(loop_span=0, loop_count=3).loop_layer_range() is None
    print("config roundtrip/validation: OK")


def test_exec_order():
    """L=9 span=4 r=2 → [0,1,2,3,4,5,2,3,4,5,6,7,8]，n_exec=13。"""
    from model.block import VibyStack

    cfg = _cfg(num_hidden_layers=9, loop_span=4, loop_count=2)
    stack = VibyStack(cfg, 9)
    assert stack.exec_order == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 8]
    assert stack.n_exec == 13 == cfg.num_exec_layers
    cfg3 = _cfg(num_hidden_layers=9, loop_span=4, loop_count=3)
    stack3 = VibyStack(cfg3, 9)
    assert stack3.exec_order == [0, 1] + [2, 3, 4, 5] * 3 + [6, 7, 8]
    stack_off = VibyStack(_cfg(num_hidden_layers=9), 9)
    assert stack_off.exec_order == list(range(9))
    # span 内层 loop_res_scale≠1.0，span 外为 1.0；MTP 末层不受影响
    scales = [layer.loop_res_scale for layer in stack.layers]
    assert scales[0] == 1.0 and scales[8] == 1.0
    assert all(abs(s - 2**-0.5) < 1e-12 for s in scales[2:6])
    assert all(s == 1.0 for s in scales[1:2] + scales[6:8])
    print("exec_order: OK")


def test_degenerate_equivalence():
    """loop_span=0 / loop_count=1 与默认模型逐位一致（同 seed）。"""
    ids = rand_ids(24, seed=1)
    ref = make_model()
    for name, kw in {
        "loop_span=0": dict(loop_span=0, loop_count=2),
        "loop_count=1": dict(loop_span=2, loop_count=1),
    }.items():
        m = make_model(**kw)
        d = maxdiff(ref(ids).logits, m(ids).logits)
        assert d == 0.0, f"[{name}] 应与默认模型逐位一致: {d}"
    print("degenerate equivalence: OK")


def test_weight_sharing_param_count():
    """loop 模型参数量 == 同 config 无 loop 模型（权重共享）。"""
    base = make_model()
    looped = make_model(loop_span=2, loop_count=2)
    n_base = sum(v.size for _, v in tree_flatten(base.parameters()))
    n_loop = sum(v.size for _, v in tree_flatten(looped.parameters()))
    assert n_base == n_loop, f"参数量应不变: {n_base} vs {n_loop}"
    names_base = {k for k, _ in tree_flatten(base.parameters())}
    names_loop = {k for k, _ in tree_flatten(looped.parameters())}
    assert names_base == names_loop
    print(f"weight sharing: {n_loop} params OK")


def test_res_scale_numerical():
    """rsqrt 与 none 下 loop 层 attn 写入幅度比例恰为 r**-0.5（构造性验证）。"""
    from model.block import VibyBlock

    cfg_none = _cfg(loop_span=2, loop_count=2, loop_res_scale="none")
    cfg_rsqrt = _cfg(loop_span=2, loop_count=2, loop_res_scale="rsqrt")
    assert cfg_none.loop_layer_range() == (1, 3)
    b_none = VibyBlock(cfg_none, layer_idx=1)
    b_rsqrt = VibyBlock(cfg_rsqrt, layer_idx=1)
    b_rsqrt.update(tree_unflatten(tree_flatten(b_none.parameters())))
    assert b_none.loop_res_scale == 1.0
    assert abs(b_rsqrt.loop_res_scale - 2**-0.5) < 1e-12
    mx.random.seed(0)
    x = mx.random.normal((1, 3, 128)).astype(mx.float32)
    w_none, w_rsqrt = [x], [x]
    b_none.eval()
    b_rsqrt.eval()
    b_none(x, residuals=w_none, mask_is_full=True)
    b_rsqrt(x, residuals=w_rsqrt, mask_is_full=True)
    mx.eval(*w_none, *w_rsqrt)
    # 第一次写入（v_attn）上游完全相同，比例必须精确为 r**-0.5
    d = maxdiff(w_rsqrt[1], w_none[1] * 2**-0.5)
    assert d < 1e-6, f"rsqrt 写入应恰为 none 的 1/√2: |Δ|={d:.3e}"
    # "r" 模式比例为 1/r
    b_r = VibyBlock(_cfg(loop_span=2, loop_count=2, loop_res_scale="r"), layer_idx=1)
    b_r.update(tree_unflatten(tree_flatten(b_none.parameters())))
    assert b_r.loop_res_scale == 0.5
    w_r = [x]
    b_r.eval()
    b_r(x, residuals=w_r, mask_is_full=True)
    mx.eval(*w_r)
    assert maxdiff(w_r[1], w_none[1] * 0.5) < 1e-6
    print("res_scale numerical: OK")


def test_loop_prefill_decode_consistency():
    """loop 生效时整段前向 == 分段 prefill == 逐 token decode。"""
    model = make_model(loop_span=2, loop_count=2)
    assert model.config.num_exec_layers == 6
    ids = rand_ids(24, seed=3)
    full = model(ids).logits

    # 分段 prefill（10 + 14）
    past = None
    outs = []
    for s, e in [(0, 10), (10, 24)]:
        o = model(ids[:, s:e], past_key_values=past, use_cache=True)
        past = o.past_key_values
        assert len(past) == 6, f"presents 应按执行层数: {len(past)}"
        outs.append(o.logits)
    d_chunk = maxdiff(full, mx.concatenate(outs, axis=1))
    assert d_chunk < ATOL, f"分段 prefill 不一致: {d_chunk:.4f}"

    # 逐 token decode（loop 层 loop_res_scale≠1.0 时回退 eager，仍须一致）
    past = None
    outs = []
    for t in range(24):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        outs.append(o.logits)
    d_dec = maxdiff(full, mx.concatenate(outs, axis=1))
    assert d_dec < ATOL, f"逐 token decode 不一致: {d_dec:.4f}"

    # 因果性：追加未来 token 不改变前缀 logits
    extra = rand_ids(8, seed=4)
    out_long = model(mx.concatenate([ids, extra], axis=1)).logits[:, :24]
    d_causal = maxdiff(full, out_long)
    assert d_causal < ATOL, f"因果性破坏: {d_causal:.4f}"
    print(f"prefill/decode consistency: max|Δ|={max(d_chunk, d_dec):.3e} OK")


def test_loop_backward():
    """loss.backward 后 loop 层参数收非零梯度（两 visit 累加，不断图）。"""
    model = make_model(loop_span=2, loop_count=2)
    model.train()
    ids = rand_ids(16, B=2, seed=7)

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    assert np.isfinite(float(val)) and float(val) > 0, f"loss 非有限: {val}"
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.stack.layers.1.mlp.router.weight",
        "model.stack.layers.2.self_attn.qkv_proj.weight",
        "model.stack.layers.1.attn_res_q_attn",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"{key} 应收到非零梯度"
    print("backward: OK")


def _train_step_smoke(compile_model, **model_kw):
    """一次微批 loss+grad 冒烟（BaseTrainer 口径），返回 (loss, grads)。"""
    from types import SimpleNamespace

    from trainer.base_trainer import BaseTrainer

    B, T = 2, 8
    ids = rand_ids(T, B=B, seed=3)
    loss_mask = mx.ones(ids.shape, dtype=mx.int32)
    attn_mask = mx.ones(ids.shape, dtype=mx.int32)
    model = make_model(**model_kw)
    model.train()
    t = BaseTrainer.__new__(BaseTrainer)
    t.model = model
    t.lm_config = model.config
    t.args = SimpleNamespace(
        accumulation_steps=1,
        compile_model=compile_model,
        cache_limit_gb=0,
        max_seq_len=T,
    )
    t._moe_gates = model.moe_gates()
    t._expl_nest_mu = 0.0
    t._en_delta = None
    unemb = model.lm_head if model.lm_head is not None else model.model.embed_tokens
    t._unembedding_input_dim = int(unemb.weight.shape[1])
    t._unembedding_output_dim = int(unemb.weight.shape[0])
    t._loss_and_grad = t._build_loss_and_grad()
    (loss, *_rest), grads = t._compute_loss_and_grad(
        ids, ids, loss_mask, attn_mask, False, None
    )
    mx.eval(loss, grads)
    return float(loss), dict(tree_flatten(grads))


def test_loop_train_step_smoke():
    """训练步冒烟：loop 开，compile 关/开各跑一次，loss 有限且梯度非零。"""
    for compile_model in (False, True):
        loss, gflat = _train_step_smoke(compile_model, loop_span=2, loop_count=2)
        assert np.isfinite(loss), f"compile={compile_model} loss 非有限"
        g_max = float(
            mx.abs(gflat["model.stack.layers.2.mlp.router.weight"]).max().item()
        )
        assert g_max > 0, f"compile={compile_model} loop 层应收到非零梯度"
        print(f"train step smoke (compile={compile_model}): loss={loss:.4f} OK")


def test_mfu_counts_loop():
    """MFU 口径：loop 生效时按执行层数计 FLOPs（GEMM span 层 ×r + attn ×n_exec）。"""
    from model.flops import attn_fwdbwd_flops_per_token, gemm_active_params

    base = make_model()
    looped = make_model(loop_span=2, loop_count=2)
    f_base = attn_fwdbwd_flops_per_token(base.config, 16)
    f_loop = attn_fwdbwd_flops_per_token(looped.config, 16)
    assert f_loop * 4 == f_base * 6, (
        f"attn FLOPs 应按执行层数 4→6 放大: {f_base} vs {f_loop}"
    )
    g_base = gemm_active_params(base)
    g_loop = gemm_active_params(looped)
    # span (1,3) 两层 GEMM 翻倍：g_loop = g_base + span 两层激活参数
    assert g_loop > g_base, "loop 后 GEMM 激活参数应放大"
    # 非 span 层不重复计：差值应恰为 layers.1/layers.2 的激活参数
    # （expert_bias 是 frozen buffer，gemm_active_params 跳过，同步排除）
    span_params = sum(
        v.size // 8 * 2 if ".experts." in path else v.size
        for path, v in tree_flatten(looped.parameters())
        if path.startswith(("model.stack.layers.1.", "model.stack.layers.2."))
        and not path.endswith("expert_bias")
    )
    assert g_loop - g_base == span_params, (
        f"GEMM 增量应恰为 span 内两层: {g_loop - g_base} vs {span_params}"
    )
    print("mfu counts loop: OK")


def test_engine_loop_smoke():
    """engine 分页/稠密 decode 在 loop 生效下与 model.generate greedy 一致。

    PagePool/KVCache 数量按 num_exec_layers 分配（每 visit 一个独立
    cache），paged 与 dense 单序列路径均应与无 cache 整段 greedy 对齐。
    """
    from engine import SamplingParams, VibyEngine

    model = make_model(loop_span=2, loop_count=2)
    assert model.config.num_exec_layers == 6
    prompt = [3, 4, 5, 6, 7, 8, 9]
    params = SamplingParams(
        max_new_tokens=6,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    engine = VibyEngine(model, page_size=4, max_num_seqs=2, max_num_pages=8)
    assert engine.pool.num_layers == 6, "pool 应按执行层数分配"
    gen_engine = engine.generate([prompt], params)[0].outputs[0].token_ids
    gen_ref = model.generate(
        mx.array([prompt]),
        max_new_tokens=6,
        do_sample=False,
        eos_token_id=None,
    )[0].tolist()[len(prompt):]
    assert gen_engine == gen_ref, f"engine {gen_engine} vs generate {gen_ref}"
    print(f"engine loop smoke: {gen_engine} OK")


def test_jfb_config():
    """loop_grad_mode 校验与 sidecar 往返。"""
    cfg = _cfg(loop_span=2, loop_count=2, loop_grad_mode="jfb")
    assert cfg.loop_grad_mode == "jfb"
    assert VibyConfig.from_dict(cfg.to_dict()).loop_grad_mode == "jfb"
    assert _cfg(loop_span=2, loop_count=2).loop_grad_mode == "full"
    try:
        _cfg(loop_span=2, loop_count=2, loop_grad_mode="detach")
    except ValueError as e:
        assert "loop_grad_mode" in str(e)
    else:
        raise AssertionError("非法 loop_grad_mode 应报错")
    print("jfb config: OK")


def test_jfb_forward_bitwise_equal():
    """jfb 前向与 full 逐位一致（replace / register / iHC 三种残差模式）。"""
    ids = rand_ids(24, seed=5)
    for name, kw in {
        "replace": {},
        "register": {"attn_res_register": True},
        "ihc": {"ihc": True, "ihc_streams": 4},
    }.items():
        full = make_model(loop_span=2, loop_count=2, loop_grad_mode="full", **kw)
        jfb = make_model(loop_span=2, loop_count=2, loop_grad_mode="jfb", **kw)
        d = maxdiff(full(ids).logits, jfb(ids).logits)
        assert d == 0.0, f"[{name}] jfb 前向应与 full 逐位一致: {d}"
    print("jfb forward bitwise (replace/register/ihc): OK")


def test_jfb_backward():
    """jfb 反向：span 层与 pre-span 层（embedding/layer0）均收非零梯度。

    embedding/layer0 的非零梯度是 x_entry 恒等通路生效的关键回归（window
    裁剪下 span 写入断梯度后，merge 链上没有其它通到 pre-span 的路径）。
    jfb 与 full 是不同梯度估计，span 层梯度应可分。
    """
    ids = rand_ids(16, B=2, seed=7)
    grads = {}
    for mode in ("full", "jfb"):
        model = make_model(loop_span=2, loop_count=2, loop_grad_mode=mode)
        model.train()

        def loss_fn(p):
            model.update(p)
            return model(ids, labels=ids).loss

        val, g = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        mx.eval(val, g)
        assert np.isfinite(float(val)), f"{mode} loss 非有限: {val}"
        grads[mode] = dict(tree_flatten(g))
    for key in (
        "model.stack.layers.1.mlp.router.weight",  # span 层
        "model.stack.layers.2.self_attn.qkv_proj.weight",  # span 层
        "model.stack.layers.0.self_attn.qkv_proj.weight",  # pre-span 层
        "model.embed_tokens.weight",  # embedding
    ):
        g_max = float(mx.abs(grads["jfb"][key]).max().item())
        assert g_max > 0, f"jfb 下 {key} 应收到非零梯度（恒等通路回归）"
    d = maxdiff(
        grads["full"]["model.stack.layers.2.mlp.router.weight"],
        grads["jfb"]["model.stack.layers.2.mlp.router.weight"],
    )
    assert d > 1e-8, f"jfb 与 full 是不同梯度估计，应可分: {d}"
    # iHC 残差模式下 pre-span 通路同样要活
    model = make_model(
        loop_span=2, loop_count=2, loop_grad_mode="jfb", ihc=True, ihc_streams=4
    )
    model.train()

    def loss_fn_ihc(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, g = mx.value_and_grad(loss_fn_ihc)(model.trainable_parameters())
    mx.eval(val, g)
    gflat = dict(tree_flatten(g))
    assert np.isfinite(float(val))
    for key in ("model.embed_tokens.weight", "model.stack.layers.1.mlp.router.weight"):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"[ihc] jfb 下 {key} 应收到非零梯度"
    print(f"jfb backward: OK (full vs jfb span grad max|Δ|={d:.3e})")


def test_jfb_train_step_smoke():
    """jfb 训练步冒烟：compile 关/开各一次，loss 有限，span 层与 embedding
    梯度均非零。"""
    for compile_model in (False, True):
        loss, gflat = _train_step_smoke(
            compile_model, loop_span=2, loop_count=2, loop_grad_mode="jfb"
        )
        assert np.isfinite(loss), f"compile={compile_model} loss 非有限"
        for key in ("model.stack.layers.2.mlp.router.weight", "model.embed_tokens.weight"):
            g_max = float(mx.abs(gflat[key]).max().item())
            assert g_max > 0, f"compile={compile_model} jfb 下 {key} 应收非零梯度"
        print(f"jfb train step smoke (compile={compile_model}): loss={loss:.4f} OK")


def test_extrap_config():
    """loop_extrap 校验与 sidecar 往返；无参数新增。"""
    cfg = _cfg(loop_span=2, loop_count=2, loop_extrap=1.0)
    assert cfg.loop_extrap == 1.0
    assert VibyConfig.from_dict(cfg.to_dict()).loop_extrap == 1.0
    assert _cfg(loop_span=2, loop_count=2).loop_extrap == 0.0
    for bad in (float("nan"), float("inf")):
        try:
            _cfg(loop_span=2, loop_count=2, loop_extrap=bad)
        except ValueError as e:
            assert "loop_extrap" in str(e)
        else:
            raise AssertionError("非有限 loop_extrap 应报错")
    # loop 未生效时报错（含 loop_span=0 与 loop_count=1 两种退化）
    for kw in (dict(loop_span=0, loop_count=2), dict(loop_span=2, loop_count=1)):
        try:
            _cfg(loop_extrap=0.5, **kw)
        except ValueError as e:
            assert "loop_extrap" in str(e)
        else:
            raise AssertionError(f"loop 未生效时 loop_extrap≠0 应报错: {kw}")
    print("extrap config: OK")


def test_extrap_zero_bitwise():
    """λ=0 与未设 extrap 逐位一致（同 seed）。"""
    ids = rand_ids(24, seed=5)
    a = make_model(loop_span=2, loop_count=2)
    b = make_model(loop_span=2, loop_count=2, loop_extrap=0.0)
    d = maxdiff(a(ids).logits, b(ids).logits)
    assert d == 0.0, f"λ=0 应与 extrap-off 逐位一致: {d}"
    print("extrap λ=0 bitwise: OK")


def test_extrap_matches_manual_reference():
    """λ=1 输出改变，且与手工参考 h₂ + (h₂ − h₁) 对齐。

    手工参考：按 exec_order 逐层跑 stack，在每次 visit 结束的 exec_pos
    记录 hidden，末次 visit 后手动外推，再跑完 post-span 层。
    """
    model0 = make_model(loop_span=2, loop_count=2)  # λ=0
    model1 = make_model(loop_span=2, loop_count=2, loop_extrap=1.0)
    ids = rand_ids(16, seed=9)
    d = maxdiff(model0(ids).logits, model1(ids).logits)
    assert d > 1e-4, f"λ=1 应改变输出: {d}"

    stack = model0.model.stack  # 与 model1 同权重（extrap 无参数）
    cfg = model0.config
    start, end = cfg.loop_layer_range()
    span_len = end - start
    visit_ends = {start + (k + 1) * span_len - 1 for k in range(cfg.loop_count)}
    h = model0.model.dropout(model0.model.embed_norm(model0.model.embed_tokens(ids)))
    pos_emb = model0.model.position_embeddings(0, ids.shape[1], h.dtype)
    residuals = [h]
    ends = {}
    final_end = max(visit_ends)
    # 锚点（默认开）：span 区段（exec_pos ∈ [start, start+r·span)，含
    # visit 1）的层读侧常驻 [span 入口 hidden] + [各 visit 出口摘要] +
    # 全部 pre-span 写入
    x_anchor = None
    pin = 0
    summaries = []
    for exec_pos, li in enumerate(stack.exec_order):
        if exec_pos > final_end:
            break
        if exec_pos == start:
            x_anchor = h
            pin = len(residuals)
        in_span = x_anchor is not None and exec_pos < start + cfg.loop_count * span_len
        h, _ = stack.layers[li](
            h,
            residuals=residuals,
            mask_is_full=True,
            position_embeddings=pos_emb,
            loop_anchor=([x_anchor] + summaries) if in_span else None,
            loop_pin=pin if in_span else 0,
        )
        if exec_pos in visit_ends:
            ends[exec_pos] = h
            summaries.append(h)
    h = ends[final_end] + 1.0 * (ends[final_end] - ends[min(visit_ends)])
    for li in stack.exec_order[final_end + 1 :]:
        h, _ = stack.layers[li](
            h, residuals=residuals, mask_is_full=True, position_embeddings=pos_emb
        )
    h = stack.final_norm(h)
    logits_ref = model0._lm_logits(h)
    d_ref = maxdiff(model1(ids).logits, logits_ref)
    assert d_ref < 1e-5, f"λ=1 应等于手工参考 h₂+(h₂−h₁): {d_ref}"
    print(f"extrap manual reference: max|Δ|={d_ref:.3e} OK")


def test_extrap_prefill_decode_consistency():
    """λ=1 下整段前向 == 分段 prefill == 逐 token decode（T=1 语义相同）。"""
    model = make_model(loop_span=2, loop_count=2, loop_extrap=1.0)
    ids = rand_ids(24, seed=3)
    full = model(ids).logits
    past = None
    outs = []
    for t in range(24):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        outs.append(o.logits)
    d_dec = maxdiff(full, mx.concatenate(outs, axis=1))
    assert d_dec < ATOL, f"λ=1 逐 token decode 不一致: {d_dec:.4f}"
    print(f"extrap prefill/decode: max|Δ|={d_dec:.3e} OK")


def test_jfb_extrap_combined():
    """jfb + λ=1 组合：前向运行且不同于纯 jfb；反向 span 与 pre-span 参数
    均收非零梯度。"""
    ids = rand_ids(16, B=2, seed=7)
    jfb = make_model(loop_span=2, loop_count=2, loop_grad_mode="jfb")
    combo = make_model(
        loop_span=2, loop_count=2, loop_grad_mode="jfb", loop_extrap=1.0
    )
    d = maxdiff(jfb(ids).logits, combo(ids).logits)
    assert d > 1e-4, f"jfb+extrap 输出应与纯 jfb 可分: {d}"
    combo.train()

    def loss_fn(p):
        combo.update(p)
        return combo(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(combo.trainable_parameters())
    mx.eval(val, grads)
    assert np.isfinite(float(val)), f"jfb+extrap loss 非有限: {val}"
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.stack.layers.2.mlp.router.weight",  # span 层
        "model.stack.layers.0.self_attn.qkv_proj.weight",  # pre-span 层
        "model.embed_tokens.weight",  # embedding
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"jfb+extrap 下 {key} 应收到非零梯度"
    print("jfb+extrap combined: OK")


def test_extrap_old_checkpoint_probe():
    """旧 checkpoint 探针路径：extrap=0 的权重可直接加载进 extrap=1.0
    的 config（无新增参数、严格 shape 对齐）并正常前向。"""
    old = make_model(loop_span=2, loop_count=2)  # 模拟旧 checkpoint
    new = make_model(loop_span=2, loop_count=2, loop_extrap=1.0)
    names_old = sorted(k for k, _ in tree_flatten(old.parameters()))
    names_new = sorted(k for k, _ in tree_flatten(new.parameters()))
    assert names_old == names_new, "extrap 不得新增任何参数"
    new.update(tree_unflatten(tree_flatten(old.parameters())))
    ids = rand_ids(16, seed=11)
    out = new(ids).logits
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item()), "探针前向输出应有限"
    # 探针语义：同权重下 λ=1 输出与旧模型（λ=0）可分
    assert maxdiff(out, old(ids).logits) > 1e-4
    print("extrap old-checkpoint probe: OK")


def test_anchor_config():
    """loop_anchor 默认开；sidecar 往返；旧 sidecar 缺键 → True；锚点关 +
    窗口不足时 config 警告，锚点开（或窗口够/全历史）时不警告。"""
    import warnings

    cfg = _cfg(loop_span=2, loop_count=2)
    assert cfg.loop_anchor is True
    assert VibyConfig.from_dict(cfg.to_dict()).loop_anchor is True
    assert _cfg(loop_span=2, loop_count=2, loop_anchor=False).loop_anchor is False
    old = {k: v for k, v in cfg.to_dict().items() if k != "loop_anchor"}
    assert VibyConfig.from_dict(old).loop_anchor is True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _cfg(loop_span=2, loop_count=2, attn_res_window=2, loop_anchor=False)
    assert any("loop_span" in str(w.message) for w in rec), "锚点关+窗口不足应警告"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _cfg(loop_span=2, loop_count=2, attn_res_window=2)  # 锚点默认开
        _cfg(loop_span=2, loop_count=2, attn_res_window=0, loop_anchor=False)
        _cfg(loop_span=2, loop_count=2, attn_res_window=5, loop_anchor=False)
    assert not any("loop_span" in str(w.message) for w in rec), "这些组合不应警告"
    print("anchor config: OK")


def test_anchor_changes_output():
    """loop 生效时锚点开/关输出可分；无 loop 时锚点不起作用（逐位一致）。"""
    ids = rand_ids(24, seed=13)
    on = make_model(loop_span=2, loop_count=2, attn_res_window=1)
    off = make_model(loop_span=2, loop_count=2, attn_res_window=1, loop_anchor=False)
    d = maxdiff(on(ids).logits, off(ids).logits)
    assert d > 1e-4, f"锚点开/关输出应可分: {d}"
    a = make_model()
    b = make_model(loop_anchor=False)
    assert maxdiff(a(ids).logits, b(ids).logits) == 0.0, "无 loop 时锚点应不起作用"
    print(f"anchor on/off: max|Δ|={d:.3e} OK")


def test_anchor_prefill_decode_consistency():
    """锚点开 + window=1（锚点实际承载输入通路）：整段前向 == 逐 token decode。"""
    model = make_model(loop_span=2, loop_count=2, attn_res_window=1)
    ids = rand_ids(24, seed=17)
    full = model(ids).logits
    past = None
    outs = []
    for t in range(24):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        outs.append(o.logits)
    d = maxdiff(full, mx.concatenate(outs, axis=1))
    assert d < ATOL, f"锚点 decode 不一致: {d:.4f}"
    print(f"anchor prefill/decode: max|Δ|={d:.3e} OK")


def test_anchor_jfb_forward_bitwise():
    """jfb+锚点 与 full+锚点前向逐位一致；window=1 极限下 jfb 的 embedding
    与 pre-span 层仍收到非零梯度（x_entry 直通 + 锚点读两条通路）。"""
    ids = rand_ids(16, B=2, seed=19)
    kw = dict(loop_span=2, loop_count=2, attn_res_window=1)
    full = make_model(**kw)
    jfb = make_model(loop_grad_mode="jfb", **kw)
    d = maxdiff(full(ids).logits, jfb(ids).logits)
    assert d == 0.0, f"jfb+锚点前向应与 full 逐位一致: {d}"
    jfb.train()

    def loss_fn(p):
        jfb.update(p)
        return jfb(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(jfb.trainable_parameters())
    mx.eval(val, grads)
    assert np.isfinite(float(val)), f"jfb+锚点 loss 非有限: {val}"
    gflat = dict(tree_flatten(grads))
    for key in (
        "model.embed_tokens.weight",
        "model.stack.layers.0.self_attn.qkv_proj.weight",
    ):
        g_max = float(mx.abs(gflat[key]).max().item())
        assert g_max > 0, f"jfb+锚点(window=1) 下 {key} 应收到非零梯度"
    print("anchor+jfb bitwise/backward: OK")


def test_anchor_pin_semantics():
    """_attn_res_read 的 pin：混合集合 = [register] + writes[:pin] +
    writes[-window:]（不足 pin+window 时全量不重复；pin=0 退化为原窗口）。"""
    from model.block import _attn_res_merge, _attn_res_read

    mx.random.seed(0)
    D = 32
    reg = mx.random.normal((2, 3, D))
    writes = [mx.random.normal((2, 3, D)) for _ in range(9)]
    w = mx.random.normal((D,))
    out = _attn_res_read(w, reg, writes, window=4, pin=3)
    ref = _attn_res_merge(w, [reg] + writes[:3] + writes[-4:], 0)
    assert maxdiff(out, ref) == 0.0
    out2 = _attn_res_read(w, reg, writes[:6], window=4, pin=3)
    ref2 = _attn_res_merge(w, [reg] + writes[:6], 0)
    assert maxdiff(out2, ref2) == 0.0
    out3 = _attn_res_read(w, reg, writes, window=4, pin=0)
    ref3 = _attn_res_merge(w, [reg] + writes[-4:], 0)
    assert maxdiff(out3, ref3) == 0.0
    # 多寄存器（loop 锚点列表 = [入口, visit 摘要...]）：全部常驻
    regs = [reg, mx.random.normal((2, 3, D))]
    out4 = _attn_res_read(w, regs, writes, window=4, pin=3)
    ref4 = _attn_res_merge(w, regs + writes[:3] + writes[-4:], 0)
    assert maxdiff(out4, ref4) == 0.0
    print("anchor pin semantics: OK")


if __name__ == "__main__":
    test_config_roundtrip_and_validation()
    test_exec_order()
    test_degenerate_equivalence()
    test_weight_sharing_param_count()
    test_res_scale_numerical()
    test_loop_prefill_decode_consistency()
    test_loop_backward()
    test_loop_train_step_smoke()
    test_mfu_counts_loop()
    test_engine_loop_smoke()
    test_jfb_config()
    test_jfb_forward_bitwise_equal()
    test_jfb_backward()
    test_jfb_train_step_smoke()
    test_extrap_config()
    test_extrap_zero_bitwise()
    test_extrap_matches_manual_reference()
    test_extrap_prefill_decode_consistency()
    test_jfb_extrap_combined()
    test_extrap_old_checkpoint_probe()
    test_anchor_config()
    test_anchor_changes_output()
    test_anchor_prefill_decode_consistency()
    test_anchor_jfb_forward_bitwise()
    test_anchor_pin_semantics()
    print("all loop tests OK")
