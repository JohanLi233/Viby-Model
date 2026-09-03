"""第 2 层 Engram n-gram：零初始化恒等、因果、prefill/decode 一致。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import mlx.core as mx
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.ngram import _POLY_MULTS, NgramEmbedding, ngram_indices


def _cfg(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        moe_latent_dim=0,
        mtp_depth=0,
        ngram_table_size=1024,
        ngram_layer=2,
        kda_v_head_ratio=1,
    )
    base.update(kw)
    return VibyConfig(**base)


def _activate_ngram(model, scale=0.05):
    """w_v 零初始化 ⇒ 初始恒等；测试需要激活通路时给 w_v 一个小随机值。"""
    ng = model.model.ngram
    ng.w_v.weight = mx.random.normal(ng.w_v.weight.shape) * scale
    mx.eval(ng.w_v.weight)


def test_ngram_indices_causal_prefix():
    ids = mx.array([[3, 7, 11, 19]], dtype=mx.int32)
    h2 = ngram_indices(ids, table_size=1024, order=2)
    h3 = ngram_indices(ids, table_size=1024, order=3)
    mx.eval(h2, h3)
    assert h2.shape == (1, 4) and h3.shape == (1, 4)
    ids2 = ids.at[0, 3].add(1)
    h2b = ngram_indices(ids2, table_size=1024, order=2)
    h3b = ngram_indices(ids2, table_size=1024, order=3)
    mx.eval(h2b, h3b)
    assert mx.array_equal(h2[:, :3], h2b[:, :3]).item()
    assert mx.array_equal(h3[:, :3], h3b[:, :3]).item()
    assert not mx.array_equal(h2[:, 3], h2b[:, 3]).item()
    print("ngram indices causal prefix: OK")


def test_ngram_zero_init_identity():
    mx.random.seed(0)
    cfg = _cfg()
    ng = NgramEmbedding(cfg)
    mx.eval(ng.parameters())
    assert float(mx.abs(ng.w_v.weight).max().item()) < 1e-8
    assert float(mx.abs(ng.conv_w).max().item()) < 1e-8
    ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    e = ng.lookup(ids)
    assert e.shape == (1, 5, ng.d_mem)
    h = mx.random.normal((1, 5, cfg.hidden_size))
    y, tail = ng.fuse(h, e)
    mx.eval(y, tail)
    assert float(mx.abs(y).max().item()) < 1e-8, "w_v=0 时 fuse 输出必须为 0"
    k, dl = 4, ng.dilation
    assert tail.shape == (1, (k - 1) * dl, cfg.hidden_size)
    print("ngram zero init identity: OK")


def test_engram_fuse_prefill_decode():
    """fuse 的卷积尾接续：整段一次算 vs 逐 token 带 conv_state 一致。"""
    mx.random.seed(5)
    cfg = _cfg()
    ng = NgramEmbedding(cfg)
    ng.w_v.weight = mx.random.normal(ng.w_v.weight.shape) * 0.1
    mx.eval(ng.parameters())
    h = mx.random.normal((1, 8, cfg.hidden_size))
    e = mx.random.normal((1, 8, ng.d_mem))
    y_full, _ = ng.fuse(h, e)
    outs = []
    state = None
    for t in range(8):
        y_t, state = ng.fuse(h[:, t : t + 1], e[:, t : t + 1], conv_state=state)
        outs.append(y_t)
    y_dec = mx.concatenate(outs, axis=1)
    mx.eval(y_full, y_dec)
    d = float(mx.max(mx.abs(y_full - y_dec)).item())
    assert d < 1e-5, d
    print(f"engram fuse prefill vs decode: max|Δ|={d:.3e}")
    print("engram fuse prefill decode: OK")


def test_ngram_nonzero_changes_logits():
    mx.random.seed(3)
    model = VibyForCausalLM(_cfg())
    model.eval()
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    y0 = model(ids).logits
    _activate_ngram(model, 0.25)
    y1 = model(ids).logits
    mx.eval(y0, y1)
    d = float(mx.max(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32))).item())
    assert d > 1e-3, d
    print("ngram nonzero w_v changes logits: OK")


def test_ngram_model_prefill_decode():
    mx.random.seed(1)
    model = VibyForCausalLM(_cfg())
    model.eval()
    _activate_ngram(model, 0.2)
    ids = mx.array([[4, 9, 14, 21, 28, 33, 40, 45]], dtype=mx.int32)
    full = model(ids).logits
    past = None
    chunks = []
    for t in range(ids.shape[1]):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        chunks.append(o.logits)
    dec = mx.concatenate(chunks, axis=1)
    mx.eval(full, dec)
    d = float(mx.max(mx.abs(full.astype(mx.float32) - dec.astype(mx.float32))).item())
    print(f"ngram prefill vs decode: max|Δ|={d:.3e}")
    assert d < 2e-3, d
    print("ngram model prefill decode: OK")


def test_ngram_future_token_does_not_leak():
    mx.random.seed(2)
    model = VibyForCausalLM(_cfg())
    model.eval()
    _activate_ngram(model, 0.2)
    ids = mx.array([[5, 8, 12, 16, 20, 24]], dtype=mx.int32)
    extra = mx.array([[99, 98]], dtype=mx.int32)
    a = model(ids).logits
    b = model(mx.concatenate([ids, extra], axis=1)).logits[:, : ids.shape[1]]
    mx.eval(a, b)
    d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
    print(f"ngram causality: max|Δ|={d:.3e}")
    assert d < 2e-3, d
    print("ngram future token no leak: OK")


def test_ngram_chunked_prefill():
    mx.random.seed(4)
    model = VibyForCausalLM(_cfg())
    model.eval()
    _activate_ngram(model, 0.2)
    ids = mx.array([[4, 9, 14, 21, 28, 33, 40, 45, 50, 55, 60]], dtype=mx.int32)
    full = model(ids).logits
    past = None
    outs = []
    for s, e in [(0, 4), (4, 8), (8, 11)]:
        o = model(ids[:, s:e], past_key_values=past, use_cache=True)
        past = o.past_key_values
        outs.append(o.logits)
    chunked = mx.concatenate(outs, axis=1)
    mx.eval(full, chunked)
    d = float(
        mx.max(mx.abs(full.astype(mx.float32) - chunked.astype(mx.float32))).item()
    )
    print(f"ngram chunked prefill: max|Δ|={d:.3e}")
    assert d < 2e-3, d
    print("ngram chunked prefill: OK")


def test_ngram_pack_doc_boundary():
    """打包两篇文档时，后篇首 token 的 n-gram 不得吃前篇尾 token。"""
    ids = mx.array([[3, 7, 11, 19, 23, 29]], dtype=mx.int32)
    seg = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)
    h_leak = ngram_indices(ids, 1024, 2)
    h_iso = ngram_indices(ids, 1024, 2, segment_ids=seg)
    mx.eval(h_leak, h_iso)
    assert mx.array_equal(h_leak[:, :3], h_iso[:, :3]).item()
    assert not mx.array_equal(h_leak[:, 3], h_iso[:, 3]).item(), (
        "文档边界处 bigram 应切断，不该仍用前篇最后一个 token"
    )
    ids_b = mx.array([[0, 0, 0, 19, 23, 29]], dtype=mx.int32)
    h_b = ngram_indices(ids_b, 1024, 2)
    h3_iso = ngram_indices(ids, 1024, 3, segment_ids=seg)
    h3_b = ngram_indices(ids_b, 1024, 3)
    mx.eval(h_b, h3_iso, h3_b)
    assert mx.array_equal(h_iso[:, 3:], h_b[:, 3:]).item()
    assert mx.array_equal(h3_iso[:, 3:], h3_b[:, 3:]).item()
    print("ngram pack doc boundary: OK")


def test_ngram_embedding_honors_segments():
    mx.random.seed(7)
    ng = NgramEmbedding(_cfg())
    mx.eval(ng.parameters())
    ids = mx.array([[3, 7, 11, 19, 23, 29]], dtype=mx.int32)
    seg = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)
    ids_b = mx.array([[0, 0, 0, 19, 23, 29]], dtype=mx.int32)
    d_pack = ng.lookup(ids, segment_ids=seg)
    d_iso = ng.lookup(ids_b)
    d_leak = ng.lookup(ids)
    mx.eval(d_pack, d_iso, d_leak)
    d_ok = float(
        mx.max(
            mx.abs(d_pack[:, 3:].astype(mx.float32) - d_iso[:, 3:].astype(mx.float32))
        ).item()
    )
    d_cross = float(
        mx.max(
            mx.abs(d_leak[:, 3:].astype(mx.float32) - d_pack[:, 3:].astype(mx.float32))
        ).item()
    )
    assert d_ok < 1e-6, d_ok
    assert d_cross > 1e-5, d_cross
    print("ngram embedding honors segments: OK")


def test_ngram_stacked_lookup_matches_loop():
    """一次 stack-gather 与逐头循环 gather+拼接逐位相同；table 梯度有限非零。"""
    mx.random.seed(4)
    cfg = _cfg()
    ng = NgramEmbedding(cfg)
    mx.eval(ng.parameters())
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    e = ng.lookup(ids)
    refs = []
    i = 0
    for order in ng.orders:
        for k in range(ng.heads):
            idx = ngram_indices(
                ids, ng.sizes[i], order, mult=_POLY_MULTS[k % len(_POLY_MULTS)]
            )
            refs.append(ng.table[idx + int(ng._offsets[i].item())])
            i += 1
    ref = mx.concatenate(refs, axis=-1)
    mx.eval(e, ref)
    assert e.shape == (1, ids.shape[1], ng.d_mem)
    assert mx.array_equal(e, ref).item()

    def loss_fn(p):
        ng.update(p)
        return ng.lookup(ids).square().sum()

    val, grads = mx.value_and_grad(loss_fn)(ng.trainable_parameters())
    mx.eval(val, grads)
    g = dict(tree_flatten(grads))["table"]
    assert mx.all(mx.isfinite(g)).item()
    assert float(mx.abs(g).max().item()) > 0
    print("ngram stacked lookup matches loop: OK")


def test_ngram_table_excluded_from_active():
    on = VibyForCausalLM(_cfg(ngram_table_size=1024))
    off = VibyForCausalLM(_cfg(ngram_table_size=0))
    table = int(on.model.ngram.table.size)
    assert on.ngram_lookup_parameters() == table
    assert off.ngram_lookup_parameters() == 0
    # 表不进激活；w_k/w_v/conv_w/norm_v 每 token 都用，仍计入
    nontable = sum(
        int(a.size)
        for p, a in tree_flatten(on.model.ngram.trainable_parameters())
        if p != "table"
    )
    assert on.num_active_parameters() - off.num_active_parameters() == nontable
    assert on.num_parameters() - off.num_parameters() == table + nontable
    print("ngram table excluded from active: OK")


def test_ngram_logit_skip_sidecar_default_off():
    cfg = _cfg()
    assert cfg.ngram_logit_skip is False
    old = {k: v for k, v in cfg.to_dict().items() if k != "ngram_logit_skip"}
    assert VibyConfig.from_dict(old).ngram_logit_skip is False
    on = _cfg(ngram_logit_skip=True)
    assert on.ngram_logit_skip is True
    assert VibyConfig.from_dict(on.to_dict()).ngram_logit_skip is True
    print("ngram logit skip sidecar default off: OK")


def test_ngram_logit_skip_zero_scale_identity():
    """s=0 时与关闭 skip 逐位一致（hidden 门仍为零）。"""
    mx.random.seed(7)
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    off = VibyForCausalLM(_cfg(ngram_logit_skip=False, mtp_depth=0))
    off.eval()
    y_off = off(ids).logits
    mx.random.seed(7)
    on = VibyForCausalLM(_cfg(ngram_logit_skip=True, mtp_depth=0))
    on.eval()
    assert float(mx.abs(on.model.ngram.logit_scale).max().item()) < 1e-8
    y_on = on(ids).logits
    mx.eval(y_off, y_on)
    d = float(mx.max(mx.abs(y_off.astype(mx.float32) - y_on.astype(mx.float32))).item())
    assert d < 1e-5, d
    print("ngram logit skip zero scale identity: OK")


def test_ngram_logit_skip_adds_unembed():
    """logits += s · lm_head(ungated n-gram)；gate=0 时 hidden 路径仍关闭。"""
    mx.random.seed(8)
    model = VibyForCausalLM(_cfg(ngram_logit_skip=True, mtp_depth=0))
    model.eval()
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    y0 = model(ids).logits
    raw = model.model.ngram.lookup(ids)
    model.model.ngram.logit_scale = mx.ones((1,))
    mx.eval(model.model.ngram.logit_scale)
    y1 = model(ids).logits
    ref = y0 + model._lm_logits(raw)
    mx.eval(y0, y1, ref)
    d = float(mx.max(mx.abs(y1.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert d < 2e-4, d
    d0 = float(mx.max(mx.abs(y1.astype(mx.float32) - y0.astype(mx.float32))).item())
    assert d0 > 1e-3, d0
    print("ngram logit skip adds unembed: OK")


def test_ngram_logit_skip_table_grad_with_zero_gate():
    """hidden 门保持 0 时，CE 仍能经 skip 回到 table。"""
    from mlx.utils import tree_flatten

    mx.random.seed(9)
    model = VibyForCausalLM(_cfg(ngram_logit_skip=True, mtp_depth=0))
    model.train()
    model.model.ngram.logit_scale = mx.ones((1,))
    mx.eval(model.model.ngram.logit_scale)
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    labels = mx.array([[9, 14, 21, 28, 33]], dtype=mx.int32)

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=labels).loss

    _, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(grads)
    gflat = dict(tree_flatten(grads))
    gt = gflat["model.ngram.table"]
    gs = gflat["model.ngram.logit_scale"]
    assert float(mx.abs(gt).max().item()) > 0, "table 应收到 skip 梯度"
    assert float(mx.abs(gs).max().item()) > 0, "logit_scale 应收到梯度"
    print("ngram logit skip table grad with zero gate: OK")


def test_ngram_logit_skip_prefill_decode():
    mx.random.seed(10)
    model = VibyForCausalLM(_cfg(ngram_logit_skip=True, mtp_depth=0))
    model.eval()
    model.model.ngram.logit_scale = mx.ones((1,)) * 0.5
    mx.eval(model.model.ngram.logit_scale)
    ids = mx.array([[4, 9, 14, 21, 28, 33, 40, 45]], dtype=mx.int32)
    full = model(ids).logits
    past = None
    chunks = []
    for t in range(ids.shape[1]):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        chunks.append(o.logits)
    dec = mx.concatenate(chunks, axis=1)
    mx.eval(full, dec)
    d = float(mx.max(mx.abs(full.astype(mx.float32) - dec.astype(mx.float32))).item())
    print(f"ngram logit skip prefill vs decode: max|Δ|={d:.3e}")
    assert d < 2e-3, d
    print("ngram logit skip prefill decode: OK")


def test_ngram_logit_skip_mtp_uses_shifted_lookup():
    """MTP step k 的 skip 用 e[:, k:]，与预测 labels[:, k:] 对齐。"""
    mx.random.seed(11)
    model = VibyForCausalLM(
        _cfg(ngram_logit_skip=True, mtp_depth=1, mtp_steps=1, mtp_loss_weight=1.0)
    )
    model.train()
    ids = mx.array([[4, 9, 14, 21, 28, 33]], dtype=mx.int32)
    labels = mx.array([[9, 14, 21, 28, 33, 40]], dtype=mx.int32)
    model.model.ngram.logit_scale = mx.zeros((1,))
    mx.eval(model.model.ngram.logit_scale)
    out0 = model(ids, labels=labels)
    mx.eval(out0.mtp_loss)
    model.model.ngram.logit_scale = mx.ones((1,))
    mx.eval(model.model.ngram.logit_scale)
    out1 = model(ids, labels=labels)
    mx.eval(out1.mtp_loss)
    d = abs(float(out1.mtp_loss.item()) - float(out0.mtp_loss.item()))
    assert d > 1e-4, d
    print("ngram logit skip mtp uses shifted lookup: OK")


def test_ngram_conf_gate_sidecar_default_off():
    cfg = _cfg()
    assert cfg.ngram_conf_gate is False
    old = {k: v for k, v in cfg.to_dict().items() if k != "ngram_conf_gate"}
    assert VibyConfig.from_dict(old).ngram_conf_gate is False
    on = _cfg(ngram_conf_gate=True)
    assert on.ngram_conf_gate is True
    assert VibyConfig.from_dict(on.to_dict()).ngram_conf_gate is True
    print("ngram conf gate sidecar default off: OK")


def test_ngram_conf_gate_requires_table():
    try:
        _cfg(ngram_conf_gate=True, ngram_table_size=0)
    except ValueError:
        print("ngram conf gate requires table: OK")
        return
    raise AssertionError("ngram_conf_gate 在 ngram_table_size=0 时应报错")


def test_ngram_conf_gate_zero_scale_identity():
    """conf_scale=0 ⇒ g=1，与关闭开关逐位一致。"""
    mx.random.seed(12)
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    off = VibyForCausalLM(_cfg(ngram_conf_gate=False, mtp_depth=0))
    off.eval()
    y_off = off(ids).logits
    mx.random.seed(12)
    on = VibyForCausalLM(_cfg(ngram_conf_gate=True, mtp_depth=0))
    on.eval()
    assert float(mx.abs(on.model.ngram.conf_scale).max().item()) < 1e-8
    y_on = on(ids).logits
    mx.eval(y_off, y_on)
    d = float(mx.max(mx.abs(y_off.astype(mx.float32) - y_on.astype(mx.float32))).item())
    assert d < 1e-5, d
    print("ngram conf gate zero scale identity: OK")


def test_write_gate_scales_register_writes():
    """寄存器模式下 write_gate 缩放 attn 写入；MLP 输入会变，只对拍 attn。"""
    from model.block import VibyBlock

    mx.random.seed(13)
    cfg = _cfg(attn_res_register=True, mtp_depth=0, ngram_table_size=0)
    block = VibyBlock(cfg, layer_idx=0)
    block.eval()
    x = mx.random.normal((2, 3, cfg.hidden_size)).astype(mx.float32)
    writes_a = []
    y_a, _ = block(x, residuals=writes_a, mask_is_full=True)
    g = mx.full((2, 3, 1), 0.5, dtype=mx.float32)
    writes_b = []
    y_b, _ = block(x, residuals=writes_b, write_gate=g, mask_is_full=True)
    mx.eval(y_a, y_b, *writes_a, *writes_b)
    d_va = float(mx.max(mx.abs(writes_b[0] - 0.5 * writes_a[0])).item())
    assert d_va < 1e-5, d_va
    ref = x + writes_b[0] + writes_b[1]
    d_y = float(mx.max(mx.abs(y_b.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert d_y < 1e-5, d_y
    print("write gate scales register writes: OK")


def test_ngram_conf_gate_scale_changes_logits():
    mx.random.seed(14)
    model = VibyForCausalLM(_cfg(ngram_conf_gate=True, mtp_depth=0))
    model.eval()
    ids = mx.array([[4, 9, 14, 21, 28, 33]], dtype=mx.int32)
    y0 = model(ids).logits
    model.model.ngram.conf_scale = mx.ones((1,))
    mx.eval(model.model.ngram.conf_scale)
    y1 = model(ids).logits
    mx.eval(y0, y1)
    d = float(mx.max(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32))).item())
    assert d > 1e-4, d
    print("ngram conf gate scale changes logits: OK")


def test_ngram_conf_gate_prefill_decode():
    mx.random.seed(15)
    model = VibyForCausalLM(_cfg(ngram_conf_gate=True, mtp_depth=0))
    model.eval()
    model.model.ngram.conf_scale = mx.ones((1,)) * 0.5
    mx.eval(model.model.ngram.conf_scale)
    ids = mx.array([[4, 9, 14, 21, 28, 33, 40, 45]], dtype=mx.int32)
    full = model(ids).logits
    past = None
    chunks = []
    for t in range(ids.shape[1]):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        chunks.append(o.logits)
    dec = mx.concatenate(chunks, axis=1)
    mx.eval(full, dec)
    d = float(mx.max(mx.abs(full.astype(mx.float32) - dec.astype(mx.float32))).item())
    print(f"ngram conf gate prefill vs decode: max|Δ|={d:.3e}")
    assert d < 2e-3, d
    print("ngram conf gate prefill decode: OK")


def test_kda_v_head_grad_finite():
    from mlx.utils import tree_flatten

    from model.kda import KDAAttention

    mx.random.seed(6)
    cfg = _cfg(hidden_size=192, num_attention_heads=8, head_dim=24, kda_v_head_ratio=2)
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    x = mx.random.normal((2, 17, 192)).astype(mx.bfloat16)

    def loss_fn(p):
        attn.update(p)
        y, _ = attn(x)
        return y.astype(mx.float32).square().sum()

    val, grads = mx.value_and_grad(loss_fn)(attn.trainable_parameters())
    mx.eval(val, grads)
    assert mx.isfinite(val).item()
    gflat = dict(tree_flatten(grads))
    for k, g in gflat.items():
        assert mx.all(mx.isfinite(g)).item(), k
    assert gflat["v_proj.weight"].shape == attn.v_proj.weight.shape
    print("kda v-head grads finite: OK")


if __name__ == "__main__":
    test_ngram_indices_causal_prefix()
    test_ngram_zero_init_identity()
    test_engram_fuse_prefill_decode()
    test_ngram_nonzero_changes_logits()
    test_ngram_model_prefill_decode()
    test_ngram_future_token_does_not_leak()
    test_ngram_chunked_prefill()
    test_ngram_pack_doc_boundary()
    test_ngram_embedding_honors_segments()
    test_ngram_stacked_lookup_matches_loop()
    test_ngram_table_excluded_from_active()
    test_ngram_logit_skip_sidecar_default_off()
    test_ngram_logit_skip_zero_scale_identity()
    test_ngram_logit_skip_adds_unembed()
    test_ngram_logit_skip_table_grad_with_zero_gate()
    test_ngram_logit_skip_prefill_decode()
    test_ngram_logit_skip_mtp_uses_shifted_lookup()
    test_ngram_conf_gate_sidecar_default_off()
    test_ngram_conf_gate_requires_table()
    test_ngram_conf_gate_zero_scale_identity()
    test_write_gate_scales_register_writes()
    test_ngram_conf_gate_scale_changes_logits()
    test_ngram_conf_gate_prefill_decode()
    test_kda_v_head_grad_finite()
    print("ALL NGRAM TESTS PASSED")
