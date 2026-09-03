"""iHC（identity Hyper-Connections）：默认关、α=0 恒等、prefill/decode。"""

import math
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from model.block import VibyBlock
from model.config import VibyConfig
from model.ihc import ihc_add_to_stream, ihc_collapse, ihc_expand
from model.model import VibyForCausalLM


def _cfg(**kw):
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
        n_shared_experts=1,
        moe_intermediate_size=48,
        moe_latent_dim=0,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _zero_ihc_alpha(block):
    for g in (block.ihc_attn, block.ihc_mlp):
        g.alpha = mx.zeros_like(g.alpha)
    mx.eval(block.ihc_attn.alpha, block.ihc_mlp.alpha)


def test_ihc_sidecar_default_off():
    cfg = _cfg()
    assert cfg.ihc is False
    assert int(cfg.ihc_streams) == 4
    d = cfg.to_dict()
    assert d["ihc"] is False
    old = {
        k: v
        for k, v in d.items()
        if k
        not in (
            "ihc",
            "ihc_streams",
            "ihc_typed",
            "ihc_collapse",
            "ihc_ngram_stream",
        )
    }
    back = VibyConfig.from_dict(old)
    assert back.ihc is False
    assert back.ihc_typed is False
    assert back.ihc_collapse == "mean"
    assert int(back.ihc_ngram_stream) == 1
    on = _cfg(ihc=True, ihc_streams=4)
    assert on.ihc is True and on.ihc_streams == 4
    assert VibyConfig.from_dict(on.to_dict()).ihc is True
    try:
        _cfg(ihc=True, ihc_streams=1)
    except ValueError as e:
        assert "ihc_streams" in str(e)
    else:
        raise AssertionError("ihc 要求 streams>=2")
    typed = _cfg(ihc=True, ihc_typed=True)
    assert typed.ihc_typed is True
    assert typed.ihc_collapse == "identity"
    assert int(typed.ihc_ngram_stream) == 1
    try:
        _cfg(ihc_typed=True)
    except ValueError as e:
        assert "ihc_typed" in str(e)
    else:
        raise AssertionError("ihc_typed 需要 ihc")
    try:
        _cfg(ihc=True, ihc_typed=True, ihc_ngram_stream=0)
    except ValueError as e:
        assert "ihc_ngram_stream" in str(e)
    else:
        raise AssertionError("identity 塌缩时 n-gram 不能写进流 0")
    print("ihc sidecar default off: OK")


def test_ihc_alpha_zero_matches_read_h():
    """α=0 时 iHC 与寄存器读 h 逐位一致（流是副本、均匀读、写门=1）。"""
    mx.random.seed(0)
    cfg_h = _cfg(attn_res_register=True, attn_res_read_h=True)
    a = VibyBlock(cfg_h, layer_idx=0)
    a.eval()
    cfg_i = _cfg(ihc=True, ihc_streams=4)
    b = VibyBlock(cfg_i, layer_idx=0)
    b.eval()
    wa = dict(tree_flatten(a.parameters()))
    wb = dict(tree_flatten(b.parameters()))
    shared = {k: wa[k] for k in wa if k in wb}
    b.update(tree_unflatten(list(shared.items())))
    _zero_ihc_alpha(b)
    x = mx.random.normal((2, 3, cfg_h.hidden_size)).astype(mx.float32)
    ya, _ = a(x, residuals=[], mask_is_full=True)
    yb, _ = b(x, mask_is_full=True)
    mx.eval(ya, yb)
    d = maxdiff(ya, yb)
    assert d < 2e-5, f"α=0 iHC 应等于 read_h, |Δ|={d:.3e}"
    print("ihc alpha=0 matches read_h: OK")


def test_ihc_nonzero_alpha_diverges():
    mx.random.seed(1)
    cfg = _cfg(ihc=True, ihc_streams=4)
    block = VibyBlock(cfg, layer_idx=0)
    block.eval()
    x = mx.random.normal((2, 3, cfg.hidden_size)).astype(mx.float32)
    y0, _ = block(x, mask_is_full=True)
    _zero_ihc_alpha(block)
    y1, _ = block(x, mask_is_full=True)
    mx.eval(y0, y1)
    assert maxdiff(y0, y1) > 1e-4, "默认 α 应让 iHC 偏离均匀读/写 1"
    print("ihc nonzero alpha diverges: OK")


def test_ihc_prefill_decode():
    mx.random.seed(2)
    model = VibyForCausalLM(_cfg(ihc=True, ihc_streams=4))
    model.eval()
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
    d = maxdiff(full, dec)
    assert d < 2e-3, f"iHC prefill/decode |Δ|={d:.3e}"
    print("ihc prefill decode: OK")


def test_ihc_trunc_normal_keeps_gates():
    mx.random.seed(3)
    model = VibyForCausalLM(_cfg(ihc=True, ihc_streams=4))
    m = 4
    want = math.log(1.0 / (m - 1))
    for layer in model.model.stack.layers:
        for g in (layer.ihc_attn, layer.ihc_mlp):
            bp = float(g.bias_pre[0].item())
            assert abs(bp - want) < 1e-5, (
                f"bias_pre 应保持 logit(1/M)={want}, 得到 {bp}"
            )
            assert float(mx.abs(g.bias_post).max().item()) < 1e-8
    print("ihc trunc_normal keeps gates: OK")


def test_ihc_collapse_identity_is_stream_zero():
    mx.random.seed(5)
    h = mx.random.normal((2, 3, 8))
    r = ihc_expand(h, 4)
    r = r + mx.arange(4).astype(mx.float32).reshape(1, 1, 4, 1)
    mx.eval(r)
    mean = ihc_collapse(r)
    ident = ihc_collapse(r, mode="identity")
    mx.eval(mean, ident)
    assert maxdiff(mean, mx.mean(r, axis=-2)) < 1e-6
    assert maxdiff(ident, r[..., 0, :]) < 1e-6
    assert maxdiff(ident, mean) > 1e-3
    print("ihc collapse identity is stream 0: OK")


def test_ihc_add_to_stream_not_broadcast():
    mx.random.seed(6)
    h = mx.random.normal((2, 3, 8))
    r = ihc_expand(h, 4)
    delta = mx.ones((2, 3, 8))
    out = ihc_add_to_stream(r, delta, 1)
    mx.eval(out)
    assert maxdiff(out[..., 0, :], r[..., 0, :]) < 1e-6
    assert maxdiff(out[..., 2, :], r[..., 2, :]) < 1e-6
    assert maxdiff(out[..., 1, :], r[..., 1, :] + delta) < 1e-6
    print("ihc add to stream not broadcast: OK")


def test_ihc_typed_zero_gate_matches_untyped():
    """n-gram 门=0 时分型与广播 iHC 起步重合（流仍是副本，identity=mean）。"""
    mx.random.seed(7)
    kw = dict(
        ihc=True,
        ihc_streams=4,
        ngram_table_size=64,
        ngram_layer=2,
        mtp_depth=0,
    )
    a = VibyForCausalLM(_cfg(**kw))
    a.eval()
    b = VibyForCausalLM(_cfg(**kw, ihc_typed=True))
    b.eval()
    wb = dict(tree_flatten(b.parameters()))
    shared = {k: v for k, v in tree_flatten(a.parameters()) if k in wb}
    b.update(tree_unflatten(list(shared.items())))
    for model in (a, b):
        for layer in model.model.stack.layers:
            _zero_ihc_alpha(layer)
    ids = mx.array([[4, 9, 14, 21]], dtype=mx.int32)
    ya = a(ids).logits
    yb = b(ids).logits
    mx.eval(ya, yb)
    d = maxdiff(ya, yb)
    assert d < 2e-5, f"typed 在 n-gram 门=0 时应等于 untyped iHC, |Δ|={d:.3e}"
    print("ihc typed zero gate matches untyped: OK")


def test_ihc_typed_ngram_diverges_from_broadcast():
    mx.random.seed(8)
    kw = dict(
        ihc=True,
        ihc_streams=4,
        ngram_table_size=64,
        ngram_layer=2,
        mtp_depth=0,
    )
    a = VibyForCausalLM(_cfg(**kw))
    a.eval()
    b = VibyForCausalLM(_cfg(**kw, ihc_typed=True))
    b.eval()
    wb = dict(tree_flatten(b.parameters()))
    shared = {k: v for k, v in tree_flatten(a.parameters()) if k in wb}
    b.update(tree_unflatten(list(shared.items())))
    gate = mx.ones((128,)) * 0.5
    a.model.ngram.gate = gate
    b.model.ngram.gate = gate
    mx.eval(a.model.ngram.gate, b.model.ngram.gate)
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    ya = a(ids).logits
    yb = b(ids).logits
    mx.eval(ya, yb)
    d = maxdiff(ya, yb)
    assert d > 1e-3, f"n-gram 开时 typed 不应再广播到四路, |Δ|={d:.3e}"
    print("ihc typed ngram diverges from broadcast: OK")


def test_ihc_typed_prefill_decode():
    mx.random.seed(9)
    model = VibyForCausalLM(
        _cfg(
            ihc=True,
            ihc_streams=4,
            ihc_typed=True,
            ngram_table_size=64,
            mtp_depth=0,
        )
    )
    model.eval()
    model.model.ngram.gate = mx.ones((128,)) * 0.25
    mx.eval(model.model.ngram.gate)
    ids = mx.array([[4, 9, 14, 21, 28, 33]], dtype=mx.int32)
    full = model(ids).logits
    past = None
    chunks = []
    for t in range(ids.shape[1]):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        chunks.append(o.logits)
    dec = mx.concatenate(chunks, axis=1)
    mx.eval(full, dec)
    d = maxdiff(full, dec)
    assert d < 2e-3, f"typed iHC prefill/decode |Δ|={d:.3e}"
    print("ihc typed prefill decode: OK")


def test_ihc_gate_params_are_1d_or_2d():
    """α/bias 1-D 进 Adam 标量组；W_h 2-D 进 Muon。"""
    mx.random.seed(4)
    model = VibyForCausalLM(_cfg(ihc=True, ihc_streams=4))
    g = model.model.stack.layers[0].ihc_attn
    assert g.alpha.ndim == 1
    assert g.bias_pre.ndim == 1 and g.bias_post.ndim == 1
    assert g.weight.ndim == 2
    print("ihc gate param ranks: OK")


if __name__ == "__main__":
    test_ihc_sidecar_default_off()
    test_ihc_alpha_zero_matches_read_h()
    test_ihc_nonzero_alpha_diverges()
    test_ihc_prefill_decode()
    test_ihc_trunc_normal_keeps_gates()
    test_ihc_collapse_identity_is_stream_zero()
    test_ihc_add_to_stream_not_broadcast()
    test_ihc_typed_zero_gate_matches_untyped()
    test_ihc_typed_ngram_diverges_from_broadcast()
    test_ihc_typed_prefill_decode()
    test_ihc_gate_params_are_1d_or_2d()
    print("all ihc tests passed")
