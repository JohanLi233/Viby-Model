"""第 2 层哈希 n-gram：零初始化门恒等、因果、prefill/decode 一致。"""

import mlx.core as mx

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.ngram import NgramEmbedding, ngram_indices


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


def test_ngram_zero_gate_identity():
    mx.random.seed(0)
    cfg = _cfg()
    ng = NgramEmbedding(cfg)
    mx.eval(ng.parameters())
    assert float(mx.abs(ng.gate).max().item()) < 1e-8
    ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    delta = ng(ids)
    mx.eval(delta)
    assert delta.shape == (1, 5, cfg.hidden_size)
    assert float(mx.abs(delta).max().item()) < 1e-8
    print("ngram zero gate identity: OK")


def test_ngram_nonzero_changes_logits():
    mx.random.seed(3)
    model = VibyForCausalLM(_cfg())
    model.eval()
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    y0 = model(ids).logits
    model.model.ngram.gate = mx.ones((128,)) * 0.25
    mx.eval(model.model.ngram.gate)
    y1 = model(ids).logits
    mx.eval(y0, y1)
    d = float(mx.max(mx.abs(y0.astype(mx.float32) - y1.astype(mx.float32))).item())
    assert d > 1e-3, d
    print("ngram nonzero gate changes logits: OK")


def test_ngram_model_prefill_decode():
    mx.random.seed(1)
    model = VibyForCausalLM(_cfg())
    model.eval()
    model.model.ngram.gate = mx.ones((128,)) * 0.2
    mx.eval(model.model.ngram.gate)
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
    model.model.ngram.gate = mx.ones((128,)) * 0.2
    mx.eval(model.model.ngram.gate)
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
    model.model.ngram.gate = mx.ones((128,)) * 0.2
    mx.eval(model.model.ngram.gate)
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
    ng.gate = mx.ones((128,)) * 0.5
    mx.eval(ng.parameters())
    ids = mx.array([[3, 7, 11, 19, 23, 29]], dtype=mx.int32)
    seg = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)
    ids_b = mx.array([[0, 0, 0, 19, 23, 29]], dtype=mx.int32)
    d_pack = ng(ids, segment_ids=seg)
    d_iso = ng(ids_b)
    d_leak = ng(ids)
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
    test_ngram_zero_gate_identity()
    test_ngram_nonzero_changes_logits()
    test_ngram_model_prefill_decode()
    test_ngram_future_token_does_not_leak()
    test_ngram_chunked_prefill()
    test_ngram_pack_doc_boundary()
    test_ngram_embedding_honors_segments()
    test_kda_v_head_grad_finite()
    print("ALL NGRAM TESTS PASSED")
