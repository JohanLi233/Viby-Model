"""默认 MLA full attention：无 KDA / ShortConv，prefill 与 decode 一致。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import mlx.core as mx
import numpy as np

from model.attention import GQAAttention, MLAAttention
from model.block import VibyBlock
from model.config import VibyConfig
from model.kda import KDAAttention
from model.model import VibyForCausalLM


def _cfg(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
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


def test_default_is_mla_not_kda():
    cfg = _cfg()
    assert cfg.use_linear_attn is False
    model = VibyForCausalLM(cfg)
    for layer in model.model.stack.layers:
        assert isinstance(layer.self_attn, MLAAttention), type(layer.self_attn)
        assert layer.mlp_out_conv is None
        assert not hasattr(layer.self_attn, "k_conv")
        assert not hasattr(layer.self_attn, "out_conv")
    print("default MLA, no KDA/ShortConv: OK")


def test_linear_attn_switch_keeps_kda():
    cfg = _cfg(use_linear_attn=True)
    model = VibyForCausalLM(cfg)
    assert isinstance(model.model.stack.layers[0].self_attn, KDAAttention)
    last = model.model.stack.layers[-1]
    assert isinstance(last.self_attn, GQAAttention)
    assert last.mlp_out_conv is not None
    assert not isinstance(last.self_attn, KDAAttention)
    print("use_linear_attn KDA/GQA: OK")


def test_mla_prefill_decode_match():
    mx.random.seed(0)
    model = VibyForCausalLM(_cfg())
    model.eval()
    rng = np.random.default_rng(1)
    ids = mx.array(rng.integers(3, 256, (1, 24)).astype(np.int64))
    full = model(ids).logits
    past = None
    parts = []
    for t in range(24):
        o = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        parts.append(o.logits)
    dec = mx.concatenate(parts, axis=1)
    d = float(mx.max(mx.abs(full.astype(mx.float32) - dec.astype(mx.float32))).item())
    assert d < 2e-3, f"MLA prefill/decode 不一致 {d:.4f}"
    print(f"MLA prefill vs decode: max|Δ|={d:.3e} OK")


def test_mla_block_no_shortconv():
    block = VibyBlock(_cfg(), layer_idx=0)
    assert isinstance(block.self_attn, MLAAttention)
    assert block.mlp_out_conv is None
    print("VibyBlock MLA 无 ShortConv: OK")


def test_position_ids_rope_survives_corrupt_cos_table():
    """position_ids 必须按 rope_freqs 建 cos/sin，不能读已被 init/ckpt 污染的表。"""
    mx.random.seed(0)
    model = VibyForCausalLM(_cfg())
    model.eval()
    ids = mx.array([[3, 7, 11, 19, 23, 29, 31]])
    a = model(ids)
    mx.eval(a.logits)
    model.model.freqs_cos = mx.random.normal(model.model.freqs_cos.shape)
    model.model.freqs_sin = mx.random.normal(model.model.freqs_sin.shape)
    pos = mx.arange(ids.shape[1], dtype=mx.int32)[None, :]
    b = model(ids, position_ids=pos)
    mx.eval(b.logits)
    np.testing.assert_allclose(
        np.array(a.logits.astype(mx.float32)),
        np.array(b.logits.astype(mx.float32)),
        atol=5e-2,
    )
    print("position_ids RoPE ignores corrupt cos table: OK")


def test_old_sidecar_keeps_linear_attn():
    old = _cfg().to_dict()
    old.pop("use_linear_attn", None)
    old.pop("kv_lora_rank", None)
    cfg = VibyConfig.from_dict(old)
    assert cfg.use_linear_attn is True
    fresh = VibyConfig.from_dict(_cfg().to_dict())
    assert fresh.use_linear_attn is False
    print("old sidecar linear attn compat: OK")


if __name__ == "__main__":
    test_default_is_mla_not_kda()
    test_linear_attn_switch_keeps_kda()
    test_mla_prefill_decode_match()
    test_mla_block_no_shortconv()
    test_position_ids_rope_survives_corrupt_cos_table()
    test_old_sidecar_keeps_linear_attn()
    print("all mla tests passed")
