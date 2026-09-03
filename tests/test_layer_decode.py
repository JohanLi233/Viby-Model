"""整层 decode 融合核：与 eager VibyBlock 对拍。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.block import VibyBlock
from model.cache import KVCache
from model.config import VibyConfig
from model.kernels import layer_decode


def _cfg(**kw):
    base = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        mtp_depth=0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_latent_dim=0,
        ngram_table_size=0,
        use_linear_attn=False,
        dropout=0.0,
        attn_res_window=4,
    )
    base.update(kw)
    return VibyConfig(**base)


def _maxdiff(a, b):
    return float((a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item())


def _run_pair(dtype, latent=0, t_past=8):
    mx.random.seed(0)
    layer_decode._DISABLED = False
    layer_decode._FAILED.clear()
    layer_decode._VERIFIED.clear()
    block = VibyBlock(_cfg(moe_latent_dim=latent), layer_idx=0)
    block.eval()
    D = 64
    attn = block.self_attn
    x = mx.random.normal((1, 1, D)).astype(dtype)
    residuals = [mx.random.normal((1, 1, D)).astype(dtype) for _ in range(3)]
    cache = KVCache()
    if t_past:
        cache.update(
            mx.random.normal((1, t_past, attn.n_heads, attn.qk_dim)).astype(dtype),
            mx.random.normal((1, t_past, attn.n_heads, attn.head_dim)).astype(dtype),
        )
    pe = attn._fallback_pos(int(cache.offset), 1, dtype)

    incoming = list(residuals)
    y_e, va_e, vm_e, nk_e, nv_e = layer_decode._eager_ref(block, x, incoming, cache, pe)
    mx.eval(y_e, va_e, vm_e, nk_e, nv_e)

    res_f = list(residuals)
    cache_f = KVCache()
    if t_past:
        cache_f.update(
            cache.keys[:, :, :t_past].transpose(0, 2, 1, 3),
            cache.values[:, :, :t_past].transpose(0, 2, 1, 3),
        )
    out = layer_decode.try_layer_decode(block, x, res_f, cache_f, pe, mask_is_full=True)
    assert out is not None, "layer_decode 回退 eager（编译或校验失败）"
    y_f, cache_f = out
    mx.eval(y_f, cache_f.keys, cache_f.values)
    tol = 2e-3 if dtype == mx.float32 else 5e-2
    kv_tol = 2e-3 if dtype == mx.float32 else 2e-2
    dy = _maxdiff(y_f, y_e)
    dva = _maxdiff(res_f[-2], va_e)
    dvm = _maxdiff(res_f[-1], vm_e)
    dk = _maxdiff(cache_f.keys[:, :, t_past : t_past + 1], nk_e.transpose(0, 2, 1, 3))
    dv = _maxdiff(cache_f.values[:, :, t_past : t_past + 1], nv_e.transpose(0, 2, 1, 3))
    assert dy <= tol, f"y |Δ|={dy:.3e}"
    assert dva <= tol, f"v_attn |Δ|={dva:.3e}"
    assert dvm <= tol, f"v_mlp |Δ|={dvm:.3e}"
    assert dk <= kv_tol, f"nk |Δ|={dk:.3e}"
    assert dv <= kv_tol, f"nv |Δ|={dv:.3e}"
    return dy, dva, dvm, dk, dv


def test_layer_decode_f32():
    d = _run_pair(mx.float32, latent=0)
    print(f"layer_decode f32 no-latent: |Δ| y/va/vm/k/v={d} OK")


def test_layer_decode_bf16():
    d = _run_pair(mx.bfloat16, latent=0)
    print(f"layer_decode bf16 no-latent: |Δ| y/va/vm/k/v={d} OK")


def test_layer_decode_latent():
    d = _run_pair(mx.float32, latent=32)
    print(f"layer_decode f32 latent32: |Δ| y/va/vm/k/v={d} OK")


def test_layer_decode_empty_cache():
    d = _run_pair(mx.float32, latent=0, t_past=0)
    print(f"layer_decode f32 empty cache: |Δ| y/va/vm/k/v={d} OK")


def test_t2_falls_back():
    mx.random.seed(1)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(), layer_idx=0)
    block.eval()
    x = mx.random.normal((1, 2, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = [x]
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None
    y, c = block(x, past_key_value=KVCache(), use_cache=True, mask_is_full=True)
    mx.eval(y)
    assert y.shape == (1, 2, 64)
    print("T=2 回退 eager: OK")


def test_layer_decode_wide_qkv():
    """QKV 很宽时 CA > NT/32，red[] 必须按 NRED=max(NT/32, CA) 分配。"""
    mx.random.seed(0)
    layer_decode._DISABLED = False
    layer_decode._FAILED.clear()
    layer_decode._VERIFIED.clear()
    block = VibyBlock(_cfg(kv_lora_rank=192), layer_idx=0)
    block.eval()
    D = 64
    attn = block.self_attn
    dtype = mx.float32
    x = mx.random.normal((1, 1, D)).astype(dtype)
    residuals = [mx.random.normal((1, 1, D)).astype(dtype) for _ in range(3)]
    cache = KVCache()
    t_past = 8
    cache.update(
        mx.random.normal((1, t_past, attn.n_heads, attn.qk_dim)).astype(dtype),
        mx.random.normal((1, t_past, attn.n_heads, attn.head_dim)).astype(dtype),
    )
    pe = attn._fallback_pos(int(cache.offset), 1, dtype)
    incoming = list(residuals)
    y_e, *_ = layer_decode._eager_ref(block, x, incoming, cache, pe)
    mx.eval(y_e)
    cache_f = KVCache()
    cache_f.update(
        cache.keys[:, :, :t_past].transpose(0, 2, 1, 3),
        cache.values[:, :, :t_past].transpose(0, 2, 1, 3),
    )
    out = layer_decode.try_layer_decode(
        block, x, list(residuals), cache_f, pe, mask_is_full=True
    )
    assert out is not None, "wide QKV 融合核应通过校验"
    y_f, _ = out
    mx.eval(y_f)
    dy = _maxdiff(y_f, y_e)
    assert dy <= 2e-3, f"wide QKV y |Δ|={dy:.3e}"
    print(f"layer_decode wide QKV (kr=192): |Δ| y={dy:.3e} OK")


def test_register_skips_fused_decode():
    """寄存器残差改了 AttnRes 接线，融合 decode 核尚未覆盖，必须回退 eager。"""
    mx.random.seed(2)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(attn_res_register=True), layer_idx=0)
    block.eval()
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = []
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "attn_res_register 下融合核应回退"
    y, _ = block(x, residuals=residuals, mask_is_full=True)
    mx.eval(y)
    assert y.shape == (1, 1, 64)
    print("register residual 回退 eager: OK")


def test_read_h_skips_fused_decode():
    """读 h 仍走寄存器写，融合 decode 核尚未覆盖。"""
    mx.random.seed(2)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(attn_res_register=True, attn_res_read_h=True), layer_idx=0)
    block.eval()
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = []
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "attn_res_read_h 下融合核应回退"
    y, _ = block(x, residuals=residuals, mask_is_full=True)
    mx.eval(y)
    assert y.shape == (1, 1, 64)
    print("read h 回退 eager: OK")


def test_conf_gate_skips_fused_decode():
    """n-gram 置信门控改了写入尺度，融合 decode 核尚未覆盖，必须回退 eager。"""
    mx.random.seed(3)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(), layer_idx=0)
    block.ngram_conf_gate = True
    block.eval()
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = [x]
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "ngram_conf_gate 下融合核应回退"
    print("ngram conf gate 回退 eager: OK")


def test_write_spread_skips_fused_decode():
    """写出基扩展要逐专家 h_k，融合 decode 核尚未覆盖，必须回退 eager。"""
    mx.random.seed(4)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(moe_latent_dim=32, moe_write_spread=True), layer_idx=0)
    block.eval()
    assert block.mlp.write_scale is not None
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = [x]
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "moe_write_spread 下融合核应回退"
    print("write spread 回退 eager: OK")


def test_ihc_skips_fused_decode():
    """iHC 改了残差读写，融合 decode 核尚未覆盖，必须回退 eager。"""
    mx.random.seed(5)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(ihc=True, ihc_streams=4), layer_idx=0)
    block.eval()
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = []
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "ihc 下融合核应回退"
    y, _ = block(x, residuals=residuals, mask_is_full=True)
    mx.eval(y)
    assert y.shape == (1, 1, 64)
    print("ihc 回退 eager: OK")


def test_route_scale_skips_fused_decode():
    """路由尺度在 latent_up 之后乘 amp，融合 decode 核尚未覆盖。"""
    mx.random.seed(6)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(moe_latent_dim=32, moe_route_scale=True), layer_idx=0)
    block.eval()
    assert block.mlp.route_scale
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = [x]
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "moe_route_scale 下融合核应回退"
    print("route scale 回退 eager: OK")


def test_dense_stem_skips_fused_decode():
    """第 0 层 dense FFN 没有 router/experts，融合 decode 核必须回退。"""
    mx.random.seed(7)
    layer_decode._DISABLED = False
    block = VibyBlock(_cfg(first_k_dense_replace=1), layer_idx=0)
    block.eval()
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    cache = KVCache()
    residuals = [x]
    r = layer_decode.try_layer_decode(
        block, x, residuals, cache, None, mask_is_full=True
    )
    assert r is None, "dense stem 下融合核应回退"
    y, _ = block(x, residuals=residuals, mask_is_full=True)
    mx.eval(y)
    assert y.shape == (1, 1, 64)
    print("dense stem 回退 eager: OK")


if __name__ == "__main__":
    test_layer_decode_f32()
    test_layer_decode_bf16()
    test_layer_decode_latent()
    test_layer_decode_empty_cache()
    test_t2_falls_back()
    test_layer_decode_wide_qkv()
    test_register_skips_fused_decode()
    test_read_h_skips_fused_decode()
    test_conf_gate_skips_fused_decode()
    test_write_spread_skips_fused_decode()
    test_ihc_skips_fused_decode()
    test_route_scale_skips_fused_decode()
    test_dense_stem_skips_fused_decode()
    print("all layer_decode tests passed")
