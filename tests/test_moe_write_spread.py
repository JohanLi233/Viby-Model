"""LatentMoE 专家写出基扩展：零初始化恒等、三路径对齐、梯度、sidecar。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from model.config import VibyConfig
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
        moe_latent_dim=64,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_write_spread_sidecar_default_off():
    cfg = _cfg()
    assert cfg.moe_write_spread is False
    old = {k: v for k, v in cfg.to_dict().items() if k != "moe_write_spread"}
    assert VibyConfig.from_dict(old).moe_write_spread is False
    on = _cfg(moe_write_spread=True)
    assert on.moe_write_spread is True
    assert VibyConfig.from_dict(on.to_dict()).moe_write_spread is True
    print("write spread sidecar default off: OK")


def test_write_spread_requires_latent():
    try:
        _cfg(moe_write_spread=True, moe_latent_dim=0)
    except ValueError:
        print("write spread requires latent: OK")
        return
    raise AssertionError("moe_write_spread 在 moe_latent_dim=0 时应报错")


def test_write_spread_zero_scale_identity():
    """s=0 时与关掉 spread 的 LatentMoE 逐位一致（含稠密/稀疏）。"""
    x = mx.random.normal((2, 5, 128))
    mx.random.seed(0)
    off = VibyForCausalLM(_cfg(moe_write_spread=False))
    off.eval()
    mx.random.seed(0)
    on = VibyForCausalLM(_cfg(moe_write_spread=True))
    on.eval()
    mlp_on = on.model.stack.layers[1].mlp
    assert mlp_on.write_scale is not None
    assert float(mx.abs(mlp_on.write_scale).max().item()) < 1e-8

    y_off = off.model.stack.layers[1].mlp(x)
    y_on = mlp_on(x)
    mx.eval(y_off, y_on)
    d = maxdiff(y_off, y_on)
    assert d < 1e-4, f"kernel/默认路径 identity |Δ|={d}"

    mlp_on._KERNEL_MAX_PAIRS = 0
    off.model.stack.layers[1].mlp._KERNEL_MAX_PAIRS = 0
    d = maxdiff(off.model.stack.layers[1].mlp(x), mlp_on(x))
    assert d < 1e-4, f"稠密路径 identity |Δ|={d}"

    mlp_on._DENSE_MAX_PAIRS = 0
    off.model.stack.layers[1].mlp._DENSE_MAX_PAIRS = 0
    d = maxdiff(off.model.stack.layers[1].mlp(x), mlp_on(x))
    assert d < 1e-4, f"稀疏路径 identity |Δ|={d}"
    print("write spread zero scale identity: OK")


def test_write_spread_nonzero_changes_output():
    mx.random.seed(3)
    model = VibyForCausalLM(_cfg(moe_write_spread=True, mtp_depth=0))
    model.eval()
    ids = mx.array([[4, 9, 14, 21, 28]], dtype=mx.int32)
    y0 = model(ids).logits
    mlp = model.model.stack.layers[0].mlp
    mlp.write_scale = mx.ones_like(mlp.write_scale) * 0.25
    mx.eval(mlp.write_scale)
    y1 = model(ids).logits
    mx.eval(y0, y1)
    d = maxdiff(y0, y1)
    assert d > 1e-3, d
    print("write spread nonzero changes output: OK")


def test_write_spread_matches_numpy_reference():
    """稠密/稀疏路径 = 共享 lat_up 混合 + Σ α_k (s_k ⊙ tile(h_k))。"""
    E, K, moe_in, D, d = 8, 2, 48, 128, 64
    mx.random.seed(11)
    model = VibyForCausalLM(
        _cfg(
            moe_write_spread=True,
            n_routed_experts=E,
            num_experts_per_tok=K,
            moe_intermediate_size=moe_in,
            moe_latent_dim=d,
        )
    )
    model.eval()
    mlp = model.model.stack.layers[1].mlp
    mlp.write_scale = (mx.random.normal(mlp.write_scale.shape) * 0.15).astype(
        mx.float32
    )
    mx.eval(mlp.write_scale)
    x = mx.random.normal((2, 4, D))
    idx, w = mlp.router(x)
    B, T, _ = x.shape
    ld = np.array(mlp.lat_down.weight)
    lu = np.array(mlp.lat_up.weight)
    gu_ = np.array(mlp.experts.gate_up_w)
    gw_, uw_ = gu_[:, :moe_in], gu_[:, moe_in:]
    dw_ = np.array(mlp.experts.down_w)
    s_ = np.array(mlp.write_scale)
    xl = np.array(x) @ ld.T
    xn = xl / np.sqrt((xl**2).mean(-1, keepdims=True) + model.config.rms_norm_eps)
    idxn, wn = np.array(idx), np.array(w)

    def silu(g, u):
        return (g / (1.0 + np.exp(-g))) * u

    mix = np.zeros((B, T, d), dtype=np.float32)
    spread = np.zeros((B, T, D), dtype=np.float32)
    n_tile = (D + d - 1) // d
    for b in range(B):
        for t in range(T):
            for j in range(K):
                e = idxn[b, t, j]
                h = silu(xn[b, t] @ gw_[e].T, xn[b, t] @ uw_[e].T)
                y = h @ dw_[e].T
                mix[b, t] += wn[b, t, j] * y
                tiled = np.concatenate([y] * n_tile, axis=-1)[..., :D]
                spread[b, t] += wn[b, t, j] * (s_[e] * tiled)
    mix = mix / np.sqrt((mix**2).mean(-1, keepdims=True) + model.config.rms_norm_eps)
    sh = 0
    for ff in mlp.shared:
        sh = sh + ff(x)
    ref = mx.array(mix @ lu.T) + mx.array(spread) + sh

    mlp._KERNEL_MAX_PAIRS = 0
    dd = maxdiff(ref, mlp(x))
    assert dd < 1e-4, f"稠密路径与参考不符: {dd}"
    mlp._DENSE_MAX_PAIRS = 0
    dd = maxdiff(ref, mlp(x))
    assert dd < 1e-4, f"稀疏路径与参考不符: {dd}"
    print("write spread matches numpy reference: OK")


def test_write_spread_scale_gets_grad():
    mx.random.seed(7)
    model = VibyForCausalLM(_cfg(moe_write_spread=True))
    model.train()
    ids = mx.array(np.random.default_rng(7).integers(3, 256, (2, 16)).astype(np.int64))

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    key = "model.stack.layers.1.mlp.write_scale"
    g_max = float(mx.abs(gflat[key]).max().item())
    assert g_max > 0, f"{key} 应收到非零梯度"
    print("write spread scale gets grad: OK")


def test_write_spread_trunc_normal_keeps_zero():
    mx.random.seed(1)
    model = VibyForCausalLM(_cfg(moe_write_spread=True))
    s = model.model.stack.layers[0].mlp.write_scale
    assert float(mx.abs(s).max().item()) < 1e-8
    print("write spread trunc_normal keeps zero: OK")


def test_write_spread_escapes_hyperball():
    """write_scale 零初始化，必须进 Adam 标量组，否则 MuonH 半径 0 钉死。"""
    import types

    from trainer.muon import create_mixed_optimizer

    mx.random.seed(7)
    model = VibyForCausalLM(_cfg(moe_write_spread=True, vocab_size=256))
    model.train()
    args = types.SimpleNamespace(
        muonh=True,
        learning_rate=3e-3,
        muon_lr=3e-3,
        adam_lr=3e-3,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        ns_steps=5,
    )
    opt = create_mixed_optimizer(model, args, "pretrain")
    ids = mx.array(np.random.default_rng(7).integers(3, 256, (2, 16)).astype(np.int64))

    def loss_fn(m):
        return m(ids, labels=ids).loss

    import mlx.nn as nn

    lg = nn.value_and_grad(model, loss_fn)
    for _ in range(3):
        _, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    s = model.model.stack.layers[0].mlp.write_scale
    n = float(mx.linalg.norm(s.astype(mx.float32)).item())
    assert n > 0, f"write_scale 仍被钉在零: {n}"
    print("write spread escapes hyperball: OK")


def test_write_spread_prefill_decode():
    mx.random.seed(1)
    model = VibyForCausalLM(_cfg(moe_write_spread=True, attn_res_register=True))
    model.eval()
    for layer in model.model.stack.layers:
        layer.mlp.write_scale = mx.ones_like(layer.mlp.write_scale) * 0.2
        mx.eval(layer.mlp.write_scale)
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
    assert d < 2e-3, d
    print("write spread prefill decode: OK")


def test_tile_to_hidden_repeats_and_clips():
    from model.moe import _tile_to_hidden

    h = mx.arange(6).reshape(2, 3).astype(mx.float32)
    t = _tile_to_hidden(h, 8)
    mx.eval(t)
    assert t.shape == (2, 8)
    np.testing.assert_allclose(
        np.array(t),
        np.array(
            [[0, 1, 2, 0, 1, 2, 0, 1], [3, 4, 5, 3, 4, 5, 3, 4]],
            dtype=np.float32,
        ),
    )
    same = _tile_to_hidden(h, 3)
    assert maxdiff(same, h) < 1e-8
    exact = _tile_to_hidden(h, 6)
    np.testing.assert_allclose(
        np.array(exact),
        np.array([[0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5]], dtype=np.float32),
    )
    clip = _tile_to_hidden(h, 2)
    np.testing.assert_allclose(
        np.array(clip), np.array([[0, 1], [3, 4]], dtype=np.float32)
    )
    print("tile to hidden repeats and clips: OK")


def test_write_spread_fused_matches_eager():
    from model.kernels.moe_write_spread import _eager, prewarm, write_spread

    dtype, d, hid, k, e, m = mx.float32, 64, 128, 2, 8, 32
    assert prewarm(dtype, d, hid, k, e, m=m), "write_spread 融合核预热失败"
    mx.random.seed(3)
    y = mx.random.normal((m, k, d)).astype(dtype)
    w = mx.random.normal((m, k)).astype(dtype)
    scale = (mx.random.normal((e, hid)) * 0.2).astype(dtype)
    idx = mx.random.randint(0, e, (m, k)).astype(mx.int32)
    fused = write_spread(y, w, scale, idx)
    assert fused is not None, "融合核回退了 eager"
    ref = _eager(y, w, scale, idx)
    mx.eval(fused, ref)
    dmax = maxdiff(fused, ref)
    assert dmax < 1e-5, f"fused vs eager |Δ|={dmax}"

    def f(y_, w_, s_):
        return (write_spread(y_, w_, s_, idx).astype(mx.float32) ** 2).sum()

    def g(y_, w_, s_):
        return (_eager(y_, w_, s_, idx).astype(mx.float32) ** 2).sum()

    lo, go = mx.value_and_grad(f, argnums=(0, 1, 2))(y, w, scale)
    lr, gr = mx.value_and_grad(g, argnums=(0, 1, 2))(y, w, scale)
    mx.eval(lo, lr, *go, *gr)
    assert abs(float(lo.item()) - float(lr.item())) < 1e-4
    for a, b in zip(go, gr):
        rel = maxdiff(a, b) / (float(mx.abs(b.astype(mx.float32)).max().item()) + 1e-12)
        assert rel < 5e-4, f"fused grad rel={rel:.2e}"
    print("write spread fused matches eager: OK")


if __name__ == "__main__":
    test_write_spread_sidecar_default_off()
    test_write_spread_requires_latent()
    test_write_spread_zero_scale_identity()
    test_write_spread_nonzero_changes_output()
    test_write_spread_matches_numpy_reference()
    test_write_spread_scale_gets_grad()
    test_write_spread_trunc_normal_keeps_zero()
    test_write_spread_escapes_hyperball()
    test_write_spread_prefill_decode()
    test_tile_to_hidden_repeats_and_clips()
    test_write_spread_fused_matches_eager()
    print("all moe_write_spread tests passed")
