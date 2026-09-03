"""iHC 融合核 vs eager：fwd / 梯度 / α=0 恒等。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from model.block import VibyBlock
from model.config import VibyConfig
from model.ihc import IHCGate
from model.kernels import ihc_fused


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


def test_ihc_fused_prewarm():
    assert ihc_fused.prewarm(4, 128, mx.float32, 1e-6)
    assert ihc_fused.prewarm(4, 64, mx.float32, 1e-6)
    assert ihc_fused.prewarm(4, 128, mx.bfloat16, 1e-6)
    print("ihc fused prewarm f32/bf16: OK")


def test_ihc_fused_mix_write_matches_python():
    mx.random.seed(0)
    m, d = 4, 128
    assert ihc_fused.prewarm(m, d, mx.float32, 1e-6)
    gate = IHCGate(d, m)
    r = mx.random.normal((2, 5, m, d)).astype(mx.float32)
    delta = mx.random.normal((2, 5, d)).astype(mx.float32)
    mx.eval(r, delta, *dict(tree_flatten(gate.parameters())).values())
    x_k, hp_k = gate.mix(r)
    h_pre, h_post = gate.gates(r)
    x_p = gate.read(r, h_pre)
    # write() 也会走核；对照 python 公式
    r_k = gate.write(r, delta, hp_k)
    r_p = r + h_post[..., None] * delta[..., None, :]
    mx.eval(x_k, x_p, r_k, r_p)
    dx = maxdiff(x_k, x_p)
    dr = maxdiff(r_k, r_p)
    assert dx < 2e-4, f"mix x |Δ|={dx:.3e}"
    assert dr < 2e-4, f"write |Δ|={dr:.3e}"
    print("ihc fused mix/write vs python: OK")


def test_ihc_fused_alpha_zero_still_read_h():
    mx.random.seed(1)
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
    for g in (b.ihc_attn, b.ihc_mlp):
        g.alpha = mx.zeros_like(g.alpha)
    mx.eval(b.ihc_attn.alpha, b.ihc_mlp.alpha)
    x = mx.random.normal((2, 3, cfg_h.hidden_size)).astype(mx.float32)
    ya, _ = a(x, residuals=[], mask_is_full=True)
    yb, _ = b(x, mask_is_full=True)
    mx.eval(ya, yb)
    d = maxdiff(ya, yb)
    assert d < 2e-5, f"fused α=0 应等于 read_h, |Δ|={d:.3e}"
    print("ihc fused alpha=0 matches read_h: OK")


if __name__ == "__main__":
    test_ihc_fused_prewarm()
    test_ihc_fused_mix_write_matches_python()
    test_ihc_fused_alpha_zero_still_read_h()
    print("all ihc fused tests passed")
