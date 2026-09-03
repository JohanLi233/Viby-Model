"""LatentMoE 路由尺度：RMSNorm 之后乘未归一化 top-k sigmoid 和。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

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
        n_shared_experts=0,
        moe_intermediate_size=48,
        moe_latent_dim=64,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_route_scale_sidecar_default_off():
    cfg = _cfg()
    assert cfg.moe_route_scale is False
    old = {k: v for k, v in cfg.to_dict().items() if k != "moe_route_scale"}
    assert VibyConfig.from_dict(old).moe_route_scale is False
    on = _cfg(moe_route_scale=True)
    assert on.moe_route_scale is True
    assert VibyConfig.from_dict(on.to_dict()).moe_route_scale is True
    print("route scale sidecar default off: OK")


def test_route_scale_requires_latent():
    try:
        _cfg(moe_route_scale=True, moe_latent_dim=0)
    except ValueError as e:
        assert "moe_route_scale" in str(e)
        print("route scale requires latent: OK")
        return
    raise AssertionError("moe_route_scale 在 moe_latent_dim=0 时应报错")


def test_latent_norm_kills_scaling_factor():
    """关着时 RMSNorm(2.5 u)=RMSNorm(u)，routed_scaling_factor 写不进残差。"""
    mx.random.seed(0)
    model = VibyForCausalLM(_cfg(moe_route_scale=False))
    model.eval()
    mlp = model.model.stack.layers[1].mlp
    mlp._KERNEL_MAX_PAIRS = 0
    x = mx.random.normal((2, 4, 128))
    mlp.router.scaling = 2.5
    y_hi = mlp(x)
    mlp.router.scaling = 1.0
    y_lo = mlp(x)
    mx.eval(y_hi, y_lo)
    d = maxdiff(y_hi, y_lo)
    assert d < 2e-3, f"无 route_scale 时 2.5× 应被 RMSNorm 吃掉, |Δ|={d:.3e}"
    print("latent norm kills scaling factor: OK")


def test_route_scale_is_amp_times_off():
    """开着：n_shared=0 时 y_on = amp ⊙ y_off。amp=未归一化 top-k sigmoid 和。"""
    mx.random.seed(1)
    off = VibyForCausalLM(_cfg(moe_route_scale=False))
    off.eval()
    mx.random.seed(1)
    on = VibyForCausalLM(_cfg(moe_route_scale=True))
    on.eval()
    wo = dict(tree_flatten(on.parameters()))
    shared = {k: v for k, v in tree_flatten(off.parameters()) if k in wo}
    on.update(tree_unflatten(list(shared.items())))
    mlp_off = off.model.stack.layers[1].mlp
    mlp_on = on.model.stack.layers[1].mlp
    mlp_off._KERNEL_MAX_PAIRS = 0
    mlp_on._KERNEL_MAX_PAIRS = 0
    x = mx.random.normal((2, 4, 128))
    y_off = mlp_off(x)
    y_on = mlp_on(x)
    _, _, amp = mlp_on.router(x, return_amp=True)
    mx.eval(y_off, y_on, amp)
    d = maxdiff(y_on, y_off * amp.astype(y_off.dtype))
    assert d < 2e-4, f"y_on 应为 amp·y_off, |Δ|={d:.3e}"
    assert float(mx.std(amp.astype(mx.float32)).item()) > 1e-6
    print("route scale is amp times off: OK")


def test_route_scale_dense_sparse_match():
    mx.random.seed(2)
    model = VibyForCausalLM(_cfg(moe_route_scale=True))
    model.eval()
    mlp = model.model.stack.layers[1].mlp
    x = mx.random.normal((2, 4, 128))
    y0 = mlp(x)
    mlp._KERNEL_MAX_PAIRS = 0
    y1 = mlp(x)
    mlp._DENSE_MAX_PAIRS = 0
    y2 = mlp(x)
    mx.eval(y0, y1, y2)
    d01 = maxdiff(y0, y1)
    d12 = maxdiff(y1, y2)
    assert d01 < 2e-4, f"稀疏/稠密 |Δ|={d01:.3e}"
    assert d12 < 2e-4, f"稠密/稀疏 |Δ|={d12:.3e}"
    print("route scale dense/sparse match: OK")


def test_route_scale_diverges_from_off():
    mx.random.seed(3)
    off = VibyForCausalLM(_cfg(moe_route_scale=False))
    off.eval()
    mx.random.seed(3)
    on = VibyForCausalLM(_cfg(moe_route_scale=True))
    on.eval()
    wo = dict(tree_flatten(on.parameters()))
    shared = {k: v for k, v in tree_flatten(off.parameters()) if k in wo}
    on.update(tree_unflatten(list(shared.items())))
    x = mx.random.normal((2, 5, 128))
    d = maxdiff(
        off.model.stack.layers[1].mlp(x),
        on.model.stack.layers[1].mlp(x),
    )
    assert d > 1e-3, f"开 route_scale 应与关掉分叉, |Δ|={d:.3e}"
    print("route scale diverges from off: OK")


if __name__ == "__main__":
    test_route_scale_sidecar_default_off()
    test_route_scale_requires_latent()
    test_latent_norm_kills_scaling_factor()
    test_route_scale_is_amp_times_off()
    test_route_scale_dense_sparse_match()
    test_route_scale_diverges_from_off()
    print("all moe route scale tests passed")
