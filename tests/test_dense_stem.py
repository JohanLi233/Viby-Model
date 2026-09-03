"""浅层 dense stem：第 0 层 FeedForward，其后保持 MoE。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten

from model.block import VibyBlock
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.moe import FeedForward, MoEFeedForward


def _cfg(**kw):
    base = dict(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        mtp_depth=0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_latent_dim=0,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def test_sidecar_default_all_moe():
    cfg = _cfg()
    assert cfg.first_k_dense_replace == 0
    assert cfg.dense_intermediate_size == 0
    d = cfg.to_dict()
    old = {
        k: v
        for k, v in d.items()
        if k not in ("first_k_dense_replace", "dense_intermediate_size")
    }
    back = VibyConfig.from_dict(old)
    assert back.first_k_dense_replace == 0
    assert back.dense_intermediate_size == 0
    layers = VibyForCausalLM(back).model.stack.layers
    assert all(isinstance(layer.mlp, MoEFeedForward) for layer in layers)


def test_first_layer_small_dense_rest_moe():
    cfg = _cfg(first_k_dense_replace=1)
    assert cfg.dense_intermediate_size == 32
    model = VibyForCausalLM(cfg)
    layers = model.model.stack.layers
    assert isinstance(layers[0].mlp, FeedForward)
    assert layers[0].mlp.gate_proj.weight.shape == (32, 64)
    assert isinstance(layers[1].mlp, MoEFeedForward)
    assert isinstance(layers[2].mlp, MoEFeedForward)
    flat = dict(tree_flatten(model.parameters()))
    assert "model.stack.layers.0.mlp.router.weight" not in flat
    assert "model.stack.layers.1.mlp.router.weight" in flat


def test_explicit_dense_width():
    cfg = _cfg(first_k_dense_replace=1, dense_intermediate_size=48)
    assert cfg.dense_intermediate_size == 48
    block = VibyBlock(cfg, layer_idx=0)
    assert block.mlp.up_proj.weight.shape == (48, 64)
    assert isinstance(VibyBlock(cfg, layer_idx=1).mlp, MoEFeedForward)


def test_first_k_out_of_range():
    try:
        _cfg(first_k_dense_replace=4)
    except ValueError as e:
        assert "num_hidden_layers" in str(e)
    else:
        raise AssertionError("first_k > L 应报错")


def test_dense_stem_forward():
    mx.random.seed(0)
    model = VibyForCausalLM(_cfg(first_k_dense_replace=1))
    model.eval()
    ids = mx.array([[3, 4, 5, 6]])
    out = model(ids)
    mx.eval(out.logits)
    assert out.logits.shape == (1, 4, 128)
    assert bool(mx.all(mx.isfinite(out.logits)).item())


if __name__ == "__main__":
    test_sidecar_default_all_moe()
    test_first_layer_small_dense_rest_moe()
    test_explicit_dense_width()
    test_first_k_out_of_range()
    test_dense_stem_forward()
    print("all dense stem tests passed")
