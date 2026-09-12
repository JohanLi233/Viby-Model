"""Rectangular existing Metal kernel against the explicit-position SDPA oracle."""

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from test_ced_recurrent import config, inputs
from trainer.utils import convert_model_dtype
import model.attention as attention


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("compiled", [False, True])
def test_fused_rectangular_logits_and_weight_gradients(monkeypatch, head_dim, compiled):
    model = build(config(n_heads=16, head_dim=head_dim, max_seq_len=64, window_size=16))
    convert_model_dtype(model, "bfloat16")
    x = inputs(32)
    params = model.trainable_parameters()
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(attention, "_RECURRENT_SPARSE", enabled)
        model.update(params)

        def objective(weights):
            model.update(weights)
            out = model(x, labels=x, need_logits=True)
            return out.loss, out.logits

        call = mx.value_and_grad(objective)
        if compiled:
            call = mx.compile(call)
        (loss, logits), grads = call(params)
        model.update(params)
        mx.eval(loss, logits, grads)
        assert bool(mx.isfinite(loss))
        assert all(bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(grads))
        outputs.append((float(loss), logits, dict(tree_flatten(grads))))
    assert abs(outputs[0][0] - outputs[1][0]) < 0.015
    assert max_abs_diff(outputs[0][1], outputs[1][1]) < 0.08
    for key in (
        "model.layers.6.attn.compressor.wkv.weight",
        "model.layers.7.attn.wq_a.weight",
        "model.layers.10.ffn.shared.w1.weight",
    ):
        native, fused = outputs[0][2][key], outputs[1][2][key]
        relative = mx.sqrt(
            mx.sum((native.astype(mx.float32) - fused.astype(mx.float32)) ** 2)
        )
        relative /= mx.maximum(mx.sqrt(mx.sum(native.astype(mx.float32) ** 2)), 1e-8)
        assert float(relative) < 0.12, (key, float(relative))


def test_interior_padding_retains_position_aware_window(monkeypatch):
    model = build(config(n_heads=16, head_dim=64))
    convert_model_dtype(model, "bfloat16")
    model.eval()
    x = inputs(17)
    mask = mx.array([[1] * 4 + [0] * 5 + [1] * 8])
    monkeypatch.setattr(attention, "_RECURRENT_SPARSE", False)
    expected = model(x, attention_mask=mask).logits
    monkeypatch.setattr(attention, "_RECURRENT_SPARSE", True)
    actual = model(x, attention_mask=mask).logits
    # Position-aware Metal now supports this mask as well as the SDPA oracle.
    assert max_abs_diff(expected, actual) < 0.08
