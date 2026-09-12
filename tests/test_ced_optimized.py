"""End-to-end reference gates for cached metadata and masked fused CED."""

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from test_ced_recurrent import config, inputs
from trainer.utils import convert_model_dtype
import model.attention as attention
import model.moe as moe


@pytest.mark.parametrize("layout", ["plain", "packed", "padded", "gap"])
def test_optimized_recurrent_masks_and_gradients(monkeypatch, layout):
    model = build(config(n_heads=16, head_dim=64, window_size=13))
    convert_model_dtype(model, "bfloat16")
    x = inputs(32, 2)
    docs, pad = None, None
    if layout == "packed":
        docs = mx.array([[0] * 5 + [1] * 12 + [2] * 15, [0] * 9 + [1] * 23])
        pad = mx.ones_like(x, mx.bool_)
    elif layout == "padded":
        pad = mx.array([[1] * 25 + [0] * 7, [1] * 19 + [0] * 13], mx.bool_)
    elif layout == "gap":
        pad = mx.array(
            [[1] * 4 + [0] * 8 + [1] * 20, [1] * 13 + [0] * 3 + [1] * 16], mx.bool_
        )
    params = model.trainable_parameters()
    outputs = []
    calls = []
    original = attention.Attention._recurrent_fast

    def observed(self, *args, **kwargs):
        calls.append(self.layer_idx)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(attention.Attention, "_recurrent_fast", observed)
    for enabled in (False, True):
        monkeypatch.setattr(attention, "_RECURRENT_OPTIMIZED", enabled)
        monkeypatch.setattr(moe, "_RECURRENT_MASKED_COUNTS", enabled)

        def objective(weights):
            model.update(weights)
            out = model(
                x,
                labels=x,
                loss_mask=pad,
                attention_mask=pad,
                segment_ids=docs,
                need_logits=True,
            )
            return out.loss, (
                out.logits,
                out.moe_loads,
                out.ced_metrics,
                out.ced_result["candidate_pool"],
            )

        (loss, aux), grads = mx.compile(mx.value_and_grad(objective))(params)
        model.update(params)
        mx.eval(loss, aux, grads)
        assert bool(mx.isfinite(loss))
        assert all(bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(grads))
        outputs.append((loss, aux, grads))
    assert len(calls) == 13 and calls.count(10) == 3
    ref, new = outputs
    assert abs(float(ref[0]) - float(new[0])) < 0.02
    # The native SDPA and existing BF16-MMA backends round differently across
    # 12 physical middle calls. Bound global relative error as well as outliers.
    logits_a, logits_b = ref[1][0].astype(mx.float32), new[1][0].astype(mx.float32)
    assert max_abs_diff(logits_a, logits_b) < 0.125
    assert (
        float(mx.sqrt(mx.sum((logits_a - logits_b) ** 2) / mx.sum(logits_a**2))) < 0.035
    )
    assert bool(mx.array_equal(ref[1][2], new[1][2]))
    assert bool(mx.array_equal(ref[1][3], new[1][3]))
    # Padding exclusion and all physical-call totals must remain exact even
    # where BF16 attention rounding can change a near-threshold expert choice.
    assert bool(mx.array_equal(mx.sum(ref[1][1], axis=1), mx.sum(new[1][1], axis=1)))
    a, b = dict(tree_flatten(ref[2])), dict(tree_flatten(new[2]))
    for path in (
        "model.layers.6.attn.compressor.wkv.weight",
        "model.layers.7.attn.wq_a.weight",
    ):
        af, bf = a[path].astype(mx.float32), b[path].astype(mx.float32)
        rel = mx.sqrt(mx.sum((af - bf) ** 2) / mx.maximum(mx.sum(af**2), 1e-8))
        assert float(rel) < 0.15, (layout, path, float(rel))
