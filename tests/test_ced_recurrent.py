"""MLX mechanism gates for residual-lifted recurrent CED; no efficacy claim."""

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from model.cache import VibyCache
from model.config import VibyConfig
from model.recurrent import (
    build_anchor_plan,
    hold_anchors,
    residual_lift,
    sample_anchors,
)


def config(**overrides):
    settings = dict(
        preset="tiny",
        n_layers=12,
        dim=64,
        n_heads=2,
        o_groups=1,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        engram_layer_ids=(),
        n_mtp_layers=0,
        vocab_size=48,
        max_seq_len=64,
        window_size=8,
        ced_recurrent_enabled=True,
        ced_recurrent_stride=4,
        ced_recurrent_rounds=3,
    )
    settings.update(overrides)
    return VibyConfig(**settings)


def inputs(length=17, batch=1):
    return (mx.arange(length * batch).reshape(batch, length) % 40 + 1).astype(mx.int32)


def test_anchor_sampling_lift_and_document_local_hold():
    positions = mx.broadcast_to(mx.arange(15)[None], (2, 15))
    segments = mx.array([[0] * 3 + [1] * 7 + [2] * 5, [0] * 15])
    pad = mx.array([[1] * 13 + [0, 0], [1] * 15], mx.bool_)
    plan = build_anchor_plan(positions, segments, pad, 4)
    anchors = [[6], [3, 7, 11]]
    for b, expected in enumerate(anchors):
        actual = [
            i
            for i, valid in zip(plan.indices[b].tolist(), plan.valid[b].tolist())
            if valid
        ]
        assert actual == expected
    h = mx.arange(2 * 15 * 2 * 3).reshape(2, 15, 2, 3).astype(mx.float32)
    initial = sample_anchors(h, plan)
    final = initial + 5
    lifted = residual_lift(h, initial, final, plan)
    assert (
        max_abs_diff(
            sample_anchors(lifted, plan) * plan.valid[..., None, None],
            final * plan.valid[..., None, None],
        )
        == 0
    )
    held = hold_anchors(initial, plan)
    assert max_abs_diff(hold_anchors(sample_anchors(held, plan), plan), held) == 0
    assert (
        max_abs_diff(
            lifted - hold_anchors(sample_anchors(lifted, plan), plan), h - held
        )
        == 0
    )
    assert plan.covered.tolist() == [
        [False] * 6 + [True] * 4 + [False] * 5,
        [False] * 3 + [True] * 12,
    ]
    assert max_abs_diff(mx.where(plan.covered[..., None, None], h, lifted), h) == 0
    # Positive mHC coefficients use replacement, with the boundary as fallback.
    pre = mx.full((2, 15, 4), 0.25)
    final_pre = mx.full((*plan.indices.shape, 4), 0.75)
    broadcast_pre = hold_anchors(final_pre, plan, fallback=pre)
    expected = mx.where(plan.covered[..., None], 0.75, 0.25)
    assert max_abs_diff(broadcast_pre, mx.broadcast_to(expected, pre.shape)) == 0


def test_lift_composition_and_unsampled_perturbation_identity():
    positions = mx.arange(13)[None]
    plan = build_anchor_plan(positions, None, None, 4)
    h = mx.random.normal((1, 13, 3))

    def transform(x, operation):
        sampled = sample_anchors(x, plan)
        return residual_lift(x, sampled, operation(sampled), plan)

    def f(x):
        return 0.5 * x + 1

    def g(x):
        return 2 * x - 3

    assert (
        max_abs_diff(transform(transform(h, g), f), transform(h, lambda x: f(g(x))))
        < 2e-6
    )
    v = mx.random.normal(h.shape)
    v = v - hold_anchors(sample_anchors(v, plan), plan)
    assert max_abs_diff(transform(h + v, f) - transform(h, f), v) < 2e-6


def test_disabled_and_k1_q1_are_exact_legacy_paths():
    x = inputs(9)
    old = build(config(ced_recurrent_enabled=False))
    for c in (config(), config(ced_recurrent_stride=1, ced_recurrent_rounds=1)):
        model = build(c)
        model.load_weights(tree_flatten(old.parameters()), strict=True)
        assert set(dict(tree_flatten(model.parameters()))) == set(
            dict(tree_flatten(old.parameters()))
        )
        result = model(x, use_ced_recurrent=False).logits
        assert bool(mx.array_equal(result, old(x).logits))
        if c.ced_recurrent_stride == 1:
            assert bool(mx.array_equal(model(x).logits, old(x).logits))


def test_configuration_roundtrip_and_incompatible_objectives():
    c = config()
    assert VibyConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
    for invalid in (
        {"n_mtp_layers": 1},
        {"ced_recurrent_stride": 0},
        {"ced_recurrent_rounds": 0},
    ):
        with pytest.raises(ValueError):
            config(**invalid)


def test_all_prefix_lengths_and_future_replacement():
    model = build(config())
    model.eval()
    x = inputs(17)
    complete = model(x).logits
    mx.eval(complete)
    for length in range(1, x.shape[1] + 1):
        prefix = model(x[:, :length]).logits
        assert max_abs_diff(prefix, complete[:, :length]) < 3e-4, length
    changed = mx.concatenate([x[:, :7], x[:, 7:] + 5], axis=1)
    assert max_abs_diff(model(changed).logits[:, :7], complete[:, :7]) < 3e-4


def test_packed_documents_short_documents_and_padding():
    model = build(config())
    model.eval()
    x = inputs(17)
    segments = mx.array([[0] * 3 + [1] * 9 + [2] * 5])
    pad = mx.array([[1] * 15 + [0, 0]])
    a = model(x, segment_ids=segments, attention_mask=pad).logits
    changed = mx.where(segments == 0, x + 20, x)
    b = model(changed, segment_ids=segments, attention_mask=pad).logits
    assert max_abs_diff(a[:, 3:15], b[:, 3:15]) < 3e-4
    changed = mx.where(pad.astype(mx.bool_), x, x + 20)
    b = model(changed, segment_ids=segments, attention_mask=pad).logits
    assert max_abs_diff(a[:, :15], b[:, :15]) < 3e-4
    for length in range(1, 18):
        b = model(
            x[:, :length],
            segment_ids=segments[:, :length],
            attention_mask=pad[:, :length],
        ).logits
        assert max_abs_diff(a[:, : min(length, 15)], b[:, : min(length, 15)]) < 3e-4, (
            length
        )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_ntp_gradients_reach_evidence_and_every_shared_middle_layer(compiled, dtype):
    model = build(config())
    model.set_dtype(dtype)
    x = inputs(13)

    def objective(m, ids):
        return m(ids, labels=(ids + 1) % m.config.vocab_size, use_mtp=False).loss

    gradient = nn.value_and_grad(model, objective)

    def call(ids):
        return gradient(model, ids)

    if compiled:
        call = mx.compile(call, inputs=model.state)
    loss, grads = call(x)
    mx.eval(loss, grads)
    flat = dict(tree_flatten(grads))
    assert bool(mx.isfinite(loss))
    assert all(bool(mx.all(mx.isfinite(g))) for g in flat.values())
    assert float(mx.max(mx.abs(flat["model.layers.6.attn.compressor.wkv.weight"]))) > 0
    for layer in (7, 8, 9, 10):
        assert any(
            float(mx.max(mx.abs(g))) > 0
            for name, g in flat.items()
            if name.startswith(f"model.layers.{layer}.attn.")
        ), layer
        assert any(
            float(mx.max(mx.abs(g))) > 0
            for name, g in flat.items()
            if name.startswith(f"model.layers.{layer}.ffn.")
        ), layer


@pytest.mark.parametrize("length", [1, 3, 8])
def test_all_padding_bfloat16_gradient_is_finite(length):
    model = build(config())
    model.set_dtype(mx.bfloat16)
    x = inputs(length)

    def objective(m):
        return m(
            x,
            labels=x,
            attention_mask=mx.zeros_like(x),
            loss_mask=mx.zeros_like(x),
            use_mtp=False,
        ).loss

    loss, grads = nn.value_and_grad(model, objective)(model)
    mx.eval(loss, grads)
    assert bool(mx.isfinite(loss))
    assert all(bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(grads))


def test_shared_moe_loads_count_every_physical_call():
    c = config()
    model = build(c)
    x = inputs(16)
    out = model(x, labels=x, use_mtp=False)
    expected_middle = (
        (16 // c.ced_recurrent_stride) * c.ced_recurrent_rounds * c.n_activated_experts
    )
    for row, gate in enumerate(model._backbone_gates):
        layer = gate.layer_idx
        expected = expected_middle if 7 <= layer <= 10 else 16 * c.n_activated_experts
        assert float(mx.sum(out.moe_loads[row])) == expected, layer


def test_prefill_then_decode_matches_every_full_prefix():
    model = build(config())
    model.eval()
    x = inputs(17)
    for initial in (1, 3, 4, 7, 9):
        cache = VibyCache(model.config)
        first = model(x[:, :initial], cache=cache).logits
        assert max_abs_diff(first, model(x[:, :initial]).logits) < 3e-4
        for pos in range(initial, x.shape[1]):
            result = model(
                x[:, pos : pos + 1], cache=cache, start_pos=pos, decode=True
            ).logits
            expected = model(x[:, : pos + 1]).logits[:, -1:]
            assert max_abs_diff(result, expected) < 3e-4, (initial, pos)
