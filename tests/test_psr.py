"""PSR semantic gates: prefix isolation, independent cache, learning and budgets."""

import numpy as np
import pytest
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from _v41_common import build, max_abs_diff
from model.config import VibyConfig
from model.psr import (COMPUTE, EvidenceMemory, IndexReference,
                       counterfactual_value_targets, exact_topk, index_change_bound)


def config(**kw):
    args = dict(preset="tiny", dim=64, n_heads=2, o_groups=1, head_dim=32,
                rope_head_dim=16, q_lora_rank=32, o_lora_rank=32,
                moe_inter_dim=32, n_routed_experts=4, n_activated_experts=2,
                engram_layer_ids=(), n_mtp_layers=0, vocab_size=32,
                max_seq_len=64, window_size=8, psr_enabled=True, psr_dim=32,
                psr_slots=2, psr_topk=2, psr_rounds=2, psr_bridge_init=0.2)
    args.update(kw)
    return VibyConfig(**args)


def run(model, ids, **kw):
    return model(ids, use_mtp=False, return_thinking=True, **kw)


def test_config_roundtrip_and_invalid_boundary():
    cfg = config()
    assert VibyConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    for kw in ({"psr_topk": 0}, {"psr_rounds": 9}, {"psr_read_cost": -1},
               {"compress_ratios": (0, 0, 2, 2)}, {"psr_update_scale": 2}):
        with pytest.raises(ValueError):
            config(**kw)


def test_disabled_path_and_zero_bridge_match_original():
    baseline = build(config(psr_enabled=False))
    model = build(config(psr_bridge_init=0))
    model.load_weights(tree_flatten(baseline.parameters()), strict=False)
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    original = run(baseline, x).logits
    disabled = run(model, x, use_thinking=False).logits
    enabled = run(model, x, thinking_prefix_lengths=4).logits
    mx.eval(original, disabled, enabled)
    assert bool(mx.array_equal(original, disabled))
    assert bool(mx.array_equal(original, enabled))


def test_no_future_answer_leak_and_no_early_prompt_injection():
    model = build(config())
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    changed = mx.array([[1, 2, 3, 4, 20, 21]])
    a = run(model, x, thinking_prefix_lengths=4)
    b = run(model, changed, thinking_prefix_lengths=4)
    off = run(model, x, use_thinking=False)
    assert max_abs_diff(a.thinking_state.slots, b.thinking_state.slots) < 1e-6
    assert max_abs_diff(a.logits[:, :3], off.logits[:, :3]) == 0
    assert max_abs_diff(a.logits[:, 3:], off.logits[:, 3:]) > 1e-5
    for idx in a.thinking_trace.indices:
        assert bool(mx.all((idx >= 0) & (idx < 4)))


def test_observation_kv_precedes_bridge_and_engine_cannot_drop_workspace():
    from engine.engine import VibyEngine
    model = build(config())
    x = mx.array([[1, 2, 3, 4]])
    _, on = model.prefill(x)
    _, off = model.prefill(x, use_thinking=False)
    boundary = model.config.n_encoder_layers
    assert max_abs_diff(on[boundary].compress_kv, off[boundary].compress_kv) == 0
    assert max_abs_diff(on[boundary].index_k, off[boundary].index_k) == 0
    assert on.thinking_state is not None and off.thinking_state is None
    with pytest.raises(ValueError, match="ThinkingState"):
        VibyEngine(model)


def test_per_example_prefix_document_and_padding_masks():
    model = build(config())
    x = mx.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    seg = mx.array([[0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 0, 0]])
    pad = mx.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 0]])
    a = run(model, x, thinking_prefix_lengths=mx.array([4, 5]), segment_ids=seg, attention_mask=pad)
    for idx in a.thinking_trace.indices:
        assert bool(mx.all((idx[0] == 2) | (idx[0] == 3)))
        assert bool(mx.all((idx[1] != 2) & (idx[1] < 5)))
    off = run(model, x, segment_ids=seg, attention_mask=pad, use_thinking=False)
    assert max_abs_diff(a.logits[:, :2], off.logits[:, :2]) == 0
    assert max_abs_diff(a.logits[0, 5], off.logits[0, 5]) < 1e-6


def test_invalid_array_boundary_fails_closed_without_nan():
    model = build(config())
    x = mx.array([[1, 2, 3]])
    a = run(model, x, thinking_prefix_lengths=mx.array([0]))
    b = run(model, x, use_thinking=False)
    assert not bool(a.thinking_state.valid[0])
    assert bool(mx.all(mx.isfinite(a.thinking_state.slots)))
    assert max_abs_diff(a.logits, b.logits) == 0
    assert bool(mx.all(a.thinking_trace.indices[0] == -1))


def test_prefill_decode_agrees_with_teacher_forcing_and_state_is_constant():
    model = build(config())
    model.eval()
    x = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    full = run(model, x, thinking_prefix_lengths=4)
    initial, cache = model.prefill(x[:, :4])
    slots = cache.thinking_state.slots
    decoded = [initial]
    for t in range(4, x.shape[1]):
        logits, _ = model.decode_step(x[:, t], cache)
        decoded.append(logits[:, None])
        assert cache.thinking_state.slots is slots
    assert cache.start_pos == x.shape[1]
    assert max_abs_diff(full.logits, mx.concatenate(decoded, axis=1)) < 2e-5
    assert max_abs_diff(full.thinking_state.slots, slots) < 2e-5
    cache.rewind(1)
    with pytest.raises(ValueError, match="boundary"):
        cache.rewind(cache.start_pos)
    with pytest.raises(ValueError, match="cannot disable"):
        model(x[:, :1], cache=cache, use_thinking=False)


def test_compute_only_ignores_memory_and_fixed_selection_is_an_ablation():
    model = build(config())
    r = model.model.reasoner
    attn = model.model.layers[2].attn
    initial = mx.random.normal((1, 64))
    keys = mx.random.normal((1, 6, 32))
    def memory(scale):
        return EvidenceMemory(mx.random.normal((1, 6, 32)) * scale, keys * scale,
                              mx.ones((1, 6), dtype=mx.bool_), mx.array([5]))
    opts = dict(actions=(COMPUTE, COMPUTE), record_trace=True)
    a, trace = r(initial, memory(1), None, mx.array([True]), attn.freq_cos, attn.freq_sin, **opts)
    b, _ = r(initial, memory(100), None, mx.array([True]), attn.freq_cos, attn.freq_sin, **opts)
    assert max_abs_diff(a.slots, b.slots) == 0
    assert trace.full_scans == 0 and trace.scores == (None, None)
    assert a.rounds == 2
    out = run(model, mx.array([[1, 2, 3, 4]]), thinking_prefix_lengths=4,
              thinking_options={"selection_mode": "fixed"})
    assert bool(mx.array_equal(out.thinking_trace.indices[0], out.thinking_trace.indices[1]))


def test_exact_topk_ties_mask_and_margin_bound():
    score = mx.array([[[1.0, 1.0, 1.0 + 5e-7, -2.0]]])
    idx, _ = exact_topk(score, mx.array([[True, True, True, False]]), 2)
    assert idx.tolist() == [[[0, 2]]]
    rng = np.random.default_rng(42)
    certified = 0
    for _ in range(24):
        keys = rng.normal(size=(1, 16, 8)).astype(np.float32)
        q = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
        w = rng.normal(size=(1, 2, 3)).astype(np.float32)
        q1 = q + rng.normal(scale=1e-4, size=q.shape).astype(np.float32)
        w1 = w + rng.normal(scale=1e-4, size=w.shape).astype(np.float32)
        score0 = (np.maximum(q @ keys[:, None].swapaxes(-1, -2), 0) * w[..., None]).sum(-2)
        score1 = (np.maximum(q1 @ keys[:, None].swapaxes(-1, -2), 0) * w1[..., None]).sum(-2)
        idx, gap = exact_topk(mx.array(score0), mx.ones((1, 16), mx.bool_), 3)
        ref = IndexReference(mx.array(q), mx.array(w), idx, gap, mx.array(np.linalg.norm(keys, axis=-1).max(-1)))
        delta = np.asarray(index_change_bound(ref, mx.array(q1), mx.array(w1)))
        assert np.all(np.abs(score1 - score0) <= delta[..., None] + 1e-5)
        if np.all(np.asarray(gap) > 2 * delta):
            certified += 1
            new_idx, _ = exact_topk(mx.array(score1), mx.ones((1, 16), mx.bool_), 3)
            assert bool(mx.array_equal(idx, new_idx))
    assert certified > 0


def targets():
    return dict(address=mx.array([[[1, 1], [2, 2]]]),
                tests=mx.array([[[0], [0], [0]]]), results=mx.array([[[1], [2], [3]]]),
                values=mx.ones((1, 3, 3)))


def test_losses_reach_indexer_state_value_and_bridge_without_label_leakage():
    model = build(config())
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    target = targets()
    def loss(m):
        return run(m, x, labels=x, thinking_prefix_lengths=4, thinking_targets=target).loss
    value, grads = nn.value_and_grad(model, loss)(model)
    mx.eval(value, grads)
    assert np.isfinite(float(value))
    flat = dict(tree_flatten(grads))
    for path in ("model.reasoner.index_query.weight", "model.reasoner.index_weights.weight",
                 "model.reasoner.blocks.0.up.weight", "model.reasoner.test_head.weight",
                 "model.reasoner.value_head.weight", "model.workspace_bridges.0.out.weight"):
        assert bool(mx.all(mx.isfinite(flat[path]))), path
        assert float(mx.max(mx.abs(flat[path]))) > 0, path
    a = run(model, x, thinking_prefix_lengths=4, thinking_targets=target)
    target["address"] = mx.zeros_like(target["address"])
    target["results"] = mx.zeros_like(target["results"])
    b = run(model, x, thinking_prefix_lengths=4, thinking_targets=target)
    assert max_abs_diff(a.thinking_state.slots, b.thinking_state.slots) == 0
    assert max_abs_diff(a.logits, b.logits) == 0
    assert abs(float(a.loss - b.loss)) > 1e-5


def test_fixed_budget_compiles_forward_and_backward():
    model = build(config())
    x = mx.array([[1, 2, 3, 4, 5, 6]])
    def loss(m, ids):
        return m(ids, labels=ids, thinking_prefix_lengths=4,
                 thinking_targets=targets(), use_mtp=False).loss
    vg = nn.value_and_grad(model, loss)
    fn = mx.compile(lambda ids: vg(model, ids), inputs=model.state, outputs=model.state)
    eager, _ = nn.value_and_grad(model, loss)(model, x)
    compiled, grads = fn(x)
    mx.eval(compiled, grads)
    assert abs(float(eager - compiled)) < 1e-4


def test_adaptive_policy_requires_calibration_and_can_stop_or_compute():
    model = build(config())
    x = mx.array([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="calibrated"):
        run(model, x, thinking_prefix_lengths=4, thinking_options={"mode": "adaptive"})
    model.eval()
    head = model.model.reasoner.value_head
    head.weight = mx.zeros_like(head.weight)
    head.bias = mx.array([-100.0, 10.0, 20.0])
    a = run(model, x, thinking_prefix_lengths=4,
            thinking_options={"mode": "adaptive", "policy_calibrated": True})
    assert a.thinking_state.rounds == 0 and a.thinking_trace.full_scans == 0
    head.bias = mx.array([100.0, -10.0, 20.0])
    b = run(model, x, thinking_prefix_lengths=4,
            thinking_options={"mode": "adaptive", "policy_calibrated": True})
    assert b.thinking_trace.actions == (COMPUTE, COMPUTE)
    assert b.thinking_trace.full_scans == 0
    assert b.thinking_state.rounds == 2


def test_zero_budget_and_one_token_prefix():
    model = build(config())
    x = mx.array([[1]])
    a = run(model, x, thinking_prefix_lengths=1, thinking_options={"rounds": 0})
    assert a.thinking_trace.states.shape[1] == 1
    assert a.thinking_state.rounds == 0
    b = run(model, x, thinking_prefix_lengths=1)
    assert bool(mx.all(mx.isfinite(b.thinking_state.slots)))
    assert b.thinking_trace.indices[0].tolist() == [[[0], [0]]]


def test_bellman_targets_value_a_zero_immediate_gain_read():
    from types import SimpleNamespace
    class TwoReadTask:
        config = SimpleNamespace(psr_group_size=1, psr_compute_cost=0.1,
                                 psr_read_cost=0.1, psr_cost_weight=1.0)
        def advance(self, slots, memory, cos, sin, action):
            return slots + (1 if action == 2 else 0), None, None
    trace = SimpleNamespace(states=mx.array([[[[0.0]], [[1.0]], [[2.0]]]]), memory=None, budget=2)
    def terminal(slots):
        return mx.where(slots[:, 0, 0] >= 2, 0.0, 0.5)
    target = counterfactual_value_targets(TwoReadTask(), trace, None, None, terminal)
    np.testing.assert_allclose(np.asarray(target[0, 0]), [0.5, 0.5, 0.1], atol=1e-6)
    assert int(mx.argmin(target[0, 0] + mx.array([0, 0.1, 0.1]))) == 2
    assert bool(mx.isnan(target[0, -1, 1:]).all())


def test_synthetic_targets_match_execution_and_bfloat16_backward():
    from trainer.psr_tasks import pointer_batch
    batch = pointer_batch(123, 2, 8, 2, 2, 2)
    ids = np.asarray(batch["input_ids"])
    for row in range(2):
        table = ids[row, :8] - 1
        start = ids[row, -2] - 1
        assert int(batch["answer"][row]) == int(table[table[start]]) + 1
    model = build(config())
    model.set_dtype(mx.bfloat16)
    def loss(m):
        return m(batch["input_ids"], labels=batch["labels"], loss_mask=batch["loss_mask"],
                 thinking_prefix_lengths=batch["prefix_length"], thinking_targets=batch["targets"],
                 use_mtp=False).loss
    value, grad = nn.value_and_grad(model, loss)(model)
    mx.eval(value, grad)
    assert bool(mx.isfinite(value))
    g = dict(tree_flatten(grad))["model.reasoner.index_query.weight"]
    assert bool(mx.isfinite(g).all()) and float(mx.max(mx.abs(g))) > 0
