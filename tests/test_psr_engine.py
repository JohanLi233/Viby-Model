"""PSR engine parity with native prefill/decode, using a nonzero correction head."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from _v41_common import build, max_abs_diff
from engine import SamplingParams, VibyEngine
from engine.memory import StatePool, capture_state, copy_state_row, restore_state
from engine.sampling import log_softmax
from model.cache import VibyCache
from model.config import VibyConfig


def make_model(**kw):
    settings = dict(
        preset="tiny",
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
        engram_layer_ids=(),
        n_mtp_layers=0,
        vocab_size=32,
        max_seq_len=48,
        window_size=8,
        psr_enabled=True,
        psr_dim=32,
        psr_slots=2,
        psr_rounds=2,
        psr_horizon=4,
    )
    settings.update(kw)
    model = build(VibyConfig(**settings))
    model.eval()
    # A zero head would let an engine silently bypass the entire workspace.
    model.psr.output.weight = mx.random.normal(model.psr.output.weight.shape) * 0.2
    model.psr.calibration_gate = mx.array(0.7)
    mx.eval(model.parameters())
    return model


def params(**kw):
    values = dict(max_new_tokens=10, do_sample=False, eos_token_id=None, logprobs=True)
    values.update(kw)
    return SamplingParams(**values)


def reference(model, prompt, steps):
    logits, cache = model.prefill(mx.array([prompt], mx.int32))
    row = logits[0, -1]
    ids, lps = [], []
    for i in range(steps):
        token = int(mx.argmax(row).item())
        ids.append(token)
        lps.append(float(log_softmax(row)[token].item()))
        if i + 1 < steps:
            logits, cache = model.decode_step(mx.array([token], mx.int32), cache)
            row = logits[0]
    return ids, lps, cache


def assert_output(output, expected):
    assert output.token_ids == expected[0]
    np.testing.assert_allclose(output.logprobs, expected[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("horizon", [1, 4])
def test_greedy_and_logprobs_match_native_across_refreshes(horizon):
    model = make_model(psr_horizon=horizon)
    prompt = [1, 5, 9, 3, 7]
    x = mx.array([prompt], mx.int32)
    corrected, _ = model.prefill(x)
    baseline, _ = model.prefill(x, use_thinking=False)
    assert max_abs_diff(corrected[:, -1], baseline[:, -1]) > 1e-3
    expected = reference(model, prompt, 10)
    assert expected[2].psr_phases >= 3
    engine = VibyEngine(model, max_num_seqs=1)
    assert engine.prefix is None
    output = engine.generate([prompt], params())[0]
    assert output.finished
    assert_output(output.outputs[0], expected)
    assert output.outputs[0].finish_reason == "length"
    assert engine.pool.cache is None


def test_batched_logits_and_row_clocks_match_independent_decode():
    model = make_model()
    prompts = [[1, 5, 9], [2, 4, 6, 8, 10, 12]]
    engine = VibyEngine(model, max_num_seqs=2)
    references = []
    for prompt in prompts:
        engine.add_request(prompt, params())
        seq = engine._waiting.pop(0)
        engine._admit(seq)
        _, cache = model.prefill(mx.array([prompt], mx.int32))
        references.append(cache)
    for _ in range(8):
        pending = [seq.token_ids[-1] for seq in engine._running]
        logits = engine._decode(engine._running)
        for row, cache in enumerate(references):
            expected, _ = model.decode_step(mx.array([pending[row]], mx.int32), cache)
            assert max_abs_diff(logits[row], expected[0]) < 2e-5
            assert engine.pool.cache.psr_phases[row].item() == cache.psr_phases
            assert (
                engine.pool.cache.psr_next_anchor[row].item() == cache.psr_next_anchor
            )
            assert (
                max_abs_diff(
                    engine.pool.cache.thinking_state.slots[row],
                    cache.thinking_state.slots[0],
                )
                < 2e-5
            )
        engine._append_step(engine._running, logits)
    assert engine.stats["max_batch"] == 2


def test_late_admission_compaction_and_independent_refresh_phases():
    model = make_model()
    engine = VibyEngine(model, max_num_seqs=2)
    prompts = [[1, 3, 5], [2, 4, 6, 8, 10], [7, 9, 11, 13]]
    limits = [4, 12, 8]
    outputs = [
        engine.add_request(prompts[i], params(max_new_tokens=limits[i]))
        for i in range(2)
    ]
    engine.step()
    # The first two rows already have an older workspace when the third arrives.
    outputs.append(engine.add_request(prompts[2], params(max_new_tokens=limits[2])))
    mixed_phases = False
    while engine.has_unfinished():
        engine.step()
        if len(engine._running) == 2:
            phases = engine.pool.cache.psr_phases.tolist()
            mixed_phases |= phases[0] != phases[1]
    assert mixed_phases
    for prompt, limit, out in zip(prompts, limits, outputs):
        assert out.finished
        assert_output(out.outputs[0], reference(model, prompt, limit))
    assert engine.stats["max_batch"] == 2


@pytest.mark.parametrize("bf16", [False, True])
def test_engram_short_prompts_and_bf16(bf16):
    model = make_model(engram_layer_ids=(1,), engram_vocab_size=128)
    if bf16:
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
        mx.eval(model.parameters())
    prompts = [[1], [2, 4, 6, 8]]
    expected = [reference(model, p, 10) for p in prompts]
    outputs = VibyEngine(model, max_num_seqs=2).generate(prompts, params())
    for out, ref in zip(outputs, expected):
        assert out.outputs[0].token_ids == ref[0]
        np.testing.assert_allclose(
            out.outputs[0].logprobs, ref[1], atol=2e-2 if bf16 else 2e-5, rtol=0
        )


def test_copy_resize_move_and_snapshot_guards():
    model = make_model()
    _, first = model.prefill(mx.array([[1, 2, 3]], mx.int32))
    _, second = model.prefill(mx.array([[5, 6, 7, 8, 9]], mx.int32))
    pool = StatePool(model.config, 1)
    copy_state_row(pool.cache, 0, first, 0, first.start_pos)
    pool.ensure(2, [first.start_pos])
    copy_state_row(pool.cache, 1, second, 0, second.start_pos)
    assert pool.cache.thinking_state.anchor.tolist() == [[2], [4]]
    assert pool.cache.psr_next_anchor.tolist() == [6, 8]
    saved = mx.array(first.thinking_state.slots)
    pool.move(0, 1)
    pool.ensure(1, [second.start_pos])
    assert (
        max_abs_diff(pool.cache.thinking_state.slots, second.thinking_state.slots) == 0
    )
    assert pool.cache.psr_next_anchor.tolist() == [8]
    assert max_abs_diff(first.thinking_state.slots, saved) == 0
    assert pool.memory_bytes() > 0
    with pytest.raises(ValueError, match="PSR"):
        capture_state(pool.cache, 0, second.start_pos)
    baseline = VibyCache(model.config, 1)
    snapshot = capture_state(baseline, 0, 0)
    with pytest.raises(ValueError, match="PSR"):
        restore_state(pool.cache, 0, snapshot)


def test_benchmark_snapshot_replays_workspace_and_refresh_clock():
    from experiments.bench_csa2_inference import snapshot, restore, eval_cache

    model = make_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], mx.int32)
    _, cache = model.prefill(tokens[:, :3])
    saved = snapshot(cache)

    def decode():
        outputs = []
        for pos in range(3, tokens.shape[1]):
            logits, _ = model.decode_step(tokens[:, pos], cache)
            outputs.append(logits)
        eval_cache(cache)
        return mx.stack(outputs, axis=1)

    first = decode()
    assert cache.psr_phases == 3
    restore(cache, saved)
    assert cache.psr_phases == 1 and cache.start_pos == 3
    assert cache.psr_next_anchor == 6
    assert cache.thinking_state.anchor.tolist() == [[2]]
    second = decode()
    assert max_abs_diff(first, second) == 0
    assert cache.psr_phases == 3


def test_scoring_anchors_at_prompt_and_matches_generation_logprobs():
    model = make_model()
    prompts = [[1, 3, 7], [2, 4, 8, 10, 12]]
    expected = [reference(model, prompt, n) for prompt, n in zip(prompts, [10, 6])]
    sequences = [p + r[0] for p, r in zip(prompts, expected)]
    sequences.append([1, 2])  # No completion: the entire output row is padding.
    scores = VibyEngine(model).score(sequences, [3, 5, 2])
    np.testing.assert_allclose(scores[0].tolist(), expected[0][1], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        scores[1, :6].tolist(), expected[1][1], atol=2e-5, rtol=2e-5
    )
    assert bool(mx.all(scores[1, 6:] == 0))
    assert bool(mx.all(scores[2] == 0))


def test_stop_stream_reset_zero_tokens_and_multiple_outputs():
    model = make_model()
    prompt = [1, 5, 9]
    engine = VibyEngine(model)
    chunks = []

    class Stream:
        def put(self, tokens):
            chunks.extend(tokens[0])

        def end(self):
            chunks.append("end")

    expected = reference(model, prompt, 1)
    stopped = engine.generate([prompt], params(eos_token_id=expected[0][0]), Stream())[
        0
    ]
    assert stopped.outputs[0].finish_reason == "stop"
    assert chunks == prompt + expected[0] + ["end"]
    empty = engine.generate([prompt], params(max_new_tokens=0))[0]
    assert empty.finished and empty.outputs[0].token_ids == []
    out = engine.generate([prompt], params(max_new_tokens=5, n=2))[0]
    expected = reference(model, prompt, 5)
    for i, output in enumerate(out.outputs):
        assert output.index == i
        assert_output(output, expected)
    with pytest.raises(ValueError, match="PSR"):
        engine.add_request(prompt, params(use_mtp_speculative=True))
