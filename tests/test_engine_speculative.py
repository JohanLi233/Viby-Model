"""DSpark engine acceptance: cache rollback, streaming and exact sampling law."""
from types import MethodType

import mlx.core as mx
import numpy as np
import pytest

from _v41_common import cfg_mix
from engine import VibyEngine, SamplingParams
from engine.sampling import speculative_correction, transform_logits
from model.model import VibyForCausalLM


@pytest.fixture
def model():
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    mx.random.seed(41)
    cfg = cfg_mix(dim=64, n_heads=4, head_dim=32, rope_head_dim=16,
                  q_lora_rank=32, o_lora_rank=16, o_groups=4, window_size=4,
                  n_routed_experts=4, n_activated_experts=2, moe_inter_dim=16,
                  index_n_heads=4, index_head_dim=32, index_topk=3,
                  dspark_n_routed_experts=4, dspark_n_activated_experts=2,
                  vocab_size=32, max_seq_len=48, dspark_block_size=4)
    model = VibyForCausalLM(cfg, skip_init=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def params(spec=False, **kw):
    values = dict(max_new_tokens=12, do_sample=False, eos_token_id=None,
                  use_mtp_speculative=spec, num_speculative_tokens=3, logprobs=True)
    values.update(kw)
    return SamplingParams(**values)


def controlled_proposals(engine, reject_at=None):
    """A deterministic oracle draft isolates verifier/rollback from draft quality."""
    def propose(self, seq, count):
        scratch = self._scratch_for(seq)
        history = list(seq.token_ids)
        result = []
        pending = history[-1]
        for i in range(count):
            row, _ = self._verify_token(scratch, pending, len(seq.token_ids) - 1 + i)
            token = int(mx.argmax(transform_logits(row, history, seq.params)).item())
            if i == reject_at:
                token = (token + 1) % self.config.vocab_size
            q = (mx.arange(self.config.vocab_size) == token).astype(mx.float32)
            result.append((token, q))
            history.append(token)
            pending = token
        return result
    engine._draft_tokens = MethodType(propose, engine)


@pytest.mark.parametrize("reject_at", [None, 0, 1, 2])
def test_greedy_matches_normal_across_acceptance_and_rollback(model, reject_at):
    prompt = [1, 5, 9, 3, 7]
    ordinary = VibyEngine(model, enable_prefix_cache=False).generate([prompt], params())[0].outputs[0]
    engine = VibyEngine(model, enable_prefix_cache=False)
    controlled_proposals(engine, reject_at)
    output = engine.generate([prompt], params(True))[0].outputs[0]
    assert output.token_ids == ordinary.token_ids
    np.testing.assert_allclose(output.logprobs, ordinary.logprobs, atol=2e-5, rtol=2e-5)
    assert engine.stats["mtp_drafted"] > 0
    assert engine.stats["mtp_accepted"] > 0 if reject_at != 0 else engine.stats["mtp_accepted"] == 0
    if reject_at is not None:
        assert engine.stats["mtp_rejected"] > 0


def test_real_dspark_prefix_forward_and_generation(model):
    engine = VibyEngine(model, enable_prefix_cache=False)
    prompt = [1, 4, 6, 8, 3]
    ordinary = engine.generate([prompt], params())[0].outputs[0].token_ids
    speculative = engine.generate([prompt], params(True))[0].outputs[0].token_ids
    assert speculative == ordinary
    assert engine.stats["mtp_rounds"] > 0
    assert engine.stats["mtp_drafted"] > 0


def test_dspark_prefix_matches_full_causal_scaffold(model):
    mains = [mx.random.normal((1, 1, model.config.dim)) for _ in model.config.dspark_target_layer_ids]
    prefix = mx.array([[1, 4]], mx.int32)
    full = mx.array([[1, 4, model.config.dspark_noise_token_id, model.config.dspark_noise_token_id]], mx.int32)
    lp, cp = model.dspark_draft(mains, prefix)
    lf, cf = model.dspark_draft(mains, full)
    mx.eval(lp, cp, lf, cf)
    np.testing.assert_allclose(np.asarray(lp), np.asarray(lf[:, :2]), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(cp), np.asarray(cf[:, :2]), rtol=2e-5, atol=2e-5)


def test_mixed_requests_compaction_prefix_cache_and_limits(model):
    engine = VibyEngine(model, max_num_seqs=2, prefix_stride=3)
    prompts = [[1, 4, 7, 2, 8], [1, 4, 7], [5, 9, 6, 8, 1, 3]]
    expected = [VibyEngine(model, enable_prefix_cache=False).generate([p], params(max_new_tokens=9))[0].outputs[0].token_ids
                for p in prompts]
    reqs = [engine.add_request(p, params(i != 1, max_new_tokens=9), request_id=str(i)) for i, p in enumerate(prompts)]
    while engine.has_unfinished():
        engine.step()
    assert [r.outputs[0].token_ids for r in reqs] == expected
    assert all(r.finished for r in reqs)
    # Full prefix hit must carry the DSpark target-layer anchor as well as KV.
    again = engine.generate([prompts[0]], params(True, max_new_tokens=9))[0]
    assert again.outputs[0].token_ids == expected[0]
    assert engine.stats["prefix_hits"] == 1
    assert engine.stats["mtp_drafted"] > 0


def test_finished_admission_eos_stream_and_confidence(model):
    class Stream:
        def __init__(self):
            self.tokens = []
        def put(self, tokens):
            self.tokens.extend(tokens[0])
        def end(self):
            pass
    prompt = [1, 3, 5]
    engine = VibyEngine(model, enable_prefix_cache=False)
    baseline = engine.generate([prompt], params())[0].outputs[0].token_ids
    stream = Stream()
    controlled_proposals(engine, 0)
    output = engine.generate([prompt], params(True, max_new_tokens=4), streamer=stream)[0].outputs[0]
    assert stream.tokens == prompt + output.token_ids
    assert output.token_ids == baseline[:4]
    output = engine.generate([prompt], params(True, eos_token_id=baseline[0]))[0].outputs[0]
    assert output.token_ids == baseline[:1]
    assert engine.stats["decode_tokens"] == 0
    assert output.finish_reason == "stop"
    output = engine.generate([prompt], params(True, max_new_tokens=0))[0].outputs[0]
    assert output.token_ids == []


def test_confidence_gate_falls_back_to_target(model):
    engine = VibyEngine(model, enable_prefix_cache=False)
    prompt = [1, 3, 7, 9]
    expected = engine.generate([prompt], params())[0].outputs[0].token_ids
    got = engine.generate([prompt], params(True, mtp_confidence_threshold=1.0))[0].outputs[0].token_ids
    assert got == expected
    assert engine.stats["mtp_confidence_stops"] > 0
    assert engine.stats["mtp_drafted"] == 0


def test_bfloat16_real_draft_and_old_prefix_upgrade(model):
    from trainer.utils import convert_model_dtype
    convert_model_dtype(model, "bfloat16")
    engine = VibyEngine(model, prefix_stride=3)
    prompt = [1, 7, 4, 6, 9]
    expected = engine.generate([prompt], params(max_new_tokens=7))[0].outputs[0].token_ids
    got = engine.generate([prompt], params(True, max_new_tokens=7))[0].outputs[0].token_ids
    assert got == expected
    # The initial ordinary full-prefix snapshot had no DSpark features.
    assert engine.stats["mtp_drafted"] > 0
    again = engine.generate([prompt], params(True, max_new_tokens=7))[0].outputs[0].token_ids
    assert again == expected
    assert engine.stats["prefix_hits"] == 1


def test_stop_strings_logprobs_n_and_context_limit(model):
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(chr(65 + t) for t in ids)
    tokenizer = Tokenizer()
    prompt = [1, 8, 3, 5]
    engine = VibyEngine(model, tokenizer, enable_prefix_cache=False)
    expected = engine.generate([prompt], params())[0].outputs[0].token_ids
    stop = tokenizer.decode(expected[:3])
    baseline = engine.generate([prompt], params(stop=[stop]))[0].outputs[0]
    controlled_proposals(engine)
    out = engine.generate([prompt], params(True, stop=[stop], n=2))[0]
    assert len(out.outputs) == 2
    for item in out.outputs:
        assert item.finish_reason == "stop"
        assert item.token_ids == baseline.token_ids
        assert len(item.logprobs) == len(item.token_ids)
    limited = VibyEngine(model, max_model_len=len(prompt) + 3, enable_prefix_cache=False)
    controlled_proposals(limited)
    output = limited.generate([prompt], params(True, max_new_tokens=100, num_speculative_tokens=100))[0].outputs[0]
    assert output.token_ids == expected[:3]
    assert output.finish_reason == "length"


def test_transformed_stochastic_distribution_and_target_logprobs(model):
    engine = VibyEngine(model, enable_prefix_cache=False, seed=19)
    p = params(True, do_sample=True, temperature=0.7, top_k=4, top_p=0.9,
               repetition_penalty=1.2, max_new_tokens=6)
    output = engine.generate([[1, 4, 8, 3, 9]], p)[0].outputs[0]
    assert len(output.token_ids) == len(output.logprobs) == 6
    assert np.isfinite(output.logprobs).all()
    assert engine.stats["mtp_drafted"] > 0


def test_sampling_rejection_uses_positive_residual():
    mx.random.seed(9)
    p = mx.array([0.2, 0.8], mx.float32)
    q = mx.array([0.75, 0.25], mx.float32)
    sampling = SamplingParams(do_sample=True, temperature=1, top_k=0, top_p=1)
    got, accepted = speculative_correction(mx.log(p), q, 0, sampling, uniform=0.99)
    assert (got, accepted) == (1, False)
    got, accepted = speculative_correction(mx.log(p), q, 1, sampling, uniform=0.99)
    assert (got, accepted) == (1, True)
    # The accepted mass plus normalized residual is exactly the target law.
    accepted_mass = mx.minimum(p, q)
    residual = mx.maximum(p - q, 0)
    recovered = accepted_mass + (1 - mx.sum(accepted_mass)) * residual / mx.sum(residual)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(p), atol=1e-7)
    count = 0
    for _ in range(600):
        draft = int(mx.random.categorical(mx.log(q)).item())
        token, _ = speculative_correction(mx.log(p), q, draft, sampling)
        count += token == 0
    assert abs(count / 600 - 0.2) < 0.07


def test_missing_draft_modules_and_invalid_params_fail_explicitly(model):
    engine = VibyEngine(model)
    with pytest.raises(ValueError, match="positive"):
        engine.add_request([1], params(True, num_speculative_tokens=0))
    model.mtp_modules = []
    with pytest.raises(ValueError, match="checkpoint"):
        engine.add_request([1], params(True))


def test_rejection_restores_engram_history_row(model):
    engine = VibyEngine(model, enable_prefix_cache=False)
    # Exercise the engine's real Engram-history bookkeeping independently of
    # the tokenizer-dependent hash module; target logits need no hash here.
    engine._w = 2
    controlled_proposals(engine, 1)
    engine.add_request([1, 4, 8, 3, 9], params(True, max_new_tokens=11))
    while engine.has_unfinished():
        engine.step()
        for seq in engine._running:
            got = np.asarray(engine.pool.cache.engram_prev[seq.row]).tolist()
            assert got == seq.token_ids[:-1][-2:]
