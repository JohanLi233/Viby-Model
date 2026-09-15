"""Repeated-state mechanism, CED adapter, and legacy execution separation."""

import json
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.iterative_thinking import IterativeEvidenceReader, IterativeThinking
from model.thinking import ThinkingCache
from trainer.config import get_pretrain_parser
from trainer.utils import (
    build_model_kwargs,
    checkpoint_execution,
    validate_checkpoint_execution,
)
from test_thinking import cfg as old_config
from test_ncp_ced import tokens, close


def config(**kw):
    return old_config(**dict(dict(thinking_arch="ced_iterative_v2"), **kw))


def model(**kw):
    mx.random.seed(83)
    return VibyForCausalLM(config(**kw))


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def test_reader_has_closed_state_dependency_and_convex_bound():
    mx.random.seed(1)
    r = IterativeEvidenceReader(16)
    keys = mx.random.normal((2, 7, 16))
    v = mx.random.normal((2, 7, 16))
    q = mx.random.normal((2, 3, 16))
    mask = mx.array([[[True] * 7, [True] * 4 + [False] * 3, [False] * 7]] * 2)
    state, trace = r(q, keys, v, 3, visible=mask, return_trace=True)
    repeated = r(q, keys, v, 3, visible=mask, fixed_query=True)
    assert float(mx.max(mx.abs(state - repeated))) > 1e-4
    assert np.isfinite(np.array(state)).all()
    close(state[:, 2], mx.zeros((2, 16)), atol=0)
    bound = mx.max(mx.sqrt(mx.sum(v * v, axis=-1)), axis=-1)
    assert np.all(
        np.array(mx.sqrt(mx.sum(state * state, axis=-1)))
        <= np.array(bound[:, None]) + 1e-5
    )
    first = r(q, keys, v, 1, visible=mask)
    continued = r(first, keys, v, 2, visible=mask)
    close(state, continued)
    assert trace.shape == (2, 3, 3, 7)
    g = mx.grad(lambda x: r(x, keys, v, 3, visible=mask).sum())(q)
    assert np.isfinite(np.array(g)).all() and float(mx.sum(mx.abs(g[:, :2]))) > 0
    close(g[:, 2], mx.zeros((2, 16)), atol=0)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_adapter_parallel_chunk_and_decode_agree(steps):
    mx.random.seed(3)
    a = IterativeThinking(config(thinking_steps=steps))
    x = mx.random.normal((2, 13, 64))
    full = a(x)
    c = ThinkingCache()
    parts = []
    for start, end in ((0, 3), (3, 4), (4, 9), (9, 13)):
        parts.append(a(x[:, start:end], cache=c, start_pos=start))
    close(full, mx.concatenate(parts, axis=1))
    assert c.memory.shape == (2, 13, 32) and c.previous is None
    assert c.memory.dtype == mx.float32
    assert c.nbytes() == 2 * 13 * 32 * 4
    changed = config(thinking_steps=steps + 1)
    with pytest.raises(ValueError, match="signature"):
        IterativeThinking(changed)(x[:, :1], cache=c, start_pos=13)


def test_packed_and_pad_causality_and_off_control():
    mx.random.seed(4)
    a = IterativeThinking(config())
    x = mx.random.normal((1, 12, 64))
    docs = mx.array([[0] * 4 + [1] * 4 + [0] * 4])
    pad = mx.array([[True] * 10 + [False] * 2])
    full = a(x, docs, pad)
    for start, end in ((0, 4), (4, 8), (8, 10)):
        close(full[:, start:end], a(x[:, start:end]))
    close(full[:, 10:], mx.zeros_like(full[:, 10:]), atol=0)
    perturbed = mx.concatenate([x[:, :6], x[:, 6:] + 100], axis=1)
    close(a(x)[:, :6], a(perturbed)[:, :6])
    close(a(x, intervention="off"), mx.zeros_like(x), atol=0)


def test_cli_legacy_sidecar_and_steps_signature(tmp_path):
    args = get_pretrain_parser().parse_args(["--thinking"])
    c = VibyConfig(**build_model_kwargs(args))
    assert c.thinking_arch == "ced_iterative_tied_v1" and c.thinking_steps == 2
    assert not args.muonh and not VibyConfig().thinking_enabled
    old = old_config().to_dict()
    old.pop("thinking_arch")
    assert VibyConfig.from_dict(old).thinking_arch == "ced_pipeline_v1"
    path = tmp_path / "state.safetensors"
    path.touch()
    path.with_suffix(".json").write_text(json.dumps({"config": config().to_dict()}))
    with pytest.raises(ValueError, match="reset_optimizer"):
        validate_checkpoint_execution(
            path,
            config(thinking_steps=4),
            SimpleNamespace(reset_optimizer=False, resume=str(path)),
        )
    assert checkpoint_execution(config())["thinking_steps"] == 2


def test_full_model_clean_kv_native_cache_and_flops():
    from trainer.flops import training_flops_per_token

    m = model()
    m.eval()
    plain = model(thinking_enabled=False)
    plain.eval()
    plain.load_weights(
        [
            (k, v)
            for k, v in tree_flatten(m.parameters())
            if not k.startswith("model.thinking.")
        ]
    )
    x = tokens(15, 2)
    close(m(x, thinking_intervention="off").logits, plain(x).logits, atol=0)
    full = m(x).logits
    first, c = m.prefill(x[:, :5])
    _, bc = plain.prefill(x[:, :5])
    parts = [first]
    boundary = m.config.n_encoder_layers
    close(c[boundary].compress_kv, bc[boundary].compress_kv, atol=0)
    close(c[boundary].index_k, bc[boundary].index_k, atol=0)
    for i in range(5, 15):
        y, c = m.decode_step(x[:, i], c)
        parts.append(y[:, None])
    close(full, mx.concatenate(parts, axis=1))
    w, d, t = m.config.thinking_dim, m.config.dim, 32
    expected = 6 * (3 * d * w + 3 * w * w) + 24 * w * t
    assert (
        training_flops_per_token(m, t) - training_flops_per_token(plain, t) == expected
    )


def test_trained_probe_uses_the_same_reader_math():
    # Deterministic equivalence at arbitrary weights; training files not required.
    from experiments.transition_state_probe import TransitionReader, worlds, batch, unit

    mx.random.seed(9)
    probe = TransitionReader(addressing="tied")
    r = IterativeEvidenceReader(64)
    r.address.weight = probe.query.weight
    data = worlds(19, 8)
    k, v, q, _ = batch(data, 0, 8, 2)
    facts = mx.concatenate([probe.entities(k), probe.entities(v)], axis=-1)
    for steps in (2, 4, 8):
        state, trace = r(
            probe.entities(q)[:, None],
            probe.entities(k),
            probe.value(facts),
            steps,
            return_trace=True,
        )
        logits = 8 * unit(state[:, 0]) @ unit(probe.entities.weight).T
        old_logits, old_trace = probe(k, v, q, steps)
        close(logits, old_logits)
        close(trace[:, :, 0], old_trace)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("arch", ["ced_iterative_v2", "ced_iterative_tied_v1"])
def test_metal_component_compiled_gradients_and_cache(dtype, arch):
    mx.set_default_device(mx.gpu)
    mx.random.seed(72)
    c = config(thinking_steps=4, thinking_arch=arch)
    a = IterativeThinking(c)
    a.set_dtype(dtype)
    x = mx.random.normal((2, 13, 64)).astype(dtype)
    weight = mx.random.normal((2, 13, 64))

    def objective(params):
        a.update(params)
        return mx.mean(a(x).astype(mx.float32) * weight)

    saved = a.parameters()
    direct = objective(saved)
    value, grad = mx.value_and_grad(objective)(saved)
    mx.eval(direct, value, grad)
    a.update(saved)
    assert abs(float(value - direct)) <= 1e-5
    cv, cg = mx.compile(mx.value_and_grad(objective))(saved)
    mx.eval(cv, cg)
    a.update(saved)
    assert abs(float(cv - value)) <= 1e-4
    assert all(
        np.isfinite(np.array(g.astype(mx.float32))).all() for _, g in tree_flatten(cg)
    )
    full = a(x)
    cache = ThinkingCache()
    parts = []
    for i in range(13):
        parts.append(a(x[:, i : i + 1], cache=cache, start_pos=i))
    close(
        full,
        mx.concatenate(parts, axis=1),
        atol=0.002 if dtype == mx.bfloat16 else 2e-5,
    )


@pytest.mark.parametrize("reject_at", [None, 1])
def test_engine_reuses_and_restores_iterative_prefix(reject_at):
    from engine import VibyEngine
    from test_engine_speculative import controlled_proposals, params

    m = model(n_mtp_layers=1, dspark_n_routed_experts=4, dspark_n_activated_experts=2)
    m.eval()
    prompt = [1, 2, 3, 4, 5]
    expected = VibyEngine(m).generate([prompt], params(max_new_tokens=6))[0].outputs[0]
    engine = VibyEngine(m, prefix_stride=3)
    controlled_proposals(engine, reject_at)
    actual = engine.generate([prompt], params(True, max_new_tokens=6))[0].outputs[0]
    assert actual.token_ids == expected.token_ids
    np.testing.assert_allclose(actual.logprobs, expected.logprobs, atol=3e-5)
    again = engine.generate([prompt], params(True, max_new_tokens=6))[0].outputs[0]
    assert again.token_ids == expected.token_ids and engine.stats["prefix_hits"] > 0


def test_legacy_weight_migration_is_explicit_and_keeps_common_backbone(tmp_path):
    from trainer.utils import (
        load_model_weights,
        ncp_warm_start_prefixes,
        ncp_warm_start_drop_prefixes,
    )

    old = VibyForCausalLM(old_config())
    new = model()
    path = tmp_path / "old.safetensors"
    old.save_weights(str(path))
    path.with_suffix(".json").write_text(json.dumps({"config": old.config.to_dict()}))
    args = SimpleNamespace(reset_optimizer=True, resume=str(path))
    converted = validate_checkpoint_execution(path, new.config, args)
    load_model_weights(
        new,
        str(path),
        allow_fresh_prefixes=ncp_warm_start_prefixes(path, new.config, converted),
        allow_drop_prefixes=ncp_warm_start_drop_prefixes(path, new.config, converted),
    )
    common = dict(tree_flatten(old.parameters()))
    for name, value in tree_flatten(new.parameters()):
        if not name.startswith("model.thinking."):
            close(value, common[name], atol=0)


def test_adjoint_readout_ties_the_actual_weight_and_counts_reuse():
    from trainer.flops import training_flops_per_token

    c = config(thinking_arch="ced_iterative_tied_v1")
    a = IterativeThinking(c)
    assert a.output is None
    assert "output.weight" not in dict(tree_flatten(a.parameters()))
    m = model(thinking_arch="ced_iterative_tied_v1")
    plain = model(thinking_enabled=False)
    w, d, t = c.thinking_dim, c.dim, 32
    extra = sum(
        v.size
        for k, v in tree_flatten(m.parameters())
        if k.startswith("model.thinking.")
    )
    assert extra == 2 * d * w + w * w
    assert (
        training_flops_per_token(m, t) - training_flops_per_token(plain, t)
        == 6 * (3 * d * w + 3 * w * w) + 24 * w * t
    )
    m.eval()
    x = tokens(13)
    full = m(x).logits
    first, cache = m.prefill(x[:, :4])
    parts = [first]
    for i in range(4, 13):
        y, cache = m.decode_step(x[:, i], cache)
        parts.append(y[:, None])
    close(full, mx.concatenate(parts, axis=1))
