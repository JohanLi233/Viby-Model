"""Latent pipeline: causal execution, gradients, packed resets and cache transport."""

import json
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.cache import VibyCache
from model.thinking import LatentThinking, ThinkingCache
from trainer.config import get_pretrain_parser
from trainer.utils import (
    build_model_kwargs,
    checkpoint_execution,
    validate_checkpoint_execution,
    ncp_warm_start_prefixes,
    ncp_warm_start_drop_prefixes,
)
from engine.memory import capture_state, restore_state, copy_state_row, state_bytes
from test_ncp_ced import config, tokens, close


def cfg(**kw):
    return config(
        **dict(
            dict(
                ncp_enabled=False,
                thinking_enabled=True,
                thinking_dim=16,
                thinking_arch="ced_pipeline_v1",
            ),
            **kw,
        )
    )


def model(**kw):
    mx.random.seed(72)
    return VibyForCausalLM(cfg(**kw))


@pytest.fixture(autouse=True)
def device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def test_default_cli_sidecar_and_explicit_legacy_control():
    c = VibyConfig()
    assert not c.thinking_enabled and not c.ncp_enabled
    for opts, enabled in (
        ([], False),
        (["--preset", "tiny"], False),
        (["--no-thinking"], False),
        (["--thinking"], True),
    ):
        c = VibyConfig(**build_model_kwargs(get_pretrain_parser().parse_args(opts)))
        assert c.thinking_enabled == enabled and not c.ncp_enabled
        assert VibyConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
    assert not VibyConfig.from_dict({"preset": "tiny"}).thinking_enabled
    assert not VibyConfig(ncp_enabled=False).thinking_enabled
    assert not VibyConfig(ncp_enabled=True).thinking_enabled
    assert checkpoint_execution(cfg())["kind"] == "ced_pipeline_v1"
    for changes in (
        dict(thinking_dim=3),
        dict(thinking_scale=float("nan")),
        dict(thinking_arch="bad"),
        dict(ncp_enabled=True),
        dict(ced_recurrent_enabled=True),
    ):
        with pytest.raises(ValueError):
            cfg(**changes)


def test_off_matches_ced_and_signal_cannot_rewrite_global_kv():
    m = model()
    m.eval()
    base = model(thinking_enabled=False)
    base.eval()
    base.load_weights(
        [
            (k, v)
            for k, v in tree_flatten(m.parameters())
            if not k.startswith("model.thinking.")
        ]
    )
    x = tokens(19)
    close(m(x, thinking_intervention="off").logits, base(x).logits, atol=0)
    y, cache = m.prefill(x)
    z, control = base.prefill(x)
    b = m.config.n_encoder_layers
    close(cache[b].compress_kv, control[b].compress_kv, atol=0)
    close(cache[b].index_k, control[b].index_k, atol=0)
    assert np.max(np.abs(np.array(y - z))) > 1e-5
    assert m(x, labels=x).ncp_loss is None


@pytest.mark.parametrize("prefix", [1, 3, 8])
def test_native_decode_and_chunked_prefill(prefix):
    m = model()
    m.eval()
    x = tokens(21, 2)
    expected = m(x).logits
    first, cache = m.prefill(x[:, :prefix])
    parts = [first]
    for i in range(prefix, 21):
        y, cache = m.decode_step(x[:, i], cache)
        parts.append(y[:, None])
    close(expected, mx.concatenate(parts, axis=1))
    first, chunks = m.prefill(x[:, :prefix])
    parts = [first]
    for start in range(prefix, 21, 5):
        parts.append(m(x[:, start : start + 5], start_pos=start, cache=chunks).logits)
    close(expected, mx.concatenate(parts, axis=1))
    close(cache.thinking_state.memory, chunks.thinking_state.memory)
    close(cache.thinking_state.previous, chunks.thinking_state.previous)
    assert cache.thinking_state.tokens == 21
    with pytest.raises(ValueError, match="rewind"):
        cache.rewind(1)
    with pytest.raises(ValueError, match="clock"):
        m(x[:, :1], start_pos=0, cache=cache)


def test_prefix_row_transport_snapshot_and_per_row_clocks():
    m = model()
    m.eval()
    a, b = tokens(9), tokens(13) + 1
    _, ca = m.prefill(a)
    _, cb = m.prefill(b)
    snap = capture_state(ca, 0, n=9)
    assert state_bytes(snap) >= snap.thinking.nbytes() > 0
    previous = np.array(snap.thinking.previous)
    m.decode_step(mx.array([2]), ca)
    np.testing.assert_array_equal(previous, np.array(snap.thinking.previous))
    restored = VibyCache(m.config)
    restore_state(restored, 0, snap)
    pool = VibyCache(m.config, 2)
    copy_state_row(pool, 0, restored)
    copy_state_row(pool, 1, cb)
    nxt = mx.array([[5], [6]])
    got = m(nxt, cache=pool, start_pos=mx.array([9, 13]), decode=True).logits
    expected_a = m(nxt[:1], cache=restored, start_pos=9, decode=True).logits
    expected_b = m(nxt[1:], cache=cb, start_pos=13, decode=True).logits
    close(got, mx.concatenate([expected_a, expected_b]))
    assert [s.tokens for s in pool.thinking_rows] == [10, 14]
    incompatible = VibyCache(cfg(thinking_scale=0.2))
    with pytest.raises(ValueError, match="execution"):
        restore_state(incompatible, 0, snap)


def test_packed_repeated_documents_pad_and_future_isolation():
    mx.random.seed(18)
    block = LatentThinking(cfg())
    x = mx.random.normal((1, 14, 64))
    docs = mx.array([[0] * 4 + [1] * 5 + [0] * 5])
    pad = mx.array([[True] * 12 + [False] * 2])
    y = block(x, docs, pad)
    for start, end in ((0, 4), (4, 9), (9, 12)):
        close(y[:, start:end], block(x[:, start:end]))
    close(y[:, 12:], mx.zeros_like(y[:, 12:]), atol=0)
    changed = mx.concatenate([x[:, :5], x[:, 5:] + 100], axis=1)
    close(block(x)[:, :5], block(changed)[:, :5])
    masked = mx.zeros((1, 14), mx.bool_)
    grads = mx.grad(lambda z: block(z, pad_mask=masked).sum())(x)
    close(grads, mx.zeros_like(grads), atol=0)


def test_bounded_signal_gradient_finite_difference_and_learning_path():
    mx.random.seed(7)
    c = cfg()
    block = LatentThinking(c)
    x = mx.random.normal((2, 9, c.dim))
    target = mx.random.normal(x.shape)

    def objective(z):
        return mx.mean(block(z) * target)

    value, grad = mx.value_and_grad(objective)(x)
    assert abs(float(value) - float(objective(x))) < 1e-6
    direction = mx.random.normal(x.shape)
    direction /= mx.sqrt(mx.sum(direction**2))
    fd = (objective(x + 0.01 * direction) - objective(x - 0.01 * direction)) / 0.02
    assert abs(float(fd - mx.sum(grad * direction))) < 1e-4
    _, pg = nn.value_and_grad(block, lambda m: mx.mean(m(x) * target))(block)
    for k, g in tree_flatten(pg):
        assert np.isfinite(np.array(g)).all() and float(mx.sum(g**2)) > 0, k
    signal = block(x)
    bound = c.thinking_scale * mx.sqrt(mx.mean(x**2, axis=-1) + c.norm_eps)
    assert np.all(
        np.array(mx.sqrt(mx.mean(signal**2, axis=-1))) <= np.array(bound) + 1e-6
    )
    assert float(mx.sum(mx.abs(block(x) - block(x, intervention="reset")))) > 1e-4
    assert float(mx.sum(mx.abs(block(x) - block(x, intervention="swap")))) > 1e-4


def test_explicit_ncp_migration_keeps_backbone_and_requires_reset(tmp_path):
    path = tmp_path / "old.safetensors"
    path.touch()
    path.with_suffix(".json").write_text(
        json.dumps({"config": config(ncp_enabled=True).to_dict()})
    )
    args = SimpleNamespace(reset_optimizer=False, resume=str(path))
    with pytest.raises(ValueError, match="reset_optimizer"):
        validate_checkpoint_execution(path, cfg(), args)
    args.reset_optimizer = True
    assert validate_checkpoint_execution(path, cfg(), args)
    assert ncp_warm_start_prefixes(path, cfg(), True) == ("model.thinking.",)
    assert ncp_warm_start_drop_prefixes(path, cfg(), True) == ("model.ncp.",)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_metal_eager_compiled_primal_gradients_and_decode(dtype):
    mx.set_default_device(mx.gpu)
    mx.random.seed(90)
    m = model()
    if dtype == mx.bfloat16:
        from trainer.utils import convert_model_dtype

        convert_model_dtype(m, "bfloat16")
    x = tokens(17, 2)
    labels = (x + 1) % m.config.vocab_size

    def loss_fn(net):
        return net(x, labels=labels, use_mtp=False).loss

    value_grad = nn.value_and_grad(m, loss_fn)
    direct = loss_fn(m)
    value, grads = value_grad(m)
    mx.eval(direct, value, grads)
    assert abs(float(direct) - float(value)) < 1e-5
    assert all(
        np.isfinite(np.array(g.astype(mx.float32))).all()
        for _, g in tree_flatten(grads)
    )

    def explicit(params):
        m.update(params)
        return value_grad(m)

    compiled = mx.compile(explicit)
    saved = m.parameters()
    cv, cg = compiled(m.trainable_parameters())
    mx.eval(cv, cg)
    m.update(saved)
    assert abs(float(cv) - float(value)) < (0.02 if dtype == mx.bfloat16 else 1e-4)
    assert all(
        np.isfinite(np.array(g.astype(mx.float32))).all() for _, g in tree_flatten(cg)
    )
    m.eval()
    expected = m(x).logits
    first, cache = m.prefill(x[:, :7])
    parts = [first]
    for i in range(7, 17):
        y, cache = m.decode_step(x[:, i], cache)
        parts.append(y[:, None])
    actual = mx.concatenate(parts, axis=1)
    if dtype == mx.float32:
        close(expected, actual)
    else:
        # The existing BF16 backbone has shape/route-sensitive prefill-decode
        # differences. Bound against the same-weight CED control, not exact parity.
        base = model(thinking_enabled=False)
        base.load_weights(
            [
                (k, v)
                for k, v in tree_flatten(m.parameters())
                if not k.startswith("model.thinking.")
            ]
        )
        base.eval()
        baseline_full = base(x).logits
        y, bc = base.prefill(x[:, :7])
        bp = [y]
        for i in range(7, 17):
            y, bc = base.decode_step(x[:, i], bc)
            bp.append(y[:, None])

        def err(a, b):
            return float(mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))

        assert (
            err(expected, actual)
            <= 1.25 * err(baseline_full, mx.concatenate(bp, axis=1)) + 0.001
        )
        # On fixed encoder states the pipeline itself must agree tightly.
        e = mx.random.normal((2, 17, m.config.dim)).astype(dtype)
        tc = ThinkingCache()
        seq = []
        for i in range(17):
            seq.append(m.model.thinking(e[:, i : i + 1], cache=tc, start_pos=i))
        close(m.model.thinking(e), mx.concatenate(seq, axis=1), atol=0.002)


def test_thinking_flops_counts_shared_block_twice():
    from trainer.flops import training_flops_per_token

    m = model()
    base = model(thinking_enabled=False)
    w, d, t = m.config.thinking_dim, m.config.dim, 32
    # input/output once, shared query/output/three FFN matrices twice; QK/AV both rounds.
    expected = 6 * (2 * d * w + 16 * w * w) + 24 * w * t
    assert (
        training_flops_per_token(m, t) - training_flops_per_token(base, t) == expected
    )


@pytest.mark.parametrize("reject_at", [None, 1])
def test_engine_speculative_snapshot_rollback_and_prefix(reject_at):
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
    repeated = engine.generate([prompt], params(True, max_new_tokens=6))[0].outputs[0]
    assert repeated.token_ids == actual.token_ids
    assert engine.stats["prefix_hits"] > 0


def test_optimizer_legacy_resume_rejected_before_model_mutation(tmp_path):
    from trainer.utils import load_checkpoint
    from trainer.muon import FusedAdamW

    m = model()
    path = tmp_path / "model.safetensors"
    m.save_weights(str(path))
    path.with_suffix(".json").write_text(
        json.dumps(
            {"config": m.config.to_dict(), "execution": checkpoint_execution(m.config)}
        )
    )
    before = [(k, np.array(v)) for k, v in tree_flatten(m.parameters())]
    args = SimpleNamespace(
        resume=str(path), reset_optimizer=False, freeze_backbone=False
    )
    with pytest.raises(ValueError, match="legacy Adam"):
        load_checkpoint(str(path), m, FusedAdamW(learning_rate=0.001), args)
    for k, v in tree_flatten(m.parameters()):
        np.testing.assert_array_equal(np.array(v), dict(before)[k])


def test_optimizer_without_precision_sidecar_cannot_silently_resume(tmp_path):
    from trainer.utils import load_checkpoint
    from trainer.muon import FusedAdamW

    m = model(thinking_enabled=False)
    path = tmp_path / "legacy.safetensors"
    m.save_weights(str(path))
    path.with_suffix(".optimizer.safetensors").touch()
    args = SimpleNamespace(
        resume=str(path), reset_optimizer=False, freeze_backbone=False
    )
    with pytest.raises(ValueError, match="reset_optimizer"):
        load_checkpoint(str(path), m, FusedAdamW(learning_rate=0.001), args)


def test_sft_sidecar_keeps_legacy_unless_thinking_is_explicit(monkeypatch, tmp_path):
    from trainer.config import get_sft_parser
    from trainer.utils import build_config_from_sidecar

    legacy = config().to_dict()
    monkeypatch.setattr("trainer.utils.load_checkpoint_config", lambda *a: legacy)
    for options, thinking in (([], False), (["--thinking"], True)):
        args = get_sft_parser().parse_args(options)
        args.save_dir = str(tmp_path)
        values, found = build_config_from_sidecar(args, "old.safetensors")
        current = VibyConfig(**values)
        assert found and current.thinking_enabled == thinking
        assert current.ncp_enabled != thinking
