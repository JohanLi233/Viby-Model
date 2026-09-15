"""CED-aware NCP mechanism gates. No quality or throughput acceptance."""

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.ncp import NextConceptPrediction, NCP_METRICS
from trainer.config import get_pretrain_parser, get_sft_parser, setup_training_args
from trainer.utils import (
    build_model_kwargs,
    checkpoint_execution,
    validate_checkpoint_execution,
    ncp_warm_start_prefixes,
    ncp_warm_start_drop_prefixes,
    load_model_weights,
    convert_model_dtype,
    build_config_from_sidecar,
)


def config(**kw):
    values = dict(
        preset="tiny",
        ncp_enabled=True,
        thinking_enabled=False,
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
        ncp_memory_dim=16,
        ncp_heads=2,
        ncp_codes=16,
    )
    values.update(kw)
    return VibyConfig(**values)


def tokens(t=17, b=1):
    return (mx.arange(t * b).reshape(b, t) % 40 + 1).astype(mx.int32)


def model(**kw):
    mx.random.seed(19)
    return VibyForCausalLM(config(**kw))


def close(a, b, atol=2e-5):
    np.testing.assert_allclose(
        np.array(a.astype(mx.float32)),
        np.array(b.astype(mx.float32)),
        atol=atol,
        rtol=atol,
    )


def norm(x):
    return float(mx.sum(mx.abs(x)).item())


def test_default_parser_preset_and_serialized_identity(monkeypatch, tmp_path):
    assert not VibyConfig().ncp_enabled and not VibyConfig().thinking_enabled
    for options in ([], ["--preset", "tiny"]):
        argv = options + ["--out_dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
        args = setup_training_args(get_pretrain_parser().parse_args(argv))
        cfg = VibyConfig(**build_model_kwargs(args))
        assert not cfg.ncp_enabled and not cfg.thinking_enabled
        assert (cfg.ncp_stride, cfg.ncp_layers, cfg.ncp_memory_dim) == (4, 2, 128)
        assert cfg.n_mtp_layers == 0
        assert VibyConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
        assert checkpoint_execution(cfg)["kind"] == "token_ced_v1"
    assert not VibyConfig.from_dict({"preset": "tiny"}).ncp_enabled
    parser = get_pretrain_parser()
    assert not any(
        a.dest in ("ncp_variant", "ncp_enabled", "ncp_kv_source")
        for a in parser._actions
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(ncp_stride=0),
        dict(ncp_groups=3),
        dict(ncp_memory_dim=15),
        dict(ncp_arch="old"),
        dict(ncp_vq_weight=float("nan")),
        dict(ncp_enabled=True, ced_recurrent_enabled=True),
        dict(kv_source_layers=(0, 3)),
    ],
)
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_zero_gate_baseline_and_clean_memory_intervention():
    m = model()
    x = tokens()
    baseline = model(ncp_enabled=False)
    baseline.load_weights(
        [
            (p, v)
            for p, v in tree_flatten(m.parameters())
            if not p.startswith("model.ncp.")
        ]
    )
    close(m(x).logits, baseline(x).logits, atol=0)
    before, c0 = m.prefill(x)
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.7])
    after, c1 = m.prefill(x)
    boundary = m.config.n_encoder_layers
    close(c0[boundary].compress_kv, c1[boundary].compress_kv, atol=0)
    close(c0[boundary].index_k, c1[boundary].index_k, atol=0)
    assert norm(after[:, 3:] - before[:, 3:]) > 1e-5
    assert norm(c0[boundary].window - c1[boundary].window) > 1e-5
    close(after[:, :3], before[:, :3], atol=0)


@pytest.mark.parametrize("prefill", [1, 3, 4, 5, 8, 15])
def test_native_cache_parity_and_exact_shared_kv_size(prefill):
    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.4])
    x = tokens(19, 2)
    expected = m(x).logits
    first, cache = m.prefill(x[:, :prefill])
    parts = [first]
    for i in range(prefill, 19):
        out, cache = m.decode_step(x[:, i], cache)
        parts.append(out[:, None])
    close(expected, mx.concatenate(parts, axis=1))
    state = cache.ncp_state
    assert state.memory.shape == (2, 4, 16)
    assert len(state.layer_memories) == 1
    assert state.layer_memories[0].shape == (2, 4, 16)
    assert state.states.shape == (2, 1, 2, 64)
    assert state.pending.shape == (2, 3, 64)
    assert state.prediction.shape == (2, 1, 64)
    assert set(vars(state)) == {
        "memory",
        "layer_memories",
        "states",
        "pending",
        "prediction",
        "tokens",
        "signature",
    }
    with pytest.raises(ValueError, match="fresh prefill"):
        cache.rewind(1)
    with pytest.raises(ValueError, match="clock"):
        m(x[:, :1], cache=cache, start_pos=0)


def test_chunked_prefill_state_parity():
    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.3])
    x = tokens(23)
    expected = m(x).logits
    first, cache = m.prefill(x[:, :5])
    parts = [first]
    for start, end in ((5, 6), (6, 11), (11, 23)):
        parts.append(m(x[:, start:end], start_pos=start, cache=cache).logits)
    close(expected, mx.concatenate(parts, axis=1))


def test_document_local_pooling_padding_and_tail_feedback():
    cfg = config()
    ncp = NextConceptPrediction(cfg)
    ncp.feedback_gate = mx.array([1.0])
    mx.random.seed(9)
    e = mx.random.normal((1, 21, 64))
    docs = mx.array([[0] * 3 + [1] * 10 + [0] * 8])
    pad = mx.array([[1] * 18 + [0, 0, 0]], mx.bool_)
    result = ncp(e, docs, pad)
    assert int(result["groups"]) == 3 and int(result["pairs"]) == 1
    close(result["signal"][:, :6], mx.zeros((1, 6, 64)), atol=0)
    assert norm(result["signal"][:, 10:13]) > 0  # last group has no next target
    close(result["signal"][:, 13:16], mx.zeros((1, 3, 64)), atol=0)
    close(result["signal"][:, 18:], mx.zeros((1, 3, 64)), atol=0)
    changed = mx.concatenate([e[:, :13] + 10, e[:, 13:]], axis=1)
    close(result["signal"][:, 13:], ncp(changed, docs, pad)["signal"][:, 13:])
    empty = ncp(e, docs, mx.zeros((1, 21), mx.bool_))
    assert (
        norm(empty["signal"]) == 0
        and float(empty["ncp_loss"]) == 0
        and float(empty["vq_loss"]) == 0
    )
    for t in (1, 2, 3):
        short = ncp(e[:, :t])
        assert norm(short["signal"]) == 0


def test_causality_packed_and_padding_gap_isolation():
    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.6])
    x = tokens(24)
    docs = mx.array([[0] * 8 + [1] * 16])
    pad = mx.ones((1, 24), mx.int32)
    first = m(x, segment_ids=docs, attention_mask=pad).logits
    changed = x.at[:, 0:8].add(2)
    close(first[:, 8:], m(changed, segment_ids=docs, attention_mask=pad).logits[:, 8:])
    for cut in (3, 4, 7, 11, 16):
        changed = mx.concatenate([x[:, :cut], (x[:, cut:] + 3) % 48], axis=1)
        close(
            first[:, :cut],
            m(changed, segment_ids=docs, attention_mask=pad).logits[:, :cut],
        )
    ncp = m.model.ncp
    e = mx.random.normal((1, 16, 64))
    pad = mx.array([[1] * 7 + [0] + [1] * 8], mx.bool_)
    a = ncp(e, pad_mask=pad)
    e2 = e.at[:, :8].add(3)
    close(a["signal"][:, 8:], ncp(e2, pad_mask=pad)["signal"][:, 8:])


def test_loss_target_and_codebook_gradient_boundaries():
    ncp = NextConceptPrediction(config())
    c = mx.random.normal((1, 3, 64))
    p = mx.random.normal((1, 3, 64))
    r = mx.random.normal((1, 3, 64)) * 0.1
    valid = mx.ones((1, 3), mx.bool_)
    adjacent = mx.ones((1, 2), mx.bool_)
    # v3: the copy base and the differential target are both detached; concepts
    # receive auxiliary gradient only through the residual head input path.
    for index in (0, 1):
        grad = mx.grad(lambda x: ncp.auxiliary_losses(x, p, r, valid, adjacent)[index])(
            c
        )
        assert norm(grad) == 0
    gr = mx.grad(lambda x: ncp.auxiliary_losses(c, p, x, valid, adjacent)[0])(r)
    assert norm(gr[:, :2]) > 0 and norm(gr[:, 2:]) == 0
    _, grads = nn.value_and_grad(
        ncp, lambda net: net.auxiliary_losses(c, p, r, valid, adjacent)[1]
    )(ncp)
    assert norm(grads["codebook"]) > 0
    assert all(norm(v) == 0 for path, v in tree_flatten(grads) if path != "codebook")
    # Historical concept inputs do receive the NCP gradient once the zero-init
    # output projection has moved.
    ncp.residual_out.weight = mx.random.normal(ncp.residual_out.weight.shape) * 0.05
    e = mx.random.normal((1, 16, 64))
    ge = mx.grad(lambda x: ncp(x)["ncp_loss"])(e)
    assert norm(ge[:, :4]) > 0 and norm(ge[:, 12:]) == 0


def test_ntp_gradient_gate_and_joint_training_contract():
    m = model()
    x = tokens(16)
    y = (x + 1) % 48
    loss, grads = nn.value_and_grad(m, lambda net: net(x, labels=y).lm_loss)(m)
    assert norm(grads["model"]["ncp"]["feedback_gate"]) > 0
    assert bool(mx.all(mx.abs(grads["model"]["ncp"]["state_gates"]) > 0))
    assert norm(grads["model"]["ncp"]["prediction_gates"]) > 0
    assert norm(grads["model"]["ncp"]["codebook"]) == 0
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.4])
    # The zero-init residual output projection blocks NTP gradient into the
    # residual MLP until the auxiliary loss moves it; model that post-warmup
    # state here (same semantics as the boundary test above).
    m.model.ncp.residual_out.weight = (
        mx.random.normal(m.model.ncp.residual_out.weight.shape) * 0.05
    )
    _, grads = nn.value_and_grad(m, lambda net: net(x, labels=y).lm_loss)(m)
    flat = dict(tree_flatten(grads))
    # v3: the codebook is only trained by VQ; NTP reaches the concept branch
    # through the copy-residual prediction and state paths, not the PQ table.
    assert norm(flat["model.ncp.codebook"]) == 0
    for key in (
        "model.ncp.residual_in.weight",
        "model.ncp.pool_slots",
        "model.ncp.layers.0.query.weight",
        "model.ncp.layers.1.query.weight",
        "model.ncp.memory_project.weight",
        "model.layers.2.attn.compressor.wkv.weight",
        "model.embed.weight",
    ):
        assert norm(flat[key]) > 0, key
    out = m(x, labels=y)
    close(
        out.loss,
        out.lm_loss
        + m.config.z_loss_weight * out.z_loss
        + m.config.aux_balance_loss_weight * out.aux_loss
        + m.config.ncp_loss_weight * out.ncp_loss
        + m.config.ncp_vq_weight * out.vq_loss,
    )
    assert all(bool(mx.all(mx.isfinite(v))) for _, v in tree_flatten(grads))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_compiled_joint_loss_and_gradients(dtype):
    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.2])
    convert_model_dtype(m, dtype)
    x = tokens(16, 2)
    y = (x + 1) % 48
    docs = mx.array([[0] * 5 + [1] * 11, [0] * 16])
    pad = mx.array([[1] * 14 + [0] * 2, [1] * 16])

    def objective(net):
        out = net(x, labels=y, loss_mask=pad, attention_mask=pad, segment_ids=docs)
        return out.loss

    eager, eg = nn.value_and_grad(m, objective)(m)

    def explicit_objective(parameters):
        m.update(parameters)
        return objective(m)

    compiled = mx.compile(mx.value_and_grad(explicit_objective))
    actual, ag = compiled(m.trainable_parameters())
    close(actual, eager, atol=0.04 if dtype == "bfloat16" else 2e-5)
    for (p, a), (_, b) in zip(tree_flatten(ag), tree_flatten(eg)):
        assert bool(mx.all(mx.isfinite(a))), p
        close(a, b, atol=0.04 if dtype == "bfloat16" else 2e-4)


def test_engine_batch_prefix_and_state_snapshots():
    from engine import VibyEngine, SamplingParams
    from engine.memory import capture_state, restore_state, state_bytes, cache_bytes
    from model.cache import VibyCache

    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.5])
    m.eval()
    prompts = [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12, 13, 14], [2, 5, 8]]
    params = SamplingParams(
        max_new_tokens=5, do_sample=False, eos_token_id=None, logprobs=True
    )
    engine = VibyEngine(m, max_num_seqs=2, prefix_stride=3)
    result = engine.generate(prompts, params)
    for prompt, out in zip(prompts, result):
        reference = VibyEngine(m, enable_prefix_cache=False).generate([prompt], params)[
            0
        ]
        assert out.outputs[0].token_ids == reference.outputs[0].token_ids
        np.testing.assert_allclose(
            out.outputs[0].logprobs, reference.outputs[0].logprobs, atol=2e-5
        )
    again = engine.generate(prompts, params)
    assert engine.stats["prefix_hits"] > 0
    assert [o.outputs[0].token_ids for o in again] == [
        o.outputs[0].token_ids for o in result
    ]
    x = tokens(9)
    _, cache = m.prefill(x[:, :5])
    snapshot = capture_state(cache, 0, 5)
    saved = mx.array(snapshot.ncp.memory)
    m.decode_step(x[:, 5], cache)
    close(snapshot.ncp.memory, saved, atol=0)
    restored = VibyCache(m.config)
    restore_state(restored, 0, snapshot)
    close(m(x[:, 5:9], cache=restored, start_pos=5).logits, m(x).logits[:, 5:9])
    assert state_bytes(snapshot) >= snapshot.ncp.nbytes() > 0
    assert cache_bytes(cache) >= cache.ncp_state.nbytes()


def test_checkpoint_migration_is_explicit_and_missing_ncp_is_strict(tmp_path):
    base = model(ncp_enabled=False)
    target = model()
    path = tmp_path / "old.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(base.parameters())))
    path.with_suffix(".json").write_text(json.dumps({"config": base.config.to_dict()}))
    args = SimpleNamespace(reset_optimizer=False, auto_resume=False)
    with pytest.raises(ValueError, match="reset_optimizer"):
        validate_checkpoint_execution(path, target.config, args)
    args.reset_optimizer = True
    changed = validate_checkpoint_execution(path, target.config, args)
    prefixes = ncp_warm_start_prefixes(path, target.config, changed)
    assert prefixes == ("model.ncp.",)
    assert load_model_weights(target, str(path), allow_fresh_prefixes=prefixes)
    with pytest.raises(ValueError, match="缺少模型参数"):
        load_model_weights(target, str(path))
    path.with_suffix(".json").write_text(
        json.dumps({"config": target.config.to_dict()})
    )
    assert not validate_checkpoint_execution(path, target.config, args)
    assert ncp_warm_start_prefixes(path, target.config, False) == ()


def test_sft_sidecar_inherits_new_and_legacy_architecture(tmp_path):
    for cfg in (config(), config(ncp_enabled=False)):
        path = tmp_path / "pretrain.json"
        path.write_text(json.dumps({"config": cfg.to_dict()}))
        args = get_sft_parser().parse_args([])
        args.save_dir = str(tmp_path)
        values, found = build_config_from_sidecar(args, "pretrain.safetensors")
        assert found and VibyConfig(**values).ncp_enabled == cfg.ncp_enabled
        assert VibyConfig(**values).ncp_memory_dim == cfg.ncp_memory_dim
    old = config(ncp_enabled=False).to_dict()
    old = {k: v for k, v in old.items() if not k.startswith("ncp_")}
    path.write_text(json.dumps({"config": old}))
    values, found = build_config_from_sidecar(args, "pretrain.safetensors")
    assert not VibyConfig(**values).ncp_enabled


def test_real_compiled_trainer_updates_ncp_and_codebook_adamw():
    from trainer.base_trainer import BaseTrainer
    from trainer.muon import create_mixed_optimizer, FusedAdamW

    m = model()
    trainer = object.__new__(BaseTrainer)
    args = get_pretrain_parser().parse_args([])
    args.learning_rate = 1e-4
    args.muon_lr = 4e-4
    args.accumulation_steps = 2
    trainer.args = args
    trainer.model = m
    trainer.lm_config = m.config
    optimizer = create_mixed_optimizer(m, args)
    # MultiOptimizer's actual routing must keep PQ codewords out of MuonH.
    routed = optimizer._split_dictionary(m.trainable_parameters())
    owners = [
        i
        for i, params in enumerate(routed)
        if "model.ncp.codebook" in dict(tree_flatten(params))
    ]
    assert len(owners) == 1
    assert isinstance(optimizer.optimizers[owners[0]], FusedAdamW)
    before = mx.array(m.model.ncp.codebook)
    trainer._expl_nest_mu = 0
    trainer._en_delta = None
    trainer._loss_and_grad = trainer._build_loss_and_grad()
    x = tokens(16)
    y = (x + 1) % 48
    mask = mx.ones(x.shape)
    values, grads = trainer._compute_loss_and_grad(x, y, mask, mask)
    mx.eval(values, grads)
    assert values[5].shape == (len(NCP_METRICS),)
    assert float(values[0]) > float(values[3])
    optimizer.update(m, grads)
    mx.eval(m.parameters(), optimizer.state)
    assert norm(m.model.ncp.codebook - before) > 0
    assert norm(m.model.ncp.feedback_gate) > 0
    second, sg = trainer._compute_loss_and_grad(x, y, mask, mask)
    trainer._loss_and_grad = mx.value_and_grad(trainer._loss_and_grad_with_params)
    eager, eg = trainer._compute_loss_and_grad(x, y, mask, mask)
    close(second[0], eager[0])
    assert all(bool(mx.all(mx.isfinite(v))) for _, v in tree_flatten(sg))
    assert norm(sg["model"]["ncp"]["layers"][0]["query"]["weight"]) > 0


def test_ncp_flops_and_memory_count_include_low_frequency_work():
    from trainer.flops import training_flops_per_token, gemm_active_params

    m = model()
    plain = model(ncp_enabled=False)
    size = sum(
        v.size
        for p, v in tree_flatten(m.trainable_parameters())
        if p.startswith("model.ncp.") and v.ndim >= 2
    )
    assert size == m.config.ncp_matrix_parameters()
    assert gemm_active_params(m) - gemm_active_params(plain) == size
    cost = training_flops_per_token(m, 16) - training_flops_per_token(plain, 16)
    assert cost > 6 * size / 4
    assert cost < 6 * size
    route_params = len(m.config.ncp_decoder_layers) * m.config.dim * m.config.ncp_layers
    assert (
        training_flops_per_token(m, 3) - training_flops_per_token(plain, 3)
        == 6 * route_params
    )


def test_global_kv_gradient_is_independent_of_concept_but_not_encoder():
    m = model()
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.7])
    x = tokens(12)

    def memory_objective(net):
        _, cache = net.prefill(x)
        return mx.sum(cache[net.config.n_encoder_layers].compress_kv[:, :12, 0])

    _, grad = nn.value_and_grad(m, memory_objective)(m)
    flat = dict(tree_flatten(grad))
    assert norm(flat["model.embed.weight"]) > 0
    assert all(
        norm(v) == 0 for path, v in flat.items() if path.startswith("model.ncp.")
    )


def test_memory_override_keeps_main_and_indexer_queries_on_decoder(monkeypatch):
    m = model()
    x = tokens(16)
    attn = m.model.layers[m.config.n_encoder_layers].attn
    q_inputs = []
    index_inputs = []
    original_q = attn._q
    original_index = attn.indexer._query_weights

    def query(x, *args, **kwargs):
        q_inputs.append(mx.array(x))
        return original_q(x, *args, **kwargs)

    def index(x, *args, **kwargs):
        index_inputs.append(mx.array(x))
        return original_index(x, *args, **kwargs)

    monkeypatch.setattr(attn, "_q", query)
    monkeypatch.setattr(attn.indexer, "_query_weights", index)
    mx.eval(m(x).logits)
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.8])
    mx.eval(m(x).logits)
    assert len(q_inputs) == 2 and len(index_inputs) == 2
    close(q_inputs[0], index_inputs[0], atol=0)
    close(q_inputs[1], index_inputs[1], atol=0)
    assert norm(q_inputs[1] - q_inputs[0]) > 0


@pytest.mark.parametrize("reject_at", [None, 1])
def test_ncp_engine_speculative_commit_and_rollback(reject_at):
    from engine import VibyEngine
    from test_engine_speculative import controlled_proposals, params

    m = model(n_mtp_layers=1, dspark_n_routed_experts=4, dspark_n_activated_experts=2)
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.5])
    m.eval()
    prompt = [1, 2, 3, 4, 5]
    expected = VibyEngine(m).generate([prompt], params(max_new_tokens=6))[0].outputs[0]
    engine = VibyEngine(m, prefix_stride=3)
    controlled_proposals(engine, reject_at)
    actual = engine.generate([prompt], params(True, max_new_tokens=6))[0].outputs[0]
    assert actual.token_ids == expected.token_ids
    np.testing.assert_allclose(actual.logprobs, expected.logprobs, atol=3e-5)


def test_bf16_sparse_and_fused_indexer_dispatch_matches_reference(monkeypatch):
    from model.kernels import sparse_attention, indexer_select

    if mx.default_device() != mx.gpu:
        pytest.skip("Metal dispatch gate")
    m = model(
        n_layers=12,
        n_heads=16,
        head_dim=128,
        index_n_heads=4,
        ncp_memory_dim=128,
        ncp_heads=4,
    )
    convert_model_dtype(m, "bfloat16")
    m.model.ncp.state_gates = mx.array([0.25] * len(m.config.ncp_decoder_layers))
    m.model.ncp.prediction_gates = mx.array(
        [0.15] * (len(m.config.ncp_decoder_layers) - 1)
    )
    m.model.ncp.feedback_gate = mx.array([0.3], mx.bfloat16)
    x = tokens(16)
    y = (x + 1) % 48
    calls = []
    original = sparse_attention.indexed_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse_attention, "indexed_attention", counted)
    monkeypatch.setattr(sparse_attention, "_ENABLED", True)
    monkeypatch.setattr(indexer_select, "_ENABLED", True)
    params = m.trainable_parameters()
    biases = m.moe_bias_stack()
    value, grad = nn.value_and_grad(m, lambda net: net(x, labels=y).loss)(m)
    mx.eval(value, grad)
    assert calls
    m.update(params)
    m.apply_moe_biases(biases)
    monkeypatch.setattr(sparse_attention, "_ENABLED", False)
    monkeypatch.setattr(indexer_select, "_ENABLED", False)
    reference, rg = nn.value_and_grad(m, lambda net: net(x, labels=y).loss)(m)
    close(value, reference, atol=0.04)
    assert all(bool(mx.all(mx.isfinite(v))) for _, v in tree_flatten(grad))
    for path in ("codebook", "feedback_gate"):
        close(grad["model"]["ncp"][path], rg["model"]["ncp"][path], atol=0.04)


def test_concept_document_offset_preserves_relative_position_semantics():
    ncp = NextConceptPrediction(config())
    ncp.feedback_gate = mx.array([0.4])
    mx.random.seed(18)
    e = mx.random.normal((1, 12, 64))
    prefix = mx.random.normal((1, 5, 64))
    standalone = ncp(e)["signal"]
    packed = ncp(mx.concatenate([prefix, e], axis=1), mx.array([[0] * 5 + [1] * 12]))[
        "signal"
    ]
    close(standalone, packed[:, 5:])


def test_state_only_ntp_path_bypasses_prediction_and_selects_depth_per_token():
    m = model(ncp_loss_weight=0, ncp_vq_weight=0)
    ncp = m.model.ncp
    ncp.state_gates = mx.array([0.4] * len(m.config.ncp_decoder_layers))
    x = tokens(16)
    _, grad = nn.value_and_grad(m, lambda net: net(x, labels=(x + 1) % 48).loss)(m)
    flat = dict(tree_flatten(grad))
    for key in (
        "memory_updates.0.weight",
        "layers.0.query.weight",
        "layers.1.query.weight",
        "state_projects.0.weight",
        "state_projects.1.weight",
        "state_routes.0.weight",
        "state_routes.1.weight",
    ):
        assert norm(flat["model.ncp." + key]) > 0, key
    for key in ("codebook", "residual_in.weight", "feedback.weight"):
        assert norm(flat["model.ncp." + key]) == 0, key
    m.eval()
    normal = m(x).logits
    off = m(x, ncp_intervention="off").logits
    assert norm(normal[:, 3:] - off[:, 3:]) > 0
    close(normal[:, :3], off[:, :3], atol=0)
    # A fixed bank and differing token queries must produce differing depth mixtures.
    states = mx.broadcast_to(
        mx.stack([mx.ones((64,)), -mx.ones((64,))])[None, None], (1, 2, 2, 64)
    )
    query = mx.stack([mx.ones((64,)), -mx.ones((64,))])[None]
    selected = ncp.state_signal(query, states, 0)
    assert norm(selected[:, 0] - selected[:, 1]) > 0


def test_second_layer_memory_contains_processed_history(monkeypatch):
    ncp = NextConceptPrediction(config())
    if ncp.residual_predictor:
        # v3's zero-init residual output would mask layer processing at init.
        ncp.residual_out.weight = mx.random.normal(ncp.residual_out.weight.shape) * 0.05
    c = mx.random.normal((1, 3, 64))
    pos = mx.array([[3, 7, 11]])
    visible = mx.tril(mx.ones((1, 3, 3), mx.bool_))
    memory = ncp.project_memory(c, pos)
    # Alter only the FIRST historical output; the last query entering layer 2 is fixed.
    original = type(ncp.layers[0]).__call__
    captured = []
    perturb = [False]

    def hook(layer, state, kv, positions, mask):
        if layer is ncp.layers[1]:
            captured.append((mx.array(state), mx.array(kv)))
        result = original(layer, state, kv, positions, mask)
        if layer is ncp.layers[0] and perturb[0]:
            result = result.at[:, 0, 0].add(2)
        return result

    monkeypatch.setattr(type(ncp.layers[0]), "__call__", hook)
    pred0, _, _ = ncp.predict_next(c, memory, pos, visible)
    perturb[0] = True
    pred1, _, _ = ncp.predict_next(c, memory, pos, visible)
    close(captured[0][0][:, -1], captured[1][0][:, -1], atol=0)
    assert norm(captured[0][1][:, 0] - captured[1][1][:, 0]) > 0
    assert norm(pred0[:, -1] - pred1[:, -1]) > 0


def test_v1_checkpoint_retains_execution_and_explicit_v3_warm_start(tmp_path):
    old = model(ncp_arch="ced_shared_kv_v1")
    old.model.ncp.feedback_gate = mx.array([0.5])
    x = tokens(13)
    expected = old(x).logits
    first, cache = old.prefill(x[:, :5])
    close(
        mx.concatenate([first, old(x[:, 5:], cache=cache, start_pos=5).logits], axis=1),
        expected,
    )
    assert cache.ncp_state.layer_memories == [] and cache.ncp_state.states is None
    path = tmp_path / "v1.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(old.parameters())))
    path.with_suffix(".json").write_text(json.dumps({"config": old.config.to_dict()}))
    loaded = VibyForCausalLM(VibyConfig.from_dict(old.config.to_dict()))
    load_model_weights(loaded, str(path), strict=True)
    close(loaded(x).logits, expected, atol=0)
    target = model()
    args = SimpleNamespace(reset_optimizer=False, auto_resume=False)
    with pytest.raises(ValueError, match="reset_optimizer"):
        validate_checkpoint_execution(path, target.config, args)
    args.reset_optimizer = True
    changed = validate_checkpoint_execution(path, target.config, args)
    prefixes = ncp_warm_start_prefixes(path, target.config, changed)
    drops = ncp_warm_start_drop_prefixes(path, target.config, changed)
    assert "model.ncp." not in prefixes
    assert "model.ncp.pool_slots" in prefixes
    assert drops == ("model.ncp.predict.", "model.ncp.predict_norm.")
    load_model_weights(
        target,
        str(path),
        allow_fresh_prefixes=prefixes,
        allow_drop_prefixes=drops,
    )
    close(target.model.ncp.codebook, old.model.ncp.codebook, atol=0)
    close(target.model.ncp.feedback_gate, old.model.ncp.feedback_gate, atol=0)
    assert VibyConfig().ncp_decoder_layers == (6, 10)


def test_v3_copy_residual_starts_as_exact_copy():
    m = model()
    x = tokens(16, 2)
    out = m(x, labels=x)
    metrics = dict(zip(NCP_METRICS, out.ncp_metrics.tolist()))
    # Zero-init residual: prediction == current concept, so the auxiliary loss
    # equals the copy (persistence) baseline and relative_mse_previous == 1.
    close(out.ncp_loss, mx.array(metrics["previous_mse"]), atol=1e-4)
    assert metrics["relative_mse_previous"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["state_gate"] == 0 and metrics["prediction_gate"] == 0


def test_v3_zero_residual_out_survives_trunc_normal_init():
    from model.init import apply_trunc_normal_init

    ncp = NextConceptPrediction(config())
    apply_trunc_normal_init(ncp, 64)
    assert norm(ncp.residual_out.weight) == 0
    assert norm(ncp.residual_in.weight) > 0


def test_v3_slot_pooling_convex_weights_and_token_order():
    ncp = NextConceptPrediction(config())
    grouped = mx.random.normal((2, 3, 4, 64))
    # All-zero logits degenerate to exact mean pooling.
    close(ncp._pool(grouped), mx.mean(grouped, axis=2), atol=1e-6)
    # Boost slot r=0 (group end -> token j = stride-1) to pin the token order.
    ncp.pool_slots = mx.concatenate([mx.full((64,), 20.0), mx.zeros((3 * 64,))])
    close(ncp._pool(grouped), grouped[:, :, 3, :], atol=1e-3)


def test_v3_zero_init_residual_blocks_concept_gradient_until_output_moves():
    ncp = NextConceptPrediction(config())
    e = mx.random.normal((1, 16, 64))
    _, grads = nn.value_and_grad(ncp, lambda net: net(e)["ncp_loss"])(ncp)
    flat = dict(tree_flatten(grads))
    assert norm(flat["residual_out.weight"]) > 0
    assert norm(flat["memory_project.weight"]) == 0


def test_v2_to_v3_warm_start_drops_pq_and_initializes_residual(tmp_path):
    old = model(ncp_arch="ced_state_v2")
    x = tokens(13)
    expected = old(x).logits
    path = tmp_path / "v2.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(old.parameters())))
    path.with_suffix(".json").write_text(json.dumps({"config": old.config.to_dict()}))
    target = model()
    args = SimpleNamespace(reset_optimizer=True, auto_resume=False)
    changed = validate_checkpoint_execution(path, target.config, args)
    prefixes = ncp_warm_start_prefixes(path, target.config, changed)
    drops = ncp_warm_start_drop_prefixes(path, target.config, changed)
    assert sorted(prefixes) == sorted(
        "model.ncp." + name
        for name in ("pool_slots", "residual_norm.", "residual_in.", "residual_out.")
    )
    assert drops == ("model.ncp.predict.", "model.ncp.predict_norm.")
    load_model_weights(
        target,
        str(path),
        allow_fresh_prefixes=prefixes,
        allow_drop_prefixes=drops,
    )
    close(target.model.ncp.codebook, old.model.ncp.codebook, atol=0)
    close(
        target.model.ncp.state_projects[0].weight,
        old.model.ncp.state_projects[0].weight,
        atol=0,
    )
    # All gates stay zero after warm start: the v3 forward must match v2 logits.
    close(target(x).logits, expected, atol=0)
