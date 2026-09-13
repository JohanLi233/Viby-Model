"""Focused DPR-JEPA causal, distributional, gradient and training contracts."""

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from _v41_common import cfg_tiny, build
from model.config import VibyConfig
from model.dpr import DistributionalPredictiveResidual, future_mask, kernel_score
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import (
    build_model_kwargs,
    convert_model_dtype,
    validate_checkpoint_execution,
)
from trainer.flops import dpr_train_flops_per_token


def cfg(**kw):
    base = dict(
        dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        n_mtp_layers=0,
        dpr_enabled=True,
        dpr_variant="legacy_v1",
        dpr_dim=4,
        dpr_width=12,
        dpr_horizon=2,
    )
    base.update(kw)
    return cfg_tiny(**base)


def ids():
    return mx.array([[5, 7, 8, 9, 10, 11, 12, 13], [15, 17, 18, 19, 20, 21, 22, 23]])


def test_cli_roundtrip_and_incompatible_modes(monkeypatch, tmp_path):
    argv = ["--dpr", "--no-psr", "--preset", "tiny", "--out_dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
    args = setup_training_args(get_pretrain_parser().parse_args(argv))
    c = VibyConfig(**build_model_kwargs(args))
    assert c.dpr_enabled and not c.psr_enabled and not c.ced_recurrent_enabled
    assert c.n_mtp_layers == 0 and c.dpr_loss_weight == 0.05 and c.dpr_dim == 32
    assert VibyConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
    for kw in (
        {"psr_enabled": True},
        {"ced_recurrent_enabled": True},
        {"n_mtp_layers": 1},
    ):
        with pytest.raises(ValueError, match="DPR requires"):
            cfg(**kw)
    path = tmp_path / "weights.safetensors"
    path.with_suffix(".json").write_text(json.dumps({"config": c.to_dict()}))
    assert not validate_checkpoint_execution(path, c, SimpleNamespace())
    c.dpr_objective = "mse"
    with pytest.raises(ValueError, match="Automatic resume"):
        validate_checkpoint_execution(path, c, SimpleNamespace(auto_resume=True))


def test_kernel_diagonal_and_bimodal_identity():
    pi = mx.array([0.5, 0.5])
    z = mx.array([[-1.0], [1.0]])
    targets = mx.array([[-1.0], [1.0]])
    correct = mx.mean(kernel_score(pi, z, targets))
    centroid = mx.mean(kernel_score(mx.array([1.0]), mx.array([[0.0]]), targets))
    assert float(centroid - correct) == pytest.approx(0.35460632219303956, abs=2e-7)
    assert float(kernel_score(mx.array([1.0]), mx.array([[3.0]]), mx.array([3.0]))) == 0


def test_exact_window_masks():
    x = mx.array([[5, 6, 2, 8, 9, 10, 0, 0]])
    seg = mx.array([[0, 0, 0, 1, 1, 1, -1, -1]])
    np.testing.assert_array_equal(
        future_mask(x, 2, segment_ids=seg), [[1, 0, 0, 1, 0, 0]]
    )
    # Check intermediate positions, not merely matching window endpoints.
    np.testing.assert_array_equal(
        future_mask(ids()[:1, :3], 2, segment_ids=mx.array([[0, 1, 0]])), [[0]]
    )
    pad = mx.array([[1, 0, 1]])
    assert not bool(future_mask(ids()[:1, :3], 2, pad_mask=pad).any())
    assert future_mask(ids()[:, :1], 2).shape == (2, 0)


def test_target_future_only_order_sensitive_and_embedding_detached():
    model = build(cfg())
    d = model.model.dpr
    x = ids()
    e = model.model.embed(x)
    changed = e.at[:, :1].add(10)
    np.testing.assert_array_equal(d.target(e)[:, 0], d.target(changed)[:, 0])
    assert float(mx.max(mx.abs(d.target(e)[:, 0] - d.target(e[:, ::-1])[:, 0]))) > 0
    g = mx.grad(lambda embeddings: d.target(embeddings).sum())(e)
    assert float(mx.max(mx.abs(g))) == 0
    _, grads = nn.value_and_grad(d, lambda m: mx.sum(m.target(e) ** 2))(d)
    assert float(mx.max(mx.abs(grads["target"]["offsets"][0]["weight"]))) > 0
    assert float(mx.max(mx.abs(grads["target"]["proj"]["weight"]))) > 0


@pytest.mark.parametrize("skip_init", [True, False])
@pytest.mark.parametrize("variant", ["legacy_v1", "contextual_v2"])
def test_base_initialization_rng_and_zero_residual(skip_init, variant):
    baseline = build(cfg(dpr_enabled=False), seed=123, skip_init=skip_init)
    rng_base = mx.random.uniform(shape=(8,))
    model = build(cfg(dpr_variant=variant), seed=123, skip_init=skip_init)
    rng_dpr = mx.random.uniform(shape=(8,))
    np.testing.assert_array_equal(rng_base, rng_dpr)
    base_params = dict(tree_flatten(baseline.parameters()))
    params = dict(tree_flatten(model.parameters()))
    for name, arr in base_params.items():
        np.testing.assert_array_equal(arr, params[name], err_msg=name)
    assert len(model.model.layers) == 4
    baseline.eval()
    model.eval()
    np.testing.assert_array_equal(baseline(ids()).logits, model(ids()).logits)
    np.testing.assert_array_equal(
        model(ids(), use_dpr=False).logits, model(ids()).logits
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_live_residual_causality_prefill_decode_and_packing(dtype):
    model = convert_model_dtype(build(cfg()), dtype)
    model.eval()
    d = model.model.dpr
    d.output.weight = mx.full(d.output.weight.shape, 0.02, d.output.weight.dtype)
    x = ids()
    full = model(x).logits
    changed = model(x.at[:, 5:].add(30)).logits
    np.testing.assert_allclose(
        full[:, :5].astype(mx.float32), changed[:, :5].astype(mx.float32), atol=2e-5
    )
    pre, cache = model.prefill(x[:, :4])
    chunks = [pre]
    for t in range(4, 8):
        step, cache = model.decode_step(x[:, t], cache)
        chunks.append(step[:, None])
    atol = 0.06 if dtype == "bfloat16" else 2e-4
    np.testing.assert_allclose(
        full.astype(mx.float32), mx.concatenate(chunks, 1).astype(mx.float32), atol=atol
    )
    with pytest.raises(ValueError, match="fresh cache"):
        model(x[:, :1], cache=cache, start_pos=8, use_dpr=False)
    seg = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 2)
    packed = model(x, segment_ids=seg).logits
    changed = model(x.at[:, :4].add(30), segment_ids=seg).logits
    np.testing.assert_allclose(
        packed[:, 4:].astype(mx.float32), changed[:, 4:].astype(mx.float32), atol=atol
    )
    # Target embeddings never enter logits, even when training loss is requested.
    a = model(x, labels=x, need_logits=True)
    d.target.bias = d.target.bias + 20
    b = model(x, labels=x, need_logits=True)
    np.testing.assert_array_equal(
        a.logits.astype(mx.float32), b.logits.astype(mx.float32)
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("compiled", [False, True])
def test_auxiliary_finite_gradients_and_live_weight(dtype, compiled):
    model = convert_model_dtype(build(cfg()), dtype)
    x = ids()
    params = model.trainable_parameters()

    def loss(p, weight):
        model.update(p)
        out = model(x, labels=x, dpr_weight=weight)
        return out.loss, (out.lm_loss, out.dpr_loss, out.dpr_metrics)

    fn = mx.value_and_grad(loss)
    if compiled:
        fn = mx.compile(fn)
    result, grads = fn(params, mx.array(0.05))
    mx.eval(result, grads)
    model.update(params)
    for name, value in tree_flatten(grads):
        assert bool(mx.all(mx.isfinite(value))), name
    flat = dict(tree_flatten(grads))
    for name in (
        "model.dpr.predict.weight",
        "model.dpr.target.offsets.0.weight",
        "model.dpr.target.proj.weight",
        "model.dpr.output.weight",
        "model.embed.weight",
    ):
        assert float(mx.max(mx.abs(flat[name]))) > 0, name
    assert result[1][2].dtype == mx.float32
    other, _ = fn(params, mx.array(0.1))
    mx.eval(other)
    model.update(params)
    assert float(other[0] - result[0]) == pytest.approx(
        float(result[1][1]) * 0.05, abs=0.005
    )
    assert float(result[1][2][3]) == 12


def test_no_future_and_single_target_keep_ntp():
    model = build(cfg())
    x = ids()[:, :2]
    out = model(x, labels=x)
    assert float(out.dpr_loss) == 0 and bool(mx.isfinite(out.lm_loss))
    x = ids()[:1, :3]
    out = model(x, labels=x)
    assert float(out.dpr_metrics[3]) == 1 and float(out.dpr_metrics[6]) == 1
    assert float(out.dpr_metrics[1]) >= 1
    assert bool(mx.isfinite(out.loss))
    x = mx.zeros((1, 8), mx.int32)
    out = model(x, labels=x)
    assert float(out.dpr_loss) == 0 and float(out.dpr_metrics[3]) == 0


def test_b_variant_skips_target_and_mse_variant_keeps_interface(monkeypatch):
    b = build(cfg(dpr_loss_weight=0))

    def fail(*a, **kw):
        raise AssertionError("target must not run")

    monkeypatch.setattr(type(b.model.dpr), "auxiliary", fail)
    out = b(ids(), labels=ids())
    assert out.dpr_loss is None
    monkeypatch.undo()
    c = build(cfg(dpr_objective="mse"))
    out = c(ids(), labels=ids())
    assert float(out.dpr_loss) == pytest.approx(
        float(out.dpr_metrics[1] + out.dpr_metrics[2])
    )


def test_parameter_budget_and_major_gemms():
    c = VibyConfig(dpr_enabled=True, n_mtp_layers=0, dpr_variant="legacy_v1")
    d = DistributionalPredictiveResidual(c)
    flat = dict(tree_flatten(d.parameters()))
    assert sum(p.size for p in flat.values()) == 799012
    assert sum(p.size for n, p in flat.items() if not n.startswith("target.")) == 270468
    assert dpr_train_flops_per_token(c, 1024) == int(1622016 + 2121728 * 1020 / 1024)


@pytest.mark.parametrize("variant", ["legacy_v1", "contextual_v2"])
def test_real_trainer_accumulation_qb_zero_head_and_checkpoint(tmp_path, variant):
    import time
    from trainer.base_trainer import BaseTrainer
    from trainer.utils import load_checkpoint, get_optimizer_steps

    args = get_pretrain_parser().parse_args(
        [
            "--dpr",
            "--no-psr",
            "--out_dir",
            str(tmp_path),
            "--learning_rate",
            ".0001",
            "--accumulation_steps",
            "2",
            "--max_steps",
            "4",
            "--cache_limit_gb",
            "0",
            "--max_seq_len",
            "8",
            "--batch_size",
            "2",
            "--log_interval",
            "1",
        ]
    )
    args.save_dir = str(tmp_path)
    args.warmup_iters = 0
    args.dpr_warmup_tokens = 32
    c = cfg(qb_stats_rows=32, dpr_variant=variant)
    model = convert_model_dtype(build(c), "bfloat16")
    model.freeze(keys=["freq_cos", "freq_sin"])
    tr = BaseTrainer(args, model, SimpleNamespace(pad_token_id=0), c, "pretrain")
    assert tr.psr_optimizer is None and args.muonh and args.compile_model
    original_target = (
        model.model.dpr.target.proj.weight
        if variant == "legacy_v1"
        else model.model.dpr.context_projection
    )
    x = ids()
    sample = (x, x + 1, mx.ones_like(x), mx.zeros_like(x))
    tr._run_epoch_steps(iter([sample] * 4), 0, 4, 4, None, 0, time.time(), 0)
    assert float(mx.max(mx.abs(model.model.dpr.output.weight))) > 0
    if variant == "legacy_v1":
        assert (
            float(mx.max(mx.abs(original_target - model.model.dpr.target.proj.weight)))
            > 0
        )
    else:
        np.testing.assert_array_equal(
            original_target, model.model.dpr.context_projection
        )
    assert bool(mx.all(mx.isfinite(model.moe_bias_stack())))
    assert args.dpr_consumed_tokens == 64
    records = [
        json.loads(line)
        for line in (tmp_path / "dpr_metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["grad/pre_clip"] > 0
    assert records[-1]["grad/dpr_predict"] >= 0
    saved = tmp_path / "pretrain_64.safetensors"
    meta = json.loads(saved.with_suffix(".json").read_text())
    assert meta["execution"]["kind"] == (
        "dpr_jepa_v1" if variant == "legacy_v1" else "dpr_jepa_v2"
    )
    assert meta["args"]["dpr_consumed_tokens"] == 64
    assert "model/dpr.py" in meta["source_sha256"]
    other = convert_model_dtype(build(c), "bfloat16")
    other.freeze(keys=["freq_cos", "freq_sin"])
    from trainer.muon import create_mixed_optimizer

    opt = create_mixed_optimizer(other, args)
    load_checkpoint(str(saved), other, opt, args)
    assert args.dpr_warmup_tokens == 32
    assert get_optimizer_steps(opt) == get_optimizer_steps(tr.optimizer)
    model.eval()
    other.eval()
    assert bool(mx.array_equal(model(x).logits, other(x).logits))


def test_explicit_bypass_and_particle_interventions(monkeypatch):
    model = build(cfg())
    model.eval()
    dpr = model.model.dpr
    dpr.output.weight = mx.random.normal(dpr.output.weight.shape) * 0.05
    x = ids()
    base = model(x).logits
    for intervention in ("mean", "shuffle"):
        changed = model(x, dpr_intervention=intervention).logits
        assert bool(mx.all(mx.isfinite(changed)))
        assert float(mx.max(mx.abs(base - changed))) > 1e-5

    def fail(*args, **kwargs):
        raise AssertionError("disabled DPR must not execute")

    monkeypatch.setattr(DistributionalPredictiveResidual, "__call__", fail)
    assert bool(mx.all(mx.isfinite(model(x, use_dpr=False).logits)))
