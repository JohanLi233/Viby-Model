"""Contextual targets, causal feedback, and progress-preserving v1 migration."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from test_dpr import cfg, ids
from _v41_common import build
from model.config import VibyConfig
from trainer.flops import dpr_train_flops_per_token
from trainer.muon import create_mixed_optimizer
from trainer.utils import (
    convert_model_dtype,
    load_checkpoint,
    save_checkpoint,
    get_optimizer_steps,
)
from scripts.upgrade_dpr_checkpoint import upgrade


def context_cfg(**kw):
    return cfg(dpr_variant="contextual_v2", **kw)


def test_projection_frozen_and_target_gradient_detached():
    model = convert_model_dtype(build(context_cfg()), "bfloat16")
    d = model.model.dpr
    assert not hasattr(d, "target")
    assert "context_projection" not in dict(tree_flatten(d.trainable_parameters()))
    assert d.context_projection.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(d.context_projection @ d.context_projection.T), np.eye(4), atol=2e-7
    )
    h = mx.random.normal((2, 8, 64))
    y = d.contextual_targets(h)
    assert y.dtype == mx.float32 and y.shape == (2, 6, 4)
    assert (
        float(mx.max(mx.abs(mx.grad(lambda a: d.contextual_targets(a).sum())(h)))) == 0
    )
    # A change outside target window cannot affect that window's target.
    np.testing.assert_array_equal(
        y[:, 0], d.contextual_targets(h.at[:, 0].add(100))[:, 0]
    )
    assert not bool(
        mx.array_equal(y[:, 0], d.contextual_targets(h.at[:, 1].add(100))[:, 0])
    )
    pi, z = d(h)[1:]
    value, metrics = d.auxiliary(pi, z, h, ids())
    assert float(value) == pytest.approx(float(metrics[0]))  # covariance is diagnostic


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("compiled", [False, True])
def test_contextual_forward_backward_causal_and_cached(dtype, compiled):
    model = convert_model_dtype(build(context_cfg()), dtype)
    model.model.dpr.output.weight = mx.full(
        model.model.dpr.output.weight.shape, 0.02, model.model.dpr.output.weight.dtype
    )
    x = ids()

    def objective(params):
        model.update(params)
        return model(x, labels=x).loss

    fn = mx.value_and_grad(objective)
    if compiled:
        fn = mx.compile(fn)
    params = model.trainable_parameters()
    loss, grads = fn(params)
    mx.eval(loss, grads)
    model.update(params)
    flat = dict(tree_flatten(grads))
    assert all(bool(mx.all(mx.isfinite(v))) for v in flat.values())
    assert float(mx.max(mx.abs(flat["model.dpr.predict.weight"]))) > 0
    assert float(mx.max(mx.abs(flat["model.embed.weight"]))) > 0
    assert not any("context_projection" in name for name in flat)
    model.eval()
    full = model(x).logits
    changed = model(x.at[:, 5:].add(30)).logits
    tol = 0.06 if dtype == "bfloat16" else 2e-4
    np.testing.assert_allclose(
        np.array(full[:, :5].astype(mx.float32)),
        np.array(changed[:, :5].astype(mx.float32)),
        atol=tol,
    )
    pre, cache = model.prefill(x[:, :4])
    pieces = [pre]
    for t in range(4, 8):
        step, cache = model.decode_step(x[:, t], cache)
        pieces.append(step[:, None])
    np.testing.assert_allclose(
        np.array(full.astype(mx.float32)),
        np.array(mx.concatenate(pieces, 1).astype(mx.float32)),
        atol=tol,
    )
    seg = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 2)
    packed = model(x, segment_ids=seg).logits
    altered = model(x.at[:, :4].add(30), segment_ids=seg).logits
    np.testing.assert_allclose(
        np.array(packed[:, 4:].astype(mx.float32)),
        np.array(altered[:, 4:].astype(mx.float32)),
        atol=tol,
    )
    before = model(x, labels=x, need_logits=True).logits
    model.model.dpr.context_projection = -model.model.dpr.context_projection
    after = model(x, labels=x, need_logits=True).logits
    np.testing.assert_array_equal(
        np.array(before.astype(mx.float32)), np.array(after.astype(mx.float32))
    )


def test_parameter_cost_and_legacy_configuration():
    c = VibyConfig(dpr_enabled=True, n_mtp_layers=0)
    from model.dpr import DistributionalPredictiveResidual

    d = DistributionalPredictiveResidual(c)
    assert sum(v.size for _, v in tree_flatten(d.trainable_parameters())) == 270468
    assert sum(v.size for _, v in tree_flatten(d.parameters())) == 303236
    assert dpr_train_flops_per_token(c, 1024) == 1622016 + 65536
    old = c.to_dict()
    old.pop("dpr_variant")
    assert VibyConfig.from_dict(old).dpr_variant == "legacy_v1"


def test_empty_and_single_contextual_target_windows():
    model = build(context_cfg())
    for x in (ids()[:, :2], mx.zeros((1, 8), mx.int32)):
        out = model(x, labels=x)
        assert float(out.dpr_loss) == 0
        assert bool(mx.isfinite(out.loss))
    out = model(ids()[:1, :3], labels=ids()[:1, :3])
    assert float(out.dpr_metrics[3]) == 1
    assert float(out.dpr_metrics[6]) == 1
    assert float(out.dpr_loss) == pytest.approx(float(out.dpr_metrics[0]))


def test_upgrade_preserves_weights_moments_progress_and_zero_target_steps(tmp_path):
    from trainer.config import get_pretrain_parser

    args = get_pretrain_parser().parse_args(
        ["--no-psr", "--dpr", "--out_dir", str(tmp_path), "--learning_rate", ".0001"]
    )
    args.save_dir = str(tmp_path)
    args.dpr_consumed_tokens, args.dpr_warmup_tokens = 1234, 5678
    args.epoch_shuffle_state = None
    c = cfg()
    model = convert_model_dtype(build(c), "bfloat16")
    model.freeze(keys=["freq_cos", "freq_sin"])
    opt = create_mixed_optimizer(model, args)
    loss, grads = nn.value_and_grad(model, lambda m: m(ids(), labels=ids()).loss)(model)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    save_checkpoint(model, opt, 0, 99, args, c)
    source = tmp_path / "pretrain_64.safetensors"
    # Exercise genuinely old sidecars without the explicit version field.
    meta = json.loads(source.with_suffix(".json").read_text())
    meta["config"].pop("dpr_variant")
    source.with_suffix(".json").write_text(json.dumps(meta))
    destination = tmp_path / "upgraded"
    assert upgrade(source, destination, dry_run=True)["step"] == 99
    assert not destination.exists()
    upgrade(source, destination)
    converted = destination / source.name
    new = convert_model_dtype(build(context_cfg()), "bfloat16")
    new.freeze(keys=["freq_cos", "freq_sin"])
    new_opt = create_mixed_optimizer(new, args)
    epoch, step = load_checkpoint(str(converted), new, new_opt, args)
    assert (epoch, step) == (0, 100)
    assert args.dpr_consumed_tokens == 1234 and args.dpr_warmup_tokens == 5678
    assert get_optimizer_steps(opt) == get_optimizer_steps(new_opt)
    old_params = dict(tree_flatten(model.parameters()))
    for name, value in tree_flatten(new.parameters()):
        if name.endswith("context_projection"):
            continue
        np.testing.assert_array_equal(
            np.array(value.astype(mx.float32)),
            np.array(old_params[name].astype(mx.float32)),
        )
    old_state = dict(tree_flatten(opt.state))
    for name, value in tree_flatten(new_opt.state):
        np.testing.assert_array_equal(
            np.array(value.astype(mx.float32)),
            np.array(old_state[name].astype(mx.float32)),
        )
    new.eval()
    model.eval()
    np.testing.assert_array_equal(
        np.array(model(ids()).logits.astype(mx.float32)),
        np.array(new(ids()).logits.astype(mx.float32)),
    )
    new.train()
    loss, grads = nn.value_and_grad(new, lambda m: m(ids(), labels=ids()).loss)(new)
    new_opt.update(new, grads)
    mx.eval(loss, new.parameters(), new_opt.state)
    assert bool(mx.isfinite(loss))
    with pytest.raises(FileExistsError):
        upgrade(source, destination)
