"""Integration gates: physical caches, repeated training, and saved execution."""

from types import SimpleNamespace
import json
import time

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from test_ced_recurrent import config, inputs
from model.config import VibyConfig
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser
from trainer.flops import training_flops_per_token
from trainer.utils import convert_model_dtype, load_checkpoint


def test_public_cache_short_memory_and_stage_isolation():
    model = build(config(index_topk=64))
    model.eval()
    x = inputs(13, batch=2)
    first, cache = model.prefill(x[:, :4])
    assert max_abs_diff(first, model(x[:, :4]).logits) < 3e-4
    assert set(cache.ced_stages) == {
        (r, layer) for r in range(3) for layer in range(7, 11)
    }
    assert len({id(c) for c in cache.ced_stages.values()}) == 12
    snapshots = {key: c["history"][0] for key, c in cache.ced_stages.items()}
    assert max_abs_diff(snapshots[(0, 7)], snapshots[(1, 7)]) > 0
    for pos in range(4, 13):
        logits, cache = model.decode_step(x[:, pos], cache)
        assert max_abs_diff(logits, model(x[:, : pos + 1]).logits[:, -1]) < 3e-4
        if pos == 4:  # A non-anchor token must not execute/update latent stages.
            assert all(
                bool(mx.array_equal(c["history"][0], snapshots[key]))
                for key, c in cache.ced_stages.items()
            )
    assert cache.start_pos == 13 and cache.ced_topk_idx.shape == (2, 1, 64)
    with pytest.raises(ValueError, match="fresh"):
        cache.rewind(1)
    with pytest.raises(ValueError, match="fresh cache"):
        model(x[:, :1], cache=cache, start_pos=13, decode=True, use_ced_recurrent=False)
    with pytest.raises(ValueError, match="single-document"):
        model.prefill(x, segment_ids=mx.zeros_like(x))
    # The public baseline switch must persist through decode as well.
    _, baseline_cache = model.prefill(x[:, :4], use_ced_recurrent=False)
    actual, _ = model.decode_step(x[:, 4], baseline_cache)
    expected = model(x[:, :5], use_ced_recurrent=False).logits[:, -1]
    assert max_abs_diff(actual, expected) < 3e-4
    assert baseline_cache.ced_stages == {}


def test_packed_capacity_routes_and_empty_qb_update():
    model = build(config())
    x = inputs(16, batch=2)
    docs = mx.array([[0] * 4 + [1] * 12, list(range(16))])
    pad = mx.array([[1] * 13 + [0] * 3, [1] * 16])
    out = model(x, labels=x, segment_ids=docs, attention_mask=pad, loss_mask=pad)
    assert out.ced_metrics.tolist() == [29, 3, 8, 12, 36]
    plan = out.ced_result["plan"]
    routes = out.ced_result["final_routes"]
    safe = mx.maximum(routes, 0)
    selected_docs = mx.take_along_axis(docs[:, None, :], safe, axis=2)
    assert bool(
        mx.all(
            (routes < 0)
            | (
                (routes <= plan.positions[..., None])
                & (selected_docs == plan.segment_ids[..., None])
            )
        )
    )
    for row, gate in enumerate(model._backbone_gates):
        if 7 <= gate.layer_idx <= 10:
            assert (
                float(mx.sum(out.moe_loads[row]))
                == 3 * 3 * model.config.n_activated_experts
            )
    model.update_moe_biases(out.moe_loads, out.moe_qb_margins)
    assert bool(mx.all(mx.isfinite(model.moe_bias_stack())))
    short = model(inputs(3), labels=inputs(3))
    previous = model.moe_bias_stack()
    model.update_moe_biases(short.moe_loads, short.moe_qb_margins)
    current = model.moe_bias_stack()
    for row, gate in enumerate(model._backbone_gates):
        if 7 <= gate.layer_idx <= 10:
            assert bool(mx.array_equal(previous[row], current[row]))


def test_k1_q1_optimizer_trajectory_is_exact():
    old = build(config(ced_recurrent_enabled=False))
    fallback = build(config(ced_recurrent_stride=1, ced_recurrent_rounds=1))
    opts = [optimizers.AdamW(1e-4), optimizers.AdamW(1e-4)]
    for step in range(3):
        x = inputs(8) + step
        for model, opt in zip((old, fallback), opts):
            loss, grads = nn.value_and_grad(model, lambda m: m(x, labels=x).loss)(model)
            mx.eval(loss, grads)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        for (_, a), (_, b) in zip(
            tree_flatten(old.parameters()), tree_flatten(fallback.parameters())
        ):
            assert bool(mx.array_equal(a, b))


def test_real_compiled_trainer_updates_and_checkpoint_roundtrip(tmp_path):
    args = get_pretrain_parser().parse_args(
        [
            "--ced-recurrent",
            "--mtp_depth",
            "0",
            "--out_dir",
            str(tmp_path),
            "--learning_rate",
            "0.0001",
            "--accumulation_steps",
            "2",
            "--max_steps",
            "4",
            "--cache_limit_gb",
            "0",
            "--max_seq_len",
            "16",
            "--log_interval",
            "1",
        ]
    )
    args.save_dir = str(tmp_path)
    args.warmup_iters = 0
    args.compile_model = True
    model = build(config(qb_stats_rows=12))
    convert_model_dtype(model, "bfloat16")
    tr = BaseTrainer(
        args, model, SimpleNamespace(pad_token_id=0), model.config, "pretrain"
    )
    for gate in model._backbone_gates:
        assert gate.qb_stats_rows == (2 if 7 <= gate.layer_idx <= 10 else 6)
    x = inputs(16)
    before = model.model.layers[7].attn.wq_a.weight
    sample = (x, (x + 1) % model.config.vocab_size, mx.ones_like(x), mx.zeros_like(x))
    tr._run_epoch_steps(iter([sample] * 4), 0, 4, 4, None, 0, time.time(), 0)
    assert max_abs_diff(before, model.model.layers[7].attn.wq_a.weight) > 0
    assert bool(mx.all(mx.isfinite(model.moe_bias_stack())))
    saved = tmp_path / "pretrain_64.safetensors"
    assert saved.exists()
    meta = json.loads(saved.with_suffix(".json").read_text())
    assert meta["execution"]["kind"] == "residual_lift_v1"
    assert "model/recurrent.py" in meta["source_sha256"]
    other = build(VibyConfig.from_dict(meta["config"]))
    convert_model_dtype(other, "bfloat16")
    args.reset_optimizer = True
    load_checkpoint(str(saved), other, optimizers.AdamW(1e-4), args)
    model.eval()
    other.eval()
    assert bool(mx.array_equal(model(x).logits, other(x).logits))


def test_nominal_matrix_estimate_respects_direct_fallback_and_short_capacity():
    old = build(config(ced_recurrent_enabled=False))
    recurrent = build(config())
    fallback = build(config(ced_recurrent_stride=1, ced_recurrent_rounds=1))
    assert training_flops_per_token(old, 16) == training_flops_per_token(fallback, 16)
    assert (
        0 < training_flops_per_token(recurrent, 16) < training_flops_per_token(old, 16)
    )
    assert 0 < training_flops_per_token(recurrent, 3) < training_flops_per_token(old, 3)
