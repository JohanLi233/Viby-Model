"""Integration gates for the revised direct-NTP protected training recipe."""

import sys
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from test_psr import pair, trainer, batch
from _v41_common import build, max_abs_diff
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import build_model_kwargs, load_checkpoint, save_checkpoint
from model.config import VibyConfig
from experiments.psr_evaluate import calibrate_gate, paired_bootstrap, evaluate


def test_cli_default_is_dense_short_local_protected_not_old_aux(monkeypatch):
    argv = [
        "--no_save",
        "--pack_sequences",
        "--doc_mask",
        "--batch_size",
        "16",
        "--accumulation_steps",
        "2",
        "--max_seq_len",
        "1024",
        "--use_swanlab",
        "--auto_resume",
    ]
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
    args = setup_training_args(get_pretrain_parser().parse_args(argv))
    cfg = VibyConfig(**build_model_kwargs(args))
    assert cfg.psr_enabled and cfg.psr_rounds == 1 and cfg.psr_horizon == 16
    assert cfg.psr_train_anchors == 2 and args.psr_learning_rate == 1e-4
    assert not hasattr(cfg, "psr_predictive_weight")
    assert not args.use_swanlab and not args.auto_resume
    off = get_pretrain_parser().parse_args(["--no-psr"])
    assert not off.psr_enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_cli_default_and_baseline_reach_compiled_training(monkeypatch, enabled):
    argv = [
        "--preset",
        "tiny",
        "--no_save",
        "--optimizer",
        "adamw",
        "--hidden_size",
        "64",
        "--num_attention_heads",
        "2",
        "--head_dim",
        "32",
        "--rope_head_dim",
        "16",
        "--q_lora_rank",
        "32",
        "--o_lora_rank",
        "32",
        "--o_groups",
        "1",
        "--moe_intermediate_size",
        "32",
        "--n_routed_experts",
        "4",
        "--num_experts_per_tok",
        "2",
        "--engram_layer_ids",
        "",
        "--vocab_size",
        "32",
        "--max_seq_len",
        "16",
        "--batch_size",
        "1",
        "--accumulation_steps",
        "1",
        "--learning_rate",
        "0.0001",
        "--psr_dim",
        "32",
        "--psr_slots",
        "2",
    ]
    if not enabled:
        argv.append("--no-psr")
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
    args = setup_training_args(get_pretrain_parser().parse_args(argv))
    args.compile_model = True
    cfg = VibyConfig(**build_model_kwargs(args))
    assert cfg.psr_enabled == enabled
    model = build(cfg)
    tr = BaseTrainer(args, model, SimpleNamespace(pad_token_id=0), cfg, "pretrain")
    assert (tr.psr_optimizer is not None) == enabled
    samples = tuple(value[:1] for value in batch())
    before = mx.array(model.model.embed.weight)
    out, grads = tr._compute_loss_and_grad(*samples)
    mx.eval(out, grads)
    assert bool(mx.isfinite(out[0]))
    tr._optimizer_step(grads, 1, out[2], moe_qb_margins=out[6])
    assert max_abs_diff(before, model.model.embed.weight) > 0
    if enabled:
        assert float(tr.psr_optimizer.step) == 1
        assert float(mx.max(mx.abs(model.psr.output.weight))) > 0
    else:
        assert model.psr is None


def test_checkpoint_upgrade_preserves_baseline_state_and_side_is_separate(tmp_path):
    base, model = pair()
    a = trainer(base)
    out, grad = a._compute_loss_and_grad(*batch())
    a._optimizer_step(grad, 1, out[2], moe_qb_margins=out[6])
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        no_save=False,
        save_interval=0,
        freeze_backbone=False,
        reset_optimizer=False,
    )
    save_checkpoint(base, a.optimizer, 2, 19, args, base.config)
    b = trainer(model)
    b.optimizer.psr_optimizer = b.psr_optimizer
    ckpt = tmp_path / "pretrain_64.safetensors"
    assert load_checkpoint(str(ckpt), model, b.optimizer, args) == (2, 20)
    sa, sb = (
        dict(tree_flatten(a.optimizer.state)),
        dict(tree_flatten(b.optimizer.state)),
    )
    assert sa.keys() == sb.keys()
    assert all(bool(mx.array_equal(sa[k], sb[k])) for k in sa)
    assert bool(mx.all(model.psr.output.weight == 0))
    out, grad = b._compute_loss_and_grad(*batch())
    b._optimizer_step(grad, 1, out[2], moe_qb_margins=out[6])
    save_checkpoint(model, b.optimizer, 2, 20, args, model.config)
    assert (tmp_path / "pretrain_64.psr_optimizer.safetensors").exists()
    _, other = pair()
    c = trainer(other)
    c.optimizer.psr_optimizer = c.psr_optimizer
    assert load_checkpoint(str(ckpt), other, c.optimizer, args) == (2, 21)
    assert float(c.psr_optimizer.step) == float(b.psr_optimizer.step)


def test_statistics_sum_tokens_and_document_pair_bootstrap():
    # Unequal lengths must not become the mean of two per-document means.
    one = {
        "a": dict(nll_sum=2.0, valid_label_count=1.0),
        "b": dict(nll_sum=8.0, valid_label_count=8.0),
    }
    two = {
        "a": dict(nll_sum=3.0, valid_label_count=1.0),
        "b": dict(nll_sum=16.0, valid_label_count=8.0),
    }
    assert sum(v["nll_sum"] for v in one.values()) / 9 == pytest.approx(10 / 9)
    low, high = paired_bootstrap(one, two, repetitions=100)
    assert low == pytest.approx(-1) and high == pytest.approx(-1)
    _, model = pair()
    model.eval()
    x, y, mask, pad, seg = batch()
    arrays = dict(
        input_ids=np.asarray(x),
        labels=np.asarray(y),
        loss_mask=np.asarray(mask),
        attention_mask=np.asarray(pad),
        segment_ids=np.asarray(seg),
        document_ids=np.asarray(seg) + np.array([[0], [10]]),
    )
    results = evaluate(model, arrays, "recurrent")
    assert results["valid_label_count"] == 15
    assert results["bridge_coverage"] == 1.0
    assert results["nll_sum"] == pytest.approx(results["base_nll_sum"], abs=1e-5)
    assert results["resampling_unit"] == "document"
    empty = evaluate(
        model, {**arrays, "loss_mask": np.zeros_like(arrays["loss_mask"])}, "off"
    )
    assert empty["status"] == "no_valid_labels" and empty["valid_label_count"] == 0


def test_gate_calibration_can_reject_and_is_separate_from_inference():
    base = np.zeros((1, 1, 2))
    labels = np.array([[0]])
    mask = np.ones((1, 1))
    assert calibrate_gate(base, np.array([[[-1.0, 1.0]]]), labels, mask) == 0
    assert calibrate_gate(base, np.array([[[1.0, -1.0]]]), labels, mask) == 1


def test_checkpoint_restores_rng_shuffle_and_side_hyperparameters(tmp_path):
    import random

    _, model = pair()
    tr = trainer(model)
    tr.optimizer.psr_optimizer = tr.psr_optimizer
    np.random.seed(23)
    state = np.random.get_state()
    order = np.random.permutation(31)
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        no_save=False,
        save_interval=0,
        freeze_backbone=False,
        reset_optimizer=False,
        epoch_shuffle_state=(state[0], state[1].tolist(), *state[2:]),
    )
    model.psr.calibration_gate = mx.array(0.37)
    save_checkpoint(model, tr.optimizer, 0, 2, args, model.config)
    expected = mx.random.uniform(shape=(5,))
    py_expected = random.random()
    mx.random.seed(99)
    np.random.seed(99)
    random.seed(99)
    _, other = pair()
    restored = trainer(other)
    restored.optimizer.psr_optimizer = restored.psr_optimizer
    restored.psr_optimizer.weight_decay = 0.9
    load_checkpoint(
        str(tmp_path / "pretrain_64.safetensors"), other, restored.optimizer, args
    )
    assert bool(mx.array_equal(expected, mx.random.uniform(shape=(5,))))
    assert random.random() == py_expected
    assert np.array_equal(order, np.random.permutation(31))
    assert restored.psr_optimizer.weight_decay == tr.psr_optimizer.weight_decay
    assert float(other.psr.calibration_gate) == pytest.approx(0.37)
