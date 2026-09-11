"""Actual default pretraining integration, causal targets and checkpoint upgrade."""

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx import optimizers
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from test_psr import config
from model.config import VibyConfig
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import build_model_kwargs, load_checkpoint
from trainer.base_trainer import BaseTrainer
from trainer.psr_pretrain import text_psr_inputs
from trainer.flops import gemm_active_params, psr_pretrain_flops_per_token


def parsed(monkeypatch, *extra):
    argv = ["--no_save", *extra]
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
    args = setup_training_args(get_pretrain_parser().parse_args(argv))
    return args, VibyConfig(**build_model_kwargs(args))


def test_exact_user_command_enables_psr_and_preserves_no_save(monkeypatch):
    args, cfg = parsed(monkeypatch, "--pack_sequences", "--doc_mask", "--batch_size", "16",
                       "--accumulation_steps", "2", "--max_seq_len", "1024", "--log_interval", "1",
                       "--seed", "1337", "--use_swanlab", "--auto_resume")
    assert cfg.psr_enabled and cfg.psr_rounds == 4 and cfg.psr_bridge_init == 0.05
    assert cfg.psr_test_classes == cfg.vocab_size
    assert cfg.n_mtp_layers == 0
    assert not args.auto_resume and not args.use_swanlab
    _, tiny = parsed(monkeypatch, "--preset", "tiny", "--vocab_size", "32")
    assert tiny.psr_enabled and tiny.psr_test_classes == 32 and tiny.psr_bridge_init == 0.05
    _, off = parsed(monkeypatch, "--preset", "tiny", "--no-psr")
    assert not off.psr_enabled


def text_batch():
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 0, 0]])
    y = mx.array([[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 9, 0, 0]])
    segments = mx.array([[0, 0, 1, 1, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0, -1, -1]])
    mask = mx.array([[1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]])
    return x, y, mask, x != 0, segments


def test_text_boundary_and_targets_are_causal_and_doc_local():
    cfg = config(psr_bridge_init=0.05)
    x, y, mask, pad, seg = text_batch()
    kw = text_psr_inputs(cfg, x, y, mask, pad, seg)
    assert kw["thinking_prefix_lengths"].tolist() == [4, 3]
    assert kw["thinking_targets"]["terminal_results"].tolist() == [[5, 6, -1, -1], [4, 5, 6, 9]]
    assert "results" not in kw["thinking_targets"]
    model = build(cfg)
    a = model(x, segment_ids=seg, attention_mask=pad, return_thinking=True, **kw)
    changed_x = mx.where(mx.arange(8)[None, :] >= mx.array([4, 3])[:, None], x + 10, x)
    changed_kw = text_psr_inputs(cfg, changed_x, y + 1, mask, pad, seg)
    b = model(changed_x, segment_ids=seg, attention_mask=pad, return_thinking=True, **changed_kw)
    assert max_abs_diff(a.thinking_state.slots, b.thinking_state.slots) == 0
    assert abs(float(a.loss - b.loss)) > 1e-5
    assert a.psr_losses["index_distillation"] is not None
    assert "address" not in a.psr_losses  # No invented oracle for natural text.
    assert "value" not in a.psr_losses


def test_real_trainer_compiled_loss_updates_workspace_and_indexer():
    cfg = config(psr_bridge_init=0.05)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model, trainer.lm_config = build(cfg), cfg
    trainer.training_type = "pretrain"
    trainer.args = SimpleNamespace(accumulation_steps=2, compile_model=True)
    trainer._expl_nest_mu, trainer._en_delta = 0.0, None
    trainer._loss_and_grad = trainer._build_loss_and_grad()
    batch = text_batch()
    output, gradients = trainer._compute_loss_and_grad(*batch)
    mx.eval(output, gradients)
    assert bool(mx.isfinite(output[0]))
    flat = dict(tree_flatten(gradients))
    for key in ("model.reasoner.index_query.weight", "model.reasoner.index_weights.weight",
                "model.reasoner.blocks.0.up.weight", "model.reasoner.test_head.weight",
                "model.workspace_bridges.0.out.weight"):
        assert bool(mx.all(mx.isfinite(flat[key]))) and float(mx.max(mx.abs(flat[key]))) > 0, key
    opt = optimizers.Adam(learning_rate=1e-3)
    before = trainer.model.model.reasoner.blocks[0].up.weight
    opt.update(trainer.model, gradients)
    mx.eval(trainer.model.parameters())
    assert max_abs_diff(before, trainer.model.model.reasoner.blocks[0].up.weight) > 0
    after, _ = trainer._compute_loss_and_grad(*batch)
    assert abs(float(output[0] - after[0])) > 1e-5


def test_no_valid_tokens_remains_finite():
    cfg = config()
    x = mx.zeros((1, 4), mx.int32)
    kw = text_psr_inputs(cfg, x, x, x, x)
    assert kw["thinking_prefix_lengths"].tolist() == [0]
    out = build(cfg)(x, labels=x, loss_mask=x, attention_mask=x, **kw)
    assert bool(mx.isfinite(out.loss))


def test_upgrade_only_allows_completely_new_psr_and_resets_progress(tmp_path):
    base = build(config(psr_enabled=False))
    new = build(config())
    checkpoint = tmp_path / "base.safetensors"
    weights = dict(tree_flatten(base.parameters()))
    mx.save_safetensors(str(checkpoint), weights)
    checkpoint.with_suffix(".json").write_text(json.dumps({"epoch": 4, "step": 99, "config": base.config.to_dict()}))
    args = SimpleNamespace(freeze_backbone=False, reset_optimizer=False)
    opt = optimizers.Adam(learning_rate=1e-3)
    initial = new.model.reasoner.slots
    assert load_checkpoint(str(checkpoint), new, opt, args) == (0, 0)
    assert max_abs_diff(base.model.embed.weight, new.model.embed.weight) == 0
    assert max_abs_diff(initial, new.model.reasoner.slots) == 0
    # A partly present PSR checkpoint is corruption, not a base-model upgrade.
    weights["model.reasoner.slots"] = initial
    mx.save_safetensors(str(checkpoint), weights)
    with pytest.raises(ValueError, match="缺少"):
        load_checkpoint(str(checkpoint), new, opt, args)
    # Complete PSR resumes its original training progress.
    mx.save_safetensors(str(checkpoint), dict(tree_flatten(new.parameters())))
    assert load_checkpoint(str(checkpoint), new, opt, args) == (4, 100)


def test_psr_flops_amortize_shared_rounds_without_counting_unused_heads():
    off = build(config(psr_enabled=False))
    on = build(config())
    assert gemm_active_params(off) == gemm_active_params(on)
    assert psr_pretrain_flops_per_token(off.config, 64) == 0
    f2 = psr_pretrain_flops_per_token(on.config, 64)
    on.config.psr_rounds = 4
    assert psr_pretrain_flops_per_token(on.config, 64) > f2 > 0
