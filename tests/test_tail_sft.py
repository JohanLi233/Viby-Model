"""TailSFT: numerical reference, data identity, compiled trainer and resume."""

import copy
import json
import random
import runpy
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from _v41_common import build
from model.config import VibyConfig
from model.model import lm_head_ce
from model.tail_sft import sequence_head_losses, tail_keep_mask, tail_objective
from trainer.base_trainer import BaseTrainer
from trainer.config import get_sft_parser
from trainer.utils import get_optimizer_steps
from trainer.tail_sft import (
    filter_fraction,
    prepare_dataset,
    reference_identity,
    validate_args,
    validate_resume,
)


def test_offset_not_absolute_loss_and_exact_ties():
    means = mx.array([1.0, 3.0, 2.0, 0.0])
    reference = mx.array([1.0, 8.0, 2.0, 100.0])
    counts = mx.array([2.0, 1.0, 7.0, 0.0])
    # Third row has equal offset to first; first loses tie. Empty row excluded.
    assert tail_keep_mask(means, reference, counts, mx.array(0.5)).tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert tail_keep_mask(means, reference, counts, mx.array(0.0)).tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert tail_keep_mask(
        means[:1], reference[:1], counts[:1], mx.array(0.99)
    ).tolist() == [True]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("compile", [False, True])
def test_loss_and_grad_match_masked_full_logits(dtype, compile):
    mx.random.seed(4)
    h = mx.random.normal((4, 7, 8)).astype(dtype)
    w = mx.random.normal((13, 8)).astype(dtype)
    y = mx.random.randint(0, 13, (4, 7))
    mask = mx.array([[1, 1, 0, 0, 0, 0, 0], [1] * 7, [0, 0, 1, 1, 1, 0, 0], [0] * 7])
    # The largest base loss makes row 1 the most improved regardless of length.
    reference = mx.array([0.0, 100.0, 0.0, 0.0])

    def actual(h, w, ref, f):
        sums, counts, zs = sequence_head_losses(h, w, y, mask, 0.01, chunk=5)
        ce, z, _, _ = tail_objective(sums, counts, zs, ref, f)
        return ce + 0.01 * z

    def expected(h, w):
        logits = h @ w.T
        ce = mx.fast.cross_entropy(logits.reshape(-1, 13), y.reshape(-1)).reshape(4, 7)
        chosen = mask * mx.array([1, 0, 1, 0])[:, None]
        z = mx.square(mx.logsumexp(logits.astype(mx.float32), axis=-1))
        return mx.sum((ce + 0.01 * z) * chosen) / mx.sum(chosen)

    fn = mx.value_and_grad(actual, argnums=(0, 1, 2))
    if compile:
        fn = mx.compile(fn)
    loss, (gh, gw, gr) = fn(h, w, reference, mx.array(0.34))
    target, (eh, ew) = mx.value_and_grad(expected, argnums=(0, 1))(h, w)
    mx.eval(loss, gh, gw, gr, target, eh, ew)
    tol = 0.04 if dtype == mx.bfloat16 else 3e-5
    np.testing.assert_allclose(np.asarray(loss), np.asarray(target), atol=tol, rtol=tol)
    np.testing.assert_allclose(
        np.asarray(gh.astype(mx.float32)),
        np.asarray(eh.astype(mx.float32)),
        atol=tol,
        rtol=tol,
    )
    np.testing.assert_allclose(
        np.asarray(gw.astype(mx.float32)),
        np.asarray(ew.astype(mx.float32)),
        atol=tol,
        rtol=tol,
    )
    assert float(mx.max(mx.abs(gr))) == 0
    assert float(mx.max(mx.abs(gh[1]))) == 0
    assert float(mx.max(mx.abs(gh[3]))) == 0
    # Same compiled function must react to a new runtime fraction/reference.
    zero, _ = fn(h, w, reference, mx.array(0.0))
    ce, z = lm_head_ce(h, w, y, mask, z_weight=0.01)
    assert abs(float(zero - ce - 0.01 * z)) < tol
    changed, _ = fn(h, w, mx.array([100.0, 0.0, 0.0, 0.0]), mx.array(0.34))
    assert abs(float(changed - loss)) > 1e-3


def test_empty_target_loss_is_zero_and_finite():
    sums, counts, zs = sequence_head_losses(
        mx.ones((2, 3, 4)),
        mx.ones((7, 4)),
        mx.zeros((2, 3), mx.int32),
        mx.zeros((2, 3)),
    )
    ce, z, keep, stats = tail_objective(sums, counts, zs, mx.zeros((2,)), mx.array(0.5))
    assert float(ce) == float(z) == 0
    assert keep.tolist() == [False, False]
    assert bool(mx.all(mx.isfinite(stats)))


def args_for(tmp_path, *extra):
    args = get_sft_parser().parse_args(
        [
            "--out_dir",
            str(tmp_path),
            "--batch_size",
            "2",
            "--max_seq_len",
            "16",
            "--accumulation_steps",
            "2",
            "--optimizer",
            "adamw",
            "--learning_rate",
            ".0001",
            "--warmup_iters",
            "0",
            "--log_interval",
            "1",
            "--save_interval",
            "2",
            *extra,
        ]
    )
    args.save_dir = str(tmp_path)
    return args


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_fraction(tmp_path, fraction):
    args = args_for(tmp_path)
    args.tail_sft_filter_fraction = fraction
    with pytest.raises(ValueError, match="filter_fraction"):
        validate_args(args)


def test_defaults_schedule_and_packing_guard(tmp_path):
    args = args_for(tmp_path)
    assert args.sft_algorithm == "tail"
    validate_args(args)
    assert filter_fraction(args, 0, 5) == 0.5
    args.tail_sft_schedule = "ramp"
    assert [filter_fraction(args, i, 5) for i in (0, 2, 4)] == [0.0, 0.25, 0.5]
    args.pack_sequences = True
    with pytest.raises(ValueError, match="individual sequences"):
        validate_args(args)
    args.sft_algorithm = "standard"
    validate_args(args)


def tiny_config():
    return VibyConfig(
        dim=64,
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=16,
        rope_head_dim=8,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_inter_dim=32,
        n_mtp_layers=0,
        vocab_size=128,
        max_seq_len=32,
        window_size=8,
        engram_layer_ids=(),
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        qb_stats_rows=8,
        psr_enabled=False,
        aux_balance_loss_weight=0.0,
    )


def test_real_compiled_model_trainer_and_partial_window(tmp_path):
    args = args_for(tmp_path, "--no_save")
    args.compile_model = True
    model = build(tiny_config())
    trainer = BaseTrainer(
        args, model, SimpleNamespace(pad_token_id=0), model.config, "sft"
    )
    assert trainer._compiled
    x = mx.array([[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 11, 12]])
    y = (x + 1) % model.config.vocab_size
    mask = (x != 0).astype(mx.float32)
    model.eval()
    out = model(
        x,
        labels=y,
        loss_mask=mask,
        attention_mask=mask,
        return_sequence_losses=True,
        psr_mode="off",
    )
    ref = out.sequence_losses + mx.array([50.0, 0.0])
    mx.eval(ref)
    model.train()
    before = model.model.embed.weight
    outputs, grads = trainer._compute_loss_and_grad(
        x, y, mask, mask, tail_reference=ref, tail_fraction=mx.array(0.5)
    )
    mx.eval(outputs, grads)
    expected = model(
        x,
        labels=y,
        loss_mask=mask * mx.array([0, 1])[:, None],
        attention_mask=mask,
        psr_mode="off",
    )
    assert abs(float(outputs[3]) * 2 - float(expected.lm_loss)) < 1e-5
    assert outputs[5][:3].tolist() == [2.0, 1.0, 8.0]
    assert all(bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(grads))
    batches = [{"X": x, "Y": y, "loss_mask": mask, "tail_reference": ref}] * 3
    trainer._run_epoch_steps(iter(batches), 0, 3, 3, None, 0, time.time(), 0)
    assert set(get_optimizer_steps(trainer.optimizer)) == {2}  # flush partial window
    assert float(mx.max(mx.abs(before - model.model.embed.weight))) > 0
    assert bool(mx.all(mx.isfinite(model.moe_bias_stack())))


def test_deterministic_sft_preprocessing(tmp_path):
    from transformers import AutoTokenizer
    from dataset.lm_dataset import SFTDataset

    tok = AutoTokenizer.from_pretrained("model")
    path = tmp_path / "sft.jsonl"
    conv = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n你好！"},
    ]
    path.write_text(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    ds = SFTDataset(
        str(path), tok, max_length=96, empty_think_ratio=0.5, deterministic_seed=3
    )
    state = random.getstate()
    first = ds[0]
    assert random.getstate() == state
    for seed in range(6):
        random.seed(seed)
        for a, b in zip(first, ds[0]):
            np.testing.assert_array_equal(a, b)


class ScoreDataset:
    def __init__(self, path):
        self.data_path = str(path)
        path.write_text("fixed samples")
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.deterministic_seed = 1
        self.max_length = 4
        self.empty_think_ratio = 0.0

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return (
            np.array([i + 1] * 4),
            np.ones(4, np.int32),
            np.array([1, 1, 0, 0]) if i != 1 else np.zeros(4),
        )


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, x, labels, loss_mask, **kw):
        self.calls += 1
        assert not self.training and kw["psr_mode"] == "off" and not kw["use_mtp"]
        return SimpleNamespace(
            sequence_losses=x[:, 0].astype(mx.float32),
            sequence_token_counts=mx.sum(loss_mask, axis=1),
        )


def test_reference_cache_identity_reuse_and_resume(tmp_path):
    args = args_for(tmp_path)
    dataset, scorer = ScoreDataset(tmp_path / "data"), Scorer()
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base weights")
    args.tail_sft_cache = str(tmp_path / "reference.npz")
    config = tiny_config()
    wrapped = prepare_dataset(dataset, scorer, config, args, checkpoint)
    assert len(wrapped) == 2 and scorer.calls == 2 and scorer.training
    assert float(wrapped[1]["tail_reference"]) == 3.0
    prepare_dataset(dataset, scorer, config, args, checkpoint)
    assert scorer.calls == 2
    saved = tmp_path / "full_sft_64.json"
    saved.write_text(json.dumps({"training_type": "sft", "args": vars(args)}))
    validate_resume(saved.with_suffix(".safetensors"), args)
    altered = copy.deepcopy(args)
    altered.tail_sft_state["filter_fraction"] = 0.25
    with pytest.raises(ValueError, match="contract changed"):
        validate_resume(saved.with_suffix(".safetensors"), altered)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_dataset(dataset, scorer, config, args, checkpoint)


def test_template_changes_reference_identity(tmp_path):
    args = args_for(tmp_path)
    ds = ScoreDataset(tmp_path / "data")
    path = tmp_path / "base"
    path.write_bytes(b"base")
    first = reference_identity(ds, tiny_config(), args, path)
    ds.tokenizer.chat_template = "changed template"
    assert (
        reference_identity(ds, tiny_config(), args, path)["tokenizer"]
        != first["tokenizer"]
    )


def test_legacy_resume_requires_explicit_new_run(tmp_path):
    args = args_for(tmp_path)
    checkpoint = tmp_path / "old.safetensors"
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"training_type": "sft", "args": {}})
    )
    with pytest.raises(ValueError, match="algorithm changed"):
        validate_resume(checkpoint, args)
    args.reset_optimizer = True
    validate_resume(checkpoint, args)


def test_actual_sft_cli_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("model")
    cfg = tiny_config().to_dict()
    cfg.update(vocab_size=len(tokenizer), max_seq_len=64)
    initial = build(VibyConfig.from_dict(cfg))
    checkpoint = tmp_path / "pretrain_64.safetensors"
    mx.save_safetensors(str(checkpoint), dict(tree_flatten(initial.parameters())))
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"training_type": "pretrain", "config": cfg})
    )
    data = tmp_path / "chat.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "你是助手。"},
                {"role": "user", "content": f"数字 {i} 后面是什么？"},
                {"role": "assistant", "content": f"{i + 1}。"},
            ]
        }
        for i in range(8)
    ]
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    cache = tmp_path / "reference.npz"

    def run(directory, resume=None):
        argv = [
            "trainer/train_full_sft.py",
            "--pretrain_checkpoint",
            str(checkpoint),
            "--data_path",
            str(data),
            "--out_dir",
            str(directory),
            "--max_seq_len",
            "64",
            "--batch_size",
            "2",
            "--accumulation_steps",
            "2",
            "--epochs",
            "1",
            "--learning_rate",
            ".0001",
            "--optimizer",
            "adamw",
            "--dtype",
            "float32",
            "--warmup_iters",
            "0",
            "--save_interval",
            "2",
            "--log_interval",
            "1",
            "--compile_model",
            "--no_swanlab",
            "--tail_sft_cache",
            str(cache),
        ]
        if resume is not None:
            argv.extend(["--resume", str(resume)])
        monkeypatch.setattr(sys, "argv", argv)
        return runpy.run_path("trainer/train_full_sft.py", run_name="__main__")

    original_limit = BaseTrainer._time_limit_exceeded
    # Stop at a completed update without changing the planned schedule.
    monkeypatch.setattr(BaseTrainer, "_time_limit_exceeded", lambda self: True)
    first = run(tmp_path / "resumed")
    saved = tmp_path / "resumed/full_sft_64.safetensors"
    meta = json.loads(saved.with_suffix(".json").read_text())
    assert meta["step"] == 1 and meta["args"]["sft_algorithm"] == "tail"
    assert meta["rng"]["numpy_epoch_start"] is not None
    assert len(first["train_ds"]) == 8
    reference_state = meta["args"]["tail_sft_state"]
    cache_mtime = cache.stat().st_mtime_ns
    monkeypatch.setattr(BaseTrainer, "_time_limit_exceeded", original_limit)
    resumed = run(tmp_path / "resumed", saved)
    complete = run(tmp_path / "complete")
    assert cache.stat().st_mtime_ns == cache_mtime
    assert resumed["args"].tail_sft_state == reference_state
    assert set(get_optimizer_steps(resumed["trainer"].optimizer)) == {2}
    actual = dict(tree_flatten(resumed["model"].parameters()))
    expected = dict(tree_flatten(complete["model"].parameters()))
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            np.asarray(expected[key]),
            atol=2e-6,
            rtol=2e-5,
            err_msg=key,
        )
