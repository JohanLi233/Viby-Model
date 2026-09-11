"""PSR logging must support the actual MultiOptimizer with saving enabled."""

import json
import time
from types import SimpleNamespace

import mlx.core as mx
from mlx import optimizers

from _v41_common import build
from test_psr import config, batch
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser
from trainer.utils import get_optimizer_steps


def test_single_nested_multi_and_restored_step_clocks():
    first, second = optimizers.SGD(0.01), optimizers.AdamW(0.01)
    nested = optimizers.MultiOptimizer(
        [optimizers.MultiOptimizer([first]), second], [lambda k, v: k == "a"]
    )
    assert get_optimizer_steps(nested) == [0, 0]
    parameters = {"a": mx.ones((2,)), "b": mx.ones((2,))}
    nested.apply_gradients(parameters, parameters)
    assert get_optimizer_steps(nested) == [1, 1]
    # Per-group counts can differ; reporting must preserve this information.
    second.apply_gradients({"b": mx.ones((2,))}, parameters)
    assert get_optimizer_steps(nested) == [1, 2]
    restored = optimizers.MultiOptimizer(
        [optimizers.SGD(0.01), optimizers.AdamW(0.01)], [lambda k, v: k == "a"]
    )
    restored.state = {"states": [first.state, second.state]}
    assert get_optimizer_steps(restored) == [1, 2]
    assert get_optimizer_steps(second) == [2]


def test_real_training_writes_metrics_before_and_after_accumulation(tmp_path):
    args = get_pretrain_parser().parse_args(
        [
            "--out_dir",
            str(tmp_path),
            "--learning_rate",
            "0.0001",
            "--batch_size",
            "2",
            "--accumulation_steps",
            "2",
            "--max_steps",
            "4",
            "--max_seq_len",
            "8",
            "--cache_limit_gb",
            "0",
        ]
    )
    args.save_dir = str(tmp_path)
    args.compile_model = False
    args.warmup_iters = 0
    assert not args.no_save
    model = build(config())
    tr = BaseTrainer(
        args, model, SimpleNamespace(pad_token_id=0), model.config, "pretrain"
    )
    assert isinstance(tr.optimizer, optimizers.MultiOptimizer)
    x, y, mask, _, segments = batch()
    batches = [(x, y, mask, segments)] * 4
    tr._run_epoch_steps(iter(batches), 0, 4, 4, None, 0, time.time(), 0)
    records = [
        json.loads(line)
        for line in (tmp_path / "psr_metrics.jsonl").read_text().splitlines()
    ]
    assert [r["microstep"] for r in records] == [1, 2, 3, 4]
    assert [r["optimizer_step"] for r in records] == [0, 0, 1, 1]
    assert all(r["phase"] == "pre_update" for r in records)
    assert all(max(r["optimizer_group_steps"]) == r["optimizer_step"] for r in records)
    assert max(get_optimizer_steps(tr.optimizer)) == 2
    assert (tmp_path / "pretrain_64.safetensors").exists()
