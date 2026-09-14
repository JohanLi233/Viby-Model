"""CPU-sized gates for recurrent CED configuration and execution identity."""

import json
import sys
from types import SimpleNamespace

import pytest

from model.config import VibyConfig
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import (
    build_model_kwargs,
    checkpoint_execution,
    validate_checkpoint_execution,
)


def config(**changes):
    values = dict(ced_recurrent_enabled=True, ncp_enabled=False, n_mtp_layers=0)
    values.update(changes)
    return VibyConfig(**values)


def checkpoint(tmp_path, cfg=None, execution=None):
    path = tmp_path / "weights.safetensors"
    metadata = {}
    if cfg is not None:
        metadata["config"] = cfg.to_dict()
    if execution is not None:
        metadata["execution"] = execution
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path


def test_actual_cli_selects_separate_ced_experiment(monkeypatch, tmp_path):
    argv = [
        "--ced-recurrent",
        "--mtp_depth",
        "0",
        "--out_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", *argv])
    args = setup_training_args(get_pretrain_parser().parse_args(argv))
    cfg = VibyConfig(**build_model_kwargs(args))
    assert cfg.ced_recurrent_enabled
    assert (cfg.ced_recurrent_stride, cfg.ced_recurrent_rounds) == (4, 3)
    assert cfg.n_mtp_layers == 0
    assert cfg.n_encoder_layers == 6
    assert VibyConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"ced_recurrent_stride": 0}, "positive"),
        ({"ced_recurrent_rounds": 0}, "positive"),
        ({"n_mtp_layers": 1}, "mtp_depth"),
        ({"n_layers": 4}, "middle decoder"),
        ({"kv_source_layers": (0, 3)}, "CED boundary"),
        ({"index_source_layers": (0, 3)}, "CED boundary"),
        ({"compress_ratios": (4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1)}, "CED boundary"),
        ({"kv_source_layers": (0, 3, 6, 8)}, "no new KV source"),
        ({"kv_source_layers": (0, 3, 6, 11)}, "no new KV source"),
        ({"compress_ratios": (4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 4)}, "no new KV source"),
        ({"engram_layer_ids": (1, 7)}, "Engram"),
    ],
)
def test_config_rejects_unsupported_execution(changes, message):
    with pytest.raises(ValueError, match=message):
        config(**changes)


def test_matching_recurrent_checkpoint_resumes_without_conversion(tmp_path):
    cfg = config()
    path = checkpoint(tmp_path, cfg, checkpoint_execution(cfg))
    args = SimpleNamespace(reset_optimizer=False, auto_resume=True)
    assert not validate_checkpoint_execution(path, cfg, args, automatic=True)


@pytest.mark.parametrize(
    "source,target",
    [
        (config(ced_recurrent_enabled=False), config()),
        (config(), config(ced_recurrent_enabled=False)),
        (config(), config(ced_recurrent_stride=2)),
        (config(), config(ced_recurrent_rounds=2)),
    ],
)
def test_execution_changes_require_explicit_optimizer_reset(tmp_path, source, target):
    path = checkpoint(tmp_path, source)
    args = SimpleNamespace(reset_optimizer=False, auto_resume=False)
    with pytest.raises(ValueError, match="reset_optimizer"):
        validate_checkpoint_execution(path, target, args)
    args.reset_optimizer = True
    assert validate_checkpoint_execution(path, target, args)
    with pytest.raises(ValueError, match="Automatic resume"):
        validate_checkpoint_execution(path, target, args, automatic=True)
    args.auto_resume = True
    with pytest.raises(ValueError, match="Automatic resume"):
        validate_checkpoint_execution(path, target, args)


def test_missing_metadata_is_not_a_recurrent_resume(tmp_path):
    path = tmp_path / "legacy.safetensors"
    args = SimpleNamespace(reset_optimizer=False, auto_resume=False)
    assert not validate_checkpoint_execution(
        path, config(ced_recurrent_enabled=False), args
    )
    with pytest.raises(ValueError, match="execution"):
        validate_checkpoint_execution(path, config(), args)


def test_explicit_execution_version_is_not_inferred_from_weight_shapes(tmp_path):
    cfg = config()
    path = checkpoint(
        tmp_path, cfg, dict(checkpoint_execution(cfg), kind="residual_lift_future")
    )
    with pytest.raises(ValueError, match="execution"):
        validate_checkpoint_execution(path, cfg, SimpleNamespace(reset_optimizer=False))
