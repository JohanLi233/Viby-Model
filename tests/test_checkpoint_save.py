"""save_checkpoint 必须在写优化器之前落 sidecar，避免中断后 eval 无配置。"""

import json
import os as _os
import sys as _sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import mlx.core as mx

from trainer.utils import save_checkpoint


class _FakeModel:
    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        return {"w": mx.zeros((2,), dtype=mx.float32)}


class _FakeOpt:
    learning_rate = 1e-8
    state = {"m": mx.zeros((2,), dtype=mx.float32)}


class _FakeCfg:
    hidden_size = 8

    def to_dict(self):
        return {"hidden_size": 8, "n_routed_experts": 4}


class TestSaveCheckpointSidecar(unittest.TestCase):
    def test_sidecar_written_if_optimizer_save_fails(self):
        def fake_save(path, _weights):
            if str(path).endswith(".optimizer.safetensors"):
                raise RuntimeError("optimizer write failed")

        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(save_dir=td, no_save=False, save_interval=0)
            with patch("trainer.utils.mx.save_safetensors", side_effect=fake_save):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(
                        _FakeModel(),
                        _FakeOpt(),
                        0,
                        3,
                        args,
                        _FakeCfg(),
                        "dpo",
                    )
            meta_path = _os.path.join(td, "dpo_8.json")
            self.assertTrue(
                _os.path.exists(meta_path), "optimizer 失败后 sidecar 必须已在"
            )
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["training_type"], "dpo")
            self.assertEqual(meta["step"], 3)
            self.assertEqual(meta["config"]["n_routed_experts"], 4)
            latest = _os.path.join(td, "latest_checkpoint.txt")
            self.assertTrue(_os.path.exists(latest))
            with open(latest, encoding="utf-8") as f:
                self.assertIn("dpo_8.safetensors", f.read())


if __name__ == "__main__":
    unittest.main()
