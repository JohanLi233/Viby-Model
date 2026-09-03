"""eval_model 的 mode / 检查点发现。"""

import os as _os
import sys as _sys
import tempfile
import unittest

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from eval_model import MODEL_MODES, _find_checkpoint, get_prompt_datas


class TestEvalModelModes(unittest.TestCase):
    def test_model_mode_2_is_dpo(self):
        self.assertEqual(MODEL_MODES[2], "dpo")
        self.assertEqual(set(MODEL_MODES), {0, 1, 2})

    def test_find_checkpoint_dpo_ignores_sft_latest(self):
        with tempfile.TemporaryDirectory() as td:
            dpo = _os.path.join(td, "dpo_768.safetensors")
            sft = _os.path.join(td, "full_sft_768.safetensors")
            open(dpo, "wb").close()
            open(sft, "wb").close()
            with open(
                _os.path.join(td, "latest_checkpoint.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(sft)
            self.assertEqual(_find_checkpoint(td, "dpo", 768), dpo)
            self.assertEqual(_find_checkpoint(td, "full_sft", 768), sft)

    def test_dpo_mode_uses_chat_prompts(self):
        class Args:
            model_mode = 2

        prompts = get_prompt_datas(Args())
        self.assertIn("请介绍一下自己。", prompts)
        self.assertNotIn("马克思主义基本原理", prompts)


if __name__ == "__main__":
    unittest.main()
