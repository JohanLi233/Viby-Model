"""Data-layout checks run without importing MLX or using the GPU."""

import unittest
from types import SimpleNamespace

from trainer.resume_layout import validate_resume_layout


class ResumeLayoutTests(unittest.TestCase):
    def test_same_layout_and_missing_legacy_fields(self):
        validate_resume_layout(
            {"args": {"batch_size": 14}}, SimpleNamespace(batch_size=14)
        )
        validate_resume_layout({}, SimpleNamespace(batch_size=14))

    def test_batch_change_cannot_silently_replay_samples(self):
        with self.assertRaisesRegex(ValueError, "batch_size.*sample/token-cursor"):
            validate_resume_layout(
                {"args": {"batch_size": 16}}, SimpleNamespace(batch_size=14)
            )
        self.assertEqual(10000 * 16 - 10000 * 14, 20000)

    def test_other_data_layout_changes(self):
        for key, old, new in [
            ("max_seq_len", 1024, 512),
            ("seed", 1337, 42),
            ("doc_mask", True, False),
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resume_layout(
                    {"args": {key: old}}, SimpleNamespace(**{key: new})
                )

    def test_path_alias_and_explicit_warm_start(self):
        validate_resume_layout(
            {"args": {"data_path": "./data.jsonl"}},
            SimpleNamespace(data_path="data.jsonl"),
        )
        validate_resume_layout(
            {"args": {"batch_size": 16}},
            SimpleNamespace(batch_size=14, reset_optimizer=True),
        )
        validate_resume_layout(
            {"args": {"batch_size": 16}},
            SimpleNamespace(batch_size=14, freeze_backbone=True),
        )


if __name__ == "__main__":
    unittest.main()
