"""SFT jsonl 打包：与逐样本口径对拍、边界 mask、缓存复用、segs 路径回退。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import json
import os
import random
import tempfile
import unittest

import numpy as np

from dataset.lm_dataset import (
    SFTDataset,
    _assistant_loss_mask,
    _render_chat_prompt,
    pack_sft_jsonl,
    post_processing_chat,
    pre_processing_chat,
)


def _write_sft_jsonl(path, conversations_list):
    with open(path, "w", encoding="utf-8") as f:
        for convs in conversations_list:
            f.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")


def _sample_ids_mask(tok, convs):
    """逐样本口径：渲染 -> 分词 -> assistant mask。ratio=0.0 与打包默认对齐。"""
    prompt = post_processing_chat(
        _render_chat_prompt(tok, pre_processing_chat(convs)), 0.0
    )
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    bos_id = tok(f"{tok.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tok(f"{tok.eos_token}\n", add_special_tokens=False).input_ids
    return ids, _assistant_loss_mask(ids, bos_id, eos_id)


class PackSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained("./model/")

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = self._td.name
        self._old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        random.seed(0)
        # 固定 system 首轮，避免 pre_processing_chat 的随机补 system
        self.convs = [
            [
                {"role": "system", "content": "你是minimind。"},
                {"role": "user", "content": "介绍一下万有引力。"},
                {"role": "assistant", "content": "万有引力是物体之间相互吸引的力。"},
            ],
            [
                {"role": "system", "content": "你是minimind。"},
                {"role": "user", "content": "杭州有什么美食？"},
                {"role": "assistant", "content": "西湖醋鱼、东坡肉。"},
                {"role": "user", "content": "还有呢？"},
                {"role": "assistant", "content": "龙井虾仁、片儿川。"},
            ],
            [
                {"role": "system", "content": "你是minimind。"},
                {"role": "user", "content": "讲一个很长的故事。"},
                {"role": "assistant", "content": "从前有座山，" * 200},
            ],
        ]

    def tearDown(self):
        self._td.cleanup()
        if self._old_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = self._old_tqdm

    def _paths(self):
        return (
            os.path.join(self.td, "sftpacked_t.npy"),
            os.path.join(self.td, "sftpackedmasks_t.npy"),
            os.path.join(self.td, "sftpackedsegs_t.npy"),
        )

    def test_pack_matches_per_sample_stream(self):
        jsonl = os.path.join(self.td, "s.jsonl")
        _write_sft_jsonl(jsonl, self.convs)
        max_length = 64
        p, m, s = self._paths()
        packed, masks, segs = pack_sft_jsonl(
            jsonl, self.tok, max_length, p, m, s, batch_size=2
        )

        want_ids, want_masks, want_segs = [], [], []
        for i, convs in enumerate(self.convs):
            ids, mask = _sample_ids_mask(self.tok, convs)
            want_ids.extend(ids)
            want_masks.extend(mask)
            want_segs.extend([i] * len(ids))
        width = max_length + 1
        n_blocks = len(want_ids) // width
        self.assertGreater(n_blocks, 1)  # 长样本确实跨块

        flat_ids = np.asarray(packed).reshape(-1)[: n_blocks * width]
        flat_masks = np.asarray(masks).reshape(-1)[: n_blocks * width]
        flat_segs = np.asarray(segs).reshape(-1)[: n_blocks * width]
        np.testing.assert_array_equal(flat_ids, want_ids[: n_blocks * width])
        np.testing.assert_array_equal(flat_masks, want_masks[: n_blocks * width])
        np.testing.assert_array_equal(flat_segs, want_segs[: n_blocks * width])
        self.assertGreater(int(np.asarray(masks).sum()), 0)

    def test_getitem_alignment_and_boundary_mask(self):
        jsonl = os.path.join(self.td, "s.jsonl")
        _write_sft_jsonl(jsonl, self.convs)
        ds = SFTDataset(
            jsonl, self.tok, max_length=64, pack_sequences=True, doc_mask=True
        )
        self.assertEqual(len(ds), len(ds._packed))
        # 找一个跨样本边界的 block
        found = False
        for i in range(len(ds)):
            X, Y, loss_mask, segX = ds[i]
            self.assertEqual(X.shape, Y.shape)
            self.assertEqual(X.shape, loss_mask.shape)
            self.assertEqual(X.shape, segX.shape)
            self.assertEqual(len(X), 64)
            segs = ds._packed_segs[i].astype(np.int64)
            segY = segs[1:]
            boundary = segX != segY
            # 边界位置必须被 mask 掉
            self.assertTrue(np.all(loss_mask[boundary] == 0))
            if boundary.any():
                found = True
        self.assertTrue(found, "测试数据应产生至少一个跨样本边界 block")
        # 监督位置必须与逐样本口径一致（非边界处）
        i = 0
        X, Y, loss_mask, segX = ds[i]
        np.asarray(ds._packed[i])
        masks = np.asarray(ds._packed_masks[i])
        segs = np.asarray(ds._packed_segs[i])
        want = (masks[1:] & (segs[:-1] == segs[1:])).astype(np.int64)
        np.testing.assert_array_equal(loss_mask, want)

    def test_no_segs_returns_3_tuple(self):
        jsonl = os.path.join(self.td, "s.jsonl")
        _write_sft_jsonl(jsonl, self.convs[:2])
        ds = SFTDataset(
            jsonl, self.tok, max_length=32, pack_sequences=True, doc_mask=False
        )
        self.assertGreater(len(ds), 0)
        out = ds[0]
        self.assertEqual(len(out), 3)
        X, Y, loss_mask = out
        np.testing.assert_array_equal(
            loss_mask, np.asarray(ds._packed_masks[0]).astype(np.int64)[1:]
        )

    def test_reload_uses_cache(self):
        jsonl = os.path.join(self.td, "s.jsonl")
        _write_sft_jsonl(jsonl, self.convs)
        p, m, s = self._paths()
        pack_sft_jsonl(jsonl, self.tok, 64, p, m, s, batch_size=2)
        mtimes = (os.path.getmtime(p), os.path.getmtime(m), os.path.getmtime(s))
        pack_sft_jsonl(jsonl, self.tok, 64, p, m, s, batch_size=2)
        self.assertEqual(
            mtimes, (os.path.getmtime(p), os.path.getmtime(m), os.path.getmtime(s))
        )

    def test_segs_path_fallback_when_override_lacks_marker(self):
        from dataset.lm_dataset import PretrainDataset

        override = os.path.join(self.td, "mytokens.npy")
        os.environ["VIBY_PACKED_CACHE"] = override
        try:
            ds = PretrainDataset.__new__(PretrainDataset)
            ds.data_path = "unused"
            ds.tokenizer = None
            ds.max_length = 8
            self.assertEqual(ds._segs_cache_path(), override + ".segs")
        finally:
            os.environ.pop("VIBY_PACKED_CACHE", None)


if __name__ == "__main__":
    unittest.main()
