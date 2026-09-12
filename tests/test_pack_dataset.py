"""预训练 jsonl 打包缓存：与旧实现口径对拍，且 doc_mask 只 tokenize 一遍。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import json
import os
import tempfile
import unittest

import numpy as np

from dataset.lm_dataset import pack_pretrain_jsonl


class CharTokenizer:
    """确定性假 tokenizer：每个字符一个 id，便于对拍。"""

    eos_token_id = 99

    def get_vocab(self):
        return {chr(i): i for i in range(128)} | {"<eos>": 99}

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) for c in texts]}
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


class CountingTokenizer(CharTokenizer):
    def __init__(self):
        self.n_texts = 0
        self.n_calls = 0

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        self.n_calls += 1
        if isinstance(texts, str):
            self.n_texts += 1
        else:
            self.n_texts += len(texts)
        return super().__call__(texts, add_special_tokens=add_special_tokens, **kwargs)


def _write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def _reference_pack(path, tokenizer, max_length, with_segs=False):
    """旧实现口径：整表读入、一次 tokenize、余数丢弃。"""
    texts = []
    with open(path, "rb") as f:
        for line in f:
            texts.append(
                json.loads(line.decode("utf-8", errors="ignore").strip())["text"]
            )
    all_ids = []
    all_segs = []
    eos = tokenizer.eos_token_id
    for doc_id, enc in enumerate(
        tokenizer(texts, add_special_tokens=False)["input_ids"]
    ):
        all_ids.extend(enc)
        all_ids.append(eos)
        if with_segs:
            all_segs.extend([doc_id] * (len(enc) + 1))
    width = max_length + 1
    arr = np.asarray(all_ids, dtype=np.int32)
    n_blocks = len(arr) // width
    packed = arr[: n_blocks * width].reshape(n_blocks, width)
    segs = None
    if with_segs:
        s = np.asarray(all_segs, dtype=np.int32)
        segs = s[: n_blocks * width].reshape(n_blocks, width)
    return packed, segs


def _reference_align_pack(
    path, tokenizer, max_length, max_doc_len=None, with_segs=False
):
    """文档边界对齐参考实现：单篇先按 max_doc_len 截断再补 eos，块首落在文档
    开头，跨块尾部丢弃（与 _raw_int32_to_npy_aligned 相同口径）。"""
    import bisect

    texts = []
    with open(path, "rb") as f:
        for line in f:
            texts.append(
                json.loads(line.decode("utf-8", errors="ignore").strip())["text"]
            )
    cap = max_length if max_doc_len is None else min(max_doc_len, max_length)
    eos = tokenizer.eos_token_id
    all_ids = []
    all_segs = []
    starts = [0]
    cur = 0
    for doc_id, enc in enumerate(
        tokenizer(texts, add_special_tokens=False)["input_ids"]
    ):
        e = enc[:cap] + [eos]
        all_ids.extend(e)
        if with_segs:
            all_segs.extend([doc_id] * len(e))
        cur += len(e)
        if doc_id > 0:
            starts.append(cur - len(e))
    starts.append(cur)  # 末尾哨兵 = 总 token 数
    arr = np.asarray(all_ids, dtype=np.int32)
    segarr = np.asarray(all_segs, dtype=np.int32) if with_segs else None
    width = max_length + 1
    pos = 0
    blocks = []
    segblocks = []
    while pos + width <= len(arr):
        blocks.append(arr[pos : pos + width])
        if segarr is not None:
            segblocks.append(segarr[pos : pos + width])
        k = bisect.bisect_left(starts, pos + width)
        if k >= len(starts):
            break
        pos = int(starts[k])
    packed = np.stack(blocks) if blocks else np.zeros((0, width), dtype=np.int32)
    segs = np.stack(segblocks) if (with_segs and segblocks) else None
    return packed, segs


class PackPretrainTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = self._td.name
        self._old_batch = os.environ.pop("VIBY_PACK_BATCH", None)
        self._old_cache = os.environ.pop("VIBY_PACKED_CACHE", None)
        self._old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"

    def tearDown(self):
        self._td.cleanup()
        if self._old_batch is None:
            os.environ.pop("VIBY_PACK_BATCH", None)
        else:
            os.environ["VIBY_PACK_BATCH"] = self._old_batch
        if self._old_cache is None:
            os.environ.pop("VIBY_PACKED_CACHE", None)
        else:
            os.environ["VIBY_PACKED_CACHE"] = self._old_cache
        if self._old_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = self._old_tqdm

    def test_matches_reference_and_drops_remainder(self):
        texts = ["abc", "de", "fghij", "k", "zzz"]
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, texts)
        tok = CharTokenizer()
        max_length = 4
        want_ids, want_segs = _reference_pack(jsonl, tok, max_length, with_segs=True)
        self.assertGreater(len(want_ids), 0)
        # 余数：总 token 不能被 width 整除
        total = sum(len(t) + 1 for t in texts)
        self.assertNotEqual(total % (max_length + 1), 0)

        packed_path = os.path.join(self.td, "packed_t.npy")
        segs_path = os.path.join(self.td, "packedsegs_t.npy")
        got_ids, got_segs = pack_pretrain_jsonl(
            jsonl,
            tok,
            max_length,
            packed_path,
            segs_path,
            batch_size=2,
        )
        np.testing.assert_array_equal(np.asarray(got_ids), want_ids)
        np.testing.assert_array_equal(np.asarray(got_segs), want_segs)

    def test_doc_mask_tokenizes_each_doc_once(self):
        texts = ["hello", "world", "foo", "bar", "baz"]
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, texts)
        tok = CountingTokenizer()
        pack_pretrain_jsonl(
            jsonl,
            tok,
            max_length=8,
            packed_path=os.path.join(self.td, "packed_t.npy"),
            segs_path=os.path.join(self.td, "packedsegs_t.npy"),
            batch_size=2,
        )
        self.assertEqual(tok.n_texts, len(texts))

    def test_reload_hits_cache_without_retokenizing(self):
        texts = ["aa", "bb", "cc"]
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, texts)
        packed_path = os.path.join(self.td, "packed_t.npy")
        segs_path = os.path.join(self.td, "packedsegs_t.npy")
        pack_pretrain_jsonl(
            jsonl,
            CharTokenizer(),
            8,
            packed_path,
            segs_path,
            batch_size=1,
        )
        tok = CountingTokenizer()
        pack_pretrain_jsonl(jsonl, tok, 8, packed_path, segs_path, batch_size=1)
        self.assertEqual(tok.n_texts, 0)

    def test_hf_backend_matches_tokenizer_call(self):
        from transformers import AutoTokenizer

        from dataset.lm_dataset import _encode_text_batch

        tok = AutoTokenizer.from_pretrained("./model/")
        texts = ["你好世界", "abc 123", "秋日清晨"]
        want = tok(texts, add_special_tokens=False)["input_ids"]
        got = _encode_text_batch(tok, texts)
        self.assertEqual([list(x) for x in got], [list(x) for x in want])

    def test_no_segs_when_not_requested(self):
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, ["abc", "defg"])
        packed_path = os.path.join(self.td, "packed_t.npy")
        packed, segs = pack_pretrain_jsonl(
            jsonl, CharTokenizer(), 8, packed_path, segs_path=None, batch_size=8
        )
        self.assertIsNone(segs)
        self.assertTrue(os.path.exists(packed_path))
        self.assertFalse(os.path.exists(os.path.join(self.td, "packedsegs_t.npy")))
        self.assertEqual(packed.shape[1], 9)

    def test_doc_align_matches_reference_and_starts_on_boundary(self):
        texts = ["abc", "de", "fghij", "k", "zzzqqq", "hello world"]
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, texts)
        tok = CharTokenizer()
        width = 5
        for max_doc_len in (None, 3):
            packed_path = os.path.join(self.td, f"packed_a{max_doc_len}.npy")
            segs_path = os.path.join(self.td, f"packedsegs_a{max_doc_len}.npy")
            got_ids, got_segs = pack_pretrain_jsonl(
                jsonl,
                tok,
                max_length=width - 1,
                packed_path=packed_path,
                segs_path=segs_path,
                batch_size=2,
                align_docs=True,
                max_doc_len=max_doc_len,
            )
            want_ids, want_segs = _reference_align_pack(
                jsonl, tok, width - 1, max_doc_len, with_segs=True
            )
            np.testing.assert_array_equal(np.asarray(got_ids), want_ids)
            np.testing.assert_array_equal(np.asarray(got_segs), want_segs)

            # 每个块的首 token 必须是某篇文档的开头：块间 seg 值从上一个块的
            # 末 token 跳到下一个块的 0 位时必然发生变化（对齐后不会跨块续篇）。
            gs = np.asarray(got_segs)
            self.assertGreater(len(gs), 0)
            for b in range(1, gs.shape[0]):
                self.assertNotEqual(gs[b, 0], gs[b - 1, -1])

    def test_doc_align_caps_doc_length(self):
        texts = ["a" * 40, "b" * 10, "c" * 3]
        jsonl = os.path.join(self.td, "d.jsonl")
        _write_jsonl(jsonl, texts)
        tok = CharTokenizer()
        max_length = 8
        max_doc_len = 12
        packed_path = os.path.join(self.td, "packed_cap.npy")
        got_ids, got_segs = pack_pretrain_jsonl(
            jsonl,
            tok,
            max_length,
            packed_path=packed_path,
            segs_path=os.path.join(self.td, "packedsegs_cap.npy"),
            batch_size=2,
            align_docs=True,
            max_doc_len=max_doc_len,
        )
        # 长文档被截到 max_doc_len（含 eos 后每篇最长 max_doc_len+1），
        # 而非原始 40 字符全部进入流。参考实现应与之一致。
        want_ids, _ = _reference_align_pack(
            jsonl, tok, max_length, max_doc_len, with_segs=False
        )
        np.testing.assert_array_equal(np.asarray(got_ids), want_ids)


if __name__ == "__main__":
    unittest.main()
