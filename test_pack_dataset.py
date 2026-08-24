"""预训练 jsonl 打包缓存：与旧实现口径对拍，且 doc_mask 只 tokenize 一遍。"""

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


if __name__ == "__main__":
    unittest.main()
