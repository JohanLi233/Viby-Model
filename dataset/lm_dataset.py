import json
from torch.utils.data import Dataset
import torch
import os
from typing import Optional, List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(
        self, data_path, tokenizer, max_length=1024, cache_size: Optional[int] = 1000
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.cache_size = cache_size
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._line_offsets = self._build_line_index()

    def _build_line_index(self) -> List[int]:
        """Build an index of line offsets for fast random access"""
        offsets = []
        with open(self.data_path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _get_line_at_offset(self, offset: int) -> str:
        """Read a specific line using its byte offset (binary-safe)."""
        with open(self.data_path, "rb") as f:
            f.seek(offset)
            return f.readline().decode("utf-8", errors="ignore").strip()

    def _load_sample(self, index: int) -> Dict[str, Any]:
        """Load a single sample with caching"""
        if index in self._cache:
            return self._cache[index]

        # Read the line at the given index
        offset = self._line_offsets[index]
        line = self._get_line_at_offset(offset)
        sample = json.loads(line)

        # Cache management: simple LRU-like behavior
        if self.cache_size and len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        if self.cache_size:
            self._cache[index] = sample

        return sample

    def __len__(self):
        return len(self._line_offsets)

    def __getitem__(self, index):
        sample = self._load_sample(index)

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(
        self, jsonl_path, tokenizer, max_length=2048, cache_size: Optional[int] = 1000
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = jsonl_path
        self.cache_size = cache_size
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._line_offsets = self._build_line_index()
        self.bos_id = tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def _build_line_index(self) -> List[int]:
        """Build an index of line offsets for fast random access"""
        offsets = []
        with open(self.data_path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _get_line_at_offset(self, offset: int) -> str:
        """Read a specific line using its byte offset (binary-safe)."""
        with open(self.data_path, "rb") as f:
            f.seek(offset)
            return f.readline().decode("utf-8", errors="ignore").strip()

    def _load_sample(self, index: int) -> Dict[str, Any]:
        """Load a single sample with caching"""
        if index in self._cache:
            return self._cache[index]

        # Read the line at the given index
        offset = self._line_offsets[index]
        line = self._get_line_at_offset(offset)
        sample = json.loads(line)

        # Cache management: simple LRU-like behavior
        if self.cache_size and len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        if self.cache_size:
            self._cache[index] = sample

        return sample

    def __len__(self):
        return len(self._line_offsets)

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < n:
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 仅在未超出实际序列长度的范围内打掩码；若未找到eos，则掩码到序列末尾
                upper = min(end + len(self.eos_id) + 1, n)
                for j in range(start + 1, upper):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < n else n
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self._load_sample(index)
        # 构建对话提示
        prompt = self._create_chat_prompt(sample["conversations"])
        # 先在未填充的序列上生成 loss mask，避免将 PAD 区域计入损失
        input_ids_trunc = self.tokenizer(prompt).input_ids[: self.max_length]
        loss_mask = self._generate_loss_mask(input_ids_trunc)
        # 对 input_ids 与 mask 分别进行填充对齐
        input_ids = input_ids_trunc + [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids_trunc)
        )
        loss_mask += [0] * (self.max_length - len(loss_mask))

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask
