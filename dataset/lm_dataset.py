import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Iterable, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 线程本地文件句柄：后台预取线程池会并发按 offset 读行，
# 每个线程持有自己的持久句柄，避免每个样本都 open/close 一次
_thread_files = threading.local()


def _tokenizer_cache_fingerprint(tokenizer) -> str:
    """打包语料缓存键中的 tokenizer 指纹。

    打包 id 与 seg 是「分词器输出」的派生缓存，但旧键只包含数据文件
    (path, mtime, size, max_length)：同一份数据换 tokenizer（或改 vocab）
    后会静默命中旧 token id 缓存。这里把完整 vocab 映射纳入键，避免
    换词表后继续使用错误打包。
    """
    import hashlib

    vocab = getattr(tokenizer, "get_vocab", None)
    try:
        payload = (
            json.dumps(vocab(), sort_keys=True, ensure_ascii=False)
            if vocab is not None
            else repr(
                (
                    getattr(tokenizer, "vocab_size", None),
                    getattr(tokenizer, "bos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None),
                    getattr(tokenizer, "pad_token_id", None),
                )
            )
        )
    except Exception:
        payload = repr(
            (
                getattr(tokenizer, "name_or_path", None),
                getattr(tokenizer, "vocab_size", None),
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            )
        )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """与 MiniMind 对齐：对话预处理。

    tool use 数据完整保留不做处理；无 system 首轮时按概率补一条
    随机 system prompt（默认 20%）。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get("tools") for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    # 概率性添加 system
    if conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """与 MiniMind 对齐：以 80% 概率移除空思考标签。"""
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


_PACK_SENTINEL = object()
_DEFAULT_PACK_BATCH = 4096


def _pack_batch_size(override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    raw = os.environ.get("VIBY_PACK_BATCH")
    if raw:
        return max(1, int(raw))
    return _DEFAULT_PACK_BATCH


def _pack_tqdm(**kwargs):
    """终端显示进度条；管道/测试（TQDM_DISABLE）下关闭，避免刷屏。"""
    env_off = os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.2)
    kwargs.setdefault("disable", env_off or not sys.stderr.isatty())
    return tqdm(**kwargs)


def _iter_text_batches(
    data_path: str, batch_size: int
) -> Iterable[Tuple[List[str], int]]:
    batch: List[str] = []
    nbytes = 0
    with open(data_path, "rb") as f:
        for line in f:
            nbytes += len(line)
            s = line.decode("utf-8", errors="ignore").strip()
            if not s:
                continue
            batch.append(json.loads(s)["text"])
            if len(batch) >= batch_size:
                yield batch, nbytes
                batch = []
                nbytes = 0
        if batch:
            yield batch, nbytes


def _prefetch(iterator: Iterable) -> Iterable:
    """读 JSON 与分词重叠：后台线程预取下一批。"""
    it = iter(iterator)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack-jsonl") as pool:
        fut = pool.submit(next, it, _PACK_SENTINEL)
        while True:
            item = fut.result()
            if item is _PACK_SENTINEL:
                return
            fut = pool.submit(next, it, _PACK_SENTINEL)
            yield item


def _encode_text_batch(tokenizer, texts: List[str]) -> List[List[int]]:
    """分批分词。有 Rust backend 时开并行，避免 TOKENIZERS_PARALLELISM=false。"""
    prev = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        backend = getattr(tokenizer, "backend", None)
        encode_batch = getattr(backend, "encode_batch", None) if backend else None
        if encode_batch is not None:
            return [enc.ids for enc in encode_batch(texts, add_special_tokens=False)]
        return tokenizer(texts, add_special_tokens=False)["input_ids"]
    finally:
        if prev is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev


def _raw_int32_to_npy(raw_path: str, out_path: str, width: int) -> int:
    """把流式写出的 int32 裸文件切成 (n_blocks, width) 的 .npy，分块拷贝避免峰值内存。"""
    n_tokens = os.path.getsize(raw_path) // 4
    n_blocks = n_tokens // width
    if n_blocks == 0:
        np.save(out_path, np.zeros((0, width), dtype=np.int32))
        return 0
    tmp_out = out_path + ".writing"
    src = np.memmap(raw_path, dtype=np.int32, mode="r", shape=(n_tokens,))
    dst = open_memmap(tmp_out, mode="w+", dtype=np.int32, shape=(n_blocks, width))
    chunk = max(1, (8 * 1024 * 1024) // (width * 4))
    pbar = _pack_tqdm(
        total=n_blocks,
        desc="[pack] write npy",
        unit="blk",
        leave=False,
    )
    try:
        for start in range(0, n_blocks, chunk):
            end = min(n_blocks, start + chunk)
            dst[start:end] = np.asarray(src[start * width : end * width]).reshape(
                end - start, width
            )
            pbar.update(end - start)
        dst.flush()
    finally:
        pbar.close()
        del dst
        del src
    os.replace(tmp_out, out_path)
    return n_blocks


def _unlink(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def pack_pretrain_jsonl(
    data_path: str,
    tokenizer,
    max_length: int,
    packed_path: str,
    segs_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """流式打包预训练 jsonl：ids（及可选文档 id）直写磁盘，峰值内存 ≈ 一个 batch。

    口径与旧实现相同：文档用 eos 拼接，切成 max_length+1 的定长块，余数丢弃。
    """
    packed_ok = os.path.exists(packed_path)
    segs_ok = segs_path is None or os.path.exists(segs_path)
    if packed_ok and segs_ok:
        packed = np.load(packed_path, mmap_mode="r")
        segs = np.load(segs_path, mmap_mode="r") if segs_path else None
        return packed, segs

    need_ids = not packed_ok
    need_segs = segs_path is not None and not segs_ok
    batch_size = _pack_batch_size(batch_size)
    parent = os.path.dirname(os.path.abspath(packed_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    width = max_length + 1
    eos_id = tokenizer.eos_token_id
    raw_ids = packed_path + ".raw" if need_ids else None
    raw_segs = segs_path + ".raw" if need_segs else None
    _unlink(raw_ids, raw_segs, packed_path + ".writing")
    if segs_path:
        _unlink(segs_path + ".writing")

    n_docs = 0
    n_tokens = 0
    t0 = time.time()
    print(
        f"[pack] 构建 {os.path.basename(data_path)} → {os.path.basename(packed_path)} "
        f"(seq={max_length}, segs={'on' if need_segs else 'off'}, batch={batch_size})",
        flush=True,
    )

    f_ids = open(raw_ids, "wb") if raw_ids else None
    f_segs = open(raw_segs, "wb") if raw_segs else None
    pbar = _pack_tqdm(
        total=os.path.getsize(data_path),
        desc="[pack] tokenize",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    try:
        for texts, n_bytes in _prefetch(_iter_text_batches(data_path, batch_size)):
            encoded = _encode_text_batch(tokenizer, texts)
            total = 0
            for ids in encoded:
                total += len(ids) + 1
            id_buf = np.empty(total, dtype=np.int32) if need_ids else None
            seg_buf = np.empty(total, dtype=np.int32) if need_segs else None
            pos = 0
            for i, ids in enumerate(encoded):
                n = len(ids) + 1
                if id_buf is not None:
                    id_buf[pos : pos + n - 1] = ids
                    id_buf[pos + n - 1] = eos_id
                if seg_buf is not None:
                    seg_buf[pos : pos + n] = n_docs + i
                pos += n
            if f_ids is not None and id_buf is not None:
                id_buf.tofile(f_ids)
            if f_segs is not None and seg_buf is not None:
                seg_buf.tofile(f_segs)
            n_docs += len(texts)
            n_tokens += total
            pbar.update(n_bytes)
            pbar.set_postfix(docs=f"{n_docs:,}", tok=f"{n_tokens:,}", refresh=False)
        if f_ids is not None:
            f_ids.flush()
        if f_segs is not None:
            f_segs.flush()
    finally:
        pbar.close()
        if f_ids is not None:
            f_ids.close()
        if f_segs is not None:
            f_segs.close()

    n_blocks = 0
    try:
        if need_ids and raw_ids:
            n_blocks = _raw_int32_to_npy(raw_ids, packed_path, width)
        if need_segs and raw_segs and segs_path:
            _raw_int32_to_npy(raw_segs, segs_path, width)
            if not need_ids:
                n_blocks = len(np.load(packed_path, mmap_mode="r"))
    finally:
        _unlink(
            raw_ids,
            raw_segs,
            packed_path + ".writing",
            (segs_path + ".writing") if segs_path else None,
        )

    print(
        f"[pack] 完成: {n_blocks} blocks, {n_docs} docs, {n_tokens} tokens, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    packed = np.load(packed_path, mmap_mode="r")
    segs = np.load(segs_path, mmap_mode="r") if segs_path else None
    return packed, segs


def _read_line_at_offset(data_path: str, offset: int) -> str:
    """Read a specific line using its byte offset (binary-safe, thread-local handle)."""
    f = getattr(_thread_files, "f", None)
    if f is None or getattr(_thread_files, "path", None) != data_path:
        f = open(data_path, "rb")
        _thread_files.f = f
        _thread_files.path = data_path
    f.seek(offset)
    return f.readline().decode("utf-8", errors="ignore").strip()


class PretrainDataset:
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=1024,
        cache_size: Optional[int] = 1000,
        pack_sequences: bool = False,
        doc_mask: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.cache_size = cache_size
        self.pack_sequences = pack_sequences
        self.doc_mask = doc_mask
        self._cache: Dict[int, Dict[str, Any]] = {}
        if pack_sequences:
            # 打包模式：文档 tokenize 后用 eos 拼接，切成 max_length+1 的定长块。
            # 流式写入磁盘，峰值内存约一个 batch；已有 .npy 则 mmap 只读。
            packed_path = self._packed_cache_path()
            segs_path = (
                packed_path.replace("packed_", "packedsegs_") if doc_mask else None
            )
            self._packed, self._packed_segs = pack_pretrain_jsonl(
                self.data_path,
                self.tokenizer,
                self.max_length,
                packed_path,
                segs_path,
            )
            self._line_offsets = None
        else:
            self._line_offsets = self._build_line_index()

    def _packed_cache_path(self) -> str:
        # VIBY_PACKED_CACHE：直接指定已构建的打包缓存（.cache/packed_<key>.npy），
        # 跳过对原始 data_path 的 stat——外接数据盘不在线时仍可用既有缓存
        # 训练；segs 缓存按 packed_/packedsegs_ 命名规则自动配对。
        override = os.environ.get("VIBY_PACKED_CACHE")
        if override:
            print(
                f"[pack] VIBY_PACKED_CACHE={override} "
                f"（忽略 data_path={self.data_path}）",
                flush=True,
            )
            return override
        import hashlib

        st = os.stat(self.data_path)
        key = hashlib.md5(
            f"{os.path.abspath(self.data_path)}:{st.st_mtime_ns}:{st.st_size}:"
            f"{self.max_length}:{_tokenizer_cache_fingerprint(self.tokenizer)}:packed".encode()
        ).hexdigest()
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"packed_{key}.npy")

    def _segs_cache_path(self) -> str:
        p = self._packed_cache_path()
        return p.replace("packed_", "packedsegs_")

    def _build_line_index(self) -> List[int]:
        """Build an index of line offsets for fast random access.

        结果按 (文件路径, mtime, size) 缓存到项目 .cache/ 下，
        大文件的索引构建（逐行扫一遍）只需做一次。
        """
        import hashlib

        st = os.stat(self.data_path)
        key = hashlib.md5(
            f"{os.path.abspath(self.data_path)}:{st.st_mtime_ns}:{st.st_size}".encode()
        ).hexdigest()
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
        )
        cache_path = os.path.join(cache_dir, f"line_offsets_{key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path).tolist()

        offsets = []
        with open(self.data_path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, np.asarray(offsets, dtype=np.int64))
        return offsets

    def _get_line_at_offset(self, offset: int) -> str:
        """Read a specific line using its byte offset (binary-safe)."""
        return _read_line_at_offset(self.data_path, offset)

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
        if self.pack_sequences:
            return len(self._packed)
        return len(self._line_offsets)

    def __getitem__(self, index):
        if self.pack_sequences:
            block = self._packed[index].astype(np.int64)
            X = block[:-1]
            Y = block[1:]
            if self._packed_segs is not None:
                segs = self._packed_segs[index].astype(np.int64)
                segX, segY = segs[:-1], segs[1:]
                # 边界位置（Y 已跨入下一篇）不计 loss：该处的预测目标在
                # 逐篇评估中不存在，是纯噪声梯度
                loss_mask = (segX == segY).astype(np.int64)
                return X, Y, loss_mask, segX
            return X, Y, np.ones_like(Y)

        sample = self._load_sample(index)

        # 与 MiniMind 对齐：不加特殊 token 分词，显式用 [bos] + text + [eos]
        # 包裹；截断长度留出 bos/eos 位置，所有非 PAD 位置都参与 next-token
        # loss（即首 token 也要预测，语义等价于 MiniMind 的 input_ids/labels）。
        encoding = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        )
        tokens = (
            [self.tokenizer.bos_token_id]
            + encoding["input_ids"]
            + [self.tokenizer.eos_token_id]
        )
        tokens = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = np.array(tokens, dtype=np.int64)
        loss_mask = (input_ids != self.tokenizer.pad_token_id).astype(np.int64)

        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = loss_mask[1:]
        return X, Y, loss_mask


class SFTDataset:
    def __init__(
        self, jsonl_path, tokenizer, max_length=2048, cache_size: Optional[int] = 1000
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = jsonl_path
        self.cache_size = cache_size
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._line_offsets = self._build_line_index()
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self._warned_zero_mask = False

    def _build_line_index(self) -> List[int]:
        """Build an index of line offsets for fast random access.

        结果按 (文件路径, mtime, size) 缓存到项目 .cache/ 下，
        大文件的索引构建（逐行扫一遍）只需做一次。
        """
        import hashlib

        st = os.stat(self.data_path)
        key = hashlib.md5(
            f"{os.path.abspath(self.data_path)}:{st.st_mtime_ns}:{st.st_size}".encode()
        ).hexdigest()
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
        )
        cache_path = os.path.join(cache_dir, f"line_offsets_{key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path).tolist()

        offsets = []
        with open(self.data_path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, np.asarray(offsets, dtype=np.int64))
        return offsets

    def _get_line_at_offset(self, offset: int) -> str:
        """Read a specific line using its byte offset (binary-safe)."""
        return _read_line_at_offset(self.data_path, offset)

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

    def create_chat_prompt(self, conversations):
        """与 MiniMind 对齐构建对话 prompt。

        解析 system 消息上的 tools 与 assistant 消息上的 tool_calls
        （JSON 字符串），完整保留 reasoning_content / tool 角色等字段，
        交由 tokenizer 的 chat template 渲染 <think> / <tool_call> /
        <tool_response> 片段。旧数据缺失 role 时仍按 user/assistant
        交替补全（system 不参与交替计数），与 MiniMind 数据兼容。
        """
        messages = []
        tools = None
        non_system_count = 0
        for turn in conversations:
            message = dict(turn)
            role = message.get("role")
            if role not in ("user", "assistant", "system", "tool"):
                role = "user" if non_system_count % 2 == 0 else "assistant"
            if role != "system":
                non_system_count += 1
            message["role"] = role
            if role == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def _create_chat_prompt(self, conversations):
        """旧接口别名，保留给外部调用。"""
        return self.create_chat_prompt(conversations)

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
                # 监督范围：从首个内容 token 到 <|im_end|>（含）为止；
                # 不多掩码下一轮的 <|im_start|>，也不跳过首个内容 token。
                # 若未找到 eos，则掩码到序列末尾
                upper = min(end + len(self.eos_id), n)
                for j in range(start, upper):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < n else n
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self._load_sample(index)
        # 与 MiniMind 对齐：概率性补 system、清洗空 <think> 标签后再渲染
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # 与 MiniMind 对齐：保留序列头部（截断尾部）。超长样本的
        # assistant 回复可能被截掉，此时 loss mask 全 0（仅警告一次）
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )
        loss_mask = self._generate_loss_mask(input_ids)
        if not any(loss_mask):
            if not self._warned_zero_mask:
                print(
                    "[Warning] SFT 样本截断后没有任何 assistant 位置可监督"
                    "（loss mask 全 0），请检查数据格式或增大 max_length。"
                )
                self._warned_zero_mask = True

        # 构建训练数据
        input_ids_arr = np.array(input_ids, dtype=np.int64)
        X = input_ids_arr[:-1]
        Y = input_ids_arr[1:]
        loss_mask = np.array(loss_mask[1:], dtype=np.int64)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset:
    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self._warned_zero_mask = False
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]  # 是一个 list，里面包含若干 {role, content}
        rejected = item["rejected"]  # 同上
        # 与 MiniMind 对齐：模板渲染后清洗空 <think> 标签
        chosen_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                chosen, tokenize=False, add_generation_prompt=False
            )
        )
        rejected_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                rejected, tokenize=False, add_generation_prompt=False
            )
        )
        # 与 MiniMind 对齐：默认右截断（保留序列头部）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        if (not any(chosen_loss_mask)) or (not any(rejected_loss_mask)):
            if not self._warned_zero_mask:
                print(
                    "[Warning] DPO 样本截断后 chosen/rejected 的 loss mask 全 0，"
                    "请检查数据格式或增大 max_length。"
                )
                self._warned_zero_mask = True
        x_chosen = np.array(chosen_input_ids[:-1], dtype=np.int64)
        y_chosen = np.array(chosen_input_ids[1:], dtype=np.int64)
        mask_chosen = np.array(chosen_loss_mask[1:], dtype=np.int64)
        x_rejected = np.array(rejected_input_ids[:-1], dtype=np.int64)
        y_rejected = np.array(rejected_input_ids[1:], dtype=np.int64)
        mask_rejected = np.array(rejected_loss_mask[1:], dtype=np.int64)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 与 SFT 一致：监督 [首个内容 token, <|im_end|>]（含），
                # 不多掩码下一轮标记，也不跳过首个内容 token
                for j in range(start, min(end + len(self.eos_id), len(input_ids))):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
