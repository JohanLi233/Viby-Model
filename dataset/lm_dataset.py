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


def post_processing_chat(prompt_content, empty_think_ratio=0.0):
    """清掉空 <think> 占位块；默认 0.0 = 始终移除（空 think 不进训练数据）。

    empty_think_ratio 为空 <think> 的保留阈值：0.0 = 全部移除（默认，现代大模型
    思路——思考内容应落在 think 内，空的占位块是脏数据）；>0 时按概率保留。
    """
    if "<think>\n\n</think>\n\n" in prompt_content and (
        empty_think_ratio <= 0 or random.random() > empty_think_ratio
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


def _raw_int32_to_npy_aligned(
    raw_ids: str,
    raw_segs: Optional[str],
    doc_starts_path: str,
    out_ids: str,
    out_segs: Optional[str],
    width: int,
) -> int:
    """文档边界对齐切片：每个 block 的首 token 都落在某篇文档的开头。

    raw 流是「文档 token + eos」顺序拼接（长文档先被 max_doc_len 截断）。
    block 起点取文档边界；若上一 block 末尾落在某文档中段，则丢弃该文档
    尾部、从下一篇文档开头重新对齐。这样每个 block 的首 token 都是真实
    文档开头（训练时跨文档注意力只发生在可分段的边界处，边界 loss 由
    doc_mask / segment_ids 处理）。丢弃尾部 token 换取对齐，代价由
    max_doc_len 控制（值越小，跨块截断的尾部越短）。

    ids 与 segs 并行切分（同一组 block 起点），额外返回 block 数。
    """
    import bisect

    n_tokens = os.path.getsize(raw_ids) // 4
    doc_starts = np.load(doc_starts_path)  # int64，(n_docs+1,)，最后一项=总 token 数
    if n_tokens == 0:
        np.save(out_ids, np.zeros((0, width), dtype=np.int32))
        if out_segs:
            np.save(out_segs, np.zeros((0, width), dtype=np.int32))
        return 0
    src_ids = np.memmap(raw_ids, dtype=np.int32, mode="r", shape=(n_tokens,))
    src_segs = None
    if raw_segs and out_segs and os.path.exists(raw_segs):
        src_segs = np.memmap(raw_segs, dtype=np.int32, mode="r", shape=(n_tokens,))

    # 逐块规划起点：0 是文档 0 开头；之后跳到「>= 上一块末尾」的第一个文档开头。
    starts: List[int] = []
    lo = 0
    pos = 0
    while pos + width <= n_tokens:
        starts.append(pos)
        k = bisect.bisect_left(doc_starts, pos + width, lo, len(doc_starts))
        if k >= len(doc_starts):
            break
        pos = int(doc_starts[k])
        lo = k
    n_blocks = len(starts)
    if n_blocks == 0:
        np.save(out_ids, np.zeros((0, width), dtype=np.int32))
        if out_segs:
            np.save(out_segs, np.zeros((0, width), dtype=np.int32))
        return 0

    tmp_ids = out_ids + ".writing"
    tmp_segs = (out_segs + ".writing") if out_segs else None
    _unlink(tmp_ids, tmp_segs)
    dst_ids = open_memmap(tmp_ids, mode="w+", dtype=np.int32, shape=(n_blocks, width))
    dst_segs = (
        open_memmap(tmp_segs, mode="w+", dtype=np.int32, shape=(n_blocks, width))
        if out_segs
        else None
    )
    chunk = max(1, (8 * 1024 * 1024) // (width * 4))
    pbar = _pack_tqdm(total=n_blocks, desc="[pack] write aligned npy", unit="blk", leave=False)
    try:
        for b in range(0, n_blocks, chunk):
            e = min(n_blocks, b + chunk)
            sb = np.asarray(starts[b:e], dtype=np.int64)
            cols = np.arange(width, dtype=np.int64)
            # 每块内容 = raw[pos : pos+width]（起点已对齐到文档边界）
            idx = sb[:, None] + cols[None, :]
            idx = np.minimum(idx, n_tokens - 1)
            dst_ids[b:e] = np.asarray(src_ids[idx])
            if dst_segs is not None and src_segs is not None:
                dst_segs[b:e] = np.asarray(src_segs[idx])
            pbar.update(e - b)
        dst_ids.flush()
        if dst_segs is not None:
            dst_segs.flush()
    finally:
        pbar.close()
        del dst_ids
        del dst_segs
        del src_ids
        del src_segs
    os.replace(tmp_ids, out_ids)
    if tmp_segs:
        os.replace(tmp_segs, out_segs)
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
    align_docs: bool = False,
    max_doc_len: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """流式打包预训练 jsonl：ids（及可选文档 id）直写磁盘，峰值内存 ≈ 一个 batch。

    align_docs=True 时：每块首 token 对齐到文档开头，长文档先按 max_doc_len
    截断（默认 max_length，即单篇最多占一个块），以丢弃跨块尾部位为代价换取
    边界对齐。align_docs=False 时与旧口径一致：文档用 eos 拼接，定长切块，
    余数丢弃。
    """
    packed_ok = os.path.exists(packed_path)
    segs_ok = segs_path is None or os.path.exists(segs_path)
    if packed_ok and segs_ok:
        packed = np.load(packed_path, mmap_mode="r")
        segs = np.load(segs_path, mmap_mode="r") if segs_path else None
        return packed, segs

    need_ids = not packed_ok
    need_segs = segs_path is not None and not segs_ok
    if align_docs and need_segs and not need_ids:
        # 对齐切片需要原始 id 流与文档起点；segs 缺失时一并重新分词构建 id。
        need_ids = True
    batch_size = _pack_batch_size(batch_size)
    parent = os.path.dirname(os.path.abspath(packed_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    width = max_length + 1
    eos_id = tokenizer.eos_token_id
    raw_ids = packed_path + ".raw" if need_ids else None
    raw_segs = segs_path + ".raw" if need_segs else None
    doc_starts_path = (raw_ids + ".starts.npy") if (align_docs and need_ids) else None
    _unlink(raw_ids, raw_segs, packed_path + ".writing")
    if segs_path:
        _unlink(segs_path + ".writing")
    _unlink(doc_starts_path)

    # 对齐口：每篇文档 token 数上限；再夹到 max_length，保证 单篇+eos ≤ width。
    cap = None
    if align_docs:
        cap = max_length if max_doc_len is None else min(max_doc_len, max_length)
        if cap <= 0:
            raise ValueError(f"--max_doc_len 必须 > 0，收到 {cap}")
    doc_starts: List[int] = [0]
    cur_tok = 0

    n_docs = 0
    n_tokens = 0
    t0 = time.time()
    print(
        f"[pack] 构建 {os.path.basename(data_path)} → {os.path.basename(packed_path)} "
        f"(seq={max_length}, segs={'on' if need_segs else 'off'}, batch={batch_size}, "
        f"align={'on' if align_docs else 'off'}, max_doc={cap or '—'})",
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
            if align_docs:
                lens = [min(len(ids), cap) + 1 for ids in encoded]
            else:
                lens = [len(ids) + 1 for ids in encoded]
            total = sum(lens)
            id_buf = np.empty(total, dtype=np.int32) if need_ids else None
            seg_buf = np.empty(total, dtype=np.int32) if need_segs else None
            pos = 0
            for i, ids in enumerate(encoded):
                if align_docs:
                    ids = ids[:cap]
                    if n_docs + i > 0:
                        doc_starts.append(cur_tok)
                n = len(ids) + 1
                if id_buf is not None:
                    id_buf[pos : pos + n - 1] = ids
                    id_buf[pos + n - 1] = eos_id
                if seg_buf is not None:
                    seg_buf[pos : pos + n] = n_docs + i
                pos += n
                cur_tok += n
            if f_ids is not None and id_buf is not None:
                id_buf.tofile(f_ids)
            if f_segs is not None and seg_buf is not None:
                seg_buf.tofile(f_segs)
            n_docs += len(texts)
            n_tokens += total
            pbar.update(n_bytes)
            pbar.set_postfix(docs=f"{n_docs:,}", tok=f"{n_tokens:,}", refresh=False)
        if align_docs and doc_starts_path:
            doc_starts.append(n_tokens)
            np.save(doc_starts_path, np.asarray(doc_starts, dtype=np.int64))
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
        if align_docs and need_ids and raw_ids and doc_starts_path:
            n_blocks = _raw_int32_to_npy_aligned(
                raw_ids,
                raw_segs if need_segs else None,
                doc_starts_path,
                packed_path,
                segs_path if need_segs else None,
                width,
            )
        elif need_ids and raw_ids:
            n_blocks = _raw_int32_to_npy(raw_ids, packed_path, width)
        if not align_docs and need_segs and raw_segs and segs_path:
            _raw_int32_to_npy(raw_segs, segs_path, width)
            if not need_ids:
                n_blocks = len(np.load(packed_path, mmap_mode="r"))
    finally:
        _unlink(
            raw_ids,
            raw_segs,
            packed_path + ".writing",
            (segs_path + ".writing") if segs_path else None,
            doc_starts_path,
        )

    print(
        f"[pack] 完成: {n_blocks} blocks, {n_docs} docs, {n_tokens} tokens, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    packed = np.load(packed_path, mmap_mode="r")
    segs = np.load(segs_path, mmap_mode="r") if segs_path else None
    return packed, segs


def _assistant_loss_mask(input_ids, bos_id, eos_id) -> List[int]:
    """监督范围：每个 assistant 轮的首个内容 token 到 <|im_end|>（含）。

    bos_id 是 "<|im_start|>assistant\\n" 的分词结果，eos_id 是
    "<|im_end|>\\n" 的分词结果；未找到 eos 时掩码到序列末尾。
    """
    loss_mask = [0] * len(input_ids)
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i : i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < n:
                if input_ids[end : end + len(eos_id)] == eos_id:
                    break
                end += 1
            upper = min(end + len(eos_id), n)
            for j in range(start, upper):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < n else n
        else:
            i += 1
    return loss_mask


def _render_chat_prompt(tokenizer, conversations) -> str:
    """渲染对话 prompt（SFTDataset.create_chat_prompt 的模块级实现）。"""
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


def _iter_conversation_batches(
    data_path: str, batch_size: int
) -> Iterable[Tuple[List[list], int]]:
    batch: List[list] = []
    nbytes = 0
    with open(data_path, "rb") as f:
        for line in f:
            nbytes += len(line)
            s = line.decode("utf-8", errors="ignore").strip()
            if not s:
                continue
            batch.append(json.loads(s)["conversations"])
            if len(batch) >= batch_size:
                yield batch, nbytes
                batch = []
                nbytes = 0
        if batch:
            yield batch, nbytes


def pack_sft_jsonl(
    data_path: str,
    tokenizer,
    max_length: int,
    packed_path: str,
    masks_path: str,
    segs_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    empty_think_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """流式打包 SFT jsonl：渲染 chat 模板后按 token 流拼接，切 max_length+1 定长块。

    与 pack_pretrain_jsonl 同口径，但不打 pad、不右截断：长样本跨块继续，
    assistant 监督不会因截断整体丢失。额外写一条逐 token 监督 mask
    （仅 assistant 内容到 <|im_end|> 为 1，int32）。segs_path 给定时
    写逐 token 的样本 id（doc_mask 用）。

    注意：pre_processing_chat / post_processing_chat 的随机增强
    （补 system、清空 <think>）在打包时固化一次，不再每 epoch 重掷。
    empty_think_ratio 默认 0.0（始终清掉空 think 占位块）。
    """
    packed_ok = os.path.exists(packed_path)
    masks_ok = os.path.exists(masks_path)
    segs_ok = segs_path is None or os.path.exists(segs_path)
    if packed_ok and masks_ok and segs_ok:
        packed = np.load(packed_path, mmap_mode="r")
        masks = np.load(masks_path, mmap_mode="r")
        segs = np.load(segs_path, mmap_mode="r") if segs_path else None
        return packed, masks, segs

    need_ids = not packed_ok
    need_masks = not masks_ok
    need_segs = segs_path is not None and not segs_ok
    batch_size = _pack_batch_size(batch_size)
    parent = os.path.dirname(os.path.abspath(packed_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    width = max_length + 1
    bos_id = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    ).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    raw_ids = packed_path + ".raw" if need_ids else None
    raw_masks = masks_path + ".raw" if need_masks else None
    raw_segs = segs_path + ".raw" if need_segs else None
    _unlink(raw_ids, raw_masks, raw_segs, packed_path + ".writing")
    if segs_path:
        _unlink(segs_path + ".writing")

    n_samples = 0
    n_tokens = 0
    n_supervised = 0
    t0 = time.time()
    print(
        f"[pack] 构建 {os.path.basename(data_path)} → {os.path.basename(packed_path)} "
        f"(sft, seq={max_length}, segs={'on' if need_segs else 'off'}, batch={batch_size})",
        flush=True,
    )

    f_ids = open(raw_ids, "wb") if raw_ids else None
    f_masks = open(raw_masks, "wb") if raw_masks else None
    f_segs = open(raw_segs, "wb") if raw_segs else None
    pbar = _pack_tqdm(
        total=os.path.getsize(data_path),
        desc="[pack] tokenize (sft)",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    try:
        for convs, n_bytes in _prefetch(
            _iter_conversation_batches(data_path, batch_size)
        ):
            prompts = [
                post_processing_chat(
                    _render_chat_prompt(tokenizer, pre_processing_chat(c)),
                    empty_think_ratio,
                )
                for c in convs
            ]
            encoded = _encode_text_batch(tokenizer, prompts)
            total = sum(len(ids) for ids in encoded)
            id_buf = np.empty(total, dtype=np.int32) if need_ids else None
            mask_buf = np.empty(total, dtype=np.int32) if need_masks else None
            seg_buf = np.empty(total, dtype=np.int32) if need_segs else None
            pos = 0
            for i, ids in enumerate(encoded):
                n = len(ids)
                if n == 0:
                    continue
                if id_buf is not None:
                    id_buf[pos : pos + n] = ids
                if mask_buf is not None:
                    m = _assistant_loss_mask(ids, bos_id, eos_id)
                    mask_buf[pos : pos + n] = m
                    n_supervised += sum(m)
                if seg_buf is not None:
                    seg_buf[pos : pos + n] = n_samples + i
                pos += n
            if pos:
                if f_ids is not None and id_buf is not None:
                    id_buf[:pos].tofile(f_ids)
                if f_masks is not None and mask_buf is not None:
                    mask_buf[:pos].tofile(f_masks)
                if f_segs is not None and seg_buf is not None:
                    seg_buf[:pos].tofile(f_segs)
            n_samples += len(convs)
            n_tokens += pos
            pbar.update(n_bytes)
            pbar.set_postfix(
                samples=f"{n_samples:,}", tok=f"{n_tokens:,}", refresh=False
            )
        for fh in (f_ids, f_masks, f_segs):
            if fh is not None:
                fh.flush()
    finally:
        pbar.close()
        for fh in (f_ids, f_masks, f_segs):
            if fh is not None:
                fh.close()

    n_blocks = 0
    try:
        if need_ids and raw_ids:
            n_blocks = _raw_int32_to_npy(raw_ids, packed_path, width)
        if need_masks and raw_masks:
            _raw_int32_to_npy(raw_masks, masks_path, width)
        if need_segs and raw_segs and segs_path:
            _raw_int32_to_npy(raw_segs, segs_path, width)
        if not need_ids:
            n_blocks = len(np.load(packed_path, mmap_mode="r"))
    finally:
        _unlink(
            raw_ids,
            raw_masks,
            raw_segs,
            packed_path + ".writing",
            (segs_path + ".writing") if segs_path else None,
        )

    print(
        f"[pack] 完成: {n_blocks} blocks, {n_samples} samples, {n_tokens} tokens, "
        f"监督 {n_supervised} tokens ({(n_supervised / max(n_tokens, 1)):.1%}), "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    packed = np.load(packed_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")
    segs = np.load(segs_path, mmap_mode="r") if segs_path else None
    return packed, masks, segs


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
        align_docs: bool = True,
        max_doc_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.cache_size = cache_size
        self.pack_sequences = pack_sequences
        self.doc_mask = doc_mask
        self.align_docs = align_docs
        self.max_doc_len = max_doc_len
        self._cache: Dict[int, Dict[str, Any]] = {}
        if pack_sequences:
            # 打包模式：文档 tokenize 后用 eos 拼接，切成 max_length+1 的定长块。
            # align_docs 默认开：每块首 token 对齐文档开头，长文档按 max_doc_len 截断，
            # 以丢弃跨块尾部位为代价换取边界对齐（减少跨文档无效 attention）。
            # 流式写入磁盘，峰值内存约一个 batch；已有 .npy 则 mmap 只读。
            packed_path = self._packed_cache_path()
            segs_path = self._segs_cache_path() if doc_mask else None
            self._packed, self._packed_segs = pack_pretrain_jsonl(
                self.data_path,
                self.tokenizer,
                self.max_length,
                packed_path,
                segs_path,
                align_docs=self.align_docs,
                max_doc_len=self.max_doc_len,
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
            f"{self.max_length}:{_tokenizer_cache_fingerprint(self.tokenizer)}:"
            f"align={int(getattr(self, 'align_docs', True))}:"
            f"maxdoc={getattr(self, 'max_doc_len', None)}:packed".encode()
        ).hexdigest()
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"packed_{key}.npy")

    def _segs_cache_path(self) -> str:
        p = self._packed_cache_path()
        segs = p.replace("packed_", "packedsegs_")
        if segs == p:
            # VIBY_PACKED_CACHE 覆盖路径不含 "packed_" 时 replace 失效，
            # 直接把 ids 缓存当 segs 会让 doc_mask 静默错乱
            segs = p + ".segs"
        return segs

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
        self,
        jsonl_path,
        tokenizer,
        max_length=2048,
        cache_size: Optional[int] = 1000,
        pack_sequences: bool = False,
        doc_mask: bool = False,
        empty_think_ratio: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = jsonl_path
        self.cache_size = cache_size
        self.pack_sequences = pack_sequences
        self.doc_mask = doc_mask
        # 空 <think> 占位块是脏数据：默认 empty_think_ratio=0.0，打包/取值时
        # 统一清掉；真实思考内容不含空 think 模式，不受影响。
        self.empty_think_ratio = empty_think_ratio
        self._cache: Dict[int, Dict[str, Any]] = {}
        if pack_sequences:
            # 打包模式（与 PretrainDataset 同口径）：渲染后按 token 流拼接成
            # max_length+1 定长块，不打 pad、不右截断；监督 mask 逐 token 打包。
            packed_path = self._sft_cache_path("sftpacked_")
            masks_path = self._sft_cache_path("sftpackedmasks_")
            segs_path = self._sft_cache_path("sftpackedsegs_") if doc_mask else None
            self._packed, self._packed_masks, self._packed_segs = pack_sft_jsonl(
                self.data_path,
                self.tokenizer,
                self.max_length,
                packed_path,
                masks_path,
                segs_path,
                empty_think_ratio=empty_think_ratio,
            )
            self._line_offsets = None
        else:
            self._line_offsets = self._build_line_index()
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self._warned_zero_mask = False

    def _sft_cache_path(self, prefix: str) -> str:
        """SFT 打包缓存路径：与 pretrain 同结构，prefix 区分 ids/masks/segs。"""
        import hashlib

        st = os.stat(self.data_path)
        key = hashlib.md5(
            f"{os.path.abspath(self.data_path)}:{st.st_mtime_ns}:{st.st_size}:"
            f"{self.max_length}:{_tokenizer_cache_fingerprint(self.tokenizer)}:"
            f"{self.empty_think_ratio}:sftpacked".encode()
        ).hexdigest()
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{prefix}{key}.npy")

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

    def create_chat_prompt(self, conversations):
        """与 MiniMind 对齐构建对话 prompt。

        解析 system 消息上的 tools 与 assistant 消息上的 tool_calls
        （JSON 字符串），完整保留 reasoning_content / tool 角色等字段，
        交由 tokenizer 的 chat template 渲染 <think> / <tool_call> /
        <tool_response> 片段。旧数据缺失 role 时仍按 user/assistant
        交替补全（system 不参与交替计数），与 MiniMind 数据兼容。
        """
        return _render_chat_prompt(self.tokenizer, conversations)

    def _create_chat_prompt(self, conversations):
        """旧接口别名，保留给外部调用。"""
        return self.create_chat_prompt(conversations)

    def _generate_loss_mask(self, input_ids):
        # 监督范围：从首个内容 token 到 <|im_end|>（含）为止；
        # 不多掩码下一轮的 <|im_start|>，也不跳过首个内容 token。
        return _assistant_loss_mask(input_ids, self.bos_id, self.eos_id)

    def __getitem__(self, index):
        if self.pack_sequences:
            block = self._packed[index].astype(np.int64)
            X = block[:-1]
            Y = block[1:]
            # mask 存的是「作为预测目标」的口径，与 Y 对齐取 [1:]
            loss_mask = self._packed_masks[index].astype(np.int64)[1:]
            if self._packed_segs is not None:
                segs = self._packed_segs[index].astype(np.int64)
                segX, segY = segs[:-1], segs[1:]
                # 跨样本边界位置（Y 已跨入下一条）不计 loss：该处的预测
                # 目标在 doc_mask 下不可见，是纯噪声梯度
                loss_mask = (loss_mask & (segX == segY)).astype(np.int64)
                return X, Y, loss_mask, segX
            return X, Y, loss_mask

        sample = self._load_sample(index)
        # 概率性补 system 与 MiniMind 对齐；空 <think> 占位块默认清掉
        # （empty_think_ratio=0.0）
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt, self.empty_think_ratio)
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
        # 与 SFT 一致：监督 [首个内容 token, <|im_end|>]（含），
        # 不多掩码下一轮标记，也不跳过首个内容 token
        return _assistant_loss_mask(input_ids, self.bos_id, self.eos_id)
