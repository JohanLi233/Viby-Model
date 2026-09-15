"""新架构的解码状态池：按"行"搬运 window / 压缩 KV / index K / 压缩器缓冲。

旧 `engine/memory.py` 的 PagePool 是 MLA + 分页 KV 专用；新模型每层状态形状
都不一样（window [B,window,hd] 环、compress_kv [B,max_seq//ratio+1,hd] 池、
index_k [B,pool_scratch+1,index_dim] 池、压缩器 kv_state [B,ratio,hd] 缓冲），
不能按固定页切分，所以改成"行 = 一条序列"的池子：

- `StatePool` 内部持有一个 `VibyCache(config, batch)`，batch = 当前并发数；
  并发数变化时整池重建并把已有行整行搬过去（一次性 O(状态) 拷贝，取代旧实现
  每步 load/spill 的分页记账）。
- `copy_state_row` / `capture_state` / `restore_state` 负责行与快照之间的搬运；
  前缀缓存（engine/prefix.py）保存的就是 `capture_state` 的结果。

注意两条已验证的约束（/tmp/dsv41_ref/API.md）：
1. `VibyCache` 是可变的：解码会原地推进 window/compress_kv/index_k/kv_state，
   所以快照必须在**跑解码之前**做（capture_state 用 mx.array 显式取值拷贝）；
2. 压缩器的 kv_state 是**批状态** [B,ratio,hd]，往池子里搬单行时必须保留其它行，
   不能把整份 state 覆盖成单行（MLX 的 `.at[]` 不支持 set，这里用 one-hot where）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from model.cache import VibyCache


@dataclass
class PrefixState:
    """某条序列在长度 n 处的整套解码状态快照（前缀缓存的复用单元）。"""

    n: int
    ncp: object = None
    thinking: object = None
    window: list = field(default_factory=list)  # 每层 [window, hd]
    compress: dict = field(default_factory=dict)  # 源层 → [n//ratio, hd]
    index_k: dict = field(
        default_factory=dict
    )  # 拥有 index K 的层 → [n//ratio, index_dim]
    kv_state: dict = field(
        default_factory=dict
    )  # ratio>1 源层 → (kv[r,hd], score[r,hd], filled[])
    engram_prev: Optional[mx.array] = None  # [w] 最近 token（末位最近）
    hidden: Optional[mx.array] = None  # [D] 末位 hidden（全命中时重算 logits）
    draft_mains: Optional[tuple] = (
        None  # target-layer inputs [D] at this prefix's last token
    )


def _batch_of(cache: VibyCache) -> int:
    return cache.layers[0].window.shape[0]


def _row_mask(batch: int, drow: int, ndim: int) -> mx.array:
    return (mx.arange(batch) == drow).reshape((batch,) + (1,) * (ndim - 1))


def _set_state_row_batched(lc, drow: int, src_state, srow: int, batch: int) -> None:
    """源是**批状态**（元素 [B,...]）：取第 srow 行写进 lc 的批状态，保留其它行。"""
    if lc.kv_state is None:
        lc.kv_state = tuple(
            mx.zeros((batch,) + s.shape[1:], dtype=s.dtype) for s in src_state
        )
    rows = []
    for a, b in zip(lc.kv_state, src_state):
        rows.append(mx.where(_row_mask(batch, drow, a.ndim), b[srow], a))
    lc.kv_state = tuple(rows)


def _set_state_row_single(lc, drow: int, row_state, batch: int) -> None:
    """源是**单行状态**（元素 [ratio,hd] / 标量，前缀快照的形状）：整行写入。"""
    if lc.kv_state is None:
        lc.kv_state = tuple(
            mx.zeros((batch,) + s.shape, dtype=s.dtype) for s in row_state
        )
    rows = []
    for a, b in zip(lc.kv_state, row_state):
        rows.append(mx.where(_row_mask(batch, drow, a.ndim), b, a))
    lc.kv_state = tuple(rows)


def _set_engram_row(dst: VibyCache, drow: int, row: Optional[mx.array]) -> None:
    if row is None:
        return
    batch = _batch_of(dst)
    if dst.engram_prev is None:
        dst.engram_prev = mx.zeros((batch, row.shape[-1]), dtype=mx.int32)
    dst.engram_prev[drow] = row


def _ncp_row(cache, row):
    if cache.ncp_rows is not None:
        return cache.ncp_rows[row].row_copy()
    return None if cache.ncp_state is None else cache.ncp_state.row_copy(row)


def _thinking_row(cache, row):
    if cache.thinking_rows is not None:
        return cache.thinking_rows[row].row_copy()
    return None if cache.thinking_state is None else cache.thinking_state.row_copy(row)


def _set_thinking_row(cache, row, state):
    from model.thinking import ThinkingCache, cache_signature

    if state is None:
        if cache.config.thinking_enabled:
            raise ValueError("Prefix is missing thinking state; use a fresh prefill")
        return
    if not cache.config.thinking_enabled or state.signature != cache_signature(
        cache.config
    ):
        raise ValueError("Thinking prefix execution differs; use a fresh prefill")
    batch = _batch_of(cache)
    if batch == 1:
        cache.thinking_state, cache.thinking_rows = state.row_copy(), None
    else:
        if cache.thinking_rows is None:
            cache.thinking_rows = [ThinkingCache() for _ in range(batch)]
            for item in cache.thinking_rows:
                item.signature = cache_signature(cache.config)
        cache.thinking_rows[row] = state.row_copy()
        cache.thinking_state = None


def _set_ncp_row(cache, row, state):
    if state is None:
        if cache.config.ncp_enabled:
            raise ValueError("Prefix state is missing NCP memory; use a fresh prefill")
        return
    if not cache.config.ncp_enabled:
        raise ValueError("NCP prefix state cannot be loaded into ordinary CED")
    from model.ncp import cache_signature

    if state.signature != cache_signature(cache.config):
        raise ValueError("NCP prefix execution differs; use a fresh prefill")
    batch = _batch_of(cache)
    if batch == 1:
        cache.ncp_state = state.row_copy()
        cache.ncp_rows = None
    else:
        if cache.ncp_rows is None:
            from model.ncp import ConceptCache

            cache.ncp_rows = [ConceptCache() for _ in range(batch)]
            for item in cache.ncp_rows:
                item.signature = cache_signature(cache.config)
        cache.ncp_rows[row] = state.row_copy()
        cache.ncp_state = None


def copy_state_row(
    dst: VibyCache, drow: int, src: VibyCache, srow: int = 0, n: Optional[int] = None
) -> None:
    """把 src 第 srow 行的状态复制到 dst 第 drow 行。

    n = 该行已处理的 token 数：给定时只复制用到的前缀（压缩池 / index K 池），
    None 表示整行复制（池子扩容/压实用，池尾多出的槽位反正不会被读到）。
    """
    batch = _batch_of(dst)
    _set_ncp_row(dst, drow, _ncp_row(src, srow))
    _set_thinking_row(dst, drow, _thinking_row(src, srow))
    for i, dl in enumerate(dst.layers):
        sl = src.layers[i]
        dl.window[drow] = sl.window[srow]
        ratio = max(sl.ratio, 1)
        k = (sl.pool_scratch + 1) if n is None else (n // ratio)
        if dl.compress_kv is not None and k:
            dl.compress_kv[drow, :k] = sl.compress_kv[srow, :k]
        if sl.kv_state is not None:
            _set_state_row_batched(dl, drow, sl.kv_state, srow, batch)
        if dl.index_k is not None and k:
            dl.index_k[drow, :k] = sl.index_k[srow, :k]
    if src.engram_prev is not None:
        _set_engram_row(dst, drow, src.engram_prev[srow])


def capture_state(
    cache: VibyCache,
    row: int,
    n: int,
    hidden: Optional[mx.array] = None,
    draft_mains=None,
) -> PrefixState:
    """从 cache 的第 row 行取长度 n 的状态快照（显式拷贝，之后 cache 再被写也不影响）。"""
    state = PrefixState(n=int(n), hidden=hidden)
    state.ncp = _ncp_row(cache, row)
    state.thinking = _thinking_row(cache, row)
    if state.thinking is not None and state.thinking.tokens != n:
        raise ValueError("Thinking snapshot must match the complete prefix clock")
    if state.ncp is not None and state.ncp.tokens != n:
        raise ValueError("NCP snapshot must use the actual processed token clock")
    if draft_mains is not None:
        state.draft_mains = tuple(mx.array(m) for m in draft_mains)
    for i, lc in enumerate(cache.layers):
        ratio = max(lc.ratio, 1)
        k = n // ratio
        state.window.append(mx.array(lc.window[row]))
        if lc.compress_kv is not None:
            state.compress[i] = mx.array(lc.compress_kv[row, :k])
        if lc.kv_state is not None:
            state.kv_state[i] = tuple(mx.array(s[row]) for s in lc.kv_state)
        if lc.index_k is not None:
            state.index_k[i] = mx.array(lc.index_k[row, :k])
    if cache.engram_prev is not None:
        state.engram_prev = mx.array(cache.engram_prev[row])
    return state


def restore_state(dst: VibyCache, drow: int, state: PrefixState) -> int:
    """把快照写回 dst 的第 drow 行，返回快照长度 n。"""
    batch = _batch_of(dst)
    if state.ncp is not None and state.ncp.tokens != state.n:
        raise ValueError("NCP prefix clock mismatch")
    _set_ncp_row(dst, drow, state.ncp)
    if state.thinking is not None and state.thinking.tokens != state.n:
        raise ValueError("Thinking snapshot clock mismatch")
    _set_thinking_row(dst, drow, state.thinking)
    for i, dl in enumerate(dst.layers):
        # Dense prefix continuation uses a Python filled count; the decode
        # compressor also carries a per-row GPU count in kv_state.
        dl.filled = state.n % dl.ratio if dl.ratio > 1 else 0
        dl.window[drow] = state.window[i]
        if dl.compress_kv is not None:
            arr = state.compress.get(i)
            if arr is not None and arr.shape[0]:
                dl.compress_kv[drow, : arr.shape[0]] = arr
        if dl.index_k is not None:
            arr = state.index_k.get(i)
            if arr is not None and arr.shape[0]:
                dl.index_k[drow, : arr.shape[0]] = arr
        if i in state.kv_state:
            _set_state_row_single(dl, drow, state.kv_state[i], batch)
    if state.engram_prev is not None:
        _set_engram_row(dst, drow, state.engram_prev)
    return int(state.n)


def _array_bytes(x) -> int:
    if x is None or not isinstance(x, mx.array):
        return 0
    return int(x.size) * int(x.dtype.size)


def cache_bytes(cache: Optional[VibyCache]) -> int:
    """一个 VibyCache 的常驻字节数（三类池 + 压缩器状态 + engram）。"""
    if cache is None:
        return 0
    total = 0
    for lc in cache.layers:
        total += _array_bytes(lc.window)
        total += _array_bytes(lc.compress_kv)
        total += _array_bytes(lc.index_k)
        if lc.kv_state is not None:
            total += sum(_array_bytes(s) for s in lc.kv_state)
    total += _array_bytes(cache.engram_prev)
    if cache.thinking_rows is not None:
        total += sum(state.nbytes() for state in cache.thinking_rows)
    elif cache.thinking_state is not None:
        total += cache.thinking_state.nbytes()
    if cache.ncp_rows is not None:
        total += sum(state.nbytes() for state in cache.ncp_rows)
    elif cache.ncp_state is not None:
        total += cache.ncp_state.nbytes()
    return total


def state_bytes(state: PrefixState) -> int:
    total = sum(_array_bytes(w) for w in state.window)
    total += sum(_array_bytes(v) for v in state.compress.values())
    total += sum(_array_bytes(v) for v in state.index_k.values())
    for tup in state.kv_state.values():
        total += sum(_array_bytes(s) for s in tup)
    total += _array_bytes(state.engram_prev)
    total += _array_bytes(state.hidden)
    if state.thinking is not None:
        total += state.thinking.nbytes()
    if state.ncp is not None:
        total += state.ncp.nbytes()
    if state.draft_mains is not None:
        total += sum(_array_bytes(m) for m in state.draft_mains)
    return total


class StatePool:
    """按行持有解码状态的池子：行号 = 请求在当前 batch 里的下标。"""

    def __init__(self, config, capacity: int = 0):
        self.config = config
        self.capacity = 0
        self.cache: Optional[VibyCache] = None
        if capacity > 0:
            self.ensure(capacity)

    def ensure(self, batch: int, lens: Optional[list] = None) -> bool:
        """把池子调整到恰好 batch 行（0 = 释放整池）；已有行按原下标保留。

        lens 给出每行已处理的 token 数：给了就只搬用到的前缀（ratio=1 源层的
        压缩池长度 = 序列长度，整行搬对长序列太贵），不给则整行搬。
        """
        batch = max(int(batch), 0)
        if self.cache is not None and self.capacity == batch:
            return False
        new = VibyCache(self.config, batch) if batch > 0 else None
        if new is not None and self.cache is not None:
            for r in range(min(self.capacity, batch)):
                copy_state_row(new, r, self.cache, r, None if lens is None else lens[r])
        self.cache, self.capacity = new, batch
        return True

    def move(self, dst_row: int, src_row: int) -> None:
        """池内搬行（请求结束时用它把末行压实到空出来的行）。"""
        if dst_row == src_row or self.cache is None:
            return
        copy_state_row(self.cache, dst_row, self.cache, src_row, None)

    def memory_bytes(self) -> int:
        return cache_bytes(self.cache)
