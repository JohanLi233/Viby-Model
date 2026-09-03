"""分页 KV 池：page 0 恒为零占位，序列从 page 1 起分配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


def _scatter_set_pages(
    pages: mx.array, linear_idx: mx.array, src: mx.array
) -> mx.array:
    """沿 linearized (page, slot) 写入。``pages``: (P, H, S, D)，``src``: (N, H, D)。"""
    p, h, s, d = pages.shape
    flat = pages.transpose(0, 2, 1, 3).reshape((p * s, h, d))
    old = flat[linear_idx]
    flat = flat.at[linear_idx].add(src - old)
    return flat.reshape((p, s, h, d)).transpose(0, 2, 1, 3)


class PagePool:
    def __init__(
        self,
        num_layers: int,
        n_heads: int,
        page_size: int,
        k_dim: int,
        v_dim: int,
        num_pages: int,
        dtype,
    ):
        if num_pages < 1:
            raise ValueError("num_pages 必须 >= 1")
        if page_size < 1:
            raise ValueError("page_size 必须 >= 1")
        self.num_layers = int(num_layers)
        self.n_heads = int(n_heads)
        self.page_size = int(page_size)
        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim)
        self.dtype = dtype
        # +1：下标 0 是永不分配的零页
        p = int(num_pages) + 1
        self.keys = [
            mx.zeros((p, n_heads, page_size, k_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.values = [
            mx.zeros((p, n_heads, page_size, v_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        mx.eval(*self.keys, *self.values)
        self.refcount = [0] * p
        self.refcount[0] = 1
        self.free = list(range(1, p))

    @property
    def num_free(self) -> int:
        return len(self.free)

    def _set_page(self, layer: int, pid: int, src_k: mx.array, src_v: mx.array) -> None:
        """把指定层的 page 写成 src（MLX ArrayAt 没有 set，用 add(src-old)）。"""
        pid = int(pid)
        layer = int(layer)
        self.keys[layer] = self.keys[layer].at[pid].add(src_k - self.keys[layer][pid])
        self.values[layer] = (
            self.values[layer].at[pid].add(src_v - self.values[layer][pid])
        )

    def _clear_pages(self, pids: list[int]) -> None:
        if not pids:
            return
        z_k = mx.zeros((self.n_heads, self.page_size, self.k_dim), dtype=self.dtype)
        z_v = mx.zeros((self.n_heads, self.page_size, self.v_dim), dtype=self.dtype)
        for pid in pids:
            for layer in range(self.num_layers):
                self._set_page(layer, pid, z_k, z_v)
        mx.eval(*self.keys, *self.values)

    def alloc(self, n: int) -> list[int]:
        n = int(n)
        if n < 0:
            raise ValueError("alloc 数量不能为负")
        if n == 0:
            return []
        if n > len(self.free):
            raise MemoryError(f"KV page 不足：需要 {n}，剩余 {len(self.free)}")
        out = self.free[:n]
        self.free = self.free[n:]
        for pid in out:
            self.refcount[pid] = 1
        self._clear_pages(out)
        return out

    def retain(self, pages: list[int]) -> None:
        for pid in pages:
            if pid == 0:
                continue
            self.refcount[pid] += 1

    def free_pages(self, pages: list[int]) -> None:
        for pid in pages:
            if pid == 0:
                continue
            self.refcount[pid] -= 1
            if self.refcount[pid] < 0:
                raise RuntimeError(f"page {pid} refcount 下溢")
            if self.refcount[pid] == 0:
                self.free.append(pid)

    def cow(self, page_id: int) -> int:
        """共享页写前拷贝。返回新 page id（已是唯一引用）。"""
        pid = int(page_id)
        if pid == 0:
            raise ValueError("不能 COW 占位页")
        if self.refcount[pid] <= 1:
            return pid
        new_id = self.alloc(1)[0]
        for layer in range(self.num_layers):
            self._set_page(
                layer, new_id, self.keys[layer][pid], self.values[layer][pid]
            )
        mx.eval(*self.keys, *self.values)
        self.free_pages([pid])
        return new_id


@dataclass
class BatchSlots:
    """一次前向里整个 batch 共享的写入起点（各层不得各自递增）。"""

    seqlens: list[int]
    q_len: int
    token_mask: Optional[mx.array] = None
    q_lens: Optional[list[int]] = None

    def end_lens(self) -> list[int]:
        add = self.q_lens
        if add is None:
            add = [self.q_len] * len(self.seqlens)
        return [s + e for s, e in zip(self.seqlens, add)]


class PagedKVCache:
    """单层 paged cache：``update`` 把新 KV 写入 page 池并 gather 成稠密 (B,H,T,D)。"""

    __slots__ = ("pool", "layer_idx", "block_table", "slots", "extras")

    def __init__(
        self,
        pool: PagePool,
        layer_idx: int,
        block_table: mx.array,
        slots: BatchSlots,
        extras: Optional[dict] = None,
    ):
        self.pool = pool
        self.layer_idx = int(layer_idx)
        self.block_table = block_table
        self.slots = slots
        self.extras = extras if extras is not None else {}

    @property
    def offset(self) -> int:
        return max(self.slots.seqlens) if self.slots.seqlens else 0

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """keys/values: (B, t, H, D)。"""
        k = keys.transpose(0, 2, 1, 3)
        v = values.transpose(0, 2, 1, 3)
        bsz, _, t, _ = k.shape
        ps = self.pool.page_size
        starts = mx.array(self.slots.seqlens, dtype=mx.int32)
        pos = starts[:, None] + mx.arange(t, dtype=mx.int32)[None, :]
        page_ix = pos // ps
        slot = pos % ps
        page_id = mx.take_along_axis(self.block_table, page_ix, axis=1)
        linear = page_id * ps + slot
        k_src = k.transpose(0, 2, 1, 3).reshape((bsz * t, k.shape[1], k.shape[3]))
        v_src = v.transpose(0, 2, 1, 3).reshape((bsz * t, v.shape[1], v.shape[3]))
        idx = linear.reshape((bsz * t,))
        mask = self.slots.token_mask
        if mask is not None:
            m = mask.reshape((bsz * t,))
            idx = mx.where(m, idx, mx.zeros_like(idx))
            k_src = mx.where(m[:, None, None], k_src, mx.zeros_like(k_src))
            v_src = mx.where(m[:, None, None], v_src, mx.zeros_like(v_src))
        self.pool.keys[self.layer_idx] = _scatter_set_pages(
            self.pool.keys[self.layer_idx], idx, k_src
        )
        self.pool.values[self.layer_idx] = _scatter_set_pages(
            self.pool.values[self.layer_idx], idx, v_src
        )
        return self.gather()

    def gather(self) -> tuple[mx.array, mx.array]:
        ps = self.pool.page_size
        end = self.slots.end_lens()
        t_pad = max(end) if end else 0
        if t_pad == 0:
            bsz = self.block_table.shape[0]
            h = self.pool.n_heads
            k0 = mx.zeros((bsz, h, 0, self.pool.k_dim), dtype=self.pool.dtype)
            v0 = mx.zeros((bsz, h, 0, self.pool.v_dim), dtype=self.pool.dtype)
            return k0, v0
        n_blocks = (t_pad + ps - 1) // ps
        table = self.block_table[:, :n_blocks]
        gk = mx.take(self.pool.keys[self.layer_idx], table.reshape((-1,)), axis=0)
        gv = mx.take(self.pool.values[self.layer_idx], table.reshape((-1,)), axis=0)
        bsz = table.shape[0]
        gk = gk.reshape((bsz, n_blocks, self.pool.n_heads, ps, self.pool.k_dim))
        gv = gv.reshape((bsz, n_blocks, self.pool.n_heads, ps, self.pool.v_dim))
        k = gk.transpose(0, 2, 1, 3, 4).reshape(
            (bsz, self.pool.n_heads, n_blocks * ps, self.pool.k_dim)
        )
        v = gv.transpose(0, 2, 1, 3, 4).reshape(
            (bsz, self.pool.n_heads, n_blocks * ps, self.pool.v_dim)
        )
        return k[:, :, :t_pad, :], v[:, :, :t_pad, :]
