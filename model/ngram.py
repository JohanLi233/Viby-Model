"""哈希 n-gram 嵌入：挂在指定层入口的可卸载参数轴。

Qwen3.8-Flash-Next 把 20M bigram/trigram 挂在 layer 2。本实现是同形的
小表：一张 `ngram_table_size` × hidden 的查找表，用多项式哈希把
(t_{i-n+1}…t_i) 映到桶；逐维门 `gate` 零初始化 ⇒ 初始 Δ=0，与关掉
n-gram 的主干逐位一致。解码把上一窗口的 token 尾存在 KVCache.extras。
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VibyConfig

# 每步先 % table_size 再乘，int32 内安全（2^18 × 131 < 2^31）。
_POLY = 131
_ORDER_SALT = {2: 2, 3: 5}


def _shift_right_ids(ids: mx.array, offset: int) -> mx.array:
    if offset <= 0:
        return ids
    pad = mx.zeros((ids.shape[0], offset), dtype=ids.dtype)
    return mx.concatenate([pad, ids[:, :-offset]], axis=1)


def ngram_indices(
    ids: mx.array,
    table_size: int,
    order: int,
    segment_ids: Optional[mx.array] = None,
) -> mx.array:
    """因果 n-gram 桶下标 (B, T)。位置 t 只看 t 及更早的 token；左侧不足补 0。

    传入 segment_ids 时，跨文档的历史 token 置 0（与 ShortConv / 注意力
    doc_mask 同口径），避免 packed 序列把上一篇的尾巴写进下一篇的查找键。
    """
    ids = ids.astype(mx.int32)
    salt = _ORDER_SALT[int(order)]
    h = mx.full(ids.shape, salt, dtype=mx.int32)
    m = int(table_size)
    seg = None if segment_ids is None else segment_ids.astype(mx.int32)
    for lag in range(int(order) - 1, -1, -1):
        tok = _shift_right_ids(ids, lag)
        if seg is not None and lag > 0:
            # 左补 -1，避免「无历史」与 segment_id=0 撞车（同 ShortConv）。
            pad = mx.full((seg.shape[0], lag), -1, dtype=seg.dtype)
            prev = mx.concatenate([pad, seg[:, :-lag]], axis=1)
            tok = mx.where(seg == prev, tok, mx.zeros_like(tok))
        h = (h * _POLY + tok) % m
    return h


class NgramEmbedding(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.table_size = int(config.ngram_table_size)
        self.orders = tuple(int(o) for o in config.ngram_orders)
        self.context_len = max(self.orders) - 1
        d = config.hidden_size
        std = d**-0.5
        self.table = mx.random.normal((self.table_size, d)) * std
        self.gate = mx.zeros((d,))

    def __call__(
        self,
        input_ids: mx.array,
        tail: Optional[mx.array] = None,
        segment_ids: Optional[mx.array] = None,
    ) -> mx.array:
        ids = input_ids.astype(mx.int32)
        take = ids.shape[1]
        seg = segment_ids
        if tail is not None:
            ids = mx.concatenate([tail.astype(mx.int32), ids], axis=1)
            # decode / 分段 prefill 不打包文档；tail 与当前段的 segment 对不齐
            seg = None
        emb = None
        for order in self.orders:
            idx = ngram_indices(ids, self.table_size, order, segment_ids=seg)
            idx = idx[:, -take:]
            e = self.table[idx]
            emb = e if emb is None else emb + e
        return self.gate.astype(emb.dtype) * emb
