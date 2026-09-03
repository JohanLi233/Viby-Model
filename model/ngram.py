"""Engram 式条件记忆：多头哈希 n-gram 检索 + 上下文门控 + 膨胀短卷积。

对照 DeepSeek/PKU《Conditional Memory via Scalable Lookup》(arXiv 2601.07372)
的完整设计，替换旧的单表+静态门 n-gram：

- 多头哈希：每个 n-gram 阶 × K 个 hash 头各一张**素数大小**的子表，
  每头不同的滚动乘子让碰撞去相关；各头检索向量**拼接**成 d_mem 维记忆
  （旧实现是单表、各阶相加，碰撞噪声混在一起）。所有头共享一张物理
  大表（加偏移一次 gather）。
- 上下文门控：当前 hidden 作 query（RMS 单位化），记忆经 W_K/W_V 投影，
  α = sigmoid(q̂·k̂/√d) ∈ (0,1) 逐 token 抑制碰撞/歧义检索（旧实现的
  静态逐维 gate 无此能力）。W_V 零初始化 ⇒ 初始 Y=0，与关掉 n-gram
  的主干逐位一致（替代旧的零初始化 gate）。
- 膨胀 depthwise 因果卷积（kernel=4，dilation=max(orders)）+ SiLU +
  残差：扩大感受野、加非线性。卷积核零初始化。
- 表走独立 Adam 组（5× lr、wd=0，稀疏 gather 更新不适合 Muon），
  见 trainer/muon.py；W_V/卷积零初始化同理不进 MuonH（范数球半径 0
  会永久钉零）。

检索只依赖输入 token id（确定性寻址），因此 pre-stack 一次 gather，
门控在注入层用当时的 hidden 完成。解码把上一窗口的 token 尾与门控后
 ṽ 的卷积尾存在 KVCache.extras（ngram_tail / ngram_conv_tail）。
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VibyConfig
from .norms import _rms_unit

# 每头不同的滚动乘子（奇素数）：让各头碰撞结构去相关。
_POLY_MULTS = (131, 137, 149, 157, 163, 173, 179, 191)
_ORDER_SALT = {2: 2, 3: 5}
_CONV_KERNEL = 4


def _shift_right_ids(ids: mx.array, offset: int) -> mx.array:
    if offset <= 0:
        return ids
    pad = mx.zeros((ids.shape[0], offset), dtype=ids.dtype)
    return mx.concatenate([pad, ids[:, :-offset]], axis=1)


def _prev_prime(n: int) -> int:
    """不超过 n 的最大素数（n >= 2）。"""
    n = int(n)
    while n >= 2:
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                break
        else:
            return n
        n -= 1
    return 2


def _split_prime_sizes(total: int, n_parts: int) -> list[int]:
    """把总桶数近似均分成 n_parts 个素数（每张素数表模运算不共振）。"""
    base = max(2, int(total) // n_parts)
    return [_prev_prime(base) for _ in range(n_parts)]


def _segment_masked_tok(tok, seg, lag):
    """跨文档的历史 token 置 0（与 ShortConv / 注意力 doc_mask 同口径）。

    左补 -1，避免「无历史」与 segment_id=0 撞车（同 ShortConv）。
    """
    if seg is None or lag <= 0:
        return tok
    pad = mx.full((seg.shape[0], lag), -1, dtype=seg.dtype)
    prev = mx.concatenate([pad, seg[:, :-lag]], axis=1)
    return mx.where(seg == prev, tok, mx.zeros_like(tok))


def ngram_indices(
    ids: mx.array,
    table_size: int,
    order: int,
    segment_ids: Optional[mx.array] = None,
    mult: int = 131,
) -> mx.array:
    """因果 n-gram 桶下标 (B, T)。位置 t 只看 t 及更早的 token；左侧不足补 0。"""
    ids = ids.astype(mx.int64)  # 大表下 h*mult 会超 int32
    salt = _ORDER_SALT[int(order)]
    h = mx.full(ids.shape, salt, dtype=mx.int64)
    m = int(table_size)
    seg = None if segment_ids is None else segment_ids.astype(mx.int32)
    for lag in range(int(order) - 1, -1, -1):
        tok = _shift_right_ids(ids, lag).astype(mx.int64)
        tok = _segment_masked_tok(tok, seg, lag)
        h = (h * int(mult) + tok) % m
    return h.astype(mx.int32)


class NgramEmbedding(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.orders = tuple(int(o) for o in config.ngram_orders)
        self.heads = int(getattr(config, "ngram_heads", 8))
        self.context_len = max(self.orders) - 1
        d = config.hidden_size
        self.d_mem = int(getattr(config, "ngram_d_mem", 0) or 0) or d
        n_tables = len(self.orders) * self.heads
        self.d_head = self.d_mem // n_tables
        # 全部 (阶 × 头) 子表压平成一张物理表，下标加偏移后一次 gather。
        self.sizes = _split_prime_sizes(config.ngram_table_size, n_tables)
        offsets = [0]
        for s in self.sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self._offsets = mx.array(offsets, dtype=mx.int32)  # (n_tables,)
        self.table_size = int(sum(self.sizes))  # 素数取整后的实际总桶数
        std = self.d_head**-0.5
        self.table = mx.random.normal((self.table_size, self.d_head)) * std
        # 上下文门控（Engram §2.3）：hidden 作 query，记忆作 key/value。
        self.w_k = nn.Linear(self.d_mem, d, bias=False)
        # W_V 零初始化 ⇒ 初始 Y≡0，与无 n-gram 主干逐位一致；不进 MuonH
        # （范数球半径=0 会钉零），由 trainer/muon.py 分组到 AdamW。
        self.w_v = nn.Linear(self.d_mem, d, bias=False)
        self.w_v.weight = mx.zeros_like(self.w_v.weight)
        # 门控后 ṽ 的 RMSNorm + 膨胀 depthwise 因果卷积（零初始化）+ SiLU 残差
        self.norm_v = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.dilation = max(self.orders)
        self.conv_w = mx.zeros((_CONV_KERNEL, d))
        # logits 残差尺度。仅 ngram_logit_skip 时存在；零初始化 ⇒ skip 起步为 0。
        if bool(getattr(config, "ngram_logit_skip", False)):
            self.logit_scale = mx.zeros((1,))
        # 置信门尺度。仅 ngram_conf_gate 时存在；零初始化 ⇒ g≡1。
        if bool(getattr(config, "ngram_conf_gate", False)):
            self.conf_scale = mx.zeros((1,))

    def lookup(
        self,
        input_ids: mx.array,
        tail: Optional[mx.array] = None,
        segment_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """多头检索并拼接成 (B, T, d_mem) 记忆向量（未过门）。"""
        ids = input_ids.astype(mx.int32)
        take = ids.shape[1]
        seg = segment_ids
        if tail is not None:
            ids = mx.concatenate([tail.astype(mx.int32), ids], axis=1)
            # decode / 分段 prefill 不打包文档；tail 与当前段的 segment 对不齐
            seg = None
        idxs = []
        i = 0
        for order in self.orders:
            for k in range(self.heads):
                idx = ngram_indices(
                    ids,
                    self.sizes[i],
                    order,
                    segment_ids=seg,
                    mult=_POLY_MULTS[k % len(_POLY_MULTS)],
                )
                idxs.append(idx[:, -take:])
                i += 1
        # 一次 gather 取全部头：(NH, B, T) + 偏移 → (NH, B, T, d_head)，
        # 再拼成 (B, T, d_mem)。逐头 gather 会产生 NH 份稠密 VJP。
        all_idx = mx.stack(idxs, axis=0) + self._offsets[:, None, None]
        nh, b, t = all_idx.shape
        e = self.table[all_idx]  # (NH, B, T, d_head)
        e = e.transpose(1, 2, 0, 3).reshape(b, t, nh * self.d_head)
        return e

    def fuse(
        self,
        hidden: mx.array,
        e: mx.array,
        conv_state: Optional[mx.array] = None,
        segment_ids: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """门控 + 卷积融合，返回 (Y, 新卷积尾)。

        hidden (B,T,D) 或 iHC 的 (B,T,M,D)（逐流独立门控，共享 W_K/W_V）。
        conv_state 是上一窗口门控后 ṽ 的尾部 (B, (K-1)*δ, ...)（decode 用，
        None 视为全零，与训练零填充同口径）。Y = SiLU(Conv(RMSNorm(ṽ))) + ṽ。
        """
        extra_stream = hidden.ndim == 4
        k = self.w_k(e)
        v = self.w_v(e).astype(hidden.dtype)
        q = _rms_unit(hidden)
        if extra_stream:
            alpha = mx.sigmoid(
                (q * _rms_unit(k)[:, :, None, :]).sum(-1, keepdims=True)
                * hidden.shape[-1] ** -0.5
            )
            vg = (alpha * v[:, :, None, :]).astype(hidden.dtype)
        else:
            alpha = mx.sigmoid(
                (q * _rms_unit(k)).sum(-1, keepdims=True) * hidden.shape[-1] ** -0.5
            )
            vg = (alpha * v).astype(hidden.dtype)
        K, dl = _CONV_KERNEL, self.dilation
        tail_len = (K - 1) * dl
        if conv_state is None:
            conv_state = mx.zeros(
                vg.shape[:1] + (tail_len,) + vg.shape[2:], dtype=vg.dtype
            )
        hist = mx.concatenate([conv_state, vg], axis=1)  # (B, tail+T, [M,] D)
        hn = self.norm_v(hist)
        w = self.conv_w.astype(hn.dtype)
        y = hn * w[0]
        for j in range(1, K):
            sh = j * dl
            xs = mx.concatenate(
                [mx.zeros(hn.shape[:1] + (sh,) + hn.shape[2:], dtype=hn.dtype),
                 hn[:, : hn.shape[1] - sh]],
                axis=1,
            )
            if segment_ids is not None:
                # 跨文档 tap 清零（同哈希侧口径）。hist 前缀是零历史，
                # 对应 mask 置 1（乘零历史无影响），只对 T 段重验边界。
                seg = segment_ids.astype(mx.int32)
                pad = mx.full((seg.shape[0], sh), -1, dtype=seg.dtype)
                prev = mx.concatenate([pad, seg[:, :-sh]], axis=1)
                same = (seg == prev).astype(hn.dtype)
                ones = mx.ones(
                    (seg.shape[0], tail_len), dtype=hn.dtype
                )
                mask = mx.concatenate([ones, same], axis=1)
                mask = mask.reshape(mask.shape[0], mask.shape[1],
                                    *([1] * (hn.ndim - 2)))
                xs = xs * mask
            y = y + xs * w[j]
        y = y[:, tail_len:]
        Y = nn.silu(y) + vg
        new_tail = hist[:, hist.shape[1] - tail_len:]
        return Y, new_tail

    def confidence_write_gate(self, ngram_raw: mx.array, unembed_weight: mx.array):
        """g = 1 − s · stopgrad(max softmax(e @ W_U^T))，形状 (B, T, 1)。

        置信度对表向量截断梯度，避免把表训成故意平坦以维持 g=1。
        s=0 时 g≡1。需要 d_mem == hidden（config 校验保证）。
        """
        logits = ngram_raw @ unembed_weight.T
        conf = mx.max(
            mx.softmax(logits.astype(mx.float32), axis=-1),
            axis=-1,
            keepdims=True,
        )
        conf = mx.stop_gradient(conf)
        scale = self.conf_scale.astype(ngram_raw.dtype)
        return 1.0 - scale * conf.astype(ngram_raw.dtype)
