"""DeepSeek-V4.1 的混合稀疏注意力（CSA2 + Lightning Indexer + 滑窗分支）。

结构（对照 V4.1 tech report §2.3 与官方 inference/model.py）：

1. **共享 K=V 的 MQA**：wkv 只产 1 个 head，同一个张量既当 K 又当 V；
   每个 head 的尾部 rope_head_dim 通道做交错对部分旋转，注意力输出端用
   同一个旋转按 -i 转回去（inverse=True），使 cache 只保留一种旋转形式。
2. **两条 KV 分支拼接**：query 能看窗口内的原始 KV（滑动窗口环），加上
   indexer 为它挑出的 index_topk 个压缩位置。
3. **分组低秩输出投影**：o_groups 组 → 组内 block-diagonal 的 wo_a →
   o_lora_rank → wo_b 回 hidden。

逐层模式（CSA2 的 Full / Reindex / Reuse）由 config.layer_mode 给出：
Full = 自己产压缩 KV 且自己跑 indexer；Reindex = 复用上游压缩 KV、自己跑
indexer；Reuse = 两者都复用上游。压缩 KV 与 indexer 的 K 由 kv_source_layers
指定的层产生（CED 边界层是解码段的全局源），其后同 ratio 的层共享同一份
cache——这正是 CED 的实现方式：解码层的全局 KV 来自编码器末层的 hidden。

位置约定：token_pos 统一是 [B,T] 的绝对位置（稠密路径用 [1,T] 广播，
解码用 [B,1]），这样同一份代码既能做整段 prefill，也能做"连续 batch 里
各请求长度不同"的逐步解码。
"""

import mlx.core as mx
from mlx import nn

from .norms import RMSNorm
from .rope import precompute_freqs_cis, rope_partial

NEG_INF = -1e30


def _cos_sin(cos_all, sin_all, pos, head_axis: bool):
    """取位置 pos 的 cos/sin；head_axis 时在 head 维前插一个广播轴。"""
    c, s = cos_all[pos], sin_all[pos]
    if head_axis:
        c, s = c[..., None, :], s[..., None, :]
    return c, s


def select_candidate_blocks(logits: mx.array, compress_lens, topk_blocks: int, block_size: int):
    """分层索引第一级：每个 query 只保留得分最高的 topk_blocks 个块。

    logits: [B,T,N]，不可达位置已经是 -inf（块得分 -inf 即"还不可达"）。
    compress_lens: 标量或可广播的每 query 可达压缩位置数。
    返回与 logits 同形状的 bool 掩码。
    """
    width = logits.shape[-1]
    pad = (-width) % block_size
    if pad:
        logits = mx.concatenate(
            [logits, mx.full((*logits.shape[:-1], pad), NEG_INF, dtype=logits.dtype)], axis=-1
        )
    blocks = logits.reshape(*logits.shape[:-1], -1, block_size)
    scores = mx.max(blocks, axis=-1)  # 块得分 = 块内最好位置
    num_blocks = scores.shape[-1]
    # 最新位置所在的那个块只填了一部分，可能被更老的满块压过去：钉住它。
    # scores 是 [B,T,num_blocks]，钉住条件必须广播成 [B,?,1]：
    #   - 标量 compress_lens（所有 query 相同）→ [1,1,1]
    #   - [B]（decode 每序列一个）→ [B,1,1]
    #   - [B,T]/[1,T]（prefill 每 query 一个）→ [B,T,1]
    # 原实现用 last[..., None]，在 decode（T=1、B>1）下会与 [B,T,num_blocks]
    # 广播成 [B,B,num_blocks]，候选池掩码凭空多出一个 batch 维。
    last = (compress_lens - 1) // block_size
    if not isinstance(last, mx.array):
        last = mx.array(last)
    if last.ndim == 0:
        last = last[None, None, None]
    elif last.ndim == 1:
        last = last[:, None, None]
    else:
        last = last[..., None]
    idx = mx.arange(num_blocks)
    scores = mx.where(idx == last, mx.array(float("inf"), dtype=scores.dtype), scores)
    k = min(topk_blocks, num_blocks)
    # tiebreak：按块下标给极小偏移，保证"阈值比较"与 argpartition 选出同一集合
    # （indexer 的 ReLU 打分里 0 值大量并列，没有 tiebreak 时两条路径会选出不同个数）
    sel = scores - mx.arange(num_blocks, dtype=scores.dtype) * 1e-7
    thr = mx.partition(sel, kth=num_blocks - k, axis=-1)[..., num_blocks - k]
    keep = (sel >= thr[..., None]) & (scores > NEG_INF)
    keep = mx.repeat(keep, block_size, axis=-1)
    return keep[..., :width]


def _topk_masks(score: mx.array, reach: mx.array, k: int, offset: int):
    """从打分矩阵里取出每 query 的 k 个压缩位置。

    返回 (keep_mask [B,T,N] bool, topk_idx [B,T,k'] int32)：
    - keep_mask："被选中且可达"的布尔掩码（稠密/prefill 路径用）；
    - topk_idx：按位置排序、加 offset、不可达为 -1 的下标（解码 gather 用）。

    索引一律 stop_gradient：MLX 不允许对 gather/scatter 的索引求 VJP。
    """
    B, T, N = score.shape
    if N == 0:
        return mx.zeros((B, T, 0), dtype=mx.bool_), mx.zeros((B, T, 0), dtype=mx.int32)
    k_eff = min(k, N)
    sg = mx.stop_gradient(score)
    # tiebreak：按位置下标给极小偏移。indexer 的 ReLU 打分并列很多，没有它
    # "阈值比较"会选出多于 k 个位置，与解码路径的 argpartition 结果不一致。
    sel_score = sg - mx.arange(N, dtype=sg.dtype) * 1e-7
    # 分数为 NEG_INF 的位置是"明确不可选"（分层索引的候选池之外），必须与
    # 不可达位置一样当成硬屏蔽：可达候选中不足 k 个时，阈值为 -inf 会让这些
    # 位置靠 tiebreak 混进 keep，候选池就不再是更深 indexer 的搜索域上界。
    selectable = sg > NEG_INF
    # keep：第 k 大分数当阈值（可达位置少于 k 时阈值为 -inf，恰好等价"全取可达"）
    thr = mx.partition(sel_score, kth=N - k_eff, axis=-1)[..., N - k_eff]
    keep = (sel_score >= thr[..., None]) & reach & selectable
    sel = mx.argpartition(-sel_score, kth=k_eff - 1, axis=-1)[..., :k_eff].astype(mx.int32)
    idx = mx.sort(sel, axis=-1)
    valid = mx.take_along_axis(reach, idx, axis=-1) & mx.take_along_axis(selectable, idx, axis=-1)
    idx = mx.where(valid, idx + offset, -1).astype(mx.int32)
    return keep, idx


class Compressor(nn.Module):
    """把 compress_ratio 个连续 token 用可学 softmax 门池化成一个 KV latent。

    ratio == 1 就是一次普通投影（CED 解码段的全局 KV 来源）。返回**未做
    RoPE** 的 latent：indexer 需要未旋转形式，由 Attention 决定何时旋转。
    组首位置 = j * ratio（一个 latent 代表它那组里第一个 token 的位置）。
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ratio = int(config.compress_ratios[layer_idx])
        self.head_dim = config.head_dim
        self.norm = RMSNorm(config.head_dim, config.norm_eps)
        self.wkv = nn.Linear(config.dim, config.head_dim, bias=False)
        self.wgate = nn.Linear(config.dim, config.head_dim, bias=False) if self.ratio > 1 else None

    def init_state(self, batch: int, dtype):
        shape = (batch, self.ratio, self.head_dim)
        return (
            mx.zeros(shape, dtype=dtype),
            mx.full(shape, -mx.inf, dtype=mx.float32),
            mx.zeros((batch,), dtype=mx.int32),
        )

    def step(self, x: mx.array, state):
        """解码一步（batch 内各序列位置可不同）。

        state: (kv [B,r,hd], score [B,r,hd], filled [B])。返回
        (pooled [B,1,hd], valid [B], 新 state)：valid 为该序列这一步是否
        刚好凑满一组（未凑满时 pooled 无效，由 reach 掩掉）。
        """
        B = x.shape[0]
        kv, sc = self.wkv(x)[:, 0], self.wgate(x)[:, 0]  # [B,hd]
        filled = state[2]
        slot = filled % self.ratio
        # 用 one-hot 掩码做函数式写入（MLX 的 .at[] 没有 set；where 版可微且可 compile）
        oh = (mx.arange(self.ratio)[None, :] == slot[:, None])[..., None]  # [B,r,1]
        kv_st = mx.where(oh, kv[:, None, :], state[0])
        sc_st = mx.where(oh, sc[:, None, :], state[1])
        new_filled = filled + 1
        valid = new_filled >= self.ratio
        pooled = self.norm(mx.sum(kv_st * mx.softmax(sc_st, axis=1), axis=1, keepdims=True))
        # 组满后必须把 score 缓冲清成 -inf：否则上一组的 token 会混进下一组的池化
        sc_st = mx.where(valid[:, None, None], mx.full_like(sc_st, -mx.inf), sc_st)
        return pooled, valid, (kv_st, sc_st, mx.where(valid, 0, new_filled))

    def __call__(self, x: mx.array, start_pos: int, state=None):
        """整段/分块 prefill：返回 (latent [B,n,hd] | None, 新 state, 组首位置 | None, first_group)。"""
        B, T, _ = x.shape
        r = self.ratio
        if r == 1:
            return self.norm(self.wkv(x)), None, start_pos + mx.arange(T)[None, :], start_pos
        kv, sc = self.wkv(x), self.wgate(x)
        filled = 0 if state is None else int(mx.max(state[2]).item())
        if filled:
            kv = mx.concatenate([state[0][:B, :filled], kv], axis=1)
            sc = mx.concatenate([state[1][:B, :filled], sc], axis=1)
        total = kv.shape[1]
        n = total // r
        rem = total - n * r
        base = (start_pos - filled) // r
        # 定长 [B,r,hd] 状态缓冲：解码时要按同一形状和 one-hot 写入，
        # 未填的位置 kv=0 / score=-inf（softmax 权重 0，不参与池化）。
        pad = r - rem
        tail_kv, tail_sc = kv[:, n * r :], sc[:, n * r :]
        if pad:
            tail_kv = mx.concatenate([tail_kv, mx.zeros((B, pad, self.head_dim), dtype=kv.dtype)], axis=1)
            tail_sc = mx.concatenate([tail_sc, mx.full((B, pad, self.head_dim), -mx.inf, dtype=mx.float32)], axis=1)
        new_state = (tail_kv, tail_sc, mx.full((B,), rem, dtype=mx.int32))
        if n == 0:
            return None, new_state, None, base
        kv_g = kv[:, : n * r].reshape(B, n, r, self.head_dim)
        sc_g = sc[:, : n * r].reshape(B, n, r, self.head_dim)
        pooled = self.norm(mx.sum(kv_g * mx.softmax(sc_g, axis=2), axis=2))
        pos = (base + mx.arange(n))[None, :] * r
        return pooled, new_state, pos, base


class Indexer(nn.Module):
    """Lightning Indexer：给每个 query 选 index_topk 个压缩位置。

    一路小注意力：q（来自与主注意力共享的 qr 低秩表示）对一个共享的 index K
    打分，ReLU 后按 head 权重（weights_proj）合并。候选层额外产出一级候选块
    掩码，更深的 indexer 只在候选池里打分（分层稀疏索引）。
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_k = layer_idx in config.kv_source_layers
        self.is_candidate_source = layer_idx == config.candidate_source_layer
        self.uses_candidates = 0 <= config.candidate_source_layer < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

    def index_keys(self, latent: mx.array, pos, cos_all, sin_all):
        """latent（未旋转）→ 归一化 + 部分 RoPE 的 index K。"""
        c, s = _cos_sin(cos_all, sin_all, pos, False)
        return rope_partial(self.k_norm(self.wk(latent)), c, s, self.rope_head_dim)

    def scores(self, x, qr, index_k, token_pos, reach, cos_all, sin_all):
        """返回 [B,T,N] 的打分（不可达位置 = -inf，fp32）。"""
        B, T, _ = x.shape
        q = self.wq_b(qr).reshape(B, T, self.n_heads, self.head_dim)
        c, s = _cos_sin(cos_all, sin_all, token_pos, True)
        q = rope_partial(q, c, s, self.rope_head_dim)
        sc = mx.einsum("bthd,bnd->bthn", q, index_k).astype(mx.float32)
        sc = mx.maximum(sc, 0.0)
        w = self.weights_proj(x).astype(mx.float32) * (self.softmax_scale * self.n_heads ** -0.5)
        sc = mx.sum(sc * w[:, :, :, None], axis=2)  # [B,T,N]
        return mx.where(reach, sc, NEG_INF)


def _group_doc_mask(ratio: int, segment_ids: mx.array):
    """压缩组的文档掩码 [B,T,N]：组内 token 与 query 同文档才可见。"""
    B, T = segment_ids.shape
    n = T // ratio
    if n == 0:
        return mx.zeros((B, T, 0), dtype=mx.bool_)
    g = segment_ids[:, : n * ratio].reshape(B, n, ratio)
    clean = mx.all(g == g[:, :, :1], axis=-1)  # 组内同文档
    return clean[:, None, :] & (g[:, :, 0][:, None, :] == segment_ids[:, :, None])


def _group_pad_mask(ratio: int, pad_mask: mx.array):
    """压缩组的 pad 掩码 [B,T,N]：组内全是真实 token 才可见。"""
    B, T = pad_mask.shape
    n = T // ratio
    if n == 0:
        return mx.zeros((B, T, 0), dtype=mx.bool_)
    g = pad_mask[:, : n * ratio].reshape(B, n, ratio)
    return mx.all(g, axis=-1)[:, None, :] & mx.ones((1, T, 1), dtype=mx.bool_)


class Attention(nn.Module):
    """一层 V4.1 注意力：滑窗 K=V MQA + 压缩分支 + 分组低秩输出。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.window_size
        self.eps = config.norm_eps
        self.ratio = int(config.compress_ratios[layer_idx])
        self.mode = config.layer_mode(layer_idx)
        self.is_kv_source = layer_idx in config.kv_source_layers
        self.is_index_source = layer_idx in config.index_source_layers
        self.softmax_scale = self.head_dim ** -0.5

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, config.dim, bias=False)

        self.compressor = Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = Indexer(config, layer_idx) if self.is_index_source else None
        # 每层一张 RoPE 表：压缩层用 compress_rope_theta + YaRN；纯滑窗层用
        # rope_theta 且不做 YaRN（对应参考实现 original_seq_len=0 的分支）。
        if self.ratio:
            seq, theta = config.original_seq_len, config.compress_rope_theta
        else:
            seq, theta = 0, config.rope_theta
        self.freq_cos, self.freq_sin = precompute_freqs_cis(
            config.rope_head_dim,
            config.max_seq_len,
            seq,
            theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        # RoPE 表是常量：留在 parameters() 里（checkpoint 随权重一起落盘、
        # 换 max_seq_len 时由 trainer 重算），但不进 trainable_parameters()，
        # 否则优化器会把它当普通矩阵更新/衰减。
        self.freeze(recurse=False, keys=["freq_cos", "freq_sin"])

    # ------------------------------------------------------------------
    def _q(self, x, token_pos):
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(*x.shape[:2], self.n_heads, self.head_dim)
        c, s = _cos_sin(self.freq_cos, self.freq_sin, token_pos, True)
        return qr, rope_partial(q, c, s, self.rope_head_dim)

    def _window_kv(self, x, token_pos):
        kv = self.kv_norm(self.wkv(x))
        c, s = _cos_sin(self.freq_cos, self.freq_sin, token_pos, False)
        return rope_partial(kv, c, s, self.rope_head_dim)

    def _out_proj(self, o, token_pos):
        c, s = _cos_sin(self.freq_cos, self.freq_sin, token_pos, True)
        o = rope_partial(o, c, s, self.rope_head_dim, inverse=True)
        B, T = o.shape[:2]
        g = self.o_groups
        o = o.reshape(B, T, g, -1)
        wo_a = self.wo_a.weight.reshape(g, self.o_lora_rank, -1)
        o = mx.einsum("btgd,grd->btgr", o.astype(wo_a.dtype), wo_a)
        return self.wo_b(o.reshape(B, T, -1))

    # ------------------------------------------------------------------
    def _compress_dense(self, x, qr, token_pos, shared, cache, start_pos,
                        segment_ids=None, pad_mask=None):
        """压缩分支（prefill / 训练）：产出/复用压缩 KV 并算出 top-k 掩码。"""
        B, T, _ = x.shape
        ratio = self.ratio
        N = (start_pos + T) // ratio
        keep = None
        latent = None
        first_group = start_pos // ratio
        pos = None
        if self.is_kv_source:
            state = None
            if cache is not None and cache.kv_state is not None:
                state = cache.kv_state
            latent, new_state, pos, first_group = self.compressor(x, start_pos, state)
            if cache is not None and new_state is not None:
                cache.kv_state = new_state
                cache.filled = int(mx.max(new_state[2]).item())
        # indexer 需要未旋转的 latent，故先跑 indexer，再写旋转后的压缩 KV
        if self.is_index_source:
            idxo = self.indexer
            if idxo.owns_k and latent is not None:
                k = idxo.index_keys(latent, pos, self.freq_cos, self.freq_sin)
                if cache is not None:
                    cache.index_k[:B, first_group : first_group + k.shape[1]] = k
                    shared.index_k = cache.index_k[:, :N]
                else:
                    shared.index_k = k
            if shared.index_k is not None:
                # 注意轴：token_pos 是 [1,T]，要显式补成 [1,T,1] 才能和 [1,1,N] 按 query 广播
                reach = mx.arange(N)[None, None, :] < ((token_pos + 1) // ratio)[..., None]
                # 文档/pad 隔离必须在选择阶段生效：否则候选块与 top-k 会按
                # 别的文档的内容来挑，packed 序列会跨文档泄漏。
                if segment_ids is not None:
                    reach = reach & _group_doc_mask(ratio, segment_ids)
                if pad_mask is not None:
                    reach = reach & _group_pad_mask(ratio, pad_mask)
                sc = idxo.scores(
                    x, qr, shared.index_k, token_pos, reach, self.freq_cos, self.freq_sin
                )
                if idxo.is_candidate_source:
                    shared.candidates = select_candidate_blocks(
                        sc,
                        (token_pos + 1) // ratio,
                        idxo.candidate_topk_blocks,
                        idxo.candidate_block_size,
                    )
                elif idxo.uses_candidates and shared.candidates is not None:
                    sc = mx.where(shared.candidates, sc, NEG_INF)
                keep, topk = _topk_masks(sc, reach, idxo.index_topk, 0)
                shared.keep_mask, shared.topk_idx = keep, topk
        else:
            keep = shared.keep_mask
        # 压缩 KV 入池：必须是 RoPE 之后的（indexer 用未旋转形式，已在上面用完）
        if self.is_kv_source and latent is not None:
            c, s = _cos_sin(self.freq_cos, self.freq_sin, pos, False)
            lat = rope_partial(latent, c, s, self.rope_head_dim)
            if cache is not None:
                cache.compress_kv[:B, first_group : first_group + lat.shape[1]] = lat
            shared.compress_kv = lat
        if cache is not None:
            # 解码/分块 prefill：池子在源层的 LayerCache 上，直接读全量前缀
            comp = cache.src_cache.compress_kv[:B, :N] if N else None
        else:
            comp = shared.compress_kv
        return comp, keep

    # ------------------------------------------------------------------
    def __call__(self, x, start_pos: int, shared, cache=None, segment_ids=None, pad_mask=None):
        """prefill / 训练路径：窗口与压缩两组掩码拼在一次 sdpa 里做稠密计算。"""
        B, T, _ = x.shape
        token_pos = (start_pos + mx.arange(T))[None, :]  # [1,T]，对所有 batch 相同
        qr, q = self._q(x, token_pos)
        kv_win = self._window_kv(x, token_pos)
        # sdpa 的 mask 用 [B,T] 的绝对位置：[1,T] 广播
        tpos = token_pos[0]

        L = 0
        kv_hist = None
        if cache is not None and start_pos > 0:
            L = min(self.window_size, start_pos)
            slots = (start_pos - L + mx.arange(L)) % self.window_size
            kv_hist = cache.window[:, slots]
        if kv_hist is not None:
            kv_all = mx.concatenate([kv_hist, kv_win], axis=1)
            pos_all = mx.concatenate([start_pos - L + mx.arange(L), tpos])
        else:
            kv_all = kv_win
            pos_all = tpos
        if cache is not None:
            n = min(self.window_size, T)
            if n:
                slots = (start_pos + T - n + mx.arange(n)) % self.window_size
                cache.window[:, slots] = kv_win[:, T - n :]

        visible = (pos_all[None, :] <= tpos[:, None]) & (pos_all[None, :] > tpos[:, None] - self.window_size)
        if segment_ids is not None:
            if start_pos != 0:
                raise ValueError("segment_ids 只在整段 prefill（start_pos=0）时支持")
            visible = visible & (segment_ids[:, None, :] == segment_ids[:, :, None])
        if pad_mask is not None:
            visible = visible & pad_mask[:, None, :]
        mask = visible[None, None, :, :] if visible.ndim == 2 else visible[:, None, :, :]

        if self.ratio > 0:
            comp, keep = self._compress_dense(
                x, qr, token_pos, shared, cache, start_pos, segment_ids, pad_mask
            )
            if comp is not None:
                kv_all = mx.concatenate([kv_all, comp], axis=1)
                cmask = mx.arange(comp.shape[1])[None, None, :] < (
                    (tpos + 1) // self.ratio
                )[None, :, None]
                if keep is not None:
                    cmask = cmask & keep
                if segment_ids is not None:
                    cmask = cmask & _group_doc_mask(self.ratio, segment_ids)
                if pad_mask is not None:
                    cmask = cmask & _group_pad_mask(self.ratio, pad_mask)
                cm = cmask[None, None, :, :] if cmask.ndim == 2 else cmask[:, None, :, :]
                if mask.shape[0] != cm.shape[0]:
                    mask = mx.broadcast_to(mask, (cm.shape[0],) + mask.shape[1:])
                mask = mx.concatenate([mask, cm], axis=-1)

        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            kv_all[:, None, :, :],
            kv_all[:, None, :, :],
            scale=self.softmax_scale,
            mask=mask,
            sinks=self.attn_sink,
        ).transpose(0, 2, 1, 3)
        return self._out_proj(o, token_pos)

    # ------------------------------------------------------------------
    def decode(self, x, start_pos, shared, cache):
        """单步解码：滑窗环 + indexer 挑出的压缩位置，gather 后做注意力。

        start_pos 可以是 int，也可以是 [B] 的逐序列位置（连续 batch 里各
        请求长度不同）：窗口按各自槽位写入、按各自绝对位置取用；压缩池与
        index K 池按各自的分组对齐写入，池长取 batch 内最大值并用 reach 掩掉。
        """
        B, T, _ = x.shape
        if T != 1:
            raise ValueError("decode 只接受单 token")
        if not isinstance(start_pos, mx.array):
            start_pos = mx.full((B,), int(start_pos), dtype=mx.int32)
        start_pos = start_pos.astype(mx.int32)
        token_pos = start_pos[:, None]  # [B,1]
        bidx = mx.arange(B)
        qr, q = self._q(x, token_pos)
        kv_new = self._window_kv(x, token_pos)
        cache.window[bidx, start_pos % self.window_size] = kv_new[:, 0]

        win_pos = start_pos[:, None] - (self.window_size - 1) + mx.arange(self.window_size)[None, :]
        win_idx = mx.where(win_pos >= 0, win_pos % self.window_size, -1)[:, None, :].astype(mx.int32)
        kv_src = cache.window
        comp_idx = None

        if self.ratio > 0:
            offset = self.window_size
            pool = cache.src_cache
            latent, pos, first_group, valid = None, None, None, None
            if self.is_kv_source:
                if self.ratio == 1:
                    latent = self.compressor.norm(self.compressor.wkv(x))
                    first_group = start_pos
                    valid = mx.ones((B,), dtype=mx.bool_)
                else:
                    state = cache.kv_state
                    if state is None:
                        state = self.compressor.init_state(B, x.dtype)
                    latent, valid, state = self.compressor.step(x, state)
                    cache.kv_state = state
                    first_group = (start_pos + 1 - self.ratio) // self.ratio
                # cos/sin 按 (*pos.shape, d/2) 取用；解码时位置是 [B,1]，
                # 与 latent/x 的 [B,1,...] 对齐（[B] 会被广播进 T 轴）
                pos = (first_group * self.ratio)[:, None]
                c, s = _cos_sin(self.freq_cos, self.freq_sin, pos, False)
                lat = rope_partial(latent, c, s, self.rope_head_dim)
                # 没凑满一组的序列不能写池子（会覆盖上一组的有效条目），
                # 改写到池尾的丢弃槽里，靠 reach 掩掉。
                wslot = mx.where(valid, first_group, cache.pool_scratch)
                cache.compress_kv[bidx, wslot] = mx.where(
                    valid[:, None], lat[:, 0], mx.zeros_like(lat[:, 0])
                )
            N = (start_pos + 1) // self.ratio  # [B]
            n_max = int(mx.max(N).item()) if B else 0
            if self.is_index_source:
                idxo = self.indexer
                if idxo.owns_k and latent is not None:
                    k = idxo.index_keys(latent, pos, self.freq_cos, self.freq_sin)
                    wslot = mx.where(valid, first_group, cache.pool_scratch)
                    cache.index_k[bidx, wslot] = mx.where(
                        valid[:, None], k[:, 0], mx.zeros_like(k[:, 0])
                    )
                if idxo.owns_k and n_max:
                    shared.index_k = cache.index_k[:, :n_max]
                if n_max:
                    reach = (mx.arange(n_max)[None, :] < N[:, None])[:, None, :]  # [B,1,N]
                    sc = idxo.scores(
                        x, qr, shared.index_k, token_pos, reach, self.freq_cos, self.freq_sin
                    )
                    if idxo.is_candidate_source:
                        shared.candidates = select_candidate_blocks(
                            sc, N, idxo.candidate_topk_blocks, idxo.candidate_block_size
                        )
                    elif idxo.uses_candidates and shared.candidates is not None:
                        sc = mx.where(shared.candidates, sc, NEG_INF)
                    _, comp_idx = _topk_masks(sc, reach, idxo.index_topk, offset)
                    shared.topk_idx = comp_idx
                else:
                    comp_idx = mx.zeros((B, 1, 0), dtype=mx.int32)
                    shared.topk_idx = comp_idx
            else:
                comp_idx = shared.topk_idx
            if n_max:
                kv_src = mx.concatenate([kv_src, pool.compress_kv[:, :n_max]], axis=1)
        if comp_idx is None:
            comp_idx = mx.zeros((B, 1, 0), dtype=mx.int32)
        idx = mx.concatenate([win_idx, comp_idx], axis=-1)
        valid_slots = idx >= 0
        gathered = mx.take_along_axis(
            kv_src[:, None, :, :], mx.maximum(idx, 0)[..., None], axis=2
        )
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            gathered,
            gathered,
            scale=self.softmax_scale,
            mask=valid_slots[:, None, :, :],
            sinks=self.attn_sink,
        ).transpose(0, 2, 1, 3)
        return self._out_proj(o, token_pos)
