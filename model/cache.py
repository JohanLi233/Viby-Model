"""V4.1 的解码状态。

每层持有三类状态（对照参考实现的 window_kv_cache / compress_kv_cache /
indexer.k_cache + 压缩器的 kv_state/score_state）：

- 滑动窗口环：最近 window_size 个 token 的 KV（已按绝对位置做过 RoPE）；
- 压缩 KV 池：只有 kv_source 层持有，被其后同 ratio 的层共享；
- 索引器 K 池：只有同时是 kv_source 的 index_source 层持有（owns_k）。

跨层共享的"当前步产物"（压缩 KV / index K / top-k 索引 / 候选块）放在
SharedAttnState 里，由层循环按顺序传递——层序保证源先写、消费者后读，
与参考实现的全局 shared_attn 运行时等价。
"""

import mlx.core as mx


class LayerCache:
    """单层解码状态。"""

    def __init__(self, config, layer_idx: int, batch_size: int = 1):
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.window_size = config.window_size
        hd = config.head_dim
        self.window = mx.zeros((batch_size, config.window_size, hd))
        self.ratio = int(config.compress_ratios[layer_idx])
        self.is_kv_source = layer_idx in config.kv_source_layers
        self.is_index_source = layer_idx in config.index_source_layers
        self.owns_index_k = self.is_index_source and self.is_kv_source
        max_seq = config.max_seq_len
        self.compress_kv = (
            mx.zeros((batch_size, max_seq // max(self.ratio, 1) + 1, hd))
            if (self.is_kv_source and self.ratio > 0)
            else None
        )
        self.kv_state = None
        self.score_state = None
        self.filled = 0  # 未成组的尾巴长度（ratio>1 时）
        self.pool_scratch = max_seq // max(self.ratio, 1)  # 池尾多留一行当丢弃槽
        self.index_k = (
            mx.zeros((batch_size, self.pool_scratch + 1, config.index_head_dim))
            if self.owns_index_k
            else None
        )


class SharedAttnState:
    """一次前向内跨层传递的注意力产物（对应参考实现的 SharedAttentionRuntime）。

    层序保证源先写、消费者后读，所以每类产物一个槽位就够，无需在两次
    前向之间清理。
    """

    __slots__ = (
        "compress_kv", "index_k", "keep_mask", "topk_idx", "candidates",
        "latent_pos", "sparse_selection", "reach", "win_idx", "pool_tokens",
    )

    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.keep_mask = None   # 稠密路径：被 indexer 选中的压缩位置 [B,T,N] bool
        self.topk_idx = None    # 解码路径：选中的压缩位置（含窗口 offset，-1 = 无效）
        self.candidates = None  # 分层索引的一级候选块掩码 [B,T,N] bool
        self.latent_pos = None  # 压缩组首的绝对位置 [n]
        self.sparse_selection = None  # (ratio, (indices, lengths)); training only
        self.reach = None       # 本段前向已算过的压缩可达掩码，同 ratio 的后续层复用
        self.win_idx = None     # decode：各层共用的滑窗 gather 下标 [B,1,W]
        self.pool_tokens = None # decode：batch 内最大 token 数（Python int，避免每层 .item()）


class VibyCache:
    """整个模型的解码缓存（逐层 LayerCache + 已处理长度）。"""

    def __init__(self, config, batch_size: int = 1):
        self.config = config
        self.layers = [LayerCache(config, i, batch_size) for i in range(config.n_layers)]
        self.start_pos = 0
        self.decode_max_pos = 0  # Python：当前步 batch 内最大绝对位置+1，decode 用来切池
        self.engram_prev = None  # [B, max_ngram-1] 最近 token id（Engram 哈希用）
        self._wire_sources()

    def _wire_sources(self):
        """消费者层直接指向源层的 LayerCache：压缩 KV 池 / index K 池都存在源层上。"""
        cfg = self.config
        for i, c in enumerate(self.layers):
            c.src_cache = self.layers[cfg.kv_source_of(i)]

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def rewind(self, offset: int):
        """回退 offset 个 token（前缀缓存命中 / 投机解码回滚用）。

        压缩池与索引器 K 池按整组回退：只有落在回退区间之外的组保留。
        """
        self.start_pos = max(0, self.start_pos - offset)
        self.decode_max_pos = self.start_pos
        for c in self.layers:
            c.filled = 0
            c.kv_state = None
            c.score_state = None
        self.engram_prev = None
        return self
