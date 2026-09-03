import mlx.core as mx


class KVCache:
    """预分配、按块增长的 KV cache。

    内部布局 (B, H, T, D)，decode 可直接送 ``scaled_dot_product_attention``，
    不必每步把整段历史从 (B, T, H, D) 转置一遍。

    兼容旧 tuple 接口：``cache[0]`` / ``cache[1]`` 返回 (B, T, H, D) 有效
    切片（与旧 ``concatenate`` 结果同布局），``shape[1]`` 即当前长度。

    ``extras``：解码期附属状态存放处（如 ShortConv 的历史输入尾部），
    随 cache 一起在生成循环里流转；rewind 截断时无法重建，直接清空。
    """

    __slots__ = ("keys", "values", "offset", "chunk", "extras")

    def __init__(self, chunk: int = 256):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk = int(chunk)
        self.extras = {}

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """写入当前段 keys/values (B, t, H, D)，返回有效 (B, H, T, D)。"""
        k = keys.transpose(0, 2, 1, 3)
        v = values.transpose(0, 2, 1, 3)
        t = k.shape[2]
        need = self.offset + t
        if self.keys is None or need > self.keys.shape[2]:
            cap = ((need + self.chunk - 1) // self.chunk) * self.chunk
            if self.keys is None:
                cap = max(cap, need + self.chunk)
            nk = mx.zeros((k.shape[0], k.shape[1], cap, k.shape[3]), dtype=k.dtype)
            nv = mx.zeros((v.shape[0], v.shape[1], cap, v.shape[3]), dtype=v.dtype)
            if self.keys is not None and self.offset > 0:
                nk[:, :, : self.offset] = self.keys[:, :, : self.offset]
                nv[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = nk, nv
        self.keys[:, :, self.offset : need] = k
        self.values[:, :, self.offset : need] = v
        self.offset = need
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]

    def rewind(self, offset: int) -> "KVCache":
        if offset < 0:
            raise ValueError(f"KVCache.rewind({offset}) 不能为负")
        # 允许 offset > 当前长度：投机解码在「全草稿命中后额外采样的
        # tail 恰好是 EOS」时，seq_len 含该 token，但 past_full 还没
        # 前向过它（EOS 截断故意跳过 tail 前向）。旧 tuple 切片
        # ``c[:, :seq_len]`` 超出时静默截到实际长度；这里对齐该语义。
        trace = self.extras.get("kda_trace")
        if offset < self.offset:
            # block 级 conv（attn_out/mlp_out）尾部：有逐步快照轨迹（投机
            # 解码在 prefill 后挂载）时按不超过 offset 的最近快照精确恢复；
            # 无轨迹时对应被截掉位置的尾部无法重建，丢弃，随后 ≤kernel-1
            # 个位置按零历史卷积（近似口径）。KDA 的 q/k/v conv 尾部与
            # SSM state 同理由 kda_trace 快照精确恢复（快照条目为
            # (offset, S, q尾, k尾, v尾)），无轨迹时才丢弃。
            for key in ("attn_out", "mlp_out"):
                ctr = self.extras.get(key + "_trace")
                if ctr:
                    base = None
                    for entry in ctr:
                        if entry[0] <= offset:
                            base = entry
                    if base is not None:
                        self.extras[key] = base[1]
                        self.extras[key + "_trace"] = [base]
                    else:
                        self.extras.pop(key, None)
                        self.extras[key + "_trace"] = []
                else:
                    self.extras.pop(key, None)
            ntr = self.extras.get("ngram_tail_trace")
            if ntr:
                base = None
                for entry in ntr:
                    if entry[0] <= offset:
                        base = entry
                if base is not None:
                    self.extras["ngram_tail"] = base[1]
                    self.extras["ngram_tail_trace"] = [base]
                else:
                    self.extras.pop("ngram_tail", None)
                    self.extras["ngram_tail_trace"] = []
            else:
                self.extras.pop("ngram_tail", None)
            nctr = self.extras.get("ngram_conv_tail_trace")
            if nctr:
                base = None
                for entry in nctr:
                    if entry[0] <= offset:
                        base = entry
                if base is not None:
                    self.extras["ngram_conv_tail"] = base[1]
                    self.extras["ngram_conv_tail_trace"] = [base]
                else:
                    self.extras.pop("ngram_conv_tail", None)
                    self.extras["ngram_conv_tail_trace"] = []
            else:
                self.extras.pop("ngram_conv_tail", None)
            if not trace:
                for key in ("q_conv", "k_conv", "v_conv", "kda_state"):
                    self.extras.pop(key, None)
        if trace:
            # 只保留不超过 offset 的最近快照作为下一轮验证的回滚点
            base = None
            for entry in trace:
                if entry[0] <= offset:
                    base = entry
            if base is not None:
                self.extras["kda_state"] = base[1]
                self.extras["q_conv"] = base[2]
                self.extras["k_conv"] = base[3]
                self.extras["v_conv"] = base[4]
                self.extras["kda_trace"] = [base]
            else:
                for key in ("kda_state", "q_conv", "k_conv", "v_conv"):
                    self.extras.pop(key, None)
                self.extras["kda_trace"] = []
        self.offset = min(int(offset), self.offset)
        return self

    def __getitem__(self, idx):
        arr = self.keys if idx == 0 else self.values
        return arr[:, :, : self.offset].transpose(0, 2, 1, 3)

    def __iter__(self):
        yield self[0]
        yield self[1]

    def __len__(self):
        return 2


def _seq_len_of_cache(cache) -> int:
    if cache is None:
        return 0
    if isinstance(cache, KVCache):
        return cache.offset
    offset = getattr(cache, "offset", None)
    if offset is not None and not isinstance(cache, (tuple, list)):
        return int(offset)
    return cache[0].shape[1]


def _rewind_cache(cache, offset: int):
    if cache is None:
        return None
    if isinstance(cache, KVCache):
        return cache.rewind(offset)
    return tuple(c[:, :offset] for c in cache)


def _rewind_cache_list(caches, offset: int) -> list:
    return [_rewind_cache(c, offset) for c in caches]


def _eval_kv_caches(caches) -> None:
    if not caches:
        return
    bufs = []
    for c in caches:
        if isinstance(c, KVCache):
            if c.keys is not None:
                bufs.append(c.keys)
                bufs.append(c.values)
        elif c is not None:
            bufs.extend(c)
    if bufs:
        mx.eval(*bufs)


def _offset_causal_mask(q_len: int, k_len: int, dtype):
    """query i 可见 key j iff j <= (k_len - q_len) + i。"""
    q = mx.arange(q_len)[:, None]
    k = mx.arange(k_len)[None, :]
    return mx.where(
        k <= q + (k_len - q_len),
        mx.array(0.0, dtype=dtype),
        mx.array(-mx.inf, dtype=dtype),
    )
