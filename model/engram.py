"""Engram：DeepSeek-V4.1 的 n-gram 条件记忆（PyTorch 参考实现 → MLX 移植）。

逐行对照 /tmp/dsv41_ref/engram.py 与 /tmp/dsv41_ref/model.py 的
ParallelEngramEmbedding / Engram（第 296-365 行）：

- build_compressed_token_map：token 的 decode 文本先过 normalizer 序列
  （NFKC → NFD → StripAccents → Lowercase → 空白折叠 → 单空格 sentinel →
  Strip → sentinel 还原）压到小 id 空间，" The" / "the" / "THE" 落到同一个
  id；含 U+FFFD 的部分字节 token 无法归一化，直接用 backend.id_to_token 的
  原始形式做 key。
- EngramLayout / compute_num_embeddings：每个 (层, n-gram 阶, 头) 在层表里
  独占一段素数长度的桶区间，素数按 层→阶→头 的顺序取用且全局不重复。
- compute_hash_multipliers：每层一个 np.random.default_rng(10007 * layer_id)，
  层间哈希去相关；乘子取奇数，上界压到 int64 不溢出。
- NgramHashState：压缩 id → 逐回看位置乘乘子 → 逐阶滚动 XOR → 各头模本桶
  素数 → 加桶偏移；回看在序列开头与 DEAD 处截断，n-gram 不跨 DEAD。
- Engram：查表 → wkv 出 (key, value) → 与残差流按 (token, hc 副本) 做 RMS
  归一化点积 → 带符号平方根 → sigmoid 门 → 写回残差流。

与参考实现的三处刻意偏差：

1. **无状态解码缓存**：参考实现预分配 [max_batch_size, max_seq_len] 的 int64
   cache，靠 start_pos 定位、跨 prefill/decode 复用；这里改为调用方显式传
   prev_tokens（最近 max_ngram_size-1 个 token），每次现算 n-gram 窗口。
   截断语义与参考实现完全一致（hash_ids 逐位相同），但没有定长大缓存，
   分块 prefill / 投机解码可直接复用同一接口。
2. **表不分片、不量化**：ParallelEngramEmbedding 的 world_size 行分片 + fp8
   块量化 + all_reduce 是训练侧并行存储实现；这里每层一张 fp32 全表一次
   gather，数学上就是同一张表反量化后的查表（fp32 精度只会更好）。
3. **int64**：mlx 0.32.2 的 Metal 后端实测支持 int64 的乘 / bitwise_xor /
   取模，故乘子与乘积全程 int64，与参考实现逐位一致。若换到只支持 int32 的
   mlx 版本，需把这里降级为 int32——此时乘积按 2³² 回绕，哈希值与参考实现
   不再相同（表仍可用，只是不再 bit-exact）。

移植注意：Engram 的初始化是「参考实现起点」的一部分——q_weight / k_weight
必须保持 ones（初始门 = 纯归一化点积），embed 表与 wkv 用 0.5/√fan_in 的截断
正态。若上层对全模型做统一的截断正态覆盖（旧 model/init.py 的
apply_trunc_normal_init 那种），需要跳过 q_weight / k_weight
（否则初始门不再是参考实现的起点）以及 embed 表（大表的二次随机化纯属浪费）。

n-gram 表是纯寻址（token → 行号），梯度只回传到被查到的行，训练侧建议单独
走 AdamW 组（稀疏 gather 更新不适合 Muon），见 trainer/muon.py。
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------- 素数 & 布局

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _trunc_normal(shape, std: float, bound: float = 2.0) -> mx.array:
    """TruncNormal(0, std²) 截在 ±bound（沿用仓库 trunc_normal 的口径）。

    这里自带一份而不是 import 全局初始化模块：engram 的参数初始化要独立
    可复现，不该跟着别的模块一起被重构掉。构造期允许少量 host 判断。
    """
    x = mx.random.normal(shape)
    for _ in range(4):
        if not bool((mx.abs(x) > bound).any().item()):
            break
        x = mx.where(mx.abs(x) > bound, mx.random.normal(shape), x)
    return mx.clip(x, -bound, bound) * std


def _is_prime(n: int) -> bool:
    """确定性 Miller-Rabin：前 12 个素数基可判定 3.3e24 以内所有 n。"""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL_PRIMES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set) -> int:
    """大于 start 且尚未分配过的最小素数（同参考实现；sympy.isprime 的等价物）。"""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def _build_primes(
    layer_ids: tuple, max_ngram_size: int, n_heads: int, vocab_size: int
) -> tuple:
    """[层][n-gram 阶][头] 的素数桶模数，按 层→阶→头 顺序取用且全局不重复。

    compute_num_embeddings 与 EngramLayout.from_config 共用这一处顺序，
    两边算出的素数必须一致（否则表行数与实际桶区间对不上）。
    """
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def _layer_rows(layer_primes: tuple) -> int:
    """该层所有 (阶, 头) 桶素数之和 = 该层表的行数。"""
    return sum(sum(heads) for heads in layer_primes)


def compute_num_embeddings(config) -> tuple:
    """按素数桶布局算出每层 n-gram 表的行数（= 该层各桶素数之和）。

    VibyConfig 拿不到 tokenizer 时先用它填 engram_num_embeddings 默认值；
    EngramLayout.from_config 用同一套素数顺序核对（断言 >= 桶总数）。
    """
    layer_ids = tuple(int(i) for i in config.engram_layer_ids)
    if not layer_ids:
        return ()
    primes = _build_primes(
        layer_ids,
        int(config.engram_max_ngram_size),
        int(config.engram_n_heads),
        int(config.engram_vocab_size),
    )
    return tuple(_layer_rows(layer) for layer in primes)


@dataclass(frozen=True)
class EngramLayout:
    """n-gram 哈希桶布局（同参考实现）。

    一个位置被 max_ngram_size - 1 个 n-gram（2-gram .. max_ngram_size-gram）
    哈希，每个阶再分 n_heads 个头。每个 (阶, 头) 在层表里独占一段素数长度的
    桶区间；素数按取用顺序生成且不重复，保证区间互不重叠。
    """

    max_ngram_size: int
    layer_ids: tuple  # 挂 Engram 的层 id（1-indexed）
    num_embeddings: tuple  # 每层表的行数
    primes: tuple  # [层][n-gram 阶][头] 桶模数
    n_heads: int
    head_dim: int

    @property
    def window(self) -> int:
        """哈希一个位置需要的回看长度 = max_ngram_size - 1。"""
        return self.max_ngram_size - 1

    @property
    def n_hash_cols(self) -> int:
        """每个位置的哈希列数 = (max_ngram_size - 1) * n_heads。"""
        return self.window * self.n_heads

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        """没有 engram 层时返回 None（与参考实现 from_args 一致）。"""
        layer_ids = tuple(int(i) for i in config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size = int(config.engram_max_ngram_size)
        n_heads = int(config.engram_n_heads)
        primes = _build_primes(
            layer_ids, max_ngram_size, n_heads, int(config.engram_vocab_size)
        )
        num_embeddings = tuple(
            int(n) for n in (getattr(config, "engram_num_embeddings", ()) or ())
        )
        if len(num_embeddings) != len(layer_ids):
            # 缺省/长度不符：按布局自算（VibyConfig 用 compute_num_embeddings 填的
            # 就是这个值，这里是没填时的兜底）
            num_embeddings = tuple(_layer_rows(layer) for layer in primes)
        for layer_id, rows, layer in zip(layer_ids, num_embeddings, primes):
            need = _layer_rows(layer)
            if rows < need:
                raise ValueError(
                    f"engram_num_embeddings[layer {layer_id}]={rows} 小于桶总数 {need}"
                )
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=primes,
            n_heads=n_heads,
            head_dim=int(config.engram_head_dim),
        )


def compute_hash_multipliers(
    layer_ids: tuple, max_ngram_size: int, tokenizer_vocab_size: int
) -> mx.array:
    """每个 (层, 回看阶) 一个奇数乘子，返回 [n_layers, max_ngram_size] int64。

    每层独立 RNG，层间哈希去相关。上界沿用参考实现的 int64 公式：保证
    token_id * multiplier 不溢出 int64（溢出回绕会让哈希依赖实现细节，
    跨设备/跨后端不可复现）。
    """
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // int(tokenizer_vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * int(layer_id))
        values = generator.integers(
            low=0, high=bound, size=(int(max_ngram_size),), dtype=np.int64
        )
        rows.append(values * 2 + 1)  # 奇数：与 2 的幂无公因子，XOR 后分布更散
    if not rows:
        return mx.zeros((0, int(max_ngram_size)), dtype=mx.int64)
    return mx.array(np.stack(rows))


# ------------------------------------------------------------- 压缩 id / 哈希


def build_compressed_token_map(tokenizer) -> tuple:
    """把每个 token 映射到「归一化后同形」的压缩 id，返回 (lookup, 表大小)。

    lookup[token_id] = 压缩 id；压缩表大小不只是边界检查——所有哈希乘子的
    上界都由它推出，两边不一致会静默改掉整张表。

    接受 transformers 的 PreTrainedTokenizerFast（取 backend_tokenizer）
    或裸的 tokenizers.Tokenizer。
    """
    from tokenizers import Regex, normalizers

    # 私有区字符：让「恰好一个空格」的 token 在 Strip() 后不与空 token/其他
    # token 撞车，之后还原成空格
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # decode 用训练时的同一把（Rust）tokenizer，不做 clean_up_tokenization_spaces
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    if hasattr(tokenizer, "__len__"):
        n_tokens = len(tokenizer)
    else:
        n_tokens = backend.get_vocab_size(with_added_tokens=True)

    key_to_new: dict = {}
    lookup = [0] * n_tokens
    for token_id in range(n_tokens):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # 半个 UTF-8 字节的 token：没有可归一化的文本，用原始形式做 key
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


class NgramHashState(nn.Module):
    """把每个位置映射到「以它结尾的各阶 n-gram」的桶下标。

    压缩 id → 逐回看位置乘乘子 → 逐阶滚动 XOR → 各头模本桶素数 → 加桶偏移。
    回看在序列开头和任意 DEAD 位置（图像 span 等）截断：一旦某个回看位置
    落在 DEAD 上，更远的回看也全部按 pad 处理，n-gram 永不跨过 DEAD。

    与参考实现的差别只在缓存：这里不持有 [B, max_seq_len] 大缓存，改为
    prev_tokens 显式传入最近 max_ngram_size-1 个 token（末位是最近的历史；
    None = 序列开头；负值 = DEAD）。截断语义不变。
    """

    DEAD = -1

    def __init__(
        self, config, layout: EngramLayout, token_map: list, compressed_vocab_size: int
    ):
        super().__init__()
        self._layout = layout
        self._max_ngram_size = int(layout.max_ngram_size)
        self._window = layout.window
        if self._max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size 必须 >= 2（至少要 2-gram）")
        self._n_hash_cols = layout.n_hash_cols

        rows = int(max(token_map)) + 1 if len(token_map) else 0
        if int(compressed_vocab_size) < rows:
            raise ValueError(
                f"compressed_vocab_size={compressed_vocab_size} < 压缩 id 上界 {rows}"
                "（乘子上界由它推出，会静默改变全部哈希）"
            )
        self._pad_id = int(token_map[int(config.engram_pad_id)])
        self._token_map = mx.array(np.asarray(token_map, dtype=np.int32))

        # 每层各桶素数的前缀和 = 每列桶区间的起点；参考实现 offsets 的形状是
        # [层, 阶×头]
        primes = np.asarray(layout.primes, dtype=np.int64)
        flat = primes.reshape(primes.shape[0], -1)
        offsets = np.zeros_like(flat)
        offsets[:, 1:] = np.cumsum(flat, axis=1)[:, :-1]
        self._primes = mx.array(primes)
        self._offsets = mx.array(offsets)
        self._multipliers = compute_hash_multipliers(
            layout.layer_ids, self._max_ngram_size, int(compressed_vocab_size)
        )

    def _compress(self, ids: mx.array, valid=None) -> mx.array:
        """token id → 压缩 id；valid 为 False（或 id 为负 = DEAD）的位置记 DEAD。

        负 id 不能直接 gather（MLX 按 Python 语义绕回表尾），先夹到 0 再抹掉。
        """
        ids = ids.astype(mx.int32)
        alive = ids >= 0 if valid is None else mx.logical_and(valid, ids >= 0)
        mapped = self._token_map[mx.maximum(ids, 0)]
        return mx.where(alive, mapped, mx.array(self.DEAD, dtype=mx.int32))

    def __call__(self, input_ids: mx.array, prev_tokens=None, token_mask=None) -> tuple:
        """返回 (hash_ids, new_prev_tokens)。

        input_ids: [B, T] 原始 token id；prev_tokens: [B, max_ngram_size-1]
        的最近历史（末位最近；None = 无历史；负值 = DEAD）；token_mask:
        [B, T] bool，False 的位置等价于 DEAD（不参与任何 n-gram）。

        hash_ids: [B, T, n_engram_layers, n_hash_cols] int64，每列已加桶偏移、
        取值落在本层表行数内，可直接 gather；
        new_prev_tokens: [B, max_ngram_size-1] int32，本步之后应传给下一步的
        历史（mask 掉的位置记 DEAD）。
        """
        input_ids = input_ids.astype(mx.int32)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids 必须是 [B, T]，得到 {input_ids.shape}")
        batch = input_ids.shape[0]
        window = self._window

        if prev_tokens is None:
            prev = mx.full((batch, window), self.DEAD, dtype=mx.int32)
        else:
            prev = prev_tokens.astype(mx.int32)
            if prev.shape != (batch, window):
                raise ValueError(f"prev_tokens 必须是 [B, {window}]，得到 {prev.shape}")
        # [B, window + T]：可回看的完整窗口（前缀是历史，负值/越界都算 DEAD）
        ext = mx.concatenate(
            [self._compress(prev), self._compress(input_ids, token_mask)], axis=1
        )
        length = ext.shape[1]
        idx = mx.arange(length, dtype=mx.int32)
        # 到 i 为止最近一个 DEAD（没有则 -1）→ run[i] = 以 i 结尾的连续有效长度。
        # 回看 s 合法 ⟺ s < run[i]；s >= run[i] 时被截断（含「跨过 DEAD」的累积
        # 屏蔽，与参考实现 rolling blocked 的语义逐位一致）。
        dead = ext < 0
        last_dead = mx.cummax(mx.where(dead, idx[None, :], -1), axis=1)
        run = idx[None, :] - last_dead

        pad_id = mx.array(self._pad_id, dtype=mx.int32)
        tokens = []
        for shift in range(self._max_ngram_size):
            source = ext[:, mx.maximum(idx - shift, 0)]
            tokens.append(mx.where(run > shift, source, pad_id))
        tokens = mx.stack(tokens, axis=-1)[:, window:, :]  # [B, T, max_ngram_size]

        # 一个回看阶一个 XOR：第 i 步的滚动值就是 (i+1)-gram 的哈希；每阶落在
        # 自己的素数桶区间里
        products = (
            tokens[:, :, None, :].astype(mx.int64) * self._multipliers[None, None]
        )
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self._max_ngram_size):
            rolling = mx.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self._primes[:, i - 1])
        hash_ids = mx.concatenate(hashes, axis=-1) + self._offsets[None, None]

        if token_mask is None:
            raw = input_ids
        else:
            raw = mx.where(token_mask, input_ids, mx.array(self.DEAD, dtype=mx.int32))
        full = mx.concatenate([prev, raw], axis=1)
        new_prev = full[:, full.shape[1] - window :]
        return hash_ids, new_prev


# ------------------------------------------------------------------ Engram 层


class EngramEmbedding(nn.Module):
    """n-gram 哈希表（对应参考实现的 ParallelEngramEmbedding，去掉分片/量化）。

    参考实现按 world_size 分片、fp8 存储 + block scale 查表时反量化；这里
    每层一张 fp32 全表一次 gather。查表是纯寻址，梯度只回传被查到的行。
    """

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        # 与全局初始化同口径：std = 0.5/√fan_in（fan_in = 表宽）。表很大时拒绝采样
        # 要多次 host 同步，纯推理加载 checkpoint 的路径应跳过构造期初始化。
        self.weight = _trunc_normal(
            (self.num_embeddings, self.dim), 0.5 / math.sqrt(self.dim)
        )

    def __call__(self, indices: mx.array) -> mx.array:
        return self.weight[indices]


class Engram(nn.Module):
    """把 n-gram 查表结果按「与残差流的匹配程度」门控后写回残差流。

    哈希 id 取出 n_hash_cols 行拼成记忆向量，wkv 变出一份每 hc 副本的 key
    和一个共享 value；门是残差流与 key 的归一化点积，逐 (token, hc 副本)
    各自 RMS 归一化（不跨副本联合）。
    """

    def __init__(self, config, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = int(layer_id)
        self.layer_hash_index = layout.layer_ids.index(self.layer_id)
        self.dim = int(config.dim)
        self.hc_mult = int(config.hc_mult)
        self.clamp_value = 1e-6
        self.eps = float(config.norm_eps)
        self._n_hash_cols = layout.n_hash_cols

        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim
        )
        kv_dim = self.dim * (self.hc_mult + 1)
        in_dim = layout.n_hash_cols * layout.head_dim
        self.wkv = nn.Linear(in_dim, kv_dim, bias=False)
        # 与全局初始化同口径（0.5/√fan_in）；单独构造 Engram 时也是这个分布
        self.wkv.weight = _trunc_normal((kv_dim, in_dim), 0.5 / math.sqrt(in_dim))
        # q/k 只以乘积出现，拆成两个 ones 不改变初始行为（初始门 = 纯归一化
        # 点积），但给了两组独立梯度
        self.q_weight = mx.ones((self.hc_mult, self.dim))
        self.k_weight = mx.ones((self.hc_mult, self.dim))

    def __call__(self, x: mx.array, hash_ids: mx.array, token_mask=None) -> mx.array:
        """x: [B, T, hc_mult, dim]；hash_ids: [B, T, n_hash_cols]（也接受
        NgramHashState 的整表输出 [B, T, n_layers, n_hash_cols]，自动取本层
        的列）；token_mask: [B, T] bool，False 的位置门压成 0、残差原样通过。
        """
        if hash_ids.ndim == 4:
            hash_ids = hash_ids[:, :, self.layer_hash_index, :]
        if hash_ids.ndim != 3 or hash_ids.shape[-1] != self._n_hash_cols:
            raise ValueError(
                f"hash_ids 必须是 [B, T, {self._n_hash_cols}]，得到 {hash_ids.shape}"
            )
        batch, seqlen, hc, dim = x.shape
        if hc != self.hc_mult or dim != self.dim:
            raise ValueError(
                f"x 必须是 [B, T, {self.hc_mult}, {self.dim}]，得到 {x.shape}"
            )

        # 表按 fp32 存储，查表后按 wkv 参数 dtype 计算（整模型 astype 时行为一致）
        rows = self.embed(hash_ids).astype(self.wkv.weight.dtype)
        kv = self.wkv(rows.reshape(batch, seqlen, -1))
        key, value = mx.split(kv, [self.hc_mult * self.dim], axis=-1)
        key = key.reshape(batch, seqlen, self.hc_mult, self.dim).astype(mx.float32)
        value = value.astype(mx.float32)
        h = x.astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        # 逐 (token, hc 副本) 的 RMS 归一化：query 与 key 各自单位化后再点积
        rstd = mx.rsqrt(h.square().mean(-1) + self.eps) * mx.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # 带符号平方根再进 sigmoid（对齐训练 kernel）；clamp 下限避免 |dot|→0
        # 时 sqrt 的梯度爆掉
        root = mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
        gate = mx.sigmoid(mx.where(dot < 0, -root, root))
        if token_mask is not None:
            gate = mx.where(token_mask[..., None], gate, 0)
        out = h + gate[..., None] * value[:, :, None, :]
        return out.astype(x.dtype)
