"""Viby 配置：DeepSeek-V4.1 架构的等比例缩小版。

架构对照（DeepSeek-V4.1-Flash，40 层 / dim 5120 / 384 路由专家）：

- **CED（Causal Encoder-Decoder）**：主干前一半是因果编码器，后一半是
  解码器；解码器的全局 KV 不由各解码层自己产生，而是从编码器最后一层的
  hidden 投影而来（`compress_ratios` 在 n_layers//2 处从 r 切到 1，
  且该层是 kv_source）。
- **CSA2（Compressed Sparse Attention 2）**：每个注意力层按
  `compress_ratios` 取三种静态模式之一——Full（自己产压缩 KV 且自己跑
  indexer）、Reindex（复用上游压缩 KV，自己跑 indexer）、Reuse（压缩 KV
  与 top-k 索引都复用上游）。压缩 KV 与 indexer 的 K 由
  `kv_source_layers` 指定层产生，被其后同 ratio 的层共享。
- **滑动窗口分支**：每层都有 window_size 的 SWA（环状 KV），与压缩分支
  拼接后一起做注意力。
- **Lightning Indexer + 分层稀疏索引**：indexer 给每个 query 选
  `index_topk` 个压缩位置；`candidate_source_layer` 先选
  `candidate_topk_blocks` 个候选块，更深的 indexer 只在候选池里打分。
- **mHC（Manifold-Constrained Hyper-Connections）**：残差流是 hc_mult 条
  并行副本，`comb` 经 Sinkhorn 迭代投影到双随机矩阵。
- **MoE**：sqrt(softplus(·)) 亲和度 + noaux_tc 的 e_score_correction_bias
  （只影响选择、不进梯度）；clamped SwiGLU 专家 + 1 个共享专家。
- **Engram**：n-gram 素数哈希条件记忆，按 token 查表、门控写入残差流。
- **MTP / DSpark**：主干后追加 draft 层，半自回归块式草稿 + 马尔可夫头 +
  置信度头。

与本仓库旧实现（Kimi Linear 3:1 KDA / iHC / AttnRes / SiTU）无关的
部分已全部移除；**Gated XSA 作为可选后处理保留**（见「注意力细节」）。
"""

import json
import os


# 默认缩放配方（≈1.0B 总参 / ≈110M 激活）：
#   dim 1024, 12 层（6 编码 + 6 解码）, 96 路由专家 × top-6。
_DEFAULT_LAYER_PRESET = dict(
    dim=1024,
    n_layers=12,
    n_heads=16,
    head_dim=128,
    rope_head_dim=32,
    q_lora_rank=512,
    o_groups=4,
    o_lora_rank=256,
    moe_inter_dim=256,
    n_routed_experts=96,
    n_activated_experts=6,
    n_shared_experts=1,
    window_size=128,
    hc_mult=4,
    index_n_heads=8,
    index_head_dim=64,
    index_topk=64,
    engram_n_heads=4,
    engram_head_dim=64,
)


def default_compress_ratios(n_layers: int, n_mtp_layers: int) -> list:
    """V4.1 的逐层压缩率：浅层纯滑窗、编码器段 r=2、解码器段 r=1、draft 层纯滑窗。

    对照 V4.1-Flash：`[0, 0] + [2]*18 + [1]*20 + [0]*3`。
    """
    ratios = [0] * (n_layers + n_mtp_layers)
    mid = max(2, n_layers // 2)  # CED 边界 = 第一个解码层
    for i in range(2, mid):
        ratios[i] = 2
    for i in range(mid, n_layers):
        ratios[i] = 1
    return ratios


def default_source_layers(n_layers: int):
    """(kv_source_layers, index_source_layers, candidate_source_layer)。

    对照 V4.1-Flash：kv=[2, 8, 14, 20]、index=[2, 8, 14, 20, 24, 28, 32, 36]、
    candidate=20。缩放规则：编码器段固定 2 个 kv 源（第 3 层 + CED 边界层），
    解码器段每 4 层一个 indexer 源。
    """
    mid = max(2, n_layers // 2)
    kv_sources = [2]
    if mid > 2:
        kv_sources.append(mid)
    else:
        kv_sources = [mid]
    index_sources = sorted(set(kv_sources) | set(range(mid, n_layers, 4)))
    return kv_sources, index_sources, mid


class VibyConfig:
    """DeepSeek-V4.1 缩放版配置。字段名尽量与官方 inference/config.json 对齐。"""

    model_type = "viby"
    arch = "deepseek_v4_1"

    def __init__(self, **kwargs):
        kw = dict(kwargs)
        # ---- 形状别名：CLI/trainer 沿用 HF 风格名字 ----
        if "hidden_size" in kw:
            kw.setdefault("dim", kw.pop("hidden_size"))
        if "num_hidden_layers" in kw:
            kw.setdefault("n_layers", kw.pop("num_hidden_layers"))
        if "num_attention_heads" in kw:
            kw.setdefault("n_heads", kw.pop("num_attention_heads"))
        if "moe_intermediate_size" in kw:
            kw.setdefault("moe_inter_dim", kw.pop("moe_intermediate_size"))
        if "num_experts_per_tok" in kw:
            kw.setdefault("n_activated_experts", kw.pop("num_experts_per_tok"))
        if "max_position_embeddings" in kw:
            kw.setdefault("max_seq_len", kw.pop("max_position_embeddings"))
        if "mtp_depth" in kw:
            kw.setdefault("n_mtp_layers", kw.pop("mtp_depth"))

        preset = kw.pop("preset", None)
        base = dict(_DEFAULT_LAYER_PRESET)
        if preset == "tiny":
            base.update(
                dim=256,
                n_layers=4,
                n_heads=4,
                head_dim=64,
                rope_head_dim=16,
                q_lora_rank=128,
                o_groups=2,
                o_lora_rank=64,
                moe_inter_dim=128,
                n_routed_experts=16,
                n_activated_experts=4,
                window_size=32,
                index_n_heads=4,
                index_head_dim=32,
                index_topk=16,
                engram_n_heads=2,
                engram_head_dim=32,
                # tiny 预设要显式压回小表，否则会继承 ≈1B 配方的大 Engram
                engram_vocab_size=8192,
            )
        for k, v in base.items():
            kw.setdefault(k, v)

        # ---- 主干 ----
        self.vocab_size = int(kw.get("vocab_size", 6400))
        self.dim = int(kw["dim"])
        self.n_layers = int(kw["n_layers"])
        self.n_heads = int(kw["n_heads"])
        self.head_dim = int(kw["head_dim"])
        self.rope_head_dim = int(kw["rope_head_dim"])
        self.q_lora_rank = int(kw["q_lora_rank"])
        self.o_groups = int(kw["o_groups"])
        self.o_lora_rank = int(kw["o_lora_rank"])
        self.norm_eps = float(kw.get("norm_eps", 1e-6))
        self.max_seq_len = int(kw.get("max_seq_len", 4096))
        self.tie_word_embeddings = bool(kw.get("tie_word_embeddings", False))
        self.bos_token_id = int(kw.get("bos_token_id", 1))
        self.eos_token_id = int(kw.get("eos_token_id", 2))
        self.pad_token_id = int(kw.get("pad_token_id", 0))
        # 报告 §4.2.2 的损失口径只有「无辅助损失负载均衡 + 权重 1e-4 的序列级均衡损失」，
        # 没有 z-loss：默认关掉（字段保留，作稳定性消融用）。
        self.z_loss_weight = float(kw.get("z_loss_weight", 0.0))
        self.aux_balance_loss_weight = float(kw.get("aux_balance_loss_weight", 1e-4))

        # ---- 稀疏注意力（CSA2）----
        self.window_size = int(kw["window_size"])
        n_mtp = int(kw.get("n_mtp_layers", 1))
        self.n_mtp_layers = n_mtp
        ratios = kw.get("compress_ratios")
        self.compress_ratios = tuple(
            int(x)
            for x in (
                ratios
                if ratios is not None
                else default_compress_ratios(self.n_layers, n_mtp)
            )
        )
        auto_kv, auto_idx, auto_cand = default_source_layers(self.n_layers)
        self.kv_source_layers = tuple(
            int(x) for x in kw.get("kv_source_layers", auto_kv)
        )
        self.index_source_layers = tuple(
            int(x) for x in kw.get("index_source_layers", auto_idx)
        )
        self.candidate_source_layer = int(kw.get("candidate_source_layer", auto_cand))
        self.candidate_topk_blocks = int(kw.get("candidate_topk_blocks", 64))
        self.candidate_block_size = int(kw.get("candidate_block_size", 8))

        # ---- Gated XSA（Exclusive Self-Attention 的可学习版）----
        # 逐 head 学 tanh(α) 扣掉 attention 输出中与自身 V 平行的分量
        #     z = y − tanh(α)·(yᵀv/‖v‖²)·v
        # α=0 初始化 ⇒ 起步恒等（层可自选剂量，故不强制限制层数也能让浅层趋于 0）。
        # 与 CSA2 正交：只对注意力输出做一次逐 head 后处理，不碰 KV cache、
        # 不碰 indexer/候选池，故 prefill/decode/prefix cache/连续 batch 口径一致。
        # 默认开；xsa_last_n=0 时按配方取最深 ≈1/3 层（参数高尔夫 XSA_LAST_N、
        # 自注意力偏置随层加深而增大，故收益集中在深层）。
        self.use_xsa = bool(kw.get("use_xsa", True))
        _xln = int(kw.get("xsa_last_n", 0) or 0)
        if _xln <= 0 and self.use_xsa:
            _xln = max(1, self.n_layers // 3)
        self.xsa_last_n = max(0, _xln)

        # ---- RoPE（滑窗分支用 rope_theta；压缩分支用 compress_rope_theta + YaRN）----
        self.rope_theta = float(kw.get("rope_theta", 10000.0))
        self.compress_rope_theta = float(kw.get("compress_rope_theta", 160000.0))
        self.original_seq_len = int(kw.get("original_seq_len", 65536))
        self.rope_factor = float(kw.get("rope_factor", 16.0))
        self.beta_fast = int(kw.get("beta_fast", 32))
        self.beta_slow = int(kw.get("beta_slow", 1))

        # ---- Lightning Indexer ----
        self.index_n_heads = int(kw["index_n_heads"])
        self.index_head_dim = int(kw["index_head_dim"])
        self.index_topk = int(kw["index_topk"])

        # ---- mHC ----
        self.hc_mult = int(kw["hc_mult"])
        self.hc_sinkhorn_iters = int(kw.get("hc_sinkhorn_iters", 20))
        self.hc_eps = float(kw.get("hc_eps", 1e-6))

        # ---- Engram ----
        _eids = kw.get("engram_layer_ids")
        if _eids is None:
            # 对照 V4.1-Flash 的 [1, 14]（40 层）：第 2 层 + 约 35% 深度处，
            # 都落在编码器段内；小模型上去重且避开第 0 层。
            _eids = []
            for cand in (1, max(2, int(self.n_layers * 0.35))):
                if cand not in _eids and 0 < cand < self.n_layers:
                    _eids.append(cand)
        self.engram_layer_ids = tuple(int(x) for x in _eids)
        self.engram_max_ngram_size = int(kw.get("engram_max_ngram_size", 4))
        # 163840 = 20 × 8192：官方 V4.1-Flash 的 Engram 检索表占**总参 26%**
        # （196B / (552B+196B)），而旧的 8192 在 ≈1B 配方下只有 1.28%（12.7M），
        # 比例差了 20 倍——Engram 是"条件记忆"，靠极稀疏的大容量查表存知识，
        # 表太小等于把它降级成一个普通特征模块。表参数量 ≈ 1536 × V
        # （2 层 × 3 阶 × 4 头 × head_dim 64），V=163840 → ≈252M，占比 ~20%。
        self.engram_vocab_size = int(kw.get("engram_vocab_size", 163840))
        self.engram_n_heads = int(kw["engram_n_heads"])
        self.engram_head_dim = int(kw["engram_head_dim"])
        self.engram_pad_id = int(kw.get("engram_pad_id", self.pad_token_id))
        self.engram_compressed_vocab_size = int(
            kw.get("engram_compressed_vocab_size", 0)
        )
        _ne = kw.get("engram_num_embeddings")
        if _ne is None and self.engram_layer_ids:
            from .engram import compute_num_embeddings

            _ne = compute_num_embeddings(self)
        self.engram_num_embeddings = tuple(int(x) for x in (_ne or ()))

        # ---- MoE ----
        self.n_routed_experts = int(kw["n_routed_experts"])
        self.n_activated_experts = int(kw["n_activated_experts"])
        self.n_shared_experts = int(kw.get("n_shared_experts", 1))
        self.moe_inter_dim = int(kw["moe_inter_dim"])
        self.score_func = str(kw.get("score_func", "sqrtsoftplus"))
        self.gate_temp = float(kw.get("gate_temp", 1.0))
        self.norm_topk_prob = bool(kw.get("norm_topk_prob", True))
        self.route_scale = float(kw.get("route_scale", 1.5))
        self.swiglu_limit = float(kw.get("swiglu_limit", 10.0))
        self.bias_update_rate = float(kw.get("bias_update_rate", 1e-3))
        self.moe_balance_method = str(kw.get("moe_balance_method", "qb"))
        self.qb_update_rate = float(kw.get("qb_update_rate", 0.5))
        self.qb_stats_rows = int(kw.get("qb_stats_rows", 8192))
        self.router_fp32 = bool(kw.get("router_fp32", True))

        # ---- MTP / DSpark ----
        self.dspark_block_size = int(kw.get("dspark_block_size", 4))
        self.dspark_noise_token_id = int(
            kw.get("dspark_noise_token_id", self.pad_token_id)
        )
        _tgt = kw.get("dspark_target_layer_ids")
        if _tgt is None:
            _tgt = (
                tuple(range(max(0, self.n_layers - 3), self.n_layers))
                if self.n_layers >= 3
                else (self.n_layers - 1,)
            )
        self.dspark_target_layer_ids = tuple(int(x) for x in _tgt)
        self.dspark_markov_rank = int(kw.get("dspark_markov_rank", 64))
        self.dspark_n_routed_experts = int(kw.get("dspark_n_routed_experts", 32))
        self.dspark_n_activated_experts = int(kw.get("dspark_n_activated_experts", 3))
        self.mtp_loss_weight = float(kw.get("mtp_loss_weight", 0.3))

        # Residual lifting reuses the existing middle decoder weights. The
        # execution identity must therefore be recorded independently of keys.
        self.ced_recurrent_enabled = bool(kw.get("ced_recurrent_enabled", False))
        self.ced_recurrent_stride = int(kw.get("ced_recurrent_stride", 4))
        self.ced_recurrent_rounds = int(kw.get("ced_recurrent_rounds", 3))
        self.ced_recurrent_arch = str(kw.get("ced_recurrent_arch", "residual_lift_v1"))

        self._validate()

    # ------------------------------------------------------------------
    @property
    def n_encoder_layers(self) -> int:
        """CED 边界（= 第一个解码层的下标）。"""
        return max(2, self.n_layers // 2)

    @property
    def n_exec_layers(self) -> int:
        return self.n_layers

    def layer_mode(self, layer_idx: int) -> str:
        """CSA2 静态模式：full / reindex / reuse / sliding。"""
        if self.compress_ratios[layer_idx] == 0:
            return "sliding"
        is_kv = layer_idx in self.kv_source_layers
        is_idx = layer_idx in self.index_source_layers
        if is_kv and is_idx:
            return "full"
        if is_idx:
            return "reindex"
        return "reuse"

    def kv_source_of(self, layer_idx: int) -> int:
        """该层读取的压缩 KV 来自哪一层（自身即源时返回自身）。"""
        src = None
        for s in self.kv_source_layers:
            if s <= layer_idx:
                src = s
        return layer_idx if src is None else src

    def index_source_of(self, layer_idx: int) -> int:
        src = None
        for s in self.index_source_layers:
            if s <= layer_idx:
                src = s
        return layer_idx if src is None else src

    def moe_of(self, layer_idx: int):
        """(路由专家数, 激活专家数)：draft 层用 DSpark 的窄 MoE。"""
        if layer_idx < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )

    def _validate(self):
        if self.ced_recurrent_stride < 1 or self.ced_recurrent_rounds < 1:
            raise ValueError("CED recurrent stride and rounds must be positive")
        if self.ced_recurrent_enabled:
            if self.ced_recurrent_arch != "residual_lift_v1":
                raise ValueError("Unsupported recurrent CED execution version")
            if self.n_mtp_layers != 0:
                raise ValueError(
                    "Recurrent CED requires --mtp_depth 0; disable MTP on both comparison sides"
                )
            boundary = self.n_encoder_layers
            middle = range(boundary + 1, self.n_layers - 1)
            if not middle:
                raise ValueError(
                    "Recurrent CED needs at least one middle decoder block"
                )
            if (
                len(self.compress_ratios) < self.n_layers
                or self.compress_ratios[boundary] != 1
                or boundary not in self.kv_source_layers
                or boundary not in self.index_source_layers
            ):
                raise ValueError("Recurrent CED requires a full ratio=1 CED boundary")
            for i in range(boundary + 1, self.n_layers):
                if self.compress_ratios[i] != 1 or i in self.kv_source_layers:
                    raise ValueError(
                        "Recurrent CED middle/output blocks must reuse token-level boundary KV (ratio=1, no new KV source)"
                    )
            if any(i in self.engram_layer_ids for i in middle):
                raise ValueError(
                    "Recurrent CED middle blocks cannot contain token Engram modules"
                )
            if self.n_layers - 1 in self.index_source_layers:
                raise ValueError(
                    "Recurrent CED output layer must reuse final anchor selection"
                )
            if self.candidate_source_layer not in (-1, boundary):
                raise ValueError(
                    "Recurrent CED candidate pool must originate at the full boundary"
                )
        if self.n_layers < 4:
            raise ValueError("n_layers 必须 >= 4（CED 至少 2 个编码层 + 2 个解码层）")
        # V4.1 的 head_dim 与 dim 解耦（官方 64 头 × 512 维，远超 dim/64）：
        # q 经 q_lora_rank 低秩再升到 n_heads*head_dim，不做 dim == n_heads*head_dim。
        if self.n_heads % self.o_groups != 0:
            raise ValueError("n_heads 必须被 o_groups 整除")
        if (
            self.rope_head_dim <= 0
            or self.rope_head_dim % 2
            or self.rope_head_dim > self.head_dim
        ):
            raise ValueError("rope_head_dim 必须是 (0, head_dim] 内的偶数")
        if self.head_dim % (self.n_heads // self.o_groups) and False:
            pass
        if self.hc_mult < 2:
            raise ValueError("hc_mult 必须 >= 2")
        if self.hc_sinkhorn_iters < 1:
            raise ValueError("hc_sinkhorn_iters 必须 >= 1")
        if self.window_size < 1:
            raise ValueError("window_size 必须 >= 1")
        if len(self.compress_ratios) != self.n_layers + self.n_mtp_layers:
            raise ValueError(
                f"compress_ratios 长度 {len(self.compress_ratios)} != n_layers + n_mtp_layers"
                f" = {self.n_layers + self.n_mtp_layers}"
            )
        if any(r < 0 for r in self.compress_ratios):
            raise ValueError("compress_ratios 不能为负")
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts 必须 > 0（V4.1 主干全层 MoE）")
        if not 0 < self.n_activated_experts <= self.n_routed_experts:
            raise ValueError("n_activated_experts 必须在 (0, n_routed_experts]")
        if self.n_shared_experts != 1:
            raise ValueError("V4.1 只有 1 个共享专家")
        if self.score_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
            raise ValueError("score_func 必须是 sqrtsoftplus / softmax / sigmoid")
        if self.moe_inter_dim <= 0:
            raise ValueError("moe_inter_dim 必须 > 0")
        if self.moe_balance_method not in ("qb", "noaux_tc"):
            raise ValueError("moe_balance_method 必须是 qb / noaux_tc")
        if not 0 <= self.qb_update_rate <= 1:
            raise ValueError("qb_update_rate 必须在 [0, 1]")
        if self.qb_stats_rows < 1:
            raise ValueError("qb_stats_rows 必须 >= 1")
        if not self.gate_temp > 0 or not self.bias_update_rate >= 0:
            raise ValueError("gate_temp 必须 > 0，bias_update_rate 必须 >= 0")
        if self.index_topk <= 0 or self.index_n_heads <= 0 or self.index_head_dim <= 0:
            raise ValueError("indexer 的 heads / head_dim / topk 必须 > 0")
        if self.candidate_source_layer >= 0 and self.candidate_block_size <= 0:
            raise ValueError("candidate_block_size 必须 > 0")
        # 压缩层必须有同 ratio 的上游源；源必须有序且第 0 层不做源。
        for lst, name in (
            (self.kv_source_layers, "kv_source_layers"),
            (self.index_source_layers, "index_source_layers"),
        ):
            if list(lst) != sorted(set(lst)):
                raise ValueError(f"{name} 必须严格递增且不重复")
            if lst and lst[0] <= 0:
                raise ValueError(f"{name} 不能包含第 0 层")
            if lst and lst[-1] >= self.n_layers:
                raise ValueError(f"{name} 的下标必须落在主干层内（< n_layers）")
        for i in range(self.n_layers):
            r = self.compress_ratios[i]
            if r == 0:
                continue
            s = self.kv_source_of(i)
            if s not in self.kv_source_layers:
                raise ValueError(f"第 {i} 层压缩率 {r} 但没有上游 kv 源")
            if self.compress_ratios[s] != r:
                raise ValueError(
                    f"第 {i} 层压缩率 {r} 与其 kv 源第 {s} 层 {self.compress_ratios[s]} 不一致"
                )
        for e in self.engram_layer_ids:
            if not 0 <= e < self.n_layers:
                raise ValueError("engram_layer_ids 必须落在主干层内")
            if self.compress_ratios[e] == 0 and e == 0:
                raise ValueError("engram 不能挂在第 0 层")
        if self.engram_layer_ids and self.engram_max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size 必须 >= 2")
        if self.n_mtp_layers > 0 and self.dspark_block_size < 1:
            raise ValueError("dspark_block_size 必须 >= 1")
        if self.n_mtp_layers > 0 and not self.dspark_target_layer_ids:
            raise ValueError("dspark_target_layer_ids 不能为空")
        for t in self.dspark_target_layer_ids:
            if not 0 <= t < self.n_layers:
                raise ValueError("dspark_target_layer_ids 必须落在主干层内")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["model_type"] = self.model_type
        d["arch"] = self.arch
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "VibyConfig":
        data = dict(data)
        data.pop("model_type", None)
        data.pop("arch", None)
        return cls(**data)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyConfig":
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    def num_parameters(self, include_engram: bool = True) -> int:
        """静态参数计数（只算权重矩阵，不含 norm/bias 等 1D 量）。"""
        d = self.dim
        hd = self.head_dim
        total = 0
        total += (
            2 * self.vocab_size * d
            if not self.tie_word_embeddings
            else self.vocab_size * d
        )
        for i in range(self.n_layers):
            r = self.compress_ratios[i]
            mode = self.layer_mode(i)
            # attention：wq_a / wq_b / wkv / wo_a / wo_b
            total += d * self.q_lora_rank + self.q_lora_rank * self.n_heads * hd
            total += d * hd
            total += (self.n_heads * hd // self.o_groups) * (
                self.o_groups * self.o_lora_rank
            )
            total += self.o_lora_rank * self.o_groups * d
            if mode == "full":  # 压缩器（+ ratio>1 时的门）
                total += d * hd * (2 if r > 1 else 1)
                total += hd * self.index_head_dim + d * self.index_n_heads
                total += self.q_lora_rank * self.index_n_heads * self.index_head_dim
            elif mode == "reindex":
                total += d * self.index_n_heads
                total += self.q_lora_rank * self.index_n_heads * self.index_head_dim
            # mHC：attn / ffn 各一个 HyperConnection，混合矩阵 (2+hc)·hc × hc·d
            total += 2 * (2 + self.hc_mult) * self.hc_mult * self.hc_mult * d
            # MoE
            n_exp, _ = self.moe_of(i)
            total += n_exp * 3 * d * self.moe_inter_dim
            total += 3 * d * self.moe_inter_dim  # 共享专家
            # Engram
            if include_engram and i in self.engram_layer_ids:
                cols = (self.engram_max_ngram_size - 1) * self.engram_n_heads
                total += cols * self.engram_head_dim * d * (self.hc_mult + 1)
                total += 2 * self.hc_mult * d  # q_weight / k_weight 两张 [hc, d]
                idx = self.engram_layer_ids.index(i)
                total += self.engram_num_embeddings[idx] * self.engram_head_dim
        if self.n_mtp_layers > 0:
            n_exp = self.dspark_n_routed_experts
            mtp = self.n_mtp_layers * (
                3 * d * self.moe_inter_dim * (n_exp + 1)
                + d * self.q_lora_rank
                + self.q_lora_rank * self.n_heads * hd
                + d * hd
                + (self.n_heads * hd // self.o_groups)
                * (self.o_groups * self.o_lora_rank)
                + self.o_lora_rank * self.o_groups * d
                + 2
                * (2 + self.hc_mult)
                * self.hc_mult
                * self.hc_mult
                * d  # draft block 的 mHC
            )
            mtp += d * len(self.dspark_target_layer_ids) * d  # main_proj（仅第一层）
            mtp += 2 * self.vocab_size * self.dspark_markov_rank  # 马尔可夫头
            total += mtp
        return total

    def num_active_parameters(self) -> int:
        """每 token 激活的主干/桥接参数。"""
        d = self.dim
        hd = self.head_dim
        per_layer = (
            self.n_activated_experts * 3 * d * self.moe_inter_dim
            + 3 * d * self.moe_inter_dim
        )
        per_layer += (
            d * self.q_lora_rank + self.q_lora_rank * self.n_heads * hd + d * hd
        )
        per_layer += (self.n_heads * hd // self.o_groups) * (
            self.o_groups * self.o_lora_rank
        )
        per_layer += self.o_lora_rank * self.o_groups * d
        total = self.n_layers * per_layer + 2 * self.vocab_size * d
        if self.n_mtp_layers > 0:
            total += self.n_mtp_layers * (
                self.dspark_n_activated_experts * 3 * d * self.moe_inter_dim + per_layer
            )
        return total
