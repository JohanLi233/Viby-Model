"""Viby：DeepSeek-V4.1 架构的等比例缩小版（Apple MLX）。

前向数据流（对照 V4.1 tech report 与官方 inference/model.py）：

    embed → 复制成 hc_mult 条残差流 → [Engram 注入] → N 个 V4.1 Block
          → hc_pre 收敛 → RMSNorm → lm_head

主干是 CED（Causal Encoder-Decoder）：前 n_layers//2 层是因果编码器，
后一半是解码器；解码段的全局压缩 KV 由编码器末层（CED 边界层）产生并被
其后所有解码层共享——这就是 config.compress_ratios 在边界处从 r 切到 1、
且边界层是 kv_source 的原因。

MTP / DSpark：主干后挂 draft 层，取主干若干目标层的"注意力输入"（mHC
流均值）拼起来，经 main_proj + main_norm 变成锚点表示；锚点 t 的草稿序列
是 [x_t, 噪声, ..., 噪声]（槽 0 是已给 token，其余交给模型自回归），
马尔可夫头按前一个 token 的嵌入给草稿 logits 加低秩偏置，置信度头预测
该草稿 token 会不会被验证接受。训练用 teacher forcing：槽 j 看 token t+j，
预测 t+j+1（与推理时"槽 j 的输入是前面已确定的 token"一致）。
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .block import Block, apply_hc_pre_norm
from .cache import SharedAttnState, VibyCache
from .config import VibyConfig
from .engram import Engram, EngramLayout, NgramHashState, build_compressed_token_map
from .hc import identity_pre_mix
from .init import apply_trunc_normal_init
from .moe import MoEFeedForward, MoEGate, update_expert_bias, update_quantile_bias
from .norms import RMSNorm
from .psr import (
    EvidenceMemory,
    ProtectedPSR,
    ReasoningTrace,
    ThinkingState,
    residual_statistics,
)
from .recurrent import recurrent_active, execution_signature, run_middle

NEG_INF = -1e30


@dataclass
class CausalLMOutput:
    loss: Optional[mx.array] = None
    lm_loss: Optional[mx.array] = None
    mtp_loss: Optional[mx.array] = None
    z_loss: Optional[mx.array] = None
    diversity_loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    moe_loads: Optional[mx.array] = None
    moe_qb_margins: Optional[mx.array] = None
    aux_loss: Optional[mx.array] = None
    hidden_states: Optional[mx.array] = None
    thinking_state: Optional[ThinkingState] = None
    thinking_trace: Optional[ReasoningTrace] = None
    corrected_loss: Optional[mx.array] = None
    corrected_lm_loss: Optional[mx.array] = None
    metrics: Optional[mx.array] = None
    base_logits: Optional[mx.array] = None
    base_token_nll: Optional[mx.array] = None
    corrected_token_nll: Optional[mx.array] = None
    bridge_mask: Optional[mx.array] = None
    ced_metrics: Optional[mx.array] = None
    ced_result: object = None
    sequence_losses: Optional[mx.array] = None
    sequence_token_counts: Optional[mx.array] = None
    tail_stats: Optional[mx.array] = None
    dpr_metrics: Optional[mx.array] = None
    dpr_loss: Optional[mx.array] = None


def lm_head_ce(
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    loss_mask=None,
    z_weight: float = 0.0,
    chunk: int = 256,
):
    """分块 lm_head + CE（+ 可选 z-loss），不物化全量 (B,T,V) logits。

    返回 (ce_mean, z_mean)；z_weight=0 时不算 z（省一次 logsumexp）。
    """
    B, T, D = hidden.shape
    n = B * T
    flat_h = hidden.reshape(n, D)
    flat_y = labels.reshape(n)
    flat_m = None if loss_mask is None else loss_mask.reshape(n).astype(mx.float32)
    ce_sum = mx.array(0.0, dtype=mx.float32)
    z_sum = mx.array(0.0, dtype=mx.float32)
    w_sum = mx.array(0.0, dtype=mx.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        logits = flat_h[s:e] @ weight.T
        ce = mx.fast.cross_entropy(logits, flat_y[s:e])
        if flat_m is None:
            ce_sum = ce_sum + mx.sum(ce)
            w_sum = w_sum + (e - s)
            if z_weight:
                lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
                z_sum = z_sum + mx.sum(lse * lse)
        else:
            m = flat_m[s:e]
            ce_sum = ce_sum + mx.sum(ce * m)
            w_sum = w_sum + mx.sum(m)
            if z_weight:
                lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
                z_sum = z_sum + mx.sum(lse * lse * m)
    ce_mean = ce_sum / mx.maximum(w_sum, 1.0)
    z_mean = z_sum / mx.maximum(w_sum, 1.0) if z_weight else mx.array(0.0)
    return ce_mean, z_mean


def sample_tokens(
    logits: mx.array, temperature: float = 1.0, top_k: int = 0
) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32) / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        thr = mx.partition(logits, kth=logits.shape[-1] - k, axis=-1)[
            ..., logits.shape[-1] - k
        ]
        logits = mx.where(logits >= thr[..., None], logits, NEG_INF)
    return mx.random.categorical(logits)


def _load_tokenizer(path: Optional[str]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or "model")


class VibyModel(nn.Module):
    """主干：embedding + N 个 V4.1 Block + 末层 norm（+ Engram）。"""

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [Block(config, i) for i in range(config.n_layers)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.dpr = None
        self.engram_layout = (
            EngramLayout.from_config(config) if config.engram_layer_ids else None
        )
        # 用 list 而不是 dict 存 Engram：参数树的整型键会被 tree_unflatten 还原成
        # list，容器类型不一致会让 nn.Module.update 走错分支。
        self.engram_layers = []
        if self.engram_layout is not None:
            tokenizer = _load_tokenizer(getattr(config, "tokenizer_path", None))
            token_map, compressed_vocab = build_compressed_token_map(tokenizer)
            if config.engram_compressed_vocab_size in (0, None):
                config.engram_compressed_vocab_size = int(compressed_vocab)
            self.engram_hash = NgramHashState(
                config, self.engram_layout, token_map, compressed_vocab
            )
            self._engram_slot = {}
            for i, lid in enumerate(config.engram_layer_ids):
                self.engram_layers.append(Engram(config, lid, self.engram_layout))
                self._engram_slot[lid] = i
        else:
            self.engram_hash = None
            self._engram_slot = {}

    def __call__(
        self,
        input_ids: mx.array,
        start_pos: int = 0,
        cache: Optional[VibyCache] = None,
        segment_ids=None,
        pad_mask=None,
        decode: bool = False,
        prev_tokens=None,
        collect_main: bool = True,
        return_memory=False,
        use_ced_recurrent=True,
        use_dpr=True,
        return_dpr=False,
        dpr_intervention=None,
    ):
        dpr_active = self.dpr is not None and use_dpr
        if cache is not None:
            signature = (dpr_active, dpr_intervention)
            if getattr(cache, "dpr_signature", signature) != signature:
                raise ValueError("DPR condition changes require a fresh cache")
            cache.dpr_signature = signature
        recurrent = recurrent_active(self.config, use_ced_recurrent)
        if cache is not None:
            signature = execution_signature(self.config, use_ced_recurrent)
            if cache.ced_signature not in (None, signature):
                raise ValueError("CED execution changes require a fresh cache")
            cache.ced_signature = signature
        if recurrent:
            if return_memory:
                raise ValueError("recurrent CED runs independently of protected PSR")
            if isinstance(start_pos, mx.array):
                raise ValueError(
                    "recurrent CED cache requires a common integer token clock"
                )
            if cache is None and (start_pos != 0 or decode):
                raise ValueError(
                    "uncached recurrent CED requires a full prefix at start_pos=0"
                )
            if cache is not None and (segment_ids is not None or pad_mask is not None):
                raise ValueError(
                    "cached recurrent CED requires unpadded single-document inputs; use uncached packed evaluation"
                )
            if cache is not None and start_pos > 0 and input_ids.shape[1] != 1:
                raise ValueError(
                    "recurrent CED cache supports prefill then single-token decode"
                )
        h = self.embed(input_ids)
        raw_embeddings = (
            h if return_dpr and self.config.dpr_variant == "legacy_v1" else None
        )
        dpr_result = None
        B, T = input_ids.shape
        positions = (
            (start_pos + mx.arange(T))[None, :]
            if not isinstance(start_pos, mx.array)
            else start_pos[:, None] + mx.arange(T)
        )
        positions = mx.broadcast_to(positions, (B, T))
        memory = None
        boundary_input = None
        hashes, new_prev = (None, None)
        if self.engram_hash is not None:
            token_mask = None if pad_mask is None else pad_mask
            hashes, new_prev = self.engram_hash(input_ids, prev_tokens, token_mask)
        h = mx.repeat(h[:, :, None, :], self.hc_mult, axis=2)
        shared = SharedAttnState()
        if decode:
            hint = getattr(cache, "decode_max_pos", None) if cache is not None else None
            if hint:
                shared.pool_tokens = int(hint)
            elif not isinstance(start_pos, mx.array):
                shared.pool_tokens = int(start_pos) + 1
        pre_mix = identity_pre_mix(h, self.hc_mult)
        mains = []
        ced_result = None
        for i, layer in enumerate(self.layers):
            if (
                recurrent
                and self.config.n_encoder_layers < i < self.config.n_layers - 1
            ):
                if i == self.config.n_encoder_layers + 1:
                    (
                        h,
                        pre_mix,
                        shared,
                        memory_positions,
                        memory_docs,
                        memory_pad,
                        ced_result,
                    ) = run_middle(
                        self,
                        h,
                        pre_mix,
                        shared,
                        positions,
                        segment_ids,
                        pad_mask,
                        cache,
                        start_pos,
                    )
                continue
            if hashes is not None and i in self._engram_slot:
                h = self.engram_layers[self._engram_slot[i]](
                    h, hashes[:, :, self._engram_slot[i], :]
                )
            if collect_main and i in self.target_layer_ids:
                # MTP 读的是目标层的"注意力输入"（mHC 均值），不是层输出
                mains.append(mx.mean(h, axis=2))
            layer_cache = None if cache is None else cache[i]
            if return_memory and i == self.config.n_encoder_layers:
                boundary_input = apply_hc_pre_norm(h, pre_mix, layer.attn_norm)
            if recurrent and i == self.config.n_layers - 1:
                h, pre_mix = layer.recurrent(
                    h,
                    pre_mix,
                    shared,
                    query_positions=positions,
                    memory_positions=memory_positions,
                    query_segment_ids=segment_ids,
                    memory_segment_ids=memory_docs,
                    query_pad_mask=pad_mask,
                    memory_pad_mask=memory_pad,
                    cache=None if cache is None else cache.ced_output_cache,
                )
            else:
                h, pre_mix = layer(
                    h,
                    start_pos,
                    pre_mix,
                    shared,
                    layer_cache,
                    segment_ids,
                    pad_mask,
                    decode,
                )
            if dpr_active and i == self.config.n_encoder_layers:
                a = apply_hc_pre_norm(h, pre_mix, self.layers[i + 1].attn_norm)
                delta, pi, z = self.dpr(a, dpr_intervention)
                h = h + delta.astype(h.dtype)[:, :, None, :]
                if return_dpr:
                    # Contextual targets come from this same encoder pass,
                    # before injection. They never enter current logits.
                    target_source = (
                        a
                        if self.config.dpr_variant == "contextual_v2"
                        else raw_embeddings
                    )
                    dpr_result = (pi, z, target_source)
            if return_memory and i == self.config.n_encoder_layers:
                # Batched decode has one position per row. Slice to the largest
                # live prefix; PSR masks each row's memory at its own anchor.
                if cache is None:
                    values = shared.compress_kv
                else:
                    memory_end = start_pos + T
                    if isinstance(start_pos, mx.array):
                        memory_end = (
                            cache.decode_max_pos or int(mx.max(start_pos).item()) + T
                        )
                    values = cache[i].compress_kv[:, :memory_end]
                memory = EvidenceMemory(
                    values, boundary_input, positions, pad_mask, segment_ids
                )
        result = (apply_hc_pre_norm(h, pre_mix, self.norm), mains, new_prev)
        if return_memory:
            result = (*result, memory)
        if recurrent:
            result = (*result, ced_result)
        if return_dpr:
            result = (*result, dpr_result)
        return result


class DSparkMarkovHead(nn.Module):
    """马尔可夫头：用前一个 token 的嵌入给草稿 logits 加低秩偏置。"""

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.head = nn.Linear(config.dspark_markov_rank, config.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array):
        e = self.embed(token_ids)
        return self.head(e), e


class DSparkConfidenceHead(nn.Module):
    """置信度头：预测该草稿 token 是否会被验证接受（推理时按置信度调度验证）。"""

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.proj = nn.Linear(config.dim + config.dspark_markov_rank, 1, bias=False)

    def __call__(self, hidden: mx.array, markov_embed: mx.array):
        return self.proj(mx.concatenate([hidden, markov_embed], axis=-1))[..., 0]


class DSparkStage(nn.Module):
    """一个 DSpark 草稿阶段：V4.1 Block +（首/末阶段）锚点投影与预测头。"""

    def __init__(self, config: VibyConfig, stage_id: int):
        super().__init__()
        self.stage_id = stage_id
        self.block_size = config.dspark_block_size
        self.layer = Block(config, config.n_layers + stage_id)
        if stage_id == 0:
            self.main_proj = nn.Linear(
                config.dim * len(config.dspark_target_layer_ids), config.dim, bias=False
            )
            self.main_norm = RMSNorm(config.dim, config.norm_eps)
        if stage_id == config.n_mtp_layers - 1:
            self.norm = RMSNorm(config.dim, config.norm_eps)
            self.markov_head = DSparkMarkovHead(config)
            self.confidence_head = DSparkConfidenceHead(config)


class VibyForCausalLM(nn.Module):
    def __init__(self, config: Optional[VibyConfig] = None, skip_init: bool = False):
        super().__init__()
        config = config or VibyConfig()
        self.config = config
        self.model = VibyModel(config)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.dim, config.vocab_size, bias=False)
        )
        self.mtp_modules = [DSparkStage(config, i) for i in range(config.n_mtp_layers)]
        self._moe_gates = [m for m in self.modules() if isinstance(m, MoEGate)]
        # 主干 gate 的 E 相同，可以堆成一张 [L, E] 计数表；draft 层 E 不同，单独处理
        self._backbone_gates = [
            g for g in self._moe_gates if g.layer_idx < config.n_layers
        ]
        self._moe_layers = [m for m in self.modules() if isinstance(m, MoEFeedForward)]
        # 融合 kernel 预热（必须在 mx.compile trace 之前、eager 上下文里做）
        try:
            from .kernels.hc_fused import prewarm_hc_post

            prewarm_hc_post(config.hc_mult)
        except Exception:  # noqa: BLE001
            pass
        if config.hc_mult == 4:
            try:
                from .kernels.hc_pre_norm import prewarm_hc_pre_norm

                prewarm_hc_pre_norm(config.dim, config.norm_eps, mx.bfloat16)
                prewarm_hc_pre_norm(config.dim, config.norm_eps, mx.float16)
            except Exception:  # noqa: BLE001
                pass
        if config.hc_mult == 4:
            from .kernels.sinkhorn_fused import prewarm_sinkhorn

            prewarm_sinkhorn(config.hc_sinkhorn_iters, config.hc_eps)
        from .kernels.moe_dispatch import prewarm_moe, prewarm_route_combine
        from .kernels.sparse_attention import prewarm_sparse_attention, prewarm_topk
        from .kernels.indexer_score import prewarm_indexer_score

        if config.n_heads == 16 and any(config.compress_ratios):
            prewarm_topk(config.max_seq_len)
        if any(config.compress_ratios):
            prewarm_indexer_score(
                config.index_n_heads, config.index_head_dim, mx.bfloat16
            )
            prewarm_indexer_score(
                config.index_n_heads, config.index_head_dim, mx.float16
            )

        for dtype in (mx.bfloat16, mx.float16):
            for dim, width, top_k in {
                (m.dim, 2 * m.moe_in, m.top_k) for m in self._moe_layers
            }:
                prewarm_moe(dim, width, top_k, dtype)
                prewarm_route_combine(dim, top_k, dtype)
            if config.n_heads == 16 and any(config.compress_ratios):
                prewarm_sparse_attention(
                    config.head_dim, config.window_size, config.head_dim**-0.5, dtype
                )
                if recurrent_active(config):
                    prewarm_sparse_attention(
                        config.head_dim,
                        (config.window_size + config.ced_recurrent_stride - 1)
                        // config.ced_recurrent_stride,
                        config.head_dim**-0.5,
                        dtype,
                    )
        if not skip_init:
            apply_trunc_normal_init(self, config.dim)
        self.psr = None
        if config.psr_enabled:
            rng = list(mx.random.state)
            self.psr = ProtectedPSR(config)
            if not skip_init:
                apply_trunc_normal_init(self.psr, config.psr_dim)
            self.psr.output.weight = mx.zeros_like(self.psr.output.weight)
            # MLX 0.32 uses a read-only per-thread RNG sentinel. key(seed)
            # packs uint64 as two uint32 words; restore via the public seed API.
            high, low = rng[0].tolist()
            mx.random.seed((int(high) << 32) | int(low))

        if config.dpr_enabled:
            from .dpr import DistributionalPredictiveResidual

            rng = list(mx.random.state)
            mx.random.seed(config.dpr_seed)
            self.model.dpr = DistributionalPredictiveResidual(config)
            if not skip_init:
                apply_trunc_normal_init(self.model.dpr, config.dim)
            self.model.dpr.output.weight = mx.zeros_like(self.model.dpr.output.weight)
            high, low = rng[0].tolist()
            mx.random.seed((int(high) << 32) | int(low))

    # ------------------------------------------------------------------
    def _head_weight(self) -> mx.array:
        return (
            self.lm_head.weight if self.lm_head is not None else self.model.embed.weight
        )

    def logits(self, hidden: mx.array) -> mx.array:
        return hidden @ self._head_weight().T

    def __call__(
        self,
        input_ids: mx.array,
        labels: Optional[mx.array] = None,
        loss_mask=None,
        attention_mask=None,
        segment_ids=None,
        need_logits=False,
        start_pos=0,
        cache=None,
        decode=False,
        prev_tokens=None,
        use_mtp=True,
        thinking_prefix_lengths=None,
        thinking_state=None,
        thinking_options=None,
        thinking_targets=None,
        use_thinking=True,
        return_thinking=False,
        psr_mode=None,
        psr_anchors=None,
        psr_gate=None,
        psr_sample_weights=None,
        return_metrics=False,
        use_ced_recurrent=True,
        return_sequence_losses=False,
        tail_reference=None,
        tail_fraction=None,
        use_dpr=True,
        dpr_weight=None,
        dpr_intervention=None,
        **kwargs,
    ):
        mode = psr_mode or (
            "recurrent" if self.config.psr_enabled and use_thinking else "off"
        )
        if not use_thinking:
            mode = "off"
        if mode not in ("off", "state_only", "recurrent"):
            raise ValueError("psr_mode must be off/state_only/recurrent")
        if tail_reference is not None and (mode != "off" or labels is None):
            raise ValueError("TailSFT requires labels and psr_mode=off")
        if cache is not None:
            previous = getattr(cache, "psr_mode", None)
            if previous is not None and previous != mode:
                raise ValueError("PSR condition changes require a fresh cache")
            cache.psr_mode = mode
        if mode == "off":
            thinking_state = None
            thinking_options = None
            thinking_targets = None
            psr_anchors = None
            thinking_prefix_lengths = None
        elif self.psr is None:
            raise ValueError("construct with psr_enabled=True")
        if thinking_targets is not None:
            raise ValueError(
                "protected PSR uses only direct next-token labels; legacy auxiliary targets were removed"
            )
        if thinking_state is None and cache is not None and mode != "off":
            thinking_state = cache.thinking_state
        if psr_anchors is None and thinking_prefix_lengths is not None:
            lengths = thinking_prefix_lengths
            if isinstance(lengths, int):
                lengths = mx.full((input_ids.shape[0],), lengths, mx.int32)
            psr_anchors = lengths[:, None] - 1
        if mode != "off" and thinking_state is None and psr_anchors is None:
            raise ValueError(
                "supply a PSR anchor plan or use prefill/generate; off bypasses it"
            )
        requested = mode != "off" and thinking_state is None
        pad = None if attention_mask is None else attention_mask.astype(mx.bool_)
        want_main = bool(use_mtp and self.mtp_modules and labels is not None)
        want_dpr = bool(
            self.config.dpr_enabled
            and use_dpr
            and labels is not None
            and self.config.dpr_loss_weight > 0
        )
        result = self.model(
            input_ids,
            start_pos=start_pos,
            cache=cache,
            segment_ids=segment_ids,
            pad_mask=pad,
            decode=decode,
            prev_tokens=prev_tokens,
            collect_main=want_main,
            return_memory=requested,
            use_ced_recurrent=use_ced_recurrent,
            use_dpr=use_dpr,
            return_dpr=want_dpr,
            dpr_intervention=dpr_intervention,
        )
        hidden, mains, _ = result[:3]
        state, trace = thinking_state, None
        if requested:
            opts = dict(thinking_options or {})
            opts.pop("mode", None)
            boundary = self.model.layers[self.config.n_encoder_layers].attn
            state, trace = self.psr(
                result[3],
                psr_anchors,
                boundary.freq_cos,
                boundary.freq_sin,
                mode=mode,
                record_trace=return_thinking,
                **opts,
            )
            if cache is not None:
                cache.thinking_state = state.for_decode()
        if state is not None and state.mode != mode:
            raise ValueError("state mode mismatch; create a fresh condition")
        out = CausalLMOutput(thinking_state=state, thinking_trace=trace)
        if recurrent_active(self.config, use_ced_recurrent):
            out.ced_result = result[-1]
            out.ced_metrics = out.ced_result["metrics"]
        b, t, _ = hidden.shape
        positions = (
            (start_pos + mx.arange(t))[None]
            if not isinstance(start_pos, mx.array)
            else start_pos[:, None] + mx.arange(t)
        )
        positions = mx.broadcast_to(positions, (b, t))
        vector, covered, offsets = (
            None,
            mx.zeros((b, t), mx.bool_),
            mx.zeros((b, t), mx.int32),
        )
        if state is not None:
            vector, covered, offsets = self.psr.read_workspace(
                hidden, state, positions, segment_ids, pad
            )
        gate = (
            self.psr.calibration_gate
            if self.psr is not None and psr_gate is None
            else psr_gate
        )
        if labels is None:
            out.base_logits = self.logits(hidden)
            out.logits = (
                out.base_logits
                if vector is None
                else mx.stop_gradient(out.base_logits) + gate * self.psr.output(vector)
            )
            return out
        z_w = float(self.config.z_loss_weight)
        if return_sequence_losses or tail_reference is not None:
            from .tail_sft import sequence_head_losses, tail_objective

            sums, counts, zs = sequence_head_losses(
                hidden, self._head_weight(), labels, loss_mask, z_w
            )
            out.sequence_losses = sums / mx.maximum(counts, 1.0)
            out.sequence_token_counts = counts
            if tail_reference is not None:
                if tail_reference.shape != (b,) or tail_fraction is None:
                    raise ValueError(
                        "TailSFT requires a reference loss per sequence and a filter fraction"
                    )
                ce, z, keep, out.tail_stats = tail_objective(
                    sums, counts, zs, tail_reference, tail_fraction
                )
                mask = mx.ones((b, t), mx.float32) if loss_mask is None else loss_mask
                loss_mask = mask * keep[:, None]
            else:
                denom = mx.maximum(mx.sum(counts), 1.0)
                ce, z = mx.sum(sums) / denom, mx.sum(zs) / denom
        else:
            ce, z = lm_head_ce(hidden, self._head_weight(), labels, loss_mask, z_w)
        out.lm_loss, out.z_loss, out.loss = ce, z, ce + z_w * z
        valid = (
            mx.ones((b, t), mx.float32)
            if loss_mask is None
            else loss_mask.astype(mx.float32)
        )
        if vector is not None or return_metrics:
            base_losses, corrected_losses, returned_logits = [], [], []
            # Baseline CE above retains its exact original execution path. Correction
            # uses detached base logits/hidden and a separate task loss.
            for start in range(0, t, 256):
                stop = min(start + 256, t)
                base = mx.stop_gradient(self.logits(hidden[:, start:stop]))
                corrected = (
                    base
                    if vector is None
                    else base + gate * self.psr.output(vector[:, start:stop])
                )
                base_losses.append(
                    nn.losses.cross_entropy(
                        base.astype(mx.float32), labels[:, start:stop], reduction="none"
                    )
                )
                corrected_losses.append(
                    nn.losses.cross_entropy(
                        corrected.astype(mx.float32),
                        labels[:, start:stop],
                        reduction="none",
                    )
                )
                if need_logits:
                    returned_logits.append(corrected)
            base_nll, corrected_nll = (
                mx.concatenate(base_losses, 1),
                mx.concatenate(corrected_losses, 1),
            )
            count = mx.maximum(mx.sum(valid), 1)
            out.corrected_lm_loss = mx.sum(corrected_nll * valid) / count
            if vector is not None:
                weights = (
                    1 if psr_sample_weights is None else psr_sample_weights[:, None]
                )
                out.corrected_loss = (
                    mx.sum(corrected_nll * valid * covered * weights) / count
                )
                out.loss = out.loss + out.corrected_loss
            prefix_mask = (
                None
                if state is None
                else positions
                < mx.min(
                    mx.where(state.valid, state.anchor, self.config.max_seq_len), axis=1
                )[:, None]
            )
            out.metrics = residual_statistics(
                base_nll, corrected_nll, valid, covered, offsets, prefix_mask
            )
            out.base_token_nll, out.corrected_token_nll, out.bridge_mask = (
                base_nll,
                corrected_nll,
                covered,
            )
            if need_logits:
                out.logits = mx.concatenate(returned_logits, 1)
        elif need_logits:
            out.logits = self.logits(hidden)
        if want_main and mains:
            mtp_loss, mtp_z = self._mtp_loss(
                mains, input_ids, labels, loss_mask, pad, segment_ids
            )
            out.mtp_loss = mtp_loss
            out.z_loss = out.z_loss + mtp_z
            out.loss = out.loss + self.config.mtp_loss_weight * mtp_loss + z_w * mtp_z
        if (
            self.training
            and self._moe_layers
            and self.config.aux_balance_loss_weight > 0
        ):
            # 报告 §4.2.2：权重 1e-4 的序列级均衡损失（逐层求和后加权）
            out.aux_loss = mx.stack(
                [m._last_aux for m in self._moe_layers], axis=0
            ).sum()
            out.loss = out.loss + self.config.aux_balance_loss_weight * out.aux_loss
        if want_dpr:
            aux, metrics = self.model.dpr.auxiliary(
                *result[-1], input_ids, pad, segment_ids
            )
            weight = self.config.dpr_loss_weight if dpr_weight is None else dpr_weight
            out.dpr_loss = aux
            out.dpr_metrics = metrics.at[7].add(mx.array(weight, mx.float32))
            out.loss = out.loss + weight * aux
        if self.training and self._backbone_gates:
            # 作为图输出返回（compile 下纯侧信道会被剪枝，见 MoEGate.__call__）
            out.moe_loads = mx.stack(
                [g._last_load for g in self._backbone_gates], axis=0
            )
            if self.config.moe_balance_method == "qb":
                samples = [g._last_qb_margins for g in self._backbone_gates]
                if recurrent_active(self.config, use_ced_recurrent):
                    rows = max(s.shape[0] for s in samples)
                    samples = [
                        mx.concatenate(
                            [s, mx.full((rows - s.shape[0], s.shape[1]), mx.nan)],
                            axis=0,
                        )
                        if s.shape[0] < rows
                        else s
                        for s in samples
                    ]
                out.moe_qb_margins = mx.stack(samples, axis=0)
        return out

    # ------------------------------------------------------------------
    def dspark_draft(self, mains, known_ids, *, last_only=False):
        """DSpark logits/confidence for a known draft prefix [B,L], L<=block.

        mains are the backbone target-layer inputs at the processed anchor,
        each [B,1,D]. known_ids starts with that anchor token. Causal draft
        stages allow computing just the known prefix: future noise slots
        cannot affect these outputs. Preserve training's Markov alignment
        exactly: previous ids are [anchor, known_ids[:-1]].
        """
        if not self.mtp_modules:
            raise ValueError(
                "DSpark speculative decoding requires a checkpoint with MTP modules"
            )
        if len(mains) != len(self.config.dspark_target_layer_ids):
            raise ValueError(
                "DSpark requires all configured backbone target-layer inputs"
            )
        b, length = known_ids.shape
        if not 1 <= length <= self.config.dspark_block_size:
            raise ValueError("DSpark prefix length exceeds dspark_block_size")
        x = mx.concatenate(mains, axis=-1) if len(mains) > 1 else mains[0]
        cur = self._stage_input(x, known_ids[:, None, :], self.mtp_modules[0])
        pre = identity_pre_mix(cur, self.model.hc_mult)
        for stage in self.mtp_modules:
            cur, pre = stage.layer(cur, 0, pre, SharedAttnState())
        last = self.mtp_modules[-1]
        h = apply_hc_pre_norm(cur, pre, last.norm).reshape(b, length, self.config.dim)
        previous = mx.concatenate([known_ids[:, :1], known_ids[:, :-1]], axis=1)
        if last_only:
            h, previous = h[:, -1:], previous[:, -1:]
        bias, markov = last.markov_head(previous)
        return self.logits(h) + bias, last.confidence_head(h, markov)

    def _mtp_loss(self, mains, input_ids, labels, loss_mask, pad_mask, segment_ids):
        """DSpark teacher-forced 草稿损失（各槽 CE 均值 + 置信度头 BCE）。"""
        cfg = self.config
        block = cfg.dspark_block_size
        B, T = input_ids.shape
        # DSpark 目标不回传到主干（论文 §2.4.3：只训 DSpark，主干冻结）
        x = (
            mx.concatenate([mx.stop_gradient(m) for m in mains], axis=-1)
            if len(mains) > 1
            else mx.stop_gradient(mains[0])
        )
        idx = mx.minimum(mx.arange(T)[:, None] + mx.arange(block)[None, :], T - 1)
        draft_ids = mx.take(input_ids, idx, axis=1)  # [B,T,block]
        seg = self._draft_broadcast(segment_ids, B, T, block)
        pad = self._draft_broadcast(pad_mask, B, T, block)
        cur = self._stage_input(x, draft_ids, self.mtp_modules[0])
        # Single-Pass mHC：pre_mix 从 identity 起步、逐子层串下去（官方 forward_spec
        # 的 pre_mix 链），草稿头的 hc_pre 用最后一个 stage 的 FFN 产出的系数，
        # 而不是重新取 identity——否则头读的流与训练出来的残差流不一致。
        pre = identity_pre_mix(cur, self.model.hc_mult)
        for stage in self.mtp_modules:
            cur, pre = stage.layer(cur, 0, pre, SharedAttnState(), None, seg, pad)
        last = self.mtp_modules[-1]
        h = apply_hc_pre_norm(cur, pre, last.norm)
        anchors = B * T
        flat_h = h.reshape(anchors, block, cfg.dim)
        flat_ids = draft_ids.reshape(anchors, block)
        flat_anchor = input_ids.reshape(anchors)
        tgt = mx.take(labels, idx, axis=1).reshape(anchors, block)
        m = None
        if loss_mask is not None:
            m = (
                mx.take(loss_mask, idx, axis=1)
                .reshape(anchors, block)
                .astype(mx.float32)
            )
        else:
            m = mx.ones((anchors, block), dtype=mx.float32)
        z_w = float(cfg.z_loss_weight)
        chunk = max(1, 4096 // max(block, 1))
        ce_sum = mx.array(0.0, dtype=mx.float32)
        conf_sum = mx.array(0.0, dtype=mx.float32)
        z_sum = mx.array(0.0, dtype=mx.float32)
        w_sum = mx.array(0.0, dtype=mx.float32)
        for s in range(0, anchors, chunk):
            e = min(s + chunk, anchors)
            hs = flat_h[s:e]
            ids = flat_ids[s:e]
            prev = mx.concatenate([flat_anchor[s:e, None], ids[:, :-1]], axis=1)
            logits = self.logits(hs)
            bias, mk = last.markov_head(prev)
            logits = logits + bias
            w = m[s:e]
            ce = mx.fast.cross_entropy(logits, tgt[s:e])
            ce_sum = ce_sum + mx.sum(ce * w)
            w_sum = w_sum + mx.sum(w)
            if z_w:
                lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
                z_sum = z_sum + mx.sum(lse * lse * w)
            hit = (
                mx.stop_gradient(
                    (mx.argmax(logits, axis=-1) == tgt[s:e]).astype(mx.float32)
                )
                * w
            )
            conf = last.confidence_head(hs, mk)
            conf_sum = conf_sum + mx.sum(
                mx.maximum(conf, 0.0)
                - conf * hit
                + mx.logaddexp(mx.zeros_like(conf), -mx.abs(conf))
            )
        denom = mx.maximum(w_sum, 1.0)
        mtp = (ce_sum + conf_sum) / denom
        z_mean = z_sum / denom if z_w else mx.array(0.0)
        return mtp, z_mean

    def _stage_input(self, x, draft_ids, stage):
        """锚点表示 main_x 注入草稿槽 0，再展开成 hc 条流。"""
        main_x = stage.main_norm(stage.main_proj(x))
        emb = self.model.embed(draft_ids)  # [B,T,block,D]
        emb = emb.at[:, :, 0, :].add(main_x)
        B, T, blk, D = emb.shape
        emb = emb.reshape(B * T, blk, D)
        return mx.repeat(emb[:, :, None, :], self.model.hc_mult, axis=2)

    @staticmethod
    def _draft_broadcast(m, B, T, block):
        """把 [B,T] 的 mask 映射到草稿序列 [B*T, block]（每条草稿序列 = 一个锚点）。"""
        if m is None:
            return None
        return mx.repeat(m.reshape(B * T)[:, None], block, axis=1)

    # ------------------------------------------------------------------
    def prefill(
        self,
        input_ids: mx.array,
        segment_ids=None,
        *,
        use_thinking=True,
        thinking_options=None,
        psr_mode=None,
        psr_gate=None,
        use_ced_recurrent=True,
        use_dpr=True,
    ):
        """整段 prefill，返回 (logits [B,T,V], VibyCache)。"""
        if (
            self.config.psr_enabled
            and use_thinking
            and psr_mode != "off"
            and segment_ids is not None
        ):
            if bool(mx.any(segment_ids != segment_ids[:, :1])):
                raise ValueError(
                    "multi-document cached PSR generation is unsupported; use uncached packed evaluation"
                )
        cache = VibyCache(self.config, input_ids.shape[0])
        out = self(
            input_ids,
            start_pos=0,
            cache=cache,
            segment_ids=segment_ids,
            use_mtp=False,
            use_thinking=use_thinking,
            thinking_options=thinking_options,
            psr_mode=psr_mode,
            psr_gate=psr_gate,
            use_ced_recurrent=use_ced_recurrent,
            use_dpr=use_dpr,
            thinking_prefix_lengths=input_ids.shape[1]
            if self.config.psr_enabled and use_thinking
            else None,
        )
        cache.start_pos = input_ids.shape[1]
        cache.psr_next_anchor = (
            input_ids.shape[1] - 1 + self.config.psr_horizon
            if cache.thinking_state is not None
            else None
        )
        cache.psr_phases = 1 if cache.thinking_state is not None else 0
        cache.psr_options = dict(thinking_options or {})
        cache.psr_gate = psr_gate
        if self.model.engram_hash is not None:
            w = max(self.config.engram_max_ngram_size - 1, 0)
            cache.engram_prev = (
                mx.concatenate(
                    [
                        mx.full((input_ids.shape[0], w), -1, mx.int32),
                        input_ids,
                    ],
                    axis=1,
                )[:, -w:]
                if w
                else mx.zeros((input_ids.shape[0], 0), dtype=mx.int32)
            )
        return out.logits, cache

    def decode_step(self, token_ids: mx.array, cache: VibyCache, *, psr_gate=None):
        """单步解码，返回 (logits [B,V], cache)。"""
        cache.decode_max_pos = cache.start_pos + 1
        if psr_gate is None:
            psr_gate = cache.psr_gate
        new_anchor = None
        if (
            cache.psr_next_anchor is not None
            and cache.start_pos >= cache.psr_next_anchor
        ):
            cache.thinking_state = None
            new_anchor = mx.full((token_ids.shape[0], 1), cache.start_pos, mx.int32)
            cache.psr_next_anchor += self.config.psr_horizon
            cache.psr_phases += 1
        out = self(
            token_ids[:, None],
            start_pos=cache.start_pos,
            cache=cache,
            decode=True,
            prev_tokens=cache.engram_prev,
            use_mtp=False,
            psr_mode=cache.psr_mode,
            psr_gate=psr_gate,
            psr_anchors=new_anchor,
            use_ced_recurrent=cache.ced_signature != ("baseline",),
            use_dpr=getattr(cache, "dpr_signature", (True, None))[0],
            dpr_intervention=getattr(cache, "dpr_signature", (True, None))[1],
            thinking_options=cache.psr_options if new_anchor is not None else None,
        )
        if self.model.engram_hash is not None:
            w = max(self.config.engram_max_ngram_size - 1, 0)
            if w:
                prev = cache.engram_prev
                if prev is None or prev.shape[1] == 0:
                    cache.engram_prev = mx.broadcast_to(
                        token_ids[:, None], (token_ids.shape[0], w)
                    )
                else:
                    cache.engram_prev = mx.concatenate(
                        [prev, token_ids[:, None]], axis=1
                    )[:, -w:]
        cache.start_pos += 1
        return out.logits[:, -1], cache

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        *,
        use_thinking=True,
        thinking_options=None,
        psr_mode=None,
        psr_gate=None,
        use_ced_recurrent=True,
    ):
        """自回归解码（prefill 一次 + 逐 token decode）。"""
        cfg = self.config
        B = input_ids.shape[0]
        logits, cache = self.prefill(
            input_ids,
            use_thinking=use_thinking,
            thinking_options=thinking_options,
            psr_mode=psr_mode,
            psr_gate=psr_gate,
            use_ced_recurrent=use_ced_recurrent,
        )
        logits = logits[:, -1]
        eos = cfg.eos_token_id if eos_token_id is None else eos_token_id
        finished = mx.zeros((B,), dtype=mx.bool_)
        toks = []
        for _ in range(max_new_tokens):
            tok = sample_tokens(logits, temperature, top_k)
            toks.append(tok)
            finished = finished | (tok == eos)
            if bool(mx.all(finished).item()):
                break
            logits, cache = self.decode_step(tok, cache, psr_gate=psr_gate)
        if not toks:
            return input_ids
        return mx.concatenate([input_ids, mx.stack(toks, axis=1)], axis=1)

    # ------------------------------------------------------------------
    @property
    def moe_gates(self):
        return self._moe_gates

    def moe_load_stats(self):
        if not self._backbone_gates:
            return None
        return mx.stack([g._last_load for g in self._backbone_gates], axis=0)

    def moe_bias_stack(self):
        return mx.stack([g.bias for g in self._backbone_gates], axis=0)

    def apply_moe_biases(self, biases: mx.array):
        for g, b in zip(self._backbone_gates, biases):
            g.bias = b.astype(mx.float32)

    def update_moe_biases(self, loads: mx.array, qb_margins=None):
        """Completed-window QB or negative-feedback sign update; no gradients.

        loads: [n_backbone_gates, E]（= forward 返回的 out.moe_loads）。
        """
        if self.config.moe_balance_method == "qb":
            if qb_margins is None:
                raise ValueError(
                    "QB requires forward-returned moe_qb_margins from the completed window"
                )
            if (
                qb_margins.ndim != 3
                or qb_margins.shape[0] != len(self._backbone_gates)
                or qb_margins.shape[2] != self.config.n_routed_experts
            ):
                raise ValueError(
                    "QB margins must have shape [backbone_gates, samples, experts]"
                )
            for g, margins in zip(self._backbone_gates, qb_margins):
                g.bias = update_quantile_bias(
                    g.bias,
                    margins,
                    g.top_k,
                    self.config.qb_update_rate,
                    ignore_padding=getattr(self.config, "ced_recurrent_enabled", False),
                )
        else:
            for g, ld in zip(self._backbone_gates, loads):
                g.bias = update_expert_bias(
                    g.bias, ld.astype(mx.float32), self.config.bias_update_rate
                )

    def num_parameters(self, include_engram: bool = True) -> int:
        total = 0
        for path, arr in tree_flatten(self.parameters()):
            if not include_engram and "engram_layers" in path and "embed" in path:
                continue
            total += arr.size
        return total

    def ngram_lookup_parameters(self) -> int:
        """Engram 查表参数（日志里单列）。"""
        total = 0
        for path, arr in tree_flatten(self.parameters()):
            if "engram_layers" in path and ".embed." in path:
                total += arr.size
        return total

    def num_active_parameters(self) -> int:
        return self.config.num_active_parameters()

    # ------------------------------------------------------------------
    def save_pretrained(self, path: str):
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str):
        return cls(VibyConfig.from_pretrained(path))
