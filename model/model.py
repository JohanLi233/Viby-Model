import os
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .block import MTPModule, VibyStack
from .init import apply_trunc_normal_init
from .cache import (
    KVCache,
    _eval_kv_caches,
    _rewind_cache,
    _rewind_cache_list,
    _seq_len_of_cache,
)
from .config import VibyConfig
from .attention import precompute_freqs_cis
from .kernels.ce import cross_entropy
from .kernels.lm_ce import lm_head_cross_entropy
from .moe import MoEGate, _col_quantile
from .ngram import NgramEmbedding
from .norms import GatedNorm


@dataclass
class CausalLMOutput:
    loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    past_key_values: Optional[list] = None
    hidden_states: Optional[mx.array] = None
    # 纯语言建模 CE（未加 MTP/aux/diversity，仅日志/评估用）
    lm_loss: Optional[mx.array] = None
    # MTP 辅助 loss 分量（未加权，仅日志展示用；无 MTP 时为 None）
    mtp_loss: Optional[mx.array] = None
    # router 输入 token 多样性正则 loss（未加权）
    diversity_loss: Optional[mx.array] = None
    # logit z-loss 实际加进总 loss 的加权贡献（仅日志；未启用/无 labels 为 None）
    z_loss: Optional[mx.array] = None
    # Qwen MTP：mtp_feature_layers 指定层在 AttnRes 合并后、final_norm 前
    # 的 hidden。现行只抽末层，list 长度 1；训练（带 labels 且 mtp_depth>0）
    # 或 output_features=True 时返回。
    features: Optional[list] = None


class VibyModel(nn.Module):
    def __init__(self, config: VibyConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 嵌入后归一化（GatedNorm 包装的 learnable-gain RMSNorm）：
        # lookup 之后、dropout 之前。嵌入表不再直接进入 AttnRes 残差流，
        # 先归一化压住逐 token 的嵌入尺度差异。
        self.embed_norm = GatedNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.ngram = NgramEmbedding(config) if config.ngram_table_size > 0 else None
        self._ngram_inject = (
            int(config.ngram_layer) - 1 if self.ngram is not None else None
        )
        # 顺序主干：num_hidden_layers 层单栈，残差流为 AttnRes（栈内维护）
        self.stack = VibyStack(config, config.num_hidden_layers)
        # Qwen MTP 只抽主干末层（final_norm 前）。
        self._mtp_feature_idx = (config.num_hidden_layers - 1,)
        if not config.use_linear_attn:
            rope_scaling = config.rope_scaling
            if rope_scaling is not None:
                rope_scaling = dict(rope_scaling)
                rope_scaling.setdefault(
                    "original_max_position_embeddings",
                    config.original_max_position_embeddings,
                )
            freqs_cos, freqs_sin, rope_freqs, rope_attn_factor = precompute_freqs_cis(
                dim=config.qk_rope_head_dim,
                end=config.max_position_embeddings,
                rope_base=config.rope_theta,
                rope_scaling=rope_scaling,
                return_freqs=True,
            )
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
            self.rope_freqs = rope_freqs
            self.rope_attn_factor = float(rope_attn_factor)
            self.freeze(
                recurse=False,
                keys=["freqs_cos", "freqs_sin", "rope_freqs"],
            )
        else:
            self.freqs_cos = None
            self.freqs_sin = None
            self.rope_freqs = None
            self.rope_attn_factor = 1.0

    def position_embeddings(
        self, start_pos: int, seq_length: int, dtype, position_ids=None
    ):
        """MLA 用的 (cos, sin, (freqs, offset, attn_factor))；linear 模式返回 None。

        ``position_ids`` 为 (B, T) 时返回逐序列 cos/sin，第三项为 None（不走
        单一 offset 的 mx.fast.rope），供连续 batch 里各请求不同缓存长度使用。
        """
        if self.freqs_cos is None:
            return None
        if position_ids is not None:
            # 必须从 rope_freqs 现算。checkpoint 里的 freqs_cos/sin 是 2D
            # 矩阵，trunc_normal init 会覆盖它们；训练/generate 走
            # mx.fast.rope(rope_freqs)，engine 连续 batch 才需要这张表。
            pos = position_ids.astype(mx.float32)
            angles = pos[..., None] / self.rope_freqs.astype(mx.float32)
            cos_h, sin_h = mx.cos(angles), mx.sin(angles)
            af = float(self.rope_attn_factor)
            freqs_cos = mx.concatenate([cos_h, cos_h], axis=-1) * af
            freqs_sin = mx.concatenate([sin_h, sin_h], axis=-1) * af
            if freqs_cos.dtype != dtype:
                freqs_cos = freqs_cos.astype(dtype)
                freqs_sin = freqs_sin.astype(dtype)
            return (freqs_cos, freqs_sin, None)
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_length]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_length]
        if freqs_cos.dtype != dtype:
            freqs_cos = freqs_cos.astype(dtype)
            freqs_sin = freqs_sin.astype(dtype)
        return (
            freqs_cos,
            freqs_sin,
            (self.rope_freqs, start_pos, self.rope_attn_factor),
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        output_features: bool = False,
        unembed_weight: Optional[mx.array] = None,
        **kwargs,
    ) -> tuple[mx.array, list, list]:
        batch_size, seq_length = input_ids.shape
        # loop 生效时 cache 按执行位置计（每 visit 一个独立 cache）
        n_exec = self.config.num_exec_layers
        if past_key_values is None:
            past_key_values = [None] * n_exec
        elif len(past_key_values) != n_exec:
            raise ValueError(
                f"past_key_values 层数 {len(past_key_values)} 与执行层数 {n_exec} 不一致"
            )
        first_cache = past_key_values[0]
        start_pos = _seq_len_of_cache(first_cache)
        if start_pos + seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"输入长度 {start_pos + seq_length} 超过模型最大上下文长度 "
                f"{self.config.max_position_embeddings}"
            )

        position_ids = kwargs.pop("position_ids", None)
        attn_bias = kwargs.pop("attn_bias", None)
        hidden_states = self.dropout(self.embed_norm(self.embed_tokens(input_ids)))
        pos_emb = self.position_embeddings(
            start_pos, seq_length, hidden_states.dtype, position_ids=position_ids
        )

        # 每微批只 sync/构造一次 causal+pad 融合 mask，所有层共享：
        # 有 padding 时普通 Attention 也能走 flash，而不是逐层退化到 O(T²)。
        # mask_has_pad 可由调用方在 eager 侧预先算好传入（mx.compile 图内
        # 不允许 .item() host sync）；未传入时按原逻辑现场判断。
        # attn_bias：engine 传入的 (B,1,Q,K) 含 past+pad，优先于现场构造。
        mask_is_full = True
        causal_bias = None
        if attn_bias is not None:
            causal_bias = attn_bias.astype(hidden_states.dtype)
            mask_is_full = False
        else:
            if mask_has_pad is None:
                mask_has_pad = attention_mask is not None and bool(
                    mx.any(attention_mask != 1).item()
                )
            if attention_mask is not None and mask_has_pad:
                mask_is_full = False
                am = attention_mask.astype(mx.bool_)
                pad_bias = mx.where(am, 0.0, -1e9)  # (B, T)
                causal = mx.triu(mx.full((seq_length, seq_length), -1e9), k=1)
                causal_bias = (
                    causal[None, None, :, :] + pad_bias[:, None, None, :]
                ).astype(hidden_states.dtype)

        # 文档边界掩码（doc_mask 打包训练）：注意力限制在同文档内因果可见，
        # 消除跨文档泄漏，与逐篇 PPL 评估口径对齐。仅在完整前向（训练）传入。
        # 同条件下的 segment_ids 同时下发给各层 ShortConv 做卷积窗段掩码。
        seg_for_conv = None
        if segment_ids is not None and first_cache is None and seq_length > 1:
            seg_for_conv = segment_ids
            same_doc = segment_ids[:, :, None] == segment_ids[:, None, :]
            causal_tril = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
            allowed = same_doc & causal_tril[None, :, :]
            seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(
                hidden_states.dtype
            )
            causal_bias = seg_bias if causal_bias is None else causal_bias + seg_bias
            mask_is_full = False

        ngram_io = None
        ngram_tail = None
        ngram_raw = None
        if self.ngram is not None:
            extras = (
                getattr(first_cache, "extras", None)
                if first_cache is not None
                else None
            )
            ngram_conv_tail = None
            if extras is not None:
                ngram_tail = extras.get("ngram_tail")
                ngram_conv_tail = extras.get("ngram_conv_tail")
            ngram_raw = self.ngram.lookup(
                input_ids,
                tail=ngram_tail,
                segment_ids=segment_ids if ngram_tail is None else None,
            )
            # 检索 pre-stack 完成；门控/卷积在注入层内用当时的 hidden 做。
            # conv_tail 由 stack 写回本 dict，随后落入 cache extras。
            ngram_io = {
                "module": self.ngram,
                "e": ngram_raw,
                "conv_state": ngram_conv_tail,
                "conv_tail": None,
            }
        self._ngram_raw = ngram_raw

        write_gate = None
        if (
            bool(getattr(self.config, "ngram_conf_gate", False))
            and ngram_raw is not None
        ):
            w = (
                unembed_weight
                if unembed_weight is not None
                else self.embed_tokens.weight
            )
            write_gate = self.ngram.confidence_write_gate(ngram_raw, w)

        hidden_states, presents, features = self.stack(
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=seg_for_conv,
            feature_layers=(self._mtp_feature_idx if output_features else None),
            ngram_io=ngram_io,
            ngram_inject=self._ngram_inject,
            position_embeddings=pos_emb,
            write_gate=write_gate,
        )
        if self.ngram is not None and use_cache and presents:
            c0 = presents[0]
            if getattr(c0, "extras", None) is not None:
                k = self.ngram.context_len
                ids_i = input_ids.astype(mx.int32)
                if ngram_tail is not None:
                    hist = mx.concatenate([ngram_tail.astype(mx.int32), ids_i], axis=1)
                else:
                    hist = ids_i
                if hist.shape[1] < k:
                    hist = mx.concatenate(
                        [
                            mx.zeros(
                                (hist.shape[0], k - hist.shape[1]), dtype=hist.dtype
                            ),
                            hist,
                        ],
                        axis=1,
                    )
                tail = hist[:, -k:]
                c0.extras["ngram_tail"] = tail
                tr = c0.extras.get("ngram_tail_trace")
                if tr is not None:
                    tr.append((c0.offset, tail))
                conv_tail = ngram_io.get("conv_tail") if ngram_io is not None else None
                if conv_tail is not None:
                    c0.extras["ngram_conv_tail"] = conv_tail
                    ctr = c0.extras.get("ngram_conv_tail_trace")
                    if ctr is not None:
                        ctr.append((c0.offset, conv_tail))
        return hidden_states, presents, features


def _transform_logits_mx(
    logits: mx.array,
    seen_ids: Optional[mx.array],
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
) -> mx.array:
    """采样前的 logits 变换（repetition penalty / top_k / top_p），全程 MLX。

    logits: (..., V) 或 (V,)；seen_ids: (N,) 或 (B, N) 已见 token。
    返回与旧 numpy 实现等价的变换后 logits，不离开 GPU。
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None]
        if seen_ids is not None:
            seen_ids = seen_ids[None]

    logits = logits / temperature

    if repetition_penalty != 1.0 and seen_ids is not None and seen_ids.size > 0:
        rows = []
        for b in range(logits.shape[0]):
            row = logits[b]
            seen = seen_ids[b] if seen_ids.ndim == 2 else seen_ids
            # 每个唯一 token id 只惩罚一次：重复索引经 scatter-max 归并为 0/1 掩码，
            # 无需 sort、无需 CPU 同步（.at 对重复索引是累加语义，不能直接 gather-scatter）。
            mask = (
                mx.zeros(row.shape[-1], dtype=mx.int32)
                .at[seen.astype(mx.int32)]
                .maximum(1)
            )
            scale = mx.where(row > 0, 1.0 / repetition_penalty, repetition_penalty)
            row = mx.where(mask.astype(mx.bool_), row * scale, row)
            rows.append(row)
        logits = mx.stack(rows)

    if do_sample and top_k > 0:
        kth = mx.partition(logits, -top_k, axis=-1)[..., -top_k : -top_k + 1]
        logits = mx.where(logits < kth, -mx.inf, logits)
    if do_sample and top_p < 1.0:
        order = mx.argsort(-logits, axis=-1).astype(mx.int32)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        exp_shifted = mx.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        sorted_probs = exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)
        cumprobs = mx.cumsum(sorted_probs, axis=-1)
        remove = cumprobs > top_p
        # 与 numpy 一致：第一个越过阈值的 token 保留，其后移除
        remove = mx.concatenate(
            [mx.zeros(remove.shape[:-1] + (1,), dtype=mx.bool_), remove[..., :-1]],
            axis=-1,
        )
        masked = mx.where(remove, -mx.inf, sorted_logits)
        logits = mx.put_along_axis(logits, order, masked, axis=-1)

    return logits[0] if squeeze else logits


def _probs_mx(logits: mx.array) -> mx.array:
    """softmax，返回概率分布（与 logits 同形状）。"""
    exp_shifted = mx.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _sample_from_logits_mx(logits: mx.array, do_sample: bool) -> mx.array:
    """从（已变换的）logits 采样一个 token 下标，全程 MLX。

    do_sample=True 时按概率采样（GPU 随机数），否则取 argmax。
    """
    if do_sample:
        return mx.random.categorical(logits, axis=-1)
    return mx.argmax(logits, axis=-1)


class VibyForCausalLM(nn.Module):
    def __init__(self, config: Optional[VibyConfig] = None, skip_init: bool = False):
        super().__init__()
        config = config or VibyConfig()
        self.config = config
        self.model = VibyModel(config)
        if config.tie_word_embeddings:
            # Tied lm_head（兼容/消融路径）：复用嵌入权重，无额外参数。
            # 默认已解绑（tie_word_embeddings=False）。
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_modules = [MTPModule(config)] if config.mtp_depth > 0 else []
        # module 树建完后缓存 gate 列表；每步前向避免重复 isinstance 遍历。
        self._moe_gates = [m for m in self.modules() if isinstance(m, MoEGate)]
        # eval 路径权重随即被 checkpoint 覆盖时传 skip_init=True：截断正态
        # 初始化会强制逐张量求值，n-gram 表（fp32 6.4GB）瞬时峰值 ~3-4 倍。
        if not skip_init:
            apply_trunc_normal_init(self, config.hidden_size)

    def _lm_logits(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states @ self.model.embed_tokens.weight.T

    def _lm_ce_from_hidden(
        self,
        hidden: mx.array,
        labels: mx.array,
        loss_mask: Optional[mx.array],
        ngram_raw: Optional[mx.array],
    ) -> tuple:
        """从 hidden 直接算 (CE, z_mean)：分块融合 lm_head+CE，不物化全量
        (B,T,V) logits（bs16×4096×V6400 bf16 ≈ 840MB，且反向还要再留一份）。

        ngram logit skip 启用时回退全量 logits 路径：logit_scale 可训练，
        融合 kernel 的 VJP 未覆盖它与 ngram_raw 的梯度。"""
        ng = self.model.ngram
        scale = getattr(ng, "logit_scale", None) if ng is not None else None
        if scale is not None and ngram_raw is not None:
            logits = self._apply_ngram_logit_skip(self._lm_logits(hidden), ngram_raw)
            return cross_entropy(logits, labels, mask=loss_mask, return_z=True)
        w = (
            self.lm_head.weight
            if self.lm_head is not None
            else self.model.embed_tokens.weight
        )
        return lm_head_cross_entropy(hidden, w, labels, mask=loss_mask, return_z=True)

    def _apply_ngram_logit_skip(
        self, logits: mx.array, ngram_raw: Optional[mx.array]
    ) -> mx.array:
        ng = self.model.ngram
        scale = getattr(ng, "logit_scale", None) if ng is not None else None
        if scale is None or ngram_raw is None:
            return logits
        return logits + scale.astype(logits.dtype) * self._lm_logits(ngram_raw)

    def _mtp_loss(
        self,
        features: list,
        input_ids: mx.array,
        labels: mx.array,
        loss_mask: Optional[mx.array],
        attention_mask: Optional[mx.array],
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        steps: Optional[int] = None,
        ngram_raw: Optional[mx.array] = None,
    ) -> tuple:
        """Qwen3.8-Next MTP loss：同一层 teacher-forced 多步展开。

        step k 在流位置 j 消费上一深度 hidden[j] 与 Emb(token j+k)，预测
        labels[:, k:]（即 token j+k+1；labels 已是 next-token 移位）。
        step 1 的 hidden 是主干末层（final_norm 前）；之后复用本模块输出。
        各步 CE / lse 取平均。--doc_mask 时 MTP 自注意力按源位置文档边界
        掩码，避免 packed 序列把跨文档泄漏回传到主干。
        """
        n_steps = int(self.config.mtp_steps if steps is None else steps)
        if n_steps < 1 or not self.mtp_modules:
            return mx.array(0.0), mx.array(0.0)
        seq_len = input_ids.shape[1]
        token_emb = self.model.embed_tokens(input_ids)
        h = features[-1] if isinstance(features, (list, tuple)) else features
        module = self.mtp_modules[0]
        terms = []
        z_terms = []
        dtype = token_emb.dtype
        for k in range(1, n_steps + 1):
            if seq_len <= k:
                break
            sub = seq_len - k
            am = attention_mask[:, :sub] if attention_mask is not None else None
            if mask_has_pad is None:
                mask_is_full = am is None or bool(mx.all(am == 1).item())
            else:
                mask_is_full = am is None or not mask_has_pad
            causal_bias = None
            if am is not None and not mask_is_full:
                pad_bias = mx.where(am.astype(mx.bool_), 0.0, -1e9)
                causal = mx.triu(mx.full((sub, sub), -1e9), k=1)
                causal_bias = (
                    causal[None, None, :, :] + pad_bias[:, None, None, :]
                ).astype(dtype)
            seg = None
            if segment_ids is not None:
                seg = segment_ids[:, :sub]
                same_doc = seg[:, :, None] == seg[:, None, :]
                causal_tril = mx.tril(mx.ones((sub, sub), dtype=mx.bool_))
                allowed = same_doc & causal_tril[None, :, :]
                seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(dtype)
                causal_bias = (
                    seg_bias if causal_bias is None else causal_bias + seg_bias
                )
                mask_is_full = False
            h, _ = module(
                h[:, :sub, :],
                token_emb[:, k:, :],
                attention_mask=am,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                segment_ids=seg,
                position_embeddings=self.model.position_embeddings(
                    0, sub, token_emb.dtype
                ),
            )
            skip = ngram_raw[:, k : k + sub] if ngram_raw is not None else None
            ce_k, z_k = self._lm_ce_from_hidden(
                h,
                labels[:, k:],
                loss_mask[:, k:] if loss_mask is not None else None,
                skip,
            )
            terms.append(ce_k)
            z_terms.append(z_k)
        if not terms:
            return mx.array(0.0), mx.array(0.0)
        return sum(terms) / len(terms), sum(z_terms) / len(z_terms)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Union[list, tuple]] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        labels: Optional[mx.array] = None,
        loss_mask: Optional[mx.array] = None,
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        output_features: bool = False,
        need_logits: bool = False,
        **kwargs,
    ) -> CausalLMOutput:
        # MoE 负载统计/辅助 loss 按"次前向"重置（见 MoEGate.__call__）。
        for g in self._moe_gates:
            g.last_div = None
            g.div_calls = 0
            if g.collect_stats:
                g.last_load = None
                g.last_margins = None
        # Qwen MTP：训练（MTP loss）与投机解码验证路径需要末层 hidden；
        # 捕获只是记录引用，代价可忽略。
        want_feats = output_features or (
            labels is not None and len(self.mtp_modules) > 0
        )
        hidden_states, past_key_values, features = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            mask_has_pad=mask_has_pad,
            segment_ids=segment_ids,
            output_features=want_feats,
            unembed_weight=(None if self.lm_head is None else self.lm_head.weight),
            **kwargs,
        )
        ngram_raw = getattr(self.model, "_ngram_raw", None)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
            if ngram_raw is not None:
                ngram_main = ngram_raw[:, -logits_to_keep:, :]
            else:
                ngram_main = None
        else:
            ngram_main = ngram_raw
        # 训练 loss 走 hidden → 分块融合 CE（见 _lm_ce_from_hidden），无需全量
        # logits；只有生成/评估（labels=None）、LAR 诊断（need_logits）或
        # logits_to_keep 截断路径才物化完整 logits。
        want_logits = (
            labels is None
            or need_logits
            or (isinstance(logits_to_keep, int) and logits_to_keep > 0)
        )
        logits = (
            self._apply_ngram_logit_skip(self._lm_logits(hidden_states), ngram_main)
            if want_logits
            else None
        )

        loss = None
        lm_loss = None
        mtp_loss = None
        z_loss = None
        if labels is not None:
            if logits is not None:
                lm_loss, lm_z = cross_entropy(
                    logits,
                    labels,
                    mask=loss_mask,
                    return_z=True,
                )
            else:
                lm_loss, lm_z = self._lm_ce_from_hidden(
                    hidden_states, labels, loss_mask, ngram_main
                )
            loss = lm_loss
            z_w = self.config.z_loss_weight
            apply_z = z_w > 0.0 and self.training
            if apply_z:
                loss = loss + z_w * lm_z
                z_loss = z_w * lm_z
            if self.config.mtp_depth > 0 and self.mtp_modules:
                mtp_loss, mtp_z = self._mtp_loss(
                    features,
                    input_ids,
                    labels,
                    loss_mask,
                    attention_mask,
                    mask_has_pad=mask_has_pad,
                    segment_ids=segment_ids,
                    ngram_raw=ngram_raw,
                )
                loss = loss + self.config.mtp_loss_weight * mtp_loss
                if apply_z:
                    # MTP 各 head 与主 LM 同 z_loss_weight；随 mtp_loss_weight 缩放
                    z_loss = z_loss + z_w * self.config.mtp_loss_weight * mtp_z
                    loss = loss + z_w * self.config.mtp_loss_weight * mtp_z
            div_loss = self.moe_diversity_loss()
            if (
                div_loss is not None
                and self.config.moe_diversity_loss_weight > 0.0
                and self.training
            ):
                loss = loss + self.config.moe_diversity_loss_weight * div_loss
        else:
            div_loss = None

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            diversity_loss=div_loss,
            z_loss=z_loss,
            features=features if want_feats else None,
        )

    def generate(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = 50,
        eos_token_id: Optional[int] = 2,
        streamer=None,
        use_cache: bool = True,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        return_kv: bool = False,
        use_mtp_speculative: bool = False,
        num_speculative_tokens: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids 必须是 2 维 (batch, seq)，实际为 {input_ids.ndim} 维"
            )
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences 必须 >= 1")
        if top_k < 0:
            raise ValueError("top_k 不能为负数")
        if top_k > self.config.vocab_size:
            raise ValueError(
                f"top_k ({top_k}) 不能超过 vocab_size ({self.config.vocab_size})"
            )

        # 生成长度护栏：总长不得超过 max_position_embeddings（前向里有同样的
        # 显式越界检查），提前在此给出清晰报错并收紧 max_new_tokens。
        prompt_len = input_ids.shape[1]
        if prompt_len > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt 长度 {prompt_len} 超过模型最大上下文 "
                f"{self.config.max_position_embeddings}"
            )
        max_new_tokens = min(
            max_new_tokens, self.config.max_position_embeddings - prompt_len
        )

        # 全 1 mask 是生成主路径的常见情形：后续 decode 步直接传
        # mask_has_pad=False，避免每步在模型内对 attention_mask 做
        # mx.any(...).item() 同步；也无需每步 concatenate 全 1 mask。
        mask_ok = attention_mask is None or bool(mx.all(attention_mask == 1).item())
        # 投机解码只支持全 1 的 attention_mask（无 padding）。
        if use_mtp_speculative:
            can_speculate = (
                len(self.mtp_modules) > 0
                and input_ids.shape[0] == 1
                and num_return_sequences == 1
                and use_cache
                and mask_ok
            )
            if can_speculate:
                generated = self._generate_speculative(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=eos_token_id,
                    streamer=streamer,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    num_speculative_tokens=num_speculative_tokens,
                )
                if return_kv:
                    return {
                        "generated_ids": generated,
                        "past_kv": getattr(self, "_last_spec_past", None),
                    }
                return generated
            import warnings

            warnings.warn(
                "use_mtp_speculative requires mtp_depth > 0, batch size 1, "
                "use_cache=True 且 attention_mask 全为 1；已回退到标准生成。"
            )

        if num_return_sequences > 1:
            input_ids = mx.concatenate([input_ids] * num_return_sequences, axis=0)
            if attention_mask is not None:
                attention_mask = mx.concatenate(
                    [attention_mask] * num_return_sequences, axis=0
                )
        past_key_values = None
        batch = input_ids.shape[0]
        finished = mx.zeros(batch, dtype=mx.bool_)

        if streamer:
            streamer.put(input_ids)

        if max_new_tokens <= 0:
            if streamer:
                streamer.end()
            if return_kv:
                return {"generated_ids": input_ids, "past_kv": None}
            return input_ids

        # 预分配 token 缓冲，避免每步 concatenate 整段序列（图膨胀 + 拷贝）。
        total_len = prompt_len + max_new_tokens
        tokens = mx.zeros((batch, total_len), dtype=input_ids.dtype)
        tokens[:, :prompt_len] = input_ids
        n = prompt_len

        for _ in range(max_new_tokens):
            past_len = _seq_len_of_cache(past_key_values[0]) if past_key_values else 0
            current_input_ids = (
                tokens[:, past_len:n] if past_len < n else tokens[:, n - 1 : n]
            )
            outputs = self(
                current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                mask_has_pad=(not mask_ok),
            )
            logits_mx = outputs.logits[:, -1, :]
            if attention_mask is not None and not mask_ok:
                attention_mask = mx.concatenate(
                    [attention_mask, mx.ones((batch, 1), dtype=attention_mask.dtype)],
                    axis=-1,
                )

            seen_ids = tokens[:, :n] if repetition_penalty != 1.0 else None
            transformed = _transform_logits_mx(
                logits_mx,
                seen_ids,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            )
            next_token = _sample_from_logits_mx(transformed, do_sample)

            if eos_token_id is not None:
                next_token = mx.where(
                    finished,
                    mx.full_like(next_token, eos_token_id),
                    next_token,
                )

            tokens[:, n] = next_token.reshape((batch,)).astype(tokens.dtype)
            n += 1
            past_key_values = outputs.past_key_values if use_cache else None
            # 每步物化：否则惰性图会把所有 decode 步串在一起，T 增大后
            # 吞吐塌缩。eval 之后的 EOS .item() 几乎不再额外同步。
            mx.eval(next_token, tokens)
            if use_cache:
                _eval_kv_caches(past_key_values)

            if streamer:
                streamer.put(next_token[:, None])

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if bool(mx.all(finished).item()):
                    break

        input_ids = tokens[:, :n]
        if streamer:
            streamer.end()
        cache_len = _seq_len_of_cache(past_key_values[0]) if past_key_values else 0
        if return_kv and use_cache and past_key_values is not None and cache_len < n:
            # 标准循环每步先 forward 当前 token、再采样并 append，因此循环
            # 结束时 cache 还差最后一个已生成 token。这里补一次纯前向，让
            # return_kv 的 cache 与 generated_ids 对齐（与投机路径一致）。
            missing = tokens[:, cache_len:n]
            outputs = self(
                missing,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                mask_has_pad=(not mask_ok),
            )
            past_key_values = outputs.past_key_values
        if return_kv:
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids

    def _mtp_draft(
        self,
        feat_last: mx.array,
        first_token: int,
        pos_idx: int,
        draft_len: int,
        seen_ids: mx.array,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        mtp_past: Optional[tuple] = None,
    ) -> tuple[list[int], list[mx.array], Optional[tuple]]:
        """Draft tokens by recursively applying the single Qwen MTP layer.

        feat_last: 主干在位置 pos_idx 的末层 hidden，(1, 1, hidden)，只用于
        第一个草稿步；step≥1 消费草稿自身上一步输出与自草稿 token 嵌入。
        first_token: bonus token at pos_idx + 1.
        draft_len: 每轮草稿 token 数（默认 mtp_steps，循环复用同一层）。
        mtp_past: MTP 层 KV cache（prefill 教师强制、验证后回滚追平）。
        """
        token = mx.array([[first_token]])
        h = None
        drafts: list[int] = []
        draft_probs: list[mx.array] = []
        module = self.mtp_modules[0]
        del pos_idx
        for i in range(draft_len):
            emb = self.model.embed_tokens(token)
            h_in = feat_last if i == 0 else h
            h, present = module(
                h_in,
                emb,
                past_key_value=mtp_past,
                use_cache=True,
                position_embeddings=self.model.position_embeddings(
                    _seq_len_of_cache(mtp_past), 1, emb.dtype
                ),
            )
            mtp_past = present
            logits = self._lm_logits(h)[0, 0]
            # repetition penalty 的上下文应包含 bonus token 与已生成的草稿
            seen_cur = mx.concatenate(
                [seen_ids, mx.array([first_token] + drafts, dtype=mx.int32)]
            )
            transformed = _transform_logits_mx(
                logits,
                seen_cur,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
            )
            probs = _probs_mx(transformed)
            tok = int(_sample_from_logits_mx(transformed, do_sample).item())
            drafts.append(tok)
            draft_probs.append(probs)
            token = mx.array([[tok]])
        return drafts, draft_probs, mtp_past

    def _generate_speculative(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_id: Optional[int],
        streamer,
        do_sample: bool,
        repetition_penalty: float,
        num_speculative_tokens: Optional[int] = None,
    ) -> mx.array:
        """MTP speculative decoding: draft with the single Qwen MTP layer,
        verify with the main model in parallel, accept the longest prefix.

        验证后直接用 vlogits / hidden 作为下一轮 bonus，不再单独整网前向
        last token（T=1 与 T=8 几乎同价，多一次主干等于白做一轮）。

        Greedy mode reproduces the main model's greedy output exactly;
        sampling mode uses standard rejection sampling against draft probs.

        注意：greedy + repetition_penalty!=1 且候选 logits 非常接近（~bf16
        噪声量级）时，批量验证与顺序解码的 KV 舍入差异可能翻转 argmax，
        导致投机路径比标准路径差一个 token。这是 bf16 精度问题，非逻辑错误；
        采样模式不受影响（拒绝采样保证分布等价）。
        """
        draft_len = num_speculative_tokens or max(1, int(self.config.mtp_steps))
        seq_len = input_ids.shape[1]
        prompt_ids = input_ids[0].astype(mx.int32)

        # Prefill 主干，取出末层 hidden 供草稿首步与 MTP cache 教师强制。
        out = self(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            mask_has_pad=False,
            output_features=True,
        )
        past = out.past_key_values
        h_trunk = out.features[-1]
        feat_last = h_trunk[:, -1:, :]
        logits_last = out.logits[0, -1]

        # MTP KV cache（vLLM/SGLang NextN）：训练时对整流做因果注意力，
        # 草稿若拿不到上下文会显著偏离。prefill 用末层 hidden + 真实
        # token 嵌入教师强制构建；验证后回滚再按接受位置追平。
        mtp_past = None
        if seq_len > 1:
            e_pref = self.model.embed_tokens(input_ids[:, 1:])
            mtp_t = e_pref.shape[1]
            _, mtp_past = self.mtp_modules[0](
                h_trunk[:, : seq_len - 1, :],
                e_pref,
                use_cache=True,
                position_embeddings=self.model.position_embeddings(
                    0, mtp_t, e_pref.dtype
                ),
            )

        # 激活 KDA 状态轨迹与 block 级 conv 尾部轨迹（rewind 精确回滚
        # 用）：以各 cache 当前位置（prefill 结束）的 state + conv 尾部
        # 为种子快照。
        for c in list(past) + [mtp_past]:
            if not isinstance(c, KVCache):
                continue
            if "kda_state" in c.extras:
                c.extras["kda_trace"] = [
                    (
                        c.offset,
                        c.extras["kda_state"],
                        c.extras.get("q_conv"),
                        c.extras.get("k_conv"),
                        c.extras.get("v_conv"),
                    )
                ]
            for key in ("attn_out", "mlp_out"):
                if key in c.extras:
                    c.extras[key + "_trace"] = [(c.offset, c.extras[key])]
            if "ngram_tail" in c.extras:
                c.extras["ngram_tail_trace"] = [(c.offset, c.extras["ngram_tail"])]
            if "ngram_conv_tail" in c.extras:
                c.extras["ngram_conv_tail_trace"] = [
                    (c.offset, c.extras["ngram_conv_tail"])
                ]

        if streamer:
            streamer.put(input_ids)

        generated: list[int] = []
        stats = {"accepted": 0, "drafted": 0}
        held: Optional[int] = None
        while len(generated) < max_new_tokens:
            seen = mx.concatenate([prompt_ids, mx.array(generated, dtype=mx.int32)])
            # 1. Bonus：拒绝采样的残差校正 token 已抽好则直接用，否则从
            # 验证 logits 采样。不再额外整网前向 last token。
            if held is not None:
                bonus = int(held)
                held = None
            else:
                bonus_logits = _transform_logits_mx(
                    logits_last,
                    seen,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                bonus = int(_sample_from_logits_mx(bonus_logits, do_sample).item())
            # 2. Draft with the MTP chain.
            remaining = max_new_tokens - len(generated)
            iter_draft_len = min(draft_len, max(0, remaining - 1))
            mtp_past_len = _seq_len_of_cache(mtp_past)
            old_cache = _seq_len_of_cache(past[0] if past else None)
            drafts, draft_probs = [], []
            if iter_draft_len > 0:
                drafts, draft_probs, mtp_past = self._mtp_draft(
                    feat_last,
                    bonus,
                    seq_len - 1,
                    iter_draft_len,
                    seen,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                    mtp_past=mtp_past,
                )
            stats["drafted"] += len(drafts)
            # 3. Verify the chain in parallel with the main model.
            # 同时取出各被验证位置的末层 hidden，供验证后追平 MTP cache。
            verify_tokens = mx.array([[bonus] + drafts], dtype=mx.int32)
            past_before_verify = past
            vout = self(
                verify_tokens,
                attention_mask=attention_mask,
                past_key_values=past_before_verify,
                use_cache=True,
                mask_has_pad=False,
                output_features=True,
            )
            past_full = vout.past_key_values
            vlogits = vout.logits[0]  # (1+D, V)

            # 4. Accept the longest matching prefix.
            n_acc = 0
            for i, d in enumerate(drafts):
                seen_i = mx.concatenate(
                    [seen, mx.array([bonus] + drafts[:i], dtype=mx.int32)]
                )
                p_logits = _transform_logits_mx(
                    vlogits[i],
                    seen_i,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                if do_sample:
                    p = _probs_mx(p_logits)
                    q = draft_probs[i]
                    ratio = float(p[d].item()) / max(float(q[d].item()), 1e-12)
                    if float(mx.random.uniform().item()) < min(1.0, ratio):
                        n_acc += 1
                    else:
                        break
                else:
                    if int(mx.argmax(p_logits).item()) == d:
                        n_acc += 1
                    else:
                        break
            stats["accepted"] += n_acc

            accepted_prefix = [bonus] + drafts[:n_acc]
            new_tokens = accepted_prefix
            stop = False
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
                stop = True
            if len(generated) + len(new_tokens) > max_new_tokens:
                new_tokens = new_tokens[: max_new_tokens - len(generated)]
                stop = True

            # 回滚到已接受前缀；下一枚 token 用 vlogits[n_acc] 当 bonus，
            # 不再单独整网前向 last token。
            keep = old_cache + len(new_tokens)
            generated.extend(new_tokens)
            if streamer and new_tokens:
                streamer.put(mx.array([new_tokens]))
            if stop:
                past = _rewind_cache_list(past_full, keep)
                break

            if mtp_past is not None:
                mtp_past = _rewind_cache(mtp_past, mtp_past_len)
                n_e = len(new_tokens)
                feats_in = mx.concatenate(
                    [feat_last, vout.features[-1][:, : n_e - 1, :]], axis=1
                )
                e_in = self.model.embed_tokens(mx.array([new_tokens], dtype=mx.int32))
                _, mtp_past = self.mtp_modules[0](
                    feats_in,
                    e_in,
                    past_key_value=mtp_past,
                    use_cache=True,
                    position_embeddings=self.model.position_embeddings(
                        mtp_past_len, e_in.shape[1], e_in.dtype
                    ),
                )

            past = _rewind_cache_list(past_full, keep)
            seq_len = keep
            last = len(new_tokens) - 1
            feat_last = vout.features[-1][:, last : last + 1, :]
            logits_last = vlogits[last]
            if (
                do_sample
                and drafts
                and n_acc < iter_draft_len
                and len(generated) < max_new_tokens
            ):
                seen_tail = mx.concatenate([seen, mx.array(new_tokens, dtype=mx.int32)])
                p_logits = _transform_logits_mx(
                    vlogits[n_acc],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    True,
                    repetition_penalty,
                )
                p = _probs_mx(p_logits)
                q = draft_probs[n_acc]
                resid = mx.clip(p - q, 0.0, None)
                if float(resid.sum().item()) > 0:
                    held = int(mx.random.categorical(mx.log(resid + 1e-12)).item())
                else:
                    held = int(mx.argmax(p_logits).item())

        generated = generated[:max_new_tokens]
        if streamer:
            streamer.end()
        out_ids = mx.concatenate(
            [input_ids, mx.array([generated], dtype=input_ids.dtype)], axis=1
        )

        # 当达到 max_new_tokens 提前截断 generated 时，让返回的 cache 与
        # out_ids 长度严格一致。
        target_len = input_ids.shape[1] + len(generated)
        if past:
            # 时间维在 axis 1
            past = _rewind_cache_list(past, target_len)
        object.__setattr__(self, "_last_spec_stats", stats)
        object.__setattr__(self, "_last_spec_past", past)
        return out_ids

    def moe_gates(self) -> list:
        """模型内全部 MoE router（含 MTP 块内的），遍历顺序固定。"""
        return list(self._moe_gates)

    def moe_diversity_loss(self) -> Optional[mx.array]:
        """本次前向的 router 输入多样性正则 loss（按调用次数平均，未加权）。"""
        total = None
        calls = 0
        for g in self.moe_gates():
            if g.last_div is None:
                continue
            total = g.last_div if total is None else total + g.last_div
            calls += max(1, g.div_calls)
        if total is None or calls == 0:
            return None
        return total / calls

    def moe_load_stats(self) -> Optional[mx.array]:
        """各 router 本步的每专家 token 计数拼接向量（无 MoE/未收集时 None）。

        仅在 router 开启 collect_stats 且 forward 执行后有效；返回的是图节点，
        可作为 compile 图的额外输出一并物化。
        """
        stats = [
            g.last_load.reshape(-1) for g in self.moe_gates() if g.last_load is not None
        ]
        return mx.concatenate(stats) if stats else None

    def qb_margin_stats(self) -> Optional[mx.array]:
        """各 router 本次前向收集的 margin 样本堆叠 (n_gates, N, E)
        （无 MoE/未收集时 None）。

        margin = 原始 sigmoid 分 − per-token 阈值 alpha（K3 Eq.14：旧
        bias 只经 cutoff 进入更新），是 QB 偏置快照的统计原料。
        仅在 router 开启 collect_stats 且训练 forward 执行后有效；返回的
        是图节点，可作为 loss 的额外输出一并物化。        MTP 块同一 gate 会把各展开步的 B·(T−k) 行拼起来，可能长于主干
        B·T；统一裁到最小 N 再 stack。
        SMELT loop 下被循环的 gate 会把各 visit 的行沿 token 轴拼接
        （r 遍即 r·B·T 行，按 visit 顺序排列）。头部裁剪会只保留 visit 1
        的样本，而 expert_bias 是全部 visit 共享的——bias 只按 visit 1 的
        分布拟合，后续 visit 的路由对 QB 不可见、失衡不被纠正（负载统计
        又是跨 visit 合并的，表现为循环层 max_load_k 持续漂移上升）。
        行数 >= 2×n_min 时改为等距抽稀，让各 visit 等比例进入分位数统计；
        余量不足 2 倍的（如 MTP 的 B·(T−1) vs B·T）保持头部裁剪。
        """
        stats = [g.last_margins for g in self.moe_gates() if g.last_margins is not None]
        if not stats:
            return None
        n_min = min(s.shape[0] for s in stats)
        mixed = []
        for s in stats:
            stride = s.shape[0] // n_min
            if stride > 1:
                s = s[::stride]
            mixed.append(s[:n_min])
        stats = mixed
        return mx.stack(stats) if len(stats) > 1 else stats[0][None]

    def moe_bias_stack(self) -> mx.array:
        """各 router 的 expert_bias 堆成 (n_gates, E)，供 compile 图当运行时入参。

        expert_bias 是 freeze buffer，不在 trainable_parameters 里；mx.compile
        会把闭包里读到的数组收成常量，QB 覆写 module 也进不了已编译前向。
        把它和 params 一起显式传入，下一步 bias 才会作用在路由上。
        """
        gates = self.moe_gates()
        if not gates:
            return mx.zeros((0,), dtype=mx.float32)
        return mx.stack([g.expert_bias.astype(mx.float32) for g in gates])

    def apply_moe_biases(self, biases: mx.array) -> None:
        """把 moe_bias_stack() 的堆叠写回各 gate（stop-gradient，不进优化器）。"""
        gates = self.moe_gates()
        if not gates or biases is None or biases.size == 0:
            return
        if biases.ndim == 1:
            biases = biases[None]
        for i, g in enumerate(gates):
            if i >= biases.shape[0]:
                break
            g.expert_bias = mx.stop_gradient(biases[i].astype(mx.float32))

    def update_moe_biases(self, margin_stats: mx.array):
        """QB（Quantile Balancing）路由偏置快照：每优化器步调用一次。

        对本步全部 token 的 margin（原始 sigmoid 分 − per-token 阈值 alpha）
        按专家取 (1−K/E) 上分位数 beta_e，下一步 bias = −beta，零均值化后
        覆写 expert_bias（frozen buffer，stop-gradient，不进优化器、不接收
        梯度）。K3 Eq.14：旧 bias 只经 cutoff alpha 进入更新，margin 不含
        expert_bias。直觉：−beta 把各专家"能进入 top-K 的上分位点"拉平到
        同一水平，长期使每专家期望负载 ≈ K/E；零均值投影消掉 top-k 不变
        的共模 gauge 自由度。替换原 V3 风格比例-截断逐步更新。
        """
        gates = self.moe_gates()
        if not gates or margin_stats is None or margin_stats.size == 0:
            return
        if margin_stats.ndim == 2:
            margin_stats = margin_stats[None]
        n = min(len(gates), margin_stats.shape[0])
        gates = gates[:n]
        stats = margin_stats[:n]
        qs = [1.0 - g.top_k / g.n_routed for g in gates]
        if all(q == qs[0] for q in qs) and stats.shape[1] >= 2:
            # 批量路径：各 gate 的 q 相同（同构 MoE 的常态）时，(G,N,E) 一次
            # partition 替代逐 gate。零均值化仍逐 gate 进行：(G,E) 批量 mean
            # 与 (E,) 单行 mean 的归约顺序不同，会有 ~1e-8 尾差，为与旧实现
            # 逐位一致不在此处批量。
            beta = _col_quantile(stats, qs[0], axis=1)
            for i, g in enumerate(gates):
                b = -beta[i]
                b = b - mx.mean(b, axis=-1, keepdims=True)
                g.expert_bias = mx.stop_gradient(b.astype(mx.float32))
            return
        for i, g in enumerate(gates):
            q = qs[i]
            beta = _col_quantile(stats[i], q)
            b = -beta
            b = b - mx.mean(b, axis=-1, keepdims=True)
            g.expert_bias = mx.stop_gradient(b.astype(mx.float32))

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def ngram_lookup_parameters(self) -> int:
        """n-gram 查找表参数量（不含 1-D gate）。"""
        from mlx.utils import tree_flatten

        n = 0
        for path, v in tree_flatten(self.parameters()):
            if path.endswith("ngram.table") or ".ngram.table" in path:
                n += v.size
        return n

    def num_active_parameters(self) -> int:
        """每 token 实际用到的参数量（Marin / DeepSeek 口径）。

        路由专家栈 ``*.experts.*`` 第一维是 E，按 top-k / E 折算；
        共享专家、latent 投影、注意力、embedding、router 全部计入。
        n-gram 查找表是稀疏 gather（每 token 两行），与未选中专家一样
        不计入；Qwen 也把 n-gram 从表内总参数里单列。
        """
        from mlx.utils import tree_flatten

        k = int(getattr(self.config, "num_experts_per_tok", 0) or 0)
        n = 0
        for path, v in tree_flatten(self.parameters()):
            if path.endswith("ngram.table") or ".ngram.table" in path:
                continue
            if ".experts." in path and v.ndim >= 3 and v.shape[0] > 0:
                e = int(v.shape[0])
                n += v.size // e * min(k, e)
            else:
                n += v.size
        return n

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        self.save_weights(os.path.join(path, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str) -> "VibyForCausalLM":
        config = VibyConfig.from_pretrained(path)
        model = cls(config)
        weights = mx.load(os.path.join(path, "model.safetensors"))
        from mlx.utils import tree_flatten

        shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
        # 形状对不上的键（含旧 HRM/Engram 等）自然丢弃
        weights = {
            k: v for k, v in weights.items() if k in shapes and v.shape == shapes[k]
        }
        model.load_weights(list(weights.items()))
        return model
