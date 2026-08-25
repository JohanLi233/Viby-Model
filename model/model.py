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
from .kernels.ce import cross_entropy
from .moe import MoEGate, _col_quantile
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
    # EAGLE-3 多层特征：mtp_feature_layers 指定的低/中/高层在 AttnRes 合并后、
    # final_norm 前的 hidden，各 (B, T, d)，按层序排列；仅在
    # output_features=True 或训练（带 labels 且 mtp_depth>0）时返回。
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
        # 顺序主干：num_hidden_layers 层单栈，残差流为 AttnRes（栈内维护）
        self.stack = VibyStack(config, config.num_hidden_layers)
        # EAGLE-3 多层特征抽取层（1-indexed 配置 → 0-indexed 层下标）
        self._mtp_feature_idx = (
            tuple(i - 1 for i in config.mtp_feature_layers)
            if config.mtp_feature_layers
            else None
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
        **kwargs,
    ) -> tuple[mx.array, list, list]:
        batch_size, seq_length = input_ids.shape
        n_layers = self.config.num_hidden_layers
        if past_key_values is None:
            past_key_values = [None] * n_layers
        elif len(past_key_values) != n_layers:
            raise ValueError(
                f"past_key_values 层数 {len(past_key_values)} 与层数 {n_layers} 不一致"
            )
        first_cache = past_key_values[0]
        start_pos = _seq_len_of_cache(first_cache)
        if start_pos + seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"输入长度 {start_pos + seq_length} 超过模型最大上下文长度 "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.dropout(self.embed_norm(self.embed_tokens(input_ids)))

        # 每微批只 sync/构造一次 causal+pad 融合 mask，所有层共享：
        # 有 padding 时普通 Attention 也能走 flash，而不是逐层退化到 O(T²)。
        # mask_has_pad 可由调用方在 eager 侧预先算好传入（mx.compile 图内
        # 不允许 .item() host sync）；未传入时按原逻辑现场判断。
        mask_is_full = True
        causal_bias = None
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

        return self.stack(
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=seg_for_conv,
            feature_layers=(self._mtp_feature_idx if output_features else None),
        )


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
    def __init__(self, config: Optional[VibyConfig] = None):
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
        self.mtp_modules = (
            [MTPModule(config) for _ in range(config.mtp_depth)]
            if config.mtp_depth > 0
            else []
        )
        # module 树建完后缓存 gate 列表；每步前向避免重复 isinstance 遍历。
        self._moe_gates = [m for m in self.modules() if isinstance(m, MoEGate)]
        apply_trunc_normal_init(self, config.hidden_size)

    def _lm_logits(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states @ self.model.embed_tokens.weight.T

    def _mtp_loss(
        self,
        features: list,
        input_ids: mx.array,
        labels: mx.array,
        loss_mask: Optional[mx.array],
        attention_mask: Optional[mx.array],
        mask_has_pad: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
    ) -> tuple:
        """EAGLE-3/K3 MTP loss：depth 1 消费主干多层特征，depth>1 链式。

        Depth 1 在流位置 j 消费 features 各层在位置 j 的 hidden（EAGLE-3
        多层特征）与 Emb(token j+1)，预测 labels[:, 1:]（即 token j+2，
        labels 已是 next-token 移位）；depth k>1 保持 V3 式链式：消费
        上一深度输出 h_{k-1}[j] 与 Emb(token j+k)，预测 labels[:, k:]。

        返回 (CE 均值, lse 均值)；lse 均值供 logit z-loss（各 head 与主 LM
        同 z_loss_weight）使用。

        --doc_mask 时 MTP 的自注意力同样按目标 token 的文档边界掩码：
        否则 MTP 块会在 packed sequence 内跨文档互相 attend，辅助 loss
        把跨文档泄漏的梯度传回主干（主 loss 已屏蔽，辅助路径却泄漏）。
        """
        seq_len = input_ids.shape[1]
        token_emb = self.model.embed_tokens(input_ids)
        h_prev = None
        terms = []
        z_terms = []
        for k, module in enumerate(self.mtp_modules, start=1):
            if seq_len <= k:
                break
            sub = seq_len - k
            dtype = token_emb.dtype
            am = attention_mask[:, :sub] if attention_mask is not None else None
            # mask_has_pad 由调用方在 eager 侧算好传入（compile 图内不允许 .item()）
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
                # MTP 序列位置 j 消费位置 j 的特征/h_prev 与
                # token_emb[:, j+k]；按源位置 j 的文档边界掩码，避免
                # MTP 自注意力跨文档。目标侧边界由 loss_mask 屏蔽。
                seg = segment_ids[:, :sub]
                same_doc = seg[:, :, None] == seg[:, None, :]
                causal_tril = mx.tril(mx.ones((sub, sub), dtype=mx.bool_))
                allowed = same_doc & causal_tril[None, :, :]
                seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(dtype)
                causal_bias = (
                    seg_bias if causal_bias is None else causal_bias + seg_bias
                )
                mask_is_full = False
            if k == 1:
                h_in = [f[:, :sub, :] for f in features]
            else:
                h_in = h_prev[:, :sub, :]
            h_k, _ = module(
                h_in,
                token_emb[:, k:, :],
                attention_mask=am,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                segment_ids=seg,
            )
            logits_k = self._lm_logits(h_k)
            ce_k, z_k = cross_entropy(
                logits_k,
                labels[:, k:],
                mask=loss_mask[:, k:] if loss_mask is not None else None,
                return_z=True,
            )
            terms.append(ce_k)
            z_terms.append(z_k)
            h_prev = h_k
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
        **kwargs,
    ) -> CausalLMOutput:
        # MoE 负载统计/辅助 loss 按"次前向"重置（见 MoEGate.__call__）。
        for g in self._moe_gates:
            g.last_div = None
            g.div_calls = 0
            if g.collect_stats:
                g.last_load = None
                g.last_margins = None
        # EAGLE-3 多层特征：训练（MTP loss）与投机解码验证路径需要；
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
            **kwargs,
        )
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self._lm_logits(hidden_states)

        loss = None
        lm_loss = None
        mtp_loss = None
        z_loss = None
        if labels is not None:
            lm_loss, lm_z = cross_entropy(
                logits,
                labels,
                mask=loss_mask,
                return_z=True,
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
        feat_last: list,
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
        """Draft tokens with the chained MTP modules (EAGLE 式自回归).

        feat_last: 目标模型在位置 pos_idx 处的多层特征（3 × (1,1,hidden)），
        只用于第一个草稿步；step≥1 消费草稿自身上一步的 block 输出
        （V3 式链式）与自草稿 token 嵌入（EAGLE 自回归）。
        first_token: the bonus token at position pos_idx + 1.
        draft_len: 每轮草稿的 token 数。超过模块数时循环复用 MTP 模块
        （与 vLLM 的 MTP self-speculation 一致，业界 depth=1 但草稿 3~7 个）。
        mtp_past: 首个 MTP 模块的注意力 KV cache（训练口径的上下文，
        由 _generate_speculative 在 prefill 时教师强制构建、每轮验证后
        回滚追平）；循环复用产生的中间条目随返回一并带出，由调用方截断。
        Returns (draft_token_ids, draft_probs, mtp_past)，len(drafts) ==
        draft_len，全程 MLX，logits 不离开 GPU。
        """
        token = mx.array([[first_token]])
        h = None
        drafts: list[int] = []
        draft_probs: list[mx.array] = []
        n_modules = len(self.mtp_modules)
        # 自复用链中每个草稿都等价于训练时“同一输出位置、更深一层”的 MTP：
        # 始终消费同一流位置 (pos_idx) 出发的表示，只把 token 逐层后移。
        # 模型已无 RoPE，pos_idx 仅作审计参数保留。
        del pos_idx
        for i in range(draft_len):
            module_idx = i % n_modules
            module = self.mtp_modules[module_idx]
            emb = self.model.embed_tokens(token)
            # KV cache 只接首个模块（depth-1 流，唯一被训练的深度）：
            # mtp_depth=1 时循环复用的每步都是 module 0，草稿全程带上下文；
            # depth>1 时更深的复用步保持无 cache（其链式输入本就 OOD）。
            # 复用步追加的中间条目随返回一并带出，由调用方截断。
            own_cache = mtp_past is not None and module_idx == 0
            # step 0：EAGLE-3 多层特征输入；step≥1：草稿自身 hidden 链式输入
            h_in = feat_last if i == 0 else h
            h, present = module(
                h_in,
                emb,
                past_key_value=mtp_past if own_cache else None,
                use_cache=True,
            )
            if own_cache:
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
        """MTP speculative decoding: draft with MTP modules, verify with the
        main model in parallel, accept the longest matching prefix.

        Greedy mode reproduces the main model's greedy output exactly;
        sampling mode uses standard rejection sampling against draft probs.

        注意：greedy + repetition_penalty!=1 且候选 logits 非常接近（~bf16
        噪声量级）时，批量验证与顺序解码的 KV 舍入差异可能翻转 argmax，
        导致投机路径比标准路径差一个 token。这是 bf16 精度问题，非逻辑错误；
        采样模式不受影响（拒绝采样保证分布等价）。
        """
        # 每轮草稿长度：默认等于 MTP 模块数，也可通过 num_speculative_tokens
        # 循环复用模块草稿更多 token（对齐 DeepSeek V4 等主流配置）。
        draft_len = num_speculative_tokens or len(self.mtp_modules)
        seq_len = input_ids.shape[1]
        prompt_ids = input_ids[0].astype(mx.int32)

        # Prefill the main model（同时取出 EAGLE-3 多层特征，供草稿
        # 首步输入与 MTP cache 教师强制构建）。
        out = self(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            mask_has_pad=False,
            output_features=True,
        )
        past = out.past_key_values
        feat_last = [f[:, -1:, :] for f in out.features]
        logits_last = out.logits[0, -1]

        # MTP 草稿层 KV cache（对齐 vLLM/SGLang 的 NextN 推理）：训练时
        # MTP block 对整流做因果注意力，草稿若拿不到上下文会显著偏离
        # （r060 实测 argmax 命中率 45%→34%）。prefill 时用主干多层特征
        # + 真实 token 嵌入教师强制构建；之后每轮验证后回滚草稿
        # 追加、再按被接受位置追平。仅维护首个模块（depth-1 流）；有
        # padding 时不启用（speculative 本就限 batch=1，eval 恒无 pad）。
        mtp_past = None
        if seq_len > 1:
            # 调用方（generate）已保证 attention_mask 全 1 或无 mask
            # （mask_ok），显式全 1 mask 同样构建草稿上下文，不再静默
            # 退化到无 cache 口径。
            feats_pref = [f[:, : seq_len - 1, :] for f in out.features]
            e_pref = self.model.embed_tokens(input_ids[:, 1:])
            _, mtp_past = self.mtp_modules[0](feats_pref, e_pref, use_cache=True)

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

        if streamer:
            streamer.put(input_ids)

        generated: list[int] = []
        stats = {"accepted": 0, "drafted": 0}
        while len(generated) < max_new_tokens:
            seen = mx.concatenate([prompt_ids, mx.array(generated, dtype=mx.int32)])
            # 1. Bonus token from the main model (always accepted).
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
            # 剩余额度/上下文位置有限时收紧草稿数：bonus 必占 1 个，
            # full-accept 的额外 tail 还要再占 1 个位置，避免越过
            # max_position_embeddings（前向里有显式越界检查）。
            remaining = max_new_tokens - len(generated)
            iter_draft_len = min(draft_len, max(0, remaining - 2))
            mtp_past_len = _seq_len_of_cache(mtp_past)
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
            # 同时取出各被验证位置的多层特征，供验证后追平 MTP cache。
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
            seen_tail = mx.concatenate(
                [seen, mx.array(accepted_prefix, dtype=mx.int32)]
            )
            if n_acc == iter_draft_len and remaining > n_acc + 1:
                # All drafts accepted: sample an extra token from the tail.
                # iter_draft_len==0（剩余额度收紧到不草稿）且额度 ≥2 时，
                # vlogits[0] 同样给出 bonus 之后的下一 token，不再白跑一轮。
                tail_logits = _transform_logits_mx(
                    vlogits[iter_draft_len],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                tail = int(_sample_from_logits_mx(tail_logits, do_sample).item())
            elif iter_draft_len == 0:
                # 剩余额度只够 1 个 token：只产出 bonus，不补 tail
                tail = None
            elif do_sample:
                # Reject: resample from the positive residual (p - q)+.
                p_logits = _transform_logits_mx(
                    vlogits[n_acc],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    do_sample,
                    repetition_penalty,
                )
                p = _probs_mx(p_logits)
                q = draft_probs[n_acc]
                resid = mx.clip(p - q, 0.0, None)
                if float(resid.sum().item()) > 0:
                    tail = int(mx.random.categorical(mx.log(resid + 1e-12)).item())
                else:
                    tail = int(mx.argmax(p_logits).item())
            else:
                # greedy：与标准路径一致，施加变换后取 argmax
                tail_logits = _transform_logits_mx(
                    vlogits[n_acc],
                    seen_tail,
                    temperature,
                    top_p,
                    top_k,
                    False,
                    repetition_penalty,
                )
                tail = int(mx.argmax(tail_logits).item())

            new_tokens = accepted_prefix + ([tail] if tail is not None else [])

            # 5. Stop at EOS (keep it, drop anything after).
            stop = False
            if eos_token_id is not None and eos_token_id in new_tokens:
                new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
                stop = True

            # 6. 回滚 cache 到实际接受的前缀，再前向 tail token。
            seq_len += len(new_tokens)
            generated.extend(new_tokens)
            if streamer:
                streamer.put(mx.array([new_tokens]))
            if stop:
                # EOS 截断后不再前向 tail。若 EOS 本身就是未验证的 tail，
                # seq_len 会比 past_full 长 1；rewind 按旧切片语义截到
                # 实际 cache 长度（见 KVCache.rewind）。
                past = _rewind_cache_list(past_full, seq_len)
                break

            # MTP cache：回滚草稿阶段的追加，再教师强制追平本轮新确定的
            # (多层特征, next-token) 对。流位置 = 追加前 cache 长度（与训练
            # 口径一致：条目 t = 主干在位置 t 的多层特征 + token t+1 的
            # 嵌入）。
            if mtp_past is not None:
                mtp_past = _rewind_cache(mtp_past, mtp_past_len)
                n_e = len(new_tokens)
                feats_in = [
                    mx.concatenate(
                        [feat_last[i], vout.features[i][:, : n_e - 1, :]], axis=1
                    )
                    for i in range(len(feat_last))
                ]
                e_in = self.model.embed_tokens(mx.array([new_tokens], dtype=mx.int32))
                _, mtp_past = self.mtp_modules[0](
                    feats_in, e_in, past_key_value=mtp_past, use_cache=True
                )

            keep = seq_len - 1  # old_seq_len + len(new_tokens[:-1])
            past = _rewind_cache_list(past_full, keep)
            tout = self(
                mx.array([[new_tokens[-1]]]),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_features=True,
            )
            past = tout.past_key_values
            feat_last = [f[:, -1:, :] for f in tout.features]
            logits_last = tout.logits[0, -1]

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

        margin = 选择分 − per-token 阈值 alpha，是 QB 偏置快照的统计原料。
        仅在 router 开启 collect_stats 且训练 forward 执行后有效；返回的
        是图节点，可作为 loss 的额外输出一并物化。MTP 块 gate 比主干少
        一个流位置（T−1 vs T），各 gate 的 N 不同，统一裁到最小 N 再
        stack（margin 跨位置近似可交换，裁尾部 B 行无系统偏差）。
        """
        stats = [g.last_margins for g in self.moe_gates() if g.last_margins is not None]
        if not stats:
            return None
        n_min = min(s.shape[0] for s in stats)
        stats = [s[:n_min] for s in stats]
        return mx.stack(stats) if len(stats) > 1 else stats[0][None]

    def update_moe_biases(self, margin_stats: mx.array):
        """QB（Quantile Balancing）路由偏置快照：每优化器步调用一次。

        对本步全部 token 的 margin（选择分 − per-token 阈值 alpha）按专家
        取 (1−K/E) 上分位数 beta_e，下一步 bias = −beta，零均值化后覆写
        expert_bias（frozen buffer，stop-gradient，不进优化器、不接收
        梯度）。直觉：−beta 把各专家"能进入 top-K 的上分位点"拉平到同一
        水平，长期使每专家期望负载 ≈ K/E；零均值投影消掉 top-k 不变的
        共模 gauge 自由度。替换原 V3 风格比例-截断逐步更新。
        """
        gates = self.moe_gates()
        if not gates or margin_stats is None or margin_stats.size == 0:
            return
        if margin_stats.ndim == 2:
            margin_stats = margin_stats[None]
        for i, g in enumerate(gates):
            if i >= margin_stats.shape[0]:
                break
            m = margin_stats[i].astype(mx.float32)  # (N, E)
            q = 1.0 - g.top_k / g.n_routed
            beta = _col_quantile(m, q)
            b = -beta
            b = b - mx.mean(b, axis=-1, keepdims=True)
            g.expert_bias = mx.stop_gradient(b.astype(mx.float32))

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))

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
        # 旧 checkpoint 中已删除子系统（MLA/HRM/Engram/value-res）的权重
        # 按键名/shape 过滤后自然丢弃
        weights = {
            k: v for k, v in weights.items() if k in shapes and v.shape == shapes[k]
        }
        model.load_weights(list(weights.items()))
        return model
