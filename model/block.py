from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import GQAAttention, MLAAttention, ShortConv
from .cache import KVCache
from .config import VibyConfig
from .kda import KDAAttention
from .kernels import attn_res_fused
from .moe import MoEFeedForward
from .norms import GatedNorm, RMSNorm


def _attn_res_merge(w: mx.array, vs: list) -> mx.array:
    """AttnRes（Attention Residuals，arXiv:2603.15031）深度加权合并：
    α_j = softmax_j(w·RMSNorm(v_j))（K3：key 归一化防范数大的层抢走混合），
    h = Σ_j α_j·v_j。
    w 是该 sublayer 的 1-D 伪查询（ndim=1 ⇒ 自然落入优化器 AdamW 标量组，
    不进 Muon）；softmax 在 f32 计算。vs 为 v_0..v_i 的 (B,T,D) 列表，
    合并逐 token 进行，缓存解码（T=1 新token）语义与训练一致。
    走 attn_res_fused 的融合 Metal kernel（失败自动回退 eager）。"""
    return attn_res_fused.merge(w, vs)


class VibyBlock(nn.Module):
    """prenorm transformer block（MLA 或 KDA/GQA + MoE FFN）。

    默认 MLA full softmax，无 ShortConv。use_linear_attn 时为 Kimi Linear
    3:1（KDA local / GQA global）且 attn/MLP 分支出口各一个 ShortConv。
    残差流为 AttnRes：把每个 sublayer 输出追加进跨层共享的 v 列表
    （v_0 = embedding 输出），再用本 sublayer 的伪查询 w 做
    softmax-over-depth 合并出下一 hidden。
    """

    def __init__(self, config: VibyConfig, layer_idx: int = 0):
        super().__init__()
        self.use_linear_attn = bool(config.use_linear_attn)
        if self.use_linear_attn:
            # local/global 分层（Kimi Linear 3:1）：global 层 full-causal
            # GQA，其余 local 层 KDA（无 RoPE/滑窗）。
            self.is_global = (
                layer_idx + 1
            ) % 4 == 0 or layer_idx == config.num_hidden_layers - 1
            self.self_attn = (
                GQAAttention(config, layer_idx=layer_idx)
                if self.is_global
                else KDAAttention(config, layer_idx=layer_idx)
            )
            self.mlp_out_conv = ShortConv(config.hidden_size)
        else:
            self.is_global = True
            self.self_attn = MLAAttention(config, layer_idx=layer_idx)
            self.mlp_out_conv = None
        self.input_layernorm = GatedNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GatedNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # 所有层的 FFN 均为 DeepSeekMoE（共享专家 + top-k 路由专家）
        self.mlp = MoEFeedForward(config)
        # AttnRes 伪查询（每个 sublayer 一个，零初始化 ⇒ 初始均匀混合）
        self.attn_res_q_attn = mx.zeros((config.hidden_size,))
        self.attn_res_q_mlp = mx.zeros((config.hidden_size,))

    def __call__(
        self,
        hidden_states: mx.array,
        past_key_value: Optional[tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        residuals: Optional[list] = None,
        segment_ids: Optional[mx.array] = None,
        position_embeddings=None,
    ):
        # residuals：跨层共享的 sublayer 输出列表 v_0..v_i。独立使用的块
        # （MTP）不传时以输入 hidden 为 v_0 自建，块内同样走 AttnRes。
        if residuals is None:
            residuals = [hidden_states]
        # 统一在 block 层建 cache：KDA 层与 GQA 层共用同一 KVCache 对象
        # （KDA 只写 extras/offset，不落 keys/values）。
        if use_cache and past_key_value is None:
            past_key_value = KVCache()
        v_attn, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=segment_ids,
        )
        residuals.append(v_attn)
        hidden_states = _attn_res_merge(self.attn_res_q_attn, residuals)
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        # site 3 卷积仅 linear attn（Kimi K3）路径；MLA 默认关掉。
        if self.mlp_out_conv is not None:
            if isinstance(present_key_value, KVCache):
                mlp_output, st = self.mlp_out_conv.cached_call(
                    mlp_output,
                    present_key_value.extras.get("mlp_out"),
                    trace=present_key_value.extras.get("mlp_out_trace"),
                    trace_base=present_key_value.offset - mlp_output.shape[1],
                )
                present_key_value.extras["mlp_out"] = st
            else:
                mlp_output = self.mlp_out_conv(mlp_output, segment_ids=segment_ids)
        residuals.append(mlp_output)
        hidden_states = _attn_res_merge(self.attn_res_q_mlp, residuals)
        return hidden_states, present_key_value


class VibyStack(nn.Module):
    """顺序 transformer 主干：N 层 + 尾部 RMSNorm，残差流为 AttnRes。"""

    def __init__(self, config: VibyConfig, n_layers: int):
        super().__init__()
        self.layers = [VibyBlock(config, layer_idx=i) for i in range(n_layers)]
        self.final_norm = GatedNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
        feature_layers: Optional[tuple] = None,
        ngram_delta: Optional[mx.array] = None,
        ngram_inject: Optional[int] = None,
        position_embeddings=None,
    ) -> tuple[mx.array, list, list]:
        # feature_layers：0-indexed 层，命中时收集该层 AttnRes 合并后、
        # final_norm 前的 hidden（Qwen MTP 只抽末层）。
        presents = []
        feat_caps = {}
        residuals = [hidden_states]  # v_0 = post-embedding hidden
        n_layers = len(self.layers)
        inject = (
            ngram_inject
            if ngram_delta is not None
            and ngram_inject is not None
            and 0 <= ngram_inject < n_layers
            else None
        )
        for layer_idx, layer in enumerate(self.layers):
            if inject is not None and layer_idx == inject:
                hidden_states = hidden_states + ngram_delta
                residuals[-1] = residuals[-1] + ngram_delta
            pv = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                past_key_value=pv,
                use_cache=use_cache,
                attention_mask=attention_mask,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                residuals=residuals,
                segment_ids=segment_ids,
                position_embeddings=position_embeddings,
            )
            presents.append(present)
            if feature_layers is not None and layer_idx in feature_layers:
                feat_caps[layer_idx] = hidden_states
        features = (
            [feat_caps[i] for i in feature_layers] if feature_layers is not None else []
        )
        hidden_states = self.final_norm(hidden_states)
        return hidden_states, presents, features


class MTPModule(nn.Module):
    """Qwen3.8-Next MTP：单层 full-attn decoder + 共享 lm_head。

    与 Qwen / DeepSeek-V3 NextN 对齐：
        x = eh_proj(concat(enorm(e), hnorm(h)))     # 2d → d
        y = DecoderBlock_GQA(x)                     # full-attn MoE
        y = output_norm(y)                          # shared_head_norm
    h 是主干末层 hidden（final_norm 前）或上一展开步的输出；e 是下一
    token 嵌入。无独立 embedding / unembedding。预训练把本模块
    teacher-forced 展开 mtp_steps 次。
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        d = config.hidden_size
        self.norm_h = RMSNorm(d, eps=config.rms_norm_eps)
        self.norm_e = RMSNorm(d, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)
        # Qwen MTP 是 dense attention block，不是线性注意力。
        self.block = VibyBlock(config, layer_idx=config.num_hidden_layers - 1)
        self.output_norm = RMSNorm(d, eps=config.rms_norm_eps)

    def __call__(
        self,
        h_in,
        token_emb: mx.array,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        segment_ids: Optional[mx.array] = None,
        position_embeddings=None,
    ) -> tuple[mx.array, Optional[tuple]]:
        if isinstance(h_in, (list, tuple)):
            h_in = h_in[-1]
        x = self.proj(
            mx.concatenate([self.norm_e(token_emb), self.norm_h(h_in)], axis=-1)
        )
        out, present = self.block(
            x,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            past_key_value=past_key_value,
            use_cache=use_cache,
            segment_ids=segment_ids,
            position_embeddings=position_embeddings,
        )
        return self.output_norm(out), present
