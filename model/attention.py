import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import KVCache, _offset_causal_mask
from .config import VibyConfig
from .kernels.attn_fused import flash_sdpa as _flash_sdpa
from .kernels.conv import causal_conv
from .norms import _rms_unit


def _shift_right_tokens(x: mx.array, offset: int) -> mx.array:
    """沿序列维右移（左填 0），用于 n-gram 窗口与因果卷积。"""
    if offset == 0:
        return x
    pad = mx.zeros_like(x[:, :offset, ...])
    return mx.concatenate([pad, x[:, :-offset, ...]], axis=1)


class ShortConv(nn.Module):
    """深度因果 1-D 卷积（depthwise，kernel=4，identity-init，dilation=1）。

    输入 (B, T, C)，沿时间轴逐通道卷积：y[t] = Σ_j w[j]·x[t-j]。
    identity 初始化（w[0]=1，其余 0）⇒ 初始严格恒等。传入 segment_ids (B, T) 时逐 tap 重验文档
    边界（跨段的历史 tap 清零），消除打包序列的跨文档泄漏。
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.dim = dim
        self.kernel_size = int(kernel_size)
        w = mx.zeros((self.kernel_size, dim))
        w = w.at[0].add(1.0)  # MLX ArrayAt 只支持 add/subtract 等
        self.weight = w

    def __call__(self, x: mx.array, segment_ids: Optional[mx.array] = None) -> mx.array:
        w = self.weight.astype(x.dtype)
        return causal_conv(x, w, seg=segment_ids, silu=False)

    def cached_call(
        self,
        x: mx.array,
        state: Optional[mx.array],
        trace: Optional[list] = None,
        trace_base: int = 0,
    ):
        """缓存解码：state 为 (B, K-1, C) 的历史输入尾部（None 视为全零），
        返回 (y, 新 state)。segment 掩码只在完整前向（训练）出现，解码侧
        不打包文档，无需传入。trace 非 None 时把每步处理后的输入尾部按
        (trace_base + t + 1, tail) 逐步追加（与 KDA kda_trace 同口径），
        供投机解码 rewind 精确恢复任意截断点的 conv 历史。"""
        B, T, C = x.shape
        K = self.kernel_size
        if state is None:
            state = mx.zeros((B, K - 1, C), dtype=x.dtype)
        hist = mx.concatenate([state, x], axis=1)  # (B, K-1+T, C)
        w = self.weight.astype(x.dtype)
        y = causal_conv(hist, w, seg=None, silu=False)[:, K - 1 :, :]
        if trace is not None:
            # 步 t 后的尾部 = 截至该步的最后 K-1 个 conv 输入
            for t in range(T):
                trace.append((trace_base + t + 1, hist[:, t + 1 : t + K, :]))
        return y, hist[:, T:, :]


class GQAAttention(nn.Module):
    """GQA global 层（full-causal NoPE，替代原 MLA；local 层由 KDA 承担）。

    仅部署在 global 层（(layer_idx+1)%4==0 或最后一层），KV head 数
    n_kv_heads_global。无 RoPE（NoPE：位置结构由 KDA 层承担）。

    结构：q_proj / kv_proj（2×n_kv_heads×head_dim）/ o_proj；Q/K 逐 head
    无参 RMS norm（全 head_dim）；K 在投影+norm 之后过 ShortConv
    （site 1，KV cache 存 conv 前的 K，拼好全历史后统一卷积，训练/解码
    口径严格一致）；注意力输出先做 XSA（逐 head 扣除与自身 V 平行的
    分量），再经可选逐 head 门（use_attn_gate）与 o_proj；o_proj 之后
    再过 ShortConv（site 2，解码状态存于本层 KVCache.extras）。
    """

    def __init__(self, config: VibyConfig, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads_global
        self.n_rep = self.n_heads // self.n_kv_heads
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(
            config.hidden_size, 2 * self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.qk_norm_eps = config.rms_norm_eps
        self.k_conv = ShortConv(self.n_kv_heads * self.head_dim)
        self.out_conv = ShortConv(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = config.flash_attn
        # 注意力输出门：2·σ(W_g x)，零初始化（初始门 1）。加在 o_proj 之前。
        self.attn_gate = None
        if config.use_attn_gate:
            self.attn_gate = nn.Linear(config.hidden_size, self.n_heads, bias=True)
            self.attn_gate.weight = mx.zeros_like(self.attn_gate.weight)
            self.attn_gate.bias = mx.zeros_like(self.attn_gate.bias)

    def _conv_k(self, k_bthd: mx.array, segment_ids) -> mx.array:
        """site 1：对 (B, T, Hkv, D) 的 pre-conv K 沿时间轴做 ShortConv，
        返回 (B, Hkv, T, D)。"""
        B, T = k_bthd.shape[0], k_bthd.shape[1]
        k = k_bthd.reshape(B, T, self.n_kv_heads * self.head_dim)
        k = self.k_conv(k, segment_ids=segment_ids)
        return k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self,
        x: mx.array,
        past_key_value: Optional[tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[tuple[mx.array, mx.array]]]:
        bsz, seq_len, _ = x.shape
        # 融合单 GEMM：[q|kv] 一次乘完再切分（参数独立存储，同 KDA 口径；
        # eval 下按身份缓存拼接结果，见 KDAAttention）
        nq = self.n_heads * self.head_dim
        srcs = (self.q_proj.weight, self.kv_proj.weight)
        w_qkv = None
        if not self.training:
            c = self.__dict__.get("_w_qkv_cache")
            if c is not None and all(s is t for s, t in zip(c[1], srcs)):
                w_qkv = c[0]
        if w_qkv is None:
            w_qkv = mx.concatenate(srcs, axis=0)
            if not self.training:
                object.__setattr__(self, "_w_qkv_cache", (w_qkv, srcs))
        nkv = self.n_kv_heads * self.head_dim
        # 一次 mx.split 而不是「切片 + 再 split」：切片的 VJP 要 scatter 进一份
        # 全宽零张量，split 的 VJP 是单次 concatenate（见 KDAAttention 同处注释）
        xq, xk, xv = mx.split(x @ w_qkv.T, [nq, nq + nkv], axis=-1)
        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 逐 head 无参 RMS qk-norm（全 head_dim）
        xq = _rms_unit(xq, self.qk_norm_eps)
        xk = _rms_unit(xk, self.qk_norm_eps)

        # KV cache 存 conv 前的 K：site 1 卷积在拼好全历史后施加。
        past_kv = None
        if use_cache:
            if isinstance(past_key_value, KVCache):
                cache = past_key_value
            else:
                cache = KVCache()
                if past_key_value is not None:
                    cache.update(past_key_value[0], past_key_value[1])
            cache.update(xk, xv)
            past_kv = cache
            k_bhtd = self._conv_k(cache[0], None)
            v_bhtd = cache.values[:, :, : cache.offset]
        elif past_key_value is not None:
            if isinstance(past_key_value, KVCache):
                pk = past_key_value[0]
                pv = past_key_value[1]
            else:
                pk, pv = past_key_value
            k_bhtd = self._conv_k(mx.concatenate([pk, xk], axis=1), None)
            v_bhtd = mx.concatenate([pv, xv], axis=1).transpose(0, 2, 1, 3)
        else:
            k_bhtd = self._conv_k(xk, segment_ids)
            v_bhtd = xv.transpose(0, 2, 1, 3)

        # (bsz, heads, seq_len, head_dim)；GQA：K/V 按 n_rep 扩展到全头数
        # （q head h 用 kv head h // n_rep，mx.repeat 的连续重复正好对应）。
        # decode/verify（Tq≠Tk，fused kernel 本就走 mlx SDPA）跳过 repeat，
        # 用 0.32.1 原生 GQA SDPA（已验证与 repeat 逐位一致），省去每步
        # O(T×n_rep) 的 K/V 物化；XSA 只重复 query 对应的末尾 seq_len 个 V。
        xq = xq.transpose(0, 2, 1, 3)
        if mask_is_full is None:
            mask_is_full = attention_mask is None or bool(
                mx.all(attention_mask == 1).item()
            )
        scale = 1.0 / math.sqrt(self.head_dim)
        key_len = k_bhtd.shape[2]
        has_past = key_len > seq_len

        # global 层 full-causal mask 快路径选择
        if seq_len == 1 and mask_is_full:
            local_mask = None
        elif has_past and seq_len > 1 and mask_is_full:
            local_mask = _offset_causal_mask(seq_len, key_len, xq.dtype)
        elif mask_is_full:
            local_mask = "causal"
        else:
            local_mask = causal_bias

        use_flash = (
            self.flash
            and self.dropout == 0.0
            and (mask_is_full or (causal_bias is not None and not has_past))
            and (
                (seq_len > 1 and not has_past)
                # decode：单 query 对全部历史可见，global 层 mask 为 None 即可
                or (seq_len == 1 and mask_is_full)
                # chunk decode / MTP 验证：qlen>1 且已有 cache，构造 offset
                # causal 后仍走 flash（内部对 Tq!=Tk 自动回退 mlx SDPA）
                or (seq_len > 1 and has_past and mask_is_full)
            )
        )
        native_gqa = self.n_rep > 1 and use_flash and (seq_len == 1 or has_past)
        if native_gqa:
            k_att, v_att = k_bhtd, v_bhtd
        elif self.n_rep > 1:
            k_att = mx.repeat(k_bhtd, self.n_rep, axis=1)
            v_att = mx.repeat(v_bhtd, self.n_rep, axis=1)
        else:
            k_att, v_att = k_bhtd, v_bhtd

        if use_flash:
            if native_gqa:
                # 手写 fused kernel 不支持 GQA（含 Tq==Tk 的边界情形，
                # 如首 token decode），native GQA 一律直连 mlx SDPA。
                output = mx.fast.scaled_dot_product_attention(
                    xq, k_att, v_att, scale=scale, mask=local_mask
                )
            else:
                output = _flash_sdpa(xq, k_att, v_att, scale=scale, mask=local_mask)
        else:
            scores = (xq @ mx.swapaxes(k_att, -1, -2)) * scale

            if local_mask is None or isinstance(local_mask, str):
                causal_mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1).astype(
                    scores.dtype
                )
                scores = scores.at[..., -seq_len:].add(causal_mask)
            else:
                scores = scores + local_mask.astype(scores.dtype)

            if attention_mask is not None:
                am = attention_mask
                if am.shape[1] < key_len:
                    pad = mx.ones((am.shape[0], key_len - am.shape[1]), dtype=am.dtype)
                    am = mx.concatenate([am, pad], axis=1)
                elif am.shape[1] > key_len:
                    am = am[:, -key_len:]
                scores = (
                    scores + (1.0 - am[:, None, None, :].astype(scores.dtype)) * -1e9
                )

            attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                xq.dtype
            )
            attn_weights = self.attn_dropout(attn_weights)
            output = attn_weights @ v_att

        # XSA（Exclusive Self-Attention）：逐 head 从输出扣除与自身 V
        # 平行的分量 z = y − (yᵀv/‖v‖²)·v；除法在 f32 做。v 取 query 对应
        # 的末尾 seq_len 个位置（有 cache 时 v_att 覆盖全部历史）。
        # native_gqa 时 v_att 是未扩展的 Hkv 头，只重复这一小段。
        yf = output.astype(mx.float32)
        vf = v_att[:, :, -seq_len:, :]
        if native_gqa:
            vf = mx.repeat(vf, self.n_rep, axis=1)
        vf = vf.astype(mx.float32)
        coef = mx.sum(yf * vf, axis=-1, keepdims=True) / mx.maximum(
            mx.sum(vf * vf, axis=-1, keepdims=True), mx.array(1e-12)
        )
        output = (yf - coef * vf).astype(output.dtype)

        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        if self.attn_gate is not None:
            # bf16 下 sigmoid 误差 ~1e-2（远小于 gate 自身量级与训练
            # 噪声），无需把 (B,T,H) 小张量抬成 f32 再降回来。
            gate = (2.0 * mx.sigmoid(self.attn_gate(x))).astype(output.dtype)
            output = output * mx.repeat(gate, self.head_dim, axis=-1)
        output = self.resid_dropout(self.o_proj(output))
        # site 2：attn 分支输出（o_proj 之后、residual 之前）的 ShortConv；
        # 解码状态挂在本层 KVCache.extras，随 cache 流转/rewind；投机解码
        # 挂了 attn_out_trace 时逐步记录尾部快照，rewind 可精确恢复。
        if isinstance(past_kv, KVCache):
            output, st = self.out_conv.cached_call(
                output,
                past_kv.extras.get("attn_out"),
                trace=past_kv.extras.get("attn_out_trace"),
                trace_base=past_kv.offset - seq_len,
            )
            past_kv.extras["attn_out"] = st
        else:
            output = self.out_conv(output, segment_ids=segment_ids)
        return output, past_kv
