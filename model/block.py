from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import GQAAttention, MLAAttention, ShortConv
from .cache import KVCache
from .config import VibyConfig
from .ihc import IHCGate, ihc_add_to_stream, ihc_collapse, ihc_expand
from .kda import KDAAttention
from .kernels import attn_res_fused
from .kernels.layer_decode import try_layer_decode
from .moe import FeedForward, MoEFeedForward
from .norms import GatedNorm, RMSNorm


def _attn_res_merge(w: mx.array, vs: list, window: int = 0) -> mx.array:
    """AttnRes（Attention Residuals，arXiv:2603.15031）深度加权合并：
    α_j = softmax_j(w·RMSNorm(v_j))（K3：key 归一化防范数大的层抢走混合），
    h = Σ_j α_j·v_j。
    w 是该 sublayer 的 1-D 伪查询（ndim=1 ⇒ 自然落入优化器 AdamW 标量组，
    不进 Muon）；softmax 在 f32 计算。vs 为 v_0..v_i 的 (B,T,D) 列表，
    合并逐 token 进行，缓存解码（T=1 新token）语义与训练一致。
    window>0 时只混合最近 W 个残差（训练默认 4）：全历史每层重读全部
    v 且 autodiff 要把每份 dv 跨后续 merge 相加，8 层是 O(L²) 带宽。
    走 attn_res_fused 的融合 Metal kernel（失败自动回退 eager）。"""
    if window > 0 and len(vs) > window:
        vs = vs[-window:]
    return attn_res_fused.merge(w, vs)


def _attn_res_read(
    w: mx.array, register, writes: list, window: int = 0, pin: int = 0
):
    """寄存器残差的读侧：softmax 混合 [register(s)] + 常驻前缀 + 最近 W 个写入。

    register 可以是单个 (B,T,D) 或它们的列表（loop 场景 = [span 入口锚点,
    各 visit 出口摘要]）。window 只裁 writes 的非常驻部分，寄存器永远在
    混合里（避免滑动窗口把恒等通路裁掉）。pin>0 时 writes 的前 pin 份
    （loop 场景 = pre-span 写入：embedding + 浅层输出）同样常驻，窗口只
    作用于其后的 span 内写入。尚无写入时：单寄存器直接返回它，多寄存器
    退化为对寄存器自身的 merge。
    """
    regs = list(register) if isinstance(register, (list, tuple)) else [register]
    if not writes:
        if len(regs) == 1:
            return regs[0]
        return _attn_res_merge(w, regs)
    if window > 0 and len(writes) > pin + window:
        vs = regs + writes[:pin] + writes[-window:]
    else:
        vs = regs + writes
    return _attn_res_merge(w, vs, 0)


def _scale_write(v: mx.array, gate: Optional[mx.array]) -> mx.array:
    if gate is None:
        return v
    return v * gate.astype(v.dtype)


class VibyBlock(nn.Module):
    """prenorm transformer block（MLA 或 KDA/GQA + MoE FFN）。

    默认 MLA full softmax，无 ShortConv。use_linear_attn 时为 Kimi Linear
    3:1（KDA local / GQA global）且 attn/MLP 分支出口各一个 ShortConv。
    残差流默认 AttnRes：把每个 sublayer 输出追加进跨层共享的 v 列表
    （v_0 = embedding 输出），再用本 sublayer 的伪查询 w 做
    softmax-over-depth 合并出下一 hidden。attn_res_register 时改为
    加法写回 hidden，AttnRes 只作为下一子层的读。attn_res_read_h 时
    读侧也改成寄存器本身（prenorm(h)），不再混合 [h]+写入。
    ihc 开时改走 identity Hyper-Connections：M 条流跨 sublayer 保持，
    读/写由本块 ihc_attn / ihc_mlp 门控；AttnRes merge/read 不走。
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
        # 前 K 层小 dense stem（默认关）；其后仍是 DeepSeekMoE
        if config.is_dense_ffn(layer_idx):
            self.mlp = FeedForward(
                config, intermediate_size=config.dense_intermediate_size
            )
        else:
            self.mlp = MoEFeedForward(config)
        # AttnRes 伪查询（每个 sublayer 一个，零初始化 ⇒ 初始均匀混合）
        self.attn_res_q_attn = mx.zeros((config.hidden_size,))
        self.attn_res_q_mlp = mx.zeros((config.hidden_size,))
        self.attn_res_window = int(getattr(config, "attn_res_window", 0) or 0)
        self.attn_res_register = bool(getattr(config, "attn_res_register", False))
        self.attn_res_read_h = bool(getattr(config, "attn_res_read_h", False))
        self.ihc_streams = (
            int(getattr(config, "ihc_streams", 4) or 0)
            if bool(getattr(config, "ihc", False))
            else 0
        )
        self.ihc_collapse = str(getattr(config, "ihc_collapse", "mean") or "mean")
        if self.ihc_streams:
            self.ihc_attn = IHCGate(
                config.hidden_size, self.ihc_streams, config.rms_norm_eps
            )
            self.ihc_mlp = IHCGate(
                config.hidden_size, self.ihc_streams, config.rms_norm_eps
            )
        self.ngram_conf_gate = bool(getattr(config, "ngram_conf_gate", False))
        # SMELT loop 残差写入缩放：本层落在 loop 跨度内且 loop 生效时
        # 按 config 取 r**-0.5 / 1/r / 1.0，其余情况 1.0（不乘）。
        self.loop_res_scale = 1.0
        loop_range = config.loop_layer_range()
        if loop_range is not None and loop_range[0] <= layer_idx < loop_range[1]:
            r = int(config.loop_count)
            if config.loop_res_scale == "rsqrt":
                self.loop_res_scale = r ** -0.5
            elif config.loop_res_scale == "r":
                self.loop_res_scale = 1.0 / r

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
        write_gate: Optional[mx.array] = None,
        loop_anchor: Optional[mx.array] = None,
        loop_pin: int = 0,
    ):
        # residuals：跨层共享的 sublayer 输出列表。替换式 AttnRes 以
        # 输入 hidden 为 v_0；寄存器模式只收集写入，空列表起步。
        # loop_anchor（仅 replace 模式、loop 生效时由 VibyStack 传入）：
        # 常驻锚点列表 [span 入口 hidden, 各 visit 出口摘要]，窗口裁不掉；
        # loop_pin 为常驻的 pre-span 写入数（窗口只裁 span 内写入）。
        if residuals is None:
            residuals = [] if self.attn_res_register else [hidden_states]
        # 统一在 block 层建 cache：KDA 层与 GQA 层共用同一 KVCache 对象
        # （KDA 只写 extras/offset，不落 keys/values）。
        if use_cache and past_key_value is None:
            past_key_value = KVCache()
        if self.ihc_streams:
            return self._forward_ihc(
                hidden_states,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                segment_ids=segment_ids,
                position_embeddings=position_embeddings,
                write_gate=write_gate,
            )
        if use_cache and write_gate is None and self.loop_res_scale == 1.0 and loop_anchor is None:
            # 融合 decode kernel 不实现 loop 缩放与锚点读出，回退 eager 路径。
            fused = try_layer_decode(
                self,
                hidden_states,
                residuals,
                past_key_value,
                position_embeddings,
                attention_mask=attention_mask,
                causal_bias=causal_bias,
                mask_is_full=mask_is_full,
                segment_ids=segment_ids,
            )
            if fused is not None:
                return fused
        if self.attn_res_register and not self.attn_res_read_h:
            attn_in = _attn_res_read(
                self.attn_res_q_attn,
                hidden_states,
                residuals,
                self.attn_res_window,
            )
        else:
            attn_in = hidden_states
        v_attn, present_key_value = self.self_attn(
            self.input_layernorm(attn_in),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=segment_ids,
        )
        v_attn = _scale_write(v_attn, write_gate)
        if self.loop_res_scale != 1.0:
            v_attn = v_attn * self.loop_res_scale
        residuals.append(v_attn)
        if self.attn_res_register:
            hidden_states = hidden_states + v_attn
            if self.attn_res_read_h:
                mlp_in = hidden_states
            else:
                mlp_in = _attn_res_read(
                    self.attn_res_q_mlp,
                    hidden_states,
                    residuals,
                    self.attn_res_window,
                )
        else:
            if loop_anchor is not None:
                hidden_states = _attn_res_read(
                    self.attn_res_q_attn,
                    loop_anchor,
                    residuals,
                    self.attn_res_window,
                    pin=loop_pin,
                )
            else:
                hidden_states = _attn_res_merge(
                    self.attn_res_q_attn, residuals, self.attn_res_window
                )
            mlp_in = hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_in))
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
        mlp_output = _scale_write(mlp_output, write_gate)
        if self.loop_res_scale != 1.0:
            mlp_output = mlp_output * self.loop_res_scale
        residuals.append(mlp_output)
        if self.attn_res_register:
            hidden_states = hidden_states + mlp_output
        elif loop_anchor is not None:
            hidden_states = _attn_res_read(
                self.attn_res_q_mlp,
                loop_anchor,
                residuals,
                self.attn_res_window,
                pin=loop_pin,
            )
        else:
            hidden_states = _attn_res_merge(
                self.attn_res_q_mlp, residuals, self.attn_res_window
            )
        return hidden_states, present_key_value

    def _forward_ihc(
        self,
        hidden_states: mx.array,
        past_key_value=None,
        use_cache: bool = False,
        attention_mask=None,
        causal_bias=None,
        mask_is_full=None,
        segment_ids=None,
        position_embeddings=None,
        write_gate=None,
    ):
        """iHC 读写：3D 输入在本块 expand/collapse；4D 则跨层保持流。"""
        own = hidden_states.ndim == 3
        if own:
            hidden_states = ihc_expand(hidden_states, self.ihc_streams)
        attn_in, h_post = self.ihc_attn.mix(hidden_states)
        v_attn, present_key_value = self.self_attn(
            self.input_layernorm(attn_in),
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=segment_ids,
        )
        v_attn = _scale_write(v_attn, write_gate)
        if self.loop_res_scale != 1.0:
            v_attn = v_attn * self.loop_res_scale
        hidden_states = self.ihc_attn.write(hidden_states, v_attn, h_post)
        mlp_in, h_post = self.ihc_mlp.mix(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_in))
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
        mlp_output = _scale_write(mlp_output, write_gate)
        if self.loop_res_scale != 1.0:
            mlp_output = mlp_output * self.loop_res_scale
        hidden_states = self.ihc_mlp.write(hidden_states, mlp_output, h_post)
        if own:
            hidden_states = ihc_collapse(hidden_states, mode=self.ihc_collapse)
        return hidden_states, present_key_value


class VibyStack(nn.Module):
    """顺序 transformer 主干：N 层 + 尾部 RMSNorm，残差流为 AttnRes。

    SMELT loop（config.loop_span>0 且 loop_count>1）时按 exec_order 展开
    执行：居中内部 span 连续跑 loop_count 遍（同一 VibyBlock 权重共享），
    如 L=9 span=4 r=2 → [0,1,2,3,4,5,2,3,4,5,6,7,8]。past_key_values 按
    执行位置索引（每 visit 一个独立 cache，Huginn 式 per-iteration KV），
    各 cache 长度均与 token 数对齐，RoPE/mask/generate 语义不变。
    loop_extrap≠0 时在末次 visit 结束后对 hidden 做 Richardson 外推
    （h += λ(h − h_prev_visit)），只动流不改 residuals；iHC 4D 流同样支持。
    loop_anchor（默认开，仅 replace AttnRes）时 span 区段的 merge 常驻
    [span 入口 hidden] + [各 visit 出口摘要] + 全部 pre-span 写入，
    窗口只裁 span 内写入，防止滑窗把输入侧/跨 visit 历史挤出后 span
    退化为闭环递归（主 loss 平台）。
    """

    def __init__(self, config: VibyConfig, n_layers: int):
        super().__init__()
        self.layers = [VibyBlock(config, layer_idx=i) for i in range(n_layers)]
        # 执行序列表：中间 span 连续重复 loop_count 次；无 loop 时为 0..L-1
        self.exec_order = list(range(n_layers))
        loop_range = config.loop_layer_range()
        if loop_range is not None:
            start, end = loop_range
            self.exec_order = (
                list(range(start))
                + list(range(start, end)) * int(config.loop_count)
                + list(range(end, n_layers))
            )
        self.n_exec = len(self.exec_order)
        # JFB（loop_grad_mode="jfb"）用的执行位置边界：span 入口与末次
        # visit 起点的 exec_pos。无 loop 时为 None（不参与判断）。
        self._loop_range = loop_range
        self.loop_grad_mode = str(getattr(config, "loop_grad_mode", "full") or "full")
        if loop_range is not None:
            start, end = loop_range
            self._jfb_entry_pos = start
            self._jfb_final_pos = start + (int(config.loop_count) - 1) * (end - start)
        else:
            self._jfb_entry_pos = None
            self._jfb_final_pos = None
        # span 每次 visit 结束的 exec_pos 列表：Richardson 外推
        # （loop_extrap≠0，末次 visit 后 h += λ(h − h_prev_visit)）与
        # loop 锚点的 per-visit 摘要都用到。外推只动 hidden 流，不改
        # residuals/已写入项；iHC 的 4D 流同样逐元素适用。
        self.loop_extrap = float(getattr(config, "loop_extrap", 0.0) or 0.0)
        if loop_range is not None:
            start, end = loop_range
            span_len = end - start
            self._visit_ends = [
                start + (k + 1) * span_len - 1 for k in range(int(config.loop_count))
            ]
            self._visit_end_set = frozenset(self._visit_ends)
        else:
            self._visit_ends = None
            self._visit_end_set = None
        # loop 锚点读出（config.loop_anchor，仅 replace 模式）：span 区段
        # （含 visit 1）的每次 merge 常驻 [span 入口 hidden] + [此前各 visit
        # 的出口摘要] + 全部 pre-span 写入（pin=入口时残差列表长度），窗口
        # 只裁 span 内写入——span 第一个子层起 window 就已挤出 embedding；
        # visit 摘要恢复跨 visit 的循环历史访问（实测 dbg_real_anchor2：
        # 只 pin 输入侧不够，被裁的 visit-1 写入是剩余差距）。exec_pos ∈
        # [start, start+r·span) 的调用收到 loop_anchor=[x_entry, ...摘要]、
        # loop_pin=res_entry_len（与 JFB 共用记录）。
        if (
            loop_range is not None
            and bool(getattr(config, "loop_anchor", True))
            and not bool(getattr(config, "attn_res_register", False))
            and not bool(getattr(config, "ihc", False))
        ):
            start, end = loop_range
            span_len = end - start
            self._anchor_lo = start
            self._anchor_hi = start + int(config.loop_count) * span_len
        else:
            self._anchor_lo = None
            self._anchor_hi = None
        self.final_norm = GatedNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_res_register = bool(getattr(config, "attn_res_register", False))
        self.ihc_streams = (
            int(getattr(config, "ihc_streams", 4) or 0)
            if bool(getattr(config, "ihc", False))
            else 0
        )
        self.ihc_typed = bool(getattr(config, "ihc_typed", False))
        self.ihc_collapse = str(getattr(config, "ihc_collapse", "mean") or "mean")
        self.ihc_ngram_stream = int(getattr(config, "ihc_ngram_stream", 1) or 0)

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
        ngram_io: Optional[dict] = None,
        ngram_inject: Optional[int] = None,
        position_embeddings=None,
        write_gate: Optional[mx.array] = None,
    ) -> tuple[mx.array, list, list]:
        # feature_layers：0-indexed 层，命中时收集该层 AttnRes 合并后、
        # final_norm 前的 hidden（Qwen MTP 只抽末层）。
        presents = []
        feat_caps = {}
        if self.ihc_streams:
            hidden_states = ihc_expand(hidden_states, self.ihc_streams)
            residuals = []
        else:
            residuals = [] if self.attn_res_register else [hidden_states]
        n_layers = len(self.layers)
        inject = (
            ngram_inject
            if ngram_io is not None
            and ngram_inject is not None
            and 0 <= ngram_inject < n_layers
            else None
        )
        if self.ihc_streams and self.ihc_typed:
            if ngram_io is not None:
                # 用流 0 的 hidden 作 query 门控，结果写进专用流
                module = ngram_io["module"]
                Y, tail = module.fuse(
                    hidden_states[:, :, 0],
                    ngram_io["e"],
                    ngram_io.get("conv_state"),
                    segment_ids,
                )
                ngram_io["conv_tail"] = tail
                hidden_states = ihc_add_to_stream(
                    hidden_states, Y, self.ihc_ngram_stream
                )
            inject = None
        visited = set()
        # JFB / 单步梯度（0th-order IFT，HRM/TRM；Attractor Models
        # arXiv:2605.12466 在 LLM 规模验证）：前 r−1 次 visit 的图在末次
        # visit 前被整体断梯度，反向只走 1 次 visit + x_entry 恒等通路。
        # 前向数值与 full 逐位一致（x_entry − sg(x_entry) 值恰为 0），
        # 仅梯度估计不同。eval 也无条件应用，无需按 training 分支。
        jfb = self.loop_grad_mode == "jfb" and self._jfb_entry_pos is not None
        anchor_on = self._anchor_lo is not None
        x_entry = None
        res_entry_len = 0
        # Richardson 外推：记录每次 visit 结束的 hidden，末次后
        # h += λ(h − h_prev)。与 JFB 组合时 h_prev 保持其图不特判。
        extrap_on = self._visit_ends is not None and self.loop_extrap != 0.0
        prev_visit_hidden = None
        # per-visit 出口摘要（loop 锚点用）：visit k 的层可读 visit 1..k−1
        # 的出口 hidden，跨 visit 的循环历史不被滑窗裁掉。
        visit_summaries = []
        for exec_pos, layer_idx in enumerate(self.exec_order):
            layer = self.layers[layer_idx]
            if (jfb or anchor_on) and exec_pos == self._jfb_entry_pos:
                # span 入口：记录 hidden（保留图）与残差列表长度。x_entry
                # 同时服务 JFB 直通与 loop 锚点读出（两者可独立开关）。
                x_entry = hidden_states
                res_entry_len = len(residuals)
            if jfb and exec_pos == self._jfb_final_pos:
                # 末次 visit 前：值不变的梯度直通（pre-span 层经恒等路
                # ∂f/∂x 拿梯度——replace 模式下 merge 链是唯一通路，缺了
                # 它 embedding/layer0 梯度全零）；前序 visit 的 span 写入
                # 原地断梯度（replace 模式末次 visit 的 merge 会读这份
                # 列表）。iHC 无共享 residuals，直通本身已覆盖。
                # delta 必须显式分组先算（元素级恰为 0），否则 (h+x)−x 的
                # 加法顺序会引入 2^-20 量级舍入，破坏与 full 的逐位一致。
                delta = x_entry - mx.stop_gradient(x_entry)
                hidden_states = mx.stop_gradient(hidden_states) + delta
                for j in range(res_entry_len, len(residuals)):
                    residuals[j] = mx.stop_gradient(residuals[j])
            if inject is not None and layer_idx == inject and layer_idx not in visited:
                # Engram：在注入层用当时的 hidden 作 query 门控检索向量；
                # loop 下只在注入层的第一次 visit 触发
                module = ngram_io["module"]
                Y, tail = module.fuse(
                    hidden_states,
                    ngram_io["e"],
                    ngram_io.get("conv_state"),
                    segment_ids,
                )
                ngram_io["conv_tail"] = tail
                hidden_states = hidden_states + Y
                if hidden_states.ndim == 3 and residuals and not self.attn_res_register:
                    residuals[-1] = residuals[-1] + Y
            visited.add(layer_idx)
            pv = past_key_values[exec_pos] if past_key_values is not None else None
            in_anchor_span = (
                anchor_on and self._anchor_lo <= exec_pos < self._anchor_hi
            )
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
                write_gate=write_gate,
                loop_anchor=([x_entry] + visit_summaries) if in_anchor_span else None,
                loop_pin=res_entry_len if in_anchor_span else 0,
            )
            presents.append(present)
            if anchor_on and exec_pos in self._visit_end_set:
                # visit 结束：出口 hidden 追加为摘要，后续 visit 常驻可读。
                # JFB 下断梯度（前序 visit 的图必须整体切掉）；值不变。
                visit_summaries.append(
                    mx.stop_gradient(hidden_states) if jfb else hidden_states
                )
            # feature_layers 按物理 layer_idx 捕获（末层不在 loop 跨度内，
            # 只 visit 一次，行为与无 loop 一致）
            if feature_layers is not None and layer_idx in feature_layers:
                feat = hidden_states
                if feat.ndim == 4:
                    feat = ihc_collapse(feat, mode=self.ihc_collapse)
                feat_caps[layer_idx] = feat
            if extrap_on and exec_pos in self._visit_end_set:
                # visit 结束。末次 visit 后做 Richardson 外推
                # z* ≈ h_r + λ(h_r − h_{r−1})；只动 hidden 流，
                # residuals/past 写入不改。cache 解码（T=1）语义相同。
                if prev_visit_hidden is not None and exec_pos == self._visit_ends[-1]:
                    hidden_states = hidden_states + self.loop_extrap * (
                        hidden_states - prev_visit_hidden
                    )
                prev_visit_hidden = hidden_states
        features = (
            [feat_caps[i] for i in feature_layers] if feature_layers is not None else []
        )
        if hidden_states.ndim == 4:
            hidden_states = ihc_collapse(hidden_states, mode=self.ihc_collapse)
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
