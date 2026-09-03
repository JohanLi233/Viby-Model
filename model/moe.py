import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .acts import glu_gu, situ_glu, situ_glu_gu
from .config import VibyConfig
from .kernels.moe_decode import _MOE_METAL_TYPE, _build_moe_decode_kernels
from .kernels.swiglu_decode import silu_mul_down_decode
from .norms import RMSNorm, _rms_unit

__all__ = [
    "FeedForward",
    "MoEGate",
    "MoEFeedForward",
    "situ_glu",
    "_col_quantile",
    "_tile_to_hidden",
]


class FeedForward(nn.Module):
    def __init__(
        self,
        config: VibyConfig,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.hidden_act = getattr(config, "hidden_act", "silu")

    def __call__(self, x: mx.array) -> mx.array:
        # gate/up 输入相同，合并为单个 (2I, D) GEMM 再 split：放大 GEMM 形状、
        # 减少 kernel 发射，数学与逐 Linear 严格等价（参数仍分别存储，
        # checkpoint/优化器路径不变）。eval 下按身份缓存拼接结果。
        srcs = (self.gate_proj.weight, self.up_proj.weight)
        gu_w = None
        if not self.training:
            c = self.__dict__.get("_gu_w_cache")
            if c is not None and c[1] is srcs[0] and c[2] is srcs[1]:
                gu_w = c[0]
        if gu_w is None:
            gu_w = mx.concatenate(srcs, axis=0)
            if not self.training:
                object.__setattr__(self, "_gu_w_cache", (gu_w, srcs[0], srcs[1]))
        gu = x @ gu_w.T
        # 推理小批量：glu·mul + down 投影融合 kernel（无梯度），按 hidden_act
        # 选 silu（默认）或 situ 公式。
        if not self.training and self.hidden_act in ("silu", "situ"):
            M = x.size // x.shape[-1]
            if M <= 512:
                r = silu_mul_down_decode(
                    gu.reshape(M, gu.shape[-1]), self.down_proj.weight,
                    self.hidden_act,
                )
                if r is not None:
                    return r.reshape(x.shape)
        return self.down_proj(glu_gu(gu, self.hidden_act))


def _col_quantile(x: mx.array, q: float, axis: int = 0) -> mx.array:
    """沿 ``axis`` 的 q 分位数（线性插值，同 numpy 默认）。

    QB 只要一个分位点。partition(kth=lo) 把第 lo 小放到位置 lo，其后全体
    ≥ 它，故 min(尾部) 即第 lo+1 小——与全量 sort 逐位相同。选中的两行转
    f32 再插值（bf16→f32 单调单射）。lo/hi 是 shape 派生的 python int，
    不触发数据 sync。

    墙钟：MLX 0.32 Metal 把 Partition::eval_gpu 直接转进 gpu_merge_sort
    （源码 "We direct partition to sort for now"），目标形状 (13,24576,384)
    与 sort 同为 ~34ms；两次 partition 则 2×。不要为了 lo/hi 调两次。
    """
    n = x.shape[axis]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    p = mx.partition(x, lo, axis=axis)
    sl = [slice(None)] * x.ndim
    sl[axis] = lo
    v_lo = p[tuple(sl)].astype(mx.float32)
    if hi == lo:
        return v_lo
    sl[axis] = slice(lo + 1, None)
    v_hi = mx.min(p[tuple(sl)], axis=axis).astype(mx.float32)
    return v_lo * (1.0 - frac) + v_hi * frac


def _tile_to_hidden(h: mx.array, dim: int) -> mx.array:
    """把末维 d 无参扩到 dim：下标 i 取 h[..., i % d]。

    用 broadcast 重复整段、余数再 concat，不要 ``concatenate([h]*n)[..., :dim]``
    或 ``take(arange % d)``：前者切片 VJP 会 scatter 进一份 n·d 宽零张量，
    后者的 gather VJP 在 compile 下同样拖垮 gather_mm 反传（见
    research/MLX_PERF.md）。broadcast 的 VJP 是沿重复轴归约。
    """
    d = int(h.shape[-1])
    if d == dim:
        return h
    if d <= 0:
        raise ValueError("tile 输入末维必须为正")
    if dim <= 0:
        raise ValueError("tile 目标维必须为正")
    n, r = divmod(dim, d)
    lead = h.shape[:-1]
    parts = []
    if n:
        tiled = mx.broadcast_to(h[..., None, :], lead + (n, d)).reshape(lead + (n * d,))
        parts.append(tiled)
    if r:
        parts.append(h[..., :r])
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=-1)


class MoEGate(nn.Module):
    """QB（Quantile Balancing）路由的 MoE router：sigmoid 打分 + 分位数
    快照偏置。

    选择分 = sigmoid(x·W) + expert_bias；对选择分取 top-(K+1)，第 K+1 个
    即 per-token 阈值 alpha，前 K 个入选。合并权重取原始 sigmoid 分
    （不含 bias）在命中集上归一化并乘 routed_scaling_factor。
    expert_bias 是 frozen 非梯度 buffer：训练期 forward 记录全部专家的
    margin = 原始 sigmoid 分 − alpha（K3 Eq.14：旧 bias 只经 cutoff
    alpha 进入更新），训练循环每优化器步取各专家 margin 的
    (1−K/E) 上分位数 beta，以 −beta（零均值化、stop-gradient）覆写
    bias（见 VibyForCausalLM.update_moe_biases）。

    collect_stats（python 属性，compile 期常量）开启时，训练 forward
    额外记录每专家 token 计数到 self.last_load、margin 样本到
    self.last_margins（均为图节点，随 loss 一同输出）；eval/生成侧
    保持关闭，零开销。
    """

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.n_routed = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.scaling = float(config.routed_scaling_factor)
        self.norm_logits = bool(getattr(config, "moe_router_logit_norm", False))
        self.logit_temp = float(getattr(config, "moe_router_logit_temp", 1.0) or 1.0)
        self.div_weight = float(
            getattr(config, "moe_diversity_loss_weight", 0.0) or 0.0
        )
        std = config.hidden_size**-0.5
        self.weight = mx.random.uniform(-std, std, (self.n_routed, config.hidden_size))
        # 负载均衡偏置（frozen 非梯度 buffer，由训练循环按 QB 分位数快照覆写）。
        self.expert_bias = mx.zeros((self.n_routed,), dtype=mx.float32)
        self.freeze(recurse=False, keys=["expert_bias"])
        # QB：扩展选择宽度 K+1（第 K+1 个选择分即 per-token 阈值 alpha）。
        # K == E 时退化为 E（阈值 = 全体最小选择分）。
        self._k1 = min(self.top_k + 1, self.n_routed)
        self.collect_stats = False
        self.last_load = None
        self.last_margins = None
        self.last_div = None
        self.div_calls = 0

    def __call__(
        self,
        x: mx.array,
        return_amp: bool = False,
    ) -> tuple:
        # 路由分数在 bf16 下计算（bf16 模型下保证 top-k 选择稳定）
        logits = (x @ self.weight.T).astype(mx.float32)
        if self.norm_logits:
            # 逐 token 标准化 logits：均值为 0、std 为 logit_temp。
            # 让 sigmoid 始终工作在敏感区，bias 不被饱和区吞掉。
            # 融合 RMSNorm kernel 做 centered/rsqrt，数学同原式。
            mean = mx.mean(logits, axis=-1, keepdims=True)
            centered = logits - mean
            logits = _rms_unit(centered, 1e-6) * self.logit_temp
        scores = mx.sigmoid(logits)
        sel = scores + self.expert_bias.astype(mx.float32)
        idx_ext = mx.argpartition(-sel, self._k1 - 1, axis=-1)[..., : self._k1]
        # 路由选择是离散操作、不可微；不断开的话 idx 会经 argpartition 链回
        # router 权重，使 autodiff 向使用 idx 的 gather/scatter 算子请求
        # indices 的 VJP（不支持）
        idx_ext = mx.stop_gradient(idx_ext)
        idx = idx_ext[..., : self.top_k]
        # per-token 阈值 alpha = 第 K+1 个选择分（扩展集中最小的那个；
        # argpartition 不保证有序，但第 _k1 个位置的元素即第 _k1 大）。
        alpha = mx.take_along_axis(sel, idx_ext[..., self._k1 - 1 : self._k1], axis=-1)
        # 合并权重来自 unbiased sigmoid 分（选择用 biased 分）。
        raw = mx.take_along_axis(scores, idx, axis=-1)
        amp = raw.sum(axis=-1, keepdims=True)
        w = raw
        if self.norm_topk_prob and self.top_k > 1:
            w = w / mx.maximum(w.sum(axis=-1, keepdims=True), mx.array(1e-9))
        w = w * self.scaling
        if self.collect_stats and self.training:
            # QB 统计原料：全部 E 个专家的 margin = 原始 sigmoid 分 − alpha
            # （K3 Eq.14：旧 bias 只经 cutoff alpha 进入更新。若误用含
            # bias 的选择分 sel，覆写式更新变成 b ← −Q − b_old，不动点
            # 被砍半且逐步周期-2 振荡）。(M, E) 图节点，跨调用沿 token
            # 轴拼接（本 forward 内多次调用时），跨微批由训练循环拼接。
            # scores/alpha 的打分与阈值保持 f32；只在收集点把样本落成
            # 激活 dtype（bf16 训练下 100MB→50MB/微批，且跨累积窗口常驻），
            # 下游 update_moe_biases 在原生 dtype 上 partition 选点，只把
            # 两行序统计量转 f32 插值；分位数对 bf16 误差不敏感。
            margins = (scores - alpha).reshape(-1, self.n_routed).astype(x.dtype)
            if self.last_margins is None:
                object.__setattr__(self, "last_margins", margins)
            else:
                object.__setattr__(
                    self,
                    "last_margins",
                    mx.concatenate([self.last_margins, margins], axis=0),
                )
        if self.collect_stats:
            counts = (
                mx.zeros((self.n_routed,), dtype=mx.float32)
                .at[idx.reshape(-1)]
                .add(1.0)
            )
            if self.last_load is not None:
                counts = counts + self.last_load
            object.__setattr__(self, "last_load", counts)
        if self.training and self.div_weight > 0.0:
            # router 输入 token 多样性正则：直接惩罚“所有位置收敛到同一
            # 方向”。loss = mean_b log1p(common² / residual²)，健康时接近 0，
            # res/common≈0.01 时约 9.2。
            xf = x.astype(mx.float32)
            mu = mx.mean(xf, axis=1, keepdims=True)
            centered = xf - mu
            common2 = mx.mean(mx.square(mu), axis=-1).reshape(-1)
            residual2 = mx.mean(mx.square(centered), axis=(1, 2))
            div = mx.mean(mx.log1p(common2 / mx.maximum(residual2, mx.array(1e-6))))
            if self.last_div is None:
                object.__setattr__(self, "last_div", div)
            else:
                object.__setattr__(self, "last_div", self.last_div + div)
            self.div_calls += 1
        if return_amp:
            return idx.astype(mx.int32), w.astype(x.dtype), amp.astype(x.dtype)
        return idx.astype(mx.int32), w.astype(x.dtype)


class _StackedExperts(nn.Module):
    """堆叠路由专家权重：(E, out, in) 张量，前向用广播 matmul 批量计算。

    gate/up 投影合并存为单张量 gate_up_w (E, 2*I, D)：前向一次 GEMM 出
    (E, C, 2I) 再 split，比两次独立 GEMM 的 kernel 数少、单 GEMM 更大
    （GPU 利用率高），数学严格等价（同分布独立初始化）。

    独立模块以便优化器按路径名（"*.experts.*"）识别 3D 专家栈：
    muonh 下（默认）逐专家 Newton-Schulz 进 MuonH；`--no_muonh` 或
    VIBY_MUONH_EXPERTS=0 时进 AdamW，避免基类 Muon 把 ndim>2 reshape 成
    (E, out*in) 整体正交化、跨专家耦合。
    """

    def __init__(self, n_routed: int, moe_in: int, dim: int):
        super().__init__()
        std_in = dim**-0.5
        std_moe = moe_in**-0.5
        self.gate_up_w = mx.random.uniform(-std_in, std_in, (n_routed, 2 * moe_in, dim))
        self.down_w = mx.random.uniform(-std_moe, std_moe, (n_routed, dim, moe_in))


class MoEFeedForward(nn.Module):
    """DeepSeekMoE FFN：共享专家（每 token 必走）+ top-k 路由专家。

    三路径实现（按 (token,choice) 对数 G = B*T*K 静态选择）：
    - G <= _KERNEL_MAX_PAIRS 且 E<=64（decode / 极小专家集）：融合 Metal
      kernel。大 E（如 384）的标量 router 慢于 gather_mm，改走 sparse。
    - 推理小 G 且 E>64：sorted gather_mm，只读 top-k，不走稠密全专家。
    - G <= _DENSE_MAX_PAIRS（小 prefill）：稠密批量路径——堆叠
      权重广播 matmul 一次算完全部专家，按路由权重（非命中为 0）加权
      合并。无 host sync、可微、可与 mx.compile 共存。
    - G 更大（训练 / 大 prefill）：sorted gather_mm 路径——按专家 argsort
      后用 mx.gather_mm(sorted_indices=True) 做免 padding 的分组 GEMM，
      每个 (token,choice) 对只算真实行（无桶容量、无 padding 白算、
      无 host sync、形状只随 (B,T,E,K) 变 ⇒ 可被 mx.compile）。
      实测（M=12288/K=8/E=256/DE=384/I=320，mlx 0.32）fwd 8.8ms、
      fwd+bwd 25.5ms，负载倾斜下耗时不变；反向峰值内存有界（~0.5GB/层）。
    """

    _DENSE_MAX_PAIRS = 4096  # G 小于该值走稠密批量路径
    _KERNEL_MAX_PAIRS = 512  # G 小于该值走融合 kernel 路径（decode/极小批量）
    _KERNEL_VERIFIED: set = set()  # 编译验证通过的 (D,I,K,dtype)
    _KERNEL_DISABLED = False  # 任一 shape 编译失败则整体禁用 kernel 路径

    def __init__(self, config: VibyConfig):
        super().__init__()
        self.n_routed = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        moe_in = int(config.moe_intermediate_size or config.intermediate_size)
        self.moe_in = moe_in
        self.hidden_act = getattr(config, "hidden_act", "silu")
        self.router = MoEGate(config)
        # Latent MoE（moe_latent_dim>0）：路由专家的输入/输出维从 hidden
        # 压缩到 d，lat_down/lat_up 两个共享投影包住专家计算；lat_down 后
        # 接 learnable-gain RMSNorm（latent_norm）再 dispatch；加权聚合
        # 输出在 lat_up 前再过一次 RMSNorm（latent_out_norm，K3 Stable
        # LatentMoE §2.3.1 Normalized LatentMoE：压低路由分支对专家
        # 选择/路由权重尺度漂移的敏感度）。router 与共享专家仍在全维
        # hidden 上。gather_mm / kernel 形状参数化，传 d 维输入即可
        # 复用，dispatch 流量同步减半。
        self.latent_dim = int(getattr(config, "moe_latent_dim", 0) or 0)
        expert_dim = self.latent_dim if self.latent_dim > 0 else config.hidden_size
        self.experts = _StackedExperts(self.n_routed, moe_in, expert_dim)
        if self.latent_dim > 0:
            D, d = config.hidden_size, self.latent_dim
            self.lat_down = nn.Linear(D, d, bias=False)
            self.lat_up = nn.Linear(d, D, bias=False)
            # 构造期均匀占位；随后 apply_trunc_normal_init 覆盖为
            # TruncNormal(0, (0.5/√fan_in)²)（默认 Linear 含 glorot 因子，
            # 方差口径不同，不能留）
            self.lat_down.weight = mx.random.uniform(-(D**-0.5), D**-0.5, (d, D))
            self.lat_up.weight = mx.random.uniform(-(d**-0.5), d**-0.5, (D, d))
            # latent RMSNorm 两个（learnable gain，1-D 自动落 AdamW 标量组）：
            # latent_norm 在 lat_down 后（dispatch 前），latent_out_norm 在
            # 聚合后、lat_up 前（K3 Eq.11：y = Σ shared + W↑·RMSNorm(u)）
            self.latent_norm = RMSNorm(d, eps=config.rms_norm_eps)
            self.latent_out_norm = RMSNorm(d, eps=config.rms_norm_eps)
        else:
            self.lat_down = None
            self.lat_up = None
            self.latent_norm = None
            self.latent_out_norm = None
        # 专家写出基扩展：每专家一条 (D,) 对角，乘在 tile(h_k) 上再按
        # 路由权重累加。零初始化 ⇒ 额外写出 ≡ 0。无 latent 时不建。
        self.write_scale = None
        if self.latent_dim > 0 and bool(getattr(config, "moe_write_spread", False)):
            self.write_scale = mx.zeros((self.n_routed, config.hidden_size))
        self.route_scale = bool(
            self.latent_dim > 0 and getattr(config, "moe_route_scale", False)
        )
        n_shared = int(config.n_shared_experts)
        # N_s 个独立共享专家，各宽 moe_in，输出相加。
        self.shared = [
            FeedForward(config, intermediate_size=moe_in) for _ in range(n_shared)
        ]

    def _latent_up(self, out: mx.array) -> mx.array:
        """latent 聚合输出 → latent_out_norm → lat_up 回全维（K3 Eq.11：
        y = Σ shared + W↑·RMSNorm(u)；三条路由路径共用同一口径）。
        moe_route_scale 时在 _combine_routed 里再乘未归一化 top-k 和。"""
        return self.lat_up(self.latent_out_norm(out))

    def _combine_routed(self, mixed: mx.array, spread, amp=None) -> mx.array:
        if self.lat_up is not None:
            mixed = self._latent_up(mixed)
            if amp is not None:
                mixed = mixed * amp.astype(mixed.dtype)
        if spread is not None:
            mixed = mixed + spread
        return mixed

    def _spread_tok(self, y_tok, idx_tok, w_tok, B, T):
        """token-major 写出扩展：y_tok (M,K,d)，对 K 求和。

        不要在 (G, hidden) 上再 scatter-add：训练图里它和 latent 混合的
        (G, d) scatter 抢融合，compile f+b 能把单层从 ~5ms 拖到 ~30ms。
        大形状走融合 Metal kernel（不物化 (M,K,hidden)）；失败回退 eager。
        """
        hid = int(self.write_scale.shape[-1])
        from .kernels.moe_write_spread import write_spread

        fused = write_spread(y_tok, w_tok, self.write_scale, idx_tok)
        if fused is not None:
            return fused.reshape(B, T, hid)
        s = self.write_scale[idx_tok].astype(y_tok.dtype)
        tiled = _tile_to_hidden(y_tok, hid)
        wt = w_tok.astype(y_tok.dtype)[..., None]
        return (tiled * s * wt).sum(axis=1).reshape(B, T, hid)

    def _dense_forward(self, x, idx, w):
        """稠密批量路径：全专家广播 matmul + 路由权重加权合并。"""
        B, T, D = x.shape
        M = B * T
        K = self.top_k
        # 路由权重稠密化：S (M, E)，每行 top-k 个非零（top-k 内专家不重复）
        S = (
            mx.zeros((M, self.n_routed), dtype=x.dtype)
            .at[mx.arange(M)[:, None], idx.reshape(M, K)]
            .add(w.reshape(M, K).astype(x.dtype))
        )
        xf = x.reshape(M, D)
        # 广播 matmul：(M,D) @ (E,D,2I) -> (E,M,2I)，末维即 [gate|up]
        h = glu_gu(xf @ self.experts.gate_up_w.swapaxes(-1, -2), self.hidden_act)
        y = h @ self.experts.down_w.swapaxes(-1, -2)  # (E,M,D)
        mixed = (y * S.T[..., None].astype(y.dtype)).sum(axis=0).reshape(B, T, D)
        spread = None
        if self.write_scale is not None:
            ii = idx.reshape(M, K)
            y_tok = y.swapaxes(0, 1)[mx.arange(M)[:, None], ii]
            spread = self._spread_tok(y_tok, ii, w.reshape(M, K), B, T)
        return mixed, spread

    def _sparse_forward(self, x, idx, w):
        """sorted gather_mm 路径：按专家 argsort → 免 padding 分段 GEMM →
        加权 scatter-add 回 token。

        mx.gather_mm(sorted_indices=True) 在 rhs_indices 有序时走分段
        GEMM（同专家行连续成段），只计算真实 (token,choice) 行——无桶
        容量、无 padding 白算、无 host sync，形状只随 (B,T,E,K) 静态
        确定 ⇒ 可进 mx.compile。反向经 autodiff（gather_mm 的 VJP 同为
        分段 GEMM），峰值内存有界（实测 ~0.5GB/层 @G=98304）。
        实测（M=12288/K=8/E=256/D=384/I=320，mlx 0.32）：fwd 8.8ms、
        fwd+bwd 25.5ms，负载倾斜下耗时不变（旧桶路径倾斜下 padding
        1.7~3× 且反向慢 3~4×）。
        K 个专家输出按 token 在激活 dtype 下 scatter-add 合并（bf16 训练
        即 bf16 累加，不再绕 f32 累加器，省 (M,D) f32 物化与两趟读写；
        K 路 bf16 累加的相对误差 ~1% 级，与稠密路径 x.dtype 加权求和、
        kernel 路径 x.dtype 输出的口径一致）。
        """
        B, T, D = x.shape
        K = self.top_k
        M = B * T
        G = M * K
        flat = idx.reshape(G)  # (G,) pair g ↔ token g//K
        order = mx.argsort(flat)  # 同专家的 pair 连续（sorted_indices 前提）
        exps_s = flat[order].astype(mx.int32)
        tok_s = (order // K).astype(mx.int32)  # 排序后各 pair 的源 token
        w_s = w.reshape(G)[order].astype(x.dtype)
        xf = x.reshape(M, D)
        xs = xf[tok_s]  # (G, D) 按专家连续
        gu_t = self.experts.gate_up_w.swapaxes(-1, -2)  # (E,D,2I)
        dw_t = self.experts.down_w.swapaxes(-1, -2)  # (E,I,D)
        if gu_t.dtype != x.dtype:
            gu_t = gu_t.astype(x.dtype)
            dw_t = dw_t.astype(x.dtype)
        h = mx.gather_mm(
            xs[:, None, :],
            gu_t,
            lhs_indices=None,
            rhs_indices=exps_s,
            sorted_indices=True,
        )  # (G,1,2I)：逐 pair 的 gate/up 投影
        act = glu_gu(h, self.hidden_act)
        y = mx.gather_mm(
            act, dw_t, lhs_indices=None, rhs_indices=exps_s, sorted_indices=True
        )[:, 0, :]  # (G, D)
        yw = y * w_s[:, None]
        # 激活 dtype 累加器 + 不再 astype(f32)：bf16 训练下 scatter-add
        # 直接 bf16 合并（见 docstring 口径说明）
        out = mx.zeros((M, D), dtype=x.dtype).at[tok_s].add(yw)
        mixed = out.reshape(B, T, D)
        spread = None
        if self.write_scale is not None:
            y_tok = y[mx.argsort(order)].reshape(M, K, D)
            spread = self._spread_tok(y_tok, idx.reshape(M, K), w.reshape(M, K), B, T)
        return mixed, spread

    def _kernel_forward(self, x):
        """融合 Metal kernel 路径：3 个 kernel 完成 router 打分+top-k 选择、
        SiTU-GLU 前半、加权合并（无梯度）。router kernel 输出 idx 按分数降序，
        与 MoEGate 的 argpartition 无序输出在加权求和下数学等价。
        Latent MoE 时 router 仍在全维 x 上打分，up/down kernel 在 latent
        维（DE=latent_dim）上跑，前后各多一次共享投影小 GEMM。"""
        B, T, D = x.shape
        M = B * T
        K, inter, E = self.top_k, self.moe_in, self.n_routed
        lat = self.lat_up is not None
        DE = self.latent_dim if lat else D
        g = self.router
        router, up, down = _build_moe_decode_kernels(
            D,
            inter,
            K,
            E,
            x.dtype,
            g.norm_topk_prob,
            g.scaling,
            logit_norm=g.norm_logits,
            logit_temp=g.logit_temp,
            latent_dim=self.latent_dim,
            hidden_act=self.hidden_act,
        )
        xf = x.reshape(M, D)
        idx, w = router(
            inputs=[xf, g.weight, g.expert_bias.astype(mx.float32)],
            output_shapes=[(M, K), (M, K)],
            output_dtypes=[mx.int32, x.dtype],
            grid=(32, M, 1),
            threadgroup=(32, 1, 1),
        )
        xf_e = self.latent_norm(self.lat_down(x)).reshape(M, DE) if lat else xf
        h = up(
            inputs=[xf_e, self.experts.gate_up_w, idx],
            output_shapes=[(M * K * inter,)],
            output_dtypes=[x.dtype],
            grid=(32, K * inter, M),
            threadgroup=(32, 1, 1),
        )[0]
        out = down(
            inputs=[h, self.experts.down_w, w, idx],
            output_shapes=[(M, DE)],
            output_dtypes=[x.dtype],
            grid=(32, DE, M),
            threadgroup=(32, 1, 1),
        )[0]
        if lat:
            out = self._latent_up(out)
        return out.reshape(B, T, D)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        B, T, D = x.shape
        G = B * T * self.top_k
        lat = self.lat_up is not None
        # 融合 kernel 仅推理路径：不可微（无 CustomKernel vjp），训练
        # （model.train()）一律旁路；collect_stats（训练负载统计）同理。
        # 大 E（如 384）上 kernel 的标量 router 点积慢于 gather_mm，只留给
        # 小专家集；decode 小 G 一律 sparse（只读 top-k），不要走稠密全专家。
        # 写出基扩展要逐专家 h_k，融合 down kernel 已在核内加权合并，旁路。
        infer_small = (
            G <= self._KERNEL_MAX_PAIRS
            and not self.training
            and not self.router.collect_stats
        )
        use_kernel = (
            infer_small
            and self.hidden_act in ("silu", "situ")
            and self.n_routed <= 64
            and self.write_scale is None
            and not self.route_scale
        )
        if use_kernel:
            cls = type(self)
            if cls._KERNEL_DISABLED or x.dtype not in _MOE_METAL_TYPE:
                idx, w = self.router(x)
                mixed, spread = self._dense_forward(
                    self.latent_norm(self.lat_down(x)) if lat else x, idx, w
                )
                out = self._combine_routed(mixed, spread)
            else:
                kk = (
                    D,
                    self.latent_dim,
                    self.moe_in,
                    self.top_k,
                    self.n_routed,
                    x.dtype,
                    self.router.norm_logits,
                    self.router.logit_temp,
                    self.hidden_act,
                )
                try:
                    out = self._kernel_forward(x)
                    if kk not in cls._KERNEL_VERIFIED:
                        mx.eval(out)  # 触发 JIT 编译，编译失败走 except 回退
                        cls._KERNEL_VERIFIED.add(kk)
                except Exception:
                    cls._KERNEL_DISABLED = True
                    idx, w = self.router(x)
                    mixed, spread = self._dense_forward(
                        self.latent_norm(self.lat_down(x)) if lat else x, idx, w
                    )
                    out = self._combine_routed(mixed, spread)
        else:
            if self.route_scale:
                idx, w, amp = self.router(x, return_amp=True)
            else:
                idx, w = self.router(x)
                amp = None
            xe = self.latent_norm(self.lat_down(x)) if lat else x
            if infer_small or G > self._DENSE_MAX_PAIRS:
                mixed, spread = self._sparse_forward(xe, idx, w)
            else:
                mixed, spread = self._dense_forward(xe, idx, w)
            out = self._combine_routed(mixed, spread, amp)
        for ff in self.shared:
            out = out + ff(x)
        return out
