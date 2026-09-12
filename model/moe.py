"""V4.1 MoE with causal QB balancing, sqrt(softplus) and clamped SwiGLU.

与 V3 的差别（V4 tech report §2.1）：
- 亲和度用 sqrt(softplus(·))，不再是 sigmoid；
- 去掉 n_group / topk_group 分组约束；
- 保留 aux-loss-free 的 e_score_correction_bias：偏置只参与 top-k 选择，
  路由权重来自未加偏置的原始分数；
- 路由专家用 clamp 过的 SwiGLU（gate 上界 clamp，up 双侧 clamp），
  1 个共享专家（普通 SwiGLU）每 token 必走。
"""

import os

import mlx.core as mx
from mlx import nn

from .init import trunc_normal
from .kernels import moe_counts, moe_decode, moe_gather

_DECODE_GATHER = os.environ.get("VIBY_MOE_DECODE_GATHER", "1") != "0"
_RECURRENT_MASKED_COUNTS = os.environ.get("VIBY_CED_MASKED_COUNTS", "1") != "0"
_QB_THRESHOLD_REUSE = os.environ.get("VIBY_QB_THRESHOLD_REUSE", "0") != "0"


def softplus(x: mx.array) -> mx.array:
    """数值稳定的 softplus：log(1+e^x) = logaddexp(x, 0)。"""
    return mx.logaddexp(x, mx.zeros_like(x))


def expert_act(gate: mx.array, up: mx.array, swiglu_limit: float) -> mx.array:
    """clamped SwiGLU 前半：silu(clamp(gate, max=L)) * clamp(up, ±L)。"""
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    return (
        gate * mx.sigmoid(gate)
    ) * up  # silu（mlx.core 无 silu，手写避免额外 import）


class Expert(nn.Module):
    """单个 clamped-SwiGLU FFN（共享专家用；路由专家走堆叠权重）。"""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # gate
        self.w2 = nn.Linear(inter_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # up
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(expert_act(self.w1(x), self.w3(x), self.swiglu_limit))


class StackedExperts(nn.Module):
    """路由专家权重堆叠：(E, 2I, D) 与 (E, D, I)。

    gate/up 合并成一张 (E, 2I, D)：一次 GEMM 出 [gate|up] 再切分，
    比两次独立 GEMM 少一半 kernel 数；数学上与两路独立权重等价。
    """

    def __init__(self, n_routed: int, moe_in: int, dim: int):
        super().__init__()
        self.gate_up_w = trunc_normal((n_routed, 2 * moe_in, dim), 0.5 / dim**0.5)
        self.down_w = trunc_normal((n_routed, dim, moe_in), 0.5 / moe_in**0.5)


class MoEGate(nn.Module):
    """路由打分 + top-k 选择。bias 是 frozen 的 e_score_correction_bias。"""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        n_routed, n_act = config.moe_of(layer_idx)
        self.layer_idx = layer_idx
        self.n_routed = n_routed
        self.top_k = n_act
        self.score_func = config.score_func
        self.gate_temp = config.gate_temp
        self.norm_topk_prob = config.norm_topk_prob
        self.route_scale = config.route_scale
        self.router_fp32 = getattr(config, "router_fp32", True)
        self.balance_method = getattr(config, "moe_balance_method", "noaux_tc")
        self.qb_stats_rows = getattr(config, "qb_stats_rows", 8192)
        self.weight = trunc_normal((n_routed, config.dim), 0.5 / config.dim**0.5)
        # 随 checkpoint 保存，但 freeze 掉；训练循环用已完成窗口的
        # QB 目标或负载 sign 更新，不进梯度或优化器。
        self.bias = mx.zeros((n_routed,), dtype=mx.float32)
        self.freeze(recurse=False, keys=["bias"])
        self._last_load = mx.zeros((n_routed,), dtype=mx.float32)
        self._last_qb_margins = mx.zeros((0, n_routed), dtype=mx.float32)

    def scores(self, x: mx.array) -> mx.array:
        """x: [M, D]（任意 dtype）→ [M, E] fp32 亲和度分数。

        默认从 GEMM 开始用 fp32；bf16 GEMM 的输出已经舍入，事后 cast
        无法恢复 top-k 边界附近的分数差。训练 dtype 转换也保留路由权重 fp32。
        """
        if self.router_fp32:
            s = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        else:
            s = (x @ self.weight.astype(x.dtype).T).astype(mx.float32)
        s = s / self.gate_temp
        if self.score_func == "softmax":
            return mx.softmax(s, axis=-1)
        if self.score_func == "sigmoid":
            return mx.sigmoid(s)
        # sqrt(0) 的导数为 Inf；极负 logits 的 softplus 下溢会造成 0*Inf。
        return mx.sqrt(mx.maximum(softplus(s), 1e-20))

    def __call__(self, x: mx.array):
        """返回 (weights [M,k] fp32, indices [M,k] int32, dense_scores [M,E] fp32)。"""
        scores = self.scores(x)
        sel = scores + self.bias
        k = self.top_k
        # argpartition 取前 k（顺序无关；加权求和与参考实现的 topk 等价）。
        # 索引必须 stop_gradient：MLX 不允许对 gather/scatter 的索引求 VJP。
        permutation = mx.argpartition(-mx.stop_gradient(sel), kth=k - 1, axis=-1)
        idx = permutation[:, :k].astype(mx.int32)
        if (
            self.training
            and self.balance_method == "qb"
            and k < self.n_routed
            and scores.shape[0] > 0
        ):
            # Only sample the detached statistics, never the actual routing.
            # The (k+1)-th biased score is the token's QB admission threshold.
            rows = min(scores.shape[0], self.qb_stats_rows)
            sample = ((mx.arange(rows) + 0.5) * (scores.shape[0] / rows)).astype(
                mx.int32
            )
            raw = mx.stop_gradient(scores[sample])
            biased = raw + mx.stop_gradient(self.bias)
            if _QB_THRESHOLD_REUSE:
                if (
                    getattr(mx, "__version__", None) == "0.32.2"
                    and mx.default_device() == mx.gpu
                ):
                    # Version-specific backend fact, NOT argpartition's API
                    # contract: MLX 0.32.2 Metal uses the complete merge sort
                    # for ArgPartition, so its kth tail element is the threshold.
                    # https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/sort.cpp#L317-L339
                    threshold_ids = permutation[sample, k : k + 1].astype(mx.int32)
                    alpha = mx.take_along_axis(biased, threshold_ids, axis=-1)
                else:
                    # Generic partition promises only the selected prefix and
                    # its complement. The largest unselected value gives the
                    # same admission threshold, including ties. Never alter
                    # the original selected prefix or its native ordering.
                    remaining = permutation[sample, k:].astype(mx.int32)
                    alpha = mx.max(
                        mx.take_along_axis(biased, remaining, axis=-1),
                        axis=-1,
                        keepdims=True,
                    )
            else:
                alpha = -mx.partition(-biased, kth=k, axis=-1)[:, k : k + 1]
            self._last_qb_margins = mx.stop_gradient(raw - alpha)
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk_prob and k > 1:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * self.route_scale
        if self.training:
            # 侧信道：每专家 token 计数。必须经 forward 返回值出图，
            # 否则 mx.compile 会把纯副作用剪掉（见 VibyForCausalLM 的 moe_loads）。
            load = mx.zeros((self.n_routed,), dtype=mx.float32)
            self._last_load = load.at[idx.reshape(-1)].add(
                mx.ones((idx.size,), dtype=mx.float32)
            )
        return w, idx, scores


class MoEFeedForward(nn.Module):
    """共享专家 + top-k 路由专家的 MoE FFN。

    两条路由路径（形状只随 (B,T,E,K) 静态确定，可 mx.compile）：
    - M <= _DENSE_MAX_TOKENS：稠密批量广播 matmul（decode / 极小 prefill，
      专家数少时比 gather_mm 的固定开销更划算）；
    - 其余（训练 / 大 prefill）：按专家 argsort 后 mx.gather_mm
      (sorted_indices=True) 的免 padding 分段 GEMM，只算真实 (token,choice) 行。
    """

    _DENSE_MAX_TOKENS = 8

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        n_routed, n_act = config.moe_of(layer_idx)
        self.n_routed = n_routed
        self.top_k = n_act
        self.moe_in = config.moe_inter_dim
        self.swiglu_limit = config.swiglu_limit
        self.router = MoEGate(config, layer_idx)
        self.aux_weight = float(getattr(config, "aux_balance_loss_weight", 0.0) or 0.0)
        self._last_aux = mx.array(0.0)
        self.experts = StackedExperts(n_routed, self.moe_in, config.dim)
        self.shared = Expert(config.dim, self.moe_in, config.swiglu_limit)

    # -- 路由写出 ------------------------------------------------------
    def _dense_forward(self, x: mx.array, idx: mx.array, w: mx.array) -> mx.array:
        B, T, D = x.shape
        M, K = B * T, self.top_k
        if (
            _DECODE_GATHER
            and not self.training
            and mx.default_device() == mx.gpu
            and self.experts.gate_up_w.dtype == x.dtype
            and self.experts.down_w.dtype == x.dtype
        ):
            # Small-batch inference: native gather_mm reads only selected
            # experts; routing, weights and the expert GEMMs are unchanged.
            exps = idx.reshape(M * K).astype(mx.int32)
            tokens = (mx.arange(M * K) // K).astype(mx.int32)
            h = mx.gather_mm(
                x.reshape(M, 1, D),
                self.experts.gate_up_w.swapaxes(-1, -2).astype(x.dtype),
                lhs_indices=tokens,
                rhs_indices=exps,
            )
            gate, up = mx.split(h, 2, axis=-1)
            act = expert_act(gate, up, self.swiglu_limit)
            y = mx.gather_mm(
                act,
                self.experts.down_w.swapaxes(-1, -2).astype(x.dtype),
                rhs_indices=exps,
            )
            from .kernels.decode_metadata import combine_selected_experts

            return combine_selected_experts(
                y, w.reshape(-1), exps, self.n_routed, K
            ).reshape(B, T, D)
        S = (
            mx.zeros((M, self.n_routed), dtype=x.dtype)
            .at[mx.arange(M)[:, None], idx.reshape(M, K)]
            .add(w.reshape(M, K).astype(x.dtype))
        )
        xf = x.reshape(M, D)
        h = xf @ self.experts.gate_up_w.swapaxes(-1, -2)  # (E, M, 2I)
        gate, up = mx.split(h, 2, axis=-1)
        act = expert_act(gate, up, self.swiglu_limit)
        y = act @ self.experts.down_w.swapaxes(-1, -2)  # (E, M, D)
        mixed = (y * S.T[..., None].astype(y.dtype)).sum(axis=0)
        return mixed.reshape(B, T, D)

    def _sparse_forward(self, x: mx.array, idx: mx.array, w: mx.array) -> mx.array:
        B, T, D = x.shape
        K = self.top_k
        M = B * T
        G = M * K
        flat = idx.reshape(G)
        order = mx.argsort(mx.stop_gradient(flat))
        exps_s = flat[order].astype(mx.int32)
        from .kernels.moe_dispatch import (
            combine_enabled_for,
            combine_routes,
            down_project_routes,
            enabled_for,
            gather_gate_up,
            route_inverse,
            route_metadata,
        )

        if enabled_for(x, self.n_routed, 2 * self.moe_in):
            metadata = route_metadata(order, exps_s, self.n_routed)
            h = gather_gate_up(
                x.reshape(M, D),
                self.experts.gate_up_w.astype(x.dtype),
                order,
                exps_s,
                metadata,
                K,
            )
            gate, up = mx.split(h, 2, axis=-1)
            act = expert_act(gate, up, self.swiglu_limit)
            return down_project_routes(
                act,
                self.experts.down_w.astype(x.dtype),
                w.reshape(G),
                order,
                metadata,
                K,
            ).reshape(B, T, D)
        tok_s = (order // K).astype(mx.int32)
        use_gather_vjp = self.training and moe_gather.enabled_for(x, K)
        use_combine = combine_enabled_for(x, K)
        # One permutation serves both input-gradient reduction and combine.
        inverse = route_inverse(order) if use_gather_vjp else None
        if use_gather_vjp:
            xs = moe_gather.gather_routes(x.reshape(M, D), order, inverse, K)
        else:
            xs = x.reshape(M, D)[tok_s]
        gu_t = self.experts.gate_up_w.swapaxes(-1, -2)
        dw_t = self.experts.down_w.swapaxes(-1, -2)
        if gu_t.dtype != x.dtype:
            gu_t, dw_t = gu_t.astype(x.dtype), dw_t.astype(x.dtype)
        h = mx.gather_mm(xs[:, None, :], gu_t, rhs_indices=exps_s, sorted_indices=True)
        gate, up = mx.split(h, 2, axis=-1)
        act = expert_act(gate, up, self.swiglu_limit)
        y = mx.gather_mm(act, dw_t, rhs_indices=exps_s, sorted_indices=True)[:, 0, :]
        if use_combine:
            # ``y`` is sorted by expert, while route weights are stored in the
            # original token-major flattening.  The fused combine handles the
            # weighting and K-way token reduction in one pass.  Its inverse is
            # the only metadata needed for this native gather_mm path.
            if inverse is None:
                inverse = route_inverse(order)
            out = combine_routes(y, w.reshape(G), order, inverse, K)
        else:
            w_s = w.reshape(G)[order].astype(x.dtype)
            out = mx.zeros((M, D), dtype=x.dtype).at[tok_s].add(y * w_s[:, None])
        return out.reshape(B, T, D)

    def seq_aux_loss(
        self, scores: mx.array, idx: mx.array, B: int, T: int, counts=None
    ) -> mx.array:
        """序列级负载均衡损失（报告 §4.2.2：权重 1e-4 的小规模序列级均衡损失）。

        aux = E · mean_b Σ_e f_{b,e}·P_{b,e}：f 是序列 b 内分到专家 e 的 (token,choice)
        占比（stop_gradient，当负载用），P 是该序列对专家 e 的平均路由概率（带梯度）。
        逐序列而不是整批，正是为了压住单条序列内部的极端不均衡。
        """
        E, k = self.n_routed, self.top_k
        M = B * T
        # Normalize per token first: shrinking all affinities must not reduce
        # the balancing loss without changing a single routing decision.
        probs = scores / mx.maximum(mx.sum(scores, axis=-1, keepdims=True), 1e-20)
        P = mx.mean(probs.reshape(B, T, E), axis=1)  # [B,E]
        if counts is None and moe_counts.enabled_for(idx):
            counts = moe_counts.sequence_route_counts(idx, B, T, E)
        if counts is not None:
            # Same sequence-level objective and /T then /K normalization.
            f = (mx.stop_gradient(counts) / T) / k
        else:
            flat = mx.zeros((M, E), dtype=mx.float32)
            rows = mx.arange(M)
            for j in range(k):
                flat = flat.at[rows, idx[:, j]].add(1.0)
            f = mx.mean(mx.stop_gradient(flat).reshape(B, T, E), axis=1) / k
        return mx.mean(mx.sum(f * P, axis=-1)) * E

    def __call__(self, x: mx.array, pad_mask=None) -> mx.array:
        B, T, D = x.shape
        w, idx, scores = self.router(x.reshape(B * T, D))
        if self.training and self.aux_weight > 0:
            counts = None
            if moe_counts.enabled_for(idx):
                counts = moe_counts.sequence_route_counts(idx, B, T, self.n_routed)
                # Supersede the router's lazy global scatter with the exact
                # same occurrence counts. This remains a forward side channel.
                self.router._last_load = mx.sum(counts, axis=0)
            self._last_aux = self.seq_aux_loss(scores, idx, B, T, counts=counts)
        if self.training and pad_mask is not None:
            # Static anchor batches contain unused slots. They execute for shape
            # stability but must contribute neither load nor balancing gradients.
            mask = pad_mask.reshape(B * T).astype(mx.float32)
            if _RECURRENT_MASKED_COUNTS and moe_counts.masked_enabled_for(
                idx, pad_mask
            ):
                loads = moe_counts.masked_sequence_route_counts(
                    idx, pad_mask, B, T, self.n_routed
                )
            else:
                loads = mx.zeros((B, self.n_routed), mx.float32)
                row = mx.repeat(mx.arange(B), T)
                for j in range(self.top_k):
                    loads = loads.at[row, idx[:, j]].add(mask)
            self.router._last_load = mx.stop_gradient(mx.sum(loads, axis=0))
            if self.aux_weight > 0:
                count = mx.sum(pad_mask, axis=1).astype(mx.float32)
                denom = mx.maximum(count, 1)[:, None]
                probs = scores / mx.maximum(
                    mx.sum(scores, axis=-1, keepdims=True), 1e-20
                )
                probability = (
                    mx.sum((probs * mask[:, None]).reshape(B, T, self.n_routed), axis=1)
                    / denom
                )
                freq = mx.stop_gradient(loads) / (denom * self.top_k)
                per_sequence = mx.sum(freq * probability, axis=-1) * self.n_routed
                self._last_aux = mx.sum(per_sequence) / mx.maximum(mx.sum(count > 0), 1)
            if self.router.balance_method == "qb" and self.top_k < self.n_routed:
                rows = min(B * T, self.router.qb_stats_rows)
                sample = ((mx.arange(rows) + 0.5) * (B * T / rows)).astype(mx.int32)
                self.router._last_qb_margins = mx.where(
                    mask[sample, None] > 0, self.router._last_qb_margins, mx.nan
                )
        if (
            moe_decode._ENABLED
            and not self.training
            and B * T <= self._DENSE_MAX_TOKENS
        ):
            decode_weights = (
                self.experts.gate_up_w,
                self.experts.down_w,
                self.shared.w1.weight,
                self.shared.w2.weight,
                self.shared.w3.weight,
            )
            if moe_decode.enabled_for(
                x, self.training, _DECODE_GATHER, self._DENSE_MAX_TOKENS, decode_weights
            ):
                return moe_decode.compiled_decode(
                    x,
                    idx,
                    w,
                    *decode_weights,
                    self.n_routed,
                    self.top_k,
                    self.swiglu_limit,
                    self.shared.swiglu_limit,
                )
        if B * T <= self._DENSE_MAX_TOKENS:
            mixed = self._dense_forward(x, idx, w)
        else:
            mixed = self._sparse_forward(x, idx, w)
        return mixed + self.shared(x)

    # -- 负载均衡（noaux_tc）-------------------------------------------
    def load_stats(self) -> mx.array:
        return self.router._last_load


def update_expert_bias(bias: mx.array, load: mx.array, rate: float) -> mx.array:
    """aux-loss-free 偏置更新：b −= γ·sign(load_frac − 1/E)。

    负载高于均值的专家降偏置、低于均值的升偏置，使选择概率向均匀收敛。
    偏置不参与梯度（与参考实现里 frozen 的 e_score_correction_bias 一致）。
    """
    total = mx.sum(load)
    frac = load / mx.maximum(total, 1.0)
    err = frac - 1.0 / load.shape[0]
    # 偏置加在选择分数上：负载高于均值 → 降偏置才能把负载压回去（符号与
    # 文档/研究笔记 b ← b − u·sign(load − mean) 一致）。原实现写成 +sign(err)
    # 是正反馈，会把热点专家越推越热。
    updated = bias - rate * mx.sign(err)
    # Common offsets do not change top-k and only consume numerical precision.
    updated = updated - mx.mean(updated)
    return mx.stop_gradient(mx.where(total > 0, updated, bias))


def update_quantile_bias(
    bias: mx.array,
    margins: mx.array,
    top_k: int,
    rate: float = 0.5,
    ignore_padding: bool = False,
) -> mx.array:
    """Causal QB update from the completed window's [samples, experts] margins.

    alpha_t = (k+1)-th largest (score_t + old_bias).
    new_bias_e = -Q_(1-k/E)(score_te - alpha_t), then EMA and recenter.
    The column order statistic leaves approximately samples*k/E values above
    the threshold. Joint expert updates are an approximation, not a guarantee
    of exact per-batch balance. No statistics enter the gradient or current step.
    """
    n, experts = margins.shape
    if n == 0 or top_k == experts or rate == 0:
        return mx.stop_gradient(bias)
    if ignore_padding:
        # Recurrent layers have different physical call counts. All-NaN rows
        # explicitly mark batch padding; partially nonfinite rows still reject
        # the update, as on the original QB path.
        padding = mx.all(mx.isnan(margins), axis=-1)
        count = mx.sum(~padding).astype(mx.int32)
        rank = mx.maximum(
            count - mx.maximum((count * top_k + experts - 1) // experts, 1), 0
        )
        columns = mx.sort(
            mx.where(padding[:, None], mx.inf, mx.stop_gradient(margins)), axis=0
        )
        target = -columns[rank]
        target = target - mx.mean(target)
        centered = bias.astype(mx.float32) - mx.mean(bias.astype(mx.float32))
        updated = (1.0 - rate) * centered + rate * target
        valid = (
            (count > 0)
            & mx.all(mx.isfinite(margins) | padding[:, None])
            & mx.all(mx.isfinite(updated))
        )
        return mx.stop_gradient(mx.where(valid, updated - mx.mean(updated), bias))
    target_count = max(1, (n * top_k + experts - 1) // experts)
    rank = n - target_count
    columns = mx.stop_gradient(margins).astype(mx.float32).T
    target = -mx.partition(columns, kth=rank, axis=-1)[:, rank]
    target = target - mx.mean(target)
    centered = bias.astype(mx.float32) - mx.mean(bias.astype(mx.float32))
    updated = (1.0 - rate) * centered + rate * target
    # A nonfinite observation must never poison the frozen checkpoint buffer.
    valid = mx.all(mx.isfinite(margins)) & mx.all(mx.isfinite(updated))
    return mx.stop_gradient(mx.where(valid, updated - mx.mean(updated), bias))
