"""V4.1 的 MoE：sqrt(softplus) 亲和度 + noaux_tc 偏置 + clamped SwiGLU。

与 V3 的差别（V4 tech report §2.1）：
- 亲和度用 sqrt(softplus(·))，不再是 sigmoid；
- 去掉 n_group / topk_group 分组约束；
- 保留 aux-loss-free 的 e_score_correction_bias：偏置只参与 top-k 选择，
  路由权重来自未加偏置的原始分数；
- 路由专家用 clamp 过的 SwiGLU（gate 上界 clamp，up 双侧 clamp），
  1 个共享专家（普通 SwiGLU）每 token 必走。
"""

import mlx.core as mx
from mlx import nn

from .init import trunc_normal


def softplus(x: mx.array) -> mx.array:
    """数值稳定的 softplus：log(1+e^x) = logaddexp(x, 0)。"""
    return mx.logaddexp(x, mx.zeros_like(x))


def expert_act(gate: mx.array, up: mx.array, swiglu_limit: float) -> mx.array:
    """clamped SwiGLU 前半：silu(clamp(gate, max=L)) * clamp(up, ±L)。"""
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    return (gate * mx.sigmoid(gate)) * up  # silu（mlx.core 无 silu，手写避免额外 import）


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
        self.weight = trunc_normal((n_routed, config.dim), 0.5 / config.dim**0.5)
        # noaux_tc 偏置：随 checkpoint 保存，但 freeze 掉（不进梯度、不进优化器），
        # 由训练循环按负载 sign 规则覆写。
        self.bias = mx.zeros((n_routed,), dtype=mx.float32)
        self.freeze(recurse=False, keys=["bias"])
        self._last_load = mx.zeros((n_routed,), dtype=mx.float32)

    def scores(self, x: mx.array) -> mx.array:
        """x: [M, D]（任意 dtype）→ [M, E] fp32 亲和度分数。

        矩阵乘留在计算 dtype（bf16 的 GEMM 在 MLX 上按 fp32 累加，权重本来
        就是 bf16，结果与"两边都升 fp32 再乘"一致），只在 [M,E] 的小结果上
        进 fp32——原来把 [M,D] 的激活和 [E,D] 的路由矩阵都升 fp32，等于让
        每层的路由 GEMM 跑在 fp32 上，还多两次 [M,D] 量级的转换。
        """
        s = (x @ self.weight.T).astype(mx.float32) / self.gate_temp
        if self.score_func == "softmax":
            return mx.softmax(s, axis=-1)
        if self.score_func == "sigmoid":
            return mx.sigmoid(s)
        return mx.sqrt(softplus(s))

    def __call__(self, x: mx.array):
        """返回 (weights [M,k] fp32, indices [M,k] int32, dense_scores [M,E] fp32)。"""
        scores = self.scores(x)
        sel = scores + self.bias
        k = self.top_k
        # argpartition 取前 k（顺序无关；加权求和与参考实现的 topk 等价）。
        # 索引必须 stop_gradient：MLX 不允许对 gather/scatter 的索引求 VJP。
        idx = mx.argpartition(-mx.stop_gradient(sel), kth=k - 1, axis=-1)[:, :k].astype(mx.int32)
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
            combine_enabled_for, combine_routes, down_project_routes,
            enabled_for, gather_gate_up, route_inverse, route_metadata,
        )

        if enabled_for(x, self.n_routed, 2 * self.moe_in):
            metadata = route_metadata(order, exps_s, self.n_routed)
            h = gather_gate_up(
                x.reshape(M, D), self.experts.gate_up_w.astype(x.dtype),
                order, exps_s, metadata, K,
            )
            gate, up = mx.split(h, 2, axis=-1)
            act = expert_act(gate, up, self.swiglu_limit)
            return down_project_routes(
                act, self.experts.down_w.astype(x.dtype), w.reshape(G),
                order, metadata, K,
            ).reshape(B, T, D)
        tok_s = (order // K).astype(mx.int32)
        xs = x.reshape(M, D)[tok_s]
        gu_t = self.experts.gate_up_w.swapaxes(-1, -2)
        dw_t = self.experts.down_w.swapaxes(-1, -2)
        if gu_t.dtype != x.dtype:
            gu_t, dw_t = gu_t.astype(x.dtype), dw_t.astype(x.dtype)
        h = mx.gather_mm(xs[:, None, :], gu_t, rhs_indices=exps_s, sorted_indices=True)
        gate, up = mx.split(h, 2, axis=-1)
        act = expert_act(gate, up, self.swiglu_limit)
        y = mx.gather_mm(act, dw_t, rhs_indices=exps_s, sorted_indices=True)[:, 0, :]
        if combine_enabled_for(x, K):
            # ``y`` is sorted by expert, while route weights are stored in the
            # original token-major flattening.  The fused combine handles the
            # weighting and K-way token reduction in one pass.  Its inverse is
            # the only metadata needed for this native gather_mm path.
            inverse = route_inverse(order)
            out = combine_routes(y, w.reshape(G), order, inverse, K)
        else:
            w_s = w.reshape(G)[order].astype(x.dtype)
            out = mx.zeros((M, D), dtype=x.dtype).at[tok_s].add(y * w_s[:, None])
        return out.reshape(B, T, D)

    def seq_aux_loss(self, scores: mx.array, idx: mx.array, B: int, T: int) -> mx.array:
        """序列级负载均衡损失（报告 §4.2.2：权重 1e-4 的小规模序列级均衡损失）。

        aux = E · mean_b Σ_e f_{b,e}·P_{b,e}：f 是序列 b 内分到专家 e 的 (token,choice)
        占比（stop_gradient，当负载用），P 是该序列对专家 e 的平均路由概率（带梯度）。
        逐序列而不是整批，正是为了压住单条序列内部的极端不均衡。
        """
        E, k = self.n_routed, self.top_k
        M = B * T
        P = mx.mean(scores.reshape(B, T, E), axis=1)  # [B,E]
        flat = mx.zeros((M, E), dtype=mx.float32)
        rows = mx.arange(M)
        for j in range(k):
            flat = flat.at[rows, idx[:, j]].add(1.0)
        f = mx.mean(mx.stop_gradient(flat).reshape(B, T, E), axis=1) / k  # [B,E]
        return mx.mean(mx.sum(f * P, axis=-1)) * E

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        w, idx, scores = self.router(x.reshape(B * T, D))
        if self.training and self.aux_weight > 0:
            self._last_aux = self.seq_aux_loss(scores, idx, B, T)
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
    return bias - rate * mx.sign(err)
