"""MoE：sqrt(softplus) 亲和度 + noaux_tc 偏置 + clamped SwiGLU。

验证的语义（报告 §2.1 Overview / §2.4.3）：
- 亲和度用 sqrt(softplus(·))（V3 的 sigmoid 已废弃）；
- 保持 DeepSeekMoE 的共享专家 + 细粒度路由专家；
- aux-loss-free 负载均衡：e_score_correction_bias 只参与 top-k 选择，路由权重
  来自未加偏置的原始分数；负载高的专家偏置下降；
- 路由专家用 clamp 过的 SwiGLU（gate 上界 clamp、up 双侧 clamp）。

运行：.venv/bin/python -m pytest tests/test_v41_moe.py -q
"""

import numpy as np
import pytest

from _v41_common import cfg_tiny, tiny_model
from model.moe import (
    MoEFeedForward,
    MoEGate,
    expert_act,
    softplus,
    update_expert_bias,
)

import mlx.core as mx


# ------------------------------------------------------------------ 打分


def test_sqrtsoftplus_score_formula():
    """scores = sqrt(logaddexp(x·Wᵀ/temp, 0))，手算对照。"""
    cfg = cfg_tiny(gate_temp=2.0)
    mx.random.seed(60)
    gate = MoEGate(cfg, 0)
    x = mx.random.normal((5, cfg.dim))
    got = gate.scores(x)
    mx.eval(got)
    s = (np.asarray(x) @ np.asarray(gate.weight).T) / 2.0
    want = np.sqrt(np.logaddexp(s, 0.0))
    assert got.shape == (5, cfg.n_routed_experts)
    assert np.allclose(np.asarray(got), want, atol=1e-5), np.abs(
        np.asarray(got) - want
    ).max()
    assert float(mx.min(got)) > 0.0
    # softplus 本身：log(1+e^x)
    v = mx.array([-3.0, 0.0, 2.5])
    mx.eval(softplus(v))
    assert np.allclose(
        np.asarray(softplus(v)), np.logaddexp(np.asarray(v), 0), atol=1e-6
    )


@pytest.mark.parametrize(
    "func,ref",
    [
        ("softmax", lambda s: np.exp(s) / np.exp(s).sum(-1, keepdims=True)),
        ("sigmoid", lambda s: 1 / (1 + np.exp(-s))),
    ],
)
def test_alternative_score_funcs(func, ref):
    cfg = cfg_tiny(score_func=func)
    mx.random.seed(61)
    gate = MoEGate(cfg, 0)
    x = mx.random.normal((4, cfg.dim))
    got = gate.scores(x)
    mx.eval(got)
    s = np.asarray(x) @ np.asarray(gate.weight).T
    assert np.allclose(np.asarray(got), ref(s), atol=1e-5)


# ------------------------------------------------------------------ noaux_tc 偏置


def test_bias_changes_selection_but_not_weights():
    """e_score_correction_bias 只影响 top-k 选择，不进路由权重。"""
    cfg = cfg_tiny(n_routed_experts=16, n_activated_experts=4)
    mx.random.seed(62)
    gate = MoEGate(cfg, 0)
    x = mx.random.normal((6, cfg.dim))
    w0, idx0, s0 = gate(x)
    mx.eval(w0, idx0, s0)
    # 给一个"本来没入选"的专家加偏置 → 它必须被选进来
    np.asarray(s0)[0]
    missed = [
        e for e in range(cfg.n_routed_experts) if e not in np.asarray(idx0)[0].tolist()
    ]
    e = missed[0]
    gate.bias = gate.bias.at[e].add(1e3)
    w1, idx1, s1 = gate(x)
    mx.eval(w1, idx1, s1)
    assert e in np.asarray(idx1)[0].tolist()
    # 分数本身不受偏置影响
    assert np.allclose(np.asarray(s1), np.asarray(s0), atol=1e-6)
    # 权重严格来自原始分数（norm_topk_prob=True → 选中分数归一化 × route_scale）
    sel_scores = np.take_along_axis(np.asarray(s1), np.asarray(idx1), axis=-1)
    want = sel_scores / sel_scores.sum(-1, keepdims=True) * cfg.route_scale
    assert np.allclose(np.asarray(w1), want, atol=1e-6)


def test_route_weights_without_norm():
    """norm_topk_prob=False：权重 = 原始分数 × route_scale（不做归一化）。"""
    cfg = cfg_tiny(norm_topk_prob=False, route_scale=2.0)
    mx.random.seed(63)
    gate = MoEGate(cfg, 0)
    x = mx.random.normal((3, cfg.dim))
    w, idx, s = gate(x)
    mx.eval(w, idx, s)
    sel = np.take_along_axis(np.asarray(s), np.asarray(idx), axis=-1)
    assert np.allclose(np.asarray(w), sel * 2.0, atol=1e-6)


def test_bias_update_rule_moves_load_toward_uniform():
    """热点 bias 下降，冷门上升；共同偏移归零、相对差为 2*rate。"""
    load = mx.array([10.0, 0.0, 0.0, 0.0])  # 专家 0 独占全部负载
    bias = mx.zeros((4,))
    new = update_expert_bias(bias, load, rate=0.1)
    mx.eval(new)
    n = np.asarray(new)
    assert n[0] < 0 and np.all(n[1:] > 0), n
    assert np.allclose(n[1:] - n[0], 0.2, atol=1e-6), n
    assert abs(n.mean()) < 1e-6
    # 均匀负载 → 不动
    even = update_expert_bias(bias, mx.ones((4,)), rate=0.1)
    mx.eval(even)
    assert np.allclose(np.asarray(even), 0, atol=1e-9)
    # 没有观测时保持原偏置。
    zero = update_expert_bias(bias, mx.zeros((4,)), rate=0.5)
    mx.eval(zero)
    assert np.allclose(np.asarray(zero), 0.0, atol=1e-6)
    # 偏置本身不产生梯度（freeze 在参数树上体现，见 test_router_bias_*）


def test_router_bias_in_parameters_but_frozen():
    """router.bias 进 checkpoint（parameters）但不进 trainable_parameters。"""
    from mlx.utils import tree_flatten

    model = tiny_model()
    params = {k for k, _ in tree_flatten(model.parameters())}
    trainable = {k for k, _ in tree_flatten(model.trainable_parameters())}
    bias_keys = [k for k in params if k.endswith("ffn.router.bias")]
    assert bias_keys, "router.bias 必须在 parameters 里（随 checkpoint 保存）"
    assert all(k not in trainable for k in bias_keys), (
        "router.bias 必须 freeze（不进优化器）"
    )
    # 也不出现在梯度树里
    ids = mx.random.randint(0, model.config.vocab_size, (1, 8))

    def loss_fn(m, ids):
        return m(ids, labels=ids, use_mtp=False).loss

    import mlx.nn as nn

    _, grads = nn.value_and_grad(model, loss_fn)(model, ids)
    mx.eval(grads)
    grad_keys = {k for k, _ in tree_flatten(grads)}
    assert not any(k.endswith("ffn.router.bias") for k in grad_keys)


# ------------------------------------------------------------------ 专家计算


def test_clamped_swiglu_boundaries():
    """clamp 边界：gate 上界、up 双侧；limit=0 表示不 clamp。"""
    L = 10.0
    u = mx.array([1.0, -1.0, 3.0])
    # gate 超过上界 → 与恰好等于上界一致；up 的钳制同理
    for g in (2 * L, 100.0, 1e6):
        a = expert_act(mx.full((3,), g), u, L)
        b = expert_act(mx.full((3,), L), u, L)
        mx.eval(a, b)
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    for up in (2 * L, 100.0, 1e6):
        a = expert_act(mx.array([1.0]), mx.array([up]), L)
        b = expert_act(mx.array([1.0]), mx.array([L]), L)
        mx.eval(a, b)
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
        a = expert_act(mx.array([1.0]), mx.array([-up]), L)
        b = expert_act(mx.array([1.0]), mx.array([-L]), L)
        mx.eval(a, b)
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    # 未触界：silu(gate)·up
    g = mx.array([1.5, -0.5, 9.999])
    v = mx.array([2.0, -3.0, 0.25])
    got = expert_act(g, v, L)
    mx.eval(got)
    gv, vv = np.asarray(g), np.asarray(v)
    assert np.allclose(np.asarray(got), (gv / (1 + np.exp(-gv))) * vv, atol=1e-6)
    # limit=0（/负数）= 不 clamp
    big = mx.array([100.0])
    got0 = expert_act(big, mx.array([-1000.0]), 0.0)
    mx.eval(got0)
    assert np.isfinite(np.asarray(got0)).all()
    assert abs(float(got0[0]) - float(100.0 / (1 + np.exp(-100.0)) * -1000.0)) < 1e-3


def test_stacked_experts_gate_up_layout():
    """堆叠专家 (E,2I,D)：前半 gate、后半 up，与逐专家独立线性等价。"""
    cfg = cfg_tiny()
    mx.random.seed(64)
    ffn = MoEFeedForward(cfg, 0)
    x = mx.random.normal((3, cfg.dim))
    h = x @ ffn.experts.gate_up_w.swapaxes(-1, -2)  # (E,M,2I)
    gate, up = mx.split(h, 2, axis=-1)
    mx.eval(gate, up)
    for e in (0, 3):
        want_gate = (
            np.asarray(x) @ np.asarray(ffn.experts.gate_up_w[e])[: cfg.moe_inter_dim].T
        )
        want_up = (
            np.asarray(x) @ np.asarray(ffn.experts.gate_up_w[e])[cfg.moe_inter_dim :].T
        )
        assert np.allclose(np.asarray(gate[e]), want_gate, atol=1e-5)
        assert np.allclose(np.asarray(up[e]), want_up, atol=1e-5)


def test_unselected_experts_do_not_affect_output():
    """top-k 之外的路由专家对输出零贡献（改它们权重输出逐位不变）。"""
    cfg = cfg_tiny(n_routed_experts=32, n_activated_experts=4)
    mx.random.seed(65)
    ffn = MoEFeedForward(cfg, 0)
    x = mx.random.normal((1, 4, cfg.dim))  # M=4 → 稠密路径（逐位可比）
    w, idx, _ = ffn.router(x.reshape(4, cfg.dim))
    mx.eval(w, idx)
    selected = set(np.asarray(idx).reshape(-1).tolist())
    unselected = [e for e in range(cfg.n_routed_experts) if e not in selected]
    assert unselected, "本用例需要至少一个未选中的专家"
    base = ffn(x)
    mx.eval(base)
    e = unselected[0]
    ffn.experts.gate_up_w = ffn.experts.gate_up_w.at[e].add(0.5)
    ffn.experts.down_w = ffn.experts.down_w.at[e].add(0.5)
    other = ffn(x)
    mx.eval(other)
    assert bool(mx.all(base == other).item()), "未选中专家的权重不应影响输出"
    e2 = sorted(selected)[0]
    ffn.experts.gate_up_w = ffn.experts.gate_up_w.at[e2].add(0.5)
    changed = ffn(x)
    mx.eval(changed)
    assert float(mx.max(mx.abs(changed - base))) > 1e-6, "选中专家的权重必须影响输出"


def test_dense_and_sparse_routing_paths_agree():
    """M<=8 的稠密广播路径与 gather_mm 稀疏路径数值一致。"""
    cfg = cfg_tiny()
    mx.random.seed(66)
    ffn = MoEFeedForward(cfg, 0)
    x = mx.random.normal((2, 5, cfg.dim))
    flat = x.reshape(10, cfg.dim)
    w, idx, _ = ffn.router(flat)
    dense = ffn._dense_forward(x, idx, w)
    sparse = ffn._sparse_forward(x, idx, w)
    mx.eval(dense, sparse)
    assert float(mx.max(mx.abs(dense - sparse))) < 1e-5


def test_shared_expert_is_always_added():
    """每 token 必走 1 个共享专家：MoE = 路由专家 + 共享专家。"""
    cfg = cfg_tiny()
    mx.random.seed(67)
    ffn = MoEFeedForward(cfg, 0)
    x = mx.random.normal((1, 4, cfg.dim))
    total = ffn(x)
    mx.eval(total)
    flat = x.reshape(4, cfg.dim)
    w, idx, _ = ffn.router(flat)
    routed = ffn._dense_forward(x, idx, w)
    shared = ffn.shared(x)
    mx.eval(routed, shared)
    assert np.allclose(
        np.asarray(total), np.asarray(routed) + np.asarray(shared), atol=1e-5
    )
    # 把共享专家清零 → 输出只剩路由专家
    ffn.shared.w2.weight = mx.zeros_like(ffn.shared.w2.weight)
    only_routed = ffn(x)
    mx.eval(only_routed)
    assert np.allclose(np.asarray(only_routed), np.asarray(routed), atol=1e-5)
    assert float(mx.max(mx.abs(routed))) > 1e-4


def test_load_stats_count_topk_assignments():
    """负载统计 = 每个专家的 (token, 选中) 计数（top-k 口径）。"""
    cfg = cfg_tiny()
    model = tiny_model()
    cfg = model.config
    ids = mx.random.randint(0, cfg.vocab_size, (2, 6))
    model.train()
    out = model(ids, labels=ids, use_mtp=False)
    mx.eval(out.moe_loads)
    assert out.moe_loads.shape == (cfg.n_layers, cfg.n_routed_experts)
    # 每个 gate 的计数和 = B·T·top_k
    assert np.allclose(
        np.asarray(mx.sum(out.moe_loads, axis=-1)),
        ids.shape[0] * ids.shape[1] * cfg.n_activated_experts,
    )
    # eval 模式不返回侧信道
    model.eval()
    out2 = model(ids, labels=ids, use_mtp=False)
    assert out2.moe_loads is None
    model.train()


def test_update_moe_biases_end_to_end():
    """forward 的 moe_loads → update_moe_biases：热点专家偏置下降。"""
    from _v41_common import build

    model = build(cfg_tiny(moe_balance_method="noaux_tc"))
    cfg = model.config
    mx.random.seed(68)
    ids = mx.random.randint(0, cfg.vocab_size, (2, 8))
    model.train()
    out = model(ids, labels=ids, use_mtp=False)
    mx.eval(out.moe_loads)
    before = np.asarray(model.moe_bias_stack())
    loads = np.asarray(out.moe_loads)
    model.update_moe_biases(out.moe_loads)
    after = np.asarray(model.moe_bias_stack())
    hot = loads[0].argmax()
    assert after[0, hot] < before[0, hot]
    assert after[0, loads[0].argmin()] > before[0, loads[0].argmin()]
    delta = after - before
    assert np.max(np.abs(delta.mean(axis=1))) < 1e-9
    assert np.max(np.ptp(delta, axis=1)) <= 2 * cfg.bias_update_rate + 1e-9
    # 偏置被写回 buffer，下一次 forward 立即生效（选择可能变化）
    assert not np.allclose(after, before)
