"""Gated XSA（Exclusive Self-Attention）语义回归。

XSA 是注意力输出的逐 head 后处理：z = y − tanh(α)·(yᵀv/‖v‖²)·v。
四条不变式：
1. α=0 严格恒等（开/关 use_xsa 的输出差异不高于模型自身 run-to-run 噪声）；
2. α≠0 真的改变输出（接线生效，不是死代码）；
3. prefill 与逐 token 解码口径一致（v 取法在两路必须对齐）；
4. 只在主干最深 N 层挂参数，DSpark 草稿层不挂。

注意：单层的 `Attention` 调用是逐位确定的（见 test_xsa_layer_is_bit_exact），
整模型前向因 MoE scatter 存在 ~1e-6 的 run-to-run 噪声，故恒等性断言用
"不超过噪声底"而不是 "== 0"。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from mlx.utils import tree_flatten

from _v41_common import build, cfg_mix, seed_of, set_param
from model.attention import Attention
from model.cache import SharedAttnState, VibyCache
from model.config import VibyConfig


def _cfg(use_xsa=True, xsa_last_n=0, **kw):
    return VibyConfig(**{**cfg_mix().to_dict(), "use_xsa": use_xsa,
                         "xsa_last_n": xsa_last_n, **kw})


def _xsa_paths(model):
    return [p for p, _ in tree_flatten(model.parameters()) if p.endswith("xsa_alpha")]


def _set_all_xsa(model, value):
    for path, arr in tree_flatten(model.parameters()):
        if path.endswith("xsa_alpha"):
            set_param(model, path, mx.full(arr.shape, value, dtype=arr.dtype))


def _noise_floor(cfg, ids, seed):
    """同一个模型跑两遍的 max|Δlogit|：MoE scatter 带来的 run-to-run 噪声底。"""
    model = build(cfg, seed=seed)
    a = model(ids, start_pos=0, use_mtp=False).logits
    b = model(ids, start_pos=0, use_mtp=False).logits
    mx.eval(a, b)
    return float(mx.max(mx.abs(a - b)).item())


def test_xsa_layer_is_bit_exact_at_alpha_zero():
    """α=0 时单层 Attention 逐位恒等（不引入任何数值扰动）。"""
    cfg = _cfg(use_xsa=True)
    mx.random.seed(seed_of("xsa_layer"))
    x = mx.random.normal((2, 16, cfg.dim))
    for layer_idx in range(cfg.n_layers):
        attn = Attention(cfg, layer_idx)
        with_xsa = attn(x, 0, SharedAttnState())
        saved = attn.xsa_alpha
        attn.xsa_alpha = None
        without = attn(x, 0, SharedAttnState())
        attn.xsa_alpha = saved
        mx.eval(with_xsa, without)
        assert float(mx.max(mx.abs(with_xsa - without)).item()) == 0.0, layer_idx


def test_xsa_alpha_zero_is_identity():
    """α=0（零初始化）⇒ 开/关 XSA 的差异不超过模型自身噪声底。"""
    mx.random.seed(seed_of("xsa_ids"))
    ids = mx.random.randint(0, 200, (2, 24))
    on = build(_cfg(use_xsa=True), seed=seed_of("xsa"))
    off = build(_cfg(use_xsa=False), seed=seed_of("xsa"))
    a = on(ids, start_pos=0, use_mtp=False).logits
    b = off(ids, start_pos=0, use_mtp=False).logits
    mx.eval(a, b)
    delta = float(mx.max(mx.abs(a - b)).item())
    floor = _noise_floor(_cfg(use_xsa=False), ids, seed_of("xsa"))
    assert delta <= max(5.0 * floor, 1e-6), (delta, floor)


def test_xsa_nonzero_alpha_changes_output():
    """α≠0 必须真的改变输出（否则接线没生效）。"""
    model = build(_cfg(use_xsa=True), seed=seed_of("xsa"))
    mx.random.seed(seed_of("xsa_ids"))
    ids = mx.random.randint(0, 200, (2, 24))
    before = model(ids, start_pos=0, use_mtp=False).logits
    paths = _xsa_paths(model)
    assert paths, "use_xsa=True 时应有 xsa_alpha 参数"
    _set_all_xsa(model, 0.7)
    after = model(ids, start_pos=0, use_mtp=False).logits
    mx.eval(before, after)
    assert float(mx.max(mx.abs(after - before)).item()) > 1e-4


def test_xsa_only_on_deepest_backbone_layers():
    """xsa_last_n 控制作用层：只有最深 N 层挂 α，DSpark 草稿层不挂。"""
    cfg = _cfg(use_xsa=True, xsa_last_n=2)
    assert cfg.xsa_last_n == 2
    for i in range(cfg.n_layers):
        attn = Attention(cfg, i)
        expected = i >= cfg.n_layers - 2
        assert (attn.xsa_alpha is not None) == expected, f"layer {i}"
    draft = Attention(cfg, cfg.n_layers)
    assert draft.xsa_alpha is None


def test_xsa_default_on_and_auto_depth():
    """默认开；xsa_last_n=0 时自动取最深 max(1, n_layers // 3) 层。"""
    cfg = cfg_mix()
    assert cfg.use_xsa is True
    assert cfg.xsa_last_n == max(1, cfg.n_layers // 3)


def test_xsa_disabled_has_no_alpha_parameter():
    """use_xsa=False ⇒ 参数树里没有 xsa_alpha（开与关是两套不同的参数树）。"""
    model = build(_cfg(use_xsa=False), seed=seed_of("xsa"))
    assert not _xsa_paths(model)


def test_xsa_prefill_matches_decode():
    """prefill 与逐 token 解码在 α≠0 时仍一致（v 取法两路口径相同）。"""
    cfg = _cfg(use_xsa=True)
    model = build(cfg, seed=seed_of("xsa"))
    _set_all_xsa(model, 0.6)
    mx.random.seed(seed_of("xsa_ids"))
    ids = mx.random.randint(0, 200, (1, 16))

    full = model(ids, start_pos=0, use_mtp=False).logits
    cache = VibyCache(cfg)
    last = None
    for t in range(ids.shape[1]):
        last = model(ids[:, t : t + 1], start_pos=t, cache=cache, use_mtp=False).logits
    mx.eval(full, last)
    diff = float(mx.max(mx.abs(full[:, -1] - last[:, 0])).item())
    assert diff < 5e-5, diff


