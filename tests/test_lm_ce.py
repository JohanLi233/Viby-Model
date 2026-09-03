"""chunked 融合 lm_head+CE（model/kernels/lm_ce.py）的等价性回归测试。

对照基线：全量物化 logits = hidden @ W.T 后走 ce.cross_entropy（本身已是
逐行融合 kernel）。覆盖：
1. loss / z_mean 等价：多种 chunk 尺寸（含不整除、单块）× 有无 mask/-100；
2. 梯度等价：grad_hidden / grad_weight，且带 z-loss 项以触发 lse 余量；
3. kernel 禁用后的 eager 回退与基线一致；
4. 端到端：labels 路径不再返回 logits（need_logits=True 时才返回），
   两种路径 loss 一致且 value_and_grad 可反向；MTP 变体冒烟。

运行：python test_lm_ce.py
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.kernels import lm_ce
from model.kernels.ce import cross_entropy
from model.kernels.lm_ce import lm_head_cross_entropy
from model.model import VibyForCausalLM


def _data(B=2, T=37, D=64, V=512, seed=0, with_mask=False):
    rng = np.random.default_rng(seed)
    hidden = mx.array(rng.normal(0, 0.5, (B, T, D))).astype(mx.bfloat16)
    weight = mx.array(rng.normal(0, 0.05, (V, D))).astype(mx.bfloat16)
    labels = mx.array(rng.integers(0, V, (B, T)).astype(np.int32))
    mask = None
    if with_mask:
        m = rng.random((B, T)) > 0.3
        labels = mx.where(mx.array(m), labels, mx.array(-100, mx.int32))
        mask = mx.array(m.astype(np.float32))
    return hidden, weight, labels, mask


def _baseline(hidden, weight, labels, mask=None):
    return cross_entropy(hidden @ weight.T, labels, mask=mask, return_z=True)


def _maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_loss_and_z_match_baseline():
    for with_mask in (False, True):
        hidden, weight, labels, mask = _data(with_mask=with_mask)
        loss_ref, z_ref = _baseline(hidden, weight, labels, mask)
        for chunk in (16, 4096):  # 不整除多块 / 单块
            loss, z = lm_head_cross_entropy(
                hidden, weight, labels, mask=mask, return_z=True, chunk=chunk
            )
            dl = _maxdiff(loss, loss_ref)
            dz = _maxdiff(z, z_ref)
            assert dl < 2e-3, f"chunk={chunk} mask={with_mask} loss 偏差 {dl:.5f}"
            assert dz < 1e-2, f"chunk={chunk} mask={with_mask} z 偏差 {dz:.5f}"


def test_grads_match_baseline():
    hidden, weight, labels, mask = _data(with_mask=True)

    # 带上 z-loss 项，确保 lse 的余量（cot_lse）也走反向
    def f_fused(h, w):
        loss, z = lm_head_cross_entropy(h, w, labels, mask=mask, return_z=True)
        return loss + 1e-4 * z

    def f_ref(h, w):
        loss, z = _baseline(h, w, labels, mask)
        return loss + 1e-4 * z

    gh, gw = mx.grad(f_fused, argnums=(0, 1))(hidden, weight)
    rh, rw = mx.grad(f_ref, argnums=(0, 1))(hidden, weight)
    mx.eval(gh, gw, rh, rw)
    dh = _maxdiff(gh, rh)
    dw = _maxdiff(gw, rw)
    assert dh < 2e-3, f"grad_hidden 最大偏差 {dh:.5f}"
    assert dw < 2e-3, f"grad_weight 最大偏差 {dw:.5f}"


def test_fallback_matches_baseline():
    hidden, weight, labels, mask = _data(with_mask=True)
    loss_ref, z_ref = _baseline(hidden, weight, labels, mask)
    old = lm_ce._LM_CE_DISABLED
    try:
        lm_ce._LM_CE_DISABLED = True
        loss, z = lm_head_cross_entropy(
            hidden, weight, labels, mask=mask, return_z=True
        )
    finally:
        lm_ce._LM_CE_DISABLED = old
    assert _maxdiff(loss, loss_ref) < 2e-3
    assert _maxdiff(z, z_ref) < 1e-2


def _tiny_model(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=512,
        mtp_depth=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        moe_latent_dim=0,
    )
    base.update(kw)
    mx.random.seed(42)
    return VibyForCausalLM(VibyConfig(**base))


def test_end_to_end_labels_path():
    """labels 路径默认不返回 logits；与 need_logits=True 的 loss 一致且可反向。"""
    rng = np.random.default_rng(1)
    X = mx.array(rng.integers(3, 256, (2, 40)).astype(np.int64))
    Y = mx.array(rng.integers(3, 256, (2, 40)).astype(np.int64))
    for kw in ({}, {"mtp_depth": 1, "mtp_steps": 2, "mtp_loss_weight": 0.1}):
        model = _tiny_model(**kw)
        model.train()
        out = model(input_ids=X, labels=Y)
        assert out.loss is not None
        assert out.logits is None, "labels 路径不应再物化全量 logits"
        out_ref = model(input_ids=X, labels=Y, need_logits=True)
        assert out_ref.logits is not None
        d = _maxdiff(out.loss, out_ref.loss)
        assert d < 2e-3, f"kw={kw} 融合/全量 loss 偏差 {d:.5f}"

        loss, grads = nn.value_and_grad(
            model, lambda: model(input_ids=X, labels=Y).loss
        )()
        mx.eval(loss, grads)
        flat = [g for g in tree_flatten(grads)]
        assert all(mx.all(mx.isfinite(g)).item() for _, g in flat)


def test_no_labels_unchanged():
    """生成/评估路径（labels=None）行为不变：仍返回全量 logits。"""
    model = _tiny_model()
    model.eval()
    X = mx.array(np.random.default_rng(2).integers(3, 256, (1, 16)))
    out = model(input_ids=X)
    assert out.logits is not None and out.loss is None


if __name__ == "__main__":
    test_loss_and_z_match_baseline()
    print("ok: loss/z 等价")
    test_grads_match_baseline()
    print("ok: 梯度等价")
    test_fallback_matches_baseline()
    print("ok: eager 回退等价")
    test_end_to_end_labels_path()
    print("ok: 端到端 labels 路径")
    test_no_labels_unchanged()
    print("ok: labels=None 路径不变")
    print("PASS")
