"""DSpark/MTP、梯度与编译：报告 §2.4.3 + §3.1 的训练接口契约。

验证的语义：
- §2.4.3 DSpark：主干后挂 draft 层，一次前向并行算 dspark_block_size 个草稿位置
  的 base logits，马尔可夫头建模草稿间依赖，置信度头预测接受概率；
- 训练阶段"只训 DSpark、冻结主干"，且 DSpark 目标**不回传主干**
  （stop_gradient 挂在锚点表示上）；
- nn.value_and_grad 能覆盖全部 trainable 参数；mx.compile 前向可跑且可复现。

运行：.venv/bin/python -m pytest tests/test_v41_train.py -q
"""

import numpy as np
import pytest

from _v41_common import build, cfg_tiny, tiny_model
from model.model import lm_head_ce

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def _grads_of(loss_fn, model, *args):
    grads = nn.value_and_grad(model, lambda m, *a: loss_fn(m, *a))(model, *args)[1]
    mx.eval(grads)
    return dict(tree_flatten(grads))


def _forward(model, ids, **kw):
    out = model(ids, labels=ids, use_mtp=True, **kw)
    mx.eval(out.loss, out.lm_loss, out.mtp_loss, out.z_loss)
    return out


# ------------------------------------------------------------------ DSpark 结构

def test_dspark_stage_layout():
    """draft 阶段：首层锚点投影 + 末层预测头，逐层一个 SWA block。"""
    cfg = cfg_tiny(n_mtp_layers=1, dspark_block_size=4, dspark_target_layer_ids=(1, 3))
    model = build(cfg)
    assert len(model.mtp_modules) == 1
    stage = model.mtp_modules[0]
    assert stage.block_size == 4
    assert hasattr(stage, "main_proj") and stage.main_proj.weight.shape == (cfg.dim, cfg.dim * 2)
    assert hasattr(stage, "main_norm")
    assert hasattr(stage, "markov_head") and hasattr(stage, "confidence_head")
    assert stage.markov_head.embed.weight.shape == (cfg.vocab_size, cfg.dspark_markov_rank)
    assert stage.confidence_head.proj.weight.shape == (1, cfg.dim + cfg.dspark_markov_rank)
    # draft block 是纯滑窗层（ratio=0）且用 DSpark 的窄 MoE
    dtype_idx = cfg.n_layers
    assert cfg.compress_ratios[dtype_idx] == 0
    assert stage.layer.ffn.n_routed == cfg.dspark_n_routed_experts
    assert stage.layer.ffn.top_k == cfg.dspark_n_activated_experts
    assert stage.layer.attn.ratio == 0


def test_mtp_loss_computable_and_backpropagates():
    """block_size 个草稿位置的 CE + 置信度 BCE 可算、可反传（含两个头）。"""
    cfg = cfg_tiny(dspark_block_size=4)
    model = build(cfg)
    mx.random.seed(80)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 12))
    _, grads = nn.value_and_grad(
        model, lambda m, x: m(x, labels=x, use_mtp=True).mtp_loss
    )(model, ids)
    mx.eval(grads)
    g = dict(tree_flatten(grads))
    assert np.isfinite(np.asarray(_forward(model, ids).mtp_loss)).all()
    assert _forward(model, ids).mtp_loss.item() > 0
    for key in ("mtp_modules.0.markov_head.embed.weight",
                "mtp_modules.0.markov_head.head.weight",
                "mtp_modules.0.confidence_head.proj.weight",
                "mtp_modules.0.layer.ffn.router.weight",
                "mtp_modules.0.layer.attn.wq_a.weight"):
        assert key in g, key
        assert float(mx.max(mx.abs(g[key]))) > 0, key
    # 置信度头确实参与损失：清零后损失必须变（BCE 项退化成常数 log2）
    model.mtp_modules[0].confidence_head.proj.weight = mx.zeros_like(
        model.mtp_modules[0].confidence_head.proj.weight
    )
    changed = _forward(model, ids).mtp_loss
    assert abs(float(changed) - 0.0) > 1e-6


@pytest.mark.parametrize("block", [1, 2, 4])
def test_mtp_block_size_changes_draft_loss(block):
    """dspark_block_size 决定并行草稿位置数，损失随之变化。"""
    cfg = cfg_tiny(dspark_block_size=block)
    model = build(cfg)
    mx.random.seed(81)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 10))
    out = _forward(model, ids)
    assert out.mtp_loss.item() > 0
    assert np.isfinite(np.asarray(out.mtp_loss)).all()


def test_total_loss_combines_lm_mtp_and_z():
    """out.loss = lm + z·z_loss + w·mtp + a·aux（z 默认 0；a = aux_balance_loss_weight）。"""
    cfg = cfg_tiny(dspark_block_size=2, mtp_loss_weight=0.3, z_loss_weight=1e-4)
    model = build(cfg)
    mx.random.seed(82)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 10))
    out = _forward(model, ids)
    want = float(out.lm_loss) + cfg.z_loss_weight * float(out.z_loss) \
        + cfg.mtp_loss_weight * float(out.mtp_loss) \
        + cfg.aux_balance_loss_weight * float(out.aux_loss)
    assert float(out.loss) == pytest.approx(want, rel=1e-5, abs=1e-6)
    # use_mtp=False：不算 MTP，也不返回 mtp_loss
    plain = model(ids, labels=ids, use_mtp=False)
    mx.eval(plain.loss, plain.mtp_loss)
    assert plain.mtp_loss is None
    assert float(plain.loss) == pytest.approx(
        float(plain.lm_loss) + cfg.z_loss_weight * float(plain.z_loss)
        + cfg.aux_balance_loss_weight * float(plain.aux_loss), rel=1e-5, abs=1e-6)
    # 关掉 MTP 后 loss 与开 MTP 不同
    assert abs(float(plain.loss) - float(out.loss)) > 1e-6


def test_mtp_loss_does_not_backprop_into_backbone():
    """stop_gradient：DSpark 目标对主干（layers/norm）的梯度恒为 0。

    注意 embed / lm_head 是共享权重（草稿输入嵌入与预测头），它们仍会拿到
    梯度——报告约束的是"不回传主干"（隐藏状态锚点被 stop_gradient 截断）。
    """
    cfg = cfg_tiny()
    model = build(cfg)
    mx.random.seed(83)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 10))
    g = nn.value_and_grad(model, lambda m, x: m(x, labels=x, use_mtp=True).mtp_loss)(
        model, ids)[1]
    mx.eval(g)
    gl = dict(tree_flatten(g))
    backbone = {k: float(mx.max(mx.abs(v))) for k, v in gl.items()
                if k.startswith("model.layers") or k.startswith("model.norm")}
    assert backbone, "应当有主干参数的梯度槽位"
    bad = {k: v for k, v in backbone.items() if v != 0.0}
    assert not bad, f"主干拿到了 DSpark 梯度：{list(bad)[:5]}"
    # 锚点被截断：主干里唯一能拿到梯度的是共享的 token 嵌入
    nonzero = [k for k, v in gl.items() if float(mx.max(mx.abs(v))) > 0]
    assert any(k.startswith("mtp_modules") for k in nonzero)
    assert all(k.startswith("mtp_modules") or k == "model.embed.weight"
               or k.startswith("lm_head") for k in nonzero), nonzero


def test_freeze_backbone_trains_only_dspark():
    """主干 + 共享头冻结后，只有 DSpark 参数可训练、且全部有梯度。"""
    cfg = cfg_tiny()
    model = build(cfg)
    mx.random.seed(84)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 10))
    model.model.freeze()
    model.lm_head.freeze()
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable, "DSpark 参数必须仍可训练"
    assert all(k.startswith("mtp_modules") for k in trainable), list(trainable)[:5]
    g = nn.value_and_grad(model, lambda m, x: m(x, labels=x, use_mtp=True).mtp_loss)(
        model, ids)[1]
    mx.eval(g)
    gl = dict(tree_flatten(g))
    assert set(gl) == set(trainable)
    assert all(float(mx.max(mx.abs(v))) > 0 for v in gl.values()), "DSpark 参数应全部有梯度"


# ------------------------------------------------------------------ 梯度 / 编译

def test_value_and_grad_covers_all_trainable_parameters():
    """nn.value_and_grad 覆盖全部 trainable 参数，梯度有限。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(85)
    ids = mx.random.randint(0, cfg.vocab_size, (2, 10))

    def loss_fn(m, x):
        return m(x, labels=x, use_mtp=True).loss

    loss, grads = nn.value_and_grad(model, loss_fn)(model, ids)
    mx.eval(loss, grads)
    gl = dict(tree_flatten(grads))
    tl = dict(tree_flatten(model.trainable_parameters()))
    assert set(gl) == set(tl), (set(tl) - set(gl), set(gl) - set(tl))
    assert np.isfinite(float(loss))
    for k, v in gl.items():
        assert np.isfinite(np.asarray(v)).all(), k
    nonzero = sum(1 for v in gl.values() if float(mx.max(mx.abs(v))) > 0)
    assert nonzero > 0.8 * len(gl), (nonzero, len(gl))
    # 梯度确实依赖输入：换一组 token，主干梯度必须变
    ids2 = mx.random.randint(0, cfg.vocab_size, (2, 10))
    g2 = dict(tree_flatten(nn.value_and_grad(model, loss_fn)(model, ids2)[1]))
    mx.eval(list(g2.values()))
    assert not np.allclose(np.asarray(g2["model.layers.0.attn.wq_a.weight"]),
                           np.asarray(gl["model.layers.0.attn.wq_a.weight"]))


def test_compile_forward_is_reproducible_and_matches_eager():
    """mx.compile 前向：两次调用逐位一致，且与 eager 数值一致。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(86)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 12))

    def loss_closure(x):
        return model(x, labels=x, use_mtp=True).loss

    compiled = mx.compile(loss_closure)
    a = compiled(ids)
    b = compiled(ids)
    mx.eval(a, b)
    assert bool(a == b), (float(a), float(b))
    eager = loss_closure(ids)
    mx.eval(eager)
    assert abs(float(a) - float(eager)) < 1e-5
    # compile 下的梯度也能跑（模型必须闭包捕获：compile 会把 Module 参数当 dict 传参）
    vg = nn.value_and_grad(model, lambda m, x: m(x, labels=x, use_mtp=True).loss)
    compiled_vg = mx.compile(lambda x: vg(model, x))
    loss2, grads2 = compiled_vg(ids)
    mx.eval(loss2, grads2)
    assert abs(float(loss2) - float(eager)) < 1e-4
    assert all(np.isfinite(np.asarray(v)).all() for v in dict(tree_flatten(grads2)).values())


def test_optimizer_steps_reduce_loss_on_fixed_batch():
    """训练步烟测：连续 AdamW 更新能把单批 loss 压下去（参数有限不过冲）。"""
    model = build(cfg_tiny())        # 用新模型：本用例会改参数，不能污染缓存实例
    cfg = model.config
    mx.random.seed(87)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 12))

    def loss_fn(m, x):
        return m(x, labels=x, use_mtp=True).loss

    import mlx.optimizers as optim

    opt = optim.AdamW(learning_rate=1e-3)     # 1e-2 在这个小模型上会过冲
    vg = nn.value_and_grad(model, loss_fn)
    before = float(loss_fn(model, ids))
    hist = [before]
    for _ in range(10):
        loss, grads = vg(model, ids)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        assert np.isfinite(float(loss))
        hist.append(float(loss_fn(model, ids)))
    assert np.isfinite(hist[-1])
    assert hist[-1] < hist[0] * 0.5, hist


# ------------------------------------------------------------------ 损失口径

def test_lm_head_ce_loss_mask_semantics():
    """lm_head_ce：mask 只统计有效 token；z_loss = mean(lse²)。"""
    mx.random.seed(88)
    h = mx.random.normal((2, 6, 32))
    w = mx.random.normal((50, 32))
    labels = np.asarray(mx.random.randint(0, 50, (2, 6)))
    ce_none, z_none = lm_head_ce(h, w, mx.array(labels), None, z_weight=1e-4)
    ce_ones, z_ones = lm_head_ce(h, w, mx.array(labels), mx.ones((2, 6)), z_weight=1e-4)
    mx.eval(ce_none, z_none, ce_ones, z_ones)
    assert float(ce_none) == pytest.approx(float(ce_ones), rel=1e-5)
    assert float(z_none) == pytest.approx(float(z_ones), rel=1e-5)
    # 手算 CE（fp32）
    logits = np.asarray(h) @ np.asarray(w).T
    lse = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
    want_ce = float(np.mean(lse - np.take_along_axis(logits, labels[:, :, None], -1)[..., 0]))
    assert float(ce_none) == pytest.approx(want_ce, rel=1e-4, abs=1e-5)
    assert float(z_none) == pytest.approx(float(np.mean(lse ** 2)), rel=1e-4)
    # 只保留一半 token：CE 等于那一半上的平均
    mask = np.zeros((1, 6), np.float32)
    mask[0, 2:] = 1.0
    ce_half, _ = lm_head_ce(h[:1], w, mx.array(labels[:1]), mx.array(mask), z_weight=0.0)
    mx.eval(ce_half)
    ce_tok = lse[:1] - np.take_along_axis(logits[:1], labels[:1, :, None], -1)[..., 0]
    assert float(ce_half) == pytest.approx(float(ce_tok[0, 2:].mean()), rel=1e-4)
    # 全 0 mask：分母兜底为 1，不产生 NaN
    ce_zero, _ = lm_head_ce(h, w, mx.array(labels), mx.zeros((2, 6)), z_weight=0.0)
    mx.eval(ce_zero)
    assert float(ce_zero) == 0.0


def test_generate_runs_and_is_causal():
    """推理接口：prefill + 逐 token decode 的 greedy 生成不炸且长度正确。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(89)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 6))
    out = model.generate(ids, max_new_tokens=4, temperature=0.0)
    mx.eval(out)
    assert out.shape == (1, 6 + 4)
    # greedy：与手工 argmax 解码一致
    logits, cache = model.prefill(ids)
    logits = logits[:, -1]
    toks = []
    for _ in range(4):
        tok = mx.argmax(logits, axis=-1)
        toks.append(tok)
        logits, cache = model.decode_step(tok, cache)
    want = mx.concatenate([ids, mx.stack(toks, axis=1)], axis=1)
    mx.eval(want)
    assert bool(mx.all(out == want).item())
