"""mHC（Manifold-Constrained Hyper-Connections）：双随机混合与恒等起点。

验证的语义（报告 §2.4.1 Single-Pass mHC，式 (2)/(6)）：
- 残差流是 n=hc_mult 条并行副本；每层预测 (A,B,C)（代码里 pre/post/comb），
  其中 B（comb）经 Sinkhorn 投影到双随机矩阵，保证跨层传播非扩张；
- Single-Pass：block 的输入混合用上一个子层产出的 A，输入混合不再依赖本层
  系数（代码接线：attention 用上一个 block 的 FFN pre，FFN 用本层 attention
  算出、但取自残差更新之前的 pre）；
- 起点是标准 pre-norm 残差：pre ≈ one-hot(第 0 条流)、post = 1。

运行：.venv/bin/python -m pytest tests/test_v41_hc.py -q
"""

import numpy as np

from _v41_common import cfg_tiny, tiny_model
from model.block import apply_hc_pre_norm
from model.hc import HyperConnection, hc_post, hc_pre, hc_split, identity_pre_mix, sinkhorn

import mlx.core as mx


def _dev(c):
    return (float(np.abs(np.asarray(c.sum(axis=-1)) - 1).max()),
            float(np.abs(np.asarray(c.sum(axis=-2)) - 1).max()))


def test_sinkhorn_is_nonnegative_and_converges():
    """Sinkhorn 迭代收敛：行列偏差随迭代单调下降，不会炸。"""
    mx.random.seed(50)
    raw = mx.random.normal((4, 6, 4, 4)) * 3.0
    devs = []
    for iters in (1, 2, 5, 20, 40):
        out = sinkhorn(raw, iters=iters, eps=1e-6)
        mx.eval(out)
        assert float(mx.min(out)) >= 0.0
        devs.append(max(_dev(out)))
    assert all(devs[i] > devs[i + 1] for i in range(len(devs) - 1)), devs
    row40, col40 = _dev(sinkhorn(raw, iters=40, eps=1e-6))
    assert col40 < 1e-5 and row40 < 1e-2, (row40, col40)
    # 极端输入仍然有限（softmax 起手 + eps 兜底）
    out = sinkhorn(mx.array(np.full((1, 1, 4, 4), 1e6, np.float32)), iters=5, eps=1e-6)
    mx.eval(out)
    assert np.isfinite(np.asarray(out)).all()
    assert abs(float(mx.sum(out[0, 0, 0])) - 1.0) < 1e-4


def test_mixes_comb_is_doubly_stochastic():
    """HyperConnection.mixes 产出的 comb 是双随机矩阵（真实 init 口径）。"""
    cfg = cfg_tiny(hc_mult=4)
    model = tiny_model()
    for layer in model.model.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            x = mx.random.normal((2, 5, cfg.hc_mult, cfg.dim))
            pre, post, comb = hc.mixes(x)
            mx.eval(pre, post, comb)
            row, col = _dev(comb)
            assert row < 1e-4 and col < 1e-4, (row, col)
            assert pre.shape == (2, 5, cfg.hc_mult)
            assert post.shape == (2, 5, cfg.hc_mult)
            assert comb.shape == (2, 5, cfg.hc_mult, cfg.hc_mult)


def test_identity_start_with_zero_mix_weights():
    """把混合矩阵清零 → 严格回到标准 pre-norm 残差起点（式 (2) 的恒等解）。"""
    hc_mult = 4
    cfg = cfg_tiny(hc_mult=hc_mult)
    hc = HyperConnection(cfg.dim, hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    hc.fn.weight = mx.zeros_like(hc.fn.weight)
    x = mx.random.normal((2, 3, hc_mult, cfg.dim))
    pre, post, comb = hc.mixes(x)
    mx.eval(pre, post, comb)
    # post = 2·σ(0) = 1
    assert np.allclose(np.asarray(post), 1.0, atol=1e-6)
    # pre = σ(base) + 1e-6：第 0 条流 sigmoid(4)、其余 sigmoid(-4)
    want = np.array([1 / (1 + np.exp(-4)) + 1e-6] + [1 / (1 + np.exp(4)) + 1e-6] * (hc_mult - 1))
    assert np.allclose(np.asarray(pre)[0, 0], want, atol=1e-6)
    assert np.asarray(pre)[0, 0, 0] > 0.9
    # comb = softmax(0) = 均匀 → 双随机
    assert np.allclose(np.asarray(comb), 1.0 / hc_mult, atol=1e-6)
    # fn 形状与参数计数一致：mix_hc × (hc·dim)
    assert hc.fn.weight.shape == ((2 + hc_mult) * hc_mult, hc_mult * cfg.dim)


def test_identity_start_in_real_model():
    """真实模型（Marin 截断正态 init）：post ≈ 1、pre 以第 0 条流为主。"""
    model = tiny_model()
    cfg = model.config
    x = mx.random.normal((1, 6, cfg.hc_mult, cfg.dim))
    for layer in model.model.layers:
        hc = layer.attn_hc
        pre, post, comb = hc.mixes(x)
        mx.eval(pre, post)
        p = np.asarray(pre)[0, 0]
        # post = 2σ(mix)、base 把中心放在 σ(0)=0.5：均值≈1，个体随 mix 小幅抖动
        assert 0.85 < float(mx.mean(post)) < 1.15, float(mx.mean(post))
        assert float(mx.min(post)) > 0.4 and float(mx.max(post)) < 1.6
        assert hc.base[hc.hc_mult:2 * hc.hc_mult].tolist() == [0.0] * hc.hc_mult
        assert p[0] > 0.9 and p[0] > 10 * max(p[1:]), p


def test_hc_split_matches_formula():
    """hc_split 手算参照：pre=σ(m·s0+b0)+ε、post=2σ(m·s1+b1)、comb=m·s2+b2。"""
    hc_mult = 3
    mix = np.random.RandomState(0).randn(2, 4, (2 + hc_mult) * hc_mult).astype(np.float32)
    scale = np.ones(3, np.float32)
    base = np.zeros((2 + hc_mult) * hc_mult, np.float32)
    base[:hc_mult] = np.array([4.0, -4.0, -4.0], np.float32)
    pre, post, comb = hc_split(mx.array(mix), mx.array(scale), mx.array(base), hc_mult)
    mx.eval(pre, post, comb)
    m = mix
    want_pre = 1.0 / (1.0 + np.exp(-(m[..., :hc_mult] + base[:hc_mult]))) + 1e-6
    want_post = 2.0 / (1.0 + np.exp(-(m[..., hc_mult:2 * hc_mult] + base[hc_mult:2 * hc_mult])))
    assert np.allclose(np.asarray(pre), want_pre, atol=1e-6)
    assert np.allclose(np.asarray(post), want_post, atol=1e-6)
    assert np.allclose(np.asarray(comb), m[..., 2 * hc_mult:], atol=1e-6)


def test_hc_pre_selects_stream_and_hc_post_formula():
    """hc_pre 是流的加权收缩；hc_post 手算参照 out[m]=post[m]·x+Σ_j comb[m,j]·res[j]。"""
    hc_mult = 3
    mx.random.seed(53)
    x = mx.random.normal((2, 4, hc_mult, 6))
    pre = mx.random.uniform(0, 1, (2, 4, hc_mult))
    y = hc_pre(x, pre)
    mx.eval(y)
    want = np.einsum("bth,bthd->btd", np.asarray(pre), np.asarray(x))
    assert np.allclose(np.asarray(y), want, atol=1e-6)
    # identity_pre_mix：只读第 0 条流（逐位相等）
    ident = identity_pre_mix(x, hc_mult)
    assert bool(mx.all(hc_pre(x, ident) == x[..., 0, :]).item())
    assert np.asarray(ident)[0, 0].tolist() == [1.0, 0.0, 0.0]
    # hc_post
    residual = mx.random.normal((2, 4, hc_mult, 6))
    h = mx.random.normal((2, 4, 6))
    post = mx.random.uniform(0.5, 1.5, (2, 4, hc_mult))
    comb = mx.random.uniform(0, 1, (2, 4, hc_mult, hc_mult))
    out = hc_post(h, residual, post, comb)
    mx.eval(out)
    want2 = (np.asarray(post)[..., None] * np.asarray(h)[:, :, None, :]
             + np.einsum("bthj,btjd->bthd", np.asarray(comb), np.asarray(residual)))
    assert out.shape == (2, 4, hc_mult, 6)
    assert np.allclose(np.asarray(out), want2, atol=1e-6)


def test_block_single_pass_wiring_matches_manual_replication():
    """Single-Pass 接线（式 6）：手写循环必须与 VibyModel 逐位一致。

    手写循环严格照 model/block.py 的接线：
      attention 输入用传进来的 pre_mix（上一个子层产出的 A_l-1）
      attention 用本层的 mixes 做残差更新
      FFN 输入用本层 attention 的 pre（A_l，取自残差更新之前）
      返回 ffn_pre 给下一个子层
    把 FFN 输入换成它自己的 pre（非 Single-Pass 写法）结果必须不同。
    """
    from model.cache import SharedAttnState

    model = tiny_model()
    cfg = model.config
    mx.random.seed(54)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))   # T=8 → MoE 稠密路径，逐位可比
    ref, _, _ = model.model(ids)
    mx.eval(ref)

    def manual(single_pass: bool):
        h = model.model.embed(ids)
        h = mx.repeat(h[:, :, None, :], cfg.hc_mult, axis=2)
        shared = SharedAttnState()
        pre = identity_pre_mix(h, cfg.hc_mult)
        for layer in model.model.layers:
            residual = h
            attn_pre, attn_post, attn_comb = layer.attn_hc.mixes(h)
            h = apply_hc_pre_norm(h, pre, layer.attn_norm)
            h = layer.attn(h, 0, shared, None, None, None)
            h = hc_post(h, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = layer.ffn_hc.mixes(h)
            h = apply_hc_pre_norm(h, attn_pre if single_pass else ffn_pre, layer.ffn_norm)
            h = layer.ffn(h)
            h = hc_post(h, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        return apply_hc_pre_norm(h, pre, model.model.norm)

    got = manual(single_pass=True)
    mx.eval(got)
    assert bool(mx.all(got == ref).item()), "Single-Pass 接线与 VibyModel 前向不一致"
    other = manual(single_pass=False)
    mx.eval(other)
    assert float(mx.max(mx.abs(other - ref))) > 1e-4, "改成 FFN 自己的 pre 后输出应当不同"


def test_block_returns_fresh_ffn_pre():
    """Block 返回的 pre_mix 是本层 FFN 的 A（供下一个子层/block 使用）。"""
    from model.cache import SharedAttnState

    model = tiny_model()
    cfg = model.config
    mx.random.seed(55)
    x = mx.random.normal((1, 4, cfg.hc_mult, cfg.dim))
    block = model.model.layers[0]
    pre_in = identity_pre_mix(x, cfg.hc_mult)
    out, pre_next = block(x, 0, pre_in, SharedAttnState(), None, None, None, False)
    mx.eval(out, pre_next)
    assert pre_next.shape == (1, 4, cfg.hc_mult)
    assert float(mx.max(mx.abs(pre_next - pre_in))) > 1e-6, "返回的应是新算出的系数"
    # 与"对块输出的残差再跑一次 ffn_hc.mixes"无关：它是从残差更新前算的，
    # 但必须与本层 attn_hc 的 pre 不同（每个子层各算各的）
    attn_pre, _, _ = block.attn_hc.mixes(x)
    mx.eval(attn_pre)
    assert float(mx.max(mx.abs(pre_next - attn_pre))) > 1e-6
