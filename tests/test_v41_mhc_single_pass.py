"""Single-Pass mHC 的接线回归（报告 §2.4.1 eq.6 + 官方 inference/model.py Block.forward）。

V4 的 mHC：X_{l+1} = B_l X_l + C_l F_l(A_l X_l)——本层系数喂本层输入。
**V4.1 换成 Single-Pass**：X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)——每个子层
消费**上一个子层**算出的输入混合系数；本层系数只喂下一个子层。这个错位正是
"残差更新与系数预测能合并成一次遍历"的前提（Mega-mHC）。

本文件用两种独立手段钉住语义：
1) **接线检查**：拦 apply_hc_pre_norm，验证每个调用点拿到的 pre_mix 就是"上一个子层算出的
   那一个"（不是本层的、也不是重新算的）——用模型自己的 HyperConnection 现算对照；
2) **扰动检查**：改某子层的系数投影，看哪些子层输入该变、哪些必须不变。
"""

import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from model import block as block_mod  # noqa: E402
from model.config import VibyConfig  # noqa: E402
from model.model import VibyForCausalLM  # noqa: E402


def _build(seed: int = 0):
    mx.random.seed(seed)
    cfg = VibyConfig(
        preset="tiny",
        max_seq_len=64,
        window_size=16,
        index_topk=8,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        dspark_block_size=2,
    )
    return VibyForCausalLM(cfg), cfg


def _trace(model, ids):
    """按执行顺序抓 (hc_pre 的输入, 每块返回的 ffn_pre)。"""
    calls = []  # [(x, pre_mix)]
    ret = {}  # layer_idx -> ffn_pre
    orig_apply = block_mod.apply_hc_pre_norm
    orig_block = block_mod.Block.__call__

    def cap_apply(x, pre_mix, norm):
        calls.append((np.array(x), np.array(pre_mix)))
        return orig_apply(x, pre_mix, norm)

    def cap_block(
        self,
        x,
        start_pos,
        pre_mix,
        shared,
        cache=None,
        segment_ids=None,
        pad_mask=None,
        decode=False,
    ):
        out, ffn_pre = orig_block(
            self, x, start_pos, pre_mix, shared, cache, segment_ids, pad_mask, decode
        )
        ret[self.layer_idx] = np.array(ffn_pre)
        return out, ffn_pre

    block_mod.apply_hc_pre_norm = cap_apply
    block_mod.Block.__call__ = cap_block
    try:
        out = model(ids, use_mtp=False)
        mx.eval(out.logits)
    finally:
        block_mod.apply_hc_pre_norm = orig_apply
        block_mod.Block.__call__ = orig_block
    return calls, ret


def test_each_sublayer_consumes_the_previous_sublayers_coefficients():
    """每个 hc_pre 调用点拿到的 pre_mix 必须来自"上一个子层"。

    每块两次 hc_pre：第 1 次（注意力输入）拿上一块 FFN 返回的 ffn_pre；
    第 2 次（FFN 输入）拿本块注意力算出的 attn_pre（用模型自己的
    HyperConnection 现算对照）。若退回 V4 的"本层系数喂本层"，两条都会挂。
    """
    model, cfg = _build()
    ids = mx.random.randint(0, cfg.vocab_size, (1, 12))
    calls, ffn_pre = _trace(model, ids)
    n = cfg.n_layers
    assert len(calls) == 2 * n, "apply_hc_pre_norm 调用次数应为 2×层数"

    first = calls[0][1]
    assert first.shape[-1] == cfg.hc_mult
    assert np.allclose(first[..., 0], 1.0) and np.allclose(first[..., 1:], 0.0), (
        "第 0 块注意力输入不是 identity pre_mix（第 0 条流）"
    )

    for l in range(n):
        x_attn, pre_attn = calls[2 * l + 0]  # 注意力输入处的 (X_l, pre_mix)
        x_ffn, pre_ffn = calls[2 * l + 1]  # FFN 输入处的   (X_l_after_attn, pre_mix)
        layer = model.model.layers[l]

        # 注意力输入用的是上一块 FFN 的系数（第 0 块是 identity，上面已单独断言）
        if l > 0:
            assert np.allclose(pre_attn, ffn_pre[l - 1], atol=1e-6), (
                "layer%d 注意力输入用的不是 layer%d FFN 的系数（Single-Pass 错位丢失）"
                % (l, l - 1)
            )

        # FFN 输入用的是本块注意力的系数（模型自己的 mixes 现算）
        attn_pre, _, _ = layer.attn_hc.mixes(mx.array(x_attn))
        assert np.allclose(pre_ffn, np.array(attn_pre), atol=1e-6), (
            "layer%d FFN 输入用的不是本块注意力算出的系数" % l
        )

        # 本块注意力自己的系数不能出现在它自己的输入里（那是 V4 旧口径）
        own = np.array(attn_pre)
        assert not np.allclose(pre_attn, own, atol=1e-4), (
            "layer%d 注意力输入等于本层系数（V4 口径），Single-Pass 未生效" % l
        )

        assert x_ffn.shape == x_attn.shape


def test_perturbation_invariances():
    """扰动检查：本层系数只影响下一个子层，不影响自己的输入混合。"""
    model, cfg = _build()
    ids = mx.random.randint(0, cfg.vocab_size, (1, 12))

    calls, _ = _trace(model, ids)
    base_attn1, base_ffn1 = calls[2][1], calls[3][1]

    # 本层注意力系数变了：自己的注意力输入不变，自己的 FFN 输入要变
    hc = model.model.layers[1].attn_hc
    hc.fn.weight = hc.fn.weight + 0.7
    calls2, _ = _trace(model, ids)
    assert np.allclose(base_attn1, calls2[2][1], atol=1e-6), (
        "本层注意力系数污染了自己的注意力输入"
    )
    assert not np.allclose(base_ffn1, calls2[3][1]), "本层注意力系数没有喂本层 FFN 输入"

    # 本层 FFN 系数变了：自己的 FFN 输入不变，但下一块注意力拿到的 pre_mix 必须变
    model2, cfg2 = _build()
    ids2 = mx.random.randint(0, cfg2.vocab_size, (1, 12))
    a, _ = _trace(model2, ids2)
    ffn_hc = model2.model.layers[1].ffn_hc
    ffn_hc.fn.weight = ffn_hc.fn.weight + 0.7
    b, _ = _trace(model2, ids2)
    assert np.allclose(a[3][1], b[3][1], atol=1e-6), (
        "本层 FFN 系数影响了自己的 FFN 输入（应只喂下一块）"
    )
    assert not np.allclose(a[4][1], b[4][1]), "本层 FFN 系数没有传给下一块的注意力输入"


def test_comb_is_doubly_stochastic_and_streams_are_hc_wide():
    model, cfg = _build()
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))
    layer = model.model.layers[0]
    h = model.model.embed(ids)
    h = mx.repeat(h[:, :, None, :], cfg.hc_mult, axis=2)
    pre, post, comb = layer.attn_hc.mixes(h)
    mx.eval(pre, post, comb)
    assert pre.shape[-1] == cfg.hc_mult and post.shape[-1] == cfg.hc_mult
    assert comb.shape[-2:] == (cfg.hc_mult, cfg.hc_mult)
    c = np.array(comb)
    assert np.allclose(c.sum(-1), 1.0, atol=1e-3), "comb 行和不为 1"
    assert np.allclose(c.sum(-2), 1.0, atol=1e-3), "comb 列和不为 1"
