"""CSA2 注意力语义：滑窗因果 / 压缩可见性 / top-k 稀疏 / 三种模式 / 分层索引 / CED。

验证的语义（报告章节）：
- §2.1：每层 = SWA（窗口内原始 KV）+ global compressed attention。
- §2.2 CED：解码层（l ≥ L/2）的全局 KV 不由本层 hidden 产生，而由第 L/2 层
  的 hidden 投影（式 1）；SWA 仍是逐层的。
- §2.3 CSA2：压缩率 r 把每 r 个 token 池化成一个 main KV；query 只看
  "可达"的压缩组（组 j 在 (j+1)·r-1 处才凑满）；indexer 为每个 query 选
  Top-K 个压缩位置，未被选中的压缩 KV 对输出零贡献。
- §2.3.1：Full 自产 KV+indexer；Reindex 复用 KV、重算 Top-K；Reuse 两者都复用。
- §2.3.2：分层稀疏索引——第一个 Full 层的候选块池是更深 indexer 的搜索域上界。

所有断言都用"手算参照"：扰动某个 token 的 hidden，逐位置比对变化集合。
运行：.venv/bin/python -m pytest tests/test_v41_attention.py -q
"""

import numpy as np
import pytest

from _v41_common import (
    assert_changed_exactly, build, ced_model, cfg_ced, cfg_mix, cfg_tiny, mix_model, row_diffs,
)
from model.attention import Attention, _topk_masks, select_candidate_blocks
from model.cache import SharedAttnState, VibyCache
from model.hc import identity_pre_mix, hc_pre
from model.config import VibyConfig
from model.rope import precompute_freqs_cis

import mlx.core as mx


# ------------------------------------------------------------------ 滑窗

def _sliding_attn(window=4):
    cfg = cfg_tiny(window_size=window)
    mx.random.seed(11)
    return cfg, Attention(cfg, 0)


def test_sliding_window_sees_exactly_last_window_tokens():
    """位置 t 只能看到 [t-window+1, t]：扰动 p 恰好影响 t∈[p, p+window-1]。"""
    W, T = 4, 24
    cfg, attn = _sliding_attn(W)
    mx.random.seed(12)
    x = mx.random.normal((1, T, cfg.dim))
    y = attn(x, 0, SharedAttnState(), None)
    mx.eval(y)
    for p in (0, 5, 12, 23):
        xp = x.at[0, p, :].add(0.5)
        yp = attn(xp, 0, SharedAttnState(), None)
        mx.eval(yp)
        expected = [t for t in range(T) if p <= t <= min(p + W - 1, T - 1)]
        assert_changed_exactly(y, yp, expected)


def test_sliding_window_is_causal():
    """因果性：未来 token 的改动对过去位置完全无影响。"""
    W, T = 8, 32
    cfg, attn = _sliding_attn(W)
    mx.random.seed(13)
    x = mx.random.normal((1, T, cfg.dim))
    y = attn(x, 0, SharedAttnState(), None)
    mx.eval(y)
    xp = x.at[0, 20, :].add(1.0)
    yp = attn(xp, 0, SharedAttnState(), None)
    mx.eval(yp)
    d = row_diffs(y, yp)[0]
    assert (d[:20] < 1e-6).all(), d[:20]
    assert (d[20:28] > 1e-9).all()          # 窗口内（含自己）
    assert (d[28:] < 1e-6).all()            # 超出窗口


# ------------------------------------------------------------------ 压缩可见性

def test_compressed_group_visible_from_group_end():
    """压缩组 j 只对 t ≥ (j+1)·r-1 的 query 可见（组凑满之后）。"""
    R, W, T = 4, 2, 32
    cfg = cfg_tiny(compress_ratios=(0, 0, R, 0, 0), kv_source_layers=(2,),
                   index_source_layers=(2,), candidate_source_layer=-1, window_size=W,
                   index_topk=64)
    mx.random.seed(14)
    attn = Attention(cfg, 2)
    mx.random.seed(15)
    x = mx.random.normal((1, T, cfg.dim))
    y = attn(x, 0, SharedAttnState(), None)
    mx.eval(y)
    for p in (7, 8, 9, 11, 12):
        xp = x.at[0, p, :].add(0.7)
        yp = attn(xp, 0, SharedAttnState(), None)
        mx.eval(yp)
        group = p // R
        first_visible = (group + 1) * R - 1
        expected = {t for t in range(T) if p <= t <= min(p + W - 1, T - 1)}
        expected |= {t for t in range(T) if t >= first_visible}
        assert_changed_exactly(y, yp, expected)


def test_compressed_group_is_average_of_ratio_tokens():
    """压缩器把 r 个 token 归一化池化成一个 latent（组首位置 = j·r）。"""
    from model.attention import Compressor

    cfg = cfg_tiny(compress_ratios=(0, 0, 4, 0, 0), kv_source_layers=(2,),
                   index_source_layers=(2,), candidate_source_layer=-1)
    mx.random.seed(16)
    comp = Compressor(cfg, 2)
    assert comp.ratio == 4 and comp.wgate is not None
    x = mx.random.normal((1, 8, cfg.dim))
    latent, state, pos, first = comp(x, 0, None)
    mx.eval(latent)
    assert latent.shape == (1, 2, cfg.head_dim)
    assert first == 0
    assert np.asarray(pos).tolist() == [[0, 4]]     # 组首绝对位置
    # 池化权重 = 组内 softmax(wgate·x)，手算对照（同一个 RMSNorm 再走一遍）
    kv, sc = comp.wkv(x), comp.wgate(x)
    w = mx.softmax(sc.reshape(1, 2, 4, cfg.head_dim), axis=2)
    expect = comp.norm((kv.reshape(1, 2, 4, cfg.head_dim) * w).sum(axis=2))
    mx.eval(expect)
    assert np.allclose(np.asarray(latent), np.asarray(expect), atol=2e-6),         np.abs(np.asarray(latent) - np.asarray(expect)).max()
    # ratio=1（CED 解码段的全局源）退化成一次普通投影
    cfg1 = cfg_tiny(compress_ratios=(0, 0, 1, 1, 0), kv_source_layers=(2,),
                    index_source_layers=(2,), candidate_source_layer=-1)
    c1 = Compressor(cfg1, 2)
    lat1, st1, pos1, _ = c1(x, 0, None)
    assert st1 is None and c1.wgate is None
    assert lat1.shape == (1, 8, cfg1.head_dim) and np.asarray(pos1).tolist() == [list(range(8))]


# ------------------------------------------------------------------ top-k 稀疏

def test_topk_masks_select_exactly_k_by_score():
    """_topk_masks 手算参照：恰好 k 个、与 numpy top-k 同集合、按位置排序。"""
    rng = np.random.RandomState(0)
    N, k = 9, 3
    score = mx.array(rng.randn(1, 4, N).astype(np.float32))
    reach = mx.array(np.ones((1, 4, N), dtype=bool))
    keep, idx = _topk_masks(score, reach, k, 0)
    mx.eval(keep, idx)
    assert np.asarray(mx.sum(keep, -1)).tolist() == [[k] * 4]
    got = np.asarray(idx)[0]
    want = np.sort(np.argsort(-np.asarray(score)[0], axis=-1)[:, :k], axis=-1)
    assert (got == want).all(), (got, want)
    # 可达位置不足 k：全部可达位置入选（索引 -1 表示无效槽）
    reach2 = mx.array((np.arange(N)[None, None, :] < 2) & np.ones((1, 4, N), bool))
    score2 = mx.where(reach2, score, mx.array(-1e30))
    keep2, idx2 = _topk_masks(score2, reach2, k, 7)
    mx.eval(keep2, idx2)
    assert np.asarray(mx.sum(keep2, -1)).tolist() == [[2] * 4]
    assert (np.asarray(idx2)[0][:, :2] == np.array([0, 1]) + 7).all()
    assert (np.asarray(idx2)[0][:, 2] == -1).all()
    # k=0 / N=0 边界
    keep0, idx0 = _topk_masks(score[:, :, :0], reach[:, :, :0], k, 0)
    assert keep0.shape[-1] == 0 and idx0.shape[-1] == 0


def test_sparse_selection_is_per_query_and_bounded():
    """每个 query 只关注 index_topk 个压缩位置（含 reach 截断）。"""
    cfg = cfg_tiny(compress_ratios=(0, 0, 1, 0, 0), kv_source_layers=(2,),
                   index_source_layers=(2,), candidate_source_layer=-1, index_topk=3)
    mx.random.seed(17)
    attn = Attention(cfg, 2)
    T = 10
    x = mx.random.normal((1, T, cfg.dim))
    shared = SharedAttnState()
    attn(x, 0, shared, None)
    mx.eval(shared.keep_mask)
    counts = np.asarray(mx.sum(shared.keep_mask, -1))[0]
    assert counts.tolist() == [min(3, t + 1) for t in range(T)], counts


def test_non_selected_compressed_kv_does_not_change_output():
    """index_topk 之外的压缩 KV 改动 → 输出 bit-level 不变。"""
    cfg = cfg_tiny(compress_ratios=(0, 0, 1, 1, 0), kv_source_layers=(2,),
                   index_source_layers=(2,), candidate_source_layer=-1, index_topk=3)
    mx.random.seed(18)
    reuse = Attention(cfg, 3)              # Reuse 层：直接用上游的 keep_mask 与 KV 池
    assert reuse.mode == "reuse" and reuse.indexer is None
    T, N = 8, 8
    x = mx.random.normal((1, T, cfg.dim))
    mx.random.seed(19)
    pool = mx.random.normal((1, N, cfg.head_dim))
    sel = np.zeros((T, N), bool)
    for t in range(T):
        sel[t, : min(3, t + 1)] = True
    mask = mx.array(sel[None])

    def run(p):
        sh = SharedAttnState()
        sh.compress_kv, sh.keep_mask = p, mask
        y = reuse(x, 0, sh, None)
        mx.eval(y)
        return y

    base = run(pool)
    never = int(np.flatnonzero(~sel.any(axis=0))[0])
    other = run(pool.at[0, never, :].add(3.0))
    assert bool(mx.all(base == other).item()), "未被任何 query 选中的压缩 KV 必须零影响"
    selected = run(pool.at[0, 2, :].add(3.0))
    assert float(mx.max(mx.abs(base - selected))) > 1e-6, "被选中的压缩 KV 必须影响输出"


# ------------------------------------------------------------------ CSA2 三模式

def _layer_outputs(model, layer_idx, x, shared=None):
    sh = shared or SharedAttnState()
    y = model.model.layers[layer_idx].attn(x, 0, sh, None)
    mx.eval(y)
    return y, sh


def test_full_layer_owns_compressor_and_indexer():
    """Full：自产压缩 KV 与 index K，并产生新的 Top-K。"""
    cfg = cfg_mix()
    model = mix_model()
    assert cfg.layer_mode(3) == "full"
    attn = model.model.layers[3].attn
    assert attn.compressor is not None and attn.indexer is not None
    assert attn.indexer.owns_k is True
    x = mx.random.normal((1, 12, cfg.dim))
    _, sh = _layer_outputs(model, 3, x)
    assert sh.compress_kv is not None and sh.compress_kv.shape[0] == 1
    assert sh.index_k is not None and sh.keep_mask is not None
    if mx.default_device() == mx.gpu:
        assert sh.sparse_selection is not None
        ratio, (idx, lens) = sh.sparse_selection
        assert ratio == 1
        assert idx.ndim == 2 and lens.ndim == 1


def test_reindex_reuses_upstream_kv_but_reselects_topk():
    """Reindex：KV/index K 来自上游，改上游 KV 会改变它自己的 Top-K 选择。"""
    cfg = cfg_mix(index_topk=2)
    model = mix_model(index_topk=2)
    layer = model.model.layers[5]
    assert cfg.layer_mode(5) == "reindex"
    assert layer.attn.compressor is None and layer.attn.indexer is not None
    assert layer.attn.indexer.owns_k is False
    x = mx.random.normal((1, 16, cfg.dim))

    def run():
        sh = SharedAttnState()
        _layer_outputs(model, 3, x, sh)          # 上游 Full 层写 KV/index K/候选池
        y, sh = _layer_outputs(model, 5, x, sh)  # Reindex 层重算 top-k
        return y, sh

    y0, sh0 = run()
    keep0 = np.asarray(sh0.keep_mask)[0].copy()
    # 改上游压缩器权重 → 上游 KV 变 → reindex 层重算出的 top-k 必须跟着变
    from _v41_common import perturbed

    with perturbed(model, "model.layers.3.attn.compressor.wkv.weight", 0.3):
        y1, sh1 = run()
    keep1 = np.asarray(sh1.keep_mask)[0]
    assert (keep0 != keep1).any(), "reindex 层应对上游 KV 变化重算 top-k"
    assert float(mx.max(mx.abs(y0 - y1))) > 1e-6


def test_reuse_layer_has_no_indexer_and_follows_upstream_selection():
    """Reuse：没有自己的 indexer 参数，Top-K 直接沿用上游 indexer 的结果。"""
    cfg = cfg_mix(index_topk=2)
    model = mix_model(index_topk=2)
    assert cfg.layer_mode(4) == "reuse"
    attn = model.model.layers[4].attn
    assert attn.indexer is None and attn.compressor is None
    x = mx.random.normal((1, 16, cfg.dim))
    from _v41_common import perturbed

    def run():
        sh = SharedAttnState()
        _layer_outputs(model, 3, x, sh)
        y, sh = _layer_outputs(model, 4, x, sh)
        return y, sh

    y0, sh0 = run()
    keep0 = np.asarray(sh0.keep_mask)[0].copy()
    # 改上游 indexer 的 query 投影 → 上游 top-k 变 → reuse 层跟着变（本层无 indexer 参数）
    with perturbed(model, "model.layers.3.attn.indexer.wq_b.weight", 0.3):
        y1, sh1 = run()
    assert (keep0 != np.asarray(sh1.keep_mask)[0]).any()
    assert float(mx.max(mx.abs(y0 - y1))) > 1e-6
    # 改本层没有的 indexer/compressor 参数是不可能的（模块不存在），故只需检查 KV 复用：
    with perturbed(model, "model.layers.3.attn.compressor.wkv.weight", 0.3):
        y2, _ = run()
    assert float(mx.max(mx.abs(y0 - y2))) > 1e-6, "reuse 层仍复用上游压缩 KV"


# ------------------------------------------------------------------ 分层索引

def test_candidate_pool_is_upper_bound_for_deeper_indexer():
    """§2.3.2：候选块之外的压缩位置对更深的 indexer 不可见（keep ⊆ 候选池）。"""
    cfg = cfg_mix(candidate_topk_blocks=1, candidate_block_size=2, index_topk=4)
    model = mix_model(candidate_topk_blocks=1, candidate_block_size=2, index_topk=4)
    assert model.model.layers[5].attn.indexer.uses_candidates is True
    T = 16
    x = mx.random.normal((1, T, cfg.dim))
    sh = SharedAttnState()
    _layer_outputs(model, 3, x, sh)                     # 候选源层（Full）
    cand = np.asarray(sh.candidates)[0].copy()
    assert cand.sum() > 0
    _layer_outputs(model, 5, x, sh)                     # 更深 indexer（Reindex）
    keep = np.asarray(sh.keep_mask)[0]
    assert not (keep & ~cand).any(), "更深 indexer 选到了候选池之外的位置"
    # 非平凡性：候选源层自己的 Top-K 不受候选池限制（它是全局打分的那一层）
    sh_src = SharedAttnState()
    _layer_outputs(model, 3, x, sh_src)
    assert (np.asarray(sh_src.keep_mask)[0] & ~cand).any(), \
        "候选源层应全局选 Top-K（候选池只是给更深层用的）"


def test_candidate_blocks_pick_highest_block_scores():
    """select_candidate_blocks 手算参照：块得分 = 块内最大值，取最高 topk 块。"""
    rng = np.random.RandomState(3)
    N, BS, nb = 12, 4, 3
    logits = mx.array(rng.randn(1, 2, N).astype(np.float32))
    lens = mx.array([[N, N]])
    keep = select_candidate_blocks(logits, lens, topk_blocks=2, block_size=BS)
    mx.eval(keep)
    got = np.asarray(keep)[0, 0].reshape(nb, BS).all(axis=1)

    # 手算参照：块得分 = 块内最大 logit；最新块（下标 (len-1)//BS）钉成 +inf；
    # 取分最高的 topk 块（并列按下标取小）。
    def ref(row, length, topk):
        rm = np.asarray(row)[0][0]
        sc = rm.reshape(nb, BS).max(axis=1).copy()
        sc[(length - 1) // BS] = np.inf
        order = np.lexsort((np.arange(nb), -sc))
        sel = np.zeros(nb, bool)
        sel[order[:topk]] = True
        return sel

    want = ref(logits, N, 2)
    assert (got == want).all(), (got, want)
    # 块得分确实是"块内最大值"：块 0 只有一个峰、块 1 整体更高，块 0 仍应
    # 凭借自己的最大值挤进 top-2（另一个名额被最新块钉住）
    peak = mx.array(np.full((1, 1, N), -50.0, dtype=np.float32))
    peak = peak.at[0, 0, 1].add(100.0)          # 块 0 的唯一峰
    peak = peak.at[0, 0, 4:8].add(-1.0)         # 块 1 整体更高但没有峰
    got2 = np.asarray(select_candidate_blocks(peak, mx.array([[N]]), 2, BS))[0, 0]
    assert got2.reshape(nb, BS).all(axis=1).tolist() == [True, False, True]
    # 最新（未填满）的块必须钉住，即使它分数最低
    low_last = mx.array(np.full((1, 1, N), -50.0, dtype=np.float32))
    low_last = low_last.at[0, 0, 4].add(100.0)  # 块 1 分数最高，但块 2 是最新块
    got3 = np.asarray(select_candidate_blocks(low_last, mx.array([[9]]), 1, BS))[0, 0]
    assert got3[8:9].all() and got3.reshape(nb, BS).all(axis=1).tolist()[2]


# ------------------------------------------------------------------ CED

def _manual_forward(model, ids, cache, perturb_before=None, delta=0.3):
    """复刻 VibyModel.__call__ 的层循环，可在某一层入口扰动 hidden。"""
    cfg = model.config
    h = model.model.embed(ids)
    h = mx.repeat(h[:, :, None, :], cfg.hc_mult, axis=2)
    shared = SharedAttnState()
    pre_mix = identity_pre_mix(h, cfg.hc_mult)
    for i, layer in enumerate(model.model.layers):
        if perturb_before is not None and i == perturb_before:
            h = h + delta
        h, pre_mix = layer(h, 0, pre_mix, shared, cache[i], None, None, False)
    return hc_pre(h, pre_mix)


def test_ced_decoder_compress_kv_comes_only_from_boundary_layer():
    """§2.2 式(1)：扰动边界层之后的 hidden 不改变解码层的压缩 KV。"""
    cfg = cfg_ced()
    model = ced_model()
    mid = cfg.n_encoder_layers
    assert mid == 3
    # 结构：0-based 的 mid = 第一个解码层（报告里 1-based 的 L/2+1）；它的
    # compressor 读的是编码器末层输出 H_{L/2}，其后的解码层不再有压缩器，
    # 缓存直接指向它的池。
    cache = VibyCache(cfg, 1)
    for l in range(mid + 1, cfg.n_layers):
        assert model.model.layers[l].attn.compressor is None
        assert cache.layers[l].src_cache is cache.layers[mid]
    assert model.model.layers[mid].attn.compressor is not None
    assert cache.layers[mid].src_cache is cache.layers[mid]
    # T=8：MoE 走稠密路径（B·T <= _DENSE_MAX_TOKENS），整段前向逐位可复现，
    # 于是"CED 不改变压缩 KV"可以断言成严格 0 而不是 1e-6 噪声。
    mx.random.seed(21)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))

    def pool_after(perturb_before):
        c = VibyCache(cfg, 1)
        _manual_forward(model, ids, c, perturb_before=perturb_before)
        mx.eval(c.layers[mid].compress_kv)
        return np.asarray(c.layers[mid].compress_kv)

    base = pool_after(None)
    assert np.abs(base - pool_after(None)).max() == 0.0, "同输入两次前向应逐位一致"
    after_boundary = pool_after(mid + 1)      # 扰动解码段入口的 hidden
    assert np.abs(base - after_boundary).max() == 0.0, \
        "解码层 hidden 不得进入全局压缩 KV（CED）"
    before_boundary = pool_after(mid)         # 扰动边界层入口 → 池必须变（非平凡）
    assert np.abs(base - before_boundary).max() > 1e-3


def test_ced_sliding_branch_still_layer_local():
    """§2.2：SWA 仍是逐层的（解码层用自己的 hidden），所以扰动解码层会改它的窗口 KV。"""
    cfg = cfg_ced()
    model = ced_model()
    mid = cfg.n_encoder_layers
    mx.random.seed(22)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))
    c0, c1 = VibyCache(cfg, 1), VibyCache(cfg, 1)
    _manual_forward(model, ids, c0, perturb_before=None)
    _manual_forward(model, ids, c1, perturb_before=mid + 1)
    w0, w1 = np.asarray(c0.layers[mid + 1].window), np.asarray(c1.layers[mid + 1].window)
    assert np.abs(w0 - w1).max() > 1e-6, "解码层的 SWA KV 必须由本层 hidden 产生"


# ------------------------------------------------------------------ RoPE 常量回归

def test_rope_tables_are_not_randomized_by_init():
    """回归：apply_trunc_normal_init 不得覆盖 freq_cos/freq_sin（RoPE 是常量）。"""
    cfg = cfg_tiny(max_seq_len=64)
    mx.random.seed(20)
    from model.model import VibyForCausalLM

    model = VibyForCausalLM(cfg)          # skip_init=False：走截断正态初始化
    for i, ratio in enumerate([cfg.compress_ratios[j] for j in range(cfg.n_layers)]):
        attn = model.model.layers[i].attn
        seq, theta = (cfg.original_seq_len, cfg.compress_rope_theta) if ratio else (0, cfg.rope_theta)
        cos, sin = precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, seq, theta,
                                        cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        mx.eval(cos, sin)
        assert float(mx.max(mx.abs(attn.freq_cos - cos))) == 0.0
        assert float(mx.max(mx.abs(attn.freq_sin - sin))) == 0.0
    from mlx.utils import tree_flatten

    trainable = {k for k, _ in tree_flatten(model.trainable_parameters())}
    assert not any("freq_cos" in k or "freq_sin" in k for k in trainable)
    params = {k for k, _ in tree_flatten(model.parameters())}
    assert any("freq_cos" in k for k in params)


# ------------------------------------------------------- 训练路径分块 ≡ 稠密

def test_chunked_window_matches_dense_path(monkeypatch):
    """分块滑窗路径必须与"全 T + 掩码"的稠密路径逐位一致（同输入同权重）。

    分块只是把不可见的 key 从 sdpa 的 S 维里拿掉（窗口 W 内的 key 必然落在
    相邻两块里），可见集完全相同，所以两条路径的输出应当相等。
    """
    import model.attention as attn_mod
    from model.attention import Attention

    W, T = 8, 32
    R = 4
    cfg = cfg_tiny(
        compress_ratios=(0, 0, R, R, 0),
        kv_source_layers=(2,),
        index_source_layers=(2, 3),
        candidate_source_layer=2,
        window_size=W,
        index_topk=8,
    )
    mx.random.seed(21)
    attn = Attention(cfg, 3)
    mx.random.seed(22)
    x = mx.random.normal((2, T, cfg.dim))
    seg = mx.repeat(mx.arange(T)[None, :] // (T // 2), 2, axis=0).astype(mx.int32)
    pad = mx.ones((2, T), dtype=mx.bool_)

    monkeypatch.setattr(attn_mod, "_CHUNK_ENABLED", True)
    a = attn(x, 0, SharedAttnState(), None, seg, pad)
    mx.eval(a)
    monkeypatch.setattr(attn_mod, "_CHUNK_ENABLED", False)
    b = attn(x, 0, SharedAttnState(), None, seg, pad)
    mx.eval(b)
    assert a.shape == b.shape
    assert mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item() < 1e-5
