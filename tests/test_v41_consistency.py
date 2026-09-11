"""一致性回归：prefill ↔ 分块 prefill ↔ 逐 token decode / packed / pad。

验证的语义（报告章节）：
- §2.3 CSA2 的压缩 KV 池是"流式"的：整段 prefill 与 prefix+逐 token decode
  必须给出同一组 logits（压缩器 kv_state 的分组对齐、窗口环、top-k 索引在
  两条路径上语义一致）。
- §2.2 CED：解码层的全局 KV 由边界层缓存共享，decode 时只能读同一份池。
- §2.1 packed 训练：segment_ids 让跨文档不可见；右 padding 下有效位置与
  无 padding 完全一致。

噪声说明：MoE 的稀疏路径（B·T > 8）用 mx.gather_mm + argsort，Metal 上
逐次运行有 ~1e-8 的累积顺序差异，经多层放大到 ~1e-6（API.md 记录的
"fp32 噪声级"）；所以模型级比较用 1e-5 容差，逐位相等只在注意力级断言。

运行：.venv/bin/python -m pytest tests/test_v41_consistency.py -q
"""

import numpy as np
import pytest

from _v41_common import (
    build, cfg_ced, cfg_mix, cfg_tiny, ced_model, engram_model, max_abs_diff, mix_model,
    tiny_model,
)
from model.cache import VibyCache

import mlx.core as mx

TOL = 1e-5


def _logits(model, ids, **kw):
    out = model(ids, use_mtp=False, **kw)
    mx.eval(out.logits)
    return out.logits


# ------------------------------------------------------------------ prefill/decode

def test_prefill_equals_prefix_plus_token_decode():
    """整段 prefill == prefix prefill + 逐 token decode（tiny 配置）。"""
    model = tiny_model()
    mx.random.seed(30)
    ids = mx.random.randint(0, model.config.vocab_size, (1, 24))
    full = _logits(model, ids)
    P = 9
    lg, cache = model.prefill(ids[:, :P])
    outs = [lg]
    for t in range(P, ids.shape[1]):
        step, cache = model.decode_step(ids[:, t], cache)
        outs.append(step[:, None, :])
    dec = mx.concatenate(outs, axis=1)
    mx.eval(dec)
    assert cache.start_pos == ids.shape[1]
    d = max_abs_diff(full, dec)
    assert d < TOL, d


def test_prefill_equals_decode_with_ratio2_and_reindex():
    """含 ratio=2 压缩器 + Reindex + 候选池的配置同样一致。"""
    model = mix_model()
    cfg = model.config
    assert 2 in cfg.compress_ratios and "reindex" in [cfg.layer_mode(i) for i in range(cfg.n_layers)]
    mx.random.seed(31)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 26))
    full = _logits(model, ids)
    P = 11
    lg, cache = model.prefill(ids[:, :P])
    outs = [lg]
    for t in range(P, ids.shape[1]):
        step, cache = model.decode_step(ids[:, t], cache)
        outs.append(step[:, None, :])
    dec = mx.concatenate(outs, axis=1)
    mx.eval(dec)
    d = max_abs_diff(full, dec)
    assert d < TOL, d


def test_chunked_prefill_equals_single_pass():
    """分块 prefill（第二块带 start_pos）== 整段 prefill：压缩器跨块续组。"""
    model = mix_model()
    cfg = model.config
    mx.random.seed(32)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 20))
    full = _logits(model, ids)
    cache = VibyCache(cfg, 1)
    outs = []
    for s, e in ((0, 7), (7, 13), (13, 20)):   # 块边界刻意不落在 ratio 上
        out = model(ids[:, s:e], start_pos=s, cache=cache, use_mtp=False)
        mx.eval(out.logits)
        outs.append(out.logits)
    chunked = mx.concatenate(outs, axis=1)
    mx.eval(chunked)
    d = max_abs_diff(full, chunked)
    assert d < TOL, d


def test_batch_decode_matches_single_sequence():
    """同位置批量解码 == 每条序列单独解码。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(33)
    ids = mx.random.randint(0, cfg.vocab_size, (3, 12))
    P = 6
    lg, cache = model.prefill(ids[:, :P])
    batch, _ = model.decode_step(ids[:, P], cache)
    mx.eval(batch)
    singles = []
    for b in range(3):
        _, c = model.prefill(ids[b:b + 1, :P])
        s, _ = model.decode_step(ids[b:b + 1, P], c)
        mx.eval(s)
        singles.append(s)
    single = mx.concatenate(singles, axis=0)
    d = max_abs_diff(batch, single)
    assert d < TOL, d


def test_decode_cache_grows_and_logits_are_causal():
    """decode 第 t 步的 logits 只由前 t+1 个 token 决定（改后面的 token 不影响）。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(34)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 10))
    lg, cache = model.prefill(ids[:, :4])
    step1, cache = model.decode_step(ids[:, 4], cache)
    mx.eval(step1)
    # 换掉最后一个 token → 该步 logits 必须变
    _, cache2 = model.prefill(ids[:, :4])
    other = ids.at[:, 4].add(1)
    step2, _ = model.decode_step(other[:, 4], cache2)
    mx.eval(step2)
    assert max_abs_diff(step1, step2) > 1e-3


# ------------------------------------------------------------------ packed / pad

def test_packed_segments_isolate_documents():
    """packed 文档隔离：改一个文档的 token 不影响另一个文档的 logits。

    文档 A（位置 0..4）在 packed 里的绝对位置与单独跑一致，所以还能直接与
    单独跑对齐；文档 B 的 RoPE 位置被前面 A 平移了，只能做"互不影响"检查。
    第 5 个位置正好是跨文档的压缩组（ratio=2 时组 (4,5)），用来验证该组作废。
    """
    model = mix_model()
    cfg = model.config
    mx.random.seed(35)
    a = mx.random.randint(0, cfg.vocab_size, (1, 5))
    b = mx.random.randint(0, cfg.vocab_size, (1, 11))
    seg = mx.concatenate([mx.zeros((1, 5), dtype=mx.int32), mx.ones((1, 11), dtype=mx.int32)], axis=1)
    packed = mx.concatenate([a, b], axis=1)
    lg = _logits(model, packed, segment_ids=seg)
    assert max_abs_diff(lg[:, :5], _logits(model, a)) < TOL      # A 段位置不变
    # 改 B → A 不变，B 自己变
    b2 = mx.random.randint(0, cfg.vocab_size, (1, 11))
    lg_b2 = _logits(model, mx.concatenate([a, b2], axis=1), segment_ids=seg)
    assert max_abs_diff(lg[:, :5], lg_b2[:, :5]) < TOL
    assert max_abs_diff(lg[:, 5:], lg_b2[:, 5:]) > 1e-3
    # 改 A → B 不变
    a2 = mx.random.randint(0, cfg.vocab_size, (1, 5))
    lg_a2 = _logits(model, mx.concatenate([a2, b], axis=1), segment_ids=seg)
    assert max_abs_diff(lg[:, 5:], lg_a2[:, 5:]) < TOL
    assert max_abs_diff(lg[:, :5], lg_a2[:, :5]) > 1e-3
    # 非平凡：不做隔离时，改 B 会通过窗口/压缩组影响 A
    loose = _logits(model, packed)
    loose_b2 = _logits(model, mx.concatenate([a, b2], axis=1))
    assert max_abs_diff(loose[:, :5], loose_b2[:, :5]) > 1e-6


def test_pad_mask_keeps_valid_positions_identical():
    """右 padding：有效位置与无 padding 完全一致（pad 不参与注意力）。"""
    model = tiny_model()
    cfg = model.config
    mx.random.seed(36)
    ids = mx.random.randint(0, cfg.vocab_size, (2, 8))
    am = mx.ones((2, 8), dtype=mx.int32)
    lg = _logits(model, ids, attention_mask=am)
    pad_ids = mx.full((2, 4), cfg.pad_token_id, dtype=mx.int32)      # 后面接 4 个 pad
    padded = mx.concatenate([ids, pad_ids], axis=1)
    am2 = mx.concatenate([am, mx.zeros((2, 4), dtype=mx.int32)], axis=1)
    lg2 = _logits(model, padded, attention_mask=am2)
    d = max_abs_diff(lg, lg2[:, :8])
    assert d < TOL, d
    # 非平凡：无 mask 时 pad 自己会进注意力，pad 段 logits 必须变。
    # 有效前缀是因果的，后面的 pad 本来就看不见，前 8 位不应作为判据。
    lg3 = _logits(model, padded)
    assert max_abs_diff(lg2[:, 8:], lg3[:, 8:]) > 1e-6


def test_segment_and_pad_together():
    """packed + pad 同时开：有效段仍与单独跑一致。"""
    model = ced_model()
    cfg = model.config
    mx.random.seed(37)
    a = mx.random.randint(0, cfg.vocab_size, (1, 6))
    pad = mx.full((1, 2), cfg.pad_token_id, dtype=mx.int32)
    packed = mx.concatenate([a, pad], axis=1)
    am = mx.concatenate([mx.ones((1, 6), dtype=mx.int32), mx.zeros((1, 2), dtype=mx.int32)], axis=1)
    lg = _logits(model, packed, attention_mask=am)
    lg_a = _logits(model, a)
    assert max_abs_diff(lg[:, :6], lg_a) < TOL


# ------------------------------------------------------------------ Engram 路径

def test_engram_prefill_decode_and_chunking_agree():
    """Engram 开启时：整段 prefill == prefix+decode == 分块 prefill。

    这条覆盖 cache.engram_prev 的线程化（prefill 写最近 max_ngram-1 个 token，
    decode_step 逐步右移），n-gram 哈希必须与整段一致。
    """
    model = engram_model()
    cfg = model.config
    assert model.model.engram_layout is not None
    mx.random.seed(38)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 18))
    full = _logits(model, ids)
    P = 7
    lg, cache = model.prefill(ids[:, :P])
    outs = [lg]
    for t in range(P, ids.shape[1]):
        step, cache = model.decode_step(ids[:, t], cache)
        outs.append(step[:, None, :])
    dec = mx.concatenate(outs, axis=1)
    mx.eval(dec)
    assert max_abs_diff(full, dec) < TOL
    # 分块 prefill：第二块要手动带上 prev_tokens（哈希窗口不能跨块丢历史）
    prev = ids[:, P - (cfg.engram_max_ngram_size - 1):P]
    out2 = model(ids[:, P:], start_pos=P, cache=model.prefill(ids[:, :P])[1],
                 prev_tokens=prev, use_mtp=False)
    mx.eval(out2.logits)
    assert max_abs_diff(full[:, P:], out2.logits) < TOL


# ------------------------------------------------------------------ 连续 batch

@pytest.mark.parametrize("l1,l2", [(12, 17), (10, 16), (12, 12), (13, 13)])
def test_continuous_batch_decode_with_unequal_lengths(l1, l2):
    """连续 batch：batch 内各请求长度不同时的逐序列解码。

    每条请求先各自 prefill，再把 cache 合并成一条 batch cache，用
    start_pos=[L1,L2] 一次解码；必须与逐条单独解码一致（含 ratio=2 的
    压缩器分组状态在不同序列上进度不同的情况）。
    """
    from _v41_common import merge_caches

    cfg = cfg_mix(window_size=8, max_seq_len=64)
    assert 2 in cfg.compress_ratios          # 必须带 ratio=2：分组进度会不同步
    model = build(cfg)
    mx.random.seed(40)
    i1 = mx.random.randint(0, cfg.vocab_size, (1, l1 + 1))
    i2 = mx.random.randint(0, cfg.vocab_size, (1, l2 + 1))
    # 参考：单条 prefill + 单条 decode
    _, a1 = model.prefill(i1[:, :l1])
    s1, _ = model.decode_step(i1[:, l1], a1)
    _, a2 = model.prefill(i2[:, :l2])
    s2, _ = model.decode_step(i2[:, l2], a2)
    mx.eval(s1, s2)
    # 批量：必须在解码之前取 cache 快照（decode 原地推进 cache）
    _, b1 = model.prefill(i1[:, :l1])
    _, b2 = model.prefill(i2[:, :l2])
    bc = merge_caches(cfg, [b1, b2], max(l1, l2))
    tok = mx.concatenate([i1[:, l1], i2[:, l2]], axis=0)
    out = model(tok[:, None], start_pos=mx.array([l1, l2], dtype=mx.int32),
                cache=bc, decode=True, use_mtp=False)
    mx.eval(out.logits)
    got = out.logits[:, -1]
    assert max_abs_diff(got[0:1], s1) < TOL, (l1, l2)
    assert max_abs_diff(got[1:2], s2) < TOL, (l1, l2)


def test_decode_step_advances_cache_in_place():
    """坑 1：decode 会原地推进 cache（窗口/压缩池 setitem + start_pos 自增）。"""
    cfg = cfg_mix(window_size=8, max_seq_len=32)
    model = build(cfg)
    mx.random.seed(41)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 9))
    lg, cache = model.prefill(ids[:, :8])
    win_before = np.asarray(cache.layers[0].window).copy()
    pool_before = np.asarray(cache.layers[3].compress_kv).copy()
    model.decode_step(ids[:, 8], cache)
    assert cache.start_pos == 9
    assert not np.array_equal(np.asarray(cache.layers[0].window), win_before)
    assert not np.array_equal(np.asarray(cache.layers[3].compress_kv), pool_before)


def test_batch_decode_needs_engram_prev_tokens():
    """坑 2：Engram 的 prev_tokens 不随 cache 推导，批量解码必须自己拼。"""
    from _v41_common import merge_caches

    model = engram_model()
    cfg = model.config
    w = max(cfg.engram_max_ngram_size - 1, 1)
    l1, l2 = 9, 13
    mx.random.seed(42)
    i1 = mx.random.randint(0, cfg.vocab_size, (1, l1 + 1))
    i2 = mx.random.randint(0, cfg.vocab_size, (1, l2 + 1))
    _, a1 = model.prefill(i1[:, :l1])
    s1, _ = model.decode_step(i1[:, l1], a1)
    _, a2 = model.prefill(i2[:, :l2])
    s2, _ = model.decode_step(i2[:, l2], a2)
    mx.eval(s1, s2)
    _, b1 = model.prefill(i1[:, :l1])
    _, b2 = model.prefill(i2[:, :l2])
    bc = merge_caches(cfg, [b1, b2], max(l1, l2))
    tok = mx.concatenate([i1[:, l1], i2[:, l2]], axis=0)
    good = model(tok[:, None], start_pos=mx.array([l1, l2], dtype=mx.int32), cache=bc,
                 decode=True, prev_tokens=mx.concatenate([i1[:, l1 - w:l1], i2[:, l2 - w:l2]], axis=0),
                 use_mtp=False)
    mx.eval(good.logits)
    assert max_abs_diff(good.logits[:, -1][0:1], s1) < TOL
    assert max_abs_diff(good.logits[:, -1][1:2], s2) < TOL
    # 漏传 prev_tokens：走"无历史"分支，偏差远超 fp32 噪声
    bc2 = merge_caches(cfg, [b1, b2], max(l1, l2))
    bad = model(tok[:, None], start_pos=mx.array([l1, l2], dtype=mx.int32), cache=bc2,
                decode=True, use_mtp=False)
    mx.eval(bad.logits)
    assert max_abs_diff(bad.logits[:, -1][0:1], s1) > 1e-2

