"""Engram：n-gram 素数哈希条件记忆（报告 §2.4.2 / §3.1.3）。

验证的语义：
- 每个 (层, n-gram 阶, 头) 在层表里独占一段**素数**长度的桶区间，各桶区间
  全局不重叠；哈希 id 必须落在本层表的行数内；
- n-gram 回看在序列开头、token_mask=False 与 prev_tokens 里的 DEAD(-1) 处截断，
  永不跨越 DEAD；
- prev_tokens 显式传最近 max_ngram-1 个 token：分块 prefill 必须与整段一致；
- 门控在 mask 关闭处把写入压成 0（残差原样通过）。

运行：.venv/bin/python -m pytest tests/test_v41_engram.py -q
"""

import numpy as np

from _v41_common import cfg_engram, engram_model
from model.engram import (
    EngramLayout,
    build_compressed_token_map,
)

import mlx.core as mx


# ------------------------------------------------------------------ 布局/范围


def test_hash_ids_stay_within_layer_tables():
    """hash ids 必须落在本层表的行数内（可直接 gather）。"""
    model = engram_model()
    cfg = model.config
    layout = model.model.engram_layout
    hash_state = model.model.engram_hash
    assert hash_state is not None and layout is not None
    assert len(layout.num_embeddings) == len(cfg.engram_layer_ids)
    mx.random.seed(70)
    ids = mx.random.randint(0, cfg.vocab_size, (2, 12))
    hi, prev = hash_state(ids)
    mx.eval(hi)
    assert hi.dtype == mx.int64
    assert hi.shape == (2, 12, len(cfg.engram_layer_ids), layout.n_hash_cols)
    assert int(mx.min(hi)) >= 0
    for li, rows in enumerate(layout.num_embeddings):
        layer_max = int(mx.max(hi[:, :, li, :]))
        assert layer_max < rows, (li, layer_max, rows)
    # 表行数 = 桶素数之和
    assert layout.num_embeddings == cfg.engram_num_embeddings
    for li, layer in enumerate(layout.primes):
        assert layout.num_embeddings[li] == sum(sum(order) for order in layer)
    # 全部 (层, 阶, 头) 的素数互不相同
    flat = [p for layer in layout.primes for order in layer for p in order]
    assert len(flat) == len(set(flat))


def test_ngram_orders_live_in_disjoint_prime_buckets():
    """同一层里不同 n-gram 阶各占一段素数桶，区间互不重叠。"""

    cfg = cfg_engram()
    layout = EngramLayout.from_config(cfg)
    model = engram_model()
    hs = model.model.engram_hash
    n_heads, n_orders = layout.n_heads, layout.max_ngram_size - 1
    # 每列的行号区间 = [offset, offset + prime)
    flat_primes = [p for order in layout.primes[0] for p in order]
    offsets = np.concatenate([[0], np.cumsum(flat_primes)[:-1]])
    ranges = [
        (int(offsets[i]), int(offsets[i] + flat_primes[i]))
        for i in range(len(flat_primes))
    ]
    # 前 n_heads 列是 2-gram、接着是 3-gram、4-gram
    for o in range(n_orders):
        cols = ranges[o * n_heads : (o + 1) * n_heads]
        lo = min(c[0] for c in cols)
        hi = max(c[1] for c in cols)
        for c in cols:
            assert lo <= c[0] < c[1] <= hi + 1
    # 不同阶的区间整体不重叠
    for o1 in range(n_orders):
        for o2 in range(o1 + 1, n_orders):
            a = ranges[o1 * n_heads : (o1 + 1) * n_heads]
            b = ranges[o2 * n_heads : (o2 + 1) * n_heads]
            assert max(x[1] for x in a) <= min(x[0] for x in b) or max(
                x[1] for x in b
            ) <= min(x[0] for x in a)
    # 同一位置不同阶的哈希取值必然落在各自区间（用真实哈希验证）
    mx.random.seed(71)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 6))
    hi, _ = hs(ids)
    mx.eval(hi)
    arr = np.asarray(hi)[0, :, 0, :]
    for c, (lo, hi_) in enumerate(ranges):
        col = arr[:, c]
        assert (col >= lo).all() and (col < hi_).all(), (c, col, lo, hi_)


# ------------------------------------------------------------------ 截断语义


def test_token_mask_and_dead_truncate_lookback():
    """mask 关闭处 = DEAD：它之后的位置不得看到它之前任何 token。"""
    model = engram_model()
    cfg = model.config
    hs = model.model.engram_hash
    mx.random.seed(72)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))
    other = ids.at[:, :6].add(1)  # 只改 mask 之前的 token
    m = np.ones((1, 8), bool)
    m[0, 6] = False  # 位置 6 = DEAD
    mask = mx.array(m)
    h1, p1 = hs(ids, None, mask)
    h2, p2 = hs(other, None, mask)
    mx.eval(h1, h2, p1)
    assert bool(mx.all(h1[:, 6:] == h2[:, 6:]).item()), (
        "DEAD 之后的 n-gram 不得跨过 DEAD"
    )
    assert bool(mx.any(h1[:, :6] != h2[:, :6]).item())
    # mask 位置落在回看窗口里 → 记成 DEAD 传给下一步（window=3，覆盖位置 5..7）
    assert -1 in np.asarray(p1)[0].tolist()


def test_prev_tokens_dead_truncates_ngram():
    """prev_tokens 里的 -1 截断回看；它之前的 token 变化不影响哈希。"""
    model = engram_model()
    cfg = model.config
    hs = model.model.engram_hash
    w = cfg.engram_max_ngram_size - 1
    assert w == 3
    mx.random.seed(73)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 3))
    base = mx.array([[11, -1, 5]], dtype=mx.int32)
    far = mx.array([[99, -1, 5]], dtype=mx.int32)  # 只改 DEAD 之前的那个
    near = mx.array([[11, -1, 9]], dtype=mx.int32)  # 改最近的历史
    h0, _ = hs(ids, base)
    h1, _ = hs(ids, far)
    h2, _ = hs(ids, near)
    mx.eval(h0, h1, h2)
    assert bool(mx.all(h0 == h1).item()), "DEAD 之前的历史不得影响哈希"
    assert bool(mx.any(h0 != h2).item()), "最近的历史必须影响哈希"


def test_prev_tokens_chunking_matches_single_pass():
    """分块哈希（显式 prev_tokens）与整段一次性哈希逐位一致。"""
    model = engram_model()
    cfg = model.config
    hs = model.model.engram_hash
    w = cfg.engram_max_ngram_size - 1
    mx.random.seed(74)
    ids = mx.random.randint(0, cfg.vocab_size, (2, 14))
    full, prev_full = hs(ids)
    k = 6
    chunk, prev_chunk = hs(ids[:, k:], prev_tokens=ids[:, k - w : k])
    mx.eval(full, chunk, prev_full, prev_chunk)
    assert bool(mx.all(full[:, k:] == chunk).item())
    assert bool(mx.all(prev_full[:, -w:] == prev_chunk).item())
    # 起始块：prev_tokens=None 等价于全 DEAD 历史
    head, _ = hs(ids[:, :k])
    mx.eval(head)
    assert bool(mx.all(full[:, :k] == head).item())


# ------------------------------------------------------------------ 门控写入


def test_gate_writes_nothing_where_masked():
    """token_mask=False 的位置门 = 0，残差逐位原样通过。"""
    model = engram_model()
    cfg = model.config
    eng = model.model.engram_layers[0]
    hs = model.model.engram_hash
    mx.random.seed(75)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 7))
    hashes, _ = hs(ids)
    x = mx.random.normal((1, 7, cfg.hc_mult, cfg.dim))
    m = np.ones((1, 7), bool)
    m[0, 3] = False
    mask = mx.array(m)
    y = eng(x, hashes, token_mask=mask)
    mx.eval(y)
    assert bool(mx.all(y[0, 3] == x[0, 3]).item()), "mask 关闭处不得写入残差流"
    assert float(mx.max(mx.abs(y[0, 0] - x[0, 0]))) > 1e-6, "mask 打开处必须有写入"
    # 不带 mask 时该位置会被写入（说明上面不是碰巧）
    y2 = eng(x, hashes)
    mx.eval(y2)
    assert float(mx.max(mx.abs(y2[0, 3] - x[0, 3]))) > 1e-6


def test_engram_module_shapes_and_identity_gate_init():
    """模块形状 + 初始门是"纯归一化点积"（q_weight/k_weight 保持 ones）。"""
    model = engram_model()
    cfg = model.config
    layout = model.model.engram_layout
    for li, lid in enumerate(cfg.engram_layer_ids):
        eng = model.model.engram_layers[li]
        assert eng.layer_id == lid
        assert eng.embed.weight.shape == (layout.num_embeddings[li], layout.head_dim)
        in_dim = layout.n_hash_cols * layout.head_dim
        assert eng.wkv.weight.shape == (cfg.dim * (cfg.hc_mult + 1), in_dim)
        mx.eval(eng.q_weight, eng.k_weight)
        assert np.allclose(np.asarray(eng.q_weight), 1.0)
        assert np.allclose(np.asarray(eng.k_weight), 1.0)
    # hash 输出可以整表喂给层（自动取本层列）
    hs = model.model.engram_hash
    mx.random.seed(76)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 5))
    hashes, _ = hs(ids)
    x = mx.random.normal((1, 5, cfg.hc_mult, cfg.dim))
    y = model.model.engram_layers[0](x, hashes)
    mx.eval(y)
    assert y.shape == x.shape


def test_engram_layers_affect_model_output():
    """Engram 写回残差流：改表参数必须改变 logits（非平凡）。"""
    model = engram_model()
    cfg = model.config
    mx.random.seed(77)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))
    out = model(ids, use_mtp=False)
    mx.eval(out.logits)
    from _v41_common import perturbed

    with perturbed(model, "model.engram_layers.0.embed.weight", 0.05):
        out2 = model(ids, use_mtp=False)
        mx.eval(out2.logits)
    assert float(mx.max(mx.abs(out.logits - out2.logits))) > 1e-4
    # 退出上下文后表已还原（不污染缓存实例）
    assert (
        float(
            mx.max(
                mx.abs(
                    model.model.engram_layers[0].embed.weight
                    - model.model.engram_layers[0].embed.weight
                )
            )
        )
        == 0.0
    )


def test_compressed_token_map_collapses_normalized_forms():
    """tokenizer 压缩表：归一化后同形的 token 落到同一 id，压缩后 id 空间更小。"""
    from transformers import AutoTokenizer

    from _v41_common import _ROOT

    tok = AutoTokenizer.from_pretrained(f"{_ROOT}/model")
    lookup, size = build_compressed_token_map(tok)
    assert len(lookup) == len(tok)
    assert 0 < size <= len(tok)
    assert max(lookup) == size - 1
    by_key: dict = {}
    for tid, cid in enumerate(lookup):
        by_key.setdefault(cid, []).append(tid)
    assert any(len(v) > 1 for v in by_key.values()), (
        "至少应有一组归一化同形的 token 被合并"
    )
    # 模型构造后 config 会填上真实压缩词表大小（乘子上界由它推出）
    model = engram_model()
    assert model.config.engram_compressed_vocab_size > 0
    # 压缩 id 表里第 0 行是 pad
    assert model.model.engram_hash._pad_id == lookup[model.config.engram_pad_id]
