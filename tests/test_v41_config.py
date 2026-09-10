"""VibyConfig：DeepSeek-V4.1 缩放版的配方与自动推导。

验证的语义（报告章节）：
- §2.1 Overview：主干 40 层 = 20 层因果编码器 + 20 层解码器；"每层同时有
  global attention 与 SWA，只有最前面两层纯 SWA" → compress_ratios 的
  [0,0] + [2]*18 + [1]*20 形状，以及 CED 边界 = n_layers//2。
- §2.2 CED：解码层的全局 KV 由第 L/2 层 hidden 投影而来 → 边界层必须是
  kv_source，且解码段 ratio 与边界层一致（缩放版为 1）。
- §2.3 CSA2：逐层静态模式 Full / Reindex / Reuse 由 kv_source_layers 与
  index_source_layers 推导；压缩层必须有同 ratio 的上游源。
- §2.4.2 Engram：每个 (层, n-gram 阶, 头) 一张素数长度的桶表。
- §2.4.3 DSpark：draft 层用更窄的 MoE。

运行：.venv/bin/python -m pytest tests/test_v41_config.py -q
"""

import pytest

from _v41_common import build, cfg_ced, cfg_engram, cfg_mix, cfg_tiny
from model.config import VibyConfig, default_compress_ratios, default_source_layers
from model.engram import EngramLayout, _build_primes, compute_num_embeddings


# ------------------------------------------------------------------ 配方可构造

def test_default_recipe_is_about_1b():
    """默认 ≈1B 配方：12 层、96 专家 top-6，静态计数 ≈1e9 / 激活 ≈1.3e8。"""
    cfg = VibyConfig()
    assert cfg.n_layers == 12
    assert cfg.n_routed_experts == 96 and cfg.n_activated_experts == 6
    assert cfg.n_shared_experts == 1
    total = cfg.num_parameters()
    assert 0.8e9 < total < 1.3e9, total
    active = cfg.num_active_parameters()
    # 激活只按 top-k 专家算：必须远小于总量（V4.1-Flash 约 8B/552B 的比例也在此列）
    assert active < total * 0.25
    assert 0.5e8 < active < 2.5e8, active


def test_default_compress_ratio_schedule_matches_flash():
    """[0,0] + [2]*18 + [1]*20 + [0]*3：浅层纯 SWA、编码段 r=2、解码段 r=1。"""
    assert default_compress_ratios(40, 3) == [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
    cfg = VibyConfig()
    assert cfg.compress_ratios[:2] == (0, 0)          # 前两层只有 SWA
    assert set(cfg.compress_ratios[2:6]) == {2}       # 编码段
    assert set(cfg.compress_ratios[6:12]) == {1}      # 解码段
    assert cfg.compress_ratios[12:] == (0,)           # draft 层纯 SWA


def test_ced_boundary_is_first_decoder_and_kv_source():
    """CED：边界层 = n_layers//2，必须是 kv_source，且解码段 ratio 由它共享。"""
    for n in (4, 6, 12, 40):
        cfg = VibyConfig(n_layers=n, engram_layer_ids=())
        mid = n // 2
        assert cfg.n_encoder_layers == mid
        assert mid in cfg.kv_source_layers
        for i in range(mid, n):
            assert cfg.compress_ratios[i] == cfg.compress_ratios[mid]
            # §2.2：解码层的压缩 KV 全部来自边界层
            assert cfg.kv_source_of(i) == mid
        assert cfg.compress_ratios[mid] == 1


def test_source_layer_derivation():
    """kv/index/candidate 源的缩放规则：编码段 2 个 kv 源，解码段每 4 层一个 indexer。"""
    assert default_source_layers(12) == ([2, 6], [2, 6, 10], 6)
    assert default_source_layers(40) == ([2, 20], [2, 20, 24, 28, 32, 36], 20)
    # n_layers 很小时只有 CED 边界层一个源
    assert default_source_layers(4) == ([2], [2], 2)
    assert default_source_layers(6) == ([2, 3], [2, 3], 3)
    # 源必须递增、去重、不落在第 0 层
    for n in (4, 6, 12, 40):
        kv, idx, cand = default_source_layers(n)
        assert list(kv) == sorted(set(kv)) and list(idx) == sorted(set(idx))
        assert min(kv) > 0 and max(kv) < n
        assert set(kv) <= set(idx)
    cfg = VibyConfig()
    assert cfg.candidate_source_layer in cfg.index_source_layers


def test_tiny_preset_constructs_and_is_small():
    cfg = cfg_tiny()
    assert cfg.dim == 256 and cfg.n_layers == 4 and cfg.n_activated_experts == 4
    model = build(cfg)
    assert model.num_parameters() < 50_000_000, model.num_parameters()
    assert model.config.arch == "deepseek_v4_1"


# ------------------------------------------------------------------ 模式推导

def test_layer_modes_full_reindex_reuse_sliding():
    """§2.3.1：Full=自产 KV+indexer，Reindex=复用 KV/新算 top-k，Reuse=两者都复用。"""
    cfg = cfg_mix()
    assert [cfg.layer_mode(i) for i in range(6)] == [
        "sliding", "sliding", "full", "full", "reuse", "reindex",
    ]
    # ratio 0 → sliding（不参与压缩 KV 共享）
    assert cfg.compress_ratios[0] == 0 and cfg.compress_ratios[1] == 0
    # Reindex 层是 index 源但不是 kv 源
    assert cfg.layer_mode(5) == "reindex"
    assert 5 in cfg.index_source_layers and 5 not in cfg.kv_source_layers
    assert cfg.kv_source_of(5) == 3 and cfg.index_source_of(5) == 5
    # Reuse 层两个源都来自上游
    assert cfg.layer_mode(4) == "reuse"
    assert cfg.kv_source_of(4) == 3 and cfg.index_source_of(4) == 3


def test_compressor_and_indexer_ownership_matches_mode():
    """模块层面的模式落地：compressor / indexer 只在对应源层存在。"""
    from model.attention import Attention

    cfg = cfg_mix()
    att = [Attention(cfg, i) for i in range(6)]
    for i, mode in enumerate(["sliding", "sliding", "full", "full", "reuse", "reindex"]):
        assert att[i].mode == mode
        assert (att[i].compressor is not None) == (i in cfg.kv_source_layers)
        assert (att[i].indexer is not None) == (i in cfg.index_source_layers)
    # reuse 层既无 compressor 也无 indexer：它没有自己的压缩/索引参数
    assert att[4].compressor is None and att[4].indexer is None
    # reindex 层复用上游 KV，所以 indexer 不拥有 K（没有 wk / k_norm）
    assert att[5].indexer.owns_k is False
    assert not hasattr(att[5].indexer, "wk"), "复用上游 index K 的层不应有 wk/k_norm"
    assert att[2].indexer.owns_k is True and hasattr(att[2].indexer, "wk")
    # 候选池只由 candidate_source_layer 建立，更深的 indexer 才消费
    assert att[3].indexer.is_candidate_source is True
    assert att[5].indexer.uses_candidates is True
    assert att[2].indexer.uses_candidates is False


# ------------------------------------------------------------------ 非法配置

def test_invalid_compress_layer_without_upstream_source():
    """压缩层没有同 ratio 上游源 → 报错（§2.3 的跨层复用前提）。"""
    with pytest.raises(ValueError, match="没有上游 kv 源"):
        VibyConfig(
            preset="tiny", n_layers=6, engram_layer_ids=(),
            compress_ratios=(0, 0, 0, 2, 2, 2, 0), kv_source_layers=(4,),
            index_source_layers=(4,), candidate_source_layer=4,
        )


def test_invalid_ratio_mismatch_with_source():
    with pytest.raises(ValueError, match="不一致"):
        VibyConfig(
            preset="tiny", n_layers=6, engram_layer_ids=(),
            compress_ratios=(0, 0, 2, 1, 1, 1, 0), kv_source_layers=(2,),
            index_source_layers=(2,), candidate_source_layer=2,
        )


def test_invalid_source_lists_and_lengths():
    with pytest.raises(ValueError, match="严格递增"):
        VibyConfig(preset="tiny", engram_layer_ids=(), kv_source_layers=(2, 2))
    with pytest.raises(ValueError, match="第 0 层"):
        VibyConfig(preset="tiny", engram_layer_ids=(), kv_source_layers=(0,))
    with pytest.raises(ValueError, match="落在主干层内"):
        VibyConfig(preset="tiny", n_layers=4, engram_layer_ids=(), kv_source_layers=(4,))
    with pytest.raises(ValueError, match="compress_ratios 长度"):
        VibyConfig(preset="tiny", engram_layer_ids=(), compress_ratios=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="不能为负"):
        VibyConfig(preset="tiny", engram_layer_ids=(), compress_ratios=(0, 0, -1, 1, 0))
    with pytest.raises(ValueError, match="n_layers"):
        VibyConfig(preset="tiny", n_layers=2, engram_layer_ids=())


def test_invalid_moe_and_rope_configs():
    with pytest.raises(ValueError, match="n_activated_experts"):
        cfg_tiny(n_activated_experts=0)
    with pytest.raises(ValueError, match="n_activated_experts"):
        cfg_tiny(n_activated_experts=99)
    with pytest.raises(ValueError, match="共享专家"):
        cfg_tiny(n_shared_experts=2)
    with pytest.raises(ValueError, match="rope_head_dim"):
        cfg_tiny(rope_head_dim=0)
    with pytest.raises(ValueError, match="rope_head_dim"):
        cfg_tiny(rope_head_dim=65)
    with pytest.raises(ValueError, match="o_groups"):
        cfg_tiny(n_heads=4, o_groups=3)
    with pytest.raises(ValueError, match="hc_mult"):
        cfg_tiny(hc_mult=1)
    with pytest.raises(ValueError, match="score_func"):
        cfg_tiny(score_func="relu")
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        cfg_tiny(dspark_target_layer_ids=())
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        cfg_tiny(dspark_target_layer_ids=(99,))


# ------------------------------------------------------------------ Engram 布局

def test_engram_num_embeddings_derived_from_prime_buckets():
    """§2.4.2：每个 (层, 阶, 头) 独占一段素数桶，表行数 = 各桶素数之和。"""
    cfg = VibyConfig()
    rows = compute_num_embeddings(cfg)
    assert len(rows) == len(cfg.engram_layer_ids)
    primes = _build_primes(cfg.engram_layer_ids, cfg.engram_max_ngram_size,
                           cfg.engram_n_heads, cfg.engram_vocab_size)
    assert len(primes) == len(cfg.engram_layer_ids)
    for layer, n_orders in zip(primes, [cfg.engram_max_ngram_size - 1]):
        assert len(layer) == n_orders
    flat = [p for layer in primes for order in layer for p in order]
    assert len(flat) == len(set(flat)), "素数桶全局不重复"
    for layer_primes, n in zip(primes, rows):
        assert n == sum(sum(order) for order in layer_primes)
        assert all(p > cfg.engram_vocab_size - 1 for order in layer_primes for p in order)
    layout = EngramLayout.from_config(cfg)
    assert layout is not None and layout.num_embeddings == rows
    assert layout.window == cfg.engram_max_ngram_size - 1
    assert layout.n_hash_cols == layout.window * cfg.engram_n_heads
    # 没有 engram 层时布局为 None（与参考实现 from_config 一致）
    assert EngramLayout.from_config(cfg_tiny()) is None


def test_engram_layout_rejects_too_small_tables():
    cfg = cfg_engram()
    bad = cfg.engram_num_embeddings[0] - 1
    with pytest.raises(ValueError, match="小于桶总数"):
        EngramLayout.from_config(
            VibyConfig(**{**cfg.to_dict(), "engram_num_embeddings": (bad, cfg.engram_num_embeddings[1])})
        )


# ------------------------------------------------------------------ draft / 序列化

def test_draft_layers_use_narrow_moe():
    """§2.4.3：DSpark draft 层是 3 层 SWA block，MoE 比主干窄。"""
    cfg = VibyConfig(dspark_n_routed_experts=32, dspark_n_activated_experts=3)
    assert cfg.moe_of(0) == (cfg.n_routed_experts, cfg.n_activated_experts)
    assert cfg.moe_of(cfg.n_layers) == (32, 3)
    assert cfg.compress_ratios[cfg.n_layers:] == (0,)
    assert all(cfg.layer_mode(i) == "sliding" for i in range(cfg.n_layers, cfg.n_layers + cfg.n_mtp_layers))
    assert cfg.n_mtp_layers == 1 and cfg.dspark_block_size == 4


def test_config_dict_roundtrip(tmp_path=None):
    cfg = cfg_ced()
    data = cfg.to_dict()
    assert data["model_type"] == "viby" and data["arch"] == "deepseek_v4_1"
    assert isinstance(data["compress_ratios"], list)
    back = VibyConfig.from_dict(data)
    assert back.to_dict() == data
    assert back.compress_ratios == cfg.compress_ratios
    assert back.kv_source_layers == cfg.kv_source_layers


def test_static_parameter_count_matches_model():
    """config.num_parameters() 与 model.num_parameters() 同量级（±2%）。"""
    for cfg in (cfg_tiny(), cfg_mix(), cfg_ced()):
        model = build(cfg)
        static, real = cfg.num_parameters(), model.num_parameters()
        assert abs(static - real) / real < 0.02, (cfg.compress_ratios, static, real)
    # 激活参数只按 top-k 专家计（远小于总量）
    cfg = cfg_mix()
    assert cfg.num_active_parameters() < cfg.num_parameters() * 0.35
