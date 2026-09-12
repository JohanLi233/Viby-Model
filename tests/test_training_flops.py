from types import SimpleNamespace

import mlx.core as mx
import pytest

from trainer.flops import attn_fwdbwd_flops_per_token, gemm_active_params


def test_sparse_length_estimate_and_observed_ties():
    cfg = SimpleNamespace(
        n_heads=2,
        head_dim=4,
        window_size=4,
        compress_ratios=[0, 1],
        n_layers=2,
        n_mtp_layers=0,
        index_topk=2,
    )
    factor = 6 * 2 * 8
    assert attn_fwdbwd_flops_per_token(cfg, 10) == int(factor * (3.4 + 3.4 + 2))
    # Measured occurrences may exceed top-k at tied thresholds. Never cap them.
    assert attn_fwdbwd_flops_per_token(cfg, 10, [3, 8]) == factor * 11
    with pytest.raises(ValueError):
        attn_fwdbwd_flops_per_token(cfg, 10, [3])


def test_draft_expansion_and_lookup_exclusion():
    cfg = SimpleNamespace(
        tie_word_embeddings=False,
        n_mtp_layers=1,
        dspark_block_size=4,
        n_activated_experts=2,
        dspark_n_activated_experts=1,
    )
    model = SimpleNamespace(
        config=cfg,
        trainable_parameters=lambda: {
            "model": {"embed": {"weight": mx.zeros((10, 3))}},
            "lm_head": {"weight": mx.zeros((10, 3))},
            "mtp_modules": [
                {
                    "main_proj": {"weight": mx.zeros((3, 3))},
                    "markov_head": {
                        "embed": {"weight": mx.zeros((10, 2))},
                        "head": {"weight": mx.zeros((10, 2))},
                    },
                }
            ],
        },
    )
    assert gemm_active_params(model) == 30 * 5 + 9 + 20 * 4
