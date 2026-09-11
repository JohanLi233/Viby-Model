"""Inference keeps fixed-k decode selection and original cache semantics."""
import mlx.core as mx
import numpy as np
import pytest

from model.attention import _topk_masks
from model.kernels import decode_metadata as dm, indexer_select as fs

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="need Metal")


@pytest.mark.parametrize("n,k", [(0, 1), (1, 1), (73, 6), (129, 64), (17, 32)])
def test_decode_keeps_original_argpartition_set(n, k):
    mx.set_default_device(mx.gpu)
    mx.random.seed(14)
    for scores in (mx.random.normal((2, 1, n)), mx.full((2, 1, n), 1e10)):
        reach = mx.random.uniform(shape=scores.shape) > 0.4
        _, ref = _topk_masks(scores, reach, k, 128)
        got = dm.topk_indices(scores, reach, k, 128)
        mx.eval(ref, got)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_decode_gather_reads_correct_pools(dtype):
    w = mx.arange(2 * 8 * 64).reshape(2, 8, 64).astype(dtype)
    c = (mx.arange(2 * 19 * 64) + 13).reshape(2, 19, 64).astype(dtype)
    wi = mx.array([[[-1, -1, 0, 1, 2, 3, 4, 5]], [[7, 0, 1, 2, 3, 4, 5, 6]]], mx.int32)
    ci = mx.array([[[8, 12, -1]], [[9, 25, 26]]], mx.int32)
    out, valid = dm.gather_pools(w, c, wi, ci)
    ref = mx.concatenate([mx.take_along_axis(w[:, None], mx.maximum(wi, 0)[..., None], axis=2),
                          mx.take_along_axis(c[:, None], mx.maximum(ci - 8, 0)[..., None], axis=2)], axis=2)
    mask = mx.concatenate([wi >= 0, ci >= 0], axis=-1)
    ref = mx.where(mask[..., None], ref, 0)
    mx.eval(out, valid, ref)
    np.testing.assert_array_equal(np.asarray(out.astype(mx.float32)), np.asarray(ref.astype(mx.float32)))
    np.testing.assert_array_equal(np.asarray(valid), np.asarray(mask))


@pytest.mark.parametrize("option", ["metadata", "experts", "prefill"])
def test_prefill_and_multistep_decode_optimized_vs_reference(monkeypatch, option):
    from _v41_common import cfg_mix
    from model.model import VibyForCausalLM
    from model import moe
    from model.kernels import moe_dispatch
    from trainer.utils import convert_model_dtype

    mx.set_default_device(mx.gpu)
    # Hold prefill combine order fixed in both arms; native bf16 scatter_add
    # otherwise produces different prefixes even for metadata-only changes.
    monkeypatch.setattr(moe_dispatch, "_COMBINE_ENABLED", True)
    mx.random.seed(15)
    cfg = cfg_mix(n_heads=16, head_dim=64, window_size=8, index_n_heads=4, index_head_dim=32,
                  candidate_block_size=7, candidate_topk_blocks=2, index_topk=5)
    model = VibyForCausalLM(cfg, skip_init=True)
    convert_model_dtype(model, "bfloat16")
    model.eval()
    tokens = mx.random.randint(0, cfg.vocab_size, (2, 38))
    results = []
    for optimized in (False, True):
        monkeypatch.setattr(dm, "_ENABLED", optimized and option == "metadata")
        monkeypatch.setattr(dm, "_PREFILL_SELECT", optimized and option == "prefill")
        monkeypatch.setattr(fs, "_ENABLED", optimized and option == "prefill")
        monkeypatch.setattr(moe, "_DECODE_GATHER", optimized and option == "experts")
        out, cache = model.prefill(tokens[:, :32])
        steps = [out]
        for i in range(32, 38):
            out, cache = model.decode_step(tokens[:, i], cache)
            steps.append(out)
        mx.eval(steps)
        assert cache.start_pos == 38
        results.append([np.asarray(a.astype(mx.float32)) for a in steps])
    for step, (actual, expected) in enumerate(zip(results[1], results[0])):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected, err_msg=f"{option}, step {step}")
