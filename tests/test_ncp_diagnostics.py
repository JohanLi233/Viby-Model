"""Diagnostic math and intervention contracts; runnable on MLX CPU."""

import mlx.core as mx
import numpy as np
import pytest
from model.ncp import concept_diagnostics, NCP_METRICS
from test_ncp_ced import model, tokens, close


def test_masked_population_and_identical_pair_baselines():
    c = mx.array(
        [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]], [[2.0, 2.0], [4.0, 4.0], [99.0, 99.0]]]
    )
    p = mx.array(
        [[[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]], [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    valid = mx.array([[True, True, False], [True, True, False]])
    pairs = mx.array([[True, False], [True, False]])
    v = concept_diagnostics(c, p, valid, pairs, mx.ones((2,)))
    close(v, mx.array([np.sqrt(7.5), 1.25, 0.25, 2, 1, 12.5, 12.5, 4, 1 / 12.5, 1 / 4]))
    # Scale shrinkage lowers raw energy/MSE, but not normalized prediction error.
    small = concept_diagnostics(c * 0.2, p * 0.2, valid, pairs, mx.ones((2,)))
    close(small[8:], v[8:])
    g = mx.grad(lambda x: concept_diagnostics(x, p, valid, pairs, mx.ones((2,))).sum())(
        c
    )
    assert float(mx.abs(g).sum()) == 0


def test_empty_and_single_sample_populations_are_finite():
    c = mx.ones((1, 1, 2))
    valid = mx.zeros((1, 1), mx.bool_)
    pairs = mx.zeros((1, 0), mx.bool_)
    v = concept_diagnostics(c, c, valid, pairs, mx.ones((2,)))
    close(v, mx.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))


def test_eval_interventions_and_diagnostic_vector():
    m = model()
    m.model.ncp.feedback_gate = mx.array([0.3])
    x = tokens(12, 2)
    with pytest.raises(ValueError, match="eval mode"):
        m(x, ncp_intervention="off")
    m.eval()
    normal = m(x, labels=x)
    assert normal.ncp_metrics.shape == (len(NCP_METRICS),)
    off = m(x, labels=x, ncp_intervention="off")
    metrics = dict(zip(NCP_METRICS, off.ncp_metrics.tolist()))
    assert metrics["feedback_rms"] == 0 and metrics["feedback_ratio"] == 0
    assert normal.ncp_metrics[-1].item() > 0
    close(normal.ncp_loss, off.ncp_loss)
    swapped = m(x, labels=x, ncp_intervention="swap")
    assert np.isfinite(swapped.lm_loss.item())
    with pytest.raises(ValueError, match="two unpadded"):
        m(x[:1], ncp_intervention="swap")
    with pytest.raises(ValueError, match="Unknown"):
        m(x, ncp_intervention="typo")


def test_eval_token_weighting_and_no_singleton_swap():
    from experiments.eval_ncp_contribution import evaluate
    from types import SimpleNamespace

    calls = []

    class Fake:
        config = SimpleNamespace(vocab_size=50, max_seq_len=20)

        def eval(self):
            pass

        def __call__(self, ids, **kw):
            calls.append((ids.shape[0], kw["ncp_intervention"]))
            weights = kw["loss_mask"]
            offset = {"normal": 0, "off": 1, "swap": 2}[kw["ncp_intervention"]]
            return SimpleNamespace(
                lm_loss=(ids.astype(mx.float32) * weights).sum() / weights.sum()
                + offset
            )

    ids = np.arange(10).reshape(5, 2)
    mask = np.ones_like(ids, dtype=np.float32)
    mask[0] = 0
    out = evaluate(Fake(), (ids, ids, mask), 2)
    assert [b["end"] - b["start"] for b in out["batches"]] == [2, 3]
    assert out["off_minus_normal"] == pytest.approx(1)
    assert out["ce"]["normal"] == pytest.approx((ids * mask).sum() / mask.sum())
    assert all(n >= 2 for n, _ in calls)


def test_diagnostics_are_detached_from_trainable_parameters():
    from mlx import nn
    from mlx.utils import tree_flatten

    m = model()
    x = tokens(12, 2)
    _, grad = nn.value_and_grad(m, lambda net: net(x, labels=x).ncp_metrics[5:].sum())(
        m
    )
    assert all(float(mx.abs(value).sum()) == 0 for _, value in tree_flatten(grad))
