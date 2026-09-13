"""Exact likelihood, causal feature wiring, gradients, and streaming state."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from _v41_common import build
from test_dpr import cfg
from model.latent_inference import (
    LatentConfig, ShortBlockLatent, LatentFilterState, block_plan, filter_nll,
    log_softmax, prepare_features, LatentRuntime,
)


def close(a, b, atol=2e-5):
    np.testing.assert_allclose(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=atol, rtol=2e-5)


def fixture(dtype=mx.float32):
    mx.random.seed(149)
    c = LatentConfig(dim=64, vocab=128, rank=8)
    branch = ShortBlockLatent(c)
    model = build(cfg(dpr_enabled=False, vocab_size=128))
    if dtype != mx.float32:
        from trainer.utils import convert_model_dtype
        model = convert_model_dtype(model, "bfloat16")
    model.eval()
    x = mx.array([[1, 10, 11, 12, 13, 14, 15, 2, 1, 12, 13, 14, 15, 16, 17, 18]])
    y = np.array([10, 11, 12, 13, 14, 15, 2, 1, 12, 13, 14, 15, 16, 17, 18, 2], np.int32)
    seg = mx.array([[0]*8 + [1]*8])
    mask = np.array([1]*7 + [0] + [1]*8)
    d, _, _, (a, e, indices) = model.model(x, segment_ids=seg, collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
    f = prepare_features(a[0], e[0], d[0], indices[0], np.array(x[0]), y, mask, np.array(seg[0]), c)
    return branch, model, x, seg, f


def test_likelihood_identity_and_exact_gradient():
    mx.random.seed(24)
    alpha = mx.random.normal((7, 4))
    observed = -mx.abs(mx.random.normal((7, 4, 4)))
    valid = mx.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]], mx.bool_)
    def block(a, p):
        return -mx.logsumexp(log_softmax(a) + mx.where(valid[..., None], p, 0).sum(1), -1).sum()
    def online(a, p):
        return filter_nll(p, log_softmax(a), valid).sum()
    close(block(alpha, observed), online(alpha, observed))
    for a, b in zip(mx.grad(block, argnums=(0, 1))(alpha, observed), mx.grad(online, argnums=(0, 1))(alpha, observed)):
        close(a, b)
    prior = log_softmax(alpha)
    total = mx.where(valid[..., None], observed, 0).sum(1)
    gamma = mx.softmax(prior + total, -1)
    close(block(alpha, observed), (gamma * (-total + mx.log(gamma) - prior)).sum())
    close(mx.grad(block)(alpha, observed), mx.softmax(alpha, -1) - gamma)
    perm = mx.array([2, 0, 3, 1])
    close(online(alpha[:, perm], observed[..., perm]), online(alpha, observed))


def test_masks_eos_tail_and_no_missing_labels():
    x = np.array([1, 8, 9, 2, 1, 5, 6, 7, 8, 9, 0])
    y = np.array([8, 9, 2, 1, 5, 6, 7, 8, 9, 2, 0])
    mask = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])
    rows = block_plan(x, y, mask, np.array([0]*4 + [1]*6 + [-1]))
    np.testing.assert_array_equal(rows, [[0, 1, 2, -1], [4, 5, 6, 7], [8, 9, -1, -1]])
    np.testing.assert_array_equal(rows[rows >= 0], np.where(mask)[0])
    assert block_plan(x, y, mask*0, x*0).shape == (0, 4)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_feature_wiring_finite_nonzero_gradients_and_recomputation(dtype, compiled):
    branch, model, x, seg, f = fixture(dtype)
    weight = mx.stop_gradient(model._head_weight())
    def objective(params):
        branch.update(params)
        return branch(f, weight)
    fn = mx.value_and_grad(objective)
    if compiled:
        fn = mx.compile(fn)
    parameters = branch.trainable_parameters()
    loss, grads = fn(parameters)
    mx.eval(loss, grads)
    branch.update(parameters)
    assert bool(mx.isfinite(loss))
    for name, value in tree_flatten(grads):
        assert bool(mx.all(mx.isfinite(value))), name
        assert float(mx.linalg.norm(value)) > 0, name
    close(branch.token_losses(f, weight)[0], branch.token_losses(f, weight, rematerialize=False)[0])
    changed = model.model(x.at[:, :8].add(20), segment_ids=seg, collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
    close(f["d"][8:], changed[0][0, 8:], atol=0.06 if dtype == mx.bfloat16 else 2e-5)
    # Reading earlier-document evidence cannot affect the second document.
    z, prior = branch.plan(f["a"], f["e"], f["candidates"], f["visible"])
    z2, p2 = branch.plan(f["a"], f["e"].at[:8].add(100), f["candidates"], f["visible"])
    close(z[2:], z2[2:]); close(prior[2:], p2[2:])


def test_future_labels_and_evidence_do_not_enter_current_prediction():
    branch, model, x, seg, f = fixture()
    z, p = branch.plan(f["a"], f["e"], f["candidates"], f["visible"])
    z2, p2 = branch.plan(f["a"], f["e"].at[1:].add(30), f["candidates"], f["visible"])
    close(z[:1], z2[:1]); close(p[:1], p2[:1])
    lp = -mx.abs(mx.random.normal((1, 4, 4)))
    nll = filter_nll(lp, p[:1], mx.ones((1, 4), mx.bool_))
    close(nll[:, :2], filter_nll(lp.at[:, 2:].add(-10), p[:1], mx.ones((1, 4), mx.bool_))[:, :2])


def test_stream_keeps_unfinished_prompt_block_and_probability_semantics():
    branch, _, _, _, _ = fixture()
    z = mx.random.normal((1, 4, 8))
    prior = log_softmax(mx.random.normal((1, 4)))
    hidden, base = mx.random.normal((1, 4, 64)), mx.random.normal((1, 4, 128))
    labels = mx.array([[20, 11, 12, 2]])
    component = branch.component_logprobs(hidden, base, z)
    observed = mx.take_along_axis(component, labels[..., None, None], -1)[..., 0]
    reference = filter_nll(observed, prior, mx.ones((1, 4), mx.bool_))
    state = LatentFilterState(z, prior)
    for j in range(4):
        out = state.predict(branch, hidden[:, j], base[:, j])
        close(mx.exp(out).sum(-1), mx.ones((1,)))
        close(-out[:, int(labels[0, j])], reference[:, j])
        done = state.observe(labels[:, j])
        assert bool(done[0]) == (j == 3)
        if j == 1:  # prompt ends; the same state resumes in decode
            assert state.count == 2


def test_small_initialization_and_baseline_fallback():
    branch, model, _, _, f = fixture()
    z, prior = branch.plan(f["a"], f["e"], f["candidates"], f["visible"])
    assert float(mx.max(mx.linalg.norm(branch.output.weight, axis=-1))) <= 1e-3 / 8**0.5 + 1e-9
    loss, base = branch.token_losses(f, model._head_weight())
    assert float(mx.max(mx.abs(loss - base))) < 0.002
    branch.output.weight = mx.zeros_like(branch.output.weight)
    loss, base = branch.token_losses(f, model._head_weight())
    close(loss, base)


def test_feature_export_preserves_backbone_and_is_causal():
    _, model, x, seg, f = fixture()
    normal = model(x, use_dpr=False, use_ced_recurrent=False, segment_ids=seg).logits
    close(model.logits(f["d"])[None], normal)
    changed = model.model(x.at[:, 2:4].add(30), segment_ids=seg, collect_main=False, use_dpr=False, use_ced_recurrent=False, return_latent_features=True)
    close(f["d"][:2], changed[0][0, :2])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_prefill_decode_unfinished_blocks_and_eos(dtype):
    branch, model, _, _, _ = fixture(dtype)
    x = mx.array([[1, 10, 11, 12, 13, 14, 15, 16, 17]])
    full = LatentRuntime(model, branch).prefill(x)
    runtime = LatentRuntime(model, branch)
    parts = [runtime.prefill(x[:, :3])]
    assert runtime.filter.count == 2
    for t in range(3, x.shape[1]):
        parts.append(runtime.decode_step(x[:, t])[:, None])
    actual = mx.concatenate(parts, 1)
    if dtype == mx.bfloat16:
        # BF16 backbone itself changes reduction paths with prefix shape. Check
        # the additional latent discrepancy against the same-state base control.
        base_full, _ = model.prefill(x, use_dpr=False, use_ced_recurrent=False)
        prefix, cache = model.prefill(x[:, :3], use_dpr=False, use_ced_recurrent=False)
        base_parts = [prefix]
        for t in range(3, x.shape[1]):
            step, cache = model.decode_step(x[:, t], cache)
            base_parts.append(step[:, None])
        close(full - log_softmax(base_full), actual - log_softmax(mx.concatenate(base_parts, 1)), atol=0.002)
    else:
        close(full, actual, atol=3e-4)
    runtime.decode_step(mx.array([2]))
    reset = runtime.decode_step(mx.array([1]))
    fresh = LatentRuntime(model, branch).prefill(mx.array([[1]]))[:, 0]
    close(reset, fresh)
    assert runtime.filter.count == 0 and runtime.cache.start_pos == 1


def test_rematerialized_gradients_checkpoint_roundtrip_and_query_intervention(tmp_path):
    branch, model, _, _, f = fixture()
    params = branch.trainable_parameters()
    def loss(p, remat):
        branch.update(p)
        return branch.token_losses(f, model._head_weight(), rematerialize=remat)[0].sum()
    g1 = mx.grad(lambda p: loss(p, True))(params)
    g2 = mx.grad(lambda p: loss(p, False))(params)
    mx.eval(g1, g2)
    branch.update(params)
    for (n1, a), (n2, b) in zip(tree_flatten(g1), tree_flatten(g2)):
        assert n1 == n2
        close(a, b)
    z, _ = branch.plan(f["a"], f["e"], f["candidates"], f["visible"])
    fixed, _ = branch.plan(f["a"], f["e"], f["candidates"], f["visible"], fixed_second_query=True)
    assert float(mx.max(mx.abs(z-fixed))) > 1e-5
    branch.save_weights(str(tmp_path / 'branch.safetensors'))
    other = ShortBlockLatent(branch.config)
    other.load_weights(str(tmp_path / 'branch.safetensors'))
    close(branch(f, model._head_weight()), other(f, model._head_weight()))
