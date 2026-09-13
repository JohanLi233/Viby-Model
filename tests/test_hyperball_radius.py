"""Persistent norm bounds under BF16 rounding and checkpoint restoration."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from trainer.muon import BatchedMuon


@pytest.mark.parametrize("layout", ["matrix", "heads", "experts"])
def test_bf16_fixed_radius_over_many_updates_and_reload(layout, tmp_path):
    mx.random.seed(123)
    name = "attn.wq_b" if layout == "heads" else "w"
    shape = (3, 8, 16) if layout == "experts" else (24, 16)
    p = mx.random.normal(shape).astype(mx.bfloat16) * 0.05
    groups = p.reshape(3, 8, 16) if layout != "matrix" else p[None]
    reference = np.linalg.norm(np.array(groups.astype(mx.float32)), axis=(-2, -1))
    opt = BatchedMuon(
        0.01, hyperball=True, ns_bf16=True, head_dim=8 if layout == "heads" else 0
    )
    params = tree_unflatten([(name, p)])
    grads = tree_unflatten([(name, mx.random.normal(shape).astype(mx.bfloat16))])
    for i in range(300):
        params = opt.apply_gradients(grads, params)
        if i % 30 == 0:
            mx.eval(params, opt.state)
    final = dict(tree_flatten(params))[name].reshape(groups.shape)
    norms = np.linalg.norm(np.array(final.astype(mx.float32)), axis=(-2, -1))
    # Rounding once to BF16 has relative error <= 2^-8; error does not accrue.
    np.testing.assert_allclose(norms, reference, rtol=2**-8, atol=1e-6)
    radii = {k: v for k, v in tree_flatten(opt.state) if k.endswith(".radius")}
    assert len(radii) == 1 and next(iter(radii.values())).dtype == mx.float32
    checkpoint = str(tmp_path / "state.safetensors")
    mx.save_safetensors(checkpoint, dict(tree_flatten(opt.state)))
    other = BatchedMuon(
        0.01, hyperball=True, ns_bf16=True, head_dim=8 if layout == "heads" else 0
    )
    other.state = tree_unflatten(list(mx.load(checkpoint).items()))
    a = opt.apply_gradients(grads, params)
    b = other.apply_gradients(grads, params)
    np.testing.assert_array_equal(
        np.array(dict(tree_flatten(a))[name].astype(mx.float32)),
        np.array(dict(tree_flatten(b))[name].astype(mx.float32)),
    )


def test_zero_radius_establishes_once_and_legacy_radius_adoption():
    opt = BatchedMuon(0.01, hyperball=True)
    params = {"w": mx.zeros((8, 16), mx.bfloat16)}
    grads = {"w": mx.ones((8, 16), mx.bfloat16)}
    params = opt.apply_gradients(grads, params)
    radius = opt.state["w"]["radius"]
    assert float(radius.item()) > 0
    params = opt.apply_gradients(grads, params)
    np.testing.assert_array_equal(np.array(radius), np.array(opt.state["w"]["radius"]))
    # Adopting old weights is deliberate; do not forcibly shrink a trained net.
    del opt.state["w"]["radius"]
    current = float(mx.linalg.norm(params["w"].astype(mx.float32)))
    step = int(opt.step)
    opt.apply_gradients(grads, params)
    assert int(opt.step) == step + 1
    assert float(opt.state["w"]["radius"].item()) == pytest.approx(current, rel=1e-6)
