"""Metal acceptance gates for opt-in MoE dataflow paths.

This module must run on Apple hardware. A Linux skip is not GPU validation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required; Metal tests not executed")
from model import moe as model_moe
from model.kernels import (
    moe_counts as counts,
    moe_decode as decode,
    moe_gather as gather,
)
from model.kernels.moe_dispatch import route_inverse

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="need Apple Metal")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    mx.set_default_device(mx.gpu)
    mx.random.seed(20260911)
    monkeypatch.setattr(gather, "_ENABLED", True)
    monkeypatch.setattr(counts, "_ENABLED", True)
    monkeypatch.setattr(decode, "_ENABLED", False)


def f32(x):
    return np.asarray(x.astype(mx.float32))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "m,d,k", [(0, 65, 6), (1, 1, 1), (9, 127, 6), (33, 1031, 1), (7, 0, 6)]
)
def test_gather_forward_and_exact_binary_vjp(dtype, m, d, k):
    order = mx.array(np.random.default_rng(3).permutation(m * k), mx.int32)
    inv = route_inverse(order)
    x = mx.random.normal((m, d)).astype(dtype)
    # Binary fractions keep the six-way reference sum exactly representable.
    g = (mx.random.randint(-8, 9, (m * k, d)).astype(mx.float32) / 8).astype(dtype)
    (got,), grads = mx.vjp(lambda a: gather.gather_routes(a, order, inv, k), [x], [g])
    expected = x[(order // k).astype(mx.int32)]
    dx = mx.sum(g[inv].reshape(m, k, d).astype(mx.float32), axis=1).astype(dtype)
    mx.eval(got, grads, expected, dx)
    np.testing.assert_array_equal(f32(got), f32(expected))
    np.testing.assert_array_equal(f32(grads[0]), f32(dx))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_compiled_gather_vjp_and_later_trainable_input(dtype):
    m, d, k = 19, 129, 6
    order = mx.array(np.random.default_rng(9).permutation(m * k), mx.int32)
    inv = route_inverse(order)
    x = mx.random.normal((m, d)).astype(dtype)
    scale = mx.random.normal((m * k, d)).astype(dtype)

    def loss(a, b):
        y = gather.gather_routes(a, order, inv, k)
        return mx.sum((y * b).astype(mx.float32))

    eager = mx.value_and_grad(loss, argnums=(0, 1))
    v0, g0 = eager(x, scale)
    mx.eval(v0, g0)  # materialize custom VJP before outer compile
    compiled = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
    v1, g1 = compiled(x, scale)
    mx.eval(v1, g1)
    np.testing.assert_allclose(f32(v1), f32(v0), rtol=1e-5, atol=1e-4)
    for a, b in zip(g1, g0):
        np.testing.assert_array_equal(f32(a), f32(b))
    expected_scale_grad = x[(order // k).astype(mx.int32)]
    np.testing.assert_array_equal(f32(g1[1]), f32(expected_scale_grad))


@pytest.mark.parametrize(
    "b,t,e,k",
    [
        (0, 3, 8, 1),
        (2, 0, 8, 6),
        (1, 1, 8, 1),
        (3, 43, 96, 6),
        (4, 1024, 96, 6),
        (2, 5, 513, 6),
    ],
)
@pytest.mark.parametrize("hot", [False, True])
def test_route_counts_are_occurrences_and_batch_local(b, t, e, k, hot):
    ids_np = (
        np.zeros((b * t, k), np.int32)
        if hot
        else np.random.default_rng(5).integers(e, size=(b * t, k), dtype=np.int32)
    )
    ids = mx.array(ids_np)
    got = counts.sequence_route_counts(ids, b, t, e)
    expected = np.zeros((b * t, e), np.float32)
    for j in range(k):
        np.add.at(expected, (np.arange(b * t), ids_np[:, j]), 1)
    expected = expected.reshape(b, t, e).sum(axis=1)
    mx.eval(got)
    assert got.dtype == mx.float32
    np.testing.assert_array_equal(f32(got), expected)
    assert float(mx.sum(got)) == b * t * k


@pytest.mark.parametrize("b,t,k", [(1, 1, 1), (3, 43, 6), (4, 1024, 6)])
def test_aux_loss_value_and_score_gradient(monkeypatch, b, t, k):
    e = 96
    owner = SimpleNamespace(n_routed=e, top_k=k)
    ids = mx.random.randint(0, e, (b * t, k)).astype(mx.int32)
    scores = mx.random.uniform(shape=(b * t, e))
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(counts, "_ENABLED", enabled)
        fn = mx.value_and_grad(
            lambda s: model_moe.MoEFeedForward.seq_aux_loss(owner, s, ids, b, t)
        )
        value, grad = fn(scores)
        mx.eval(value, grad)
        results.append((f32(value), f32(grad)))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=2e-6, atol=1e-8)


def make_layer(e, k, dtype):
    cfg = SimpleNamespace(
        dim=64,
        moe_inter_dim=32,
        swiglu_limit=10.0,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.0,
        aux_balance_loss_weight=1e-4,
        moe_of=lambda layer: (e, k),
    )
    layer = model_moe.MoEFeedForward(cfg)
    layer.router.weight = layer.router.weight.astype(dtype)
    layer.experts.gate_up_w = layer.experts.gate_up_w.astype(dtype)
    layer.experts.down_w = layer.experts.down_w.astype(dtype)
    for linear in (layer.shared.w1, layer.shared.w2, layer.shared.w3):
        linear.weight = linear.weight.astype(dtype)
    return layer


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("b,e,k", [(1, 96, 6), (3, 12, 6), (8, 12, 1)])
def test_decode_exact_parity_and_no_captured_weights(monkeypatch, dtype, b, e, k):
    monkeypatch.setattr(model_moe, "_DECODE_GATHER", True)
    layer = make_layer(e, k, dtype)
    layer.eval()
    decode._compiled_region.cache_clear()
    # Second call uses new weights and new routing, but identical shapes.
    for step in range(2):
        x = mx.random.normal((b, 1, 64)).astype(dtype)
        if step:
            layer.experts.gate_up_w = layer.experts.gate_up_w * 0.75
            layer.experts.down_w = layer.experts.down_w * 1.125
            for linear in (layer.shared.w1, layer.shared.w2, layer.shared.w3):
                linear.weight = linear.weight * 0.875
            layer.router.weight = -layer.router.weight
            layer.shared.swiglu_limit = 0.125
        monkeypatch.setattr(decode, "_ENABLED", False)
        expected = layer(x)
        mx.eval(expected)
        monkeypatch.setattr(decode, "_ENABLED", True)
        got = layer(x)
        mx.eval(got)
        np.testing.assert_array_equal(f32(got), f32(expected))


def test_training_aux_load_side_channel_survives_compile(monkeypatch):
    layer = make_layer(12, 6, mx.float32)
    layer.train()
    x = mx.random.normal((2, 17, 64))
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(counts, "_ENABLED", enabled)

        def forward(a):
            y = layer(a)
            return y, layer._last_aux, layer.load_stats()

        fn = mx.compile(forward)
        y, aux, load = fn(x)
        mx.eval(y, aux, load)
        outputs.append((f32(y), f32(aux), f32(load)))
    # The count optimization must not change expert forward or global load.
    np.testing.assert_allclose(outputs[1][0], outputs[0][0], rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(outputs[1][2], outputs[0][2])
    np.testing.assert_allclose(outputs[1][1], outputs[0][1], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("batch", [1, 2])
def test_full_model_prefill_and_eight_decode_steps(monkeypatch, batch):
    from _v41_common import cfg_mix
    from model.model import VibyForCausalLM
    from model.kernels import moe_dispatch
    from trainer.utils import convert_model_dtype

    monkeypatch.setattr(moe_dispatch, "_COMBINE_ENABLED", True)
    monkeypatch.setattr(model_moe, "_DECODE_GATHER", True)
    cfg = cfg_mix(
        n_heads=16,
        head_dim=64,
        window_size=8,
        index_n_heads=4,
        index_head_dim=32,
        candidate_block_size=7,
        candidate_topk_blocks=2,
        index_topk=5,
    )
    model = VibyForCausalLM(cfg, skip_init=True)
    convert_model_dtype(model, "bfloat16")
    model.eval()
    tokens = mx.random.randint(0, cfg.vocab_size, (batch, 40))
    arms = []
    for enabled in (False, True):
        monkeypatch.setattr(decode, "_ENABLED", enabled)
        out, cache = model.prefill(tokens[:, :32])
        steps = [out]
        for i in range(32, 40):
            out, cache = model.decode_step(tokens[:, i], cache)
            steps.append(out)
        mx.eval(steps)
        assert cache.start_pos == 40
        arms.append([f32(a) for a in steps])
    for step, (got, expected) in enumerate(zip(arms[1], arms[0])):
        np.testing.assert_array_equal(got, expected, err_msg=f"decode step {step}")


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_native_sorted_moe_complete_vjp(monkeypatch, dtype):
    from model.kernels import moe_dispatch

    monkeypatch.setattr(moe_dispatch, "_ENABLED", False)  # never custom GEMMs
    monkeypatch.setattr(moe_dispatch, "_COMBINE_ENABLED", True)
    b, t, d, e, k, width = 2, 17, 64, 12, 6, 32
    x = mx.random.normal((b, t, d)).astype(dtype)
    gu = (mx.random.normal((e, 2 * width, d)) / d**0.5).astype(dtype)
    dw = (mx.random.normal((e, d, width)) / width**0.5).astype(dtype)
    ids_np = np.stack(
        [np.random.default_rng(i).choice(e, k, replace=False) for i in range(b * t)]
    )
    ids = mx.array(ids_np, mx.int32)
    w = mx.random.uniform(shape=(b * t, k)) / k
    cot = mx.random.normal((b, t, d)).astype(dtype)

    def forward(a, gate_up, down, weights):
        owner = SimpleNamespace(
            top_k=k,
            n_routed=e,
            moe_in=width,
            swiglu_limit=10.0,
            training=True,
            experts=SimpleNamespace(gate_up_w=gate_up, down_w=down),
        )
        return model_moe.MoEFeedForward._sparse_forward(owner, a, ids, weights)

    results = []
    for enabled in (False, True):
        monkeypatch.setattr(gather, "_ENABLED", enabled)
        (out,), grad = mx.vjp(forward, [x, gu, dw, w], [cot])
        mx.eval(out, grad)
        results.append((f32(out), [f32(g) for g in grad]))
    tol = {
        mx.float32: (1e-4, 1e-5),
        mx.float16: (5e-3, 3e-3),
        mx.bfloat16: (3e-2, 2e-2),
    }[dtype]
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=tol[0], atol=tol[1])
    for actual, expected in zip(results[1][1], results[0][1]):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=tol[0], atol=tol[1])
