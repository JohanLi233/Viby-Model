"""Causal QB, precision, compile and checkpoint contracts; no long training."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from _v41_common import build, cfg_tiny
from model.moe import MoEGate, MoEFeedForward, update_quantile_bias
from trainer.config import get_pretrain_parser
from trainer.utils import build_model_kwargs, convert_model_dtype, load_model_weights


def test_cli_roundtrip_and_validation():
    args = get_pretrain_parser().parse_args(
        [
            "--moe_balance_method",
            "qb",
            "--qb_update_rate",
            "0.25",
            "--qb_stats_rows",
            "512",
            "--aux_balance_loss_weight",
            "0",
        ]
    )
    cfg = cfg_tiny(
        **{
            k: v
            for k, v in build_model_kwargs(args).items()
            if k
            in (
                "moe_balance_method",
                "qb_update_rate",
                "qb_stats_rows",
                "aux_balance_loss_weight",
            )
        }
    )
    assert cfg.moe_balance_method == "qb" and cfg.qb_update_rate == 0.25
    assert cfg.qb_stats_rows == 512 and cfg.aux_balance_loss_weight == 0
    for kw in (
        {"qb_update_rate": 1.1},
        {"qb_stats_rows": 0},
        {"gate_temp": 0},
        {"moe_balance_method": "unknown"},
    ):
        with pytest.raises(ValueError):
            cfg_tiny(**kw)


def test_qb_numpy_reference_shift_invariance_and_guards():
    rng = np.random.default_rng(42)
    margins = rng.normal(size=(157, 16)).astype(np.float32)
    bias = rng.normal(size=16).astype(np.float32)
    got = np.asarray(update_quantile_bias(mx.array(bias), mx.array(margins), 4))
    count = int(np.ceil(157 * 4 / 16))
    target = -np.sort(margins, axis=0)[157 - count]
    target -= target.mean()
    expected = (bias - bias.mean() + target) * 0.5
    np.testing.assert_allclose(got, expected, atol=2e-6)
    shifted = update_quantile_bias(mx.array(bias + 10), mx.array(margins - 10), 4)
    np.testing.assert_allclose(np.asarray(shifted), got, atol=2e-6)
    for invalid, k in (
        (mx.zeros((0, 16)), 4),
        (mx.full((8, 16), float("nan")), 4),
        (mx.ones((8, 16)), 16),
    ):
        np.testing.assert_array_equal(
            np.asarray(update_quantile_bias(mx.array(bias), invalid, k)), bias
        )


def test_compiled_gate_statistics_are_detached_and_bias_is_dynamic():
    cfg = cfg_tiny(qb_stats_rows=32)
    gate = MoEGate(cfg)
    x = mx.random.normal((97, cfg.dim)).astype(mx.bfloat16)
    weight = gate.weight

    def forward(weight, bias, inputs):
        gate.weight, gate.bias = weight, bias
        weights, ids, scores = gate(inputs)
        return weights.sum(), ids, scores, gate._last_qb_margins

    compiled = mx.compile(mx.value_and_grad(forward))
    old_bias = mx.zeros((16,))
    (loss, ids, scores, margins), grad = compiled(weight, old_bias, x)
    mx.eval(loss, ids, scores, margins, grad)
    assert margins.shape == (32, 16) and margins.dtype == mx.float32
    sample = ((np.arange(32) + 0.5) * (97 / 32)).astype(np.int32)
    raw = np.asarray(scores)[sample]
    threshold = np.sort(raw, axis=1)[:, -5:-4]
    np.testing.assert_allclose(np.asarray(margins), raw - threshold, atol=1e-6)
    new_bias = mx.arange(16).astype(mx.float32) * 100
    (_, changed_ids, _, _), _ = compiled(weight, new_bias, x)
    mx.eval(changed_ids)
    assert set(np.asarray(changed_ids).reshape(-1)) == {12, 13, 14, 15}
    # Statistics exported from the loss have no gradient path of their own.
    stats_grad = mx.grad(lambda w: forward(w, old_bias, x)[3].sum())(weight)
    assert float(mx.abs(stats_grad).max()) == 0
    gate.weight, gate.bias = weight, old_bias
    gate.eval()
    before = np.asarray(gate.bias).copy()
    gate(x)
    np.testing.assert_array_equal(np.asarray(gate.bias), before)


def test_extreme_logits_have_finite_gradients():
    cfg = cfg_tiny()
    gate = MoEGate(cfg)
    gate.weight = mx.full(gate.weight.shape, -100.0)
    x = mx.ones((5, cfg.dim))
    grad = mx.grad(lambda x: gate.scores(x).sum())(x)
    assert bool(mx.all(mx.isfinite(grad)))


def test_aux_balance_cannot_shrink_all_scores_to_reduce_loss():
    owner = SimpleNamespace(n_routed=4, top_k=2)
    scores = mx.array([[4.0, 2.0, 1.0, 1.0], [2.0, 4.0, 1.0, 1.0]])
    ids = mx.array([[0, 1], [0, 1]], dtype=mx.int32)
    def fn(s):
        return MoEFeedForward.seq_aux_loss(owner, s, ids, 1, 2)
    assert float(fn(scores)) == pytest.approx(float(fn(scores * 0.01)), abs=1e-6)
    grad = mx.grad(fn)(scores)
    assert abs(float((grad * scores).sum())) < 1e-6


def test_fp32_router_survives_dtype_conversion_and_old_checkpoint(tmp_path):
    model = build(cfg_tiny(n_mtp_layers=0))
    convert_model_dtype(model, "bfloat16")
    params = dict(tree_flatten(model.parameters()))
    for k, v in params.items():
        if k.endswith(("router.weight", "router.bias")):
            assert v.dtype == mx.float32
    path = tmp_path / "legacy.safetensors"
    mx.save_safetensors(
        str(path),
        {k: v.astype(mx.bfloat16) if "router." in k else v for k, v in params.items()},
    )
    assert load_model_weights(model, str(path))
    assert all(
        v.dtype == mx.float32
        for k, v in tree_flatten(model.parameters())
        if "router." in k
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_accumulated_training_updates_qb_once_and_skips_nonfinite(
    compiled, monkeypatch
):
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from trainer.base_trainer import BaseTrainer

    cfg = cfg_tiny(
        dim=64,
        n_heads=2,
        head_dim=32,
        rope_head_dim=16,
        o_groups=1,
        o_lora_rank=32,
        q_lora_rank=32,
        moe_inter_dim=32,
        n_mtp_layers=0,
        qb_stats_rows=32,
    )
    model = build(cfg)
    convert_model_dtype(model, "bfloat16")
    tr = BaseTrainer.__new__(BaseTrainer)
    tr.model, tr.lm_config, tr.training_type = model, cfg, "pretrain"
    tr.args = SimpleNamespace(
        accumulation_steps=2, compile_model=compiled, grad_clip=1.0
    )
    tr._expl_nest_mu, tr._en_delta = 0, None
    tr._moe_gates = model.moe_gates
    for gate in tr._moe_gates:
        gate.qb_stats_rows = cfg.qb_stats_rows // 2
    tr.optimizer = optim.AdamW(learning_rate=1e-4)
    tr._loss_and_grad = tr._build_loss_and_grad()
    before = np.asarray(model.moe_bias_stack()).copy()
    grads, loads, samples = None, None, []
    for _ in range(2):
        x = mx.random.randint(1, cfg.vocab_size, (2, 16))
        output, g = tr._compute_loss_and_grad(x, x, mx.ones_like(x), mx.ones_like(x))
        mx.eval(output, g)
        assert bool(mx.isfinite(output[0]))
        grads = g if grads is None else tree_map(mx.add, grads, g)
        loads = output[2] if loads is None else loads + output[2]
        samples.append(output[6])
        np.testing.assert_array_equal(np.asarray(model.moe_bias_stack()), before)
    margins = mx.concatenate(samples, axis=1)
    expected = mx.stack(
        [
            update_quantile_bias(mx.array(b), m, cfg.n_activated_experts)
            for b, m in zip(before, margins)
        ]
    )
    norm = tr._optimizer_step(grads, 2, loads, margins)
    assert np.isfinite(norm) and norm > 0
    np.testing.assert_allclose(
        np.asarray(model.moe_bias_stack()), np.asarray(expected), atol=1e-6
    )
    after = np.asarray(model.moe_bias_stack()).copy()
    tr._optimizer_step(
        tree_map(lambda g: mx.full(g.shape, float("nan")), grads), 2, loads, margins
    )
    np.testing.assert_array_equal(np.asarray(model.moe_bias_stack()), after)
    assert all(
        v.dtype == mx.float32
        for k, v in tree_flatten(model.parameters())
        if "router." in k
    )

    # Exercise the real epoch loop too: bounded samples from both microbatches
    # must reach each optimizer step, and normalized load metrics must log.
    import time
    import trainer.base_trainer as base_trainer

    tr.tokenizer = SimpleNamespace(pad_token_id=0)
    tr.args.log_interval, tr.args.warmup_iters = 1, 1
    tr._save_if_needed = lambda *a: None
    logged, windows = [], []
    monkeypatch.setattr(
        base_trainer, "log_training_progress", lambda *a, **kw: logged.append(kw)
    )
    optimizer_step = tr._optimizer_step

    def checked_step(grads, count, moe_loads=None, moe_qb_margins=None):
        windows.append((count, tuple(moe_qb_margins.shape)))
        return optimizer_step(grads, count, moe_loads, moe_qb_margins)

    tr._optimizer_step = checked_step
    batches = [(x, x, mx.ones_like(x))] * 4
    tr._run_epoch_steps(
        loader_iter=iter(batches),
        epoch=0,
        iter_per_epoch=4,
        total_training_steps=4,
        swanlab=object(),
        skip_steps=0,
        start_time=time.time(),
        base_step_offset_for_speed=0,
    )
    assert windows == [(2, (cfg.n_layers, 32, cfg.n_routed_experts))] * 2
    assert len(logged) == 4
    assert logged[-1]["extra"]["moe/gate0_max_load_ratio"] >= 1.0


@pytest.mark.parametrize("scale", [0.02, 1.0, 20.0])
def test_qb_recovers_hot_experts_across_score_scales(scale):
    rng = np.random.default_rng(13)
    raw = rng.normal(0, 0.1, (4096, 16)).astype(np.float32)
    raw[:, :4] += 1.0
    raw = mx.array(raw * scale)
    bias = mx.zeros((16,))
    ratios = []
    for _ in range(24):
        biased = raw + bias
        ids = mx.argpartition(-biased, kth=3, axis=1)[:, :4]
        counts = np.bincount(np.asarray(ids).reshape(-1), minlength=16)
        ratios.append(counts.max() / counts.mean())
        alpha = -mx.partition(-biased, kth=4, axis=1)[:, 4:5]
        bias = update_quantile_bias(bias, raw - alpha, 4)
        mx.eval(bias)
    assert ratios[0] > 3.9
    assert max(ratios[-4:]) < 1.15, ratios
