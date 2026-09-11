"""NCP core gates: causal alignment, gradient directions and real trainer entry."""

import json
import time
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import pytest

from _v41_common import build, max_abs_diff
from model.config import VibyConfig
from model.ncp import NextConceptModule
from model.model import VibyForCausalLM
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import build_model_kwargs, convert_model_dtype, _reset_rope_tables
from trainer.flops import gemm_active_params, ncp_flops_per_token


def cfg(**kw):
    settings = dict(
        preset="tiny",
        dim=64,
        n_heads=2,
        o_groups=1,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        engram_layer_ids=(),
        n_mtp_layers=0,
        vocab_size=32,
        max_seq_len=32,
        window_size=8,
        ncp_enabled=True,
        ncp_layers=1,
        ncp_heads=1,
        ncp_codebooks=2,
        ncp_codebook_size=8,
        ncp_inter_dim=128,
    )
    settings.update(kw)
    return VibyConfig(**settings)


def test_configuration_and_pretrain_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train_pretrain.py", "--no_save", "--preset", "tiny"]
    )
    args = setup_training_args(
        get_pretrain_parser().parse_args(["--no_save", "--preset", "tiny"])
    )
    c = VibyConfig(**build_model_kwargs(args))
    assert c.ncp_enabled and not c.psr_enabled and c.ncp_chunk_size == 4
    assert c.ncp_layers == 1 and c.ncp_codebooks == 2
    assert VibyConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()
    with pytest.raises(ValueError):
        cfg(psr_enabled=True)
    with pytest.raises(ValueError):
        cfg(ncp_codebooks=3)
    with pytest.raises(ValueError):
        cfg(ncp_merge="hard")


def test_disabled_ncp_equals_same_weight_backbone():
    baseline = build(cfg(ncp_enabled=False))
    model = build(cfg())
    model.load_weights(tree_flatten(baseline.parameters()), strict=False)
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    assert bool(mx.array_equal(model(x, use_ncp=False).logits, baseline(x).logits))
    assert model.psr is None


def test_token_shift_matches_released_index_formula_and_no_future_leak():
    model = build(cfg())
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    a = model(x)
    assert a.ncp_result.active.tolist() == [
        [False, False, False, True, True, True, True, True, True, True]
    ]
    changed = mx.concatenate([x[:, :4], x[:, 4:] + 10], axis=1)
    b = model(changed)
    assert max_abs_diff(a.logits[:, :4], b.logits[:, :4]) < 2e-5
    assert (
        max_abs_diff(a.ncp_result.predicted[:, :1], b.ncp_result.predicted[:, :1])
        < 2e-5
    )
    # The completed second group can first affect logit 7, not logit 6.
    changed = x.at[:, 7].add(5)
    b = model(changed)
    assert max_abs_diff(a.logits[:, :7], b.logits[:, :7]) < 2e-5


def test_packed_groups_and_padding_are_excluded():
    model = build(cfg())
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    seg = mx.array([[0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]])
    pad = mx.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    out = model(x, labels=x, segment_ids=seg, attention_mask=pad)
    assert out.ncp_result.valid.tolist() == [[False, True, False]]
    assert out.ncp_result.active.tolist() == [
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
    ]
    assert float(out.ncp_loss) == 0
    changed = mx.where(seg == 0, x + 15, x)
    other = model(changed, segment_ids=seg, attention_mask=pad)
    assert (
        max_abs_diff(out.ncp_result.predicted[:, 1], other.ncp_result.predicted[:, 1])
        < 2e-5
    )


@pytest.mark.parametrize("merge", ["softmax", "raw_logits"])
def test_ntp_ncp_vq_gradients(merge):
    model = build(cfg(ncp_merge=merge))
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    for objective in ("lm_loss", "ncp_loss", "vq_loss"):
        _, grads = nn.value_and_grad(
            model, lambda m: getattr(m(x, labels=x), objective)
        )(model)
        mx.eval(grads)
        flat = dict(tree_flatten(grads))
        assert all(bool(mx.all(mx.isfinite(g))) for g in flat.values())
        assert float(mx.max(mx.abs(flat["ncp.codebooks"]))) > 0
        if objective == "vq_loss":
            assert all(
                float(mx.max(mx.abs(g))) == 0
                for k, g in flat.items()
                if k != "ncp.codebooks"
            )
        else:
            assert float(mx.max(mx.abs(flat["ncp.prediction_heads.0.weight"]))) > 0
            assert float(mx.max(mx.abs(flat["ncp.layers.0.q.weight"]))) > 0
            assert float(mx.max(mx.abs(flat["model.embed.weight"]))) > 0
    out = model(x, labels=x)
    expected = (
        out.lm_loss
        + out.ncp_loss
        + out.vq_loss
        + model.config.aux_balance_loss_weight * out.aux_loss
    )
    assert float(out.loss) == pytest.approx(float(expected), abs=1e-5)


def test_future_concept_target_is_detached():
    module = NextConceptModule(cfg())
    x = mx.random.normal((1, 2, 64))
    valid = mx.ones((1, 2), mx.bool_)

    def loss(c):
        prediction = module.predict(c, valid, None)
        return module.losses(c, prediction, valid, None)[0]

    gradient = mx.grad(loss)(x)
    assert float(mx.max(mx.abs(gradient[:, 0]))) > 0
    assert float(mx.max(mx.abs(gradient[:, 1]))) == 0


@pytest.mark.parametrize("length", [3, 4, 8])
def test_empty_short_and_all_masked_gradients(length):
    model = build(cfg())
    model.set_dtype(mx.bfloat16)
    x = mx.ones((1, length), mx.int32)
    mask = mx.zeros_like(x)
    loss, grads = nn.value_and_grad(
        model, lambda m: m(x, labels=x, attention_mask=mask, loss_mask=mask).loss
    )(model)
    mx.eval(loss, grads)
    assert bool(mx.isfinite(loss))
    assert all(bool(mx.all(mx.isfinite(g))) for _, g in tree_flatten(grads))


def test_bf16_compilation_and_concept_rope_reset():
    model = build(cfg())
    convert_model_dtype(model, "bfloat16")
    _reset_rope_tables(model, model.config)
    assert model.ncp.freq_cos.shape == (9, 32)
    params = dict(tree_flatten(model.trainable_parameters()))
    assert "ncp.freq_cos" not in params
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    def f(params, ids):
        model.update(params)
        return model(ids, labels=ids).loss

    original = model.trainable_parameters()
    fn = mx.compile(mx.value_and_grad(f))
    a, grads = fn(original, x)
    model.update(original)
    mx.eval(a, grads)
    assert bool(mx.isfinite(a))
    opt = optimizers.Adam(1e-3)
    opt.update(model, grads)
    mx.eval(model.parameters())
    updated = model.trainable_parameters()
    b, _ = fn(updated, x)
    model.update(updated)
    assert float(mx.abs(a - b)) > 1e-5


def test_reference_prefill_decode_is_causal_and_reload(tmp_path):
    model = build(cfg())
    model.eval()
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    full = model(x).logits
    first, cache = model.prefill(x[:, :3])
    outputs = [first]
    for i in range(3, 8):
        out, _ = model.decode_step(x[:, i], cache)
        outputs.append(out[:, None])
    assert max_abs_diff(full, mx.concatenate(outputs, axis=1)) < 2e-5
    path = tmp_path / "model.safetensors"
    model.save_weights(str(path))
    other = VibyForCausalLM(model.config, skip_init=True)
    other.load_weights(str(path))
    other.eval()
    assert max_abs_diff(full, other(x).logits) == 0


def test_real_pretrain_logger_and_muon_update(tmp_path):
    args = get_pretrain_parser().parse_args(
        [
            "--out_dir",
            str(tmp_path),
            "--learning_rate",
            "0.0001",
            "--accumulation_steps",
            "2",
            "--max_steps",
            "4",
            "--cache_limit_gb",
            "0",
            "--max_seq_len",
            "8",
        ]
    )
    args.save_dir = str(tmp_path)
    args.warmup_iters = 0
    args.compile_model = True
    model = build(cfg())
    convert_model_dtype(model, "bfloat16")
    _reset_rope_tables(model, model.config)
    tr = BaseTrainer(
        args, model, SimpleNamespace(pad_token_id=0), model.config, "pretrain"
    )
    assert tr.psr_optimizer is None
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    sample = (x, x + 1, mx.ones_like(x), mx.zeros_like(x))
    before = model.ncp.codebooks
    tr._run_epoch_steps(iter([sample] * 4), 0, 4, 4, None, 0, time.time(), 0)
    assert max_abs_diff(before, model.ncp.codebooks) > 0
    groups = tr.optimizer._split_dictionary(model.trainable_parameters())
    for opt, group in zip(tr.optimizer.optimizers, groups):
        if "ncp.codebooks" in dict(tree_flatten(group)):
            assert "Muon" not in type(opt).__name__
    rows = [
        json.loads(line)
        for line in (tmp_path / "ncp_metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and rows[-1]["valid_pairs"] == 1
    assert (tmp_path / "pretrain_64.safetensors").exists()


def test_concept_cost_is_at_pooled_resolution():
    a = build(cfg(ncp_enabled=False))
    b = build(cfg())
    assert gemm_active_params(a) == gemm_active_params(b)
    assert ncp_flops_per_token(b.config, 32) > 0
    assert ncp_flops_per_token(a.config, 32) == 0
