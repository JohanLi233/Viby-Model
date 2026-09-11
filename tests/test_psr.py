"""Scientific PSR contracts, explicitly checked on MLX rather than toy algebra."""

from types import SimpleNamespace
import pytest
import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten, tree_map
from _v41_common import build, max_abs_diff
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.init import apply_trunc_normal_init
from trainer.psr_pretrain import anchor_plan, compact_anchor_plan, text_psr_inputs
from trainer.base_trainer import BaseTrainer
from trainer.psr_optim import ParameterView


def config(**kw):
    args = dict(
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
        psr_enabled=True,
        psr_dim=32,
        psr_slots=2,
        psr_rounds=1,
        psr_horizon=4,
        psr_train_anchors=2,
    )
    args.update(kw)
    return VibyConfig(**args)


def pair():
    baseline = build(config(psr_enabled=False))
    revised = build(config())
    shared = tree_flatten(baseline.parameters())
    revised.load_weights(shared, strict=False)
    for k, v in shared:
        assert bool(mx.array_equal(v, dict(tree_flatten(revised.parameters()))[k])), k
    return baseline, revised


def batch():
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]])
    y = x + 1
    mask = mx.ones_like(x)
    pad = mx.ones_like(x)
    seg = mx.array([[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
    mask = mask.at[0, 5].add(-1)
    return x, y, mask, pad, seg


def trainer(model, compiled=False, mode="recurrent"):
    tr = BaseTrainer.__new__(BaseTrainer)
    tr.model, tr.lm_config, tr.training_type = model, model.config, "pretrain"
    tr.args = SimpleNamespace(
        accumulation_steps=1,
        compile_model=compiled,
        psr_training_mode=mode,
        psr_grad_clip=0.2,
        grad_clip=0.3,
        seed=123,
        psr_freeze_base=False,
    )
    tr._expl_nest_mu, tr._en_delta, tr._psr_step = 0.0, None, 0
    tr.optimizer = optimizers.AdamW(learning_rate=1e-3, weight_decay=0.1)
    tr.psr_optimizer = (
        optimizers.AdamW(learning_rate=2e-3, weight_decay=0.02)
        if model.config.psr_enabled
        else None
    )
    tr._loss_and_grad = tr._build_loss_and_grad()
    return tr


def test_t01_t02_t06_off_shared_weights_and_zero_initialization():
    base, model = pair()
    x = batch()[0]
    off = model(
        x,
        psr_mode="off",
        thinking_targets={"invalid": None},
        thinking_options={"rounds": 0},
    )
    assert bool(mx.array_equal(off.logits, base(x).logits))
    assert off.thinking_state is None
    apply_trunc_normal_init(model, model.config.dim)
    assert bool(mx.all(model.psr.output.weight == 0))
    enabled = model(x, psr_anchors=mx.array([[1], [1]]))
    assert bool(mx.array_equal(enabled.logits, model(x, psr_mode="off").logits))
    # Construction also keeps baseline RNG consumption unchanged, but pairing
    # always copies/checks shared tensors instead of relying on this convenience.
    mx.random.seed(99)
    a = VibyForCausalLM(config(psr_enabled=False))
    ra = mx.random.uniform(shape=(8,))
    mx.random.seed(99)
    b = VibyForCausalLM(config())
    rb = mx.random.uniform(shape=(8,))
    assert bool(mx.array_equal(ra, rb))
    for k, v in tree_flatten(a.parameters()):
        assert bool(mx.array_equal(v, dict(tree_flatten(b.parameters()))[k]))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["off", "recurrent"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_t03_ten_protected_updates_match_parameters_optimizer_and_moe(
    compiled, mode, dtype
):
    base, model = pair()
    base.set_dtype(dtype)
    model.set_dtype(dtype)
    a, b = trainer(base, compiled), trainer(model, compiled, mode)
    # Copy optimizer moments explicitly after a shared warmup update.
    outputs, g = a._compute_loss_and_grad(*batch())
    a._optimizer_step(g, 1, outputs[2])
    model.load_weights(tree_flatten(base.parameters()), strict=False)
    b.optimizer.state = tree_map(lambda v: mx.array(v), a.optimizer.state)
    for _ in range(10):
        oa, ga = a._compute_loss_and_grad(*batch())
        ob, gb = b._compute_loss_and_grad(*batch())
        a._optimizer_step(ga, 1, oa[2])
        b._optimizer_step(gb, 1, ob[2])
        shared = dict(tree_flatten(model.parameters()))
        for k, v in tree_flatten(base.parameters()):
            assert max_abs_diff(v, shared[k]) < 2e-6, k
        sa, sb = (
            dict(tree_flatten(a.optimizer.state)),
            dict(tree_flatten(b.optimizer.state)),
        )
        assert sa.keys() == sb.keys()
        for k in sa:
            assert max_abs_diff(sa[k], sb[k]) < 2e-6, k
        assert max_abs_diff(base.moe_bias_stack(), model.moe_bias_stack()) < 2e-6


def test_t04_t05_corrected_gradient_isolation_and_two_stage_zero_head():
    _, model = pair()
    x, y, mask, pad, seg = batch()
    anchors = mx.array([[1], [1]])

    def loss(m):
        return m(
            x,
            labels=y,
            loss_mask=mask,
            attention_mask=pad,
            segment_ids=seg,
            psr_anchors=anchors,
        ).corrected_loss

    _, grads = nn.value_and_grad(model, loss)(model)
    flat = dict(tree_flatten(grads))
    assert float(mx.max(mx.abs(flat["psr.output.weight"]))) > 0
    for k, v in flat.items():
        if k != "psr.output.weight":
            assert float(mx.max(mx.abs(v))) == 0, k
    opt = optimizers.SGD(learning_rate=0.1)
    opt.update(ParameterView(model, True), {"psr": grads["psr"]})
    _, grads = nn.value_and_grad(model, loss)(model)
    flat = dict(tree_flatten(grads))
    assert float(mx.max(mx.abs(flat["psr.read_query.weight"]))) > 0
    assert float(mx.max(mx.abs(flat["psr.blocks.0.up.weight"]))) > 0
    for k, v in flat.items():
        if not k.startswith("psr."):
            assert float(mx.max(mx.abs(v))) == 0, k


def test_t07_t08_t09_t15_future_doc_pad_and_nonbridge_isolation():
    _, model = pair()
    model.psr.output.weight = mx.random.normal(model.psr.output.weight.shape) * 0.02
    x, y, mask, pad, seg = batch()
    anchors = mx.array([[1], [1]])
    a = model(
        x,
        psr_anchors=anchors,
        segment_ids=seg,
        attention_mask=pad,
        return_thinking=True,
    )
    changed = mx.where(mx.arange(8)[None, :] > 1, x + 10, x)
    b = model(
        changed,
        psr_anchors=anchors,
        segment_ids=seg,
        attention_mask=pad,
        return_thinking=True,
    )
    assert max_abs_diff(a.thinking_state.slots, b.thinking_state.slots) == 0
    assert max_abs_diff(a.logits[:, :2], b.logits[:, :2]) == 0
    off = model(x, psr_mode="off", segment_ids=seg, attention_mask=pad)
    assert max_abs_diff(a.logits[:, 0], off.logits[:, 0]) == 0
    assert max_abs_diff(a.logits[:, 5:], off.logits[:, 5:]) == 0
    assert max_abs_diff(a.logits[:, 1:5], off.logits[:, 1:5]) > 0
    # The state/correction of document 1 cannot affect document 2.
    assert max_abs_diff(a.logits[0, 6:], off.logits[0, 6:]) == 0
    none = model(
        mx.zeros((1, 3), mx.int32),
        labels=mx.zeros((1, 3), mx.int32),
        loss_mask=mx.zeros((1, 3)),
        attention_mask=mx.zeros((1, 3)),
        psr_anchors=mx.array([[-1]]),
    )
    assert (
        float(none.lm_loss) == 0
        and float(none.corrected_loss) == 0
        and bool(mx.isfinite(none.loss))
    )


def test_t10_t11_same_horizon_explicit_modes_and_fresh_cache():
    _, model = pair()
    x = batch()[0][:1]
    model.eval()
    model.psr.output.weight = mx.random.normal(model.psr.output.weight.shape) * 0.02
    for rounds in (1, 2, 4):
        out = model(
            x,
            psr_anchors=mx.array([[2]]),
            thinking_options={"rounds": rounds},
            return_thinking=True,
        )
        assert (
            out.thinking_state.horizon == 4
            and out.thinking_state.anchor.tolist() == [[2]]
        )
        assert out.thinking_state.rounds == rounds
    with pytest.raises(ValueError, match="R>=1"):
        model(x, psr_anchors=mx.array([[2]]), thinking_options={"rounds": 0})
    state = model(
        x, psr_anchors=mx.array([[2]]), psr_mode="state_only", return_thinking=True
    )
    assert state.thinking_state.rounds == 0
    assert max_abs_diff(state.logits, model(x, psr_mode="off").logits) > 0
    _, cache = model.prefill(x[:, :4], psr_mode="state_only")
    with pytest.raises(ValueError, match="fresh cache"):
        model(x[:, :1], cache=cache, psr_mode="off")


def test_t12_t16_t17_dense_sparse_bound_and_true_fixed_scan_count():
    _, model = pair()
    x = batch()[0][:1]
    ids = mx.array([[[[0, 1], [0, 1]]]])
    audit = model(
        x,
        psr_anchors=mx.array([[3]]),
        thinking_options={
            "read_mode": "sparse_diagnostic",
            "fixed_indices": ids,
            "min_rho": 0.01,
        },
        return_thinking=True,
    )
    tr = audit.thinking_trace
    assert tr.full_scans == 1
    assert bool(mx.all(tr.read_error[0] <= tr.error_bound[0] + 1e-5))
    assert bool(mx.all((tr.rho[0] >= 0) & (tr.rho[0] <= 1)))
    fixed = model(
        x,
        psr_anchors=mx.array([[3]]),
        thinking_options={"read_mode": "fixed", "fixed_indices": ids},
        return_thinking=True,
    )
    assert fixed.thinking_trace.full_scans == 0
    with pytest.raises(ValueError, match="indices"):
        model(x, psr_anchors=mx.array([[3]]), thinking_options={"read_mode": "fixed"})


def test_t13_t14_t18_cache_invariance_compilation_runtime_gate_and_no_labels():
    base, model = pair()
    base.eval()
    model.eval()
    model.psr.output.weight = mx.random.normal(model.psr.output.weight.shape) * 0.02
    x = batch()[0][:1]
    full = model(x, psr_anchors=mx.array([[3, 7]]))
    first, cache = model.prefill(x[:, :4])
    _, base_cache = base.prefill(x[:, :4])
    outputs = [first]
    for t in range(4, 8):
        token, _ = model.decode_step(x[:, t], cache)
        base.decode_step(x[:, t], base_cache)
        outputs.append(token[:, None])
    assert max_abs_diff(full.logits, mx.concatenate(outputs, 1)) < 2e-5
    for ca, cb in zip(cache.layers, base_cache.layers):
        for key in ("window", "compress_kv", "index_k"):
            if getattr(ca, key) is not None:
                assert max_abs_diff(getattr(ca, key), getattr(cb, key)) < 2e-5

    def forward(params, ids, gate):
        model.update(params)
        return model(ids, psr_anchors=mx.array([[3, 7]]), psr_gate=gate).logits

    fn = mx.compile(forward)
    params = model.trainable_parameters()
    a = fn(params, x, mx.array(0.0))
    b = fn(params, x, mx.array(1.0))
    model.update(params)
    assert max_abs_diff(a, model(x, psr_mode="off").logits) < 2e-5
    assert max_abs_diff(a, b) > 0
    assert max_abs_diff(b, full.logits) < 2e-5


def test_layout_sampling_not_content_and_full_coverage():
    x, y, mask, pad, seg = batch()
    a, w = anchor_plan(x, pad, seg, 4, count=2, key=mx.array(71, mx.uint32))
    b, _ = anchor_plan(x + 99, pad, seg, 4, count=2, key=mx.array(71, mx.uint32))
    assert bool(mx.array_equal(a, b))
    full = compact_anchor_plan(x, pad, seg, 4)
    assert full.tolist() == [[0, 4, 6], [0, 4, -1]]
    assert w.tolist() == [1.5, 1.0]
    assert "thinking_targets" not in text_psr_inputs(config(), x, y, mask, pad, seg)


def test_independent_nan_handling_and_baseline_lr_groups():
    base, model = pair()
    a, b = trainer(base), trainer(model)
    oa, ga = a._compute_loss_and_grad(*batch())
    ob, gb = b._compute_loss_and_grad(*batch())
    gb["psr"]["output"]["weight"] = mx.full_like(gb["psr"]["output"]["weight"], mx.nan)
    a._optimizer_step(ga, 1, oa[2])
    b._optimizer_step(gb, 1, ob[2])
    for k, v in tree_flatten(base.parameters()):
        assert max_abs_diff(v, dict(tree_flatten(model.parameters()))[k]) < 2e-6
    assert bool(mx.all(model.psr.output.weight == 0))


def test_t03_actual_muon_parameter_groups_and_ten_updates():
    from trainer.muon import create_mixed_optimizer
    from trainer.config import get_pretrain_parser

    base, model = pair()
    a, b = trainer(base), trainer(model)
    args = get_pretrain_parser().parse_args(["--learning_rate", "0.0001"])
    a.optimizer = create_mixed_optimizer(base, args)
    b.optimizer = create_mixed_optimizer(ParameterView(model), args)
    for _ in range(10):
        oa, ga = a._compute_loss_and_grad(*batch())
        ob, gb = b._compute_loss_and_grad(*batch())
        a._optimizer_step(ga, 1, oa[2])
        b._optimizer_step(gb, 1, ob[2])
        shared = dict(tree_flatten(model.parameters()))
        for key, value in tree_flatten(base.parameters()):
            assert max_abs_diff(value, shared[key]) < 2e-6, key
        sa, sb = (
            dict(tree_flatten(a.optimizer.state)),
            dict(tree_flatten(b.optimizer.state)),
        )
        assert sa.keys() == sb.keys()
        for key in sa:
            assert max_abs_diff(sa[key], sb[key]) < 2e-6, key


def test_t16_generation_block_refresh_matches_full_plan_and_actual_reads():
    _, model = pair()
    model.eval()
    model.psr.output.weight = mx.random.normal(model.psr.output.weight.shape) * 0.02
    x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    full = model(x, psr_anchors=mx.array([[3, 7, 11]]))
    first, cache = model.prefill(x[:, :4])
    outputs = [first]
    for t in range(4, 13):
        step, _ = model.decode_step(x[:, t], cache)
        outputs.append(step[:, None])
    assert cache.psr_phases == 3
    assert cache.thinking_state.anchor.tolist() == [[11]]
    assert max_abs_diff(full.logits, mx.concatenate(outputs, 1)) < 2e-5


def test_padded_anchor_bfloat16_backward_is_finite():
    # T<H creates one actual anchor and one padding anchor, as in the CLI probe.
    model = build(config(psr_horizon=16, vocab_size=6400))
    model.set_dtype(mx.bfloat16)
    tr = trainer(model, compiled=True)
    for _ in range(2):
        out, grads = tr._compute_loss_and_grad(*batch())
        mx.eval(out, grads)
        for key, value in tree_flatten(grads):
            assert bool(mx.all(mx.isfinite(value))), key
        tr._optimizer_step(grads, 1, out[2])
    assert float(mx.max(mx.abs(model.psr.output.weight))) > 0
