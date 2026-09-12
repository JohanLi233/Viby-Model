"""Benchmark state must be replayable, including mutable optimizer containers."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from types import SimpleNamespace
from mlx.utils import tree_flatten

from experiments.kernel_bench_utils import (
    abba_blocks,
    restore_train_state,
    snapshot_train_state,
)


def test_optimizer_restore_does_not_mutate_snapshot():
    model = nn.Linear(3, 2, bias=False)
    optimizer = optim.MultiOptimizer([optim.Adam(0.01)])
    grads = {"weight": mx.ones_like(model.weight)}
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    snap = snapshot_train_state(model, optimizer)
    frozen = [(p, a.tolist()) for p, a in tree_flatten(snap["optimizer"])]
    results = []
    for _ in range(3):
        restore_train_state(model, optimizer, snap)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        results.append(model.weight.tolist())
        assert [(p, a.tolist()) for p, a in tree_flatten(snap["optimizer"])] == frozen
    assert results[0] == results[1] == results[2]


def test_aa_keeps_both_slot_groups():
    calls = []
    result = abba_blocks(
        lambda: calls.append("same"),
        lambda: calls.append("same"),
        warmup=0,
        block_iters=2,
        n_blocks=1,
    )
    assert len(calls) == 8
    assert result["A"]["n"] == result["B"]["n"] == 4


def test_restores_psr_optimizer_rng_counter_and_lookahead():
    from trainer.psr_optim import ParameterView

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2, bias=False)
            self.psr = nn.Linear(3, 2, bias=False)
            self.config = SimpleNamespace()

    model = Model()
    optimizer = optim.Adam(0.01)
    side = optim.Adam(0.02)
    optimizer.psr_optimizer = side
    main_grad = {"base": {"weight": mx.ones_like(model.base.weight)}}
    side_grad = {"psr": {"weight": mx.ones_like(model.psr.weight)}}
    trainer = SimpleNamespace(
        _psr_step=17, _en_delta={"x": mx.array(0.5)}, args=SimpleNamespace()
    )

    def update():
        optimizer.update(ParameterView(model), main_grad)
        side.update(ParameterView(model, side=True), side_grad)
        mx.eval(model.parameters(), optimizer.state, side.state)

    update()
    saved = snapshot_train_state(model, optimizer, trainer)
    results = []
    for _ in range(3):
        restore_train_state(model, optimizer, saved, trainer)
        assert trainer._psr_step == trainer.args.psr_microstep == 17
        assert float(trainer._en_delta["x"]) == 0.5
        update()
        results.append([(p, a.tolist()) for p, a in tree_flatten(model.parameters())])
        trainer._psr_step += 8
        trainer._en_delta = None
    assert results[0] == results[1] == results[2]
