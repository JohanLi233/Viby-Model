"""Benchmark state must be replayable, including mutable optimizer containers."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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
