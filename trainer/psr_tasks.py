"""Executable pointer tasks. Teacher trajectories are returned ONLY as targets."""

import hashlib
import numpy as np
import mlx.core as mx


def pointer_batch(seed, batch_size, nodes, depth, rounds, slots):
    """Memory position i holds pointer table[i]; query gives start and depth.

    Tokens 1..nodes encode node IDs; 0 is padding. Input contains no execution
    trace or final answer. Address targets are physical prefix positions.
    Behavior U=0 tests the current node; U=1 tests whether it is the start node.
    U=1 labels are retained for held-out behavior diagnostics, not trained here.
    """
    rng = np.random.default_rng(seed)
    table = rng.integers(nodes, size=(batch_size, nodes), dtype=np.int32)
    start = rng.integers(nodes, size=batch_size, dtype=np.int32)
    inputs = np.concatenate(
        [
            table + 1,
            np.full((batch_size, 1), nodes + 1),
            start[:, None] + 1,
            np.full((batch_size, 1), nodes + 2 + depth),
        ],
        axis=1,
    ).astype(np.int32)
    node = start.copy()
    trajectory = [node.copy()]
    for _ in range(max(rounds, depth)):
        node = table[np.arange(batch_size), node]
        trajectory.append(node.copy())
    answer = trajectory[depth] + 1
    address = (
        np.stack(trajectory[:rounds], axis=1)
        if rounds
        else np.empty((batch_size, 0), dtype=np.int32)
    )
    # Once the requested computation is complete, no further address is required.
    if rounds > depth:
        address[:, depth:] = -1
    states = np.stack(
        [trajectory[min(r, depth)] + 1 for r in range(rounds + 1)], axis=1
    )
    labels = np.zeros_like(inputs)
    labels[:, -1] = answer
    mask = np.zeros_like(inputs, dtype=np.float32)
    mask[:, -1] = 1
    return {
        "input_ids": mx.array(inputs),
        "labels": mx.array(labels),
        "loss_mask": mx.array(mask),
        "answer": mx.array(answer),
        "prefix_length": inputs.shape[1],
        "targets": {
            "address": mx.array(np.repeat(address[..., None], slots, axis=-1)),
            "tests": mx.zeros((batch_size, rounds + 1, 1), dtype=mx.int32),
            "results": mx.array(states[..., None]),
        },
        "heldout_tests": mx.ones((batch_size, rounds + 1, 1), dtype=mx.int32),
        "heldout_results": mx.array(
            ((states - 1) == start[:, None]).astype(np.int32)[..., None]
        ),
        "identity": hashlib.sha256(inputs.tobytes() + answer.tobytes()).hexdigest(),
    }
