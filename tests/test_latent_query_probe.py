"""Scoped CPU checks for the probe's scientific controls."""

import mlx.core as mx
import numpy as np
import pytest

from experiments.latent_query_probe import QueryProbe, arrays, gradient_audit, worlds


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_world_answers_are_two_hop_and_order_independent():
    data = worlds(42, 64, 16)
    for keys, values, query, answer in zip(
        *(data[k] for k in ("keys", "values", "query", "answer"))
    ):
        graph = dict(zip(keys.tolist(), values.tolist()))
        assert graph[graph[query]] == answer
        assert answer != query and graph[query] != query
    other = worlds(43, 64, 16)
    assert not np.array_equal(data["values"], other["values"])


def test_fixed_query_reference_and_permutation_invariance():
    mx.random.seed(5)
    model = QueryProbe()
    k, v, q, _ = arrays(worlds(1, 4, 16), 0, 4)
    initial, memory = model.prepare(k, v, q)
    np.testing.assert_allclose(
        np.array(model.step(initial, memory)),
        np.array(model.step(initial, memory, initial)),
        atol=1e-6,
    )
    for arm in ("recurrent", "fixed_query", "one_read"):
        np.testing.assert_allclose(
            np.array(model(k, v, q, arm)),
            np.array(model(k[:, ::-1], v[:, ::-1], q, arm)),
            atol=2e-6,
        )
    # Reset before second shared block is exactly the one-read computation.
    np.testing.assert_allclose(
        np.array(model(k, v, q, intervention="reset")),
        np.array(model(k, v, q, "one_read")),
        atol=1e-6,
    )


def test_answer_gradient_reaches_intermediate_state_and_donor_changes_output():
    mx.random.seed(9)
    model = QueryProbe()
    batch = arrays(worlds(8, 8, 16), 0, 8)
    gradient_audit(model, batch)
    k, v, q, _ = batch
    own = np.array(model(k, v, q))
    donor = np.array(model(k, v, q, intervention="donor"))
    assert np.isfinite(donor).all() and np.max(np.abs(own - donor)) > 1e-4
    with pytest.raises(ValueError, match="at least two"):
        model(k[:1], v[:1], q[:1], intervention="donor")
