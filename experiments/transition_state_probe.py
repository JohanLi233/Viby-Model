"""CPU-only mechanism gate: learned read results become subsequent queries."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def unit(x):
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)


class TransitionReader(nn.Module):
    def __init__(self, nodes=16, dim=64, addressing="learned"):
        super().__init__()
        self.entities = nn.Embedding(nodes, dim)
        self.addressing = addressing
        self.key = (
            nn.Linear(2 * dim, dim, bias=False) if addressing == "learned" else None
        )
        self.value = nn.Linear(2 * dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)

    def prepare(self, keys, values, query):
        facts = mx.concatenate([self.entities(keys), self.entities(values)], axis=-1)
        projected_keys = (
            self.key(facts) if self.key is not None else self.query(self.entities(keys))
        )
        return self.entities(query)[:, None], unit(projected_keys), self.value(facts)

    def step(self, state, keys, values):
        weights = mx.softmax(
            8 * (unit(self.query(state)) @ keys.swapaxes(-1, -2)), axis=-1
        )
        return weights @ values, weights

    def __call__(self, keys, values, query, hops=2, fixed=False, donor=None):
        state, k, v = self.prepare(keys, values, query)
        initial = state
        trace = []
        for hop in range(hops):
            state, weights = self.step(initial if fixed else state, k, v)
            trace.append(weights[:, 0])
            if donor is not None and hop == 0:
                state = state[donor]
        logits = 8 * (unit(state[:, 0]) @ unit(self.entities.weight).T)
        return logits, mx.stack(trace, axis=1)


def worlds(seed, count, nodes=16):
    rng = np.random.default_rng(seed)
    cycle = np.argsort(rng.random((count, nodes)), axis=1)
    edges = np.empty_like(cycle)
    np.put_along_axis(edges, cycle, np.roll(cycle, -1, axis=1), axis=1)
    keys = np.argsort(rng.random((count, nodes)), axis=1)
    return dict(
        keys=keys.astype(np.int32),
        values=np.take_along_axis(edges, keys, axis=1).astype(np.int32),
        query=rng.integers(nodes, size=count, dtype=np.int32),
        edges=edges.astype(np.int32),
    )


def target(data, hops):
    state = data["query"].copy()
    path = []
    for _ in range(hops):
        path.append(state.copy())
        state = data["edges"][np.arange(len(state)), state]
    return state, np.stack(path, axis=1)


def batch(data, start, end, hops):
    y, _ = target({k: v[start:end] for k, v in data.items()}, hops)
    return tuple(mx.array(data[k][start:end]) for k in ("keys", "values", "query")) + (
        mx.array(y),
    )


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(9001)
    means = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)]
    return dict(mean=float(d.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist())


def evaluate(m, data, hops=2, fixed=False, patch=False):
    if patch and hops != 2:
        raise ValueError("Counterfactual audit currently requires exactly two hops")
    losses, predictions, chosen = [], [], []
    answer, path = target(data, hops)
    donors = np.concatenate(
        [
            np.roll(np.arange(i, min(i + 128, len(answer))), -1)
            for i in range(0, len(answer), 128)
        ]
    )
    if patch:
        first = data["edges"][np.arange(len(answer)), data["query"]]
        answer = data["edges"][np.arange(len(answer)), first[donors]]
    for i in range(0, len(answer), 128):
        k, v, q, _ = batch(data, i, i + 128, hops)
        donor = mx.array(donors[i : i + len(q)] - i) if patch else None
        logits, weights = m(k, v, q, hops, fixed, donor)
        ce = nn.losses.cross_entropy(
            logits, mx.array(answer[i : i + len(q)]), reduction="none"
        )
        mx.eval(ce, logits, weights)
        losses.extend(ce.tolist())
        predictions.extend(mx.argmax(logits, axis=-1).tolist())
        ids = np.array(mx.argmax(weights, axis=-1))
        chosen.extend(
            np.take_along_axis(data["keys"][i : i + len(q)], ids, axis=1).tolist()
        )
    predictions = np.array(predictions)
    chosen = np.array(chosen)
    losses = np.array(losses)
    result = dict(
        nll=float(losses.mean()), accuracy=float(np.mean(predictions == answer))
    )
    if not patch:
        result["address_accuracy_by_hop"] = np.mean(chosen == path, axis=0).tolist()
    return result, dict(
        nll=losses, prediction=predictions, answer=answer, chosen=chosen, donor=donors
    )


def audit(m, data):
    k, v, q, y = batch(data, 0, 8, 2)

    def objective(net):
        return nn.losses.cross_entropy(net(k, v, q)[0], y, reduction="mean")

    direct = objective(m)
    value, g = nn.value_and_grad(m, objective)(m)
    mx.eval(direct, value, g)
    state, keys, values = m.prepare(k, v, q)
    state, _ = m.step(state, keys, values)

    def final(z):
        z, _ = m.step(z, keys, values)
        logits = 8 * (unit(z[:, 0]) @ unit(m.entities.weight).T)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    grad = mx.grad(final)(state)
    direction = mx.array(
        np.random.default_rng(19).normal(size=state.shape).astype(np.float32)
    )
    direction /= mx.sqrt(mx.sum(direction**2))
    analytic = mx.sum(grad * direction)
    mx.eval(analytic, grad)
    # FP64 reference prevents FP32 subtraction and large-step curvature from
    # determining the gradient gate. The learned model remains MLX FP32.
    wq, key64, value64, emb = [
        np.array(v).astype(np.float64)
        for v in (m.query.weight, keys, values, m.entities.weight)
    ]
    labels = np.array(y)

    def normalize(z):
        return z / np.sqrt(np.sum(z * z, axis=-1, keepdims=True) + 1e-6)

    def reference(z):
        scores = 8 * (normalize(z @ wq.T) @ key64.swapaxes(-1, -2))
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        h = (weights @ value64)[:, 0]
        logits = 8 * normalize(h) @ normalize(emb).T
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return (
            np.log(np.exp(shifted).sum(axis=-1))
            - shifted[np.arange(len(labels)), labels]
        ).mean()

    state64 = np.array(state).astype(np.float64)
    direction64 = np.array(direction).astype(np.float64)
    fd_errors = []
    for eps in (1e-4, 1e-5):
        fd = (
            reference(state64 + eps * direction64)
            - reference(state64 - eps * direction64)
        ) / (2 * eps)
        fd_errors.append(abs(float(fd) - float(analytic)))
    result = dict(
        primal_gap=abs(float(direct) - float(value)),
        fd_error=max(fd_errors),
        fd_epsilons=[1e-4, 1e-5],
        fd_reference="numpy_float64_same_weights_and_intermediate_state",
        fd_errors=fd_errors,
        state_grad_norm=float(mx.sqrt(mx.sum(grad**2))),
    )
    assert (
        result["primal_gap"] <= 1e-5
        and result["fd_error"] <= 2e-4
        and result["state_grad_norm"] > 1e-8
    ), result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--addressing", choices=("learned", "tied"), default="learned")
    a = p.parse_args()
    mx.set_default_device(mx.cpu)
    out = Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    phases = [
        (1, 128, worlds(10000 + a.seed, 128 * 64)),
        (2, 512, worlds(20000 + a.seed, 512 * 64)),
    ]
    heldout = worlds(30000 + a.seed, 2048)
    for h, _, data in phases:
        np.savez_compressed(out / f"train_{h}hop.npz", **data)
    np.savez_compressed(out / "heldout.npz", **heldout)
    mx.random.seed(a.seed)
    proto = TransitionReader(addressing=a.addressing)
    mx.eval(proto.parameters())
    initial = [(k, mx.array(v)) for k, v in tree_flatten(proto.parameters())]
    proto.save_weights(str(out / "initial.safetensors"))
    before, _ = evaluate(proto, heldout)
    manifest = dict(
        seed=a.seed,
        addressing=a.addressing,
        device="cpu",
        dtype="float32",
        batch=64,
        dim=64,
        nodes=16,
        optimizer="Adam",
        learning_rate=0.002,
        betas=[0.9, 0.999],
        phases=[
            {"hops": h, "updates": n, "answer_labels": n * 64} for h, n, _ in phases
        ],
        initial=before,
        gradient_audit=audit(proto, heldout),
        parameters=sum(v.size for _, v in initial),
        python=sys.version,
        mlx=mx.__version__,
        hardware=platform.platform(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        gate={
            "accuracy_min": 0.9,
            "vs_fixed_pp_min": 0.2,
            "address_min": 0.9,
            "counterfactual_min": 0.8,
            "initial_max": 0.25,
        },
    )
    for name in (
        "experiments/transition_state_probe.py",
        "research/TRANSITION_THINKING_20260914.md",
    ):
        raw = (ROOT / name).read_bytes()
        (out / Path(name).name).write_bytes(raw)
        manifest.setdefault("source_sha256", {})[name] = hashlib.sha256(raw).hexdigest()
    (out / "workspace.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if before["accuracy"] >= 0.25:
        raise RuntimeError("Interface already solves task; invalid learning gate")
    summary = {}
    raw = {}
    for arm in ("recurrent", "fixed_query"):
        m = TransitionReader(addressing=a.addressing)
        m.load_weights(initial)
        mx.eval(m.parameters())
        assert all(
            np.array_equal(np.array(v), np.array(dict(initial)[k]))
            for k, v in tree_flatten(m.parameters())
        )
        opt = optimizers.Adam(learning_rate=0.002)
        elapsed = time.monotonic()
        with (out / f"{arm}_steps.jsonl").open("x") as log:
            for hops, steps, data in phases:

                def objective(net, k, v, q, y):
                    return nn.losses.cross_entropy(
                        net(k, v, q, hops, arm == "fixed_query")[0], y, reduction="mean"
                    )

                vg = nn.value_and_grad(m, objective)
                for step in range(steps):
                    loss, g = vg(m, *batch(data, step * 64, (step + 1) * 64, hops))
                    mx.eval(loss, g)
                    gn = float(mx.sqrt(sum(mx.sum(v * v) for _, v in tree_flatten(g))))
                    if not np.isfinite(float(loss)) or not np.isfinite(gn):
                        raise FloatingPointError(f"{arm}/{hops}/{step}")
                    opt.update(m, g)
                    mx.eval(m.parameters(), opt.state)
                    log.write(
                        json.dumps(
                            dict(
                                hops=hops, step=step + 1, loss=float(loss), grad_norm=gn
                            )
                        )
                        + "\n"
                    )
                    if (step + 1) % 128 == 0:
                        print(
                            json.dumps(
                                dict(
                                    seed=a.seed,
                                    arm=arm,
                                    hops=hops,
                                    step=step + 1,
                                    loss=float(loss),
                                )
                            ),
                            flush=True,
                        )
        m.eval()
        summary[arm] = {}
        for hops in (1, 2, 4, 8):
            result, samples = evaluate(m, heldout, hops, arm == "fixed_query")
            summary[arm][str(hops)] = result
            for k, v in samples.items():
                raw[f"{arm}_{hops}_{k}"] = v
        if arm == "recurrent":
            result, samples = evaluate(m, heldout, 2, patch=True)
            summary["counterfactual"] = result
            for k, v in samples.items():
                raw[f"counterfactual_{k}"] = v
        summary[arm]["elapsed_seconds"] = time.monotonic() - elapsed
        m.save_weights(str(out / f"{arm}.safetensors"))
        print(json.dumps({arm: summary[arm]}), flush=True)
    delta = paired(raw["recurrent_2_nll"], raw["fixed_query_2_nll"])
    r, f = summary["recurrent"]["2"], summary["fixed_query"]["2"]
    passed = (
        r["accuracy"] >= 0.9
        and r["accuracy"] - f["accuracy"] >= 0.2
        and delta["ci95"][1] < 0
        and min(r["address_accuracy_by_hop"]) >= 0.9
        and summary["counterfactual"]["accuracy"] >= 0.8
    )
    result = dict(
        summary=summary,
        recurrent_minus_fixed=delta,
        passed=bool(passed),
        status="mechanism_screen_pass_needs_confirmation"
        if passed
        else "stop_this_parameterization",
        scope="Structured relation interface with one-hop curriculum; not Viby or language-model token efficiency",
    )
    np.savez_compressed(out / "per_world.npz", **raw)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
