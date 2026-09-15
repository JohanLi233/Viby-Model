"""Answer-only, structured two-hop screening; not a Viby quality benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import numpy as np

from model.ncp import ConceptBlock
from model.norms import RMSNorm


ARMS = ("recurrent", "fixed_query", "one_read")


def worlds(seed, count, nodes):
    """Independent random single cycles: no fixed points or two-hop self shortcut.

    Facts are encoded independently; no contextual encoder can pre-solve a chain.
    Row order is randomized independently of graph edges and query.
    """
    rng = np.random.default_rng(seed)
    cycles = np.argsort(rng.random((count, nodes)), axis=1)
    edges = np.empty_like(cycles)
    np.put_along_axis(edges, cycles, np.roll(cycles, -1, axis=1), axis=1)
    keys = np.argsort(rng.random((count, nodes)), axis=1)
    values = np.take_along_axis(edges, keys, axis=1)
    query = rng.integers(nodes, size=count)
    middle = edges[np.arange(count), query]
    answer = edges[np.arange(count), middle]
    return dict(
        keys=keys.astype(np.int32),
        values=values.astype(np.int32),
        query=query.astype(np.int32),
        answer=answer.astype(np.int32),
    )


class QueryProbe(nn.Module):
    """One shared existing ConceptBlock, with an optional frozen attention query.

    K=V and pre-norm/FFN follow ConceptBlock. Position zero for all facts is
    deliberate permutation invariance in this structured interface, not CED RoPE.
    """

    def __init__(self, nodes=16, dim=64):
        super().__init__()
        self.entities = nn.Embedding(nodes, dim // 2)
        self.input = nn.Linear(dim // 2, dim, bias=False)
        self.memory_norm = RMSNorm(dim)
        self.block = ConceptBlock(
            SimpleNamespace(dim=dim, ncp_memory_dim=dim, ncp_heads=2, norm_eps=1e-6)
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, nodes, bias=False)

    def prepare(self, keys, values, query):
        memory = self.memory_norm(
            mx.concatenate([self.entities(keys), self.entities(values)], axis=-1)
        )
        initial = self.input(self.entities(query))[:, None]
        return initial, memory

    def step(self, state, memory, query_state=None):
        if query_state is None:
            return self.block(
                state,
                memory,
                mx.zeros(state.shape[:2], mx.int32),
                mx.ones((*state.shape[:2], memory.shape[1]), mx.bool_),
            )
        # Same residual/FFN as ConceptBlock; only attention Q input is changed.
        block = self.block
        b, n, d = state.shape
        q = block.q_norm(
            block.query(block.attn_norm(query_state)).reshape(
                b, n, block.heads, block.width
            )
        ).transpose(0, 2, 1, 3)
        score = q @ memory[:, None].swapaxes(-1, -2) * block.width**-0.5
        out = (
            (mx.softmax(score, axis=-1) @ memory[:, None])
            .transpose(0, 2, 1, 3)
            .reshape(b, n, block.heads * block.width)
        )
        state = state + block.output(out)
        y = block.ffn_norm(state)
        return state + block.down(nn.silu(block.gate(y)) * block.up(y))

    def __call__(self, keys, values, query, arm="recurrent", intervention="own"):
        if arm not in ARMS or intervention not in ("own", "reset", "donor"):
            raise ValueError("Unknown arm/intervention")
        initial, memory = self.prepare(keys, values, query)
        state = self.step(initial, memory)
        if arm != "one_read":
            if intervention == "reset":
                state = initial
            elif intervention == "donor":
                if len(query) < 2:
                    raise ValueError("Donor needs at least two independent worlds")
                # Swap only the computed correction, preserving recipient query.
                delta = state - initial
                donor = mx.concatenate([delta[1:], delta[:1]], axis=0)
                def norm(x):
                    return mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
                state = initial + donor * norm(delta) / mx.maximum(norm(donor), 1e-12)
            state = self.step(state, memory, initial if arm == "fixed_query" else None)
        return self.head(self.norm(state[:, 0]))


def arrays(data, start, end):
    return tuple(
        mx.array(data[k][start:end]) for k in ("keys", "values", "query", "answer")
    )


def paired_interval(delta):
    rng = np.random.default_rng(9001)
    means = np.array(
        [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)]
    )
    return dict(
        mean=float(delta.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist()
    )


def evaluate(model, data, arm, intervention="own"):
    losses, predictions = [], []
    for start in range(0, len(data["query"]), 128):
        k, v, q, y = arrays(data, start, start + 128)
        logits = model(k, v, q, arm, intervention)
        ce = nn.losses.cross_entropy(logits, y, reduction="none")
        mx.eval(ce, logits)
        losses.append(np.array(ce))
        predictions.append(np.array(mx.argmax(logits, axis=-1)))
    loss, pred = np.concatenate(losses), np.concatenate(predictions)
    if not np.isfinite(loss).all():
        raise FloatingPointError("Non-finite held-out loss")
    return (
        dict(nll=float(loss.mean()), accuracy=float((pred == data["answer"]).mean())),
        loss,
        pred,
    )


def gradient_audit(model, batch):
    k, v, q, y = batch
    initial, memory = model.prepare(k, v, q)
    state = model.step(initial, memory)

    def objective(z):
        logits = model.head(model.norm(model.step(z, memory)[:, 0]))
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    direct = objective(state)
    primal, grad = mx.value_and_grad(objective)(state)
    direction = mx.array(
        np.random.default_rng(18).normal(size=state.shape).astype(np.float32)
    )
    direction /= mx.sqrt(mx.sum(direction * direction))
    eps = 0.01
    fd = (objective(state + eps * direction) - objective(state - eps * direction)) / (
        2 * eps
    )
    analytic = mx.sum(grad * direction)
    mx.eval(direct, primal, grad, fd, analytic)
    result = dict(
        primal_gap=abs(float(direct) - float(primal)),
        state_gradient_norm=float(mx.sqrt(mx.sum(grad * grad))),
        finite_difference=float(fd),
        analytic=float(analytic),
        fd_error=abs(float(fd) - float(analytic)),
    )
    if (
        not all(np.isfinite(x) for x in result.values())
        or result["primal_gap"] > 1e-5
        or result["state_gradient_norm"] <= 1e-8
        or result["fd_error"] > 2e-4
    ):
        raise FloatingPointError(f"Gradient gate failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--eval-worlds", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if (
        args.nodes < 4
        or args.steps < 1
        or args.batch_size < 2
        or args.eval_worlds % 128
    ):
        parser.error(
            "nodes >= 4, steps >= 1, batch >= 2; eval-worlds positive multiple of 128"
        )
    if args.eval_worlds < 128:
        parser.error("eval-worlds must be >= 128")
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    train = worlds(1000 + args.seed, args.steps * args.batch_size, args.nodes)
    heldout = worlds(2000 + args.seed, args.eval_worlds, args.nodes)
    np.savez_compressed(out / "train.npz", **train)
    np.savez_compressed(out / "heldout.npz", **heldout)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "experiments/latent_query_probe.py",
        "model/ncp.py",
        "model/norms.py",
        "research/LATENT_QUERY_20260914.md",
    )
    source_hashes = {}
    for name in sources:
        raw = (root / name).read_bytes()
        (out / Path(name).name).write_bytes(raw)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    (out / "workspace.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    )
    meta = dict(
        config=vars(args),
        command=sys.argv,
        python=sys.version,
        mlx=mx.__version__,
        hardware=platform.platform(),
        dtype="float32",
        compile=False,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_hashes=source_hashes,
        learning_rate=0.002,
        optimizer="Adam, defaults",
        answer_labels_per_arm=len(train["query"]),
        symbolic_prompt_ids_per_arm=len(train["query"]) * (2 * args.nodes + 1),
        natural_language_tokens=0,
        tokenizer=None,
        gate=dict(
            accuracy_min=0.8,
            accuracy_gap_min=0.1,
            nll_gap_max=-0.05,
            donor_nll_gap_min=0.05,
            paired_nll_ci_upper_max=0.0,
        ),
    )
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    mx.random.seed(args.seed)
    prototype = QueryProbe(args.nodes)
    mx.eval(prototype.parameters())
    initial = [
        (key, mx.array(value)) for key, value in tree_flatten(prototype.parameters())
    ]
    np.savez(out / "initial.npz", **{key: np.array(value) for key, value in initial})
    summary, raw = {}, {}
    for arm in ARMS:
        model = QueryProbe(args.nodes)
        model.load_weights(initial)
        mx.eval(model.parameters())
        assert all(
            np.array_equal(np.array(v), np.array(dict(initial)[k]))
            for k, v in tree_flatten(model.parameters())
        )
        optimizer = optimizers.Adam(learning_rate=0.002)

        def loss_fn(m, k, v, q, y):
            return nn.losses.cross_entropy(m(k, v, q, arm), y, reduction="mean")

        value_grad = nn.value_and_grad(model, loss_fn)
        audit = gradient_audit(model, arrays(train, 0, 8))
        started = time.monotonic()
        with (out / f"{arm}_train.jsonl").open("x") as log:
            for step in range(args.steps):
                loss, grads = value_grad(
                    model,
                    *arrays(
                        train, step * args.batch_size, (step + 1) * args.batch_size
                    ),
                )
                finite = mx.all(
                    mx.stack([mx.all(mx.isfinite(g)) for _, g in tree_flatten(grads)])
                )
                mx.eval(loss, finite)
                if not bool(finite) or not np.isfinite(float(loss)):
                    raise FloatingPointError(
                        f"Invalid gradient/loss: {arm} step {step}"
                    )
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                log.write(
                    json.dumps(
                        dict(
                            step=step + 1,
                            answer_labels=(step + 1) * args.batch_size,
                            loss=float(loss),
                        )
                    )
                    + "\n"
                )
                if (step + 1) % 64 == 0:
                    print(
                        json.dumps(dict(arm=arm, step=step + 1, loss=float(loss))),
                        flush=True,
                    )
        model.eval()
        metrics, losses, pred = evaluate(model, heldout, arm)
        summary[arm] = dict(
            **metrics,
            gradient_audit=audit,
            train_seconds=time.monotonic() - started,
            parameters=sum(v.size for _, v in tree_flatten(model.parameters())),
        )
        raw[arm + "_nll"], raw[arm + "_prediction"] = losses, pred
        if arm == "recurrent":
            for intervention in ("donor", "reset"):
                metrics, losses, pred = evaluate(model, heldout, arm, intervention)
                summary[intervention] = metrics
                raw[intervention + "_nll"], raw[intervention + "_prediction"] = (
                    losses,
                    pred,
                )
        model.save_weights(str(out / f"{arm}.safetensors"))
        print(json.dumps(dict(arm=arm, result=summary[arm])), flush=True)
    np.savez_compressed(out / "per_world.npz", **raw)
    comparisons = {
        name: paired_interval(raw["recurrent_nll"] - raw[name + "_nll"])
        for name in ("fixed_query", "one_read", "donor", "reset")
    }
    passed = (
        summary["recurrent"]["accuracy"] >= 0.8
        and summary["recurrent"]["accuracy"] - summary["fixed_query"]["accuracy"] >= 0.1
        and comparisons["fixed_query"]["mean"] <= -0.05
        and comparisons["fixed_query"]["ci95"][1] < 0
        and comparisons["donor"]["mean"] <= -0.05
        and comparisons["donor"]["ci95"][1] < 0
    )
    result = dict(
        summary=summary,
        recurrent_minus_control=comparisons,
        passed=bool(passed),
        status="screen_pass_requires_independent_seeds"
        if passed
        else "stop_no_viby_integration",
        scope="Structured answer-only toy; no CED encoder, packing, cache, tokenizer, Viby quality or token-efficiency evidence.",
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
