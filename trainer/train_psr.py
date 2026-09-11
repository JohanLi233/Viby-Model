"""Bounded, reproducible PSR mechanism training; no tokenizer/download required.

Run: .venv/bin/python -m trainer.train_psr --out-dir research_runs/psr_demo
This is a synthetic research harness, not evidence of language-task improvement.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn, optimizers

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.psr import counterfactual_value_targets, psr_losses
from trainer.psr_tasks import pointer_batch


def tiny_config(nodes, budgets, bridge):
    return VibyConfig(preset="tiny", dim=64, n_heads=2, o_groups=1, head_dim=32,
                      rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, moe_inter_dim=32,
                      n_routed_experts=4, n_activated_experts=2, engram_layer_ids=(),
                      n_mtp_layers=0, vocab_size=nodes + max(budgets) + 4,
                      max_seq_len=max(nodes + 8, 32), window_size=8,
                      psr_enabled=True, psr_slots=8, psr_dim=64, psr_topk=4,
                      psr_rounds=min(budgets), psr_max_rounds=max(budgets),
                      psr_bridge_init=bridge, psr_test_classes=nodes + 1,
                      psr_cost_weight=0.01)


def restricted_logits(model, slots):
    tests = mx.zeros((slots.shape[0], 1, 1), dtype=mx.int32)
    return model.model.reasoner.predictive_logits(slots[:, None], tests)[:, 0, 0]


def evaluate(model, batch, rounds, selection="adaptive"):
    opts = {"rounds": rounds, "selection_mode": selection}
    # Warm the same path, then time a synchronized complete CED + PSR + decoder
    # + vocabulary readout. No generation/queue overhead is claimed here.
    def forward():
        return model(batch["input_ids"], thinking_prefix_lengths=batch["prefix_length"],
                     thinking_options=opts, use_mtp=False, return_thinking=True)
    out = forward()
    mx.eval(out.logits, out.thinking_state.slots)
    start = time.perf_counter()
    out = forward()
    mx.eval(out.logits, out.thinking_state.slots)
    elapsed = time.perf_counter() - start
    pred = mx.argmax(out.logits[:, -1], axis=-1)
    restricted = restricted_logits(model, out.thinking_state.slots)
    heldout = model.model.reasoner.predictive_logits(out.thinking_trace.states, batch["heldout_tests"])
    exact = mx.mean((pred == batch["answer"]).astype(mx.float32))
    state_exact = mx.mean((mx.argmax(restricted, -1) == batch["answer"]).astype(mx.float32))
    unseen = mx.mean((mx.argmax(heldout, -1) == batch["heldout_results"]).astype(mx.float32))
    reads = sum(int(mx.sum(idx >= 0)) for idx in out.thinking_trace.indices if idx is not None)
    certificates = [p for p in out.thinking_trace.reuse_possible if p is not None]
    return dict(rounds=rounds, selection=selection, accuracy=float(exact),
                restricted_accuracy=float(state_exact), untrained_test_accuracy=float(unseen),
                prefill_seconds=elapsed, full_scans=out.thinking_trace.full_scans,
                selected_slot_positions=reads, cost_units=float(out.thinking_trace.actual_cost),
                diagnostic_reuse_fraction=float(mx.mean(mx.stack(certificates).astype(mx.float32))) if certificates else 0,
                identity=batch["identity"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--value-updates", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-predictive", action="store_true")
    parser.add_argument("--fixed-selection", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if (min(args.budgets) < 1 or max(args.budgets) > 8 or args.nodes < 2
            or min(args.steps, args.batch_size, args.eval_size) < 1
            or args.warmup_steps < 0 or args.value_updates < 0
            or args.learning_rate <= 0 or args.max_seconds <= 0):
        parser.error("invalid task, budget, or training limits")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.checkpoint:
        cfg = VibyConfig.from_pretrained(str(args.checkpoint))
        if not cfg.psr_enabled or cfg.psr_max_rounds < max(args.budgets):
            parser.error("checkpoint must enable PSR and cover all requested budgets")
        if cfg.vocab_size <= args.nodes + 2 + max(args.budgets) or cfg.psr_test_classes <= args.nodes:
            parser.error("checkpoint vocabulary/test classes are too small for the task")
    else:
        cfg = tiny_config(args.nodes, args.budgets, 0.0)
    if args.no_predictive:
        cfg.psr_predictive_weight = 0.0
    mx.random.seed(args.seed)
    model = VibyForCausalLM(cfg)
    if args.checkpoint:
        model.load_weights(str(args.checkpoint / "model.safetensors"))
    opt = optimizers.Adam(learning_rate=args.learning_rate)
    model.train()
    selection = "fixed" if args.fixed_selection else "adaptive"
    cfg.save_pretrained(str(args.out_dir))
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    mlx=mx.__version__, platform=platform.platform(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], text=True),
                    parameters=model.num_parameters(), policy_calibrated=False,
                    timing_scope="synchronized full prefill; excludes generation and request scheduling")
    root = Path(__file__).resolve().parents[1]
    source_paths = ("model/psr.py", "model/model.py", "model/config.py", "model/cache.py",
                    "trainer/psr_tasks.py", "trainer/train_psr.py")
    manifest["source_sha256"] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    functions = {}
    def make_step(rounds, warmup):
        def loss(m, x, y, mask, address, tests, results):
            out = m(x, labels=y, loss_mask=mask, use_mtp=False,
                    thinking_prefix_lengths=x.shape[1],
                    thinking_options={"rounds": rounds, "selection_mode": selection},
                    thinking_targets={"address": address, "tests": tests, "results": results})
            return out.psr_losses["total"] if warmup else out.loss
        vg = nn.value_and_grad(model, loss)
        def step(*arrays):
            loss, grads = vg(model, *arrays)
            opt.update(model, grads)
            return loss
        if args.compile:
            return mx.compile(step, inputs=[model.state, opt.state], outputs=[model.state, opt.state])
        return step
    started = time.perf_counter()
    log = []
    for step in range(args.steps):
        if time.perf_counter() - started >= args.max_seconds:
            break
        warmup = step < args.warmup_steps
        if step == args.warmup_steps:
            for bridge in model.model.workspace_bridges:
                bridge.gate = mx.array(0.05)
        rounds = args.budgets[step % len(args.budgets)]
        batch = pointer_batch(args.seed + step, args.batch_size, args.nodes, rounds, rounds, cfg.psr_slots)
        key = rounds, warmup
        if key not in functions:
            functions[key] = make_step(*key)
        t = batch["targets"]
        loss = functions[key](batch["input_ids"], batch["labels"], batch["loss_mask"],
                              t["address"], t["tests"], t["results"])
        mx.eval(loss, model.parameters(), opt.state)
        value = float(loss)
        if not bool(mx.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss at step {step}; run retained")
        row = dict(step=step, rounds=rounds, phase="warmup" if warmup else "bridge",
                   loss=value, seconds=time.perf_counter() - started, identity=batch["identity"])
        log.append(row)
        with (args.out_dir / "train.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        if step % 10 == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)
    # Optional finite-horizon value fitting on own-state rollouts. Representation
    # is held fixed by optimizing only the value loss on detached states.
    value_opt = optimizers.Adam(learning_rate=args.learning_rate)
    model.eval()
    value_steps = 0
    for step in range(args.value_updates):
        if time.perf_counter() - started >= args.max_seconds:
            break
        rounds = args.budgets[step % len(args.budgets)]
        batch = pointer_batch(args.seed + 50000 + step, args.batch_size, args.nodes, rounds, rounds, cfg.psr_slots)
        out = model(batch["input_ids"], thinking_prefix_lengths=batch["prefix_length"],
                    thinking_options={"rounds": rounds}, use_mtp=False, return_thinking=True)
        trace = out.thinking_trace
        boundary = model.model.layers[cfg.n_encoder_layers].attn
        def terminal(slots):
            return nn.losses.cross_entropy(restricted_logits(model, slots), batch["answer"], reduction="none")
        targets = counterfactual_value_targets(model.model.reasoner, trace, boundary.freq_cos, boundary.freq_sin, terminal)
        states = mx.stop_gradient(trace.states)
        mx.eval(states, targets)
        def value_loss(m):
            from dataclasses import replace
            values = mx.stack([m.model.reasoner.action_values(states[:, r], rounds - r)
                               for r in range(rounds + 1)], axis=1)
            return psr_losses(m.model.reasoner, replace(trace, values=values), {"values": targets})["value"]
        loss, grads = nn.value_and_grad(model, value_loss)(model)
        value_opt.update(model, grads)
        mx.eval(loss, model.parameters(), value_opt.state)
        if not bool(mx.isfinite(loss)):
            raise RuntimeError("non-finite value loss; run retained")
        value_steps += 1
        with (args.out_dir / "value.jsonl").open("a") as file:
            file.write(json.dumps(dict(step=step, loss=float(loss), rounds=rounds)) + "\n")
    train_seconds = time.perf_counter() - started
    evaluations = []
    for depth in args.budgets:
        for rounds in [0] + sorted(set(args.budgets)):
            batch = pointer_batch(args.seed + 100000 + depth, args.eval_size, args.nodes, depth, rounds, cfg.psr_slots)
            for mode in ("adaptive", "fixed"):
                row = evaluate(model, batch, rounds, mode)
                row["depth"] = depth
                evaluations.append(row)
    model.save_weights(str(args.out_dir / "model.safetensors"))
    report = dict(training_steps=len(log), value_updates=value_steps, training_seconds=train_seconds,
                  peak_memory_bytes=mx.get_peak_memory(), evaluations=evaluations,
                  policy_calibrated=False,
                  limitations=["untrained test ID is a diagnostic, not a sufficiency proof",
                               "fixed-selection ablation still scans to keep the training score path identical",
                               "costs are configured units, not calibrated milliseconds",
                               "no matched-parameter ordinary-compute or explicit-CoT baseline",
                               "finite-horizon value fitting is not held-out policy calibration"])
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "evaluations"}), flush=True)


if __name__ == "__main__":
    main()
