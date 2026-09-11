"""Bounded E2 mechanism harness: frozen baseline, dense direct NTP residual.

Inputs are a baseline checkpoint + config/sidecar and explicit pretokenized NPZ
training data. Use a separate evaluation NPZ with experiments/psr_evaluate.py.
No claim of joint-pretraining efficiency follows from this frozen-base probe.
"""

import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx import optimizers
from mlx.utils import tree_flatten
import numpy as np
from experiments.psr_evaluate import load_model, sha256
from trainer.psr_pretrain import anchor_plan
from trainer.psr_optim import ParameterView


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config")
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--mode", choices=["state_only", "recurrent"], default="recurrent")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-seconds", type=float, default=120)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if (
        min(args.steps, args.rounds, args.horizon, args.max_seconds, args.learning_rate)
        <= 0
    ):
        p.error("positive limits required")
    # Load base exactly before attaching a new side. Constructor state preservation
    # makes shared initialization predictable, but checkpoint tensors are copied.
    base = load_model(args.checkpoint, args.config)
    from model.config import VibyConfig
    from model.model import VibyForCausalLM

    cfg = VibyConfig.from_dict(
        {
            **base.config.to_dict(),
            "psr_enabled": True,
            "psr_rounds": args.rounds,
            "psr_horizon": args.horizon,
        }
    )
    mx.random.seed(args.seed)
    model = VibyForCausalLM(cfg, skip_init=True)
    model.load_weights(tree_flatten(base.parameters()), strict=False)
    model.eval()
    opt = optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    data = dict(np.load(args.data))
    n = len(data["input_ids"])
    initial = {k: mx.array(v) for k, v in tree_flatten(base.parameters())}

    def loss(params, x, y, mask, pad, seg, anchors, weights, gate):
        model.update(params)
        out = model(
            x,
            labels=y,
            loss_mask=mask,
            attention_mask=pad,
            segment_ids=seg,
            psr_anchors=anchors,
            psr_sample_weights=weights,
            psr_mode=args.mode,
            psr_gate=gate,
            return_metrics=True,
            use_mtp=False,
        )
        return out.corrected_loss, out.metrics

    vg = mx.value_and_grad(loss)
    if args.compile:
        vg = mx.compile(vg)
    rows = []
    begin = time.perf_counter()
    for step in range(args.steps):
        if time.perf_counter() - begin >= args.max_seconds:
            break
        i = step % n
        x = mx.array(data["input_ids"][i : i + 1], mx.int32)
        y = mx.array(data["labels"][i : i + 1], mx.int32)
        mask = mx.array(data["loss_mask"][i : i + 1])
        pad = mx.array(
            data.get("attention_mask", np.ones_like(data["input_ids"]))[i : i + 1]
        )
        seg = (
            mx.array(data["segment_ids"][i : i + 1]) if "segment_ids" in data else None
        )
        anchors, weights = anchor_plan(
            x,
            pad,
            seg,
            cfg.psr_horizon,
            count=cfg.psr_train_anchors,
            key=step + args.seed,
        )
        params = ParameterView(model, True).trainable_parameters()
        (value, metrics), grads = vg(
            params, x, y, mask, pad, seg, anchors, weights, model.psr.calibration_gate
        )
        model.update(params)
        grads, norm = optimizers.clip_grad_norm(grads, 1.0)
        mx.eval(value, metrics, norm)
        if not bool(mx.isfinite(norm)):
            raise RuntimeError("non-finite PSR gradient; stop probe")
        opt.update(ParameterView(model, True), grads)
        mx.eval(model.psr.parameters(), opt.state)
        row = dict(
            microstep=step + 1,
            optimizer_step=step + 1,
            loss=float(value),
            sums=mx.sum(metrics, 0).tolist(),
            elapsed_seconds=time.perf_counter() - begin,
        )
        rows.append(row)
        with (args.out_dir / "train.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    current = dict(tree_flatten(model.parameters()))
    unchanged = all(bool(mx.array_equal(v, current[k])) for k, v in initial.items())
    if not unchanged:
        raise AssertionError("frozen baseline changed")
    model.save_weights(str(args.out_dir / "model.safetensors"))
    (args.out_dir / "model.json").write_text(
        json.dumps({"config": cfg.to_dict()}, indent=2)
    )
    mx.save_safetensors(
        str(args.out_dir / "side_optimizer.safetensors"), dict(tree_flatten(opt.state))
    )
    report = dict(
        steps=len(rows),
        baseline_unchanged=unchanged,
        train_data_sha256=sha256(args.data),
        baseline_checkpoint_sha256=sha256(args.checkpoint),
        seed=args.seed,
        mode=args.mode,
        rounds=args.rounds,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        seconds_including_compile=time.perf_counter() - begin,
        peak_memory_bytes=mx.get_peak_memory(),
        limits="mechanism smoke only; no calibration or held-out efficiency claim",
    )
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
