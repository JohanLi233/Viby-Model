"""Bounded, paired real-text pilot through the actual compiled training update.

This is a small-model launch gate, not large-model token-efficiency acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.thinking import LatentThinking
from model.iterative_thinking import IterativeThinking
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser
from trainer.utils import convert_model_dtype, get_optimizer_steps


class ResetThinking(LatentThinking):
    """Probe-only matched-parameter/compute arm, always resetting cross-token input."""

    def __call__(self, *args, **kwargs):
        kwargs["intervention"] = "reset"
        return super().__call__(*args, **kwargs)


class ResetIterativeThinking(IterativeThinking):
    def __call__(self, *args, **kwargs):
        kwargs["intervention"] = "reset"
        return super().__call__(*args, **kwargs)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def data(source, tokenizer, out, steps, batch, length, source_offset=0):
    tok = Tokenizer.from_file(tokenizer)
    sets = {"train": [], "eval": []}
    need = {"train": steps * batch, "eval": 256}
    seen = set()
    with open(source) as stream, (out / "selected_documents.jsonl").open("x") as raw:
        for line_no, line in enumerate(stream):
            if line_no < source_offset:
                continue
            record = json.loads(line)
            text = record["text"]
            ids = [1] + tok.encode(text, add_special_tokens=False).ids + [2]
            if len(ids) < length + 1:
                continue
            ids = ids[: length + 1]
            digest = hashlib.sha256(np.array(ids, np.int32).tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            split = "eval" if line_no % 5 == 0 else "train"
            if len(sets[split]) >= need[split]:
                continue
            raw.write(
                json.dumps(
                    dict(line=line_no, split=split, text=text, prefix_sha256=digest),
                    ensure_ascii=False,
                )
                + "\n"
            )
            sets[split].append(ids)
            if all(len(sets[k]) == need[k] for k in sets):
                break
    if any(len(sets[k]) != need[k] for k in sets):
        raise ValueError("Insufficient independent long documents")
    sets = {k: np.array(v, np.int32) for k, v in sets.items()}
    np.savez_compressed(out / "data.npz", **sets)
    return sets


def evaluate(m, ids, intervention="normal"):
    m.eval()
    values = []
    for i in range(0, len(ids), 4):
        x = mx.array(ids[i : i + 4])
        res = m(
            x[:, :-1],
            labels=x[:, 1:],
            return_sequence_losses=True,
            thinking_intervention=intervention,
            use_mtp=False,
        )
        mx.eval(res.sequence_losses)
        values.extend(res.sequence_losses.astype(mx.float32).tolist())
    return np.array(values, np.float64)


def paired(a, b):
    delta = a - b
    rng = np.random.default_rng(42)
    means = [
        delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)
    ]
    return dict(
        mean=float(delta.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--thinking-arch",
        choices=("ced_pipeline_v1", "ced_iterative_v2", "ced_iterative_tied_v1"),
        default="ced_pipeline_v1",
    )
    ap.add_argument("--source-offset", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = ap.parse_args()
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    batch, length = 4, 128
    c = VibyConfig(
        preset="tiny",
        dim=128,
        n_layers=6,
        n_heads=4,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=64,
        o_groups=1,
        o_lora_rank=32,
        moe_inter_dim=64,
        n_routed_experts=8,
        n_activated_experts=2,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=16,
        candidate_block_size=4,
        candidate_topk_blocks=4,
        engram_layer_ids=(),
        n_mtp_layers=0,
        vocab_size=6400,
        max_seq_len=length,
        window_size=32,
        thinking_enabled=True,
        thinking_arch=args.thinking_arch,
        thinking_dim=32,
        thinking_scale=0.1,
        qb_stats_rows=512,
    )
    root = Path(__file__).resolve().parents[1]
    sets = data(
        args.data,
        str(root / "model/tokenizer.json"),
        out,
        args.steps,
        batch,
        length,
        args.source_offset,
    )
    order = np.random.default_rng(args.seed).permutation(len(sets["train"]))
    sets["train"] = sets["train"][order]
    np.save(out / "train_order.npy", order)
    dump(
        out / "manifest.json",
        dict(
            args=vars(args),
            config=c.to_dict(),
            batch=batch,
            length=length,
            updates=args.steps,
            labels_per_arm=args.steps * batch * length,
            eval_labels=256 * length,
            seed=args.seed,
            mlx=mx.__version__,
            python=sys.version,
            optimizer="actual mixed Muon/Sinkhorn/FP32 AdamW; no MuonH; no clipping",
            lr=0.001,
            muon_lr=13 / 3 * 0.001,
            beta2=0.9997499061952749,
            eps=9.059340702369018e-16,
            warmup_steps=32,
            accumulation=1,
            compile=True,
            scope="Long-document prefixes, no packing/Engram; pilot is not the 1024-dim training recipe",
            gates={
                "finite": True,
                "grad_max_over_median_max": 20,
                "normal_minus_ced_mean_max": -0.01,
                "normal_minus_reset_mean_max": -0.005,
                "paired_ci_upper_max": 0,
            },
            code_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            tokenizer_sha256=hashlib.sha256(
                (root / "model/tokenizer.json").read_bytes()
            ).hexdigest(),
        ),
    )
    (out / "workspace.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"])
    )
    for folder in ("model", "trainer", "experiments"):
        for src in (root / folder).glob("*.py"):
            if folder == "experiments" and src.name != Path(__file__).name:
                continue
            dest = out / "source" / folder / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
    mx.random.seed(args.seed)
    prototype = VibyForCausalLM(c)
    convert_model_dtype(prototype, args.dtype)
    mx.eval(prototype.parameters())
    initial = [(k, mx.array(v)) for k, v in tree_flatten(prototype.parameters())]
    prototype.save_weights(str(out / "initial.safetensors"))
    summaries = {}
    scores = {}
    arms = (
        ("normal", "reset", "ced", "untied_readout")
        if args.thinking_arch == "ced_iterative_tied_v1"
        else ("normal", "reset", "ced")
    )
    for arm in arms:
        sub = out / arm
        sub.mkdir()
        config = VibyConfig.from_dict(
            dict(
                c.to_dict(),
                thinking_enabled=arm != "ced",
                thinking_arch="ced_iterative_v2"
                if arm == "untied_readout"
                else c.thinking_arch,
            )
        )
        m = VibyForCausalLM(config)
        if arm == "reset":
            cls = (
                ResetIterativeThinking
                if args.thinking_arch in ("ced_iterative_v2", "ced_iterative_tied_v1")
                else ResetThinking
            )
            m.model.thinking = cls(config)
        weights = [
            (k, v)
            for k, v in initial
            if arm != "ced" or not k.startswith("model.thinking.")
        ]
        if arm == "untied_readout":
            # All common parameters match exactly. Only its independent readout
            # is new, with the same Marin initialization and working dtype.
            m.model.thinking.output.set_dtype(getattr(mx, args.dtype))
            missing = set(dict(tree_flatten(m.parameters()))) - set(dict(weights))
            assert missing == {"model.thinking.output.weight"}, missing
            m.load_weights(weights, strict=False)
        else:
            m.load_weights(weights)
        mx.eval(m.parameters())
        assert all(
            np.array_equal(
                np.array(v.astype(mx.float32)),
                np.array(dict(weights)[k].astype(mx.float32)),
            )
            for k, v in tree_flatten(m.parameters())
            if k in dict(weights)
        )
        ta = get_pretrain_parser().parse_args([])
        ta.learning_rate = 0.001
        ta.muon_lr = 13 / 3 * 0.001
        ta.adam_beta2 = 0.9997499061952749
        ta.adam_eps = 9.059340702369018e-16
        ta.accumulation_steps = 1
        ta.grad_clip = 0
        ta.muonh = False
        ta.auto_resume = False
        ta.resume = None
        ta.compile_model = True
        ta.cache_limit_gb = 2
        ta.save_dir = str(sub)
        ta.reset_optimizer = True
        trainer = BaseTrainer(ta, m, None, config)
        start = time.monotonic()
        norms = []
        train_loss = []
        with (sub / "steps.jsonl").open("x") as log:
            for step in range(args.steps):
                m.train()
                factor = min(1.0, (step + 1) / 32)
                for opt in trainer.optimizer.optimizers:
                    opt.learning_rate = opt.base_lr * factor
                x = mx.array(sets["train"][step * batch : (step + 1) * batch])
                mask = mx.ones((batch, length))
                values, grads = trainer._compute_loss_and_grad(
                    x[:, :-1], x[:, 1:], mask, mask
                )
                mx.eval(values, grads)
                loss = float(values[0])
                ce = float(values[3])
                if not np.isfinite(loss):
                    raise FloatingPointError(f"{arm}/{step}: invalid loss")
                norm = trainer._optimizer_step(
                    grads, 1, moe_loads=values[2], moe_qb_margins=values[6]
                )
                if not np.isfinite(norm):
                    raise FloatingPointError(f"{arm}/{step}: invalid gradient")
                norms.append(norm)
                train_loss.append(ce)
                log.write(
                    json.dumps(
                        dict(
                            update=step + 1,
                            labels=(step + 1) * batch * length,
                            loss=loss,
                            ce=ce,
                            grad_norm=norm,
                        )
                    )
                    + "\n"
                )
                log.flush()
                if (step + 1) % 32 == 0:
                    print(
                        json.dumps(
                            dict(arm=arm, update=step + 1, ce=ce, grad_norm=norm)
                        ),
                        flush=True,
                    )
        elapsed = time.monotonic() - start
        scores[arm] = evaluate(m, sets["eval"])
        if not np.isfinite(scores[arm]).all():
            raise FloatingPointError("invalid eval loss")
        if arm == "normal":
            scores["normal_reset"] = evaluate(m, sets["eval"], "reset")
            scores["normal_swap"] = evaluate(m, sets["eval"], "swap")
        n = np.array(norms)
        summaries[arm] = dict(
            eval_nll=float(scores[arm].mean()),
            grad_min=float(n.min()),
            grad_median=float(np.median(n)),
            grad_max=float(n.max()),
            grad_max_over_median=float(n.max() / np.median(n)),
            parameters=sum(v.size for _, v in tree_flatten(m.parameters())),
            optimizer_steps=get_optimizer_steps(trainer.optimizer),
            elapsed_seconds=elapsed,
        )
        m.save_weights(str(sub / "model.safetensors"))
        dump(sub / "config.json", config.to_dict())
        dump(sub / "summary.json", summaries[arm])
        print(json.dumps(summaries[arm]), flush=True)
    np.savez_compressed(out / "per_document.npz", **scores)
    comparisons = {
        k: paired(scores["normal"], scores[k]) for k in scores if k != "normal"
    }
    stable = all(x["grad_max_over_median"] <= 20 for x in summaries.values())
    gate = (
        stable
        and comparisons["ced"]["mean"] <= -0.01
        and comparisons["ced"]["ci95"][1] < 0
        and comparisons["reset"]["mean"] <= -0.005
        and comparisons["reset"]["ci95"][1] < 0
    )
    result = dict(
        summaries=summaries,
        comparisons=comparisons,
        stable=stable,
        passed=gate,
        status="pilot_pass_requires_confirmation"
        if gate
        else "stop_no_large_pretraining",
        scope="Single-seed tiny-model real-text pilot; no production quality/token-efficiency claim",
    )
    dump(out / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
