"""主配方 kernel 优化基准：同状态 ABBA 与真实 accumulation window。

默认对照是当前默认开关，不是「全部 kernel 关闭」。f+b 对照不更新权重；
整窗口每次从冻结快照恢复后再跑两次微批 + clip + optimizer + bias。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.kernel_bench_utils import (
    abba_blocks,
    active_viby_flags,
    append_jsonl,
    apply_default_flags,
    input_identity,
    memory_snapshot,
    param_identity,
    restore_train_state,
    snapshot_train_state,
    summarize_times,
)

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.flops import training_flops_per_token
from trainer.utils import build_model_kwargs, convert_model_dtype, resolve_compute_scaled_hparams


def packed_segments(batch, seq, mean_len, seed):
    mx.random.seed(seed)
    p = 1.0 / float(mean_len)
    cuts = (mx.random.uniform(shape=(batch, seq)) < p).astype(mx.int32)
    cuts = cuts.at[:, 0].add(1 - cuts[:, 0])
    return mx.cumsum(cuts, axis=1).astype(mx.int32)


def make_batch(cfg, batch, seq, seed, seg):
    mx.random.seed(seed)
    x = mx.random.randint(0, cfg.vocab_size, (batch, seq))
    y = mx.random.randint(0, cfg.vocab_size, (batch, seq))
    loss_mask = mx.ones((batch, seq), mx.float32)
    attn_mask = mx.ones((batch, seq), mx.int32)
    seg_ids = packed_segments(batch, seq, seg, seed + 17) if seg else None
    mx.eval(x, y, loss_mask, attn_mask)
    if seg_ids is not None:
        mx.eval(seg_ids)
    return x, y, loss_mask, attn_mask, seg_ids


def build_trainer(args):
    cli = [
        "--out_dir", "research_runs/_bench",
        "--no_save",
        "--batch_size", str(args.batch),
        "--accumulation_steps", str(args.accum),
        "--max_seq_len", str(args.seq),
        "--cache_limit_gb", str(args.cache_limit_gb),
        "--dtype", "bfloat16",
    ]
    if args.no_compile:
        cli.append("--no_compile")
    targs = setup_training_args(get_pretrain_parser().parse_args(cli), "pretrain")
    kwargs = build_model_kwargs(targs)
    cfg = VibyConfig(**kwargs)
    targs = resolve_compute_scaled_hparams(targs, 467617)
    model = VibyForCausalLM(cfg)
    convert_model_dtype(model, getattr(targs, "dtype", ""))
    trainer = BaseTrainer(targs, model, None, cfg, "pretrain")
    return targs, cfg, model, trainer


def eval_fwd_bwd(outputs, grads):
    mx.eval(*[o for o in outputs if o is not None])
    mx.eval(grads)


def run_fb(trainer, batch):
    outputs, grads = trainer._compute_loss_and_grad(*batch)
    eval_fwd_bwd(outputs, grads)
    return outputs, grads


def run_window(trainer, batches):
    accum_grads = None
    last_moe = None
    for batch in batches:
        outputs, grads = trainer._compute_loss_and_grad(*batch)
        eval_fwd_bwd(outputs, grads)
        moe_loads = outputs[2]
        if getattr(moe_loads, "size", 0) > 0:
            last_moe = moe_loads if last_moe is None else last_moe + moe_loads
        accum_grads = grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
        mx.eval(accum_grads)
        if last_moe is not None:
            mx.eval(last_moe)
    trainer._optimizer_step(accum_grads, len(batches), moe_loads=last_moe)


def record_base(path, rec):
    append_jsonl(path, rec)
    print(json.dumps({k: rec[k] for k in rec if k != "blocks"}, ensure_ascii=False, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["aa-fb", "aa-window", "cfg-dump"], default="aa-fb")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seg", type=float, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-iters", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    apply_default_flags()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = run_dir / "benchmarks.jsonl"

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    targs, cfg, model, trainer = build_trainer(args)
    cfg_dump = {
        "dim": cfg.dim,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "head_dim": cfg.head_dim,
        "window_size": cfg.window_size,
        "n_routed_experts": cfg.n_routed_experts,
        "n_activated_experts": cfg.n_activated_experts,
        "moe_inter_dim": getattr(cfg, "moe_inter_dim", None),
        "hc_mult": cfg.hc_mult,
        "index_n_heads": cfg.index_n_heads,
        "index_head_dim": cfg.index_head_dim,
        "index_topk": cfg.index_topk,
        "compress_ratios": list(cfg.compress_ratios),
        "dtype": str(getattr(targs, "dtype", "")),
        "compile_model": bool(trainer._compiled),
        "grad_clip": getattr(targs, "grad_clip", None),
        "accumulation_steps": targs.accumulation_steps,
        "batch_size": targs.batch_size,
        "max_seq_len": targs.max_seq_len,
        "cache_limit_gb": getattr(targs, "cache_limit_gb", None),
        "params": param_identity(model),
        "flags": active_viby_flags(),
        "flops_per_token": training_flops_per_token(model, args.seq),
    }
    (run_dir / "resolved_cfg.json").write_text(json.dumps(cfg_dump, indent=2, ensure_ascii=False) + "\n")
    if args.mode == "cfg-dump":
        print(json.dumps(cfg_dump, indent=2, ensure_ascii=False))
        return

    fb_batch = make_batch(cfg, args.batch, args.seq, args.seed, args.seg)
    window_batches = [
        make_batch(cfg, args.batch, args.seq, args.seed + i, args.seg)
        for i in range(args.accum)
    ]
    ident = input_identity(*[a for batch in [fb_batch, *window_batches] for a in batch if a is not None])

    # 常驻优化器状态（计时外）。
    _, g0 = run_fb(trainer, fb_batch)
    trainer.optimizer.update(model, g0)
    mx.eval(model.parameters(), trainer.optimizer.state)
    del g0
    snap = snapshot_train_state(model, trainer.optimizer)

    tokens_fb = args.batch * args.seq
    tokens_window = args.accum * tokens_fb
    base_rec = {
        "stage": args.mode,
        "shape": {"B": args.batch, "T": args.seq, "accum": args.accum, "seg": args.seg},
        "dtype": "bfloat16",
        "compiled": bool(trainer._compiled),
        "flags": active_viby_flags(),
        "input_hash": ident,
        "cfg": cfg_dump,
        "variant": "baseline",
    }

    if args.mode == "aa-fb":
        def arm():
            run_fb(trainer, fb_batch)

        result = abba_blocks(arm, arm, args.warmup, args.block_iters, args.blocks)
        med = result["A"]["median_s"]
        rec = {
            **base_rec,
            **result,
            "tokens_s_median": tokens_fb / med if med else None,
            "exit": "ok",
        }
        record_base(jsonl, rec)
        (run_dir / "aa_fb.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n")
        print(
            "A/A f+b median %.4fs  %.0f tok/s  drift=%.3f%%  inconclusive=%s"
            % (med, tokens_fb / med, 100 * (result["aa_end_over_start_abs_rel"] or 0),
               result["inconclusive_aa_drift"]),
            flush=True,
        )
        return

    if args.mode == "aa-window":
        def arm():
            run_window(trainer, window_batches)

        def reset():
            restore_train_state(model, trainer.optimizer, snap)

        result = abba_blocks(
            arm, arm, args.warmup, args.block_iters, args.blocks,
            before_a=reset, before_b=reset,
        )
        med = result["A"]["median_s"]
        rec = {
            **base_rec,
            **result,
            "tokens_s_median": tokens_window / med if med else None,
            "memory_after": memory_snapshot(),
            "exit": "ok",
        }
        record_base(jsonl, rec)
        (run_dir / "aa_window.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n")
        print(
            "A/A window median %.4fs  %.0f tok/s  drift=%.3f%%  inconclusive=%s"
            % (med, tokens_window / med, 100 * (result["aa_end_over_start_abs_rel"] or 0),
               result["inconclusive_aa_drift"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
