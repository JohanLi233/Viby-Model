"""V4.1 组件消融：一次跑完多个结构变体，输出 min 口径 tok/s + MFU + 峰值。

用途是 `research/MLX_PERF.md` §1 SOP 的①定位：先看整步，再把大项按组件拆开，
判断哪个组件吃掉了墙钟（不是 FLOPs——FLOPs 口径见 trainer/flops.py）。

用法：
    .venv/bin/python experiments/ab_v41_components.py --batch 4 --seq 1024
    .venv/bin/python experiments/ab_v41_components.py --only base,hc_mult=2
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402

from model.config import VibyConfig  # noqa: E402
from model.model import VibyForCausalLM  # noqa: E402
from trainer.base_trainer import BaseTrainer  # noqa: E402
from trainer.config import get_pretrain_parser, setup_training_args  # noqa: E402
from trainer.flops import training_flops_per_token  # noqa: E402
from trainer.utils import build_model_kwargs, resolve_compute_scaled_hparams  # noqa: E402

VARIANTS = [
    ("base", {}, True),
    ("no_engram", {"engram_layer_ids": ()}, True),
    ("hc_mult=2", {"hc_mult": 2}, True),
    ("moe_8_exp", {"n_routed_experts": 8}, True),
    ("no_xsa", {"use_xsa": False}, True),
    ("all_sliding", {"compress_ratios": (0,) * 12}, True),
    ("small_index", {"candidate_topk_blocks": 8, "index_topk": 16}, True),
    ("half_layers", {"n_layers": 6}, True),
]


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def run_one(name, cfg_kw, compile_model, batch, seq, iters, warmup):
    cli = [
        "--out_dir",
        "research_runs/_bench",
        "--no_save",
        "--batch_size",
        str(batch),
        "--accumulation_steps",
        "2",
        "--max_seq_len",
        str(seq),
    ]
    if not compile_model:
        cli.append("--no_compile")
    targs = setup_training_args(get_pretrain_parser().parse_args(cli), "pretrain")
    kwargs = build_model_kwargs(targs)
    kwargs.update(cfg_kw)
    cfg = VibyConfig(**kwargs)
    targs = resolve_compute_scaled_hparams(targs, 467617)

    t0 = time.time()
    from trainer.utils import convert_model_dtype

    model = VibyForCausalLM(cfg)
    convert_model_dtype(model, getattr(targs, "dtype", ""))
    trainer = BaseTrainer(targs, model, None, cfg, "pretrain")
    time.time() - t0

    B, T = batch, seq
    X = mx.random.randint(0, cfg.vocab_size, (B, T))
    Y = mx.random.randint(0, cfg.vocab_size, (B, T))
    loss_mask = mx.ones((B, T), dtype=mx.float32)
    attn_mask = mx.ones((B, T), dtype=mx.int32)
    mx.eval(X, Y, loss_mask, attn_mask)

    def step():
        outputs, grads = trainer._compute_loss_and_grad(
            X, Y, loss_mask, attn_mask, None
        )
        mx.eval(*[o for o in outputs if o is not None])
        mx.eval(grads)

    mx.reset_peak_memory()
    t_first = time.time()
    step()  # 首次调用 = mx.compile trace + Metal JIT
    trace_s = time.time() - t_first
    mn, med = bench(step, iters, warmup)
    peak = mx.get_peak_memory() / 1e9
    fpt = training_flops_per_token(model, T)
    tps = B * T / mn
    mfu = tps * fpt / float(targs.peak_tflops) / 1e12 * 100
    print(
        "%-18s %6.0f tok/s  MFU %5.1f%%  min %.3fs med %.3fs  trace %.0fs  "
        "峰值 %5.2fGB  参 %.0fM/%.0fM  FLOPs/tok %.3fG"
        % (
            name,
            tps,
            mfu,
            mn,
            med,
            trace_s,
            peak,
            model.num_parameters() / 1e6,
            cfg.num_active_parameters() / 1e6,
            fpt / 1e9,
        ),
        flush=True,
    )
    del trainer, model, X, Y, loss_mask, attn_mask
    gc.collect()
    mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--only", default="", help="逗号分隔的变体名过滤")
    args = ap.parse_args()

    wanted = [s for s in args.only.split(",") if s]
    print(
        "=== B=%d T=%d iters=%d warmup=%d（min 口径）==="
        % (args.batch, args.seq, args.iters, args.warmup),
        flush=True,
    )
    for name, cfg_kw, compile_model in VARIANTS:
        if wanted and name not in wanted:
            continue
        try:
            run_one(
                name,
                cfg_kw,
                compile_model,
                args.batch,
                args.seq,
                args.iters,
                args.warmup,
            )
        except Exception as e:  # noqa: BLE001
            print("%-18s FAILED %s: %s" % (name, type(e).__name__, e), flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
