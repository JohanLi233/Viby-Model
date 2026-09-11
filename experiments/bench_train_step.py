"""V4.1 训练整步基准：分段计时（fwd / fwd+bwd / optimizer）+ MFU + 峰值显存。

这是 2026-09-10 V4.1 重构把旧 `experiments/bench_train_step.py` 一起删掉之后
重建的计时口径（`research/MLX_PERF.md` §1 的 SOP 第①步）。

用法：
    .venv/bin/python experiments/bench_train_step.py --batch 4 --seq 1024
    .venv/bin/python experiments/bench_train_step.py --cfg hc_mult=2
    .venv/bin/python experiments/bench_train_step.py --cfg engram_layer_ids=
    .venv/bin/python experiments/bench_train_step.py --no-compile --iters 5

计时纪律（`research/MLX_PERF.md` §2.1）：
- MLX 惰性求值，只有 `mx.eval()` 之后的墙钟才算数；
- warmup 之后取 min（短 kernel / 整步口径都用 min）；
- 形状口径 = `--batch × --seq` 的微批，与训练日志的 tokens/s 同口径
  （tokens/s = B·T / (fwd+bwd 墙钟)）。

注意：本脚本走的是 `BaseTrainer._compute_loss_and_grad` 本体，所以
`--compile_model` 的 `mx.compile`、MoE noaux_tc 侧信道、梯度物化顺序都与
真实训练逐步一致；测出来的 tokens/s 可以直接和训练日志对比。
"""

import argparse
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


def parse_cfg(pairs):
    """`--cfg key=value` → VibyConfig kwargs（空 value = 空 tuple，用于关组件）。"""
    kw = {}
    for item in pairs or []:
        key, _, value = item.partition("=")
        if value == "":
            kw[key] = ()
        elif value.lower() in ("true", "false"):
            kw[key] = value.lower() == "true"
        else:
            try:
                kw[key] = int(value)
            except ValueError:
                try:
                    kw[key] = float(value)
                except ValueError:
                    kw[key] = value
    return kw


def bench(fn, iters, warmup, label=""):
    """min-of-N 墙钟；fn 自己负责 mx.eval。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    if os.environ.get("BENCH_VERBOSE"):
        print("  [%s] 逐轮 %s" % (label, " ".join("%.3f" % float(t) for t in times)), flush=True)
    times.sort()
    return times[0], times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=["base", "tiny"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-opt", action="store_true", help="跳过 optimizer 计时")
    ap.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="结构覆盖，可重复：--cfg hc_mult=2 --cfg use_xsa=false",
    )
    ap.add_argument(
        "--seg",
        type=float,
        default=None,
        help="生成 packed 文档边界 segment_ids 的平均文档长度（token）；"
        "默认 None＝不传 segment_ids（比真实 --doc_mask 训练少一层掩码计算）",
    )
    ap.add_argument(
        "--cache-limit-gb",
        type=float,
        default=0.0,
        help="转发给训练器的 --cache_limit_gb（Metal 分配器缓存上限）；0=不限（默认）",
    )
    ap.add_argument(
        "--fwd-only",
        action="store_true",
        help="额外测一次纯前向（compile 同口径），用来拆 fwd / bwd 占比",
    )
    ap.add_argument(
        "--no-opt-state",
        action="store_true",
        help="不计时前不先跑一次 optimizer.update（默认跑，让优化器状态常驻，"
        "接近真实训练的稳态内存）",
    )
    args = ap.parse_args()

    cli = [
        "--out_dir", "research_runs/_bench",
        "--no_save",
        "--batch_size", str(args.batch),
        "--accumulation_steps", str(args.accum),
        "--max_seq_len", str(args.seq),
        "--cache_limit_gb", str(args.cache_limit_gb),
    ]
    if args.no_compile:
        cli.append("--no_compile")
    targs = get_pretrain_parser().parse_args(cli)
    targs = setup_training_args(targs, "pretrain")

    kwargs = build_model_kwargs(targs)
    if args.preset == "tiny":
        kwargs["preset"] = "tiny"
    kwargs.update(parse_cfg(args.cfg))
    cfg = VibyConfig(**kwargs)
    targs = resolve_compute_scaled_hparams(targs, 467617)

    t0 = time.time()
    model = VibyForCausalLM(cfg)
    # 真实训练走 build_model_and_tokenizer → convert_model_dtype(args.dtype)，
    # 之前这里漏了这一步，导致基准一直在测 fp32（也顺带让 bf16-only 的
    # 自定义 kernel 全部被 enabled_for 挡掉）。
    from trainer.utils import convert_model_dtype

    convert_model_dtype(model, getattr(targs, "dtype", ""))
    trainer = BaseTrainer(targs, model, None, cfg, "pretrain")
    build_s = time.time() - t0

    B, T = args.batch, args.seq
    X = mx.random.randint(0, cfg.vocab_size, (B, T))
    Y = mx.random.randint(0, cfg.vocab_size, (B, T))
    loss_mask = mx.ones((B, T), dtype=mx.float32)
    attn_mask = mx.ones((B, T), dtype=mx.int32)
    seg_ids = None
    if args.seg:
        # 模拟 --pack_sequences --doc_mask 的 packed 块：平均每 seg 个 token 换一次文档
        p = 1.0 / float(args.seg)
        cuts = (mx.random.uniform(shape=(B, T)) < p).astype(mx.int32)
        cuts = cuts.at[:, 0].add(1 - cuts[:, 0])
        seg_ids = mx.cumsum(cuts, axis=1).astype(mx.int32)
    mx.eval(X, Y, loss_mask, attn_mask)
    if seg_ids is not None:
        mx.eval(seg_ids)

    fpt = training_flops_per_token(model, T)
    peak_tflops = float(getattr(targs, "peak_tflops", 13.5))
    modes = {}
    for i in range(cfg.n_layers):
        modes[cfg.layer_mode(i)] = modes.get(cfg.layer_mode(i), 0) + 1

    print(
        "配置：dim=%d layers=%d heads=%d×%d experts=%d/%d hc=%d engram=%s "
        "compile=%s B=%d T=%d accum=%d"
        % (
            cfg.dim, cfg.n_layers, cfg.n_heads, cfg.head_dim,
            cfg.n_activated_experts, cfg.n_routed_experts, cfg.hc_mult,
            list(cfg.engram_layer_ids), trainer._compiled, B, T, args.accum,
        )
    )
    print(
        "      逐层模式 %s  参数量 %.1fM/%.1fM  FLOPs/token %.3fG  构建 %.1fs"
        % (
            modes, model.num_parameters() / 1e6,
            cfg.num_active_parameters() / 1e6, fpt / 1e9, build_s,
        ),
        flush=True,
    )

    def run_step():
        outputs, grads = trainer._compute_loss_and_grad(
            X, Y, loss_mask, attn_mask, seg_ids
        )
        mx.eval(*[o for o in outputs if o is not None])
        mx.eval(grads)
        return grads

    if not args.no_opt_state:
        # 先走一次完整的窗口（fwd+bwd + optimizer），让优化器状态真正分配出来：
        # 真实训练从第 2 个窗口起是"优化器状态常驻 + 分配器缓存已长起来"的状态，
        # 冷启动测出来的数会偏乐观。
        _g = run_step()
        trainer.optimizer.update(model, _g)
        mx.eval(model.parameters(), trainer.optimizer.state)

    mx.reset_peak_memory()
    if args.fwd_only:
        # 纯前向（同样 compile 口径），用来把 fwd / bwd 拆开：
        # 反向 = fwd+bwd − fwd，正常应 ≈ 2×fwd，明显更高说明 VJP 里有重算/低效。
        # 必须在**返回值**里带上图输出：函数里直接 mx.eval 的话，mx.compile
        # 追踪时看不到任何返回值，整张图会被 DCE 掉（实测 0.000s）。
        def fwd_impl():
            params = model.trainable_parameters()
            biases = (
                model.moe_bias_stack()
                if hasattr(model, "moe_bias_stack")
                else mx.zeros((0,), dtype=mx.float32)
            )
            out = trainer._loss_and_grad_with_params(
                params, biases, X, Y, loss_mask, attn_mask, seg_ids
            )
            return out[0], out[2]

        fwd_fn = mx.compile(fwd_impl) if trainer._compiled else fwd_impl

        def run_fwd():
            loss, loads = fwd_fn()
            mx.eval(loss, loads)
        f_min, f_med = bench(run_fwd, args.iters, args.warmup, "fwd")
        print(
            "fwd      min %.3fs (med %.3fs)  %6.0f tok/s  峰值 %.2f GB"
            % (f_min, f_med, B * T / f_min, mx.get_peak_memory() / 1e9),
            flush=True,
        )
    fb_min, fb_med = bench(run_step, args.iters, args.warmup, "fwd+bwd")
    peak_fb = mx.get_peak_memory() / 1e9
    tps = B * T / fb_min
    print(
        "fwd+bwd  min %.3fs (med %.3fs)  %6.0f tok/s  MFU %5.1f%%  峰值 %.2f GB"
        % (fb_min, fb_med, tps, tps * fpt / peak_tflops / 1e12 * 100, peak_fb),
        flush=True,
    )

    if not args.no_opt:
        _, grads = trainer._compute_loss_and_grad(X, Y, loss_mask, attn_mask, seg_ids)
        mx.eval(grads)

        def run_opt():
            trainer.optimizer.update(model, grads)
            mx.eval(model.parameters(), trainer.optimizer.state)

        opt_min, _ = bench(run_opt, args.iters, 1)
        window = fb_min * args.accum + opt_min
        print(
            "optimizer min %.3fs  → accum=%d 窗口 %.2fs  %6.0f tok/s  opt 占比 %.1f%%  峰值 %.2f GB"
            % (
                opt_min, args.accum, window, args.accum * B * T / window,
                100 * opt_min / window, mx.get_peak_memory() / 1e9,
            ),
            flush=True,
        )
    print(
        "注：MFU 口径不含优化器 FLOPs（trainer/flops.py），墙钟含；"
        "旧实现 1551M 配方在 bs12×1024 上的历史基线见 research/MLX_PERF.md §9。"
    )


if __name__ == "__main__":
    main()
