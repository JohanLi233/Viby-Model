"""P1：真实调用形状下的 kernel 路径归因。

确认 Indexer VJP 是否进入 loss graph，并比较稀疏 Attention 反向与
Indexer 打分（含无效 tile）的可节省整步时间。输出 profile.json / profile.md。

捕获形状不靠 wrap 整图（会改变融合）；按配置静态计数调用次数，孤立 kernel
在配方形状上计时。整步分母来自同进程一次 compiled f+b。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.kernel_bench_utils import (
    active_viby_flags,
    apply_default_flags,
    memory_snapshot,
    param_identity,
    summarize_times,
)

import mlx.core as mx
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.kernels import indexer_score as idxk
from model.kernels import sparse_attention as sa
from model.model import VibyForCausalLM
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.flops import training_flops_per_token
from trainer.utils import (
    build_model_kwargs,
    convert_model_dtype,
    resolve_compute_scaled_hparams,
)


def isolated_indexer(shape, dtype, iters=8, warmup=3):
    b, t, h, d, n = shape["B"], shape["T"], shape["H"], shape["D"], shape["N"]
    q = mx.random.normal((b, t, h, d)).astype(dtype)
    k = mx.random.normal((b, n, d)).astype(dtype)
    w = mx.random.normal((b, t, h)).astype(dtype)
    reach = mx.broadcast_to(
        (mx.arange(n)[None, None, :] <= mx.arange(t)[None, :, None]), (b, t, n)
    )
    mx.eval(q, k, w, reach)
    cot = mx.ones((b, t, n), mx.float32)

    def fwd():
        y = idxk.indexer_score(q, k, w, reach)
        mx.eval(y)

    def fb():
        y, g = mx.vjp(idxk.indexer_score, [q, k, w, reach], [cot])
        mx.eval(y, g)

    idxk.prewarm_indexer_score(h, d, dtype)
    for _ in range(warmup):
        fwd()
    ft = [
        (_t0 := time.perf_counter(), fwd(), time.perf_counter() - _t0)[2]
        for _ in range(iters)
    ]
    for _ in range(warmup):
        fb()
    bt = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fb()
        bt.append(time.perf_counter() - t0)
    empty = 0.0
    if n:
        bk = 16
        tiles = (n + bk - 1) // bk
        r = mx.pad(reach.astype(mx.int32), [(0, 0), (0, 0), (0, tiles * bk - n)])
        r = r.reshape(b, t, tiles, bk)
        empty = float(mx.mean(mx.all(r == 0, axis=-1).astype(mx.float32)))
    return {
        "fwd": summarize_times(ft),
        "fb": summarize_times(bt),
        "causal_empty_tile_frac": empty,
        "reach_density": float(mx.mean(reach.astype(mx.float32))),
    }


def isolated_sparse(shape, window_size, dtype, iters=8, warmup=3):
    b, t, h, d, n = shape["B"], shape["T"], shape["H"], shape["D"], shape["N"]
    q = mx.random.normal((b, t, h, d)).astype(dtype)
    window = mx.random.normal((b, t, d)).astype(dtype)
    compressed = mx.random.normal((b, n, d)).astype(dtype)
    visible = mx.broadcast_to(
        (mx.arange(n)[None, None, :] <= mx.arange(t)[None, :, None]), (b, t, n)
    )
    # 训练 keep 远稀于因果上三角；再留约 topk/N 的可见率作对照。
    keep = mx.random.uniform(shape=visible.shape) < min(1.0, 64 / max(n, 1))
    visible = visible & keep
    sinks = mx.zeros((h,), mx.float32)
    scale = d**-0.5
    mx.eval(q, window, compressed, visible, sinks)
    cot = mx.ones((b, h, t, d), dtype)

    def fn(a, wkv, ckv, s):
        return sa.indexed_attention(
            a, wkv, ckv, visible, None, None, s, window_size, scale
        )

    def fwd():
        y = fn(q, window, compressed, sinks)
        mx.eval(y)

    def fb():
        y, g = mx.vjp(fn, [q, window, compressed, sinks], [cot])
        mx.eval(y, g)

    sa.prewarm_sparse_attention(d, window_size, scale, dtype)
    for _ in range(warmup):
        fwd()
    ft = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fwd()
        ft.append(time.perf_counter() - t0)
    for _ in range(warmup):
        fb()
    bt = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fb()
        bt.append(time.perf_counter() - t0)
    return {
        "fwd": summarize_times(ft),
        "fb": summarize_times(bt),
        "visible_density": float(mx.mean(visible.astype(mx.float32))),
    }


def indexer_grad_report(grads):
    rows = []
    total_abs = 0.0
    indexer_abs = 0.0
    for path, g in tree_flatten(grads):
        mag = float(mx.sum(mx.abs(g.astype(mx.float32))))
        total_abs += mag
        if ".indexer." in path:
            indexer_abs += mag
            rows.append({"path": path, "abs_sum": mag, "shape": list(g.shape)})
    return {
        "indexer_param_abs_sum": indexer_abs,
        "all_param_abs_sum": total_abs,
        "indexer_share_of_abs": indexer_abs / total_abs if total_abs else 0.0,
        "any_nonzero_indexer_grad": indexer_abs > 0.0,
        "indexer_params": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seg", type=float, default=200)
    ap.add_argument("--cache-limit-gb", type=float, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    apply_default_flags()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    cli = [
        "--out_dir",
        "research_runs/_bench",
        "--no_save",
        "--batch_size",
        str(args.batch),
        "--accumulation_steps",
        "2",
        "--max_seq_len",
        str(args.seq),
        "--cache_limit_gb",
        str(args.cache_limit_gb),
        "--dtype",
        "bfloat16",
    ]
    targs = setup_training_args(get_pretrain_parser().parse_args(cli), "pretrain")
    cfg = VibyConfig(**build_model_kwargs(targs))
    targs = resolve_compute_scaled_hparams(targs, 467617)
    model = VibyForCausalLM(cfg)
    convert_model_dtype(model, getattr(targs, "dtype", ""))
    trainer = BaseTrainer(targs, model, None, cfg, "pretrain")

    modes = [cfg.layer_mode(i) for i in range(cfg.n_layers)]
    mode_counts = dict(Counter(modes))
    ratios = list(cfg.compress_ratios[: cfg.n_layers])
    cfg.n_layers // 2
    n_index_calls = sum(
        1 for i in range(cfg.n_layers) if ratios[i] > 0 and cfg.layer_mode(i) != "reuse"
    )
    n_sparse_calls = sum(1 for i in range(cfg.n_layers) if ratios[i] > 0)
    encoder_n = args.seq // 2
    decoder_n = args.seq

    x = mx.random.randint(0, cfg.vocab_size, (args.batch, args.seq))
    y = mx.random.randint(0, cfg.vocab_size, x.shape)
    mask = mx.ones(x.shape, mx.float32)
    attn = mx.ones(x.shape, mx.int32)
    cuts = (mx.random.uniform(shape=x.shape) < 1 / args.seg).astype(mx.int32)
    cuts = cuts.at[:, 0].add(1 - cuts[:, 0])
    seg = mx.cumsum(cuts, axis=1).astype(mx.int32)
    mx.eval(x, y, mask, attn, seg)

    outputs, grads = trainer._compute_loss_and_grad(x, y, mask, attn, seg)
    mx.eval(*[o for o in outputs if o is not None])
    mx.eval(grads)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    outputs, grads = trainer._compute_loss_and_grad(x, y, mask, attn, seg)
    mx.eval(*[o for o in outputs if o is not None])
    mx.eval(grads)
    compiled_fb_s = time.perf_counter() - t0
    mem = memory_snapshot()
    finite = all(
        bool(mx.all(mx.isfinite(g.astype(mx.float32)))) for _, g in tree_flatten(grads)
    )
    grad_info = indexer_grad_report(grads)

    idx_shapes = []
    for n, count in (
        (
            encoder_n,
            sum(1 for i, r in enumerate(ratios) if r == 2 and modes[i] != "reuse"),
        ),
        (
            decoder_n,
            sum(1 for i, r in enumerate(ratios) if r == 1 and modes[i] != "reuse"),
        ),
    ):
        if count <= 0:
            continue
        sh = {
            "B": args.batch,
            "T": args.seq,
            "H": cfg.index_n_heads,
            "D": cfg.index_head_dim,
            "N": n,
            "count": count,
            "role": "encoder" if n == encoder_n else "decoder",
        }
        iso = isolated_indexer(sh, mx.bfloat16)
        iso["shape"] = sh
        iso["calls"] = count
        idx_shapes.append(iso)

    sa_shapes = []
    for n, count in (
        (encoder_n, sum(1 for r in ratios if r == 2)),
        (decoder_n, sum(1 for r in ratios if r == 1)),
    ):
        if count <= 0:
            continue
        sh = {
            "B": args.batch,
            "T": args.seq,
            "H": cfg.n_heads,
            "D": cfg.head_dim,
            "N": n,
            "count": count,
            "role": "encoder" if n == encoder_n else "decoder",
        }
        iso = isolated_sparse(sh, cfg.window_size, mx.bfloat16)
        iso["shape"] = sh
        iso["calls"] = count
        sa_shapes.append(iso)

    def weighted(items, field):
        return sum(it[field]["median_s"] * it["calls"] for it in items)

    idx_fwd = weighted(idx_shapes, "fwd")
    idx_fb = weighted(idx_shapes, "fb")
    sa_fwd = weighted(sa_shapes, "fwd")
    sa_fb = weighted(sa_shapes, "fb")
    idx_bwd = max(0.0, idx_fb - idx_fwd)
    sa_bwd = max(0.0, sa_fb - sa_fwd)
    empty_save = sum(
        it["fwd"]["median_s"] * it["calls"] * it.get("causal_empty_tile_frac", 0.0)
        for it in idx_shapes
    )
    indexer_in_graph = bool(grad_info["any_nonzero_indexer_grad"])
    indexer_train = idx_fb if indexer_in_graph else idx_fwd

    def amdahl(p, s):
        if p <= 0:
            return 1.0
        return 1.0 / ((1.0 - p) + p / s)

    step_ref = compiled_fb_s
    p_sparse_bwd = sa_bwd / step_ref if step_ref else 0
    p_idx = indexer_train / step_ref if step_ref else 0
    p_idx_skip = empty_save / step_ref if step_ref else 0
    # Reindex 候选池不会减少 causal empty tile；P3a 还额外跳过 candidate 外的
    # 可达 tile。没有实测 candidate 密度时，用「空因果 tile + 一半可达打分」
    # 作为乐观上界，避免低估 P3a。
    p3_optimistic = (
        (empty_save + 0.5 * max(0.0, idx_fwd - empty_save)) / step_ref
        if step_ref
        else 0
    )

    pick = "P2"
    reason = "稀疏 Attention 反向的可节省整步时间更大"
    if p3_optimistic > p_sparse_bwd * 1.1:
        pick = "P3a"
        reason = "Indexer 无效打分（空 tile + 候选外）的乐观可节省整步时间更大"
    if not sa_shapes:
        pick = "P3a"
        reason = "本形状未命中稀疏 Attention kernel"

    profile = {
        "flags": active_viby_flags(),
        "params": param_identity(model),
        "shape": {"B": args.batch, "T": args.seq, "seg": args.seg},
        "layer_modes": mode_counts,
        "n_index_calls": n_index_calls,
        "n_sparse_calls": n_sparse_calls,
        "compiled_fb_s": compiled_fb_s,
        "finite_grads": finite,
        "memory": mem,
        "indexer_isolated": idx_shapes,
        "sparse_isolated": sa_shapes,
        "totals_s": {
            "indexer_fwd": idx_fwd,
            "indexer_fb": idx_fb,
            "indexer_bwd": idx_bwd,
            "indexer_causal_empty_save_fwd": empty_save,
            "sparse_fwd": sa_fwd,
            "sparse_fb": sa_fb,
            "sparse_bwd": sa_bwd,
            "indexer_train_cost": indexer_train,
        },
        "indexer_grads": grad_info,
        "amdahl_clues": {
            "denominator": "one compiled f+b after warmup eval",
            "p_sparse_bwd": p_sparse_bwd,
            "p_indexer_train": p_idx,
            "p_indexer_causal_empty": p_idx_skip,
            "p_p3a_optimistic": p3_optimistic,
            "whole_step_if_sparse_bwd_1_3x": amdahl(p_sparse_bwd, 1.3),
            "whole_step_if_p3a_optimistic_all_saved": amdahl(p3_optimistic, 8.0),
        },
        "pick": pick,
        "pick_reason": reason,
        "flops_per_token": training_flops_per_token(model, args.seq),
        "notes": [
            "孤立计时 × 静态层数是上界：compile 可能融合/重叠，不能当已测整步差分。",
            "candidate 密度未在本脚本实测；P3a 乐观上界把可达打分的一半算作可跳过。",
            "Indexer 参数梯度为 0 时，训练预算不花在 indexer VJP/dK atomic 上。",
        ],
    }
    (run_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    md = [
        "# Kernel 路径归因",
        "",
        f"- compiled 单次 f+b：{compiled_fb_s:.3f}s",
        f"- 层模式 {mode_counts}；indexer 调用 {n_index_calls}，sparse 调用 {n_sparse_calls}",
        f"- Indexer 孤立 fwd {idx_fwd:.4f}s，f+b {idx_fb:.4f}s；参数梯度非零：{indexer_in_graph}",
        f"- 稀疏 Attention 孤立 fwd {sa_fwd:.4f}s，bwd {sa_bwd:.4f}s",
        f"- 因果空 tile 可省 fwd {empty_save:.4f}s；P3a 乐观占比 {p3_optimistic:.3f}",
        f"- 稀疏 bwd 占比 {p_sparse_bwd:.3f}；若组件 1.3× 则整步上限 {amdahl(p_sparse_bwd, 1.3):.3f}×",
        f"- 选择 **{pick}**：{reason}",
    ]
    (run_dir / "profile.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
