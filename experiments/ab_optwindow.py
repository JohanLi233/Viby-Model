"""优化器窗口策略同进程交替 A/B（目标 2B 配置，真实 fwd/bwd 产生的梯度）。

prof_target_step 里窗口段 = win_norm 109 + win_build 7 + win_eval 277
≈ 393ms/窗口（每微批 197ms，占墙钟 13%），而分组孤立计时只解释
Muon 73 + AdamW 75 + QB 35 + 累加 30 + 范数 31 = 244ms。差的 ~150ms
怀疑是「一次 mx.eval 把新旧参数 / m / v / 梯度 / Muon 临时量同时顶到
峰值」。跨进程对比不可信（本机分钟尺度漂移可达 15%），故同进程逐窗口
轮换臂：

  base    : 现状（全量 grad_norm + 一次性 eval 全部参数与 state）
  nonorm  : 跳过 grad_norm（NaN 防护改看 loss），其余同 base
  chunked : 保留 grad_norm，参数按块 eval（让中间量提前释放）
  both    : nonorm + chunked
  qbsub   : both + QB margin 沿 token 轴 1/8 下采样

用法: .venv/bin/python experiments/ab_optwindow.py [windows_per_arm]
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from experiments.prof_target_step import TARGET_ARGS, build_everything
from trainer.utils import apply_lr_schedule, resolve_lr_horizon, resolve_warmup_iters

ARMS = ("base", "nonorm", "chunked", "both", "qbsub")
CHUNK_BYTES = 512 << 20  # 每块参数字节上限


def _chunks(flat):
    out, cur, n = [], [], 0
    for _, v in flat:
        cur.append(v)
        n += v.size * v.dtype.size
        if n >= CHUNK_BYTES:
            out.append(cur)
            cur, n = [], 0
    if cur:
        out.append(cur)
    return out


def run(trainer, args, n_win):
    model = trainer.model
    accum = args.accumulation_steps
    total = resolve_lr_horizon(args, 27863)
    resolve_warmup_iters(args, total)
    B, T = args.batch_size, args.max_seq_len
    rng = np.random.default_rng(0)

    def synth():
        X = mx.array(rng.integers(1, args.vocab_size, size=(B, T), dtype=np.int64))
        Y = mx.array(rng.integers(1, args.vocab_size, size=(B, T), dtype=np.int64))
        m = mx.ones((B, T), dtype=mx.int64)
        seg = mx.array(
            np.cumsum(rng.random((B, T)) < 1.0 / 340.0, axis=1).astype(np.int64)
        )
        return X, Y, m, seg

    res = {a: [] for a in ARMS}
    accum_grads = None
    last_stats = None
    n_mb = (2 + n_win * len(ARMS)) * accum

    for step in range(n_mb):
        win_idx = step // accum
        arm = ARMS[(win_idx - 2) % len(ARMS)] if win_idx >= 2 else "base"
        rec = win_idx >= 2

        X, Y, loss_mask, seg = synth()
        apply_lr_schedule(trainer.optimizer, step, total, args.warmup_iters)
        attn_mask = mx.ones((B, T), dtype=mx.int32)
        (out, grads) = trainer._compute_loss_and_grad(
            X, Y, loss_mask, attn_mask, False, seg, compute_lar=False
        )
        mx.eval(*out)
        mx.eval(grads)
        moe_stats = out[2]
        if last_stats is None or last_stats.size == 0:
            last_stats = moe_stats
        elif moe_stats.size > 0:
            last_stats = mx.concatenate([last_stats, moe_stats], axis=1)
        accum_grads = (
            grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
        )

        if (step + 1) % accum != 0:
            continue

        t0 = time.perf_counter()
        g_flat = tree_flatten(accum_grads)
        if arm in ("base", "chunked"):
            gn = mx.sqrt(sum(mx.sum(mx.square(g)) for _, g in g_flat))
            ok = np.isfinite(float(gn))
        else:
            # NaN 防护退化为看 loss（前向已物化，只多一次标量同步）
            ok = np.isfinite(float(out[0]))
            mx.eval([g for _, g in g_flat])  # 累加仍需物化
        assert ok
        stats = last_stats
        if arm == "qbsub" and stats is not None and stats.size > 0:
            stats = stats[:, ::8]
        trainer.optimizer.update(model, accum_grads)
        if stats is not None and stats.size > 0:
            model.update_moe_biases(stats)
        if arm in ("chunked", "both", "qbsub"):
            for blk in _chunks(tree_flatten(model.parameters())):
                mx.eval(blk)
            mx.eval(trainer.optimizer.state)
        else:
            mx.eval(model.parameters(), trainer.optimizer.state)
        dt = (time.perf_counter() - t0) * 1e3
        if rec:
            res[arm].append(dt)
        accum_grads = None
        last_stats = None

    print(f"\n=== 优化器窗口 A/B（每臂 {n_win} 窗口，同进程轮换）===")
    print(f"{'臂':<10}{'中位 ms/窗口':>14}{'min':>9}{'每微批':>9}")
    base = statistics.median(res["base"]) if res["base"] else 0.0
    for a in ARMS:
        if not res[a]:
            continue
        m = statistics.median(res[a])
        print(
            f"{a:<10}{m:>14.1f}{min(res[a]):>9.1f}{m / accum:>9.1f}"
            f"   {'' if a == 'base' else f'({m - base:+.1f}ms)'}"
        )
    print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")


def main():
    n_win = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    args, _, _, _, trainer, _, _ = build_everything(TARGET_ARGS)
    run(trainer, args, n_win)


if __name__ == "__main__":
    main()
