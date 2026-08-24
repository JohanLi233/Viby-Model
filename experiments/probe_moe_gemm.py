"""MoE 专家 GEMM 的裸速度上限 vs 稀疏桶路径的实际速度。

稀疏桶路径实测只有 3~4 TFLOPS（M4 Max bf16 GEMM 峰值 ~25），需要区分是
「(E,C,D)@(E,D,2I) 这种细粒度 batched GEMM 本身就慢」还是「散布/收回的
索引算子吃掉了时间」。本脚本分别测：

  1. 裸 batched GEMM：(E,C,D)@(E,D,2I) + SwiGLU + (E,C,I)@(E,I,D)
  2. 同 FLOPs 的单个大 GEMM：(E·C,D)@(D,2I)（同一份权重，非等价，仅作上限）
  3. 稀疏桶路径的非 GEMM 部分：argsort / scatter / gather / scatter-add

用法: uv run experiments/probe_moe_gemm.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def gemm_flops(rows, D, I):  # noqa: E741
    return rows * (D * 2 * I + I * D) * 2


def bench_grouped(E, C, D, I, EG):  # noqa: E741
    """按 EG 分组的 batched GEMM（复现 _sparse_forward 的 GEMM 部分）。"""
    n_groups = E // EG
    xb = (mx.random.normal((E * C, D)) * 0.1).astype(mx.bfloat16)
    gu = (mx.random.normal((E, D, 2 * I)) * 0.02).astype(mx.bfloat16)
    dw = (mx.random.normal((E, I, D)) * 0.02).astype(mx.bfloat16)
    mx.eval(xb, gu, dw)

    def run():
        parts = []
        for gi in range(n_groups):
            s0, s1 = gi * EG, (gi + 1) * EG
            xg = xb[s0 * C : s1 * C].reshape(EG, C, D)
            g_, u_ = mx.split(xg @ gu[s0:s1], 2, axis=-1)
            parts.append(((nn.silu(g_) * u_) @ dw[s0:s1]).reshape(EG * C, D))
        return mx.concatenate(parts, axis=0)

    return timed(run)


def bench_single_gemm(rows, D, I):  # noqa: E741
    """同 FLOPs 的单个大 GEMM（非等价，作为纯 GEMM 上限参考）。"""
    x = (mx.random.normal((rows, D)) * 0.1).astype(mx.bfloat16)
    gu = (mx.random.normal((D, 2 * I)) * 0.02).astype(mx.bfloat16)
    dw = (mx.random.normal((I, D)) * 0.02).astype(mx.bfloat16)
    mx.eval(x, gu, dw)

    def run():
        g_, u_ = mx.split(x @ gu, 2, axis=-1)
        return (nn.silu(g_) * u_) @ dw

    return timed(run)


def bench_scatter(M, G, D, rows):
    """稀疏桶路径的非 GEMM 部分：argsort + 散布 + 收回 + scatter-add。"""
    K = G // M
    exps = mx.random.randint(0, 288, (G,)).astype(mx.int32)
    xf = (mx.random.normal((M, D)) * 0.1).astype(mx.bfloat16)
    y = (mx.random.normal((rows + 1, D)) * 0.1).astype(mx.bfloat16)
    w = mx.random.uniform(shape=(G,)).astype(mx.bfloat16)
    mx.eval(exps, xf, y, w)

    def sort_part():
        order = mx.argsort(exps)
        counts = mx.zeros((288,), dtype=mx.int32).at[exps].add(1)
        offsets = mx.cumsum(counts) - counts
        tok_s = (order // K).astype(mx.int32)
        rank = mx.arange(G, dtype=mx.int32) - offsets[exps[order]]
        return order, tok_s, rank

    def _row():
        _, tok_s, rank = sort_part()
        return mx.minimum(rank * 3 + tok_s % 7, rows), tok_s

    def build_scatter():
        """现行做法：zeros(rows+1,D) 上 scatter-add 源 token（bf16 原子加）。"""
        row, tok_s = _row()
        return mx.zeros((rows + 1, D), dtype=mx.bfloat16).at[row].add(xf[tok_s])

    def build_gather():
        """候选做法：先散布 int32 的「桶行 → 源 token」表（1MB 级），
        再一次连续 gather 出整桶（无原子加、无 (rows,D) 清零）。"""
        row, tok_s = _row()
        src = mx.clip(
            mx.full((rows + 1,), M, dtype=mx.int32).at[row].add(tok_s - M), 0, M
        )
        xf0 = mx.concatenate([xf, mx.zeros((1, D), dtype=mx.bfloat16)], axis=0)
        return xf0[src]

    def collect_scatter():
        """现行做法：桶行 gather → 加权 → f32 scatter-add 回 token。"""
        row, tok_s = _row()
        yw = y[row] * w[:, None]
        return mx.zeros((M, D), dtype=mx.float32).at[tok_s].add(yw.astype(mx.float32))

    def collect_reduce():
        """候选做法：把桶行号还原成 token 序 (M,K)，gather 成 (M,K,D) 后
        沿 K 规约。无原子加、无 f32 中间缓冲，代价是多一次 int32 散布。"""
        order, tok_s, rank = sort_part()
        row = mx.minimum(rank * 3 + tok_s % 7, rows)
        row_tok = mx.zeros((M * K,), dtype=mx.int32).at[order].add(row).reshape(M, K)
        yw = y[row_tok] * w.reshape(M, K)[..., None]
        return yw.astype(mx.float32).sum(axis=1)

    return (
        timed(sort_part),
        timed(build_scatter),
        timed(build_gather),
        timed(collect_scatter),
        timed(collect_reduce),
    )


def main():
    D = 768
    E = int(os.environ.get("VIBY_BENCH_E", 288))
    M = int(os.environ.get("VIBY_BENCH_M", 12 * 1024))
    K = int(os.environ.get("VIBY_BENCH_K", 6))
    G = M * K
    print(f"D={D} E={E} tokens={M} K={K} 真实 pair={G}")

    print(f"\n{'I':>5}{'C':>6}{'EG':>5}{'桶行数':>9}{'GEMM(ms)':>10}{'TFLOPS':>9}")
    for I in (104, 128):  # noqa: E741
        for C in (256, 448, 512):
            rows = E * C
            fl = gemm_flops(rows, D, I)
            for EG in (8, 32, E):
                t = bench_grouped(E, C, D, I, EG)
                print(
                    f"{I:>5}{C:>6}{EG:>5}{rows:>9}{t * 1e3:>10.2f}{fl / t / 1e12:>9.2f}"
                )
            t = bench_single_gemm(rows, D, I)
            print(
                f"{I:>5}{C:>6}{'单GEMM':>5}{rows:>9}{t * 1e3:>10.2f}"
                f"{fl / t / 1e12:>9.2f}"
            )

    rows = E * 448
    s, sc, gt, c1, c2 = bench_scatter(M, G, D, rows)
    print(f"\n非 GEMM 部分（桶行数={rows}，均含前置 argsort）")
    print(f"  argsort+bincount+rank        {s * 1e3:7.2f}ms")
    print(f"  构桶：zeros+scatter-add      {sc * 1e3:7.2f}ms")
    print(f"  构桶：int 索引表+gather      {gt * 1e3:7.2f}ms")
    print(f"  收回：加权 f32 scatter-add   {c1 * 1e3:7.2f}ms")
    print(f"  收回：(M,K,D) gather+规约    {c2 * 1e3:7.2f}ms")


if __name__ == "__main__":
    main()
