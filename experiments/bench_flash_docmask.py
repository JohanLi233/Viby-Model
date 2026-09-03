"""doc_mask 偏置在手写 flash 里的开销量化（目标注意力形状）。

MLA 目标形状：B=12 H=12 T=1024 Dk=112(64 nope+48 rope) Dv=64。
三臂同进程交替（抗本机漂移）：
  1. mask="causal"   无 mask buffer（快路径假想上限）
  2. mask=零偏置     同流量同指令、零语义 —— 纯 mask 机制开销
  3. mask=真实 doc 偏置（dataset 实采 segment_ids 构造）

另统计真实 segment_ids 下 (RES=32 × STR=32) tile 的「干净」比例
（整块无任何文档边界/因果外元素 ⇒ 可走无 mask 快路径），给 block 级
快路径改造的命中率上限。

用法: .venv/bin/python experiments/bench_flash_docmask.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

from model.kernels.attn_fused import flash_sdpa

B, H, T, DK, DV = 12, 12, 1024, 112, 64


def timed(fn, it=10, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def make_doc_bias(seg):
    """segment_ids (B,T) → 加性偏置 (B,1,T,T) bf16（同 model.py 的构造）。"""
    same_doc = seg[:, :, None] == seg[:, None, :]
    causal_tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    allowed = same_doc & causal_tril[None, :, :]
    return mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(mx.bfloat16)


def tile_stats(seg):
    """(RES×STR) tile 三态占比：全合法（干净可快路径）/ 全屏蔽 / 混合。"""
    RES, STR = 32, 32
    same_doc = seg[:, :, None] == seg[:, None, :]
    causal_tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    allowed = same_doc & causal_tril[None, :, :]  # (B,T,T)
    # 只统计因果下三角内、kernel 实际会处理的 tile（kb*STR <= qb*RES+RES-1）
    n_clean = n_mixed = n_blk = 0
    for b in range(seg.shape[0]):
        a = allowed[b]
        for qb in range(T // RES):
            q0, q1 = qb * RES, qb * RES + RES
            nkb = (q1 + STR - 1) // STR
            for kb in range(nkb):
                k0, k1 = kb * STR, kb * STR + STR
                tile = a[q0:q1, k0:k1]
                s = tile.sum().item()
                n_blk += 1
                if s == RES * STR:
                    n_clean += 1
                elif s > 0:
                    n_mixed += 1
    return n_clean, n_mixed, n_blk


def real_segments(n_batches=4):
    """从真实 pretrain 打包缓存取几个 segment_ids（pack+doc_mask 路径）。"""
    from dataset.lm_dataset import PretrainDataset

    ds = PretrainDataset(
        "/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl",
        None,
        max_length=T,
        pack_sequences=True,
        doc_mask=True,
    )
    segs = []
    for i in range(0, len(ds), max(1, len(ds) // (n_batches * B)))[: n_batches * B]:
        item = ds[i]
        segs.append(item[3])  # (X, Y, loss_mask, segX)
    return mx.array(np.stack(segs[: n_batches * B]))


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.5).astype(mx.bfloat16)
    cot = (mx.random.normal((B, H, T, DV)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v, cot)
    scale = DK**-0.5

    def loss(mk):
        return (flash_sdpa(q, k, v, scale=scale, mask=mk).astype(mx.float32)).sum()

    def vg(mk):
        return mx.value_and_grad(
            lambda a, b, c: (
                flash_sdpa(a, b, c, scale=scale, mask=mk).astype(mx.float32) * cot
            ).sum()
        )(q, k, v)

    zero_bias = mx.zeros((B, 1, T, T), dtype=mx.bfloat16)
    mx.eval(zero_bias)

    # 真实 doc 偏置（拿得到数据就用，拿不到退化为合成 doc 长 300）
    doc_bias = None
    try:
        seg = real_segments()
        doc_bias = make_doc_bias(seg)
        mx.eval(doc_bias)
        nc, nm, nb = tile_stats(seg)
        print(
            f"真实 segment_ids tile 统计：干净 {nc}/{nb} ({100 * nc / nb:.1f}%) "
            f"混合 {nm} ({100 * nm / nb:.1f}%)"
        )
    except Exception as e:
        print(f"真实数据不可用（{type(e).__name__}: {e}），用合成 doc 长 300")
    if doc_bias is None:
        bounds = mx.arange(0, T + 300, 300)
        seg = mx.zeros((B, T), dtype=mx.int32)
        for i in range(B):
            seg_i = mx.concatenate(
                [
                    mx.full((min(bounds[j + 1], T) - min(bounds[j], T),), j)
                    for j in range(len(bounds) - 1)
                    if min(bounds[j], T) < T
                ]
            )[:T]
            seg = seg.at[i].add(seg_i)
        doc_bias = make_doc_bias(seg)
        mx.eval(doc_bias)

    arms = [("causal", "causal"), ("零偏置", zero_bias), ("doc偏置", doc_bias)]
    # 同进程交替，取每臂中位
    res = {n: {"f": [], "fb": []} for n, _ in arms}
    for rnd in range(7):
        for name, mk in arms:
            t0 = time.perf_counter()
            mx.eval(loss(mk))
            t1 = time.perf_counter()
            mx.eval(vg(mk))
            t2 = time.perf_counter()
            if rnd >= 2:
                res[name]["f"].append((t1 - t0) * 1e3)
                res[name]["fb"].append((t2 - t1) * 1e3)
    import statistics

    print(f"\nB={B} H={H} T={T} Dk={DK} Dv={DV}（单层）")
    print(f"{'臂':<10}{'fwd':>9}{'f+b':>9}{'bwd':>9}")
    for name, _ in arms:
        f = statistics.median(res[name]["f"])
        fb = statistics.median(res[name]["fb"])
        print(f"{name:<10}{f:>9.2f}{fb:>9.2f}{fb - f:>9.2f}")


if __name__ == "__main__":
    main()
