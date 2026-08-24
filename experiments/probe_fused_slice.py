"""融合投影切片对 causal_conv 的代价，以及拆分投影的对照。

KDA 把 q/k/v/f_a/g_a/b 六个投影拼成一次 GEMM，输出 (B,T,2384)，再用
fused[..., a:b] 切出各路。切片是 stride=2384 的非连续视图，causal_conv
读它比读连续张量慢得多（隔离测 0.24ms/conv，KDA 内 0.89ms/conv）。

对照三种投影方案在「投影 + 3×conv」全链上的 fwd / fwd+bwd：
  A 融合单 GEMM + 切片（现状）
  B 融合单 GEMM + 切片后显式连续化
  C q/k/v 三个独立 GEMM（各自连续）+ 小门控 GEMM

用法: uv run python experiments/probe_fused_slice.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels.conv import causal_conv

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768
H, Dh = 8, 96
P = H * Dh  # 768
EXTRA = 2 * Dh + H  # f_a(96) + g_a(96) + b(8) = 200
K = 4

x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
w_fused = (mx.random.normal((3 * P + EXTRA, D)) * 0.02).astype(mx.bfloat16)
w_q = w_fused[:P].astype(mx.bfloat16)
w_k = mx.array(w_fused[P : 2 * P])
w_v = mx.array(w_fused[2 * P : 3 * P])
w_g = mx.array(w_fused[3 * P :])
cw = (mx.random.normal((K, P)) * 0.3).astype(mx.bfloat16)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
mx.eval(x, w_fused, w_q, w_k, w_v, w_g, cw, seg)
print(f"B={B} T={T} D={D} 融合输出宽度 {3 * P + EXTRA}  切片宽度 {P}\n")


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def rep(label, f, fb):
    print(f"{label:<34}{f:>8.2f}{fb - f:>9.2f}{fb:>9.2f}")


print(f"{'方案':<34}{'fwd':>8}{'bwd':>9}{'f+b':>9}")

# ---- 纯 conv 基线：连续输入 ----
q0 = mx.array(mx.random.normal((B, T, P)).astype(mx.bfloat16))
mx.eval(q0)


def conv3_contig(a, cw_):
    return sum(
        (causal_conv(a, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        for _ in range(3)
    )


f = timed(lambda: conv3_contig(q0, cw))
fb = timed(lambda: mx.value_and_grad(conv3_contig, argnums=(0, 1))(q0, cw))
rep("3×conv 连续输入（基线）", f, fb)

# ---- 切片输入的 conv ----
fused0 = mx.array((x @ w_fused.T))
mx.eval(fused0)


def conv3_slice(fu, cw_):
    tot = 0.0
    for i in range(3):
        s = fu[..., i * P : (i + 1) * P]
        tot = (
            tot
            + (causal_conv(s, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        )
    return tot


f = timed(lambda: conv3_slice(fused0, cw))
fb = timed(lambda: mx.value_and_grad(conv3_slice, argnums=(0, 1))(fused0, cw))
rep("3×conv 切片输入", f, fb)


def conv3_slice_contig(fu, cw_):
    tot = 0.0
    for i in range(3):
        s = mx.contiguous(fu[..., i * P : (i + 1) * P])
        tot = (
            tot
            + (causal_conv(s, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        )
    return tot


f = timed(lambda: conv3_slice_contig(fused0, cw))
fb = timed(lambda: mx.value_and_grad(conv3_slice_contig, argnums=(0, 1))(fused0, cw))
rep("3×conv 切片+contiguous", f, fb)

print()


# ---- 全链 A: 融合 GEMM + 切片 ----
def chain_A(x_, wf, cw_):
    fu = x_ @ wf.T
    tot = 0.0
    for i in range(3):
        s = fu[..., i * P : (i + 1) * P]
        tot = (
            tot
            + (causal_conv(s, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        )
    return tot + (fu[..., 3 * P :].astype(mx.float32) ** 2).sum()


cA = mx.compile(chain_A)
cAg = mx.compile(mx.value_and_grad(chain_A, argnums=(0, 1, 2)))
f = timed(lambda: cA(x, w_fused, cw))
fb = timed(lambda: cAg(x, w_fused, cw))
rep("A 融合GEMM+切片（现状）", f, fb)


# ---- 全链 B: 融合 GEMM + contiguous ----
def chain_B(x_, wf, cw_):
    fu = x_ @ wf.T
    tot = 0.0
    for i in range(3):
        s = mx.contiguous(fu[..., i * P : (i + 1) * P])
        tot = (
            tot
            + (causal_conv(s, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        )
    return tot + (fu[..., 3 * P :].astype(mx.float32) ** 2).sum()


cB = mx.compile(chain_B)
cBg = mx.compile(mx.value_and_grad(chain_B, argnums=(0, 1, 2)))
f = timed(lambda: cB(x, w_fused, cw))
fb = timed(lambda: cBg(x, w_fused, cw))
rep("B 融合GEMM+contiguous", f, fb)


# ---- 全链 C: q/k/v 独立 GEMM + 门控小 GEMM ----
def chain_C(x_, wq, wk, wv, wg, cw_):
    tot = 0.0
    for wt in (wq, wk, wv):
        s = x_ @ wt.T
        tot = (
            tot
            + (causal_conv(s, cw_, seg=seg, silu=True).astype(mx.float32) ** 2).sum()
        )
    return tot + ((x_ @ wg.T).astype(mx.float32) ** 2).sum()


cC = mx.compile(chain_C)
cCg = mx.compile(mx.value_and_grad(chain_C, argnums=(0, 1, 2, 3, 4, 5)))
f = timed(lambda: cC(x, w_q, w_k, w_v, w_g, cw))
fb = timed(lambda: cCg(x, w_q, w_k, w_v, w_g, cw))
rep("C 三独立GEMM（各自连续）", f, fb)

print()
# ---- 只测 GEMM 部分，隔离投影方案本身的差异 ----
f = timed(lambda: (x @ w_fused.T,))
print(f"{'纯 GEMM 融合 (768→2384)':<34}{f:>8.2f}")
f = timed(lambda: (x @ w_q.T, x @ w_k.T, x @ w_v.T, x @ w_g.T))
print(f"{'纯 GEMM 拆分 (3×768 + 200)':<34}{f:>8.2f}")
print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
