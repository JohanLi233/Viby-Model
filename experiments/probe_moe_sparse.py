"""MoE _sparse_forward（gather_mm 路径）逐段剖析与替代方案对比。

当前实现把每个 (token,choice) 对做成 (1,D) 的 lhs 交给 mx.gather_mm，
即每对一次 GEMV。本脚本量化：
  1. 整条 _sparse_forward 的 fwd / fwd+bwd
  2. 逐段：router / argsort+索引 / gather xs / gather_mm×2 / scatter-add
  3. gather_mm 与「同 FLOPs 单个大 GEMM」的效率差（分段是否真的成段）
  4. 候选：把 lhs 预先按段折叠成 (G/1, ...) 之外的形状

用法: uv run python experiments/probe_moe_sparse.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = int(os.environ.get("VIBY_BENCH_D", 768))
DE = int(os.environ.get("VIBY_BENCH_DE", 384))  # latent 维（专家输入维）
I = int(os.environ.get("VIBY_BENCH_I", 384))  # noqa: E741
E = int(os.environ.get("VIBY_BENCH_E", 256))
K = int(os.environ.get("VIBY_BENCH_K", 8))

M = B * T
G = M * K
FL = G * (DE * 2 * I + I * DE) * 2  # 专家 GEMM 的真实 FLOPs


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


print(f"M={M} G={G} DE={DE} I={I} E={E} K={K}  专家 GEMM {FL / 1e9:.1f} GFLOP")

# ---- 输入 ----
mx.random.seed(0)
xe = (mx.random.normal((M, DE)) * 0.1).astype(mx.bfloat16)
gu_w = (mx.random.normal((E, 2 * I, DE)) * 0.02).astype(mx.bfloat16)
dw_w = (mx.random.normal((E, DE, I)) * 0.02).astype(mx.bfloat16)
# Zipf 倾斜的路由（贴真实负载）
p = 1.0 / mx.arange(1, E + 1).astype(mx.float32)
p = p / p.sum()
idx = mx.random.categorical(mx.log(p), num_samples=G).astype(mx.int32)
w = mx.random.uniform(shape=(G,)).astype(mx.bfloat16)
mx.eval(xe, gu_w, dw_w, idx, w)

gu_t = gu_w.swapaxes(-1, -2)  # (E, DE, 2I)
dw_t = dw_w.swapaxes(-1, -2)  # (E, I, DE)
mx.eval(gu_t, dw_t)


def index_part():
    order = mx.argsort(idx)
    exps_s = idx[order].astype(mx.int32)
    tok_s = (order // K).astype(mx.int32)
    w_s = w[order]
    return order, exps_s, tok_s, w_s


order, exps_s, tok_s, w_s = index_part()
mx.eval(order, exps_s, tok_s, w_s)
xs = xe[tok_s]
mx.eval(xs)


def gm1():
    return mx.gather_mm(
        xs[:, None, :], gu_t, lhs_indices=None, rhs_indices=exps_s, sorted_indices=True
    )


h = gm1()
mx.eval(h)
act = (nn.silu(h[..., :I]) * h[..., I:]).astype(mx.bfloat16)
mx.eval(act)


def gm2():
    return mx.gather_mm(
        act, dw_t, lhs_indices=None, rhs_indices=exps_s, sorted_indices=True
    )[:, 0, :]


y = gm2()
mx.eval(y)

# ---- 上限：同 FLOPs 的单个大 GEMM（非等价，纯 GEMM 天花板）----
gu1 = gu_t[0]
dw1 = dw_t[0]
mx.eval(gu1, dw1)

t_idx = timed(lambda: index_part())
t_gather = timed(lambda: xe[tok_s])
t_gm1 = timed(gm1)
t_act = timed(lambda: (nn.silu(h[..., :I]) * h[..., I:]).astype(mx.bfloat16))
t_gm2 = timed(gm2)
t_scat = timed(
    lambda: mx.zeros((M, DE), dtype=mx.float32)
    .at[tok_s]
    .add((y * w_s[:, None]).astype(mx.float32))
)
t_big1 = timed(lambda: xs @ gu1)
t_big2 = timed(lambda: act[:, 0, :] @ dw1)

fl1 = G * DE * 2 * I * 2
fl2 = G * I * DE * 2
print(f"\n{'段':<26}{'ms':>9}{'TFLOPS':>9}")
print(f"{'argsort+索引':<26}{t_idx * 1e3:>9.2f}{'':>9}")
print(f"{'gather xs (G,DE)':<26}{t_gather * 1e3:>9.2f}{'':>9}")
print(f"{'gather_mm gate_up':<26}{t_gm1 * 1e3:>9.2f}{fl1 / t_gm1 / 1e12:>9.2f}")
print(f"{'  └ 单大GEMM 上限':<26}{t_big1 * 1e3:>9.2f}{fl1 / t_big1 / 1e12:>9.2f}")
print(f"{'SwiGLU':<26}{t_act * 1e3:>9.2f}{'':>9}")
print(f"{'gather_mm down':<26}{t_gm2 * 1e3:>9.2f}{fl2 / t_gm2 / 1e12:>9.2f}")
print(f"{'  └ 单大GEMM 上限':<26}{t_big2 * 1e3:>9.2f}{fl2 / t_big2 / 1e12:>9.2f}")
print(f"{'加权 f32 scatter-add':<26}{t_scat * 1e3:>9.2f}{'':>9}")
tot = t_idx + t_gather + t_gm1 + t_act + t_gm2 + t_scat
print(f"{'逐段和':<26}{tot * 1e3:>9.2f}{FL / tot / 1e12:>9.2f}")
ideal = t_idx + t_gather + t_big1 + t_act + t_big2 + t_scat
print(f"{'若 GEMM 打满':<26}{ideal * 1e3:>9.2f}{FL / ideal / 1e12:>9.2f}")


# ---- 端到端 fwd / fwd+bwd ----
def sparse_forward(xe_, gu_, dw_):
    _o, e_s, t_s, ws = order, exps_s, tok_s, w_s
    xs_ = xe_[t_s]
    h_ = mx.gather_mm(
        xs_[:, None, :],
        gu_.swapaxes(-1, -2),
        lhs_indices=None,
        rhs_indices=e_s,
        sorted_indices=True,
    )
    a_ = nn.silu(h_[..., :I]) * h_[..., I:]
    y_ = mx.gather_mm(
        a_, dw_.swapaxes(-1, -2), lhs_indices=None, rhs_indices=e_s, sorted_indices=True
    )[:, 0, :]
    yw = y_ * ws[:, None]
    return mx.zeros((M, DE), dtype=mx.float32).at[t_s].add(yw.astype(mx.float32))


C = mx.random.normal((M, DE))
mx.eval(C)


def loss(xe_, gu_, dw_):
    return (sparse_forward(xe_, gu_, dw_) * C).sum()


vg = mx.value_and_grad(loss, argnums=(0, 1, 2))
t_f = timed(lambda: sparse_forward(xe, gu_w, dw_w))
t_fb = timed(lambda: vg(xe, gu_w, dw_w))
print(
    f"\n端到端 fwd {t_f * 1e3:.2f}ms ({FL / t_f / 1e12:.2f} TF) | "
    f"fwd+bwd {t_fb * 1e3:.2f}ms ⇒ bwd {(t_fb - t_f) * 1e3:.2f}ms "
    f"(bwd/fwd={(t_fb - t_f) / t_f:.2f})"
)
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
