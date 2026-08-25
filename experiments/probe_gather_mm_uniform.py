"""gather_mm 的 4.2ms/层差距：内在尾块量化还是 kernel 实现开销？

判定用手写分段 GEMM 之前必须回答的问题（probe_moe_parts 的 follow-up）：
sorted gather_mm 比同 FLOPs 均匀稠密 batched GEMM 慢 ~22%（fwd）。差距有两个
来源：
  a) 真实路由负载倾斜 → 每专家段长围绕均值 384 波动 → 段尾 tile 量化浪费
     （任何 kernel 都无法避免，除非 tile 跨段——那就不是分段 GEMM 了）
  b) MLX gather_mm kernel 自身的调度/发射开销（手写 kernel 可追回）

用「均匀合成索引（每专家恰好 384 行）」把 (a) 归零：
  - gather_mm(均匀) ≈ 稠密 → 差距全是倾斜内在浪费 ⇒ 手写 kernel 没意义
  - gather_mm(均匀) ≈ 当前倾斜水平 → 纯实现开销 ⇒ 手写值得做

顺带把 VJP 拆成 dA（argnums=(0,)）和 dB（argnums=(1,)）分别计时，看反向
哪一半慢。dB 是分段归约 GEMM，若 MLX 用稠密 mask 或原子加实现会显著慢。

用法: .venv/bin/python experiments/probe_gather_mm_uniform.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
E, K, DE, I = 256, 8, 384, 384
M, G = B * T, B * T * K
ROWS = G // E  # 384


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


mx.random.seed(0)
# 真实形态输入（连续 bf16）
xs = (mx.random.normal((G, DE)) * 0.5).astype(mx.bfloat16)
act = (mx.random.normal((G, 1, I)) * 0.5).astype(mx.bfloat16)
gu_t = (mx.random.normal((E, DE, 2 * I)) * 0.02).astype(mx.bfloat16)
dw_t = (mx.random.normal((E, I, DE)) * 0.02).astype(mx.bfloat16)
# 均匀索引：每专家恰好 ROWS 行，按段连续（sorted_indices 前提）
uni = mx.repeat(mx.arange(E, dtype=mx.int32), ROWS)
# 倾斜索引：Zipf -ish，用随机打分 argsort 模拟真实路由（段长方差大）
rng_scores = mx.random.normal((G, E))
imb = mx.argsort(mx.argsort(rng_scores, axis=-1), axis=-1)[:, 0] % E  # 伪随机专家
imb = mx.sort(imb).astype(mx.int32)  # 排序成段连续
mx.eval(xs, act, gu_t, dw_t, uni, imb)

# 倾斜度统计
cnt = mx.zeros((E,), dtype=mx.float32).at[imb].add(1.0)
mx.eval(cnt)
print(
    f"段长: 均匀 {ROWS} | 倾斜 min {cnt.min().item():.0f} max {cnt.max().item():.0f} "
    f"std {cnt.std().item():.1f}"
)

F1 = 2 * G * DE * 2 * I
F2 = 2 * G * I * DE


def f_mm1(a, b, idx):
    return mx.gather_mm(a[:, None, :], b, None, idx, sorted_indices=True).sum()


def f_mm2(a, b, idx):
    return mx.gather_mm(a, b, None, idx, sorted_indices=True).sum()


def report(label, fn, a, b, idx, flops):
    cf = mx.compile(lambda x, y: fn(x, y, idx))
    da = mx.compile(mx.value_and_grad(lambda x, y: fn(x, y, idx), argnums=(0,)))
    db = mx.compile(mx.value_and_grad(lambda x, y: fn(x, y, idx), argnums=(1,)))
    dab = mx.compile(mx.value_and_grad(lambda x, y: fn(x, y, idx), argnums=(0, 1)))
    f = timed(lambda: cf(a, b))
    t_da = timed(lambda: da(a, b)) - f  # 仅 dA 的反向增量
    t_db = timed(lambda: db(a, b)) - f  # 仅 dB
    t_dab = timed(lambda: dab(a, b)) - f
    print(
        f"{label:<28}{f:>7.2f}{t_da:>7.2f}{t_db:>7.2f}{t_dab:>7.2f}"
        f"{flops / f / 1e9:>8.1f}"
    )


print(f"\n{'口径':<28}{'fwd':>7}{'dA':>7}{'dB':>7}{'dA+dB':>7}{'TF/s_f':>8}")
report("mm1 倾斜 (G,1,DE)@(E,DE,2I)", f_mm1, xs, gu_t, imb, F1)
report("mm1 均匀", f_mm1, xs, gu_t, uni, F1)
report("mm2 倾斜 (G,1,I)@(E,I,DE)", f_mm2, act, dw_t, imb, F2)
report("mm2 均匀", f_mm2, act, dw_t, uni, F2)

# 稠密对照（同 FLOPs，无索引）
a1 = mx.random.normal((E, ROWS, DE)).astype(mx.bfloat16)
a2 = mx.random.normal((E, ROWS, I)).astype(mx.bfloat16)
mx.eval(a1, a2)


def report_dense(label, a, b, flops):
    cf = mx.compile(lambda x, y: (x @ y).sum())
    dab = mx.compile(mx.value_and_grad(lambda x, y: (x @ y).sum(), argnums=(0, 1)))
    f = timed(lambda: cf(a, b))
    fb = timed(lambda: dab(a, b)) - f
    print(f"{label:<28}{f:>7.2f}{'':>7}{'':>7}{fb:>7.2f}{flops / f / 1e9:>8.1f}")


report_dense("稠密 mm1 (E,384,DE)@(E,DE,2I)", a1, gu_t, F1)
report_dense("稠密 mm2 (E,384,I)@(E,I,DE)", a2, dw_t, F2)

# ---- 真实路由索引对照（probe_moe_parts 的索引来源）----
from mlx.utils import tree_map  # noqa: E402

from model.config import VibyConfig  # noqa: E402
from model.moe import MoEFeedForward  # noqa: E402

cfg = VibyConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    vocab_size=6400,
    max_position_embeddings=T,
    mtp_depth=1,
    use_attn_gate=True,
    n_routed_experts=E,
    num_experts_per_tok=K,
    n_shared_experts=2,
    moe_intermediate_size=I,
    moe_latent_dim=DE,
)
mod = MoEFeedForward(cfg)
mod.update(
    tree_map(
        lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a,
        mod.parameters(),
    )
)
mx.eval(mod.parameters())
mod.train()
x_real = (mx.random.normal((B, T, 768)) * 0.5).astype(mx.bfloat16)
idx_r, _w = mod.router(x_real)
flat_r = idx_r.reshape(G)
order_r = mx.argsort(flat_r)
exps_r = flat_r[order_r].astype(mx.int32)
cnt_r = mx.zeros((E,), dtype=mx.float32).at[exps_r].add(1.0)
mx.eval(exps_r, cnt_r)
print(
    f"\n真实路由段长: min {cnt_r.min().item():.0f} max {cnt_r.max().item():.0f} "
    f"std {cnt_r.std().item():.1f}"
)
report("mm1 真实路由", f_mm1, xs, gu_t, exps_r, F1)
report("mm2 真实路由", f_mm2, act, dw_t, exps_r, F2)
