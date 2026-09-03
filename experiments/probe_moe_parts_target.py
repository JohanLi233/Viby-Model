"""MoE 层内部逐段计时 · 目标配置版（E=384 K=8 I=512 latent=256，B12 T1024）。

probe_moe_parts.py 的目标配置复刻：shape matrix 已证明 gather_mm 在新形状
健康（~12.5-12.9 TFLOPS ≈ 稠密参考的 90-95%），本脚本量化整层 fwd 与
GEMM 地板之间的非 GEMM 余项：argsort / gather / situ_glu / scatter-add /
router / QB margins 收集（含逐层 concat 的累积复制）/ latent 投影 / 共享专家。

用法: .venv/bin/python experiments/probe_moe_parts_target.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.acts import situ_glu, situ_glu_gu
from model.config import VibyConfig
from model.moe import MoEFeedForward

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768

cfg = VibyConfig(
    hidden_size=D,
    num_hidden_layers=12,
    num_attention_heads=12,
    vocab_size=6400,
    max_position_embeddings=T,
    kv_lora_rank=256,
    qk_rope_head_dim=48,
    use_linear_attn=False,
    n_routed_experts=384,
    num_experts_per_tok=8,
    n_shared_experts=2,
    moe_intermediate_size=512,
    moe_latent_dim=256,
    routed_scaling_factor=2.5,
    mtp_depth=1,
    use_attn_gate=True,
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

E, K, I = mod.n_routed, mod.top_k, mod.moe_in  # noqa: E741
DE = mod.latent_dim or D
M, G = B * T, B * T * K
print(f"B={B} T={T} D={D} E={E} K={K} I={I} latent={DE}  M={M} G={G}\n")

mx.random.seed(0)
x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
mx.eval(x)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def bench(label, fn, args, argnums, flops=0.0):
    cf = mx.compile(fn)
    cvg = mx.compile(mx.value_and_grad(fn, argnums=argnums))
    f = timed(lambda: cf(*args))
    fb = timed(lambda: cvg(*args))
    tf = f"{flops / (f / 1e3) / 1e12:.1f}" if flops else "—"
    tfb = f"{3 * flops / (fb / 1e3) / 1e12:.1f}" if flops else "—"
    print(f"{label:<26}{f:>8.2f}{fb:>8.2f}{fb - f:>8.2f}{tf:>9}{tfb:>9}")
    mx.clear_cache()
    return f, fb


p = mod.trainable_parameters()
mod.router.collect_stats = True
idx, w_route = mod.router(x)
margins = mod.router.last_margins
xe = mod.latent_norm(mod.lat_down(x))
mx.eval(idx, w_route, xe, margins)
flat = idx.reshape(G)
order = mx.argsort(flat)
exps_s = flat[order].astype(mx.int32)
tok_s = (order // K).astype(mx.int32)
w_s = w_route.reshape(G)[order].astype(x.dtype)
xf = xe.reshape(M, DE)
xs = xf[tok_s]
gu_t = mod.experts.gate_up_w.swapaxes(-1, -2)
dw_t = mod.experts.down_w.swapaxes(-1, -2)
mx.eval(order, exps_s, tok_s, w_s, xs, gu_t, dw_t)

print(f"{'段':<26}{'fwd':>8}{'f+b':>8}{'bwd':>8}{'TF/s_f':>9}{'TF/s_fb':>9}")

bench(
    "整层 __call__(含QB统计)",
    lambda a, p_: (mod.update(p_), mod(a))[1].sum(),
    (x, p),
    (0, 1),
)
mod.router.collect_stats = False
bench(
    "整层 __call__(无QB统计)",
    lambda a, p_: (mod.update(p_), mod(a))[1].sum(),
    (x, p),
    (0, 1),
)
mod.router.collect_stats = True
bench(
    "router(含margins)",
    lambda a, p_: (mod.update(p_), mod.router(a))[1][1].sum(),
    (x, p),
    (0, 1),
    2 * M * D * E,
)
bench(
    "lat_down+norm",
    lambda a, p_: (mod.update(p_), mod.latent_norm(mod.lat_down(a)))[1].sum(),
    (x, p),
    (0, 1),
    2 * M * D * DE,
)
bench(
    "共享专家 ×2",
    lambda a, p_: (mod.update(p_), sum(ff(a) for ff in mod.shared))[1].sum(),
    (x, p),
    (0, 1),
    2 * 2 * M * (2 * D * I + I * D),
)

# QB margins 的逐层 concat：模拟 13 层累积（每层 concat 复制整份已有缓冲）
m0 = (mx.random.normal((M, E)) * 0.1).astype(mx.bfloat16)
mx.eval(m0)


def f_concat_chain():
    buf = None
    for _ in range(13):
        buf = m0 if buf is None else mx.concatenate([buf, m0], axis=0)
    return buf


def f_list_stack():
    return mx.stack([m0] * 13, axis=0)


print(
    f"{'QB margins 13层 concat':<26}{timed(lambda: mx.eval(f_concat_chain())):>8.2f}"
    f"（逐层 concat，累积复制 ~858MB）"
)
print(
    f"{'QB margins list+stack':<26}{timed(lambda: mx.eval(f_list_stack())):>8.2f}"
    f"（末尾一次 stack，118MB）"
)


def f_sort(xe_, iw):
    fl = iw.reshape(G)
    o = mx.argsort(fl)
    return (o // K).sum() + fl[o].sum()


print(f"{'argsort+索引派生':<26}{timed(lambda: mx.eval(f_sort(xe, idx))):>8.2f}")


def f_gather(xf_, ts):
    return xf_[ts]


print(f"{'xs = xf[tok_s] gather':<26}{timed(lambda: f_gather(xf, tok_s)):>8.2f}")

bench(
    "gather_mm1 (G,1,DE)@(E,DE,2I)",
    lambda a, b: mx.gather_mm(
        a[:, None, :], b, None, exps_s, sorted_indices=True
    ).sum(),
    (xs, gu_t),
    (0, 1),
    2 * G * DE * 2 * I,
)
h = mx.gather_mm(xs[:, None, :], gu_t, None, exps_s, sorted_indices=True)
act = situ_glu_gu(h)
mx.eval(h, act)
bench(
    "situ_glu 切片两参数", lambda a: situ_glu(a[..., :I], a[..., I:]).sum(), (h,), (0,)
)
bench("situ_glu 打包", lambda a: situ_glu_gu(a).sum(), (h,), (0,))
bench(
    "gather_mm2 (G,1,I)@(E,I,DE)",
    lambda a, b: mx.gather_mm(a, b, None, exps_s, sorted_indices=True).sum(),
    (act, dw_t),
    (0, 1),
    2 * G * I * DE,
)
y = mx.gather_mm(act, dw_t, None, exps_s, sorted_indices=True)[:, 0, :]
mx.eval(y)
bench(
    "加权 scatter-add (bf16)",
    lambda a: mx.zeros((M, DE), dtype=x.dtype).at[tok_s].add(a * w_s[:, None]).sum(),
    (y,),
    (0,),
)
merged = (
    mx.zeros((M, DE), dtype=x.dtype).at[tok_s].add(y * w_s[:, None]).reshape(B, T, DE)
)
mx.eval(merged)
bench(
    "lat_up(+out_norm)",
    lambda a, p_: (mod.update(p_), mod._latent_up(a))[1].sum(),
    (merged, p),
    (0, 1),
    2 * M * DE * D,
)

print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
