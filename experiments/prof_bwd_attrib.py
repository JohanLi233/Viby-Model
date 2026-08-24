"""compile 口径的 fwd/bwd 组件归因（真实 1B 配置）。

bench_train_step / bench_components 都是 eager 口径，会高估 elementwise
链；真实训练走 mx.compile。本脚本逐组件在 compile 下测 fwd 与 fwd+bwd，
把整步的反向时间归因到各组件，并按层数外推。

逐个构建、测完即释放，避免同时持有多份大权重。
用法: uv run python experiments/prof_bwd_attrib.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.kernels import prewarm_all

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768
E = int(os.environ.get("VIBY_BENCH_E", 256))
MI = int(os.environ.get("VIBY_BENCH_I", 384))
V = 6400


def cfg_of(**kw):
    base = dict(
        hidden_size=D,
        num_hidden_layers=8,
        num_attention_heads=8,
        vocab_size=V,
        max_position_embeddings=T,
        mtp_depth=1,
        use_attn_gate=True,
        n_routed_experts=E,
        num_experts_per_tok=8,
        n_shared_experts=2,
        moe_intermediate_size=MI,
        latent_dim=384,
    )
    base.update(kw)
    return VibyConfig(**base)


def bf16(mod):
    mod.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            mod.parameters(),
        )
    )
    mx.eval(mod.parameters())
    return mod


def timed(fn, it=6, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


rows = []


def bench(label, mod, count, fwd_call, x_shape=(B, T, D)):
    """fwd_call(mod, x, params) -> 标量 loss。count 为整模型里的层数。"""
    # compile 之前预热：否则融合 kernel 首调用的在线校验落在 trace 内，
    # 异常被吞掉并永久回退 eager，归因会整体偏向 eager 图。
    prewarm_all(mod, base_cfg, mx.bfloat16, T)
    x = (mx.random.normal(x_shape) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    p = mod.trainable_parameters()

    def loss(x_, p_):
        mod.update(p_)
        return fwd_call(mod, x_)

    c_f = mx.compile(loss)
    c_vg = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
    f = timed(lambda: c_f(x, p))
    fb = timed(lambda: c_vg(x, p))
    rows.append((label, count, f, fb - f, fb))
    print(
        f"  {label:<22}×{count:<3} fwd {f:>7.2f}  bwd {fb - f:>7.2f}  "
        f"f+b {fb:>7.2f}  外推 {fb * count:>7.1f}ms"
    )
    mx.clear_cache()


seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
cot = mx.random.normal((B, T, D))
mx.eval(seg, cot)

base_cfg = cfg_of()
main_kda = sum(
    not (((i + 1) % 4 == 0) or i == base_cfg.num_hidden_layers - 1)
    for i in range(base_cfg.num_hidden_layers)
)
mtp_kda = base_cfg.mtp_depth if base_cfg.num_hidden_layers > 1 else 0
kda_count = int(main_kda + mtp_kda)
gqa_count = int(base_cfg.num_hidden_layers - main_kda)
moe_count = int(base_cfg.num_hidden_layers + base_cfg.mtp_depth)
print(
    f"配置 B={B} T={T} D={D} E={E} MI={MI} "
    f"KDA={main_kda}+{mtp_kda} GQA={gqa_count} MoE={moe_count}\n"
)
print("组件（compile 口径，min over 6）")

# ---- KDA 层 ----
from model.kda import KDAAttention  # noqa: E402

m = bf16(KDAAttention(cfg_of(), layer_idx=0))
m.train()
bench(
    "KDA 层",
    m,
    kda_count,
    lambda mo, x: (mo(x, segment_ids=seg)[0].astype(mx.float32) * cot).sum(),
)
del m
mx.clear_cache()

# ---- GQA 层 ----
from model.attention import GQAAttention  # noqa: E402

m = bf16(GQAAttention(cfg_of(), layer_idx=1))
m.train()
bench(
    "GQA 层",
    m,
    gqa_count,
    lambda mo, x: (mo(x, segment_ids=seg)[0].astype(mx.float32) * cot).sum(),
)
del m
mx.clear_cache()

# ---- MoE 层 ----
from model.moe import MoEFeedForward  # noqa: E402

m = bf16(MoEFeedForward(cfg_of()))
m.train()
bench(
    "MoE 层",
    m,
    moe_count,
    lambda mo, x: (mo(x)[0].astype(mx.float32) * cot).sum(),
)
del m
mx.clear_cache()

# ---- lm_head + CE ----
import mlx.nn as nn  # noqa: E402

from model.kernels.ce import cross_entropy  # noqa: E402

head = bf16(nn.Linear(D, V, bias=False))
tgt = mx.random.randint(0, V, (B, T))
mx.eval(tgt)


def ce_call(mo, x):
    return cross_entropy(
        mo(x).astype(mx.float32).reshape(-1, V), tgt.reshape(-1)
    ).mean()


bench("lm_head+CE", head, 2, ce_call)
del head
mx.clear_cache()

tot_f = sum(f * c for _, c, f, _, _ in rows)
tot_b = sum(b * c for _, c, _, b, _ in rows)
print(f"\n外推整步: fwd {tot_f:.0f}ms + bwd {tot_b:.0f}ms = {tot_f + tot_b:.0f}ms")
print("反向占比归属:")
for label, c, _, b, _ in sorted(rows, key=lambda r: -r[3] * r[1]):
    print(f"  {label:<22}{b * c:>7.1f}ms  {b * c / tot_b * 100:>5.1f}%")
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
