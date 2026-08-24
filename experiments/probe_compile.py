"""全步 mx.compile 冒烟 + 测速（gather_mm MoE 路径下 compile 首次可用）。

对比 eager / compiled 的 loss 值与整步耗时。用法:
    .venv/bin/python experiments/probe_compile.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM

B, T, V = 12, 1024, 6400
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

cfg = VibyConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    vocab_size=V,
    max_position_embeddings=T,
    mtp_depth=1,
    mtp_loss_weight=0.3,
    use_attn_gate=True,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=2,
    moe_intermediate_size=320,
    routed_scaling_factor=2.5,
    moe_router_noise=0.0,
)

X = mx.random.randint(1, V, (B, T))
Y = mx.random.randint(1, V, (B, T))
mask = mx.ones((B, T), dtype=mx.int32)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)


def build():
    model = VibyForCausalLM(cfg)
    model.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    model.train()
    for g in model.moe_gates():
        g.collect_stats = True
    return model


model = build()
model_c = build()
model_c.update(model.parameters())
mx.eval(model_c.parameters())


def make_loss(m):
    def loss_fn(params):
        m.update(params)
        res = m(input_ids=X, labels=Y, loss_mask=mask, segment_ids=seg)
        stats = m.qb_margin_stats()
        return res.loss, stats if stats is not None else mx.zeros((0,))

    return loss_fn


vg = mx.value_and_grad(make_loss(model), argnums=0)
cvg = mx.compile(mx.value_and_grad(make_loss(model_c), argnums=0))

p = model.trainable_parameters()
(l_e, s_e), g_e = vg(p)
mx.eval(l_e, s_e, g_e)
pc = model_c.trainable_parameters()
(l_c, s_c), g_c = cvg(pc)
model_c.update(pc)  # compiled 内部 update 只在 trace 生效，立即恢复真实参数
mx.eval(l_c, s_c, g_c)
print(f"loss eager {float(l_e):.6f} vs compiled {float(l_c):.6f}")
ge = tree_map(lambda a: a.astype(mx.float32), g_e)
gc = tree_map(lambda a: a.astype(mx.float32), g_c)


def tree_maxdiff(a, b):
    from mlx.utils import tree_flatten

    fa = dict(tree_flatten(a))
    fb = dict(tree_flatten(b))
    return max(float((fa[k] - fb[k]).abs().max()) for k in fa)


print(f"grad max diff: {tree_maxdiff(ge, gc):.2e}")


def step(fn, m, compiled):
    pp = m.trainable_parameters()
    (l, s), g = fn(pp)  # noqa: E741
    if compiled:
        # compiled 内部的 m.update(pp) 只在 trace 时执行，留下占位数组；
        # 立即用真实参数恢复（与 BaseTrainer 同口径）
        m.update(pp)
    mx.eval(l, s)
    mx.eval(g)
    return l


for label, fn, m, c in [
    ("eager", vg, model, False),
    ("compiled", cvg, model_c, True),
]:
    for _ in range(2):
        mx.eval(step(fn, m, c))
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(step(fn, m, c))
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    print(f"{label:<10} 整步 {best * 1e3:8.1f}ms   吞吐 {B * T / best:7.0f} tok/s")
print(f"峰值内存: {mx.get_peak_memory() / 2**30:.2f} GB")
