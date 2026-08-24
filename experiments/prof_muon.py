"""MuonH 优化器分项剖析：r081 真实口径（L8+MTP, E256/K8/I320, bf16）。

构建真实模型跑一次 fwd/bwd 拿真实梯度，然后连续跑 N 步 optimizer，
用计时 wrapper 包住 BatchedMuon 的 _ns5 / _apply_cached_Q / mom/apply
kernel，拆出：
  - 2D 矩阵组 NS5（每步都跑）
  - 专家堆叠组 refresh 步 NS5(+Q)
  - 专家堆叠组 cache-hit 步 Q@U
  - 逐元素 mom/apply 融合 kernel

用法: .venv/bin/python experiments/prof_muon.py [steps]
"""

import os
import sys
import time
from collections import defaultdict
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import BatchedMuon, create_mixed_optimizer

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 18

cfg = VibyConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    vocab_size=6400,
    max_position_embeddings=1024,
    mtp_depth=1,
    mtp_loss_weight=0.3,
    use_attn_gate=True,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=1,
    moe_intermediate_size=320,
    routed_scaling_factor=2.5,
    moe_router_noise=0.05,
    moe_router_logit_norm=True,
    moe_router_logit_temp=1.0,
    moe_diversity_loss_weight=0.0,
)

model = VibyForCausalLM(cfg)
model.update(
    tree_map(
        lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a,
        model.parameters(),
    )
)
mx.eval(model.parameters())
model.train()


class _Args:
    learning_rate = 0.01
    muon_ns_steps = 5
    router_lr_mult = 0.01
    muonh = True


opt = create_mixed_optimizer(model, _Args(), "pretrain")

# 真实梯度：一次 fwd/bwd
B, T = 12, 1024
rng = np.random.default_rng(0)
X = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
Y = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
mask = mx.ones((B, T), dtype=mx.int64)
seg = mx.array(np.cumsum(rng.random((B, T)) < (1.0 / 340.0), axis=1).astype(np.int64))


def loss_fn(params, X, Y, mask, seg):
    model.update(params)
    res = model(
        input_ids=X,
        labels=Y,
        loss_mask=mask,
        attention_mask=None,
        mask_has_pad=False,
        segment_ids=seg,
    )
    stats = model.qb_margin_stats()
    return res.loss, stats


vg = mx.value_and_grad(loss_fn, argnums=0)
params = model.trainable_parameters()
(loss, stats), grads = vg(params, X, Y, mask, seg)
mx.eval(grads)
print(f"loss={float(loss):.3f}  梯度就绪")

# ---- 计时 wrapper ----
TIMES = defaultdict(list)
SHAPES = set()

orig_ns5 = BatchedMuon._ns5
orig_ns5_gram = BatchedMuon._ns5_gram
orig_cached = BatchedMuon._apply_cached_Q


def _tag(args):
    X = args[1]
    return tuple(X.shape)


def timed_ns5(self, X, *a, **kw):
    key = ("ns5(+Q)" if kw.get("return_Q") else "ns5", _tag((self, X)))
    SHAPES.add(key)
    t0 = time.perf_counter()
    r = orig_ns5(self, X, *a, **kw)
    mx.eval(r)
    TIMES[key[0]].append((time.perf_counter() - t0, key[1]))
    return r


def timed_ns5g(self, X, *a, **kw):
    key = ("ns5_gram(+Q)" if kw.get("return_Q") else "ns5_gram", _tag((self, X)))
    SHAPES.add(key)
    t0 = time.perf_counter()
    r = orig_ns5_gram(self, X, *a, **kw)
    mx.eval(r)
    TIMES[key[0]].append((time.perf_counter() - t0, key[1]))
    return r


def timed_cached(self, Q, X, tr):
    key = ("cache_Q@U", tuple(X.shape))
    SHAPES.add(key)
    t0 = time.perf_counter()
    r = orig_cached(self, Q, X, tr)
    mx.eval(r)
    TIMES[key[0]].append((time.perf_counter() - t0, key[1]))
    return r


BatchedMuon._ns5 = timed_ns5
BatchedMuon._ns5_gram = timed_ns5g
BatchedMuon._apply_cached_Q = timed_cached

# 找到 MuonH 子优化器
muon_opt = None
for o in opt.optimizers:
    if isinstance(o, BatchedMuon):
        muon_opt = o
assert muon_opt is not None, "未找到 BatchedMuon"
print(
    f"BatchedMuon: ns_steps={muon_opt.ns_steps} ns_bf16={muon_opt.ns_bf16} "
    f"stack_ns_every={muon_opt.stack_ns_every} cache_q={muon_opt.stack_cache_q} "
    f"stack_ns_steps={muon_opt.stack_ns_steps}"
)

# 直接对 BatchedMuon 子优化器跑（跳过 MultiOptimizer 分发开销，但 retain
# 真实梯度形状）；梯度属于非 Muon 组的张量会被 filter 前被剔——这里手动筛

flat_g = dict(tree_flatten(grads))
muon_grads = {
    k: v for k, v in flat_g.items() if v.ndim >= 2 and ".experts." in k or False
}
# 实际上直接跑整个 opt.update 更接近真实：
step_ts = []
for i in range(STEPS):
    g2, gn = optim.clip_grad_norm(grads, 1.0)
    mx.eval(g2)
    t0 = time.perf_counter()
    opt.update(model, g2)
    mx.eval(model.parameters(), opt.state)
    dt = time.perf_counter() - t0
    step_ts.append(dt)
    print(f"  opt step {i:>2}: {dt * 1e3:7.1f}ms")

print("\n=== 分项累计 ===")
for k, v in sorted(TIMES.items()):
    tot = sum(t for t, _ in v)
    shapes = defaultdict(float)
    for t, s in v:
        shapes[s] += t
    print(f"{k:14s} 共 {tot * 1e3:8.1f}ms  ({len(v)} 次)")
    for s, t in sorted(shapes.items(), key=lambda x: -x[1]):
        print(f"    {str(s):24s} {t * 1e3:8.1f}ms")

tot = sum(step_ts)
print(f"\nopt 总耗时 {tot * 1e3:.1f}ms / {STEPS} 步 = {tot / STEPS * 1e3:.1f}ms/步")
print(f"峰值内存: {mx.get_peak_memory() / 2**30:.2f} GB")
