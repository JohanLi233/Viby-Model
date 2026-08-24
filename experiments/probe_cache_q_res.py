"""诊断 Temporal MuonH 的残差门（stack_cache_q_res）实际触发率。

背景：muonh 默认 stack_ns_every=8 + stack_cache_q_res=0.15。命中步先算
Q@U_norm，再测 ||DDᵀ−I||_F/n，超阈值就改跑完整 Gram-NS5。若阈值不可达，
每步都会「先做一遍 Q@U 再做一遍完整 NS」，比不开缓存还慢。

本探针跑真实模型 + 每步不同的合成 batch，打印：
  - 每步各堆叠组的残差与是否 refresh
  - 对照：完整 NS5 自身输出的残差（阈值可达性的下界）

用法: uv run python experiments/probe_cache_q_res.py [steps]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import BatchedMuon, create_mixed_optimizer

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

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
    n_shared_experts=2,
    moe_intermediate_size=384,
    routed_scaling_factor=2.5,
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
muon = next(o for o in opt.optimizers if isinstance(o, BatchedMuon))
print(
    f"stack_ns_every={muon.stack_ns_every} cache_q={muon.stack_cache_q} "
    f"cache_q_res={muon.stack_cache_q_res} ns_bf16={muon.ns_bf16}"
)

B, T = 12, 1024
rng = np.random.default_rng(0)


def make_batch():
    X = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
    Y = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
    seg = mx.array(np.cumsum(rng.random((B, T)) < (1 / 340.0), axis=1).astype(np.int64))
    return X, Y, seg


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
    return res.loss, model.qb_margin_stats()


vg = mx.value_and_grad(loss_fn, argnums=0)
mask = mx.ones((B, T), dtype=mx.int64)

# ---- 插桩：记录每步各组的残差 / refresh 决策 ----
LOG: list = []
orig_res = BatchedMuon._orth_residual
orig_gram = BatchedMuon._ns5_gram
orig_cached = BatchedMuon._apply_cached_Q
cur: dict = {}


@staticmethod
def probe_res(D, probe=32):
    r = orig_res(D, probe)
    mx.eval(r)
    cur.setdefault("res", []).append((tuple(D.shape), float(r.item())))
    return r


def probe_gram(self, X, *a, **kw):
    cur.setdefault("refresh", []).append(tuple(X.shape))
    return orig_gram(self, X, *a, **kw)


def probe_cached(self, Q, X, tr):
    cur.setdefault("hit", []).append(tuple(X.shape))
    return orig_cached(self, Q, X, tr)


BatchedMuon._orth_residual = probe_res
BatchedMuon._ns5_gram = probe_gram
BatchedMuon._apply_cached_Q = probe_cached

print(f"\n{'步':>3} {'耗时ms':>8} {'命中':>4} {'刷新':>4}  残差(各组)")
n_refresh = 0
for i in range(STEPS):
    X, Y, seg = make_batch()
    params = model.trainable_parameters()
    (loss, stats), grads = vg(params, X, Y, mask, seg)
    mx.eval(grads)
    cur.clear()
    t0 = time.perf_counter()
    opt.update(model, grads)
    if stats is not None:
        model.update_moe_biases(stats)
    mx.eval(model.parameters(), opt.state)
    dt = (time.perf_counter() - t0) * 1e3
    res = cur.get("res", [])
    nh, nr = len(cur.get("hit", [])), len(cur.get("refresh", []))
    n_refresh += nr
    rs = "  ".join(f"{s[0]}x{s[1]}x{s[2]}:{v:.3f}" for s, v in res)
    print(f"{i:>3} {dt:>8.1f} {nh:>4} {nr:>4}  {rs}")

print(f"\n刷新总次数 {n_refresh} / {STEPS} 步（每 8 步应为 1 组次）")

# ---- 阈值可达性：完整 NS5 输出自身的残差 ----
print("\n=== 完整 NS5 输出自身的残差（阈值下界）===")
BatchedMuon._orth_residual = orig_res
for shape in [(2304, 640, 384), (2304, 384, 384)]:
    U = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
    Dg, Q, tr = orig_gram(muon, U, return_Q=True)
    Ds = orig_ns5(muon, U) if (orig_ns5 := BatchedMuon._ns5) else None
    mx.eval(Dg, Ds)
    print(
        f"  {shape}: Gram-NS5 残差 {float(orig_res(Dg).item()):.4f}  "
        f"标准 NS5 残差 {float(orig_res(Ds).item()):.4f}"
    )
