"""对同一份真实动量 U，直接对比三者的正交残差：
  full  = Gram-NS5(U)            —— 刷新步的输出
  cache = Q_prev @ normalize(U)  —— 命中步的输出
  以及 Q 来源那一步的 U 与当前 U 的相对变化。

用于区分「残差阈值不可达（full 自身残差就超阈值）」与「命中路径实现有
bug（full 小、cache 大）」。

用法: uv run python experiments/probe_cache_q_gap.py [steps]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import BatchedMuon, create_mixed_optimizer

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

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

B, T = 12, 1024
rng = np.random.default_rng(0)
mask = mx.ones((B, T), dtype=mx.int64)


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

# 拦截 Uflat：在 apply_gradients 的堆叠组里，_ns5_gram / _apply_cached_Q
# 的第一个数组参数就是当前 Uflat
SEEN: dict = {}
orig_gram = BatchedMuon._ns5_gram
orig_cached = BatchedMuon._apply_cached_Q


def cap_gram(self, X, *a, **kw):
    SEEN.setdefault(tuple(X.shape), {})["U"] = X
    return orig_gram(self, X, *a, **kw)


def cap_cached(self, Q, X, tr):
    SEEN.setdefault(tuple(X.shape), {})["U"] = X
    return orig_cached(self, Q, X, tr)


BatchedMuon._ns5_gram = cap_gram
BatchedMuon._apply_cached_Q = cap_cached

prev_U: dict = {}
prev_Q: dict = {}

print(f"{'步':>3} {'形状':>18} {'full残差':>9} {'cache残差':>10} {'U相对变化':>10}")
for i in range(STEPS):
    X = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
    Y = mx.array(rng.integers(1, cfg.vocab_size, size=(B, T)))
    seg = mx.array(np.cumsum(rng.random((B, T)) < (1 / 340.0), axis=1).astype(np.int64))
    params = model.trainable_parameters()
    (loss, stats), grads = vg(params, X, Y, mask, seg)
    mx.eval(grads)
    SEEN.clear()
    opt.update(model, grads)
    if stats is not None:
        model.update_moe_biases(stats)
    mx.eval(model.parameters(), opt.state)

    for shape, d in sorted(SEEN.items()):
        U = d["U"]
        # full：本步动量上跑完整 Gram-NS5
        D_full, Q_new, tr = orig_gram(muon, U, return_Q=True)
        r_full = float(BatchedMuon._orth_residual(D_full).item())
        # cache：用上一步的 Q 作用在本步动量上
        if shape in prev_Q:
            D_cache = orig_cached(muon, prev_Q[shape], U, tr)
            r_cache = float(BatchedMuon._orth_residual(D_cache).item())
        else:
            r_cache = float("nan")
        if shape in prev_U:
            dU = float(
                (
                    mx.linalg.norm((U - prev_U[shape]).astype(mx.float32))
                    / (mx.linalg.norm(prev_U[shape].astype(mx.float32)) + 1e-12)
                ).item()
            )
        else:
            dU = float("nan")
        print(f"{i:>3} {str(shape):>18} {r_full:>9.4f} {r_cache:>10.4f} {dU:>10.4f}")
        prev_U[shape] = U
        prev_Q[shape] = Q_new
