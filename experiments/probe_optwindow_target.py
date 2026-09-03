"""优化器窗口逐段计时 · 目标 2B 配置（真实参数树，合成梯度）。

prof_target_step 里 win_norm 115ms + win_build 7ms + win_eval 289ms
= 411ms/窗口（每微批 206ms，占墙钟 13%），而分组孤立计时只有
BatchedMuon 73 + FusedAdamW 75 = 148ms。本探针把窗口拆到底：

  accum_add      : tree_map(mx.add) 物化 4.1GB 梯度树
  grad_norm      : Σ sum(square(g)) + sqrt + float() 同步
  accum+norm 融合 : mx.compile 一把算完（少一趟 4.1GB 写+读）
  opt.update     : 构图
  opt eval       : 物化参数 + optimizer state
  QB 分位数       : update_moe_biases（13 gate × (24576,384) partition）

用法: .venv/bin/python experiments/probe_optwindow_target.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer

B, T = 12, 1024


class _Args:
    learning_rate = 1.5e-3
    muon_ns_steps = 5
    router_lr_mult = 0.01
    muonh = False


def build():
    cfg = VibyConfig(
        hidden_size=768,
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
        mtp_steps=1,
        use_attn_gate=True,
    )
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
    opt = create_mixed_optimizer(model, _Args(), "pretrain")
    return model, opt


def med(fn, reps=5, warm=1):
    ts = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if i >= warm:
            ts.append(dt * 1e3)
    return statistics.median(ts)


def main():
    model, opt = build()
    params = model.trainable_parameters()
    flat_p = tree_flatten(params)
    n_par = sum(v.size for _, v in flat_p)
    n_tensor = len(flat_p)
    gb = n_par * 2 / 2**30
    print(
        f"可训练 {n_par / 1e6:.1f}M 参数 / {n_tensor} 个张量 / bf16 {gb:.2f}GB\n"
        f"（参考：纯流式 add 在本机约 550 GB/s）"
    )

    def mk_grads(seed):
        mx.random.seed(seed)
        g = tree_map(
            lambda a: (mx.random.normal(a.shape) * 1e-3).astype(a.dtype), params
        )
        mx.eval(g)
        return g

    g1 = mk_grads(0)
    g2 = mk_grads(1)

    print(f"\n{'段':<30}{'ms':>9}{'GB/s':>9}")

    def accum():
        s = tree_map(mx.add, g1, g2)
        mx.eval(s)

    t = med(accum)
    print(f"{'tree_map(add) 累加两份梯度':<30}{t:>9.1f}{3 * gb / (t / 1e3):>9.0f}")

    gs = tree_map(mx.add, g1, g2)
    mx.eval(gs)
    gs_flat = tree_flatten(gs)

    def norm():
        v = mx.sqrt(sum(mx.sum(mx.square(g)) for _, g in gs_flat))
        float(v)

    t = med(norm)
    print(f"{'grad_norm（已物化梯度上）':<30}{t:>9.1f}{gb / (t / 1e3):>9.0f}")

    @mx.compile
    def _fused(a, b):
        s = [x + y for x, y in zip(a, b)]
        return s, mx.sqrt(sum(mx.sum(mx.square(x)) for x in s))

    a_list = [v for _, v in tree_flatten(g1)]
    b_list = [v for _, v in tree_flatten(g2)]

    def fused():
        s, nv = _fused(a_list, b_list)
        mx.eval(s)
        float(nv)

    t = med(fused)
    print(f"{'compile(累加+范数) 一把':<30}{t:>9.1f}{3 * gb / (t / 1e3):>9.0f}")

    def upd():
        opt.update(model, gs)
        mx.eval(model.parameters(), opt.state)

    t = med(upd, reps=4)
    print(f"{'opt.update + eval':<30}{t:>9.1f}{'—':>9}")

    # QB：13 gate × (24576, 384) bf16 margin 取 (1−K/E) 分位数
    n_gates = len(model.moe_gates())
    stats = (mx.random.normal((n_gates, B * T * 2, 384)) * 0.1).astype(mx.bfloat16)
    mx.eval(stats)
    sgb = stats.size * 2 / 2**30

    def qb():
        model.update_moe_biases(stats)
        mx.eval(mx.stack([g.expert_bias for g in model.moe_gates()]))

    t = med(qb)
    print(
        f"{'QB 分位数（partition）':<30}{t:>9.1f}{'—':>9}"
        f"   ({n_gates}×{B * T * 2}×384 = {sgb:.2f}GB)"
    )

    # 备选：只对 token 轴做 argpartition 选择（O(n) 而非 O(n log n)）
    def qb_topk():
        k = int(round((8 / 384) * stats.shape[1]))
        v = mx.topk(stats.astype(mx.float32).swapaxes(1, 2), k, axis=-1)[..., 0]
        mx.eval(v)

    t = med(qb_topk)
    print(f"{'QB 分位数（topk 选择）':<30}{t:>9.1f}{'—':>9}")

    # 备选：margin 在收集端就下采样（每 8 个 token 取 1）
    sub = stats[:, ::8]
    mx.eval(sub)

    def qb_sub():
        model.update_moe_biases(sub)
        mx.eval(mx.stack([g.expert_bias for g in model.moe_gates()]))

    t = med(qb_sub)
    print(f"{'QB 分位数（1/8 下采样 sort）':<30}{t:>9.1f}{'—':>9}")

    print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
