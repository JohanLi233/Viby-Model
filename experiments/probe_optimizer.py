"""optimizer 段耗时分解：clip_grad_norm / Muon NS / AdamW（分组）。

655M 参数里 552M 是 3D 堆叠专家权重（走 AdamW 标量组），整步 optimizer
实测占 13%。这里按参数组拆开计时，看是 Newton-Schulz、AdamW 的逐元素
带宽、还是 clip_grad_norm 的全参数归约在主导。

用法: uv run experiments/probe_optimizer.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer


class _Args:
    learning_rate = 0.01
    muon_ns_steps = 5
    router_lr_mult = 0.01


def timed(fn, it=6, w=2):
    for _ in range(w):
        fn()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    E = int(os.environ.get("VIBY_BENCH_E", 288))
    I = int(os.environ.get("VIBY_BENCH_I", 104))  # noqa: E741
    L = int(os.environ.get("VIBY_BENCH_L", 8))
    cfg = VibyConfig(
        hidden_size=768,
        num_hidden_layers=L,
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=1024,
        mtp_depth=1,
        use_attn_gate=True,
        n_routed_experts=E,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=I,
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

    params = model.trainable_parameters()
    flat = tree_flatten(params)
    n_all = sum(v.size for _, v in flat)
    n_exp = sum(v.size for p, v in flat if ".experts." in p)
    print(f"可训练参数 {n_all / 1e6:.1f}M，其中堆叠专家权重 {n_exp / 1e6:.1f}M")

    grads = tree_map(lambda a: mx.ones_like(a) * 0.01, params)
    mx.eval(grads)

    opt = create_mixed_optimizer(model, _Args(), "pretrain")
    # 先跑一步把 optimizer state 建起来（含 AdamW 的 m/v 分配）
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)

    def do_clip():
        g2, gn = optim.clip_grad_norm(grads, 1.0)
        mx.eval(g2, gn)

    def do_update():
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    def do_accum():
        mx.eval(tree_map(mx.add, grads, grads))

    t_clip = timed(do_clip)
    t_upd = timed(do_update)
    t_acc = timed(do_accum)
    print(f"\nclip_grad_norm(全参数)      {t_clip * 1e3:7.1f}ms")
    print(f"opt.update(全部分组)        {t_upd * 1e3:7.1f}ms")
    print(f"梯度累加 tree_map(add)      {t_acc * 1e3:7.1f}ms")

    # 分组隔离：只喂某一组的梯度，其余组梯度置零形状不变不可行（MultiOptimizer
    # 按 filter 分派），改为直接对各组参数子集用独立优化器计时。
    print("\n按参数组隔离（各自新建同类优化器，state 已预热）")
    groups = {
        "堆叠专家权重(AdamW)": [(p, v) for p, v in flat if ".experts." in p],
        "Muon 核心矩阵": [
            (p, v)
            for p, v in flat
            if v.ndim >= 2
            and ".experts." not in p
            and ".router." not in p
            and "embed" not in p
            and "lm_head" not in p
        ],
        "嵌入(AdamW)": [(p, v) for p, v in flat if "embed" in p or "lm_head" in p],
    }
    from trainer.muon import BatchedMuon

    for name, items in groups.items():
        if not items:
            continue
        sub = {p: v for p, v in items}
        sub_g = {p: mx.ones_like(v) * 0.01 for p, v in items}
        mx.eval(list(sub_g.values()))
        o = (
            BatchedMuon(learning_rate=mx.array(0.01), momentum=0.95, ns_steps=5)
            if name.startswith("Muon")
            else optim.AdamW(learning_rate=mx.array(0.01), weight_decay=0.1)
        )
        state = {"p": sub}

        def step():
            state["p"] = o.apply_gradients(sub_g, state["p"])
            mx.eval(state["p"], o.state)

        step()
        t = timed(step)
        n = sum(v.size for _, v in items)
        print(f"  {name:<22}{len(items):>4} 张量 {n / 1e6:>7.1f}M  {t * 1e3:>7.1f}ms")


if __name__ == "__main__":
    main()
