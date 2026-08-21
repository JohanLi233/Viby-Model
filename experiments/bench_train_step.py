"""端到端训练单步分解基准（合成数据，不依赖外部语料）。

按 r073 真实配置构建模型，跑若干微批并把整步墙钟拆成：
  数据准备 / 前向 / 反向 / 梯度累加 / optimizer（含 Muon NS）/ 容量表滚动。

用法:
    .venv/bin/python experiments/bench_train_step.py [iters] [--bs N] [--seq N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from model.model import VibyConfig, VibyForCausalLM
from trainer.muon import create_mixed_optimizer


def build_config(args):
    return VibyConfig(
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=8,
        kv_lora_rank=192,
        qk_rope_head_dim=32,
        vocab_size=6400,
        max_position_embeddings=args.seq,
        mtp_depth=args.mtp,
        mtp_loss_weight=0.3,
        use_value_res=True,
        use_attn_gate=True,
        hrm_H_cycles=2,
        hrm_L_cycles=3,
        hrm_bp_cycles=[2],
        hrm_emb_scale=27.7128,
        hrm_state_norm=False,
        hrm_input_skip=0.0,
        hrm_token_gate_scale=0.0,
        hrm_cycle_router=1,
        hrm_cycle_router_rank=8,
        hrm_cycle_film=1,
        engram_layers=(0,) if args.engram else (),
        engram_orders=(2, 3),
        engram_heads=0,
        engram_slots=8192,
        engram_sub_dim=128,
        engram_scale=1.0,
        engram_inject_every_cycle=0,
        n_routed_experts=112,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=104,
        routed_scaling_factor=2.5,
        moe_router_noise=0.05,
        moe_aux_loss_weight=0.001,
        moe_router_logit_norm=True,
        moe_router_logit_temp=1.0,
        moe_diversity_loss_weight=0.0,
        cycle_delta_max=0.0,
        moe_bias_update_rate=0.02,
        scale_logits_by_emb_scale=False,
    )


class _Args:
    learning_rate = 0.01
    muon_ns_steps = 5
    router_lr_mult = 0.01
    cycle_router_lr_mult = 0.1
    engram_lr_mult = 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("iters", nargs="?", type=int, default=8)
    p.add_argument(
        "--ab",
        action="store_true",
        help="同进程交替 A/B：逐次在「桶按专家 id 成组」与「按负载排序装槽」"
        "之间切换并分别取中位数。机器状态在分钟尺度上漂移可达数倍，跨进程"
        "对比不可信，交替才能把漂移摊掉。",
    )
    p.add_argument("--bs", type=int, default=6)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--mtp", type=int, default=1)
    p.add_argument("--engram", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--skew",
        type=float,
        default=0.0,
        help="路由倾斜强度（expert_bias 的标准差倍数，Zipf 形状、热专家散布到"
        "随机 id）。随机初始化的 router + 随机 token 路由近似均匀，而真实训练"
        "会发展出 max/mean 4-10 的负载倾斜，桶 padding 由此放大；0.35 左右可"
        "复现该量级（跑完看输出的 max/mean 标定）。",
    )
    args = p.parse_args()

    cfg = build_config(args)
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

    gates = model.moe_gates()
    for g in gates:
        g.collect_stats = True

    def apply_skew():
        """把固定的 Zipf 形状偏置写回 expert_bias（每步重写，抵消训练循环的
        负载均衡更新），使路由维持目标倾斜度。"""
        if args.skew <= 0:
            return
        rng = np.random.default_rng(7)
        E = cfg.n_routed_experts
        z = 1.0 / np.arange(1, E + 1)
        z = z[rng.permutation(E)]
        z = (z - z.mean()) / (z.std() + 1e-9)
        for g in gates:
            b = g.expert_bias
            v = mx.array((z * args.skew).astype(np.float32))
            g.expert_bias = mx.broadcast_to(v, b.shape).astype(b.dtype)

    apply_skew()
    mx.eval(model.parameters())

    moe_mods = [m for m in model.modules() if hasattr(m, "_cap_peak")]

    def force_id_grouping():
        """把容量表改回按专家 id 成组（容量推导规则不变），作为 A/B 的对照臂"""
        for mod in moe_mods:
            E = mod.n_routed
            EG = min(mod._SPARSE_GROUP, E)
            AL = mod._SPARSE_ALIGN
            ng = (E + EG - 1) // EG
            for k in list(mod._cap_peak):
                pk = mod._cap_peak[k]
                mod._cap_order[k] = list(range(E))
                mod._cap_perm[k] = list(range(E))
                mod._cap_table[k] = [
                    max(
                        (int(max(pk[gi * EG : (gi + 1) * EG])) * 5 // 4 + 64 + AL - 1)
                        // AL
                        * AL,
                        AL,
                    )
                    for gi in range(ng)
                ]

    opt = create_mixed_optimizer(model, _Args(), "pretrain")

    B, T = args.bs, args.seq
    rng = np.random.default_rng(0)
    # 打包语料口径：无 padding，doc_mask 段长约 340 token
    Xn = rng.integers(1, cfg.vocab_size, size=(B, T), dtype=np.int64)
    Yn = rng.integers(1, cfg.vocab_size, size=(B, T), dtype=np.int64)
    segn = np.cumsum(rng.random((B, T)) < (1.0 / 340.0), axis=1).astype(np.int64)

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
        stats = model.moe_load_stats()
        return res.loss, stats

    vg = mx.value_and_grad(loss_fn, argnums=0)

    X = mx.array(Xn)
    Y = mx.array(Yn)
    mask = mx.ones((B, T), dtype=mx.int64)
    seg = mx.array(segn)

    n_params = sum(
        v.size
        for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
            model.trainable_parameters()
        )
    )
    print(
        f"配置: bs{B}×seq{T} mtp={args.mtp} engram={args.engram} 可训练参数 {n_params / 1e6:.2f}M"
    )

    acc = {"fwd": [], "bwd": [], "accum": [], "opt": [], "cap": [], "total": []}
    ab = {False: [], True: []}  # 按 id 成组 / 按负载装槽 的整步耗时
    ab_pad = {False: [0, 0], True: [0, 0]}
    arm = True  # 本次迭代使用的分组方式（--ab 时逐次翻转）

    for it in range(args.warmup + args.iters):
        rec = it >= args.warmup
        rows0 = sum(m._rows_seen for m in moe_mods)
        pairs0 = sum(m._pairs_seen for m in moe_mods)
        t0 = time.perf_counter()

        params = model.trainable_parameters()
        (loss, stats), grads = vg(params, X, Y, mask, seg)
        mx.eval(loss, stats)
        t1 = time.perf_counter()
        mx.eval(grads)
        t2 = time.perf_counter()

        accum = tree_map(mx.add, grads, grads)
        mx.eval(accum)
        t3 = time.perf_counter()

        ov = 0
        for m in model.modules():
            f = getattr(m, "update_capacity_table", None)
            if f is not None:
                ov += f()
        if args.ab:
            arm = not arm  # 下一次迭代换另一条臂
            if not arm:
                force_id_grouping()
        t4 = time.perf_counter()

        import mlx.optimizers as optim

        g2, gn = optim.clip_grad_norm(accum, 1.0)
        opt.update(model, g2)
        model.update_moe_biases(stats, 0.02)
        apply_skew()  # 维持目标倾斜，抵消 bias 均衡更新
        mx.eval(model.parameters(), opt.state)
        t5 = time.perf_counter()

        if rec and args.ab:
            # 本次迭代跑的是切换「之前」的那条臂
            used = not arm if args.ab else True
            ab[used].append(t5 - t0)
            ab_pad[used][0] += sum(m._rows_seen for m in moe_mods) - rows0
            ab_pad[used][1] += sum(m._pairs_seen for m in moe_mods) - pairs0
        if rec:
            acc["fwd"].append(t1 - t0)
            acc["bwd"].append(t2 - t1)
            acc["accum"].append(t3 - t2)
            acc["cap"].append(t4 - t3)
            acc["opt"].append(t5 - t4)
            acc["total"].append(t5 - t0)
        print(
            f"  iter {it:>2}{'(warm)' if not rec else '      '} "
            f"total {(t5 - t0) * 1e3:7.1f}ms  fwd {(t1 - t0) * 1e3:6.1f}  "
            f"bwd {(t2 - t1) * 1e3:6.1f}  opt {(t5 - t4) * 1e3:5.1f}"
        )

    def stat(key):
        v = sorted(acc[key])
        n = len(v)
        return v[n // 2], v[0]

    tot_avg = stat("total")[0]
    print(f"\n{'段':<12}{'中位(ms)':>10}{'min(ms)':>10}{'占比':>8}")
    for key, label in [
        ("fwd", "前向"),
        ("bwd", "反向"),
        ("accum", "梯度累加"),
        ("cap", "容量表"),
        ("opt", "optimizer"),
    ]:
        a, m = stat(key)
        print(f"{label:<12}{a * 1e3:>10.1f}{m * 1e3:>10.1f}{a / tot_avg * 100:>7.1f}%")
    a, m = stat("total")
    print(f"{'整步':<12}{a * 1e3:>10.1f}{m * 1e3:>10.1f}{100.0:>7.1f}%")
    print(f"吞吐: {B * T / a:.0f} tokens/s（中位口径） / {B * T / m:.0f}（min）")
    print(f"峰值内存: {mx.get_peak_memory() / 2**30:.2f} GB")
    rows = pairs = 0
    for mod in model.modules():
        if getattr(mod, "_calls_seen", 0):
            rows += mod._rows_seen
            pairs += mod._pairs_seen
    if pairs:
        print(f"MoE 桶 padding: {rows / pairs:.2f}x（桶行数 / 真实 pair 数）")
    if args.ab and ab[True] and ab[False]:
        import statistics

        a = statistics.median(ab[False])
        b = statistics.median(ab[True])
        pa = ab_pad[False][0] / max(ab_pad[False][1], 1)
        pb = ab_pad[True][0] / max(ab_pad[True][1], 1)
        print(
            f"\n同进程交替 A/B（各 {len(ab[False])}/{len(ab[True])} 次，中位）:\n"
            f"  桶按专家 id 成组   {a * 1e3:7.1f}ms  padding {pa:.2f}x\n"
            f"  桶按负载排序装槽   {b * 1e3:7.1f}ms  padding {pb:.2f}x\n"
            f"  整步提速           {a / b:.2f}x"
        )


if __name__ == "__main__":
    main()
