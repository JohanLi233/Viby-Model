"""端到端训练单步分解基准（合成数据，不依赖外部语料）。

默认保留历史单微批口径；``--preset 1080m`` 使用当前真实配置和两个微批的
梯度累积窗口。墙钟拆成前向 / 反向 / 梯度累加 / optimizer（含 Muon NS）。

用法:
    .venv/bin/python experiments/bench_train_step.py [iters] [--bs N] [--seq N]
    .venv/bin/python experiments/bench_train_step.py 8 --preset 1080m
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer


def optimizer_due(microbatch_index: int, accumulation_steps: int) -> bool:
    """返回当前微批是否结束一个梯度累积窗口。"""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps 必须 >= 1")
    return (microbatch_index + 1) % accumulation_steps == 0


def benchmark_accounting(
    batch_size: int,
    seq_len: int,
    accumulation_steps: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    """统一窗口 token 数、吞吐和归一化单微批耗时的计算口径。"""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps 必须 >= 1")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds 必须 > 0")
    tokens = batch_size * seq_len * accumulation_steps
    return {
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed_seconds,
        "seconds_per_microbatch": elapsed_seconds / accumulation_steps,
    }


def kda_layer_counts(num_hidden_layers: int, mtp_depth: int) -> tuple[int, int]:
    """按 VibyBlock 的 3:1 local/global 规则推导 KDA 层数。"""
    main = sum(
        not (((i + 1) % 4 == 0) or i == num_hidden_layers - 1)
        for i in range(num_hidden_layers)
    )
    # 每个 MTP block 是 full-attn GQA（Qwen 口径），不计入 KDA。
    del mtp_depth
    return int(main), 0


def summarize_samples(samples) -> tuple[float, float, float]:
    """返回历史上中位/min 口径，并保留周期性刷新所需的算术均值。"""
    values = sorted(samples)
    if not values:
        raise ValueError("samples 不能为空")
    return values[len(values) // 2], values[0], sum(values) / len(values)


def build_config(args):
    return VibyConfig(
        hidden_size=768,
        num_hidden_layers=args.layers,
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=args.seq,
        mtp_depth=args.mtp,
        mtp_loss_weight=0.3,
        use_attn_gate=True,
        n_routed_experts=args.experts,
        num_experts_per_tok=args.topk,
        n_shared_experts=args.shared,
        moe_intermediate_size=args.moe_in,
        routed_scaling_factor=2.5,
        moe_router_logit_norm=True,
        moe_router_logit_temp=1.0,
        moe_diversity_loss_weight=0.0,
    )


class _Args:
    learning_rate = 0.01
    muon_ns_steps = 5
    router_lr_mult = 0.01
    # 与 train_pretrain 默认口径一致（--muonh 默认开启）；置 0 可对照旧分组
    muonh = os.environ.get("VIBY_BENCH_MUONH", "1") == "1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("iters", nargs="?", type=int, default=8)
    p.add_argument(
        "--preset",
        choices=("1080m",),
        help="应用当前 1080M 训练口径（会覆盖模型形状、compile/no-clip 与 accum）",
    )
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
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--compile",
        action="store_true",
        help="用 mx.compile 编译 loss（与真实训练 --compile_model 同口径，"
        "含融合 kernel 预热）。不加则是 eager 口径。",
    )
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--experts", type=int, default=112)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--moe_in", type=int, default=104)
    p.add_argument("--shared", type=int, default=2)
    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="每个 optimizer 窗口包含的微批数；默认 1 保留历史基准口径",
    )
    p.add_argument(
        "--no_clip",
        action="store_true",
        help="与真实训练默认 grad_clip=0 同口径：不算 clip_grad_norm，"
        "只做优化器更新（bench 默认仍 clip=1，便于和 1380ms 基线对照）。",
    )
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
    if args.preset == "1080m":
        args.bs = 12
        args.seq = 1024
        args.layers = 8
        args.experts = 256
        args.topk = 8
        args.moe_in = 384
        args.shared = 2
        args.mtp = 1
        args.compile = True
        args.no_clip = True
        args.accumulation_steps = 2
    if args.accumulation_steps < 1:
        p.error("--accumulation_steps 必须 >= 1")
    if args.ab:
        print("提示: 当前训练使用 gather_mm，旧 capacity-table --ab 已停用并忽略")

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

    opt = create_mixed_optimizer(model, _Args(), "pretrain")

    B, T = args.bs, args.seq
    rng = np.random.default_rng(0)
    # 打包语料口径：无 padding，doc_mask 段长约 340 token
    Xn = rng.integers(1, cfg.vocab_size, size=(B, T), dtype=np.int64)
    Yn = rng.integers(1, cfg.vocab_size, size=(B, T), dtype=np.int64)
    segn = np.cumsum(rng.random((B, T)) < (1.0 / 340.0), axis=1).astype(np.int64)

    def loss_fn(params, biases, X, Y, mask, seg):
        model.update(params)
        model.apply_moe_biases(biases)
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
    if args.compile:
        # 与 base_trainer 同口径：compile 前必须预热融合 kernel，否则首次
        # 校验的 host sync 落在 trace 内、被吞掉后整轮回退 eager
        # （整步 fwd+bwd 807→1189ms、峰值内存 17.4→31.0GB）。
        from mlx.utils import tree_flatten

        from model.kernels import prewarm_all

        prewarm_all(
            model,
            cfg,
            tree_flatten(model.parameters())[0][1].dtype,
            args.seq,
            log=print,
        )
        vg = mx.compile(vg)

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
    main_kda, mtp_kda = kda_layer_counts(args.layers, args.mtp)
    print(
        f"配置: bs{B}×seq{T} accum={args.accumulation_steps} mtp={args.mtp} "
        f"可训练参数 {n_params / 1e6:.2f}M"
    )
    print(f"KDA 层: 主干 {main_kda} + MTP {mtp_kda} = {main_kda + mtp_kda}")

    acc = {"fwd": [], "bwd": [], "accum": [], "opt": [], "total": []}
    for it in range(args.warmup + args.iters):
        rec = it >= args.warmup
        t0 = time.perf_counter()
        fwd_s = bwd_s = accum_s = opt_s = 0.0
        accum_grads = None
        window_stats = None
        for microbatch in range(args.accumulation_steps):
            params = model.trainable_parameters()
            biases = model.moe_bias_stack()
            (loss, stats), grads = vg(params, biases, X, Y, mask, seg)
            if args.compile:
                # compiled fn 内 model.update 只在 trace 时执行，恢复真实参数引用。
                model.update(params)
                model.apply_moe_biases(biases)

            ts = time.perf_counter()
            mx.eval(loss, stats)
            fwd_s += time.perf_counter() - ts
            ts = time.perf_counter()
            mx.eval(grads)
            bwd_s += time.perf_counter() - ts

            ts = time.perf_counter()
            if args.accumulation_steps == 1:
                # 历史基准用 grads+grads 单独测一次累加，保留可比性。
                accum_grads = tree_map(mx.add, grads, grads)
                mx.eval(accum_grads)
            elif accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(mx.add, accum_grads, grads)
                mx.eval(accum_grads)
            accum_s += time.perf_counter() - ts
            if window_stats is None or window_stats.size == 0:
                window_stats = stats
            elif stats is not None and stats.size > 0:
                window_stats = mx.concatenate([window_stats, stats], axis=1)

            if optimizer_due(microbatch, args.accumulation_steps):
                import mlx.optimizers as optim

                ts = time.perf_counter()
                g2 = accum_grads
                if not args.no_clip:
                    g2, _gn = optim.clip_grad_norm(accum_grads, 1.0)
                opt.update(model, g2)
                if window_stats is not None and window_stats.size > 0:
                    model.update_moe_biases(window_stats)
                apply_skew()
                mx.eval(model.parameters(), opt.state)
                opt_s += time.perf_counter() - ts
        t5 = time.perf_counter()

        if rec:
            acc["fwd"].append(fwd_s)
            acc["bwd"].append(bwd_s)
            acc["accum"].append(accum_s)
            acc["opt"].append(opt_s)
            acc["total"].append(t5 - t0)
        per_mb = (t5 - t0) / args.accumulation_steps
        print(
            f"  window {it:>2}{'(warm)' if not rec else '      '} "
            f"total {(t5 - t0) * 1e3:7.1f}ms  /micro {per_mb * 1e3:6.1f}  "
            f"fwd {fwd_s * 1e3:6.1f}  bwd {bwd_s * 1e3:6.1f}  "
            f"opt {opt_s * 1e3:5.1f}"
        )

    def stat(key):
        return summarize_samples(acc[key])

    tot_med, tot_min, tot_mean = stat("total")
    print(f"\n{'段':<12}{'中位(ms)':>10}{'min(ms)':>10}{'周期均值':>10}{'均值占比':>9}")
    for key, label in [
        ("fwd", "前向"),
        ("bwd", "反向"),
        ("accum", "梯度累加"),
        ("opt", "optimizer"),
    ]:
        med_v, min_v, mean_v = stat(key)
        print(
            f"{label:<12}{med_v * 1e3:>10.1f}{min_v * 1e3:>10.1f}"
            f"{mean_v * 1e3:>10.1f}{mean_v / tot_mean * 100:>8.1f}%"
        )
    print(
        f"{'累积窗口':<12}{tot_med * 1e3:>10.1f}{tot_min * 1e3:>10.1f}"
        f"{tot_mean * 1e3:>10.1f}{100.0:>8.1f}%"
    )
    med = benchmark_accounting(B, T, args.accumulation_steps, tot_med)
    cycle = benchmark_accounting(B, T, args.accumulation_steps, tot_mean)
    best = benchmark_accounting(B, T, args.accumulation_steps, tot_min)
    print(
        f"归一化单微批: {med['seconds_per_microbatch'] * 1e3:.1f}ms（中位） / "
        f"{cycle['seconds_per_microbatch'] * 1e3:.1f}ms（周期均值） / "
        f"{best['seconds_per_microbatch'] * 1e3:.1f}ms（min）"
    )
    print(
        f"吞吐: {med['tokens_per_second']:.0f} tokens/s（中位口径） / "
        f"{cycle['tokens_per_second']:.0f}（周期均值） / "
        f"{best['tokens_per_second']:.0f}（min）"
    )
    print(f"峰值内存: {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
