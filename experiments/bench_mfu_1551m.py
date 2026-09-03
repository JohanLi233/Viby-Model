"""1551M 配方整步分段基准 + 同进程交替 A/B（定位 MFU 缺口）。

与 ``trainer/train_pretrain.py`` 同口径：用 ``get_pretrain_parser`` 解析同一
套参数、同样的 VibyConfig 拼装、同样的 prewarm + mx.compile、同样带 LAR
分支的 loss 签名。合成打包语料（无 padding、doc_mask 段长约 340）。

跨进程对比在本机有 ±30% 漂移（research/MLX_PERF.md §2.2），所以所有 arm
必须**同进程轮转**、丢弃前 2 轮、取中位数。

用法:
    uv run experiments/bench_mfu_1551m.py --rounds 4
    uv run experiments/bench_mfu_1551m.py --rounds 3 --ns_arms 5,3
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from model.config import VibyConfig
from model.flops import (
    DEFAULT_PEAK_TFLOPS,
    model_flops_utilization,
    training_flops_per_token,
)
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer

# 与用户实际训练命令一致的参数（--data_path 是占位，bench 用合成数据）
BASE_ARGV = (
    "--data_path /dev/null --hidden_size 768 --num_hidden_layers 9 "
    "--num_attention_heads 8 --kv_lora_rank 256 --qk_rope_head_dim 64 "
    "--no-use_linear_attn --first_k_dense_replace 1 "
    "--dense_intermediate_size 768 --n_routed_experts 128 "
    "--num_experts_per_tok 8 --n_shared_experts 2 --moe_intermediate_size 512 "
    "--moe_latent_dim 384 --routed_scaling_factor 2.5 --moe_route_scale "
    "--ngram_table_size 1048576 --ngram_layer 2 --mtp_depth 1 --mtp_steps 1 "
    "--use_attn_gate --batch_size 12 --max_seq_len 1024 --epochs 1 "
    "--compile_model --cache_limit_gb 0 --grad_clip 0"
)


def build_config():
    """完全复刻 train_pretrain.py 里 VibyConfig 的拼装，保证口径一致。"""
    from trainer.config import get_pretrain_parser

    parser = get_pretrain_parser()
    args = parser.parse_args(BASE_ARGV.split())
    return VibyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        mtp_steps=args.mtp_steps,
        use_attn_gate=args.use_attn_gate,
        attn_res_window=getattr(args, "attn_res_window", 4),
        attn_res_register=getattr(args, "attn_res_register", False),
        attn_res_read_h=getattr(args, "attn_res_read_h", False),
        ihc=getattr(args, "ihc", False),
        ihc_streams=getattr(args, "ihc_streams", 4),
        ihc_typed=getattr(args, "ihc_typed", False),
        ihc_collapse=getattr(args, "ihc_collapse", None),
        ihc_ngram_stream=getattr(args, "ihc_ngram_stream", 1),
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        routed_scaling_factor=args.routed_scaling_factor,
        moe_router_logit_norm=args.moe_router_logit_norm,
        moe_router_logit_temp=args.moe_router_logit_temp,
        moe_diversity_loss_weight=args.moe_diversity_loss_weight,
        z_loss_weight=args.z_loss_weight,
        moe_latent_dim=args.moe_latent_dim,
        moe_write_spread=getattr(args, "moe_write_spread", False),
        moe_route_scale=getattr(args, "moe_route_scale", False),
        first_k_dense_replace=getattr(args, "first_k_dense_replace", 0),
        **(
            {"dense_intermediate_size": args.dense_intermediate_size}
            if getattr(args, "dense_intermediate_size", None) is not None
            else {}
        ),
        kda_v_head_ratio=args.kda_v_head_ratio,
        ngram_table_size=args.ngram_table_size,
        ngram_layer=args.ngram_layer,
        ngram_logit_skip=getattr(args, "ngram_logit_skip", False),
        ngram_conf_gate=getattr(args, "ngram_conf_gate", False),
        tie_word_embeddings=args.tie_word_embeddings,
        use_linear_attn=args.use_linear_attn,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
    )


def make_loss_fn(model, accum_steps, unemb_in, unemb_out):
    """与 BaseTrainer._loss_fn 同结构（含 LAR 分支），返回可 compile 的函数。"""

    def _loss_fn(
        params,
        moe_biases,
        X,
        Y,
        loss_mask,
        attn_mask,
        mask_has_pad,
        seg_ids,
        compute_lar,
    ):
        model.update(params)
        model.apply_moe_biases(moe_biases)
        res = model(
            input_ids=X,
            labels=Y,
            loss_mask=loss_mask,
            attention_mask=attn_mask,
            mask_has_pad=mask_has_pad,
            segment_ids=seg_ids,
        )
        mtp_loss = res.mtp_loss if res.mtp_loss is not None else mx.array(0.0)
        lm_loss = res.lm_loss if res.lm_loss is not None else res.loss
        z_loss = res.z_loss if res.z_loss is not None else mx.array(0.0)
        moe_stats = model.qb_margin_stats()
        if moe_stats is None:
            moe_stats = mx.zeros((0,), dtype=mx.float32)
        load_stats = model.moe_load_stats()
        if load_stats is None:
            load_stats = mx.zeros((0,), dtype=mx.float32)
        if compute_lar:
            valid_f32 = (loss_mask > 0).astype(mx.float32)
            unemb_weight = model.lm_head.weight.astype(mx.float32)
            hidden_f32 = res.hidden_states.astype(mx.float32)
            logits_f32 = res.logits.astype(mx.float32)
            token_count = mx.maximum(mx.sum(valid_f32), 1.0)
            output_dim = unemb_out
            input_dim = unemb_in
            wx_ms = mx.sum(logits_f32 * logits_f32 * valid_f32[:, :, None]) / (
                token_count * output_dim
            )
            weight_ms = mx.sum(unemb_weight * unemb_weight) / (output_dim * input_dim)
            hidden_ms = mx.sum(hidden_f32 * hidden_f32 * valid_f32[:, :, None]) / (
                token_count * input_dim
            )
            lar = (
                0.5
                * (mx.log(wx_ms) - mx.log(weight_ms) - mx.log(hidden_ms))
                / mx.log(mx.array(input_dim, dtype=mx.float32))
            )
        else:
            lar = mx.array(0.0)
        return (
            res.loss / accum_steps,
            mtp_loss,
            moe_stats,
            lm_loss / accum_steps,
            mx.array(0.0),
            lar,
            z_loss,
            load_stats,
        )

    return _loss_fn


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4, help="每个 arm 的轮数（丢弃前 2）")
    p.add_argument(
        "--accum_arms",
        type=str,
        default="2,4,8",
        help="梯度累积窗口大小候选（逗号分隔）",
    )
    p.add_argument(
        "--ns_arms",
        type=str,
        default="5",
        help="Muon NS 迭代步数候选（逗号分隔）；>1 个时为每组各建一个优化器",
    )
    p.add_argument("--lar", type=int, default=1, help="是否每步算 LAR（1=复刻线上口径）")
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    accum_arms = [int(x) for x in args.accum_arms.split(",") if x.strip()]
    ns_arms = [int(x) for x in args.ns_arms.split(",") if x.strip()]

    cfg = build_config()
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

    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"可训练参数 {n_params / 1e6:.1f}M  dtype bf16  48GB 统一内存")

    class _OptArgs:
        learning_rate = 8.0e-4
        muon_lr = 3.47e-3
        adam_beta2 = 0.99981
        adam_eps = 2.949e-15
        muonh = True
        router_lr_mult = 0.01

    opts = {}
    for ns in ns_arms:
        oa = _OptArgs()
        oa.muon_ns_steps = ns
        opts[ns] = create_mixed_optimizer(model, oa, "pretrain")
        print(f"  优化器已建: muon_ns_steps={ns}")

    accum_default = max(accum_arms)
    unemb = model.lm_head if model.lm_head is not None else model.model.embed_tokens
    loss_fn = make_loss_fn(
        model, accum_default, int(unemb.weight.shape[1]), int(unemb.weight.shape[0])
    )
    vg = mx.value_and_grad(loss_fn, argnums=0)

    from model.kernels import prewarm_all

    prewarm_all(
        model, cfg, tree_flatten(model.parameters())[0][1].dtype, 1024, log=print
    )
    print("编译 loss 函数 ...")
    vg = mx.compile(vg)
    print("编译完成\n")

    B, T = 12, 1024
    rng = np.random.default_rng(0)
    vocab = int(cfg.vocab_size)
    Xs, Ys, Ss = [], [], []
    for _ in range(max(accum_arms) + 2):
        Xs.append(mx.array(rng.integers(1, vocab, size=(B, T), dtype=np.int64)))
        Ys.append(mx.array(rng.integers(1, vocab, size=(B, T), dtype=np.int64)))
        Ss.append(
            mx.array(np.cumsum(rng.random((B, T)) < (1.0 / 340.0), axis=1, dtype=np.int64))
        )
    loss_mask = mx.ones((B, T), dtype=mx.int64)
    attn_mask = mx.ones((B, T), dtype=mx.int32)
    flops = training_flops_per_token(model, T)
    print(f"FLOPs/token: {flops / 1e9:.3f}G   peak {DEFAULT_PEAK_TFLOPS} TFLOPS\n")

    results = {}

    def run_window(accum, ns_step, s_idx):
        opt = opts[ns_step]
        rec = {"fwd": 0.0, "bwd": 0.0, "accum": 0.0, "opt": 0.0, "total": 0.0}
        t0 = time.perf_counter()
        accum_grads = None
        window_stats = None
        for m in range(accum):
            params = model.trainable_parameters()
            biases = model.moe_bias_stack()
            outs, grads = vg(
                params,
                biases,
                Xs[(s_idx + m) % len(Xs)],
                Ys[(s_idx + m) % len(Ys)],
                loss_mask,
                attn_mask,
                False,
                Ss[(s_idx + m) % len(Ss)],
                bool(args.lar),
            )
            model.update(params)
            model.apply_moe_biases(biases)
            ts = time.perf_counter()
            mx.eval(outs)
            rec["fwd"] += time.perf_counter() - ts
            ts = time.perf_counter()
            mx.eval(grads)
            rec["bwd"] += time.perf_counter() - ts

            ts = time.perf_counter()
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(mx.add, accum_grads, grads)
                mx.eval(accum_grads)
            rec["accum"] += time.perf_counter() - ts

            st = outs[2]
            if window_stats is None or window_stats.size == 0:
                window_stats = st
            elif st.size > 0:
                window_stats = mx.concatenate([window_stats, st], axis=1)

        ts = time.perf_counter()
        opt.update(model, accum_grads)
        if window_stats is not None and window_stats.size > 0:
            model.update_moe_biases(window_stats)
        mx.eval(model.parameters(), opt.state)
        rec["opt"] += time.perf_counter() - ts
        rec["total"] = time.perf_counter() - t0
        return rec

    arms = [(a, ns) for ns in ns_arms for a in accum_arms]
    for a in arms:
        results[a] = {k: [] for k in ("fwd", "bwd", "accum", "opt", "total")}

    print(f"轮转 {args.warmup + args.rounds} 轮 × {len(arms)} arm ...")
    for r in range(args.warmup + args.rounds):
        for (a, ns) in arms:
            rec = run_window(a, ns, r)
            if r >= args.warmup:
                for k in rec:
                    results[(a, ns)][k].append(rec[k])
            tag = "warm" if r < args.warmup else "    "
            per_mb = rec["total"] / a * 1e3
            tps = B * T / (rec["total"] / a)
            print(
                f"  r{r} accum={a} ns={ns} {tag} /micro {per_mb:6.1f}ms  "
                f"fwd {rec['fwd'] * 1e3:6.1f}  bwd {rec['bwd'] * 1e3:7.1f}  "
                f"acc {rec['accum'] * 1e3:5.1f}  opt {rec['opt'] * 1e3:6.1f}  "
                f"{tps:6.0f} tok/s  MFU {model_flops_utilization(tps, flops):5.1%}"
            )

    print(f"\n{'arm':<16}{'/micro':>9}{'fwd':>8}{'bwd':>9}{'acc':>7}{'opt':>8}"
          f"{'opt占比':>8}{'tok/s':>9}{'MFU':>8}")
    for (a, ns) in arms:
        r = results[(a, ns)]
        tot = median(r["total"])
        per_mb = tot / a
        tps = B * T / per_mb
        mfu = model_flops_utilization(tps, flops)
        opt = median(r["opt"])
        print(
            f"accum={a} ns={ns:<4}{per_mb * 1e3:9.1f}{median(r['fwd']) * 1e3:8.1f}"
            f"{median(r['bwd']) * 1e3:9.1f}{median(r['accum']) * 1e3:7.1f}"
            f"{opt * 1e3:8.1f}{opt / tot * 100:7.1f}%{tps:9.0f}{mfu:8.1%}"
        )
    print(f"\n峰值内存: {mx.get_peak_memory() / 2**30:.2f} GB")
    print("注：fwd/bwd/acc/opt 各自 mx.eval 分段，合计 ≈ total（差值为 Python 调度）")


if __name__ == "__main__":
    main()
