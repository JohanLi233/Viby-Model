"""V4.1 缩放版的自检 + 性能基准。

用法：
    python experiments/bench_v41.py --preset tiny --batch 4 --seq 256
    python experiments/bench_v41.py --preset base --batch 1 --seq 512 --grad

输出：参数/激活量、逐层模式表、fwd / fwd+bwd 墙钟与吞吐、峰值内存，
以及 prefill ↔ 逐 token 解码的一致性（max|Δlogit|）。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402
from mlx import nn  # noqa: E402

from model.config import VibyConfig  # noqa: E402
from model.model import VibyForCausalLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["tiny", "base"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--layers", type=int, default=0, help="覆盖层数（0 = 用预设）")
    ap.add_argument(
        "--experts", type=int, default=0, help="覆盖路由专家数（0 = 用预设）"
    )
    ap.add_argument("--grad", action="store_true", help="额外测 fwd+bwd")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument(
        "--parity", action="store_true", help="额外跑 prefill/decode 一致性"
    )
    args = ap.parse_args()

    kw = {"preset": None if args.preset == "base" else "tiny"}
    if args.layers:
        kw["n_layers"] = args.layers
        # 层数变了要重推压缩率与源层（交给 VibyConfig 的默认推导）
        kw.pop("compress_ratios", None)
    if args.experts:
        kw["n_routed_experts"] = args.experts
        kw["n_activated_experts"] = min(6, args.experts)
    cfg = VibyConfig(**kw)
    print(
        "配置：dim=%d layers=%d (编码 %d) heads=%d×%d window=%d hc=%d experts=%d/%d"
        % (
            cfg.dim,
            cfg.n_layers,
            cfg.n_encoder_layers,
            cfg.n_heads,
            cfg.head_dim,
            cfg.window_size,
            cfg.hc_mult,
            cfg.n_activated_experts,
            cfg.n_routed_experts,
        )
    )
    modes = {}
    for i in range(cfg.n_layers):
        modes[cfg.layer_mode(i)] = modes.get(cfg.layer_mode(i), 0) + 1
    print("逐层模式：", modes, " compress_ratios:", list(cfg.compress_ratios))
    print(
        "kv 源:",
        list(cfg.kv_source_layers),
        " indexer 源:",
        list(cfg.index_source_layers),
        " 候选层:",
        cfg.candidate_source_layer,
    )

    t0 = time.time()
    model = VibyForCausalLM(cfg)
    print(
        "构建 %.1fs  总参 %.1fM  激活 %.1fM  Engram 表 %.1fM"
        % (
            time.time() - t0,
            model.num_parameters() / 1e6,
            cfg.num_active_parameters() / 1e6,
            model.ngram_lookup_parameters() / 1e6,
        )
    )

    B, T = args.batch, args.seq
    ids = mx.random.randint(0, cfg.vocab_size, (B, T))
    lab = mx.random.randint(0, cfg.vocab_size, (B, T))
    mask = mx.ones((B, T))
    use_mtp = not args.no_mtp

    mx.reset_peak_memory()
    t0 = time.time()
    out = model(ids, labels=lab, loss_mask=mask, use_mtp=use_mtp)
    mx.eval(out.loss)
    dt = time.time() - t0
    print(
        "fwd   B=%d T=%d  %.3fs  %.0f tok/s  peak %.2f GB  loss %.3f (lm %.3f mtp %.3f)"
        % (
            B,
            T,
            dt,
            B * T / dt,
            mx.get_peak_memory() / 1e9,
            out.loss.item(),
            out.lm_loss.item(),
            (out.mtp_loss or mx.array(0.0)).item(),
        )
    )
    if out.moe_loads is not None:
        print(
            "      MoE 负载：每层计数和 %s（= B*T*top_k）"
            % mx.sum(out.moe_loads, axis=-1).tolist()[:4]
        )

    if args.grad:

        def loss_fn(m):
            return m(ids, labels=lab, loss_mask=mask, use_mtp=use_mtp).loss

        mx.reset_peak_memory()
        t0 = time.time()
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)
        dt = time.time() - t0
        print(
            "f+bwd B=%d T=%d  %.3fs  %.0f tok/s  peak %.2f GB"
            % (B, T, dt, B * T / dt, mx.get_peak_memory() / 1e9)
        )

    if args.parity:
        pref, cache = model.prefill(ids[:, :-1])
        step, _ = model.decode_step(ids[:, -1], cache)
        full, _ = model.prefill(ids)
        d = mx.max(mx.abs(full[:, -1] - step)).item()
        print(
            "prefill/decode 一致性 max|Δlogit| = %.3e（fp32 噪声级 ≈1e-6 为合格）" % d
        )


if __name__ == "__main__":
    main()
