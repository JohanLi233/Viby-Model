"""V4.1 编译口径的分组桩测：把某一类算子整体换成常数/恒等，量它对整步的贡献。

`ab_v41_components.py` 只能按"结构开关"消融（层数/专家数/组件有无），
剩下的"每层零碎算子"（RoPE、RMSNorm、mask、indexer、compressor、MoE 侧信道）
它测不到——而它们加起来可能就是大头。这个脚本用 monkeypatch 把整类算子桩掉，
形状保持不变，所以计时仍然可比（数值当然不对，只看墙钟）。

用法：
    .venv/bin/python experiments/prof_v41_stub.py --batch 4 --seq 1024
    .venv/bin/python experiments/prof_v41_stub.py --only base,no_rope
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402

import model.attention as attn_mod  # noqa: E402
import model.moe as moe_mod  # noqa: E402
import model.norms as norm_mod  # noqa: E402
import model.rope as rope_mod  # noqa: E402
from model.config import VibyConfig  # noqa: E402
from model.model import VibyForCausalLM  # noqa: E402
from trainer.base_trainer import BaseTrainer  # noqa: E402
from trainer.config import get_pretrain_parser, setup_training_args  # noqa: E402
from trainer.flops import training_flops_per_token  # noqa: E402
from trainer.utils import build_model_kwargs, resolve_compute_scaled_hparams  # noqa: E402


def _bench(fn, iters=2, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def _patch(stubs):
    """按名字返回 (恢复函数, 打桩函数)。"""
    saved = []

    def save(mod, name, new):
        old = getattr(mod, name)
        setattr(mod, name, new)
        saved.append((mod, name, old))

    if "no_rope" in stubs:
        ident = lambda x, cos, sin, rope_dim, inverse=False: x  # noqa: E731
        save(attn_mod, "rope_partial", ident)
        save(rope_mod, "rope_partial", ident)
        save(rope_mod, "apply_rope", lambda x, cos, sin, inverse=False: x)
    if "no_norm" in stubs:
        save(norm_mod.RMSNorm, "__call__", lambda self, x: x)
        save(norm_mod, "rms_unit", lambda x, eps=1e-6: x)
    if "no_masks" in stubs:
        save(attn_mod, "_group_doc_mask", lambda ratio, seg: True)
        save(attn_mod, "_group_pad_mask", lambda ratio, pad: True)
    if "no_indexer" in stubs:
        def _scores(self, x, qr, index_k, token_pos, reach, cos_all, sin_all):
            B, T, _ = x.shape
            return mx.where(reach, mx.zeros((B, T, index_k.shape[1]), dtype=mx.float32), attn_mod.NEG_INF)

        save(attn_mod.Indexer, "scores", _scores)
    if "no_compressor" in stubs:
        def _comp(self, x, start_pos, state=None):
            B, T, _ = x.shape
            n = max(1, (start_pos + T) // self.ratio)
            lat = mx.zeros((B, n, self.dim_out), dtype=x.dtype) if hasattr(self, "dim_out") else None
            if lat is None:
                lat = mx.zeros((B, n, self.wkv.weight.shape[0]), dtype=x.dtype)
            base = start_pos // self.ratio
            return lat, None, (base + mx.arange(n))[None, :] * self.ratio, base

        save(attn_mod.Compressor, "__call__", _comp)
    if "no_topk" in stubs:
        save(attn_mod, "_topk_masks",
             lambda score, reach, k, offset, need_idx=True: (mx.ones_like(score, dtype=mx.bool_), None))
    if "no_router" in stubs:
        save(moe_mod.MoEGate, "scores",
             lambda self, x: mx.zeros((x.shape[0], self.n_routed), dtype=mx.float32))
    if "no_router_topk" in stubs:
        def _gate_call(self, x):
            M = x.shape[0]
            sc = mx.zeros((M, self.n_routed), dtype=mx.float32)
            idx = mx.broadcast_to(mx.arange(self.top_k, dtype=mx.int32)[None, :], (M, self.top_k))
            w = mx.full((M, self.top_k), 1.0 / self.top_k, dtype=mx.float32)
            self._last_load = mx.zeros((self.n_routed,), dtype=mx.float32)
            return w, idx, sc
        save(moe_mod.MoEGate, "__call__", _gate_call)
    if "no_moe" in stubs:
        save(moe_mod.MoEFeedForward, "__call__", lambda self, x: self.shared(x))
    if "no_attn" in stubs:
        save(attn_mod.Attention, "__call__",
             lambda self, x, start_pos, shared, cache=None, segment_ids=None, pad_mask=None: mx.zeros_like(x))
    if "no_aux" in stubs:
        save(moe_mod.MoEFeedForward, "seq_aux_loss",
             lambda self, scores, idx, B, T: mx.array(0.0))
    if "no_load" in stubs:
        old_call = moe_mod.MoEGate.__call__

        def _call(self, x):
            w, idx, scores = old_call(self, x)
            self._last_load = mx.zeros((self.n_routed,), dtype=mx.float32)
            return w, idx, scores

        save(moe_mod.MoEGate, "__call__", _call)
    return saved


def _restore(saved):
    for mod, name, old in reversed(saved):
        setattr(mod, name, old)


STUBS = ["base", "no_topk", "no_compressor", "no_router", "no_router_topk", "no_sdpa", "no_moe", "no_attn", "no_moe+no_attn", "no_rope", "no_norm",
         "no_masks", "no_indexer+no_compressor", "no_aux+no_load"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    wanted = [s for s in args.only.split(",") if s]

    cli = ["--out_dir", "research_runs/_bench", "--no_save",
           "--batch_size", str(args.batch), "--accumulation_steps", "2",
           "--max_seq_len", str(args.seq), "--cache_limit_gb", "8"]
    targs = setup_training_args(get_pretrain_parser().parse_args(cli), "pretrain")
    cfg = VibyConfig(**build_model_kwargs(targs))
    targs = resolve_compute_scaled_hparams(targs, 467617)
    B, T = args.batch, args.seq

    X = mx.random.randint(0, cfg.vocab_size, (B, T))
    Y = mx.random.randint(0, cfg.vocab_size, (B, T))
    loss_mask = mx.ones((B, T), dtype=mx.float32)
    attn_mask = mx.ones((B, T), dtype=mx.int32)
    cuts = (mx.random.uniform(shape=(B, T)) < 1 / 200).astype(mx.int32)
    cuts = cuts.at[:, 0].add(1 - cuts[:, 0])
    seg = mx.cumsum(cuts, axis=1).astype(mx.int32)
    mx.eval(X, Y, loss_mask, attn_mask, seg)

    base_t = None
    for name in STUBS:
        if wanted and name not in wanted:
            continue
        stubs = [] if name == "base" else name.split("+")
        saved = _patch(stubs)
        try:
            from trainer.utils import convert_model_dtype

            model = VibyForCausalLM(cfg)
            convert_model_dtype(model, getattr(targs, 'dtype', ''))
            trainer = BaseTrainer(targs, model, None, cfg, "pretrain")

            def step():
                out, grads = trainer._compute_loss_and_grad(X, Y, loss_mask, attn_mask, seg)
                mx.eval(*[o for o in out if o is not None])
                mx.eval(grads)

            mx.reset_peak_memory()
            mn, med = _bench(step, args.iters, 1)
            tps = B * T / mn
            if name == "base":
                base_t = mn
                delta = ""
            else:
                delta = " Δ%+.1f%%" % (100 * (mn - base_t) / base_t)
            print("%-58s %6.0f tok/s  min %.3fs med %.3fs%s  峰值 %.1fGB"
                  % (name, tps, mn, med, delta, mx.get_peak_memory() / 1e9), flush=True)
            del trainer, model
            gc.collect()
            mx.clear_cache()
        except Exception as exc:  # noqa: BLE001
            print("%-58s FAILED %s: %s" % (name, type(exc).__name__, exc), flush=True)
        finally:
            _restore(saved)


if __name__ == "__main__":
    main()
