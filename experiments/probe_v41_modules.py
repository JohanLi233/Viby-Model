"""逐组件前向计时：把一次 V4.1 前向按模块切开，用 mx.eval 封口测墙钟。

`bench_train_step.py` 只能给整步；这个 probe 回答"墙钟花在哪个模块"，
对应 MLX_PERF.md §1 SOP 的②归因。注意 mx.eval 会切断图融合，绝对值和
训练整步不可比，只看**相对占比**。

用法：
    .venv/bin/python experiments/probe_v41_modules.py --batch 4 --seq 1024
    .venv/bin/python experiments/probe_v41_modules.py --cfg hc_mult=2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx  # noqa: E402

from model.cache import SharedAttnState  # noqa: E402
from model.config import VibyConfig  # noqa: E402
from model.hc import hc_post, hc_pre, identity_pre_mix  # noqa: E402
from model.model import VibyForCausalLM, lm_head_ce  # noqa: E402


def parse_cfg(pairs):
    kw = {}
    for item in pairs or []:
        key, _, value = item.partition("=")
        if value == "":
            kw[key] = ()
        elif value.lower() in ("true", "false"):
            kw[key] = value.lower() == "true"
        else:
            try:
                kw[key] = int(value)
            except ValueError:
                kw[key] = float(value)
    return kw


class Timer:
    def __init__(self):
        self.acc = {}
        self.order = []

    def add(self, name, dt):
        if name not in self.acc:
            self.acc[name] = 0.0
            self.order.append(name)
        self.acc[name] += dt

    def report(self, total_s):
        print("\n%-24s %8s %8s" % ("模块", "ms", "占比"))
        for name in self.order:
            ms = self.acc[name] * 1000
            print("%-24s %8.2f %7.1f%%" % (name, ms, 100 * self.acc[name] / total_s))
        print("%-24s %8.2f" % ("合计", total_s * 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    kwargs = {"preset": None if args.preset == "base" else "tiny", "n_mtp_layers": 0}
    kwargs.update(parse_cfg(args.cfg))
    cfg = VibyConfig(**kwargs)
    model = VibyForCausalLM(cfg)
    m = model.model
    B, T = args.batch, args.seq
    ids = mx.random.randint(0, cfg.vocab_size, (B, T))
    labels = mx.random.randint(0, cfg.vocab_size, (B, T))
    loss_mask = mx.ones((B, T), dtype=mx.float32)

    timer = Timer()

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        timer.add(name, time.perf_counter() - t0)
        return out

    # warmup（含 Metal JIT）
    model(ids, labels=labels, loss_mask=loss_mask, use_mtp=False)
    mx.eval(model.parameters())

    t_all = time.perf_counter()
    for _ in range(args.repeat):
        h = timed(
            "embed+repeat",
            lambda: mx.repeat(m.embed(ids)[:, :, None, :], m.hc_mult, axis=2),
        )
        hashes = None
        if m.engram_hash is not None:
            hashes, _ = timed("engram_hash", lambda: m.engram_hash(ids, None, None))
        shared = SharedAttnState()
        pre_mix = identity_pre_mix(h, m.hc_mult)
        for i, layer in enumerate(m.layers):
            if hashes is not None and i in m._engram_slot:
                slot = m._engram_slot[i]
                eng = m.engram_layers[slot]
                h = timed(
                    "engram[%d]" % i,
                    lambda eng=eng, slot=slot: eng(h, hashes[:, :, slot, :]),
                )

            def sub(name, fn, h):
                return timed(name, fn)

            residual = h
            attn_pre, attn_post, attn_comb = timed(
                "hc.mixes(attn)", lambda: layer.attn_hc.mixes(h)
            )
            hh = timed("hc_pre(attn)", lambda: hc_pre(h, pre_mix))
            hh = timed("attn_norm", lambda: layer.attn_norm(hh))
            hh = timed(
                "attn[%s]" % cfg.layer_mode(i),
                lambda: layer.attn(hh, 0, shared, None, None, None),
            )
            h = timed(
                "hc_post(attn)", lambda: hc_post(hh, residual, attn_post, attn_comb)
            )

            residual = h
            ffn_pre, ffn_post, ffn_comb = timed(
                "hc.mixes(ffn)", lambda: layer.ffn_hc.mixes(h)
            )
            hh = timed("hc_pre(ffn)", lambda: hc_pre(h, attn_pre))
            hh = timed("ffn_norm", lambda: layer.ffn_norm(hh))
            hh = timed("moe", lambda: layer.ffn(hh))
            h = timed("hc_post(ffn)", lambda: hc_post(hh, residual, ffn_post, ffn_comb))
            pre_mix = ffn_pre
        h = timed("hc_pre(final)", lambda: hc_pre(h, pre_mix))
        h = timed("final_norm", lambda: m.norm(h))
        timed(
            "lm_head_ce",
            lambda: lm_head_ce(h, model._head_weight(), labels, loss_mask, 0.0),
        )
    total = time.perf_counter() - t_all
    print(
        "配置 dim=%d layers=%d hc=%d B=%d T=%d repeat=%d"
        % (cfg.dim, cfg.n_layers, cfg.hc_mult, B, T, args.repeat)
    )
    timer.report(total)


if __name__ == "__main__":
    main()
