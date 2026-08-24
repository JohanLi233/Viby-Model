"""BatchedMuon 段耗时分解：形状分组 / Newton-Schulz / 动量与回写。

optimizer 里 Muon 组只有 ~20M 参数却要 36ms（同期 AdamW 621M 参数只要
30ms），需要看是 NS 的 GEMM 本身慢，还是分组太碎导致 batch 维太小、
kernel 发射与堆叠拷贝占主导。

用法: uv run experiments/probe_muon.py
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
from trainer.muon import BatchedMuon


def timed(fn, it=8, w=3):
    for _ in range(w):
        fn()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    cfg = VibyConfig(
        hidden_size=768,
        num_hidden_layers=int(os.environ.get("VIBY_BENCH_L", 8)),
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=1024,
        mtp_depth=1,
        use_attn_gate=True,
        n_routed_experts=int(os.environ.get("VIBY_BENCH_E", 288)),
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=int(os.environ.get("VIBY_BENCH_I", 104)),
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

    def is_muon(path, arr):
        return (
            arr.ndim >= 2
            and "embed" not in path
            and "lm_head" not in path
            and ".experts." not in path
            and ".router." not in path
        )

    flat = [(p, v) for p, v in tree_flatten(model.trainable_parameters())]
    items = [(p, v) for p, v in flat if is_muon(p, v)]
    n = sum(v.size for _, v in items)
    print(f"Muon 组：{len(items)} 个张量，{n / 1e6:.2f}M 参数")

    groups: dict = {}
    for p, v in items:
        r = v.shape[0]
        c = v.size // r
        groups.setdefault((r, c), []).append(p)
    print(f"\n{'(r, c)':>16}{'张量数':>8}{'NS(ms)':>9}{'GFLOP':>9}{'TFLOPS':>9}")
    total_ns = 0.0
    opt = BatchedMuon(learning_rate=mx.array(0.01), ns_steps=5)
    for (r, c), paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        N = len(paths)
        X = mx.random.normal((N, r, c)).astype(mx.bfloat16)
        mx.eval(X)
        fn = lambda: mx.eval(opt._ns5(X))  # noqa: E731
        rows = [(r, c)]
        t = timed(fn)
        total_ns += t
        # NS5：每步 X@Xᵀ、A@A、B@X，短边 m=min(r,c)
        fl = 0.0
        for rr, cc in rows:
            m, k = min(rr, cc), max(rr, cc)
            fl += N * 5 * 2 * (m * m * k + m * m * m + m * m * k)
        print(
            f"{str((r, c)):>16}{N:>8}{t * 1e3:>9.2f}"
            f"{fl / 1e9:>9.2f}{fl / t / 1e12:>9.2f}"
        )
    print(f"\nNS 合计 {total_ns * 1e3:.1f}ms")

    grads = {p: mx.ones_like(v) * 0.01 for p, v in items}
    params = {p: v for p, v in items}
    mx.eval(list(grads.values()))
    state = {"p": params}

    def full_step():
        state["p"] = opt.apply_gradients(grads, state["p"])
        mx.eval(state["p"], opt.state)

    full_step()
    t_full = timed(full_step)
    print(
        f"apply_gradients 全流程 {t_full * 1e3:.1f}ms"
        f"（NS 之外 {t_full * 1e3 - total_ns * 1e3:.1f}ms 是堆叠/动量/回写）"
    )


if __name__ == "__main__":
    main()
