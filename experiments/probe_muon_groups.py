"""BatchedMuon 逐组逐阶段耗时归属（真实 1B 配置）。

复制 apply_gradients 的分组逻辑（不改生产代码），对每个形状组分别测
stack / mom / ns5(标准) / ns5_gram / apply / scatter，找出稳态步的时间
去向。重点看 2D 组：它们每步都跑完整 NS5（堆叠专家组已降频到 1/8）。

用法: uv run python experiments/probe_muon_groups.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import BatchedMuon, _stack_apply_kernel, _stack_mom_kernel


def timed(fn, it=5, w=2):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    cfg = VibyConfig(
        hidden_size=768,
        num_hidden_layers=int(os.environ.get("VIBY_BENCH_L", 8)),
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=1024,
        mtp_depth=1,
        use_attn_gate=True,
        n_routed_experts=int(os.environ.get("VIBY_BENCH_E", 256)),
        num_experts_per_tok=8,
        n_shared_experts=2,
        moe_intermediate_size=int(os.environ.get("VIBY_BENCH_I", 384)),
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
    flat_p = dict(tree_flatten(params))
    n_all = sum(v.size for v in flat_p.values())
    print(f"可训练参数 {n_all / 1e6:.1f}M")

    # 只保留会进 Muon 组的张量（ndim>=2、非 embedding/router，与
    # create_mixed_optimizer 的 filter 一致性无关——这里只是量化 NS 成本）
    flat_g = {
        k: (mx.random.normal(v.shape) * 0.01).astype(v.dtype)
        for k, v in flat_p.items()
        if v.ndim >= 2
    }
    flat_p = {k: v for k, v in flat_p.items() if v.ndim >= 2}
    state_v = {k: mx.zeros_like(v) for k, v in flat_p.items()}
    mx.eval(flat_g, state_v)

    groups: dict = {}
    stack_groups: dict = {}
    for path, g in flat_g.items():
        if g.ndim > 2:
            b, r = g.shape[0], g.shape[1]
            stack_groups.setdefault((b, r, g.size // (b * r)), []).append(path)
        else:
            groups.setdefault(g.shape, []).append(path)

    opt = BatchedMuon(learning_rate=mx.array(0.01), momentum=0.95, ns_steps=5)
    mom_fn = _stack_mom_kernel(0.95, True, 0.0)
    apply_fn = _stack_apply_kernel(opt.hyperball)
    lr = mx.array(0.001, dtype=mx.bfloat16)

    hdr = (
        f"{'形状':<20}{'张量':>5}{'参数M':>8}{'stack':>8}{'mom':>7}"
        f"{'ns5':>8}{'gram':>8}{'apply':>8}{'scat':>7}{'合计':>8}"
    )
    tot = {k: 0.0 for k in ("stack", "mom", "ns5", "gram", "apply", "scat")}

    def measure(label, paths, r, c, is_stack, b=1):
        def do_stack():
            return (mx.stack([flat_g[p].reshape(-1, r, c).squeeze() for p in paths]),)

        if is_stack:

            def st():
                return (
                    mx.stack([flat_g[p] for p in paths]),
                    mx.stack([flat_p[p] for p in paths]),
                    mx.stack([state_v[p] for p in paths]),
                )
        else:

            def st():
                return (
                    mx.stack([flat_g[p].reshape(r, c) for p in paths]),
                    mx.stack([flat_p[p].reshape(r, c) for p in paths]),
                    mx.stack([state_v[p].reshape(r, c) for p in paths]),
                )

        t_st = timed(st)
        G, P, V = st()
        mx.eval(G, P, V)
        t_mom = timed(lambda: mom_fn(G, P, V))  # noqa: F821
        U, Vn = mom_fn(G, P, V)
        mx.eval(U, Vn)
        N = U.shape[0]
        Uf = U.reshape(N * b, r, c) if is_stack else U
        t_ns = timed(lambda: opt._ns5(Uf))  # noqa: F821
        t_gr = timed(lambda: opt._ns5_gram(Uf))  # noqa: F821
        X = opt._ns5(Uf)
        X = X.reshape(U.shape) if is_stack else X
        mx.eval(X)
        t_ap = timed(lambda: apply_fn(P, X, lr))  # noqa: F821
        NP = apply_fn(P, X, lr)
        mx.eval(NP)
        t_sc = timed(lambda: [(Vn[i], NP[i].astype(mx.bfloat16)) for i in range(N)])  # noqa: F821
        n = sum(flat_g[p].size for p in paths)
        s = t_st + t_mom + min(t_ns, t_gr) + t_ap + t_sc
        for k, v in (
            ("stack", t_st),
            ("mom", t_mom),
            ("ns5", min(t_ns, t_gr)),
            ("apply", t_ap),
            ("scat", t_sc),
        ):
            tot[k] += v
        print(
            f"{label:<20}{len(paths):>5}{n / 1e6:>8.1f}{t_st:>8.2f}{t_mom:>7.2f}"
            f"{t_ns:>8.2f}{t_gr:>8.2f}{t_ap:>8.2f}{t_sc:>7.2f}{s:>8.2f}"
        )
        del G, P, V, U, Vn, X, NP
        mx.clear_cache()

    print(f"\n=== 2D 组（每步都跑完整 NS5，不降频）===\n{hdr}")
    for shp, paths in sorted(
        groups.items(), key=lambda kv: -sum(flat_g[p].size for p in kv[1])
    ):
        measure(f"{shp[0]}x{shp[1]}", paths, shp[0], shp[1], False)
    print(f"\n=== 堆叠专家组（每步全量 Gram-NS，不降频）===\n{hdr}")
    for (b, r, c), paths in sorted(
        stack_groups.items(), key=lambda kv: -sum(flat_g[p].size for p in kv[1])
    ):
        measure(f"{b}x{r}x{c}", paths, r, c, True, b)

    print("\n各阶段合计: " + "  ".join(f"{k}={v:.1f}ms" for k, v in tot.items()))
    print(
        f"总计 {sum(tot.values()):.1f}ms   峰值内存 "
        f"{mx.get_peak_memory() / 2**30:.2f} GB"
    )


if __name__ == "__main__":
    main()
