"""验证 prewarm_all 能让 compile 下的融合 kernel 保持启用。

对照三种顺序下整模型一步 fwd+bwd 的耗时与各 kernel 禁用标志：
  (a) 直接 compile（修复前的真实训练行为）
  (b) prewarm_all 后再 compile（修复后）
  (c) 先 eager 跑一次再 compile（上界参考）

用法: uv run python experiments/verify_prewarm.py
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASE = os.environ.get("VIBY_PREWARM_CASE")


def run(case):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    from model.config import VibyConfig
    from model.model import VibyForCausalLM

    B, T = 12, 1024
    cfg = VibyConfig(
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        vocab_size=6400,
        max_position_embeddings=T,
        mtp_depth=1,
        use_attn_gate=True,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=2,
        moe_intermediate_size=384,
        latent_dim=384,
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
    dtype = tree_flatten(model.parameters())[0][1].dtype

    X = mx.random.randint(0, 6400, (B, T))
    Y = mx.random.randint(0, 6400, (B, T))
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    mx.eval(X, Y, seg)
    p = model.trainable_parameters()

    from model.kernels.ce import cross_entropy

    def loss(p_, X_, Y_):
        model.update(p_)
        out = model(X_, segment_ids=seg)
        lg = out.logits if hasattr(out, "logits") else out[0]
        return cross_entropy(
            lg.astype(mx.float32).reshape(-1, 6400), Y_.reshape(-1)
        ).mean()

    vg = mx.value_and_grad(loss)

    if case == "prewarm":
        from model.kernels import prewarm_all

        prewarm_all(model, cfg, dtype, T, log=lambda m: print(f"    {m}"))
    elif case == "eager_first":
        mx.eval(vg(p, X, Y))

    cvg = mx.compile(vg)
    for _ in range(2):
        mx.eval(cvg(p, X, Y))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(cvg(p, X, Y))
        ts.append(time.perf_counter() - t0)

    import model.kda as kda
    from model.kernels import attn_res_fused, conv, kda_prep

    flags = {
        "scan": kda._SCAN_KERNEL_DISABLED,
        "conv": getattr(conv, "_DISABLED", "?"),
        "prep": kda_prep._DISABLED,
        "attn_res": getattr(attn_res_fused, "_DISABLED", "?"),
    }
    labels = {
        "plain": "(a) 直接 compile（修复前）",
        "prewarm": "(b) prewarm_all 后 compile（修复后）",
        "eager_first": "(c) 先 eager 再 compile（参考）",
    }
    print(
        f"  {labels[case]:<34} 整步 f+b {min(ts) * 1e3:>8.1f}ms   "
        + " ".join(f"{k}禁用={v}" for k, v in flags.items())
    )
    print(f"    峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")


if CASE:
    run(CASE)
else:
    print("整模型一步 fwd+bwd（bs12×1024, 1080M）\n")
    for c in ("plain", "prewarm", "eager_first"):
        subprocess.run(
            [sys.executable, __file__],
            env=dict(os.environ, VIBY_PREWARM_CASE=c),
            check=False,
        )
