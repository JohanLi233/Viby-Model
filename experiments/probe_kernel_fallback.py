"""检查 mx.compile 下手写 Metal kernel 是否被静默禁用。

各 kernel 的首次在线校验都要 .item() 对照 eager 参考。若首次调用发生在
mx.compile 的 trace 内，host sync 抛异常 → except 分支把模块级 _DISABLED
置真 → 该 kernel 此后永久走 eager。base_trainer 只 prewarm 了
attn_res_fused，conv / kda_scan / kda_prep 都没有 prewarm 调用点。

本脚本分别在「先 eager 跑一次」与「直接 compile」两种顺序下构建 KDA 层，
打印各模块的禁用标志与层耗时。

用法: uv run python experiments/probe_kernel_fallback.py
"""

import importlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = os.environ.get("VIBY_FALLBACK_CHILD")


def run_case(compile_first: bool):
    """在子进程里跑一种顺序（模块级 _DISABLED 是进程内全局状态）。"""
    import mlx.core as mx
    from mlx.utils import tree_map

    from model.config import VibyConfig
    from model.kda import KDAAttention

    B, T, D = 12, 1024, 768
    cfg = VibyConfig(
        hidden_size=D,
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
    m = KDAAttention(cfg, layer_idx=0)
    m.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            m.parameters(),
        )
    )
    mx.eval(m.parameters())
    m.train()
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    cot = mx.random.normal((B, T, D))
    mx.eval(x, seg, cot)
    p = m.trainable_parameters()

    def loss(x_, p_):
        m.update(p_)
        return (m(x_, segment_ids=seg)[0].astype(mx.float32) * cot).sum()

    vg = mx.value_and_grad(loss, argnums=(0, 1))
    if not compile_first:
        mx.eval(vg(x, p))  # 先在 eager 下触发在线校验
    cvg = mx.compile(vg)
    for _ in range(3):
        mx.eval(cvg(x, p))
    ts = []
    for _ in range(6):
        t0 = time.perf_counter()
        mx.eval(cvg(x, p))
        ts.append(time.perf_counter() - t0)

    import model.kda as kda
    from model.kernels import conv, kda_prep

    flags = {
        "kda_scan": kda._SCAN_KERNEL_DISABLED,
        "causal_conv": getattr(conv, "_DISABLED", "n/a"),
        "kda_prep": kda_prep._DISABLED,
    }
    label = "直接 compile（模拟真实训练）" if compile_first else "先 eager 再 compile"
    print(
        f"{label:<28} f+b {min(ts) * 1e3:>7.2f}ms   "
        + "  ".join(f"{k} 禁用={v}" for k, v in flags.items())
    )


if CHILD:
    run_case(CHILD == "1")
else:
    print("KDA 层 fwd+bwd（compile 口径）与各 kernel 的禁用标志\n")
    for v in ("1", "0"):
        env = dict(os.environ, VIBY_FALLBACK_CHILD=v)
        subprocess.run([sys.executable, __file__], env=env, check=False)
    importlib.invalidate_caches()
