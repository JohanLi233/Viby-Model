"""注意力段耗时分解：投影 GEMM / SDPA / mask 形式。

把 GQA 一层拆成「投影 GEMM」与「SDPA」，并对比 mask 的三种形式（causal
字符串 / bool 数组 / 加性 bf16 数组），看 --doc_mask 的段掩码是否把 SDPA
打到慢路径。

用法: uv run experiments/probe_attn.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_real_module(B, T, H, hd, rope, D):
    """真实 GQAAttention 模块：整体 vs 把 SDPA 换成直通，差值即注意力打分部分。

    合成的投影链路和真实模块对不上（真实模块还有 QK-norm、partial RoPE、
    ShortConv、XSA、attn gate、两次 transpose/reshape），所以这里直接替换
    真身里的 SDPA，剩下的完全一致，相减才是干净的 SDPA 成本。
    """
    from model.attention import GQAAttention as Attention
    from model.config import VibyConfig

    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=2,
        num_attention_heads=H,
        vocab_size=6400,
        max_position_embeddings=T,
        use_attn_gate=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    attn = Attention(cfg, layer_idx=0)  # local 层：partial RoPE + 滑窗
    attn.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            attn.parameters(),
        )
    )
    mx.eval(attn.parameters())

    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    Cx = mx.random.normal((B, T, D)) * 0.5
    cos = mx.zeros((T, rope), dtype=mx.bfloat16)
    sin = mx.zeros((T, rope), dtype=mx.bfloat16)
    pos = (cos, sin, (mx.ones((rope // 2,)), 0, 1.0))
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(x, Cx, bias)

    real_sdpa = mx.fast.scaled_dot_product_attention

    def stub(q, k, v, scale=None, mask=None):
        # 直通：形状与 SDPA 输出一致，但不做任何 T×T 打分
        return v[..., : q.shape[-1]]

    def mk():
        def loss(x_, p):
            attn.update(p)
            o, _ = attn(x_, pos, causal_bias=bias, mask_is_full=False)
            return (o.astype(mx.float32) * Cx).sum()

        return loss

    loss = mk()
    p = attn.trainable_parameters()
    vg = mx.value_and_grad(loss, argnums=(0, 1))

    arms = {"完整模块": real_sdpa, "SDPA 直通": stub}
    samples = {k: {"f": [], "fb": []} for k in arms}
    for rnd in range(8):
        for name, fn in arms.items():
            mx.fast.scaled_dot_product_attention = fn
            t0 = time.perf_counter()
            mx.eval(loss(x, p))
            t1 = time.perf_counter()
            mx.eval(vg(x, p))
            t2 = time.perf_counter()
            if rnd >= 2:
                samples[name]["f"].append(t1 - t0)
                samples[name]["fb"].append(t2 - t1)
    mx.fast.scaled_dot_product_attention = real_sdpa

    import statistics

    med = {
        n: (statistics.median(s["f"]), statistics.median(s["fb"]))
        for n, s in samples.items()
    }
    print("\n真实 Attention 模块（doc_mask 数组 mask）")
    print(f"{'':<12}{'fwd':>9}{'bwd':>9}{'合计':>9}")
    for n, (f, b) in med.items():
        print(f"{n:<12}{f * 1e3:>9.2f}{b * 1e3:>9.2f}{(f + b) * 1e3:>9.2f}")
    df = med["完整模块"][0] - med["SDPA 直通"][0]
    db = med["完整模块"][1] - med["SDPA 直通"][1]
    tot = med["完整模块"][0] + med["完整模块"][1]
    print(
        f"{'⇒ SDPA 净额':<12}{df * 1e3:>9.2f}{db * 1e3:>9.2f}{(df + db) * 1e3:>9.2f}"
        f"   占模块 {(df + db) / tot * 100:.0f}%"
    )


def main():
    B = int(os.environ.get("VIBY_BENCH_B", 12))
    T = int(os.environ.get("VIBY_BENCH_T", 1024))
    H, hd, rope = 8, 96, 32
    qk = hd + rope
    D = 768
    print(f"B={B} T={T} H={H} head_dim={hd} qk_dim={qk} D={D}")

    q = (mx.random.normal((B, H, T, qk)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, qk)) * 0.3).astype(mx.bfloat16)
    v_pad = (mx.random.normal((B, H, T, qk)) * 0.3).astype(mx.bfloat16)
    v_raw = (mx.random.normal((B, H, T, hd)) * 0.3).astype(mx.bfloat16)
    scale = qk**-0.5
    Cq = mx.random.normal((B, H, T, hd)) * 0.5
    mx.eval(q, k, v_pad, v_raw, Cq)

    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    keep = (same & tril[None])[:, None]  # (B,1,T,T) bool
    bias = mx.where(keep, 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(keep, bias)

    # SDPA 的 FLOPs：QK^T + softmax·V，causal 只算下三角（≈一半）
    fl_full = 2 * B * H * T * T * (qk + qk) * 0.5

    def mk(mask, v):
        def loss(q_, k_, v_):
            o = mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale, mask=mask)
            return (o[..., :hd].astype(mx.float32) * Cq).sum()

        return loss

    cases = [
        ('"causal" 字符串', "causal", v_pad, "qk"),
        ('"causal" 字符串', "causal", v_raw, "hd"),
        ("bool 数组 (B,1,T,T)", keep, v_pad, "qk"),
        ("bool 数组 (B,1,T,T)", keep, v_raw, "hd"),
        ("加性 bf16 数组", bias, v_pad, "qk"),
        ("加性 bf16 数组", bias, v_raw, "hd"),
    ]
    # 机器在分钟尺度上漂移可达 ±30%，逐 case 连续计时不可比；改成轮转交替
    # 采样再取中位数，把漂移摊到所有 case 上。
    fns = []
    for label, mask, v, vw in cases:
        loss = mk(mask, v)
        fns.append((label, vw, v, loss, mx.value_and_grad(loss, argnums=(0, 1, 2))))
    samples = {i: {"f": [], "fb": []} for i in range(len(fns))}
    for rnd in range(9):
        for i, (_, _, v, loss, vg) in enumerate(fns):
            t0 = time.perf_counter()
            mx.eval(loss(q, k, v))
            t1 = time.perf_counter()
            mx.eval(vg(q, k, v))
            t2 = time.perf_counter()
            if rnd >= 2:
                samples[i]["f"].append(t1 - t0)
                samples[i]["fb"].append(t2 - t1)

    import statistics

    print(f"\n{'mask 形式':<22}{'V 宽':>6}{'fwd':>9}{'bwd':>9}{'fwd TFLOPS':>12}")
    for i, (label, vw, _, _, _) in enumerate(fns):
        f = statistics.median(samples[i]["f"])
        b = statistics.median(samples[i]["fb"])
        print(
            f"{label:<22}{vw:>6}{f * 1e3:>9.2f}{b * 1e3:>9.2f}"
            f"{fl_full / f / 1e12:>12.2f}"
        )

    bench_real_module(B, T, H, hd, hd // 2, D)  # rope 段 = partial RoPE 前半维

    # 投影 GEMM：qkv (D→H·qk+rank+rope)、kv_up (rank→2·H·hd)、o (H·hd→D)
    M = B * T
    rank = 192
    x = (mx.random.normal((M, D)) * 0.5).astype(mx.bfloat16)
    w_qkv = (mx.random.normal((D, H * qk + rank + rope)) * 0.02).astype(mx.bfloat16)
    w_kvup = (mx.random.normal((rank, 2 * H * hd)) * 0.02).astype(mx.bfloat16)
    w_o = (mx.random.normal((H * hd, D)) * 0.02).astype(mx.bfloat16)
    Cp = mx.random.normal((M, D)) * 0.5
    mx.eval(x, w_qkv, w_kvup, w_o, Cp)

    def proj(x_, a, b, c):
        qkv = x_ @ a
        kv = qkv[:, H * qk : H * qk + rank] @ b
        return ((qkv[:, : H * hd] + kv[:, : H * hd]) @ c).astype(mx.float32)

    def ploss(x_, a, b, c):
        return (proj(x_, a, b, c) * Cp).sum()

    pf = timed(lambda: proj(x, w_qkv, w_kvup, w_o))
    pvg = mx.value_and_grad(ploss, argnums=(0, 1, 2, 3))
    pfb = timed(lambda: pvg(x, w_qkv, w_kvup, w_o))
    fl_proj = 2 * M * (D * (H * qk + rank + rope) + rank * 2 * H * hd + H * hd * D)
    print(
        f"\n投影 GEMM（qkv+kv_up+o）  fwd {pf * 1e3:.2f}ms  fwd+bwd {pfb * 1e3:.2f}ms"
        f"  fwd {fl_proj / pf / 1e12:.2f} TFLOPS"
    )


if __name__ == "__main__":
    main()
