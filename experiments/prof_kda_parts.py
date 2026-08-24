"""KDA 层内部逐项计时：投影 / conv / 门 / chunk 分解 / scan / 输出段。

用法: .venv/bin/python experiments/prof_kda_parts.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.config import VibyConfig
from model.kda import KDAAttention, _chunk_kda

B, T, D, H = 12, 1024, 768, 8


def timed(fn, iters=8, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=8,
        num_attention_heads=H,
        vocab_size=6400,
        max_position_embeddings=T,
        n_routed_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=2,
        moe_intermediate_size=320,
    )
    kda = KDAAttention(cfg, layer_idx=0)
    HD = cfg.head_dim
    print(f"head_dim={HD}")

    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )

    t_proj = timed(lambda: (kda.q_proj(x), kda.k_proj(x), kda.v_proj(x)))
    q_in = kda.q_proj(x)
    mx.eval(q_in)
    t_conv = timed(
        lambda: (
            kda.q_conv(q_in, segment_ids=seg),
            kda.k_conv(q_in, segment_ids=seg),
            kda.v_conv(q_in, segment_ids=seg),
        )
    )
    t_gates = timed(
        lambda: (
            kda.f_b_proj(kda.f_a_proj(x)),
            kda.b_proj(x),
            kda.g_b_proj(kda.g_a_proj(x)),
        )
    )

    # 构造 chunk 输入（f32, (B,H,T,D)）
    q = mx.random.normal((B, H, T, HD)) * 0.1
    k = mx.random.normal((B, H, T, HD)) * 0.1
    v = mx.random.normal((B, H, T, HD)) * 0.5
    log_g = -mx.random.uniform(0.001, 1.0, (B, H, T, HD))
    beta = mx.random.uniform(0, 1, (B, H, T))
    q, k, v, log_g, beta = [a.astype(mx.float32) for a in (q, k, v, log_g, beta)]
    mx.eval(q, k, v, log_g, beta)

    t_chunk = timed(lambda: _chunk_kda(q, k, v, log_g, beta))

    # 分解 _chunk_kda 内部：准备段（cumsum/GEMM 建矩阵）vs scan 段
    import math

    import mlx.core as mxx
    from model.kda import KDA_CHUNK, _kda_scan

    def prep():
        C = KDA_CHUNK
        NC = T // C

        def ch(a):
            return a.reshape(B, H, NC, C, *a.shape[3:])

        qc, kc, vc, lgc, bc = ch(q), ch(k), ch(v), ch(log_g), ch(beta)
        gc = mxx.cumsum(lgc, axis=-2)
        eg = mxx.exp(gc)
        qe = qc * eg
        ke = kc * eg
        ki = kc * mxx.exp(-gc)
        sl = mxx.tril(mxx.ones((C, C), dtype=mxx.bool_), k=-1)
        Lm = (bc[..., None] * ke) @ mxx.swapaxes(ki, -1, -2)
        Lm = mxx.where(sl, Lm, mxx.zeros_like(Lm))
        P = -Lm
        X = mxx.eye(C, dtype=mxx.float32) + P
        p = P
        for _ in range(int(math.log2(C)) - 1):
            p = p @ p
            X = X + p @ X
        Afb = X * bc[..., None, :]
        w = Afb @ ke
        u = Afb @ vc
        lower = mxx.tril(mxx.ones((C, C), dtype=mxx.bool_))
        Aqk = qe @ mxx.swapaxes(ki, -1, -2)
        Aqk = mxx.where(lower, Aqk, mxx.zeros_like(Aqk))
        gl = gc[:, :, :, -1, :]
        kd = kc * mxx.exp(gl[:, :, :, None, :] - gc)
        return qe, w, u, Aqk, kd, mxx.exp(gl)

    t_prep = timed(prep)
    qe, w, u, Aqk, kd, egl = prep()
    S0 = mxx.zeros((B, H, HD, HD), dtype=mxx.float32)
    mx.eval(qe, w, u, Aqk, kd, egl, S0)
    t_scan = timed(lambda: _kda_scan(qe, w, u, Aqk, kd, egl, S0))

    # 输出段
    out = mx.random.normal((B, T, H, HD)).astype(mx.bfloat16)
    t_out = timed(
        lambda: kda.o_proj(
            (
                kda.o_norm(out)
                * mx.sigmoid(kda.g_b_proj(kda.g_a_proj(x))).reshape(B, T, -1)
                if False
                else kda.o_norm(out).reshape(B, T, -1)
            )
        )
    )

    t_full = timed(lambda: kda(x, segment_ids=seg)[0])

    print(f"qkv 投影      {t_proj:7.2f}ms")
    print(f"3×conv+silu   {t_conv:7.2f}ms")
    print(f"门投影(3低秩) {t_gates:7.2f}ms")
    print(f"chunk 分解段  {t_prep:7.2f}ms")
    print(f"scan 段(NC64) {t_scan:7.2f}ms")
    print(f"chunk 合计    {t_chunk:7.2f}ms")
    print(f"输出 norm+proj{t_out:7.2f}ms")
    print(f"整层前向      {t_full:7.2f}ms")


if __name__ == "__main__":
    main()
