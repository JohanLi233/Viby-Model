"""KDA 层内部 fwd+bwd 分项计时（反向杠杆定位）。

用法: .venv/bin/python experiments/prof_kda_bwd.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.config import VibyConfig
from model.kda import KDAAttention, _chunk_kda

B, T, D, H = 12, 1024, 768, 8


def timed(fn, iters=6, warm=2):
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
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)

    def fb_of(fn, *args):
        """对标量和的 value_and_grad 计时（全部入参求梯度）。"""
        vg = mx.value_and_grad(
            lambda *a: fn(*a).sum()
            if not isinstance(fn(*a), tuple)
            else fn(*a)[0].sum(),
            argnums=list(range(len(args))),
        )
        return timed(lambda: vg(*args))

    # qkv 投影 fwd+bwd
    def proj_sum(x_):
        return kda.q_proj(x_) + kda.k_proj(x_) + kda.v_proj(x_)

    t = fb_of(proj_sum, x)
    print(f"qkv 投影  fwd+bwd {t:7.2f}ms")

    # conv fwd+bwd（对 q_in）
    q_in = kda.q_proj(x)
    mx.eval(q_in)

    def conv_sum(qi):
        a, _ = kda.q_conv(qi, segment_ids=seg)
        b, _ = kda.k_conv(qi, segment_ids=seg)
        c, _ = kda.v_conv(qi, segment_ids=seg)
        return a + b + c

    t = fb_of(conv_sum, q_in)
    print(f"3×conv    fwd+bwd {t:7.2f}ms")

    # chunk fwd+bwd（f32 输入）
    q = (mx.random.normal((B, H, T, HD)) * 0.1).astype(mx.float32)
    k = (mx.random.normal((B, H, T, HD)) * 0.1).astype(mx.float32)
    v = (mx.random.normal((B, H, T, HD)) * 0.5).astype(mx.float32)
    log_g = -mx.random.uniform(0.001, 1.0, (B, H, T, HD))
    beta = mx.random.uniform(0, 1, (B, H, T))
    mx.eval(q, k, v, log_g, beta)

    def chunk_sum(q_, k_, v_, g_, b_):
        o, S = _chunk_kda(q_, k_, v_, g_, b_)
        return o.sum() + S.sum()

    vg = mx.value_and_grad(chunk_sum, argnums=[0, 1, 2, 3, 4])
    t = timed(lambda: vg(q, k, v, log_g, beta))
    print(f"chunk 合计 fwd+bwd {t:7.2f}ms")

    # 整层 fwd+bwd
    def layer_sum(x_):
        return kda(x_, segment_ids=seg)[0].sum()

    vg = mx.value_and_grad(layer_sum, argnums=0)
    t = timed(lambda: vg(x))
    print(f"整层     fwd+bwd {t:7.2f}ms")


if __name__ == "__main__":
    main()
