"""KDA 层内部分段归因（compile 口径，累积前缀差分）。

反向归因显示 KDA 占整步反向的 51%，但 chunk 段只占 KDA 的 ~1/3，其余在
投影/conv/门/transpose/输出。这里从 x 出发按真实计算链构造递增前缀，
每个前缀都编译后计时，相邻差分即为该段成本（避免用合成中间张量测量
带来的 dtype/连续性偏差）。

用法: uv run python experiments/prof_kda_stages.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_map

from model.config import VibyConfig
from model.kda import (
    KDA_CHUNK,
    KDA_G_MIN,
    KDAAttention,
    _chunk_kda,
    _repeat_heads,
    _rms_unit,
    kda_out_gate,
)
from model.kernels import prewarm_all

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = 768

cfg = VibyConfig(
    hidden_size=D,
    num_hidden_layers=8,
    num_attention_heads=8,
    vocab_size=6400,
    max_position_embeddings=T,
    mtp_depth=1,
    use_attn_gate=True,
    use_linear_attn=True,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=2,
    moe_intermediate_size=384,
    latent_dim=384,
)
main_kda = sum(
    not (((i + 1) % 4 == 0) or i == cfg.num_hidden_layers - 1)
    for i in range(cfg.num_hidden_layers)
)
mtp_kda = cfg.mtp_depth if cfg.num_hidden_layers > 1 else 0
kda_count = int(main_kda + mtp_kda)
m = KDAAttention(cfg, layer_idx=0)
m.update(
    tree_map(
        lambda a: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a,
        m.parameters(),
    )
)
mx.eval(m.parameters())
m.train()
# 必须在 mx.compile 之前预热：融合 kernel 首调用的在线校验含 host sync，
# 落在 compile trace 里会被吞掉并永久回退 eager（分段数据会整体失真）。
prewarm_all(m, cfg, mx.bfloat16, T, log=print)
H, Dh = m.n_heads, m.head_dim
Hv, n_rep = m.n_v_heads, m.n_rep_v

x = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
seg = mx.cumsum((mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1)
mx.eval(x, seg)
print(
    f"B={B} T={T} H={H} Hv={Hv} head_dim={Dh} chunk={KDA_CHUNK} NC={T // KDA_CHUNK}\n"
)


def timed(fn, it=6, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


# ---- 按真实 __call__ 顺序切成阶段函数，返回该阶段所有活跃张量 ----
def s1_proj(x_):
    w_in = mx.concatenate(
        (
            m.q_proj.weight,
            m.k_proj.weight,
            m.v_proj.weight,
            m.f_a_proj.weight,
            m.g_proj.weight,
            m.b_proj.weight,
        ),
        axis=0,
    )
    qk, vv = H * Dh, Hv * Dh
    return mx.split(
        x_ @ w_in.T,
        [qk, 2 * qk, 2 * qk + vv, 2 * qk + vv + Dh, 2 * qk + vv + Dh + vv],
        axis=-1,
    )


def s2_conv(x_):
    q_in, k_in, v_in, fa, ga, bl = s1_proj(x_)
    q, _ = m.q_conv(q_in, segment_ids=seg)
    k, _ = m.k_conv(k_in, segment_ids=seg)
    v, _ = m.v_conv(v_in, segment_ids=seg)
    return q, k, v, fa, ga, bl


def s3_gates(x_):
    q, k, v, fa, g, bl = s2_conv(x_)
    q = (m.scale**2) * _rms_unit(q.reshape(B, T, H, Dh))
    k = m.scale * _rms_unit(k.reshape(B, T, H, Dh))
    v = v.reshape(B, T, Hv, Dh)
    q = _repeat_heads(q, n_rep, 2)
    k = _repeat_heads(k, n_rep, 2)
    a = m.f_b_proj(fa).reshape(B, T, H, Dh)
    z = a.astype(mx.float32) + m.dt_bias.reshape(H, Dh)
    e_a = mx.broadcast_to(mx.exp(m.A_log)[:, None], (H, Dh))
    log_g = KDA_G_MIN * mx.sigmoid(e_a * z)
    log_g = _repeat_heads(log_g, n_rep, 2)
    beta = mx.sigmoid(bl.astype(mx.float32))
    beta = _repeat_heads(beta, n_rep, 2)
    return q, k, v, log_g, beta, g


def s4_transpose(x_):
    q, k, v, log_g, beta, ga = s3_gates(x_)
    return (
        q.astype(mx.float32).transpose(0, 2, 1, 3),
        k.astype(mx.float32).transpose(0, 2, 1, 3),
        v.astype(mx.float32).transpose(0, 2, 1, 3),
        log_g.transpose(0, 2, 1, 3),
        beta.transpose(0, 2, 1),
        ga,
    )


def s5_chunk(x_):
    q, k, v, log_g, beta, ga = s4_transpose(x_)
    # 训练路径：末态 S 不被下游消费 → ZSC 特化；S 不进 loss（否则违反
    # zero_state_cot 前提）。
    out, _S = _chunk_kda(q, k, v, log_g, beta, None, zero_state_cot=True)
    return out, ga


def s6_out(x_):
    out, g = s5_chunk(x_)
    out = out.transpose(0, 2, 1, 3)
    gate = kda_out_gate(g).reshape(B, T, Hv, Dh)
    out = m.o_norm(out.astype(x_.dtype)) * gate
    return (m.o_proj(out.reshape(B, T, -1).astype(x_.dtype)),)


STAGES = [
    ("融合投影 x@w_in.T", s1_proj),
    ("+ 3× causal_conv", s2_conv),
    ("+ rms_unit/门/softplus", s3_gates),
    ("+ transpose→f32", s4_transpose),
    ("+ _chunk_kda", s5_chunk),
    ("+ o_norm/gate/o_proj", s6_out),
]

p = m.trainable_parameters()
prev_f = prev_b = 0.0
print(f"{'累积前缀':<26}{'fwd':>8}{'Δfwd':>8}{'bwd':>8}{'Δbwd':>8}{'Δf+b':>8}")
res = []
for label, fn in STAGES:

    def loss(x_, p_, _fn=fn):
        m.update(p_)
        outs = _fn(x_)
        return sum((o.astype(mx.float32) ** 2).sum() for o in outs)

    cf = mx.compile(loss)
    cvg = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
    f = timed(lambda: cf(x, p))
    fb = timed(lambda: cvg(x, p))
    b = fb - f
    res.append((label, f - prev_f, b - prev_b))
    print(
        f"{label:<26}{f:>8.2f}{f - prev_f:>8.2f}{b:>8.2f}{b - prev_b:>8.2f}"
        f"{(f - prev_f) + (b - prev_b):>8.2f}"
    )
    prev_f, prev_b = f, b
    mx.clear_cache()

print(f"\n各段 f+b 降序（×{kda_count} 层外推：主干 {main_kda} + MTP {mtp_kda}）")
for label, df, db in sorted(res, key=lambda r: -(r[1] + r[2])):
    print(f"  {label:<26}{df + db:>7.2f}ms  → {(df + db) * kda_count:>7.1f}ms/步")
print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
