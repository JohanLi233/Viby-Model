"""KDA 数值验证：chunk 并行 vs 逐 token 递推。

1. _chunk_kda 与 _recurrent_kda 输出/终态一致（f32，含 T 非 chunk 整数倍）。
2. 两条路径对 q/k/v 的梯度一致。
3. 模块级：训练前向（chunk）与逐步 decode（cache 递推）一致（bf16 容差）。
"""

import mlx.core as mx

from model.kda import KDAAttention, _chunk_kda, _recurrent_kda
from model.config import VibyConfig


def _rand_inputs(B=2, H=3, T=37, D=32, decay_lo=0.0, decay_hi=0.3, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, T, D)) * 0.5
    k = mx.random.normal((B, H, T, D)) * (D**-0.5)  # ~单位 L2
    v = mx.random.normal((B, H, T, D))
    rate = mx.random.uniform(decay_lo, decay_hi, (B, H, T, D))
    log_g = -rate
    beta = mx.sigmoid(mx.random.normal((B, H, T)))
    return q, k, v, log_g, beta


def test_chunk_vs_recurrent():
    for name, (lo, hi) in [(" mild", (0.0, 0.3)), ("strong", (0.5, 2.0))]:
        q, k, v, log_g, beta = _rand_inputs(decay_lo=lo, decay_hi=hi)
        o1, s1 = _chunk_kda(q, k, v, log_g, beta)
        o2, s2 = _recurrent_kda(q, k, v, log_g, beta)
        mx.eval(o1, s1, o2, s2)
        assert not mx.any(mx.isnan(o1)).item(), f"chunk 输出 NaN ({name} 衰减)"
        do = (o1 - o2).abs().max().item()
        ds = (s1 - s2).abs().max().item()
        print(f"[{name} decay] out max|Δ|={do:.3e}  state max|Δ|={ds:.3e}")
        assert do < 2e-4 and ds < 2e-4, (do, ds)
    print("chunk vs recurrent: OK")


def test_grads():
    q, k, v, log_g, beta = _rand_inputs(T=33, seed=1)

    def loss_c(q, k, v):
        o, s = _chunk_kda(q, k, v, log_g, beta)
        return (o**2).sum() + (s**2).sum()

    def loss_r(q, k, v):
        o, s = _recurrent_kda(q, k, v, log_g, beta)
        return (o**2).sum() + (s**2).sum()

    lc, gc = mx.value_and_grad(loss_c, argnums=(0, 1, 2))(q, k, v)
    lr, gr = mx.value_and_grad(loss_r, argnums=(0, 1, 2))(q, k, v)
    mx.eval(lc, lr, *gc, *gr)
    assert abs(lc.item() - lr.item()) < 1e-3, (lc.item(), lr.item())
    for name, a, b in zip("qkv", gc, gr):
        d = (a - b).abs().max().item()
        rel = d / (b.abs().max().item() + 1e-12)
        print(f"grad {name}: max|Δ|={d:.3e} rel={rel:.3e}")
        assert rel < 1e-3, (name, d, rel)
    print("grads: OK")


def test_module_train_vs_decode():
    mx.random.seed(2)
    cfg = VibyConfig(
        hidden_size=192,
        num_hidden_layers=8,
        vocab_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kda_v_head_ratio=1,
        ngram_table_size=0,
    )
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    x = mx.random.normal((2, 37, 192))
    x = x.astype(mx.bfloat16)
    y_train, _ = attn(x)
    # 逐步 decode：首步建 cache，逐 token 递推
    from model.cache import KVCache

    cache = KVCache()
    ys = []
    for t in range(37):
        y_t, _ = attn(x[:, t : t + 1], past_key_value=cache, use_cache=True)
        ys.append(y_t)
    y_dec = mx.concatenate(ys, axis=1)
    mx.eval(y_train, y_dec)
    d = (y_train.astype(mx.float32) - y_dec.astype(mx.float32)).abs().max().item()
    scale = y_train.astype(mx.float32).abs().max().item()
    print(f"module train vs decode: max|Δ|={d:.3e} (|y|max={scale:.3e})")
    assert d < 2e-2, d  # bf16 容差
    assert cache.offset == 37
    print("module train vs decode: OK")


def test_decay_clamp_no_overflow():
    """r081 NaN 根因的最小复现与钳制回归。

    chunk 分解要 materialize ki = k·e^{−gc}：每步衰减 6 nats 时 15 步累计
    e^90 越过 f32 上限 → inf → NaN（r081_gqa_qb step ~254 的机制）。
    模块路径改 K3 下界门后 g∈(g_min,0)，极端 A_h/dt_bias 仍应有限。
    """
    q, k, v, _, beta = _rand_inputs(T=64, seed=3)
    # 机制复现：6 nats/步 × 15 步 ≈ e^90 → f32 溢出（函数级，无钳制）
    log_g_extreme = mx.full(q.shape, -6.0)
    o_bad, _ = _chunk_kda(q, k, v, log_g_extreme, beta)
    mx.eval(o_bad)
    print(f"6 nats/step 无钳制: finite={mx.all(mx.isfinite(o_bad)).item()}")
    assert not mx.all(mx.isfinite(o_bad)).item(), "6 nats/步应复现溢出（机制确认）"

    # 钳制上界处（4 nats/步）：chunk 有限且与 recurrent 一致
    log_g_clamped = mx.full(q.shape, -4.0)
    o1, s1 = _chunk_kda(q, k, v, log_g_clamped, beta)
    o2, s2 = _recurrent_kda(q, k, v, log_g_clamped, beta)
    mx.eval(o1, s1, o2, s2)
    assert mx.all(mx.isfinite(o1)).item(), "4 nats/步 chunk 输出应有限"
    do = (o1 - o2).abs().max().item()
    print(f"4 nats/step(钳制上界) chunk vs recurrent: max|Δ|={do:.3e}")
    assert do < 5e-3, do

    # 模块级：把衰减参数推到溢出区（模拟 Adam 步长异常后的漂移终点），
    # 钳制必须保证前向与梯度都有限
    mx.random.seed(4)
    cfg = VibyConfig(
        hidden_size=192,
        num_hidden_layers=8,
        vocab_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kda_v_head_ratio=1,
        ngram_table_size=0,
    )
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    attn.A_log = mx.full(attn.A_log.shape, 4.0)  # e^4 ≈ 55
    attn.dt_bias = mx.full(
        attn.dt_bias.shape, 8.0
    )  # e^4·8≈435 → σ≈1 → g 饱和至 g_min=−5/步
    x = mx.random.normal((2, 64, 192)).astype(mx.bfloat16)

    def loss_fn(a, d):
        attn.A_log, attn.dt_bias = a, d
        y, _ = attn(x)
        return y.astype(mx.float32).abs().sum()

    loss, (g_a, g_d) = mx.value_and_grad(loss_fn, argnums=(0, 1))(
        attn.A_log, attn.dt_bias
    )
    mx.eval(loss, g_a, g_d)
    assert mx.all(mx.isfinite(loss)).item(), "模块前向必须有限（钳制生效）"
    assert mx.all(mx.isfinite(g_a)).item() and mx.all(mx.isfinite(g_d)).item(), (
        "极端衰减下梯度必须有限"
    )
    print("decay clamp no-overflow: OK")


def test_kda_inner_matches_eager():
    """kda_inner 前向/梯度贴原式（合成 GEMM + 闭式 (I+L)⁻¹ VJP）。"""
    from model.kernels.kda_inner import _inner_eager, kda_inner

    mx.random.seed(8)
    B, H, NC, C, D = 2, 2, 3, 16, 32
    qe = mx.random.normal((B, H, NC, C, D)) * 0.3
    ke = mx.random.normal((B, H, NC, C, D)) * 0.3
    ki = mx.random.normal((B, H, NC, C, D)) * 0.3
    v = mx.random.normal((B, H, NC, C, D)) * 0.5
    beta = mx.sigmoid(mx.random.normal((B, H, NC, C)))
    mx.eval(qe, ke, ki, v, beta)

    w1, u1, a1 = kda_inner(qe, ke, ki, v, beta)
    w0, u0, a0 = _inner_eager(qe, ke, ki, v, beta)
    mx.eval(w1, u1, a1, w0, u0, a0)
    for name, x, y in (("w", w1, w0), ("u", u1, u0), ("Aqk", a1, a0)):
        d = (x - y).abs().max().item()
        assert d < 2e-5, (name, d)

    def loss_new(qe, ke, ki, v, beta):
        w, u, a = kda_inner(qe, ke, ki, v, beta)
        return (w**2).sum() + (u**2).sum() + (a**2).sum()

    def loss_old(qe, ke, ki, v, beta):
        w, u, a = _inner_eager(qe, ke, ki, v, beta)
        return (w**2).sum() + (u**2).sum() + (a**2).sum()

    ln, gn = mx.value_and_grad(loss_new, argnums=(0, 1, 2, 3, 4))(qe, ke, ki, v, beta)
    lo, go = mx.value_and_grad(loss_old, argnums=(0, 1, 2, 3, 4))(qe, ke, ki, v, beta)
    mx.eval(ln, lo, *gn, *go)
    assert abs(ln.item() - lo.item()) < 1e-4, (ln.item(), lo.item())
    for name, a, b in zip(("qe", "ke", "ki", "v", "beta"), gn, go):
        d = (a - b).abs().max().item()
        rel = d / (b.abs().max().item() + 1e-12)
        assert rel < 2e-4, (name, d, rel)
    print("kda_inner vs eager: OK")


def _kda_cfg(**kw):
    base = dict(
        hidden_size=192,
        num_hidden_layers=8,
        vocab_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        ngram_table_size=0,
    )
    base.update(kw)
    return VibyConfig(**base)


def test_kda_v_head_ratio_shapes():
    """Qwen GDN 口径：V 头 = ratio × QK 头；ratio=1 时与旧正方形投影同宽。"""
    cfg1 = _kda_cfg(kda_v_head_ratio=1)
    a1 = KDAAttention(cfg1, layer_idx=0)
    d, h = cfg1.hidden_size, cfg1.num_attention_heads
    hd = cfg1.head_dim
    assert a1.n_k_heads == h and a1.n_v_heads == h
    assert a1.q_proj.weight.shape == (h * hd, d)
    assert a1.v_proj.weight.shape == (h * hd, d)
    assert a1.g_proj.weight.shape == (h * hd, d)
    assert a1.o_proj.weight.shape == (d, h * hd)

    cfg2 = _kda_cfg(kda_v_head_ratio=2)
    a2 = KDAAttention(cfg2, layer_idx=0)
    assert a2.n_k_heads == h and a2.n_v_heads == 2 * h
    assert a2.q_proj.weight.shape == (h * hd, d)
    assert a2.k_proj.weight.shape == (h * hd, d)
    assert a2.v_proj.weight.shape == (2 * h * hd, d)
    assert a2.g_proj.weight.shape == (2 * h * hd, d)
    assert a2.o_proj.weight.shape == (d, 2 * h * hd)
    assert a2.v_conv.weight.shape[-1] == 2 * h * hd
    print("kda v-head ratio shapes: OK")


def test_kda_v_head_ratio_train_vs_decode():
    mx.random.seed(5)
    cfg = _kda_cfg(kda_v_head_ratio=2)
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    x = mx.random.normal((2, 37, 192)).astype(mx.bfloat16)
    y_train, _ = attn(x)
    from model.cache import KVCache

    cache = KVCache()
    ys = []
    for t in range(37):
        y_t, _ = attn(x[:, t : t + 1], past_key_value=cache, use_cache=True)
        ys.append(y_t)
    y_dec = mx.concatenate(ys, axis=1)
    mx.eval(y_train, y_dec)
    d = (y_train.astype(mx.float32) - y_dec.astype(mx.float32)).abs().max().item()
    print(f"v-head×2 train vs decode: max|Δ|={d:.3e}")
    assert y_train.shape == (2, 37, 192)
    assert d < 2e-2, d
    assert cache.extras["kda_state"].shape == (2, 16, 24, 24), cache.extras[
        "kda_state"
    ].shape
    print("kda v-head ratio train vs decode: OK")


if __name__ == "__main__":
    test_chunk_vs_recurrent()
    test_grads()
    test_module_train_vs_decode()
    test_decay_clamp_no_overflow()
    test_kda_inner_matches_eager()
    test_kda_v_head_ratio_shapes()
    test_kda_v_head_ratio_train_vs_decode()
    print("ALL KDA TESTS PASSED")
