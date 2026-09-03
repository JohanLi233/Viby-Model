"""EigenGate：状态谱高通的数值契约。

1. λ=0 是精确恒等（不跑多项式）。
2. 目标 A（keep）：小奇异值被压、大奇异值幅度近似保留；左右 Gram 等价。
3. 酉等变：Φ(Q S R) = Q Φ(S) R。
4. 默认关：_chunk_kda 与基线逐位一致。
5. 打开后 chunk 扫描与逐 token 递推（含门控点）对齐，梯度也对齐。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import os

import mlx.core as mx
import numpy as np


def _svd_np(a):
    return np.linalg.svd(a, full_matrices=False)


def test_lambda_zero_is_identity():
    from model.eigengate import apply

    mx.random.seed(0)
    S = mx.random.normal((2, 3, 16, 16)).astype(mx.float32)
    out = apply(S, lam=0.0)
    mx.eval(out)
    assert out is S or (out - S).abs().max().item() == 0.0
    print("lambda 0 identity: OK")


def test_highpass_kills_small_keeps_large():
    from model.eigengate import apply, cubic_coeffs

    # 构造两个正交 spike：强信号 + 弱干涉
    u = np.zeros((8, 2), dtype=np.float32)
    v = np.zeros((8, 2), dtype=np.float32)
    u[0, 0] = 1.0
    u[1, 1] = 1.0
    v[0, 0] = 1.0
    v[2, 1] = 1.0
    sig = 1.0 * np.outer(u[:, 0], v[:, 0]) + 0.01 * np.outer(u[:, 1], v[:, 1])
    S = mx.array(sig[None, None])  # (1,1,8,8)
    coeffs = cubic_coeffs(l0=0.05)
    out = apply(S, lam=1.0, coeffs=coeffs, target="keep")
    mx.eval(out)
    s_in = _svd_np(sig)[1]
    s_out = _svd_np(np.array(out[0, 0]))[1]
    # 大模式保留到 20% 以内；小模式（σ_F≈0.01）相对被压到一半以下
    assert abs(s_out[0] - s_in[0]) / s_in[0] < 0.2, (s_in[0], s_out[0])
    assert s_out[1] < 0.5 * s_in[1], (s_in[1], s_out[1])
    print(
        f"highpass keep: σ {s_in[0]:.3f}/{s_in[1]:.3e} → {s_out[0]:.3f}/{s_out[1]:.3e} OK"
    )


def test_left_right_gram_equivalent():
    from model.eigengate import highpass, cubic_coeffs

    mx.random.seed(1)
    S = mx.random.normal((1, 1, 12, 8)).astype(mx.float32)  # 非方
    coeffs = cubic_coeffs(l0=0.05)
    a = highpass(S, coeffs, target="keep", prefer="left")
    b = highpass(S, coeffs, target="keep", prefer="right")
    mx.eval(a, b)
    d = (a - b).abs().max().item()
    rel = d / (a.abs().max().item() + 1e-12)
    print(f"left vs right gram: max|Δ|={d:.3e} rel={rel:.3e}")
    assert rel < 2e-5, (d, rel)


def test_equivariance():
    from model.eigengate import apply, cubic_coeffs

    rng = np.random.default_rng(2)
    s = rng.normal(size=(8, 8)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)).astype(np.float32))
    r, _ = np.linalg.qr(rng.normal(size=(8, 8)).astype(np.float32))
    S = mx.array(s[None, None])
    Q = mx.array(q)
    R = mx.array(r)
    coeffs = cubic_coeffs(l0=0.05)
    phi_s = apply(S, lam=1.0, coeffs=coeffs)
    S_rot = (Q @ S[0, 0] @ mx.swapaxes(R, -1, -2))[None, None]
    phi_rot = apply(S_rot, lam=1.0, coeffs=coeffs)
    recon = Q @ phi_s[0, 0] @ mx.swapaxes(R, -1, -2)
    mx.eval(phi_rot, recon)
    d = (phi_rot[0, 0] - recon).abs().max().item()
    rel = d / (recon.abs().max().item() + 1e-12)
    print(f"equivariance: max|Δ|={d:.3e} rel={rel:.3e}")
    assert rel < 2e-4, (d, rel)


def test_disabled_matches_baseline():
    os.environ["VIBY_EIGENGATE"] = "0"
    from model.kda import _chunk_kda, _recurrent_kda
    from test_kda import _rand_inputs

    q, k, v, log_g, beta = _rand_inputs(T=37, seed=3)
    o1, s1 = _chunk_kda(q, k, v, log_g, beta)
    o2, s2 = _recurrent_kda(q, k, v, log_g, beta)
    mx.eval(o1, s1, o2, s2)
    do = (o1 - o2).abs().max().item()
    ds = (s1 - s2).abs().max().item()
    print(f"disabled chunk vs recurrent: out {do:.3e} state {ds:.3e}")
    assert do < 2e-4 and ds < 2e-4, (do, ds)


def test_enabled_chunk_vs_recurrent():
    os.environ["VIBY_EIGENGATE"] = "1"
    os.environ["VIBY_EIGENGATE_K"] = "1"  # 每 chunk（16 token）一次
    os.environ["VIBY_EIGENGATE_LAM"] = "1"
    from model.kda import _chunk_kda, _recurrent_kda
    from test_kda import _rand_inputs

    q, k, v, log_g, beta = _rand_inputs(B=2, H=2, T=37, D=16, seed=4)
    o1, s1 = _chunk_kda(q, k, v, log_g, beta)
    o2, s2 = _recurrent_kda(q, k, v, log_g, beta)
    mx.eval(o1, s1, o2, s2)
    do = (o1 - o2).abs().max().item()
    ds = (s1 - s2).abs().max().item()
    print(f"enabled chunk vs recurrent: out {do:.3e} state {ds:.3e}")
    assert do < 5e-4 and ds < 5e-4, (do, ds)
    # 门控确实改了状态：与关闭路径不同
    os.environ["VIBY_EIGENGATE"] = "0"
    o0, s0 = _chunk_kda(q, k, v, log_g, beta)
    mx.eval(o0, s0)
    os.environ["VIBY_EIGENGATE"] = "1"
    assert (s1 - s0).abs().max().item() > 1e-4, "K=1 应在 T=37 触发门控"


def test_enabled_grads():
    os.environ["VIBY_EIGENGATE"] = "1"
    os.environ["VIBY_EIGENGATE_K"] = "1"
    os.environ["VIBY_EIGENGATE_LAM"] = "1"
    from model.kda import _chunk_kda, _recurrent_kda
    from test_kda import _rand_inputs

    q, k, v, log_g, beta = _rand_inputs(B=1, H=2, T=33, D=16, seed=5)

    def loss_c(q, k, v):
        o, s = _chunk_kda(q, k, v, log_g, beta)
        return (o**2).sum() + (s**2).sum()

    def loss_r(q, k, v):
        o, s = _recurrent_kda(q, k, v, log_g, beta)
        return (o**2).sum() + (s**2).sum()

    lc, gc = mx.value_and_grad(loss_c, argnums=(0, 1, 2))(q, k, v)
    lr, gr = mx.value_and_grad(loss_r, argnums=(0, 1, 2))(q, k, v)
    mx.eval(lc, lr, *gc, *gr)
    assert abs(lc.item() - lr.item()) < 2e-3, (lc.item(), lr.item())
    for name, a, b in zip("qkv", gc, gr):
        d = (a - b).abs().max().item()
        rel = d / (b.abs().max().item() + 1e-12)
        print(f"enabled grad {name}: max|Δ|={d:.3e} rel={rel:.3e}")
        assert rel < 5e-3, (name, d, rel)


def test_short_seq_no_gate_is_baseline():
    """T < period 时不插入门控，应与关闭路径逐位一致。"""
    os.environ["VIBY_EIGENGATE"] = "1"
    os.environ["VIBY_EIGENGATE_K"] = "16"  # period = 256 token
    os.environ["VIBY_EIGENGATE_LAM"] = "1"
    from model.kda import _chunk_kda
    from test_kda import _rand_inputs

    q, k, v, log_g, beta = _rand_inputs(T=37, seed=6)
    o_on, s_on = _chunk_kda(q, k, v, log_g, beta)
    os.environ["VIBY_EIGENGATE"] = "0"
    o_off, s_off = _chunk_kda(q, k, v, log_g, beta)
    mx.eval(o_on, s_on, o_off, s_off)
    do = (o_on - o_off).abs().max().item()
    ds = (s_on - s_off).abs().max().item()
    print(f"short seq no-gate vs off: out {do:.3e} state {ds:.3e}")
    assert do == 0.0 and ds == 0.0, (do, ds)


if __name__ == "__main__":
    test_lambda_zero_is_identity()
    test_highpass_kills_small_keeps_large()
    test_left_right_gram_equivalent()
    test_equivariance()
    test_disabled_matches_baseline()
    test_enabled_chunk_vs_recurrent()
    test_enabled_grads()
    test_short_seq_no_gate_is_baseline()
    print("ALL EIGENGATE TESTS PASSED")
