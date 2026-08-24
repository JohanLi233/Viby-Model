"""Temporal MuonH（Cache Q）：Gram-NS≈NS5、命中步跟动量、墙钟摊薄。

用法: .venv/bin/python experiments/test_muonh_cache.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import BatchedMuon

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


def cosine(A, B):
    a = A.astype(mx.float32).reshape(A.shape[0], -1)
    b = B.astype(mx.float32).reshape(B.shape[0], -1)
    num = (a * b).sum(-1)
    den = mx.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-12
    return float(mx.mean(num / den).item())


def rel_f(A, B):
    d = mx.linalg.norm((A - B).astype(mx.float32), axis=(-2, -1))
    n = mx.linalg.norm(B.astype(mx.float32), axis=(-2, -1)) + 1e-12
    return float(mx.mean(d / n).item())


def test_gram_matches_ns5():
    """刷新步用的 Gram-NS 应与标准 NS5 方向高度一致。"""
    mx.random.seed(0)
    opt = BatchedMuon(learning_rate=0.01, ns_bf16=True)
    for shape in ((32, 384, 512), (32, 512, 384), (16, 64, 48)):
        X = mx.random.normal(shape).astype(mx.bfloat16)
        D = opt._ns5(X)
        G = opt._ns5_gram(X)
        mx.eval(D, G)
        cos = cosine(G, D)
        rf = rel_f(G, D)
        check(
            f"Gram≈NS5 {shape} cos={cos:.4f}",
            cos > 0.99,
            f"relF={rf:.4f}",
        )


def test_Q_apply_equals_ns5_on_same_X():
    """同一 X 上 NS5 累积的 Q@normalize(X) 应等于 NS5 输出。"""
    mx.random.seed(1)
    opt = BatchedMuon(learning_rate=0.01, ns_bf16=True)
    X = mx.random.normal((24, 384, 512)).astype(mx.bfloat16)
    Y, Q, tr = opt._ns5(X, return_Q=True)
    Y2 = opt._apply_cached_Q(Q, X, tr)
    D = opt._ns5(X)
    mx.eval(Y, Y2, D)
    d = float(mx.max(mx.abs(Y.astype(mx.float32) - Y2.astype(mx.float32))).item())
    d2 = float(mx.max(mx.abs(Y.astype(mx.float32) - D.astype(mx.float32))).item())
    check("Q@X == NS5(X) 同输入", d < 2e-2, f"maxdiff={d}")
    check("return_Q 的 Y 与纯 NS5 一致", d2 < 2e-2, f"maxdiff={d2}")


def test_cache_tracks_momentum_better_than_stale_D():
    """EMA 动量下，命中步 Cache Q 应不差于（通常优于）复用旧 D。"""
    mx.random.seed(2)
    N, r, c = 64, 96, 128
    m_mom = 0.95
    every = 8
    opt = BatchedMuon(learning_rate=0.01, ns_bf16=True)

    V = mx.zeros((N, r, c), dtype=mx.bfloat16)
    G = mx.random.normal((N, r, c)).astype(mx.bfloat16) * 0.01
    Q = None
    tr = False
    D_stale = None
    cos_q, cos_d = [], []

    for step in range(32):
        G = 0.9 * G + 0.1 * mx.random.normal((N, r, c)).astype(mx.bfloat16) * 0.01
        V = m_mom * V + (1.0 - m_mom) * G
        U = G * (1.0 - m_mom) + V * m_mom
        D5 = opt._ns5(U)
        mx.eval(D5)
        if step % every == 0 or Q is None:
            _, Q, tr = opt._ns5(U, return_Q=True)
            D_stale = D5
            mx.eval(Q)
            continue
        if step < every:
            continue
        Dq = opt._apply_cached_Q(Q, U, tr)
        mx.eval(Dq)
        cos_q.append(cosine(Dq, D5))
        cos_d.append(cosine(D_stale, D5))

    aq, ad = sum(cos_q) / len(cos_q), sum(cos_d) / len(cos_d)
    check(
        f"Cache Q cos≥复用D ({aq:.4f}≥{ad:.4f}-0.005)",
        aq + 1e-6 >= ad - 0.005,
        f"n={len(cos_q)}",
    )
    check(f"Cache Q 命中 cos>0.95 ({aq:.4f})", aq > 0.95)


def test_end_to_end_stack_cache_q():
    """apply_gradients：stack_cache_q + every=4 跑通，范数球仍冻结。"""
    mx.random.seed(3)
    E, r, c = 8, 16, 24
    P = {"e": mx.random.normal((E, r, c)).astype(mx.bfloat16)}
    G = {"e": mx.random.normal((E, r, c)).astype(mx.bfloat16)}
    # 逐专家 Frobenius（与 hyperball_project 的 axis 一致）
    n0 = mx.linalg.norm(P["e"].astype(mx.float32), axis=(-2, -1))
    opt = BatchedMuon(
        learning_rate=0.05,
        hyperball=True,
        ns_bf16=True,
        stack_ns_every=4,
        stack_cache_q=True,
    )
    for _ in range(8):
        P = opt.apply_gradients(dict(G), P)
        mx.eval(P)
    n1 = mx.linalg.norm(P["e"].astype(mx.float32), axis=(-2, -1))
    rel = float(mx.mean(mx.abs(n1 - n0) / (n0 + 1e-12)).item())
    check(
        "cache_q 多步后逐专家范数冻结",
        rel < 5e-3,
        f"mean rel={rel:.2e}",
    )


def test_amortized_faster_than_ns5_every_step():
    """小规模：every=8 + cache_q 摊薄应明显快于 every=1。"""
    mx.random.seed(4)
    # 缩小以免 CI/本机过慢，但仍含 stack 路径
    params = {
        "e0": mx.random.normal((64, 128, 192)).astype(mx.bfloat16),
        "e1": mx.random.normal((64, 128, 192)).astype(mx.bfloat16),
    }
    grads = {
        k: mx.random.normal(v.shape).astype(mx.bfloat16) for k, v in params.items()
    }
    mx.eval(params, grads)

    def bench(every, cache_q, iters=12):
        opt = BatchedMuon(
            learning_rate=0.01,
            hyperball=True,
            ns_bf16=True,
            stack_ns_every=every,
            stack_cache_q=cache_q,
        )
        p = dict(params)
        for _ in range(2):
            p = opt.apply_gradients(dict(grads), p)
            mx.eval(p)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            p = opt.apply_gradients(dict(grads), p)
            mx.eval(p)
            ts.append(time.perf_counter() - t0)
        return sum(ts) / len(ts)

    t_full = bench(1, False)
    t_cache = bench(8, True)
    check(
        f"cache_q e8 快于每步 NS5 ({t_cache * 1e3:.1f}<{t_full * 1e3:.1f}*0.55)",
        t_cache < t_full * 0.55,
    )


if __name__ == "__main__":
    test_gram_matches_ns5()
    test_Q_apply_equals_ns5_on_same_X()
    test_cache_tracks_momentum_better_than_stale_D()
    test_end_to_end_stack_cache_q()
    test_amortized_faster_than_ns5_every_step()
    print(f"\n{'全部通过' if FAIL == 0 else f'{FAIL} 项失败'}")
    sys.exit(1 if FAIL else 0)
