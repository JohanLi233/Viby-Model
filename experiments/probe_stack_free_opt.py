"""优化器专家组的跨层 stack 纯搬运：三个等价变体的墙钟与数值对照。

现状（A）：G/P/V 各跨层 stack 成 (9,256,r,c) → mom_fn → Gram-NS(2304,·,·)
→ apply_fn → 逐层切回。三份整组拷贝是纯搬运（NS 的 batch 维无耦合、动量
与投影均逐矩阵/逐元素），只是为了「一次大调用」。

变体：
  B  逐层 mom_fn（零拷贝）→ 只 stack U 一份 → 同一次大 NS → 逐层 apply_fn。
     NS 输入形状与 A 完全相同 ⇒ NS 输出逐位一致；mom/apply 逐元素/逐矩阵，
     预期整链逐位一致。
  C  全逐层：mom_fn ×9 + NS(256,·,·) ×9 + apply_fn ×9，零 stack。
     NS 的 GEMM batch 从 2304 变 256，tile 调度可能动低 bits ⇒ 只保证数学
     等价（相对差 ~1e-3 内），需要单独确认 GEMM 效率不掉。

用法: .venv/bin/python experiments/probe_stack_free_opt.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import _ns_core_fn, _stack_apply_kernel, _stack_mom_kernel

GROUPS = {
    "gate_up (9,256,768,384)": (9, 256, 768, 384),
    "down    (9,256,384,384)": (9, 256, 384, 384),
}
M_, WD, NEST = 0.95, 0.0, True
NS = _ns_core_fn("gram", True, 5, "cubic5b05")
mom_fn = _stack_mom_kernel(M_, NEST, WD)
apply_fn = _stack_apply_kernel(True)  # hyperball
LR = mx.array(0.01, mx.bfloat16)


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def variant_A(Gs, Ps, Vs):
    G = mx.stack(Gs)
    P = mx.stack(Ps)
    V = mx.stack(Vs)
    U, Vn = mom_fn(G, P, V)
    N, b, r, c = U.shape
    X = NS(U.reshape(N * b, r, c)).reshape(N, b, r, c)
    NP = apply_fn(P, X, LR)
    return [Vn[i] for i in range(N)], [NP[i] for i in range(N)]


def variant_B(Gs, Ps, Vs):
    outs = [mom_fn(g[None], p[None], v[None]) for g, p, v in zip(Gs, Ps, Vs)]
    U = mx.stack([o[0][0] for o in outs])
    Vn = [o[1][0] for o in outs]
    N, b, r, c = U.shape
    X = NS(U.reshape(N * b, r, c)).reshape(N, b, r, c)
    NP = [apply_fn(p[None], X[i : i + 1], LR)[0] for i, p in enumerate(Ps)]
    return Vn, NP


def variant_C(Gs, Ps, Vs):
    Vn, NP = [], []
    for g, p, v in zip(Gs, Ps, Vs):
        U, v_ = mom_fn(g, p, v)  # (b,r,c)
        b, r, c = U.shape
        X = NS(U)  # (b,r,c)，batch=256
        NP.append(apply_fn(p, X, LR))
        Vn.append(v_)
    return Vn, NP


def rel_diff(a, b):
    d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max()
    s = b.astype(mx.float32).abs().max()
    return (d / (s + 1e-12)).item()


for name, (N, b, r, c) in GROUPS.items():
    mx.random.seed(0)
    Gs = [(mx.random.normal((b, r, c)) * 0.02).astype(mx.bfloat16) for _ in range(N)]
    Ps = [(mx.random.normal((b, r, c)) * 0.02).astype(mx.bfloat16) for _ in range(N)]
    Vs = [mx.zeros_like(p) for p in Ps]
    mx.eval(Gs, Ps, Vs)

    tA = timed(lambda: variant_A(Gs, Ps, Vs))
    tB = timed(lambda: variant_B(Gs, Ps, Vs))
    tC = timed(lambda: variant_C(Gs, Ps, Vs))

    vA, pA = variant_A(Gs, Ps, Vs)
    vB, pB = variant_B(Gs, Ps, Vs)
    vC, pC = variant_C(Gs, Ps, Vs)
    mx.eval(vA, pA, vB, pB, vC, pC)
    bit_B = all(mx.array_equal(x, y).item() for x, y in zip(pA + vA, pB + vB))
    rd_C = max(rel_diff(x, y) for x, y in zip(pA, pC))
    print(f"{name}:  A {tA:7.2f}ms | B {tB:7.2f}ms | C {tC:7.2f}ms")
    print(f"    B 逐位一致={bit_B} | C 最大相对差={rd_C:.2e}")
