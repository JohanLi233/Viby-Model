"""验证 BatchedMuon 改动的数值等价性。

两处改动需要证明只是「同一算术的不同排布」：
  1. _ns_auto 按形状在 _ns5 / _ns5_gram 间切换（两者应算术等价）
  2. 堆叠组命中步改逐张量（免 mx.stack），须与堆叠路径逐位一致

用法: uv run python experiments/verify_muon_hit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten

from trainer.muon import BatchedMuon

mx.random.seed(0)
ok = True


def rel(a, b):
    return (
        mx.linalg.norm((a - b).astype(mx.float32))
        / mx.maximum(mx.linalg.norm(b.astype(mx.float32)), 1e-9)
    ).item()


def cos(a, b):
    a32, b32 = a.astype(mx.float32).flatten(), b.astype(mx.float32).flatten()
    return (
        (a32 * b32).sum() / mx.maximum(mx.linalg.norm(a32) * mx.linalg.norm(b32), 1e-9)
    ).item()


# ---- 1. _ns_auto vs 两个底层实现 ----
print("=== _ns_auto 选路与 _ns5/_ns5_gram 等价性 ===")
opt = BatchedMuon(learning_rate=mx.array(0.01), momentum=0.95, ns_steps=5)
for r, c in [(768, 768), (6400, 768), (768, 384), (384, 384), (384, 640)]:
    X = (mx.random.normal((8, r, c)) * 0.02).astype(mx.bfloat16)
    mx.eval(X)
    a = opt._ns5(X)
    g = opt._ns5_gram(X)
    auto = opt._ns_auto(X)
    mx.eval(a, g, auto)
    picked = "ns5" if r == c else "gram"
    hit = rel(auto, a if picked == "ns5" else g)
    print(
        f"  ({r:>4},{c:>4}) 选 {picked:<4} 与所选一致 rel={hit:.2e}  "
        f"ns5↔gram cos={cos(a, g):.6f} rel={rel(g, a):.4f}"
    )
    if hit > 1e-6 or cos(a, g) < 0.99:
        ok = False
        print("    ✗ 不一致")

# ---- 2. 命中步：逐张量 vs 堆叠 ----
print("\n=== 堆叠组命中步：逐张量路径 vs 旧堆叠路径 ===")
N, b, r, c = 4, 32, 768, 384
params = {
    f"l{i}.w": (mx.random.normal((b, r, c)) * 0.02).astype(mx.bfloat16)
    for i in range(N)
}
grads = {
    f"l{i}.w": (mx.random.normal((b, r, c)) * 0.01).astype(mx.bfloat16)
    for i in range(N)
}
mx.eval(params, grads)


def run(every, steps=3):
    o = BatchedMuon(
        learning_rate=mx.array(0.01), momentum=0.95, ns_steps=5, stack_ns_every=every
    )
    p = dict(params)
    for _ in range(steps):
        p = o.apply_gradients(dict(grads), p)
        mx.eval(p, o.state)
    return dict(tree_flatten(p))


# every=1：每步刷新（无命中步）；every=3：第 2、3 步走命中路径
p_hit = run(3)
# 参考：手工复现旧堆叠路径（stack G/P/V + mom + apply）
from trainer.muon import _stack_apply_kernel, _stack_mom_kernel  # noqa: E402

mom = _stack_mom_kernel(0.95, True, 0.0)
app = _stack_apply_kernel(True)
paths = [f"l{i}.w" for i in range(N)]
P = mx.stack([params[q] for q in paths])
V = mx.zeros_like(P)
G = mx.stack([grads[q] for q in paths])
Xc = None
for s in range(3):
    U, V = mom(G, P, V)
    if s == 0:
        Xc = (
            BatchedMuon(learning_rate=mx.array(0.01), ns_steps=5)
            ._ns_auto(U.reshape(N * b, r, c))
            .reshape(N, b, r, c)
        )
    lr = mx.array(0.01, dtype=mx.bfloat16) * (max(1.0, r / c) ** 0.5)
    P = app(P, Xc, lr).astype(mx.bfloat16)
mx.eval(P)

worst = max(rel(p_hit[f"l{i}.w"], P[i]) for i in range(N))
print(f"  3 步（1 刷新 + 2 命中）最大相对偏差 rel={worst:.2e}")
if worst > 1e-6:
    ok = False
    print("    ✗ 命中步与堆叠路径不一致")

# ---- 3. 命中步动量与 nesterov 组合正确 ----
print("\n=== momv_fn 与 mom_fn 的 V 输出一致 ===")
from trainer.muon import _stack_momv_kernel  # noqa: E402

momv = _stack_momv_kernel(0.95, 0.0)
Gt = (mx.random.normal((b, r, c)) * 0.01).astype(mx.bfloat16)
Pt = (mx.random.normal((b, r, c)) * 0.02).astype(mx.bfloat16)
Vt = (mx.random.normal((b, r, c)) * 0.01).astype(mx.bfloat16)
mx.eval(Gt, Pt, Vt)
_, v_full = mom(Gt, Pt, Vt)
v_only = momv(Gt, Pt, Vt)
mx.eval(v_full, v_only)
d = rel(v_only, v_full)
print(f"  rel={d:.2e}")
if d > 0:
    ok = False
    print("    ✗ V 不逐位一致")

print(f"\n{'全部通过' if ok else '存在不一致'}")
sys.exit(0 if ok else 1)
