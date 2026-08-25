"""验证 BatchedMuon._ns_auto 的数值等价性。

_ns_auto 按形状在 _ns5 / _ns5_gram 间切换（两者应算术等价）。
（原命中步/降频复用的验证随该机制一起删除——r082 归因：EVERY=8
早期损失 0.4-0.5 nat，负面结果见 BatchedMuon 类 docstring。）

用法: uv run python experiments/verify_muon_hit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

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

print(f"\n{'全部通过' if ok else '存在不一致'}")
sys.exit(0 if ok else 1)
