"""MoE 块隔离微基准：真实形状 bs6×2048 下 _sparse_forward 的 fwd / fwd+bwd
耗时，旧逐组 GEMM vs 融合 kernel，均衡容量 vs 集中态容量。

用途：量化 MoE 专家计算在整步（8 次 MoE 调用）中的真实占比，决定优化
杠杆在前向 kernel、反向 kernel 还是根本不在 MoE。

用法: .venv/bin/python experiments/bench_moe_block.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.model import MoEFeedForward
from experiments.test_moe_fused import build_moe, make_idx_w, prime_caps

B, T, D, I, E, K = 6, 2048, 768, 104, 112, 6
G = B * T * K


def timed(fn, iters, warm=3):
    for _ in range(warm):
        out = fn()
        mx.eval(*out if isinstance(out, tuple) else out)
    ts = []
    for _ in range(iters):
        t0 = time.time()
        out = fn()
        mx.eval(*out if isinstance(out, tuple) else out)
        ts.append(time.time() - t0)
    return min(ts), sum(ts) / len(ts)


def run_regime(name, caps_mode, iters):
    m = build_moe(E, K, I, D)
    mx.random.seed(0)
    x0 = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    gu0 = m.experts.gate_up_w.astype(mx.bfloat16)
    dw0 = m.experts.down_w.astype(mx.bfloat16)
    idx, w = make_idx_w(B, T, K, E, seed=7)
    if caps_mode == "balanced":
        caps = prime_caps(m, idx)  # 按实测计数 ×1.25+64 对齐
    else:
        caps = prime_caps(m, idx, flat=1758)  # 集中稳态 ≈ 3.3× 均值（cap≈2176）
    rows = sum(caps) * min(m._SPARSE_GROUP, E)
    C = mx.random.normal((B, T, D)) * 0.5

    def loss_fn(x, gu, dw):
        m.experts.gate_up_w = gu
        m.experts.down_w = dw
        out = m._sparse_forward(x, idx, w, step_idx=None)
        return (out.astype(mx.float32) * C).sum()

    vg = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))
    for label, dis in [("旧逐组GEMM", True), ("融合kernel", False)]:
        MoEFeedForward._FUSED_DISABLED = dis
        def fwd():
            return (loss_fn(x0, gu0, dw0),)
        def fb():
            return vg(x0, gu0, dw0)
        fmin, favg = timed(fwd, iters)
        bmin, bavg = timed(fb, iters)
        print(
            f"  {label}: fwd {fmin * 1e3:.1f}ms | fwd+bwd {bmin * 1e3:.1f}ms"
            f"（avg {favg * 1e3:.1f}/{bavg * 1e3:.1f}）⇒ bwd≈{(bmin - fmin) * 1e3:.1f}ms"
        )
    print(f"  [{name}] rows/调用={rows}（真实 pair={G}，padding {rows / G:.1f}×）")
    MoEFeedForward._FUSED_DISABLED = True


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("== 均衡容量 ==")
    run_regime("均衡", "balanced", iters)
    print("== 集中态容量（flat 2176/组）==")
    run_regime("集中", "flat", iters)


if __name__ == "__main__":
    main()
