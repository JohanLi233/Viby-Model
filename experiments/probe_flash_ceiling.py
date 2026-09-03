"""手写 flash 与可达上限的对照（目标 MLA 形状 B12 H12 T1024 Dk112 Dv64）。

三条参考线，交替测量抗漂移：
  1. 手写 flash（现状，causal）
  2. mx.fast.scaled_dot_product_attention（V 零填到 Dk，等宽快路径），
     以及它的 autodiff f+b —— mlx 自己的天花板
  3. 同 FLOPs 的稠密 batched GEMM（(BH,T,Dk)@(BH,Dk,T) + (BH,T,T)@(BH,T,Dv)），
     不含 softmax / mask —— 纯 tensor core 地板

另测 Dk=128/Dv=64（把 rope_dim 48→64 后的形状）与 Dk=64+Dv=64，看
d_qk=112 这个非 2 幂宽度本身有多贵。

用法: .venv/bin/python experiments/probe_flash_ceiling.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels.attn_fused import flash_sdpa

B, H, T = 12, 12, 1024
DT = mx.bfloat16


def _mk(dk, dv):
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, dk)) * 0.5).astype(DT)
    k = (mx.random.normal((B, H, T, dk)) * 0.5).astype(DT)
    v = (mx.random.normal((B, H, T, dv)) * 0.5).astype(DT)
    cot = (mx.random.normal((B, H, T, dv)) * 0.5).astype(DT)
    mx.eval(q, k, v, cot)
    return q, k, v, cot


def _med(fn, rounds=7, warm=2):
    ts = []
    for r in range(rounds):
        t0 = time.perf_counter()
        mx.eval(fn())
        dt = time.perf_counter() - t0
        if r >= warm:
            ts.append(dt * 1e3)
    return statistics.median(ts)


def causal_flops(dk, dv):
    """因果下三角实际 MAC 的 fwd FLOPs（QKᵀ + AV）。"""
    return B * H * T * T * (dk + dv)  # 2*(1/2 T^2)*(dk+dv) per (b,h)


def run(dk, dv, label):
    q, k, v, cot = _mk(dk, dv)
    scale = dk**-0.5
    fl = causal_flops(dk, dv)

    def hand_f():
        return flash_sdpa(q, k, v, scale=scale, mask="causal")

    def hand_fb():
        return mx.value_and_grad(
            lambda a, b, c: (
                flash_sdpa(a, b, c, scale=scale, mask="causal").astype(mx.float32) * cot
            ).sum()
        )(q, k, v)

    vpad = (
        v
        if dv == dk
        else mx.concatenate([v, mx.zeros((B, H, T, dk - dv), dtype=DT)], axis=-1)
    )
    cpad = (
        cot
        if dv == dk
        else mx.concatenate([cot, mx.zeros((B, H, T, dk - dv), dtype=DT)], axis=-1)
    )
    mx.eval(vpad, cpad)

    def mlx_f():
        return mx.fast.scaled_dot_product_attention(
            q, k, vpad, scale=scale, mask="causal"
        )

    def mlx_fb():
        return mx.value_and_grad(
            lambda a, b, c: (
                mx.fast.scaled_dot_product_attention(
                    a, b, c, scale=scale, mask="causal"
                ).astype(mx.float32)
                * cpad
            ).sum()
        )(q, k, vpad)

    q3 = q.reshape(B * H, T, dk)
    k3 = k.reshape(B * H, T, dk)
    v3 = v.reshape(B * H, T, dv)
    p3 = mx.zeros((B * H, T, T), dtype=DT)
    mx.eval(q3, k3, v3, p3)

    def dense_f():
        s = q3 @ k3.swapaxes(-1, -2)
        return s.astype(DT) @ v3

    rows = [("手写 flash", hand_f, hand_fb), ("mlx SDPA(V pad)", mlx_f, mlx_fb)]
    res = {}
    for rnd in range(7):
        for name, f, fb in rows:
            t0 = time.perf_counter()
            mx.eval(f())
            t1 = time.perf_counter()
            try:
                mx.eval(fb())
                ok = True
            except Exception:
                ok = False
            t2 = time.perf_counter()
            if rnd >= 2:
                d = res.setdefault(name, {"f": [], "fb": [], "ok": ok})
                d["f"].append((t1 - t0) * 1e3)
                d["fb"].append((t2 - t1) * 1e3)
                d["ok"] = ok
    t_dense = _med(dense_f)

    print(f"\n== {label}：Dk={dk} Dv={dv}，因果 fwd {fl / 1e9:.1f} GFLOP ==")
    print(f"{'臂':<20}{'fwd':>9}{'TF/s':>8}{'f+b':>9}{'TF/s':>8}")
    for name, _, _ in rows:
        d = res[name]
        f = statistics.median(d["f"])
        fb = statistics.median(d["fb"])
        tf = fl / (f / 1e3) / 1e12
        tfb = 3 * fl / (fb / 1e3) / 1e12 if d["ok"] else 0.0
        fbs = f"{fb:>9.2f}" if d["ok"] else f"{'n/a':>9}"
        tfbs = f"{tfb:>8.1f}" if d["ok"] else f"{'—':>8}"
        print(f"{name:<20}{f:>9.2f}{tf:>8.1f}{fbs}{tfbs}")
    # 稠密全 T² 参考（无因果裁剪）：FLOPs 是因果的 2 倍
    print(
        f"{'稠密 GEMM(全T²)':<20}{t_dense:>9.2f}"
        f"{2 * fl / (t_dense / 1e3) / 1e12:>8.1f}"
        f"{'—':>9}{'—':>8}   ← tensor core 地板"
    )


if __name__ == "__main__":
    run(112, 64, "目标 MLA")
    run(128, 64, "rope 48→64")
    run(128, 128, "等宽 128")
    run(64, 64, "等宽 64")
