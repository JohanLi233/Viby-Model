"""MoE 路由 GEMM 替代布局：目标形状下 gather_mm vs 稠密 batched vs 对齐填充。

目标：(E=384, DE=256, I=512, G=98304) 的 mm1+mm2。对照：
  1. 现状 gather_mm (G,1,DE)@(E,DE,2I)  sorted
  2. 同 FLOPs 均匀稠密 (E, C, DE)@(E, DE, 2I)，C=G/E=256
  3. 把 lhs 改成 (G, DE) 再 expand —— 确认 M=1 是否可避免
  4. 32 对齐填充后的 batched GEMM（模拟真实 max 段长）

用法: .venv/bin/python experiments/probe_moe_alt_gemm.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B, T, K = 12, 1024, 8
E, DE, I = 384, 256, 512  # noqa: E741
M, G = B * T, B * T * K
C = G // E
DT = mx.bfloat16


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    mx.random.seed(0)
    xs = (mx.random.normal((G, DE)) * 0.5).astype(DT)
    act = (mx.random.normal((G, I)) * 0.5).astype(DT)
    gu = (mx.random.normal((E, DE, 2 * I)) * 0.02).astype(DT)
    dw = (mx.random.normal((E, I, DE)) * 0.02).astype(DT)
    uni = mx.repeat(mx.arange(E, dtype=mx.int32), C)
    imb = mx.sort(mx.random.randint(0, E, (G,)).astype(mx.int32))
    mx.eval(xs, act, gu, dw, uni, imb)
    cnt = mx.zeros((E,), dtype=mx.float32).at[imb].add(1.0)
    mx.eval(cnt)
    cmin, cmax = int(cnt.min().item()), int(cnt.max().item())
    print(f"G={G} E={E} C_mean={C} DE={DE} I={I}  倾斜段长 {cmin}-{cmax}")

    F1 = 2 * G * DE * 2 * I
    F2 = 2 * G * I * DE

    def gm1(a, w, idx):
        return mx.gather_mm(a[:, None, :], w, None, idx, sorted_indices=True)

    def gm2(a, w, idx):
        return mx.gather_mm(a[:, None, :], w, None, idx, sorted_indices=True)

    print(f"\n{'口径':<36}{'fwd':>8}{'f+b':>8}{'TF_f':>8}")

    def report(label, fwd, flops, args, argnums=(0, 1)):
        cf = mx.compile(fwd)
        cv = mx.compile(mx.value_and_grad(lambda *x: fwd(*x).sum(), argnums=argnums))
        f = timed(lambda: cf(*args))
        fb = timed(lambda: cv(*args))
        print(f"{label:<36}{f:>8.2f}{fb:>8.2f}{flops / f / 1e9:>8.1f}")
        mx.clear_cache()
        return f, fb

    report("gmm1 uni (G,1,DE)", lambda a, w: gm1(a, w, uni).sum(), F1, (xs, gu))
    report("gmm1 imb (G,1,DE)", lambda a, w: gm1(a, w, imb).sum(), F1, (xs, gu))
    report("gmm2 uni (G,1,I)", lambda a, w: gm2(a, w, uni).sum(), F2, (act, dw))
    report("gmm2 imb (G,1,I)", lambda a, w: gm2(a, w, imb).sum(), F2, (act, dw))

    a1 = xs.reshape(E, C, DE)
    a2 = act.reshape(E, C, I)
    mx.eval(a1, a2)
    report("dense mm1 (E,C,DE)@(E,DE,2I)", lambda a, w: (a @ w).sum(), F1, (a1, gu))
    report("dense mm2 (E,C,I)@(E,I,DE)", lambda a, w: (a @ w).sum(), F2, (a2, dw))

    # 32 对齐填充：C_pad = ceil(cmax/32)*32，白算比例 = C_pad/C_mean
    cpad = ((cmax + 31) // 32) * 32
    xpad = (mx.random.normal((E, cpad, DE)) * 0.5).astype(DT)
    apad = (mx.random.normal((E, cpad, I)) * 0.5).astype(DT)
    mx.eval(xpad, apad)
    Fp1 = 2 * E * cpad * DE * 2 * I
    Fp2 = 2 * E * cpad * I * DE
    print(f"\n32 对齐填充 C_pad={cpad}  白算 {cpad / C:.2f}×")
    report(f"pad mm1 (E,{cpad},DE)", lambda a, w: (a @ w).sum(), Fp1, (xpad, gu))
    report(f"pad mm2 (E,{cpad},I)", lambda a, w: (a @ w).sum(), Fp2, (apad, dw))

    # 旧 docstring 形状 sanity
    print("\n== 旧形状 E256 DE384 I320 ==")
    E0, DE0, I0 = 256, 384, 320
    C0 = G // E0
    xs0 = (mx.random.normal((G, DE0)) * 0.5).astype(DT)
    gu0 = (mx.random.normal((E0, DE0, 2 * I0)) * 0.02).astype(DT)
    dw0 = (mx.random.normal((E0, I0, DE0)) * 0.02).astype(DT)
    uni0 = mx.repeat(mx.arange(E0, dtype=mx.int32), C0)
    a10 = xs0.reshape(E0, C0, DE0)
    mx.eval(xs0, gu0, dw0, uni0, a10)
    F10 = 2 * G * DE0 * 2 * I0
    F20 = 2 * G * I0 * DE0
    act0 = (mx.random.normal((G, I0)) * 0.5).astype(DT)
    a20 = act0.reshape(E0, C0, I0)
    mx.eval(act0, a20)
    report("old gmm1", lambda a, w: gm1(a, w, uni0).sum(), F10, (xs0, gu0))
    report("old dense mm1", lambda a, w: (a @ w).sum(), F10, (a10, gu0))
    report(
        "old gmm2",
        lambda a, w: gm2(a, w, uni0).sum(),
        F20,
        (act0, dw0),
    )
    report("old dense mm2", lambda a, w: (a @ w).sum(), F20, (a20, dw0))


if __name__ == "__main__":
    main()
