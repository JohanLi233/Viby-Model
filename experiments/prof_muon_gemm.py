"""Muon 专家 NS 的原始件微基准：raw GEMM 上限 vs 现行实现的逐段耗时。

用法: .venv/bin/python experiments/prof_muon_gemm.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        r = fn()
        mx.eval(r)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    # 专家栈 NS 内部形状（tr 后短边在左）
    for (N, r, c), tag in [
        ((2304, 384, 640), "gu(2304,384,640)"),
        ((2304, 320, 384), "dw(2304,320,384)"),
    ]:
        X = mx.random.normal((N, r, c)).astype(mx.bfloat16)
        Q = mx.random.normal((N, r, r)).astype(mx.bfloat16)
        mx.eval(X, Q)

        t_gemm_xt = bench(lambda: X @ X.swapaxes(-1, -2))
        A = X @ X.swapaxes(-1, -2)
        mx.eval(A)
        t_gemm_aa = bench(lambda: A @ A)
        t_gemm_bx = bench(lambda: A @ X)
        t_qu = bench(lambda: Q @ X)

        fl_xt = N * r * r * c * 2 / 1e12
        fl_aa = N * r * r * r * 2 / 1e12
        fl_bx = N * r * c * r * 2 / 1e12
        fl_qu = fl_bx
        print(f"== {tag} ==")
        print(
            f"  X@Xᵀ  {t_gemm_xt:7.2f}ms  {fl_xt:.3f}TF  {fl_xt / t_gemm_xt * 1e3:6.1f} TFLOPS"
        )
        print(
            f"  A@A   {t_gemm_aa:7.2f}ms  {fl_aa:.3f}TF  {fl_aa / t_gemm_aa * 1e3:6.1f} TFLOPS"
        )
        print(
            f"  B@X   {t_gemm_bx:7.2f}ms  {fl_bx:.3f}TF  {fl_bx / t_gemm_bx * 1e3:6.1f} TFLOPS"
        )
        print(f"  Q@U   {t_qu:7.2f}ms  {fl_qu:.3f}TF  {fl_qu / t_qu * 1e3:6.1f} TFLOPS")
        ns_iter = t_gemm_xt + t_gemm_aa + t_gemm_bx
        print(f"  NS单迭代下限 {ns_iter:7.2f}ms  → NS5 {ns_iter * 5:7.2f}ms")

        # 逐元素/归约件
        P = mx.random.normal(
            (9, 256, 640 if r == 384 else 384, 384 if r == 384 else 320)
        ).astype(mx.bfloat16)
        mx.eval(P)

        def norm_f32():
            return mx.linalg.norm(P.astype(mx.float32), axis=(-2, -1), keepdims=True)

        def norm_sq_sum():
            # bf16 输入直接 square+sum（MLX 归约内部累加精度待验证）
            return mx.sum(mx.square(P), axis=(-2, -1), keepdims=True)

        t_norm32 = bench(norm_f32)
        t_normsq = bench(norm_sq_sum)
        gb = P.size * 2 / 1e9
        print(
            f"  norm(astype f32) {t_norm32:6.2f}ms  ({gb:.2f}GB → {gb / t_norm32 * 1e3:.0f}GB/s eff)"
        )
        print(f"  sum(square bf16) {t_normsq:6.2f}ms")

        # astype 全拷贝
        t_astype = bench(lambda: P.astype(mx.float32))
        print(f"  astype f32 拷贝  {t_astype:6.2f}ms")

        # mom 融合 kernel 参照
        from trainer.muon import _stack_apply_kernel, _stack_mom_kernel

        G = mx.random.normal(P.shape).astype(mx.bfloat16)
        V = mx.zeros(P.shape, dtype=mx.bfloat16)
        mx.eval(G, V)
        mom = _stack_mom_kernel(0.95, True, 0.0)
        app = _stack_apply_kernel(True)
        lr = mx.array(0.01, dtype=mx.bfloat16)
        t_mom = bench(lambda: mom(G, P, V))
        Xd = mx.random.normal(P.shape).astype(mx.bfloat16)
        mx.eval(Xd)
        t_app = bench(lambda: app(P, Xd, lr))
        print(f"  mom kernel      {t_mom:6.2f}ms")
        print(f"  apply(hyperball){t_app:6.2f}ms")


if __name__ == "__main__":
    main()
