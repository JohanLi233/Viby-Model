"""AttnRes 残差链的真实成本（含 dv 跨 merge 求和），B12 T1024 D768 bf16。

probe_attnres_gatednorm 逐 N 单独测 merge，每个 v 是独立输入，梯度就是
kernel 的 dv 切片、不再相加。真实模型里 v_j 参与 j..23 全部 merge，
autodiff 要把 324 份 (B,T,D) cotangent 加起来，而 dv_all 的布局是
(B,T,N,D) —— 逐 j 切片沿倒数第二维跨步，加法走非连续路径。

本探针复刻 VibyStack 的残差流结构（24 次 merge，ΣN=324），测：
  A) 融合 kernel 现状
  B) eager 参考
并打印每档 N 的 dv 切片是否连续，给布局改造定量依据。

用法: .venv/bin/python experiments/probe_attnres_chain.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_res_fused

B, T, D = 12, 1024, 768
DT = mx.bfloat16
NSUB = 24  # 12 层 × 2 sublayer
BYTES_V = B * T * D * 2


def _med(fn, reps=7, warm=2):
    ts = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r)
        dt = time.perf_counter() - t0
        if i >= warm:
            ts.append(dt * 1e3)
    return statistics.median(ts)


def chain(merge_fn, x0, ws, gains):
    """复刻 VibyStack：每个 sublayer 产出一个 v 追加进共享列表再 merge。

    sublayer 用一次逐元素缩放替代（真实 attn/MoE 的成本单独归因），
    结构上保证 v_j 参与后续全部 merge、cotangent 要跨 merge 相加。
    """
    residuals = [x0]
    h = x0
    for i in range(NSUB):
        residuals.append(h * gains[i])
        h = merge_fn(ws[i], residuals)
    return h


def main():
    mx.random.seed(0)
    x0 = (mx.random.normal((B, T, D)) * 0.5).astype(DT)
    ws = [(mx.random.normal((D,)) * 0.25).astype(DT) for _ in range(NSUB)]
    gains = [mx.array(1.0 + 0.01 * i).astype(DT) for i in range(NSUB)]
    cot = (mx.random.normal((B, T, D)) * 0.5).astype(DT)
    mx.eval(x0, cot, *ws, *gains)

    for n in range(2, NSUB + 2):
        attn_res_fused.prewarm(D, DT, [n])

    sigma_n = sum(range(2, NSUB + 2))
    print(
        f"残差链：{NSUB} 次 merge，ΣN={sigma_n}，单 v={BYTES_V / 1e6:.1f}MB\n"
        f"  一趟读完全部历史 v = {sigma_n * BYTES_V / 1e9:.2f}GB"
    )

    for label, fn in (
        ("融合 kernel", attn_res_fused.merge),
        ("eager", attn_res_fused._merge_eager),
    ):

        def fwd():
            return chain(fn, x0, ws, gains)

        def loss(x_, *w_):
            h = chain(fn, x_, list(w_), gains)
            return (h.astype(mx.float32) * cot.astype(mx.float32)).sum()

        vg = mx.value_and_grad(loss, argnums=range(NSUB + 1))

        f = _med(fwd)
        fb = _med(lambda: vg(x0, *ws))
        print(
            f"  {label:<12} fwd {f:7.1f}ms  f+b {fb:7.1f}ms  bwd {fb - f:7.1f}ms"
            f"   fwd {2 * sigma_n * BYTES_V / 1e9 / (f / 1e3):.0f} GB/s"
        )

    # dv_all 切片的连续性：(B,T,N,D) 布局下 dv_all[:,:,j] 沿倒数第二维跨步
    dv = mx.zeros((B, T, 8, D), dtype=DT)
    mx.eval(dv)
    sl = dv[:, :, 3]
    print(
        f"\ndv_all (B,T,N,D) 的 dv_all[:,:,j]：shape={sl.shape} "
        f"contiguous={sl.flags['C_CONTIGUOUS'] if hasattr(sl, 'flags') else 'n/a'}"
    )
    dv2 = mx.zeros((8, B, T, D), dtype=DT)
    mx.eval(dv2)
    dv2[3]

    # 用一次加法链量化两种布局把 N 份切片加起来的差距
    def add_strided():
        s = dv[:, :, 0]
        for j in range(1, 8):
            s = s + dv[:, :, j]
        return s

    def add_contig():
        s = dv2[0]
        for j in range(1, 8):
            s = s + dv2[j]
        return s

    print(
        f"8 份 (B,T,D) 相加：跨步布局 {_med(add_strided):.2f}ms  "
        f"连续布局 {_med(add_contig):.2f}ms"
    )
    print(f"峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
