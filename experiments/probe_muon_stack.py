"""BatchedMuon 堆叠组每步的「非 NS」开销剖析。

命中步（复用旧极因子 D）不需要任何 GEMM，理论上只剩逐元素的动量更新与
参数写回。但 apply_gradients 每步都要把 9 层专家权重 mx.stack 成
(N,b,r,c)（G/P/V 三份）再 scatter_back 切回去——纯内存搬运。本脚本量化：

  stack×3 / mom kernel / apply(hyperball) / scatter_back / _fro_norm

用法: uv run python experiments/probe_muon_stack.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import (
    _fro_norm_auto,
    _stack_apply_kernel,
    _stack_mom_kernel,
)

N = int(os.environ.get("VIBY_N_LAYERS", 9))  # 8 主干 + 1 MTP
E = int(os.environ.get("VIBY_BENCH_E", 256))
DE = int(os.environ.get("VIBY_BENCH_DE", 384))
I = int(os.environ.get("VIBY_BENCH_I", 384))  # noqa: E741
BW = 400e9


def timed(fn, it=6, w=2):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


for label, r, c in [("gate_up", 2 * I, DE), ("down", DE, I)]:
    nbytes = N * E * r * c * 2
    print(f"\n=== {label} ({N}×{E}×{r}×{c}) bf16 每份 {nbytes / 2**30:.2f}GB ===")
    G = [mx.random.normal((E, r, c)).astype(mx.bfloat16) for _ in range(N)]
    P = [mx.random.normal((E, r, c)).astype(mx.bfloat16) for _ in range(N)]
    V = [mx.random.normal((E, r, c)).astype(mx.bfloat16) for _ in range(N)]
    mx.eval(G, P, V)

    t_stack = timed(lambda: (mx.stack(G), mx.stack(P), mx.stack(V)))  # noqa: F821
    Gs, Ps, Vs = mx.stack(G), mx.stack(P), mx.stack(V)
    mx.eval(Gs, Ps, Vs)

    mom = _stack_mom_kernel(0.95, True, 0.0)
    t_mom = timed(lambda: mom(Gs, Ps, Vs))  # noqa: F821
    U, Vn = mom(Gs, Ps, Vs)
    mx.eval(U, Vn)

    Xd = mx.random.normal((N, E, r, c)).astype(mx.bfloat16)
    lr = mx.array(0.001, dtype=mx.bfloat16)
    mx.eval(Xd, lr)
    app_h = _stack_apply_kernel(True)
    app_n = _stack_apply_kernel(False)
    t_app_h = timed(lambda: app_h(Ps, Xd, lr))  # noqa: F821
    t_app_n = timed(lambda: app_n(Ps, Xd, lr))  # noqa: F821
    NP = app_h(Ps, Xd, lr)
    mx.eval(NP)

    def scatter_back():  # noqa: F821
        out = []
        for i in range(N):
            out.append(Vn[i].reshape(E, r, c))  # noqa: F821
            out.append(NP[i].reshape(E, r, c).astype(mx.bfloat16))  # noqa: F821
        return out

    t_scat = timed(scatter_back)
    t_fro = timed(lambda: _fro_norm_auto(Ps.reshape(N * E, r, c)))  # noqa: F821

    # 逐张量（不 stack）做同样的逐元素工作
    def per_tensor_mom():
        return [mom(G[i], P[i], V[i]) for i in range(N)]  # noqa: F821

    def per_tensor_apply():
        return [app_h(P[i], Xd[i], lr) for i in range(N)]  # noqa: F821

    t_pt_mom = timed(per_tensor_mom)
    t_pt_app = timed(per_tensor_apply)

    print(
        f"{'stack×3 (G/P/V)':<26}{t_stack:>8.2f}ms  "
        f"（下界 {3 * 2 * nbytes / BW * 1e3:.2f}ms）"
    )
    print(f"{'mom kernel (堆叠)':<26}{t_mom:>8.2f}ms")
    print(f"{'mom kernel (逐张量)':<26}{t_pt_mom:>8.2f}ms")
    print(f"{'apply hyperball (堆叠)':<26}{t_app_h:>8.2f}ms")
    print(f"{'apply hyperball (逐张量)':<26}{t_pt_app:>8.2f}ms")
    print(f"{'apply 无投影':<26}{t_app_n:>8.2f}ms")
    print(f"{'scatter_back 切片':<26}{t_scat:>8.2f}ms")
    print(f"{'_fro_norm 单次':<26}{t_fro:>8.2f}ms")
    print(f"{'命中步现状合计':<26}{t_stack + t_mom + t_app_h + t_scat:>8.2f}ms")
    print(f"{'命中步免 stack 合计':<26}{t_pt_mom + t_pt_app:>8.2f}ms")
    del G, P, V, Gs, Ps, Vs, U, Vn, Xd, NP
    mx.clear_cache()

print(f"\n峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
