"""AttnRes 合并 + GatedNorm eager 链 · 目标配置量化探针（B12 T1024 D768 bf16）。

背景：组件归因里 "other" 大头是 AttnRes 合并（12 层 ×2 次，第 i 层合并
N=2i+2 / 2i+3 个 v，B·T·D bf16 = 18.87MB/个；主栈合计 ΣN=324 次读/趟，
MTP 块再 +N=2,3）。融合 kernel fwd 读 v 两趟（score+mix），bwd 读 v 三趟
（rstd 复算 + da 点积 + 混合写出 dv_all）+ 写 N 份 dv。

本探针量化：
  A) 逐 N 的 merge fwd / fwd+bwd 耗时 → 拼出每微批 AttnRes 总成本；
     附 N=25 eager 参考（量化融合已拿到的收益）。
  B) GatedNorm 训练链（rms_norm → 两个 rank-128 GEMM → 2·sigmoid → 乘）
     eager vs mx.compile 的 fwd / fwd+bwd，×27 次/微批估算占比。

用法: .venv/bin/python experiments/probe_attnres_gatednorm.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_res_fused
from model.norms import GatedNorm

B, T, D = 12, 1024, 768
DT = mx.bfloat16
BYTES_V = B * T * D * 2  # 18.87MB


def _time(fn, reps=15, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def _mk_vs(n, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((D,)) * 0.25).astype(DT)
    vs = [(mx.random.normal((B, T, D)) * 0.5).astype(DT) for _ in range(n)]
    mx.eval(w, *vs)
    return w, vs


def probe_attn_res():
    print(
        f"== AttnRes 融合 merge 逐 N（B{B} T{T} D{D} bf16，单 v={BYTES_V / 1e6:.2f}MB）=="
    )
    print(f"{'N':>3} {'fwd_ms':>8} {'bwd_ms':>8} {'fwd_GB/s':>9} {'bwd_GB/s':>9}")
    tot_fwd = tot_bwd = 0.0
    seq = [2 * i + 2 for i in range(12)] + [2 * i + 3 for i in range(12)] + [2, 3]
    per_n = {}
    for n in sorted(set(seq)):
        w, vs = _mk_vs(n, seed=n)
        attn_res_fused.prewarm(D, DT, [n])

        def fwd():
            return attn_res_fused.merge(w, vs)

        vg = mx.value_and_grad(
            lambda w_, *v_: (
                attn_res_fused.merge(w_, list(v_)).astype(mx.float32) ** 2
            ).sum()
            / 2,
            argnums=range(n + 1),
        )

        def fwdbwd():
            return vg(w, *vs)

        f_min, f_med = _time(fwd)
        t_min, t_med = _time(fwdbwd)
        _b_min, b_med = max(t_min - f_min, 0.0), max(t_med - f_med, 0.0)
        per_n[n] = (f_med, b_med)
        fwd_gb = (2 * n + 1) * BYTES_V / 1e9
        bwd_gb = (4 * n + 3) * BYTES_V / 1e9  # v×3 + dv_all×1 + cot×2 + out
        print(
            f"{n:>3} {f_med * 1e3:>8.3f} {b_med * 1e3:>8.3f} "
            f"{fwd_gb / f_med:>9.0f} {bwd_gb / max(b_med, 1e-9):>9.0f}"
        )
    for n in seq:
        f, b = per_n[n]
        tot_fwd += f
        tot_bwd += b
    print(
        f"每微批 AttnRes 合计（主栈 24 次 + MTP 2 次，ΣN={sum(seq)}）："
        f"fwd {tot_fwd * 1e3:.1f}ms  bwd {tot_bwd * 1e3:.1f}ms"
    )

    # eager 参考（N=25，融合 vs eager 的单次对比）
    n = 25
    w, vs = _mk_vs(n, seed=999)
    f_min, f_med = _time(lambda: attn_res_fused.merge(w, vs))
    e_min, e_med = _time(lambda: attn_res_fused._merge_eager(w, vs))
    vg_f = mx.value_and_grad(
        lambda w_, *v_: (
            attn_res_fused.merge(w_, list(v_)).astype(mx.float32) ** 2
        ).sum()
        / 2,
        argnums=range(n + 1),
    )
    vg_e = mx.value_and_grad(
        lambda w_, *v_: (
            attn_res_fused._merge_eager(w_, list(v_)).astype(mx.float32) ** 2
        ).sum()
        / 2,
        argnums=range(n + 1),
    )
    tf_min, tf_med = _time(lambda: vg_f(w, *vs))
    te_min, te_med = _time(lambda: vg_e(w, *vs))
    print(
        f"N=25 参考：fwd 融合 {f_med * 1e3:.3f} vs eager {e_med * 1e3:.3f}ms；"
        f"fwd+bwd 融合 {tf_med * 1e3:.3f} vs eager {te_med * 1e3:.3f}ms"
    )


def probe_gated_norm():
    print(f"\n== GatedNorm 训练链（B{B} T{T} D{D} rank128 bf16）==")
    gn = GatedNorm(D, eps=1e-5, rank=128)
    from mlx.utils import tree_map

    gn.update(
        tree_map(
            lambda a: a.astype(DT) if mx.issubdtype(a.dtype, mx.floating) else a,
            gn.parameters(),
        )
    )
    gn.train()
    mx.random.seed(0)
    x = (mx.random.normal((B, T, D)) * 0.5).astype(DT)
    params = gn.parameters()

    # 先快照权重并做逐段分解：后面的 vg compile 会在 trace 期执行
    # gn.update(p)，把模块权重替换成 tracer，之后再用 gn.* 会炸。
    import mlx.nn as nn

    w_norm = mx.array(gn.norm.weight)
    wd = mx.array(gn.gate_down)
    wu = mx.array(gn.gate_up)
    y = mx.fast.rms_norm(x, w_norm, gn.norm.eps)
    h = y @ wd
    z = mx.sigmoid(nn.silu(h) @ wu) * 2.0
    mx.eval(y, h, z)

    segs = {
        "rms_norm": lambda: mx.fast.rms_norm(x, w_norm, gn.norm.eps),
        "y@Wd": lambda: y @ wd,
        "silu": lambda: nn.silu(h),
        "silu@Wu": lambda: nn.silu(h) @ wu,
        "2sig*y": lambda: y * (2.0 * mx.sigmoid(z)).astype(y.dtype),
    }
    for name, fn in segs.items():
        f, _ = _time(fn)
        print(f"  {name:<14} {f * 1e3:.3f}ms")

    def loss_fn(p, x_):
        gn.update(p)
        y = gn(x_)
        return (y.astype(mx.float32) ** 2).sum() / 2

    vg_eager = mx.value_and_grad(loss_fn)
    vg_comp = mx.compile(mx.value_and_grad(loss_fn))

    f_e, _ = _time(lambda: gn(x))
    f_c, _ = _time(lambda: mx.compile(lambda x_: gn(x_))(x))
    te_min, te_med = _time(lambda: vg_eager(params, x))
    tc_min, tc_med = _time(lambda: vg_comp(params, x))
    print(f"fwd       eager {f_e * 1e3:.3f}ms  compile {f_c * 1e3:.3f}ms")
    print(f"fwd+bwd   eager {te_med * 1e3:.3f}ms  compile {tc_med * 1e3:.3f}ms")
    n_calls = 12 * 2 + 1 + 1 + 2  # 12层×2 + final_norm + embed_norm + MTP块×2
    print(
        f"每微批 ×{n_calls}：compile fwd {f_c * n_calls * 1e3:.1f}ms  "
        f"fwd+bwd {tc_med * n_calls * 1e3:.1f}ms"
    )


if __name__ == "__main__":
    probe_attn_res()
    probe_gated_norm()
