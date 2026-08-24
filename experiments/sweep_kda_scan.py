"""kda_scan 融合核的发射几何扫描（dv_split / nt_fwd / nt_bwd）。

probe_chunk_parts：scan f+b 15.01ms，访存下界 3.24ms（4.6×），fwd 有效
算力仅 ~1.1 TFLOPS。根因是发射几何而非算法——fwd 只有 B·H·dv_split=192
个线程组，每组 (D·DVH + C·DVH)·4 = 21.5KB 共享内存，超过每核 32KB 的
一半，每核只能驻留 1 组；bwd 更只有 B·H=96 组。

这三个参数只改变工作在线程/线程组间的分配，每个输出元素的累加顺序不变
⇒ 结果逐位等价。脚本对每个候选同时验证 fwd 输出与全部 7 个梯度与当前
默认几何逐位相同。

用法: uv run python experiments/sweep_kda_scan.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import kda_scan as ks

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H = 8
D = DV = 96
C = 16
NC = T // C

mx.random.seed(0)
qe = (mx.random.normal((B, H, NC, C, D)) * 0.05).astype(mx.float32)
w = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
u = (mx.random.normal((B, H, NC, C, DV)) * 0.3).astype(mx.float32)
Aqk = (mx.random.normal((B, H, NC, C, C)) * 0.05).astype(mx.float32)
kd = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
egl = mx.random.uniform(0.9, 1.0, (B, H, NC, D)).astype(mx.float32)
S0 = (mx.random.normal((B, H, D, DV)) * 0.1).astype(mx.float32)
INS = [qe, w, u, Aqk, kd, egl, S0]
cot_o = mx.random.normal((B, H, NC, C, DV)).astype(mx.float32)
mx.eval(INS, cot_o)


def timed(fn, it=8, w_=3):
    for _ in range(w_):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def make(dv_split, nt_fwd, nt_bwd):
    def fwd(*a):
        o, _ = ks.kda_scan_metal(*a, dv_split=dv_split, nt_fwd=nt_fwd, nt_bwd=nt_bwd)
        return o

    def loss(*a):
        # 只把 o 接进 loss：训练里最终状态在无 cache 时不被消费，
        # cot_Sall 恒为全零，与真实反向同构。
        return (fwd(*a) * cot_o).sum()

    return mx.compile(fwd), mx.compile(mx.value_and_grad(loss, argnums=tuple(range(7))))


DVH_OK = lambda dv, nt: DV % dv == 0 and (DV // dv) % (nt // C) == 0  # noqa: E731

print(f"B={B} H={H} D={D} DV={DV} C={C} NC={NC}")
print(f"当前默认 dv_split={ks._DV_SPLIT} nt_fwd={ks._NT_FWD} nt_bwd={ks._NT_BWD}\n")

ref_f, ref_vg = make(ks._DV_SPLIT, ks._NT_FWD, ks._NT_BWD)
ref_o = ref_f(*INS)
_, ref_g = ref_vg(*INS)
mx.eval(ref_o, ref_g)

print(
    f"{'dv':>4}{'nt_fwd':>8}{'nt_bwd':>8}{'TG共享KB':>10}{'#TG_f':>7}{'fwd':>8}{'f+b':>8}{'逐位':>6}"
)
rows = []
for dv in (1, 2, 3, 4, 6, 8):
    for ntf in (64, 128, 256, 512):
        if not DVH_OK(dv, ntf) or ntf % C:
            continue
        dvh = DV // dv
        tg_kb = (D * dvh + C * dvh) * 4 / 1024
        if tg_kb > 30:
            continue
        for ntb in (128, 256, 512, 1024):
            if ntb % C or DV % (ntb // C) or D % (ntb // C):
                continue
            try:
                f_, vg_ = make(dv, ntf, ntb)
                tf = timed(lambda: f_(*INS))
                tfb = timed(lambda: vg_(*INS))
                o = f_(*INS)
                _, g = vg_(*INS)
                mx.eval(o, g)
                exact = bool((o == ref_o).all().item()) and all(
                    bool((a == b).all().item()) for a, b in zip(g, ref_g)
                )
            except Exception as e:  # 编译/资源失败：跳过该候选
                print(f"{dv:>4}{ntf:>8}{ntb:>8}  失败 {type(e).__name__}: {e}")
                continue
            rows.append((tfb, tf, dv, ntf, ntb, exact))
            print(
                f"{dv:>4}{ntf:>8}{ntb:>8}{tg_kb:>10.1f}{B * H * dv:>7}"
                f"{tf:>8.2f}{tfb:>8.2f}{'是' if exact else '否':>6}"
            )
            mx.clear_cache()

rows.sort()
print("\n最优 5 组（按 f+b）")
for tfb, tf, dv, ntf, ntb, exact in rows[:5]:
    print(
        f"  dv={dv} nt_fwd={ntf} nt_bwd={ntb}: fwd {tf:.2f} f+b {tfb:.2f}ms"
        f"  逐位一致={exact}"
    )
base = [r for r in rows if (r[2], r[3], r[4]) == (ks._DV_SPLIT, ks._NT_FWD, ks._NT_BWD)]
if base and rows:
    print(
        f"\n当前默认 f+b {base[0][0]:.2f}ms → 最优 {rows[0][0]:.2f}ms "
        f"（{(1 - rows[0][0] / base[0][0]) * 100:.1f}% 更快）"
    )
