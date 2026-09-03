"""gather_mm 效率的形状矩阵：定位目标配置（E=384/K=8/latent=256/I=512）比
docstring 旧记录（E=256/K=8/latent=384/I=320，fwd 8.8ms）慢 68% 的原因。

两种嫌疑正交分解：
  a) latent 384→256 的瘦 GEMM（K 维变浅，tile 利用率掉）
  b) E 256→384 的分段变细（G 相同 ⇒ 段均长 384→256，段尾 tile 量化
     + 每段固定开销摊薄）

固定 M=12288/K=8（G=98304 不变），扫 (E, DE, I)：
  old      (256,384,320)  docstring 记录形状
  ablE     (384,384,320)  只变 E
  ablDE    (256,256,320)  只变 DE
  ablI     (256,384,512)  只变 I
  ablDEI   (256,256,512)  DE+I 合变（E 不动）
  new      (384,256,512)  目标形状

每个形状测 mm1/mm2 的 fwd（均匀索引 vs 伪随机倾斜索引）与 dA/dB 增量，
外加同 FLOPs 稠密 batched GEMM 上限。均匀≈倾斜 ⇒ 实现开销主导；
均匀明显快 ⇒ 段长量化/倾斜内在浪费主导。

用法: .venv/bin/python experiments/probe_moe_shape_matrix.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
K = 8
M, G = B * T, B * T * K

SHAPES = [
    ("old   (256,384,320)", 256, 384, 320),
    ("ablE  (384,384,320)", 384, 384, 320),
    ("ablDE (256,256,320)", 256, 256, 320),
    ("ablI  (256,384,512)", 256, 384, 512),
    ("ablDEI(256,256,512)", 256, 256, 512),
    ("new   (384,256,512)", 384, 256, 512),
]


def timed(fn, it=8, w=3):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def bench_shape(E, DE, I):  # noqa: E741
    mx.random.seed(0)
    xs = (mx.random.normal((G, DE)) * 0.5).astype(mx.bfloat16)
    act = (mx.random.normal((G, 1, I)) * 0.5).astype(mx.bfloat16)
    gu_t = (mx.random.normal((E, DE, 2 * I)) * 0.02).astype(mx.bfloat16)
    dw_t = (mx.random.normal((E, I, DE)) * 0.02).astype(mx.bfloat16)
    rows = G // E
    uni = mx.repeat(mx.arange(E, dtype=mx.int32), rows)
    # 伪随机倾斜（段长 min~0/max~3× 均值，模拟真实路由方差量级）
    imb = mx.sort(mx.random.randint(0, E, (G,)).astype(mx.int32))
    mx.eval(xs, act, gu_t, dw_t, uni, imb)
    cnt = mx.zeros((E,), dtype=mx.float32).at[imb].add(1.0)
    mx.eval(cnt)
    skew = f"{cnt.min().item():.0f}/{cnt.max().item():.0f}"

    F1 = 2 * G * DE * 2 * I
    F2 = 2 * G * I * DE

    def mm1(a, b, idx):
        return mx.gather_mm(a[:, None, :], b, None, idx, sorted_indices=True).sum()

    def mm2(a, b, idx):
        return mx.gather_mm(a, b, None, idx, sorted_indices=True).sum()

    res = {}
    for tag, idx in (("uni", uni), ("imb", imb)):
        for name, fn, a, b, fl in (
            ("mm1", mm1, xs, gu_t, F1),
            ("mm2", mm2, act, dw_t, F2),
        ):
            cf = mx.compile(lambda x, y: fn(x, y, idx))
            cda = mx.compile(
                mx.value_and_grad(lambda x, y: fn(x, y, idx), argnums=(0,))
            )
            cdb = mx.compile(
                mx.value_and_grad(lambda x, y: fn(x, y, idx), argnums=(1,))
            )
            f = timed(lambda: cf(a, b))
            da = timed(lambda: cda(a, b)) - f
            db = timed(lambda: cdb(a, b)) - f
            res[f"{name}_{tag}"] = (f, da, db, fl)

    # 稠密上限（无索引，同 FLOPs）
    a1 = (mx.random.normal((E, rows, DE)) * 0.5).astype(mx.bfloat16)
    a2 = (mx.random.normal((E, rows, I)) * 0.5).astype(mx.bfloat16)
    mx.eval(a1, a2)
    d1 = timed(lambda: a1 @ gu_t)
    d2 = timed(lambda: a2 @ dw_t)
    res["dense"] = (d1, d2)
    res["skew"] = skew
    return res


def main():
    print(f"M={M} K={K} G={G}（段均长=G/E）")
    hdr = f"{'形状':<20}{'段均长':>7}{'mm1 fwd':>9}{'dA':>7}{'dB':>7}{'TF_f':>6}"
    print("\n== mm1 (G,1,DE)@(E,DE,2I) / mm2 (G,1,I)@(E,I,DE)，均匀索引 ==")
    print(hdr)
    for label, E, DE, I in SHAPES:  # noqa: E741
        r = bench_shape(E, DE, I)
        f1, da1, db1, fl1 = r["mm1_uni"]
        f2, da2, db2, fl2 = r["mm2_uni"]
        fi1, dai1, dbi1, _ = r["mm1_imb"]
        fi2, dai2, dbi2, _ = r["mm2_imb"]
        d1, d2 = r["dense"]
        tot = f1 + f2
        print(
            f"{label:<20}{G // E:>7}{f1:>9.2f}{da1:>7.2f}{db1:>7.2f}"
            f"{fl1 / f1 / 1e9:>6.1f}   mm1"
        )
        print(
            f"{'':<20}{'':>7}{f2:>9.2f}{da2:>7.2f}{db2:>7.2f}"
            f"{fl2 / f2 / 1e9:>6.1f}   mm2 | 倾斜: mm1 {fi1:.2f}/{dai1:.2f}/{dbi1:.2f} "
            f"mm2 {fi2:.2f}/{dai2:.2f}/{dbi2:.2f} (段长 min/max {r['skew']})"
        )
        print(
            f"{'':<20}{'':>7}两GEMM fwd 合计 {tot:>6.2f}ms | 稠密上限 mm1 {d1:.2f} "
            f"mm2 {d2:.2f} 合计 {d1 + d2:.2f}ms"
        )
        mx.clear_cache()


if __name__ == "__main__":
    main()
