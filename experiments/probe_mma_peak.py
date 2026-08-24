"""量 simdgroup MMA 在 M4 Max 上的实际峰值，给 flash kernel 定天花板。

flash 反向手写 kernel 卡在 3.2 TFLOPS，加行距填充无效，分块扫描还出现
STR=16 反常凹陷。继续猜没有意义，先把「一条 simdgroup_multiply_accumulate
到底能跑多快」量出来：

A 臂：两个操作数都在寄存器里循环 MMA —— 纯算力上限。
B 臂：一个操作数每次从 threadgroup 载入 —— 真实 kernel 的形态。
C 臂：B 再叠加转置载入 —— flash 里算 QKᵀ 的形态。

A 和 B/C 的落差就是访存开销占比；A 本身若不高，说明这条路线到顶了。

用法: uv run experiments/probe_mma_peak.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

HDR = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

NT = 256
NSG = NT // 32
NACC = 4  # 累加器片段数
NA = 16  # A 片段数（相当于 DK/8）
ITER = 64
NGROUP = 2048  # threadgroup 数


def build(mode):
    """mode: reg / tg / tgT"""
    if mode == "reg":
        load_b = "Bf[d][c]"
        decl_b = f"""
        simdgroup_matrix<bfloat16_t, 8, 8> Bf[{NA}][{NACC}];
        for (uint d = 0; d < {NA}; d++)
            for (uint c = 0; c < {NACC}; c++)
                simdgroup_load(Bf[d][c], Bs + (size_t)c * 8 * SB + d * 8, SB);
        """
        inner = ""
    else:
        tr = ", ulong2(0, 0), true" if mode == "tgT" else ""
        decl_b = ""
        inner = f"""
                    simdgroup_matrix<bfloat16_t, 8, 8> Bf;
                    simdgroup_load(Bf, Bs + (size_t)c * 8 * SB + d * 8, SB{tr});
        """
        load_b = "Bf"
    src = f"""
        #define SB {128 + 8}
        uint tid = thread_position_in_grid.x;
        uint sg = tid / 32;
        threadgroup bfloat16_t As[32 * SB];
        threadgroup bfloat16_t Bs[32 * SB];
        for (uint i = tid; i < 32 * SB; i += {NT}) {{
            As[i] = bfloat16_t(x[i % 1024]);
            Bs[i] = bfloat16_t(x[(i * 7) % 1024]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> Af[{NA}];
        simdgroup_matrix<float, 8, 8> acc[{NACC}];
        for (uint d = 0; d < {NA}; d++)
            simdgroup_load(Af[d], As + (size_t)sg * 8 * SB + d * 8, SB);
        for (uint c = 0; c < {NACC}; c++)
            acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        {decl_b}
        for (uint it = 0; it < {ITER}; it++) {{
            for (uint d = 0; d < {NA}; d++) {{
                for (uint c = 0; c < {NACC}; c++) {{
                    {inner}
                    simdgroup_multiply_accumulate(acc[c], Af[d], {load_b}, acc[c]);
                }}
            }}
        }}
        threadgroup float outs[8 * {NACC} * 8];
        for (uint c = 0; c < {NACC}; c++)
            simdgroup_store(acc[c], outs + c * 8, {NACC} * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float s = 0.0f;
            for (uint i = 0; i < 8 * {NACC} * 8; i++) s += outs[i];
            out[thread_position_in_grid.z] = s;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mma_peak_{mode}",
        input_names=["x"],
        output_names=["out"],
        source=src,
        header=HDR,
    )


def main():
    x = mx.random.normal((1024,))
    mx.eval(x)
    flops = NGROUP * NSG * ITER * NA * NACC * 2 * 512
    print(
        f"每臂 {flops / 1e9:.1f} GFLOP（{NGROUP} threadgroup × {NSG} simdgroup"
        f" × {ITER}×{NA}×{NACC} 次 8×8×8 MMA）"
    )
    res = {}
    kerns = {m: build(m) for m in ("reg", "tg", "tgT")}
    samples = {m: [] for m in kerns}
    for rnd in range(6):
        for m, kern in kerns.items():
            t0 = time.perf_counter()
            mx.eval(
                kern(
                    inputs=[x],
                    output_shapes=[(NGROUP,)],
                    output_dtypes=[mx.float32],
                    grid=(NT, 1, NGROUP),
                    threadgroup=(NT, 1, 1),
                )[0]
            )
            samples[m].append(time.perf_counter() - t0)
    label = {
        "reg": "两操作数都在寄存器",
        "tg": "B 每次从 threadgroup 载入",
        "tgT": "B 从 threadgroup 转置载入",
    }
    for m in kerns:
        t = statistics.median(samples[m][2:])
        res[m] = flops / t / 1e12
        print(f"  {label[m]:<28}{t * 1e3:>8.2f}ms{res[m]:>8.1f} TFLOPS")
    print(
        f"\n  访存开销占比：普通载入 {1 - res['tg'] / res['reg']:.0%}"
        f"，转置载入 {1 - res['tgT'] / res['reg']:.0%}"
    )


if __name__ == "__main__":
    main()
