"""simdgroup MMA 操作数放置矩阵：哪些「A/B 来源」组合在 M4 Max 上真正快。

背景：probe_mma_peak 意外测出 A、B 都在寄存器时只有 1.3 TFLOPS，而 A 在
寄存器、B 从 threadgroup 载入是 10.4。flash 反向里的 MMA 形态更多样：
- dP = dO·Vᵀ：dO 常驻（device 载入寄存器）、V 每块从 device 载入
- dQ/dK/dV：dS/P 来自 threadgroup、K/Q 来自 threadgroup
- S = Q·Kᵀ：Q 常驻寄存器、K 从 threadgroup 转置载入

这里把 A×B 的来源矩阵扫一遍（reg / TG / device），决定 v2 kernel 每个
MMA 的操作数该怎么放。

用法: uv run experiments/probe_mma_operand.py
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
NA = 12  # A 片段数（dP 的 dO 侧 = DV/8=12）
NACC = 2  # B 片段数（STR/8=2）
ITER = 64
NGROUP = 1536  # threadgroup 数


def build(mode):
    """mode ∈ {rr, rT, TT, dd, dT, Td}：A 来源 × B 来源（r=reg 常驻,
    T=TG 每次载入, d=device 每次载入）。"""
    pa, pb = mode[0], mode[1]

    if pa == "r":
        decl_a = f"""
        simdgroup_matrix<bfloat16_t, 8, 8> Af[{NA}];
        for (uint d = 0; d < {NA}; d++)
            simdgroup_load(Af[d], As + (size_t)sg * 8 * SA + d * 8, SA);
        """
        load_a = "Af[d]"
    elif pa == "T":
        decl_a = ""
        load_a = "Af_"
    else:  # device
        decl_a = ""
        load_a = "Afd"

    if pb == "r":
        decl_b = f"""
        simdgroup_matrix<bfloat16_t, 8, 8> Bf[{NA}][{NACC}];
        for (uint d = 0; d < {NA}; d++)
            for (uint c = 0; c < {NACC}; c++)
                simdgroup_load(Bf[d][c], Bs + (size_t)c * 8 * SB + d * 8, SB);
        """
        load_b = "Bf[d][c]"
    elif pb == "T":
        decl_b = ""
        load_b = "Bf_"
    else:  # device
        decl_b = ""
        load_b = "Bfd"

    inner = []
    if pa == "T":
        inner.append(
            "simdgroup_matrix<bfloat16_t, 8, 8> Af_;"
            "simdgroup_load(Af_, As + (size_t)sg * 8 * SA + d * 8, SA);"
        )
    if pa == "d":
        inner.append(
            "simdgroup_matrix<bfloat16_t, 8, 8> Afd;"
            "simdgroup_load(Afd, x + (size_t)((d * 7 + it) % 512) * 64, 8);"
        )
    if pb == "T":
        inner.append(
            "simdgroup_matrix<bfloat16_t, 8, 8> Bf_;"
            "simdgroup_load(Bf_, Bs + (size_t)c * 8 * SB + d * 8, SB);"
        )
    if pb == "d":
        inner.append(
            "simdgroup_matrix<bfloat16_t, 8, 8> Bfd;"
            "simdgroup_load(Bfd, x + (size_t)((d * 13 + c * 5 + it) % 512) * 64, 8);"
        )
    inner.append(
        "simdgroup_multiply_accumulate(acc[c], " + load_a + ", " + load_b + ", acc[c]);"
    )
    inner_src = "\n".join("    " + line for line in inner)

    src = f"""
        #define SA 72
        #define SB 72
        uint tid = thread_position_in_grid.x;
        uint sg = tid / 32;
        threadgroup bfloat16_t As[64 * SA];
        threadgroup bfloat16_t Bs[64 * SB];
        for (uint i = tid; i < 64 * SA; i += {NT}) As[i] = bfloat16_t(x[i % 1024]);
        for (uint i = tid; i < 64 * SB; i += {NT}) Bs[i] = bfloat16_t(x[(i * 7) % 1024]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> acc[{NACC}];
        for (uint c = 0; c < {NACC}; c++)
            acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        {decl_a}
        {decl_b}
        for (uint it = 0; it < {ITER}; it++) {{
            for (uint d = 0; d < {NA}; d++) {{
                for (uint c = 0; c < {NACC}; c++) {{
{inner_src}
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
        name=f"mma_op_{mode}",
        input_names=["x"],
        output_names=["out"],
        source=src,
        header=HDR,
    )


def main():
    x = (mx.random.normal((4096,)) * 0.1).astype(mx.bfloat16)
    mx.eval(x)
    flops = NGROUP * NSG * ITER * NA * NACC * 2 * 512
    print(
        f"每臂 {flops / 1e9:.1f} GFLOP（{NGROUP} TG × {NSG} sg × {ITER}×{NA}×{NACC} MMA）"
    )
    label = {
        "rT": "A reg × B TG（probe_mma_peak 快臂）",
        "TT": "A TG × B TG",
        "dd": "A device × B device（= 现在 dP 的形态）",
        "dT": "A device × B TG",
        "Td": "A TG × B device",
    }
    kerns = {m: build(m) for m in label}
    samples = {m: [] for m in label}
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
    print(f"{'':<44}{'ms':>9}{'TFLOPS':>9}")
    for m in label:
        t = statistics.median(samples[m][2:])
        print(f"  {label[m]:<42}{t * 1e3:>9.2f}{flops / t / 1e12:>9.1f}")


if __name__ == "__main__":
    main()
