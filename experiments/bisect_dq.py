"""dq kernel 成本 bisect：按组件开关合成 kernel，定位时间去向。

用法: uv run experiments/bisect_dq.py
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

B, H, T, DK, DV = 12, 8, 1024, 128, 96
NT, RES, STRB = 128, 32, 16
SCALE = 1.0 / (128**0.5)


def build(name, copy, smma, dp, ew, dq, bar, nt, strb):
    res = nt // 32 * 8
    qbs = T // res
    src = f"""
        #define DK {DK}
        #define DV {DV}
        #define TQ {T}
        #define TK {T}
        #define NH {H}
        #define NT {nt}
        #define RES {res}
        #define STR {strb}
        #define SK {DK + 8}
        #define SV {DV + 8}
        #define SF {strb + 4}
        #define SP {strb + 8}
        #define SCALE {SCALE}f
        #define LOG2E 1.4426950408889634f
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint qs  = qb * RES;

        threadgroup bfloat16_t Ks[STR * SK];
        threadgroup float Ss[RES * SF];
        threadgroup float DPs[RES * SF];
        threadgroup bfloat16_t Ps[RES * SP];
        threadgroup float AccHi[RES * 68];
        for (uint i = tid; i < RES * 68; i += NT) AccHi[i] = 0.0f;

        simdgroup_matrix<float, 8, 8> accLo[8];
        for (uint dc = 0; dc < 8; dc++)
            accLo[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<bfloat16_t, 8, 8> Qf[DK / 8];
        const device bfloat16_t* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
        for (uint d = 0; d < DK / 8; d++) simdgroup_load(Qf[d], qp + d * 8, DK);
        simdgroup_matrix<bfloat16_t, 8, 8> dOf[DV / 8];
        const device bfloat16_t* op0 = dout + ((size_t)bh * TQ + qs + sg * 8) * DV;
        for (uint e = 0; e < DV / 8; e++) simdgroup_load(dOf[e], op0 + e * 8, DV);

        float sink = 0.0f;
        uint nkb = (qs + RES + STR - 1) / STR;
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            {"threadgroup_barrier(mem_flags::mem_threadgroup);" if bar else ""}
            {"for (uint i = tid; i < STR * DK; i += NT) Ks[(i / DK) * SK + i % DK] = k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];" if copy else ""}
            {"threadgroup_barrier(mem_flags::mem_threadgroup);" if bar else ""}

            simdgroup_matrix<float, 8, 8> Sf[STR / 8];
            simdgroup_matrix<float, 8, 8> Pf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                Sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                Pf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            if ({"true" if smma else "false"}) {{
                for (uint d = 0; d < DK / 8; d++)
                    for (uint c = 0; c < STR / 8; c++) {{
                        simdgroup_matrix<bfloat16_t, 8, 8> Kf;
                        simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + d * 8, SK, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                    }}
            }}
            if ({"true" if dp else "false"}) {{
                const device bfloat16_t* vp = v + ((size_t)bh * TK + ks) * DV;
                for (uint e = 0; e < DV / 8; e++)
                    for (uint c = 0; c < STR / 8; c++) {{
                        simdgroup_matrix<bfloat16_t, 8, 8> Vf;
                        simdgroup_load(Vf, vp + (size_t)c * 8 * DV + e * 8, DV, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(Pf[c], dOf[e], Vf, Pf[c]);
                    }}
            }}
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(Sf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
                simdgroup_store(Pf[c], DPs + (size_t)sg * 8 * SF + c * 8, SF);
            }}
            {"threadgroup_barrier(mem_flags::mem_threadgroup);" if bar else ""}
            if ({"true" if ew else "false"}) {{
                for (uint t = tid; t < RES * STR; t += NT) {{
                    uint i = t / STR, j = t % STR;
                    uint gi = qs + i, gj = ks + j;
                    float s = Ss[i * SF + j] * SCALE;
                    s = (gj <= gi) ? s : -INFINITY;
                    Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (DPs[i * SF + j] - 0.5f));
                }}
            }}
            {"threadgroup_barrier(mem_flags::mem_threadgroup);" if bar else ""}
            if ({"true" if dq else "false"}) {{
                simdgroup_matrix<bfloat16_t, 8, 8> dSf[STR / 8];
                for (uint c = 0; c < STR / 8; c++)
                    simdgroup_load(dSf[c], Ps + (size_t)sg * 8 * SP + c * 8, SP);
                for (uint c = 0; c < STR / 8; c++)
                    for (uint dc = 0; dc < 8; dc++) {{
                        simdgroup_matrix<bfloat16_t, 8, 8> Kf;
                        simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + dc * 8, SK);
                        simdgroup_multiply_accumulate(accLo[dc], dSf[c], Kf, accLo[dc]);
                    }}
                for (uint h0 = 0; h0 < 8; h0 += 4) {{
                    simdgroup_matrix<float, 8, 8> accH[4];
                    for (uint h = 0; h < 4; h++)
                        simdgroup_load(accH[h], AccHi + (size_t)sg * 8 * 68 + (h0 + h) * 8, 68);
                    for (uint c = 0; c < STR / 8; c++)
                        for (uint h = 0; h < 4; h++) {{
                            simdgroup_matrix<bfloat16_t, 8, 8> Kf;
                            simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + (8 + h0 + h) * 8, SK);
                            simdgroup_multiply_accumulate(accH[h], dSf[c], Kf, accH[h]);
                        }}
                    for (uint h = 0; h < 4; h++)
                        simdgroup_store(accH[h], AccHi + (size_t)sg * 8 * 68 + (h0 + h) * 8, 68);
                }}
            }}
            {"if (tid == 0) { float z = 0; for (uint i = 0; i < RES * SF; i++) z += Ss[i] + DPs[i]; for (uint i = 0; i < RES * SP; i++) z += float(Ps[i]); sink += z; }" if ew else ""}
        }}
        threadgroup float sink_t[NT];
        sink_t[tid] = sink;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float z = 0.0f;
            for (uint i = 0; i < NT; i++) z += sink_t[i];
            out[bh * {qbs} + qb] = z;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"bisect_dq_{name}_{nt}_{strb}",
        input_names=["q", "k", "v", "dout"],
        output_names=["out"],
        source=src,
        header=HDR,
    )


def main():
    mx.random.seed(0)
    q = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.4).astype(mx.bfloat16)
    mx.eval(q, k, v, do)

    variants = [
        ("all", dict(copy=1, smma=1, dp=1, ew=1, dq=1, bar=1)),
        ("no_copy", dict(copy=0, smma=1, dp=1, ew=1, dq=1, bar=1)),
        ("no_S", dict(copy=1, smma=0, dp=1, ew=1, dq=1, bar=1)),
        ("no_dP", dict(copy=1, smma=1, dp=0, ew=1, dq=1, bar=1)),
        ("no_ew", dict(copy=1, smma=1, dp=1, ew=0, dq=1, bar=1)),
        ("no_dQ", dict(copy=1, smma=1, dp=1, ew=1, dq=0, bar=1)),
        ("copy_only", dict(copy=1, smma=0, dp=0, ew=0, dq=0, bar=1)),
        ("all_nobar", dict(copy=1, smma=1, dp=1, ew=1, dq=1, bar=0)),
        ("mma_only", dict(copy=0, smma=1, dp=1, ew=0, dq=1, bar=0)),
    ]
    kern = {n: build(n, nt=NT, strb=STRB, **f) for n, f in variants}
    qbs = T // RES
    outs = {n: None for n in kern}
    for n, kk in kern.items():
        outs[n] = kk(
            inputs=[q, k, v, do],
            output_shapes=[(B * H, qbs)],
            output_dtypes=[mx.float32],
            grid=(NT, qbs, B * H),
            threadgroup=(NT, 1, 1),
        )[0]
        mx.eval(outs[n])

    ss = {n: [] for n in kern}
    for rnd in range(8):
        for n, kk in kern.items():
            t0 = time.perf_counter()
            mx.eval(
                kk(
                    inputs=[q, k, v, do],
                    output_shapes=[(B * H, qbs)],
                    output_dtypes=[mx.float32],
                    grid=(NT, qbs, B * H),
                    threadgroup=(NT, 1, 1),
                )[0]
            )
            ss[n].append(time.perf_counter() - t0)
    print(f"{'variant':<26}{'ms':>9}")
    for n, _ in variants:
        t = statistics.median(ss[n][2:])
        print(f"  {n:<24}{t * 1e3:>9.2f}")


if __name__ == "__main__":
    main()
