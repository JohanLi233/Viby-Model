"""elementwise 循环的二级 bisect：定位 exp 循环为什么这么贵。

用法: uv run experiments/bisect_ew.py
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


def build(name, ew_variant, nt, strb, use_mask=0):
    res = nt // 32 * 8
    qbs = T // res
    ew = {
        "full": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (DPs[i * SF + j] - 0.5f));
        """,
        "no_exp": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * (s + s) * (DPs[i * SF + j] - 0.5f));
        """,
        "clamp_exp": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            s = metal::max(s, -87.0f);
            Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (DPs[i * SF + j] - 0.5f));
        """,
        "precise_exp": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * metal::exp(s - 1.0f) * (DPs[i * SF + j] - 0.5f));
        """,
        "no_dp_read": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (float(i + j) - 0.5f));
        """,
        "no_ps_write": """
            float s = Ss[i * SF + j] * SCALE;
            s = (gj <= gi) ? s : -INFINITY;
            sink_l += SCALE * fast::exp2((s - 1.0f) * LOG2E) * (DPs[i * SF + j] - 0.5f);
        """,
        "no_tg_read": """
            float s = float(i + j);
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (float(i - j) - 0.5f));
        """,
        "alu_only": """
            Ps[i * SP + j] = bfloat16_t(float(i) * 0.5f + float(j));
        """,
        "mask_load": """
            float s = Ss[i * SF + j] * SCALE + float(maskb[((size_t)(bh / NH) * TQ + gi) * TK + gj]);
            s = (gj <= gi) ? s : -INFINITY;
            Ps[i * SP + j] = bfloat16_t(SCALE * fast::exp2((s - 1.0f) * LOG2E) * (DPs[i * SF + j] - 0.5f));
        """,
    }[ew_variant]
    loop = f"""
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Ks[(i / DK) * SK + i % DK] =
                    k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < RES * SF; i += NT) Ss[i] = float(i);
            for (uint i = tid; i < RES * SF; i += NT) DPs[i] = float(i * 3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sink_l = 0.0f;
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint i = t / STR, j = t % STR;
                uint gi = qs + i, gj = ks + j;
                {ew}
            }}
            sink += sink_l;
        }}
    """
    names = ["k"] + (["maskb"] if use_mask else [])
    "[k]" + (", maskb" if use_mask else "")
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
        #define SF {strb + 4}
        #define SP {strb + 8}
        #define SCALE {SCALE}f
        #define LOG2E 1.4426950408889634f
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint qs  = qb * RES;

        threadgroup bfloat16_t Ks[STR * SK];
        threadgroup float Ss[RES * SF];
        threadgroup float DPs[RES * SF];
        threadgroup bfloat16_t Ps[RES * SP];
        float sink = 0.0f;
        uint nkb = (qs + RES + STR - 1) / STR;
        {loop}
        threadgroup float sink_t[NT];
        sink_t[tid] = sink;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float z = 0.0f;
            for (uint i = 0; i < NT; i++) z += sink_t[i];
            z += Ps[0];
            out[bh * {qbs} + qb] = z;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"bisect_ew_{name}_{nt}_{strb}",
        input_names=names,
        output_names=["out"],
        source=src,
        header=HDR,
    )


def main():
    mx.random.seed(0)
    k = (mx.random.normal((B, H, T, DK)) * 0.4).astype(mx.bfloat16)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    maskb = mx.where(same[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(k, maskb)

    variants = [
        ("full", "full", 0),
        ("no_exp", "no_exp", 0),
        ("clamp_exp", "clamp_exp", 0),
        ("precise_exp", "precise_exp", 0),
        ("no_dp_read", "no_dp_read", 0),
        ("no_ps_write", "no_ps_write", 0),
        ("no_tg_read", "no_tg_read", 0),
        ("alu_only", "alu_only", 0),
        ("mask_load", "mask_load", 1),
    ]
    kern = {n: build(n, v, nt=NT, strb=STRB, use_mask=m) for n, v, m in variants}
    qbs = T // RES
    for n, kk in kern.items():
        ins = [k, maskb] if n == "mask_load" else [k]
        out = kk(
            inputs=ins,
            output_shapes=[(B * H, qbs)],
            output_dtypes=[mx.float32],
            grid=(NT, qbs, B * H),
            threadgroup=(NT, 1, 1),
        )[0]
        mx.eval(out)

    ss = {n: [] for n in kern}
    for rnd in range(8):
        for n, kk in kern.items():
            ins = [k, maskb] if n == "mask_load" else [k]
            t0 = time.perf_counter()
            mx.eval(
                kk(
                    inputs=ins,
                    output_shapes=[(B * H, qbs)],
                    output_dtypes=[mx.float32],
                    grid=(NT, qbs, B * H),
                    threadgroup=(NT, 1, 1),
                )[0]
            )
            ss[n].append(time.perf_counter() - t0)
    print(f"{'variant':<14}{'ms':>9}")
    for n, _, _ in variants:
        t = statistics.median(ss[n][2:])
        print(f"  {n:<12}{t * 1e3:>9.2f}")


if __name__ == "__main__":
    main()
