"""kda_scan 前向换 simdgroup MMA 的可行性 probe。

probe_chunk_parts（H=16 真实口径）：scan_zsc f+b 25.65ms（fwd 7.57 /
bwd 18.08），fwd 11.5 GFLOP / 7.57ms ≈ 1.5 TF/s，bwd 25.4 GFLOP /
18.08ms ≈ 1.4 TF/s——两者都卡在「标量 FMA + JB 寄存器分块」的天花板，
与几何无关（sweep_kda_scan H=16 重扫仅 2.9% 空间）。§5.5：MMA 两操作数
从 threadgroup 载入可达 11-12 TF/s。

本 probe 验证：跨 chunk 顺序扫描（NC=64 步循环 + 每步 ~6 个 barrier）
的结构下，chunk 步的三个矩阵乘（vt = u−w@S、o = qe@S+Aqk@vt、
S = egl⊙S + kdᵀ@vt）换成 simdgroup MMA 后的实际速率。

布局：每 threadgroup 负责 (bh, Dv 半片)，S 半片 96×48 f32 常驻 TG
（18KB，无行距填充——S 只做普通载入）；w/qe/kd 分时共用一块 16×(96+4)
staging（kd 要转置载入，+4 防 bank 冲突）；NT=192 = 6 个 simdgroup，
每个 SG 拥有一个 8 列 N tile。

用法: uv run python experiments/probe_kda_scan_mma.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import _kda_scan
from model.kernels.kda_scan import kda_scan_metal

B = int(os.environ.get("VIBY_BENCH_B", 12))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
H = int(os.environ.get("VIBY_BENCH_H", 16))
C = 16
D = DV = 96
DVH = DV // 2
NC = T // C
NT = 192  # 6 simdgroup = DVH/8 个 N tile

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


def _build_fwd_mma():
    SD = D + 4  # kd 转置载入的行距（100 mod 32 = 4，8 行错开不撞 bank）
    src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        uint dvh = thread_position_in_grid.z;
        uint sg  = tid / 32;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint DVH = {DVH};
        constexpr uint NT = {NT};
        constexpr uint NC = {NC};
        constexpr uint SD = {SD};
        uint j0 = dvh * DVH;
        size_t cb = (size_t)bh * NC;
        size_t sab = (size_t)bh * (NC + 1) * D * DV;

        threadgroup float S[D * DVH];      // 18KB
        threadgroup float vt[C * DVH];     // 3KB
        threadgroup float Wst[C * SD];     // 6.25KB，w/qe/kd 分时复用
        threadgroup float egls[D];

        for (uint i = tid; i < D * DVH; i += NT)
            S[i] = S0[(size_t)bh * D * DV + (i / DVH) * DV + j0 + (i % DVH)];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < NC; c++) {{
            size_t cbh = cb + c;
            const device float* qe_c  = qe  + (size_t)cbh * C * D;
            const device float* w_c   = w   + (size_t)cbh * C * D;
            const device float* u_c   = u   + (size_t)cbh * C * DV;
            const device float* kd_c  = kd  + (size_t)cbh * C * D;
            const device float* Aqk_c = Aqk + (size_t)cbh * C * C;
            device float* o_c         = o   + (size_t)cbh * C * DV;

            // Sall[c] = 入态；Wst ← −w；egl 装载
            for (uint i = tid; i < D * DVH; i += NT)
                Sall[sab + ((size_t)c * D + i / DVH) * DV + j0 + (i % DVH)] = S[i];
            for (uint i = tid; i < C * D; i += NT)
                Wst[(i / D) * SD + (i % D)] = -w_c[i];
            if (tid < D) egls[tid] = egl[cbh * D + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P1: vt = u + (−w)@S（acc 以 u tile 初始化）
            {{
                simdgroup_matrix<float, 8, 8> acc0, acc1, Af, Bf;
                simdgroup_load(acc0, u_c + 0 * DV + j0 + sg * 8, DV);
                simdgroup_load(acc1, u_c + 8 * DV + j0 + sg * 8, DV);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + sg * 8, DVH);
                    simdgroup_load(Af, Wst + 0 * SD + kk * 8, SD);
                    simdgroup_multiply_accumulate(acc0, Af, Bf, acc0);
                    simdgroup_load(Af, Wst + 8 * SD + kk * 8, SD);
                    simdgroup_multiply_accumulate(acc1, Af, Bf, acc1);
                }}
                simdgroup_store(acc0, vt + 0 * DVH + sg * 8, DVH);
                simdgroup_store(acc1, vt + 8 * DVH + sg * 8, DVH);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < C * D; i += NT)
                Wst[(i / D) * SD + (i % D)] = qe_c[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P2: o = qe@S + Aqk@vt（Aqk 小块 device 直读作 A）
            {{
                simdgroup_matrix<float, 8, 8> acc0, acc1, Af, Bf;
                acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + sg * 8, DVH);
                    simdgroup_load(Af, Wst + 0 * SD + kk * 8, SD);
                    simdgroup_multiply_accumulate(acc0, Af, Bf, acc0);
                    simdgroup_load(Af, Wst + 8 * SD + kk * 8, SD);
                    simdgroup_multiply_accumulate(acc1, Af, Bf, acc1);
                }}
                for (uint kk = 0; kk < C / 8; kk++) {{
                    simdgroup_load(Bf, vt + kk * 8 * DVH + sg * 8, DVH);
                    simdgroup_load(Af, Aqk_c + 0 * C + kk * 8, C);
                    simdgroup_multiply_accumulate(acc0, Af, Bf, acc0);
                    simdgroup_load(Af, Aqk_c + 8 * C + kk * 8, C);
                    simdgroup_multiply_accumulate(acc1, Af, Bf, acc1);
                }}
                simdgroup_store(acc0, o_c + 0 * DV + j0 + sg * 8, DV);
                simdgroup_store(acc1, o_c + 8 * DV + j0 + sg * 8, DV);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Wst ← kd（P3 转置载入）；S ⊙= egl（每 2 线程一行，免除法）
            for (uint i = tid; i < C * D; i += NT)
                Wst[(i / D) * SD + (i % D)] = kd_c[i];
            {{
                uint d = tid >> 1;
                float ev = egls[d];
                uint jb = (tid & 1) * (DVH / 2);
                for (uint j = 0; j < DVH / 2; j++)
                    S[d * DVH + jb + j] *= ev;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3: S += kdᵀ@vt（acc 以 S tile 初始化，逐 M tile 滚动）
            {{
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                for (uint mm = 0; mm < D / 8; mm++) {{
                    simdgroup_load(acc, S + mm * 8 * DVH + sg * 8, DVH);
                    for (uint kk = 0; kk < C / 8; kk++) {{
                        simdgroup_load(Af, Wst + kk * 8 * SD + mm * 8, SD,
                                       ulong2(0, 0), true);
                        simdgroup_load(Bf, vt + kk * 8 * DVH + sg * 8, DVH);
                        simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                    }}
                    simdgroup_store(acc, S + mm * 8 * DVH + sg * 8, DVH);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint i = tid; i < D * DVH; i += NT)
            Sall[sab + ((size_t)NC * D + i / DVH) * DV + j0 + (i % DVH)] = S[i];
    """
    return mx.fast.metal_kernel(
        name=f"probe_kda_scan_mma_fwd_{NC}_{C}_{D}_{DV}_{NT}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall"],
        source=src,
        header=_HEADER,
    )


def _build_fwd_mma_v2(NT2: int = 384, stage: bool = True):
    """V2：NT=384 → 12 SG。P1/P2 每 SG 一个 (M,N) tile 组合（2×6=12），
    P3 每 SG 一个 M tile（12）。Sall[c+1] 由 P3 的 acc 直写 device，
    省掉下一 chunk 头部的 TG 重读。stage=False 时 A 操作数 device 直读
    （省 2 个 barrier 和 staging 拷贝，代价是 A 载入更慢）。"""
    SD = D + 4
    NSG = NT2 // 32
    assert NSG == 12
    stage_w = f"""
            for (uint i = tid; i < C * D; i += {NT2})
                Wst[(i / D) * SD + (i % D)] = -w_c[i];
    """
    tg_decl = f"threadgroup float Wst[C * {SD}];"
    barrier_p1 = "threadgroup_barrier(mem_flags::mem_threadgroup);"
    a_p1 = "simdgroup_load(Af, Wst + mrow * 8 * SD + kk * 8, SD);"
    if stage:
        stage_qe = f"""
            for (uint i = tid; i < C * D; i += {NT2})
                Wst[(i / D) * SD + (i % D)] = qe_c[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        stage_kd = f"""
            for (uint i = tid; i < C * D; i += {NT2})
                Wst[(i / D) * SD + (i % D)] = kd_c[i];
        """
        a_p2 = "simdgroup_load(Af, Wst + mrow * 8 * SD + kk * 8, SD);"
        a_p3_hoist = """
                simdgroup_load(A0, Wst + 0 * SD + sg * 8, SD, ulong2(0, 0), true);
                simdgroup_load(A1, Wst + 8 * SD + sg * 8, SD, ulong2(0, 0), true);
        """
    else:
        stage_qe = ""
        stage_kd = ""
        a_p2 = "simdgroup_load(Af, qe_c + mrow * 8 * D + kk * 8, D);"
        a_p3_hoist = """
                simdgroup_load(A0, kd_c + 0 * D + sg * 8, D, ulong2(0, 0), true);
                simdgroup_load(A1, kd_c + 8 * D + sg * 8, D, ulong2(0, 0), true);
        """
    src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        uint dvh = thread_position_in_grid.z;
        uint sg  = tid / 32;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint DVH = {DVH};
        constexpr uint NT = {NT2};
        constexpr uint NC = {NC};
        constexpr uint SD = {SD};
        uint j0 = dvh * DVH;
        size_t cb = (size_t)bh * NC;
        size_t sab = (size_t)bh * (NC + 1) * D * DV;

        threadgroup float S[D * DVH];
        threadgroup float vt[C * DVH];
        {tg_decl}
        threadgroup float egls[D];

        for (uint i = tid; i < D * DVH; i += NT) {{
            float sv = S0[(size_t)bh * D * DV + (i / DVH) * DV + j0 + (i % DVH)];
            S[i] = sv;
            Sall[sab + (size_t)(i / DVH) * DV + j0 + (i % DVH)] = sv;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < NC; c++) {{
            size_t cbh = cb + c;
            const device float* qe_c  = qe  + (size_t)cbh * C * D;
            const device float* w_c   = w   + (size_t)cbh * C * D;
            const device float* u_c   = u   + (size_t)cbh * C * DV;
            const device float* kd_c  = kd  + (size_t)cbh * C * D;
            const device float* Aqk_c = Aqk + (size_t)cbh * C * C;
            device float* o_c         = o   + (size_t)cbh * C * DV;
            device float* sall_c      = Sall + sab + ((size_t)c + 1) * D * DV;

            {stage_w}
            if (tid < D) egls[tid] = egl[cbh * D + tid];
            {barrier_p1}

            // P1: vt = u + (−w)@S，SG → (mrow, ncol)
            {{
                uint mrow = sg & 1, ncol = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                simdgroup_load(acc, u_c + mrow * 8 * DV + j0 + ncol * 8, DV);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + ncol * 8, DVH);
                    {a_p1}
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, vt + mrow * 8 * DVH + ncol * 8, DVH);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {stage_qe}
            // P2: o = qe@S + Aqk@vt
            {{
                uint mrow = sg & 1, ncol = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + ncol * 8, DVH);
                    {a_p2}
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                for (uint kk = 0; kk < C / 8; kk++) {{
                    simdgroup_load(Bf, vt + kk * 8 * DVH + ncol * 8, DVH);
                    simdgroup_load(Af, Aqk_c + mrow * 8 * C + kk * 8, C);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, o_c + mrow * 8 * DV + j0 + ncol * 8, DV);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {stage_kd}
            // S ⊙= egl（4 线程一行）
            {{
                uint d = tid >> 2;
                float ev = egls[d];
                uint jb = (tid & 3) * (DVH / 4);
                for (uint j = 0; j < DVH / 4; j++)
                    S[d * DVH + jb + j] *= ev;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3: S += kdᵀ@vt，SG → M tile；A tile 每 SG 固定，hoist 出
            // N 循环；acc 直写 S 与 Sall[c+1]
            {{
                simdgroup_matrix<float, 8, 8> acc, A0, A1, Bf;
{a_p3_hoist}
                for (uint nn = 0; nn < DVH / 8; nn++) {{
                    simdgroup_load(acc, S + sg * 8 * DVH + nn * 8, DVH);
                    simdgroup_load(Bf, vt + 0 * DVH + nn * 8, DVH);
                    simdgroup_multiply_accumulate(acc, A0, Bf, acc);
                    simdgroup_load(Bf, vt + 8 * DVH + nn * 8, DVH);
                    simdgroup_multiply_accumulate(acc, A1, Bf, acc);
                    simdgroup_store(acc, S + sg * 8 * DVH + nn * 8, DVH);
                    simdgroup_store(acc, sall_c + (size_t)sg * 8 * DV + j0 + nn * 8, DV);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"probe_kda_scan_mma_fwd_v2_{NC}_{C}_{D}_{DV}_{NT2}_{int(stage)}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall"],
        source=src,
        header=_HEADER,
    )


def _build_bwd_mma(NT2: int = 384):
    """bwd MMA（ZSC 变体）：每 TG 一个 (bh, Dv 半片)，dS 半片常驻 TG。

    递推链只需 dS/dvt（j 可分），j 全维归约的 dqe/dw hoist 到 MLX batched
    GEMM（dqe = do@Scᵀ、dw = −du@Scᵀ，都不依赖递推）；dAqk/dkd/degl 按
    半片写部分和 (B,H,2,NC,...)，MLX 侧求和。dS 的 device scratch 整个
    消失（旧版每 chunk 从 global 读写 dSw 多次）。"""
    SW = DVH + 4  # 52 mod 32 = 20：转置载入 dSᵀ/vtᵀ 的 8 行互不撞 bank
    src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        uint dvh = thread_position_in_grid.z;
        uint sg  = tid / 32;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint DVH = {DVH};
        constexpr uint NT = {NT2};
        constexpr uint NC = {NC};
        constexpr uint SW = {SW};
        uint j0 = dvh * DVH;
        size_t cb = (size_t)bh * NC;
        size_t sab = (size_t)bh * (NC + 1) * D * DV;
        size_t pb2 = ((size_t)bh * 2 + dvh) * NC;   // 部分和的 (bh, half) 基址

        threadgroup float dS[D * SW];     // 19968B
        threadgroup float vt[C * SW];     // 3328B（P2c 转置载入）
        threadgroup float ndvt[C * DVH];  // 3072B
        threadgroup float egls[D];

        // ZSC：dS 初始为零（cot_Sall 恒零）
        for (uint i = tid; i < D * DVH; i += NT)
            dS[(i / DVH) * SW + (i % DVH)] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int c = NC - 1; c >= 0; c--) {{
            size_t cbh = cb + c;
            const device float* qe_c  = qe  + (size_t)cbh * C * D;
            const device float* w_c   = w   + (size_t)cbh * C * D;
            const device float* u_c   = u   + (size_t)cbh * C * DV;
            const device float* kd_c  = kd  + (size_t)cbh * C * D;
            const device float* Aqk_c = Aqk + (size_t)cbh * C * C;
            const device float* do_c  = cot_o + (size_t)cbh * C * DV;
            const device float* sc    = Sall + sab + (size_t)c * D * DV;

            // P1a: vt ← w@Sc（MMA，SG → (m,n)）
            {{
                uint m = sg & 1, n = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Af, w_c + m * 8 * D + kk * 8, D);
                    simdgroup_load(Bf, sc + kk * 8 * DV + j0 + n * 8, DV);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, vt + m * 8 * SW + n * 8, SW);
            }}
            if (tid < D) egls[tid] = egl[cbh * D + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P1b: vt = u − vt
            for (uint i = tid; i < C * DVH; i += NT) {{
                uint r = i / DVH, col = i % DVH;
                vt[r * SW + col] = u_c[r * DV + j0 + col] - vt[r * SW + col];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P2a: dvt = Aqkᵀ@do + kd@dS → ndvt（暂存正值）
            {{
                uint m = sg & 1, n = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < C / 8; kk++) {{
                    simdgroup_load(Af, Aqk_c + kk * 8 * C + m * 8, C,
                                   ulong2(0, 0), true);
                    simdgroup_load(Bf, do_c + kk * 8 * DV + j0 + n * 8, DV);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Af, kd_c + m * 8 * D + kk * 8, D);
                    simdgroup_load(Bf, dS + kk * 8 * SW + n * 8, SW);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, ndvt + m * 8 * DVH + n * 8, DVH);
            }}
            // P2c: dAqk_p = do@vtᵀ（4 个 tile，sg<4）
            if (sg < 4) {{
                uint m = sg & 1, l = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < DVH / 8; kk++) {{
                    simdgroup_load(Af, do_c + m * 8 * DV + j0 + kk * 8, DV);
                    simdgroup_load(Bf, vt + l * 8 * SW + kk * 8, SW,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, dAqk_p + (pb2 + c) * C * C + m * 8 * C + l * 8, C);
            }}
            // P2d: dkd_p = vt@dSᵀ（M=2 × N=12，每 SG 两个 (m, nn=sg)）
            {{
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                for (uint m = 0; m < 2; m++) {{
                    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (uint kk = 0; kk < DVH / 8; kk++) {{
                        simdgroup_load(Af, vt + m * 8 * SW + kk * 8, SW);
                        simdgroup_load(Bf, dS + sg * 8 * SW + kk * 8, SW,
                                       ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                    }}
                    simdgroup_store(acc, dkd_p + (pb2 + c) * C * D + m * 8 * D + sg * 8, D);
                }}
            }}
            // P2e: degl_p[d] = Σ_j Sc[d,j]·dS[d,j]（半片）
            if (tid < D) {{
                float acc = 0.0f;
                for (uint j = 0; j < DVH; j++)
                    acc += sc[tid * DV + j0 + j] * dS[tid * SW + j];
                degl_p[(pb2 + c) * D + tid] = acc;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // EL: du 写出 + dvt 取负 + dS ⊙= egl
            for (uint i = tid; i < C * DVH; i += NT) {{
                uint r = i / DVH, col = i % DVH;
                float v = ndvt[i];
                du[cbh * C * DV + r * DV + j0 + col] = v;
                ndvt[i] = -v;
            }}
            {{
                uint d = tid >> 2;
                float ev = egls[d];
                uint jb = (tid & 3) * (DVH / 4);
                for (uint j = 0; j < DVH / 4; j++)
                    dS[d * SW + jb + j] *= ev;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3a: dS += qeᵀ@do + wᵀ@ndvt（SG → M tile，A 转置 device hoist）
            {{
                simdgroup_matrix<float, 8, 8> acc, A0, A1, A2, A3, Bf;
                simdgroup_load(A0, qe_c + 0 * D + sg * 8, D, ulong2(0, 0), true);
                simdgroup_load(A1, qe_c + 8 * D + sg * 8, D, ulong2(0, 0), true);
                simdgroup_load(A2, w_c + 0 * D + sg * 8, D, ulong2(0, 0), true);
                simdgroup_load(A3, w_c + 8 * D + sg * 8, D, ulong2(0, 0), true);
                for (uint nn = 0; nn < DVH / 8; nn++) {{
                    simdgroup_load(acc, dS + sg * 8 * SW + nn * 8, SW);
                    simdgroup_load(Bf, do_c + 0 * DV + j0 + nn * 8, DV);
                    simdgroup_multiply_accumulate(acc, A0, Bf, acc);
                    simdgroup_load(Bf, do_c + 8 * DV + j0 + nn * 8, DV);
                    simdgroup_multiply_accumulate(acc, A1, Bf, acc);
                    simdgroup_load(Bf, ndvt + 0 * DVH + nn * 8, DVH);
                    simdgroup_multiply_accumulate(acc, A2, Bf, acc);
                    simdgroup_load(Bf, ndvt + 8 * DVH + nn * 8, DVH);
                    simdgroup_multiply_accumulate(acc, A3, Bf, acc);
                    simdgroup_store(acc, dS + sg * 8 * SW + nn * 8, SW);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        // dS0（ZSC：= dS）
        for (uint i = tid; i < D * DVH; i += NT)
            dS0[(size_t)bh * D * DV + (i / DVH) * DV + j0 + (i % DVH)] =
                dS[(i / DVH) * SW + (i % DVH)];
    """
    return mx.fast.metal_kernel(
        name=f"probe_kda_scan_mma_bwd_{NC}_{C}_{D}_{DV}_{NT2}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "Sall", "cot_o"],
        output_names=["du", "dAqk_p", "dkd_p", "degl_p", "dS0"],
        source=src,
        header=_HEADER,
    )


def make_mma_op():
    """fwd v3 + bwd MMA 的 custom_function（ZSC 语义：loss 只依赖 o）。"""
    k_fwd = _build_fwd_mma_v2(384, False)
    k_bwd = _build_bwd_mma(384)

    @mx.custom_function
    def _op(qe, w, u, Aqk, kd, egl, S0):
        Bx, Hx = qe.shape[0], qe.shape[1]
        o, Sall = k_fwd(
            inputs=[qe, w, u, Aqk, kd, egl, S0],
            output_shapes=[(Bx, Hx, NC, C, DV), (Bx, Hx, NC + 1, D, DV)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(384, Bx * Hx, 2),
            threadgroup=(384, 1, 1),
        )
        return o, Sall

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, w, u, Aqk, kd, egl, _S0 = primals
        cot_o = cotangent[0]
        Sall = output[1]
        Bx, Hx = qe.shape[0], qe.shape[1]
        du, dAqk_p, dkd_p, degl_p, dS0 = k_bwd(
            inputs=[qe, w, u, Aqk, kd, egl, Sall, cot_o],
            output_shapes=[
                (Bx, Hx, NC, C, DV),
                (Bx, Hx, 2, NC, C, C),
                (Bx, Hx, 2, NC, C, D),
                (Bx, Hx, 2, NC, D),
                (Bx, Hx, D, DV),
            ],
            output_dtypes=[mx.float32] * 5,
            grid=(384, Bx * Hx, 2),
            threadgroup=(384, 1, 1),
        )
        # dqe/dw 共享一次 Sall 流读：拼接后单 GEMM（Sall 439MB，分开算要读两遍）
        ScT = mx.swapaxes(Sall[:, :, :NC], -1, -2)
        dodu = mx.concatenate([cot_o, du], axis=3) @ ScT
        dqe = dodu[..., :C, :]
        dw = -dodu[..., C:, :]
        dAqk = dAqk_p[:, :, 0] + dAqk_p[:, :, 1]
        dkd = dkd_p[:, :, 0] + dkd_p[:, :, 1]
        degl = degl_p[:, :, 0] + degl_p[:, :, 1]
        return [dqe, dw, du, dAqk, dkd, degl, dS0]

    return _op


def timed(fn, it=10, w_=3):
    for _ in range(w_):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    mx.random.seed(0)
    qe = (mx.random.normal((B, H, NC, C, D)) * 0.05).astype(mx.float32)
    w = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
    u = (mx.random.normal((B, H, NC, C, DV)) * 0.3).astype(mx.float32)
    Aqk = (mx.random.normal((B, H, NC, C, C)) * 0.05).astype(mx.float32)
    kd = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
    egl = mx.random.uniform(0.9, 1.0, (B, H, NC, D)).astype(mx.float32)
    S0 = (mx.random.normal((B, H, D, DV)) * 0.1).astype(mx.float32)
    ins = [qe, w, u, Aqk, kd, egl, S0]
    mx.eval(ins)

    def runner(kern, nt):
        def fn():
            o, Sall = kern(
                inputs=ins,
                output_shapes=[(B, H, NC, C, DV), (B, H, NC + 1, D, DV)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, B * H, 2),
                threadgroup=(nt, 1, 1),
            )
            return o, Sall

        return fn

    variants = [
        ("MMA v1 NT=192 全stage", runner(_build_fwd_mma(), NT)),
        ("MMA v2 NT=384 全stage", runner(_build_fwd_mma_v2(384, True), 384)),
        ("MMA v3 NT=384 只stage w", runner(_build_fwd_mma_v2(384, False), 384)),
    ]

    # 正确性：对照 eager 参考（相对误差，状态链口径 §5.9）
    o_r, sall_r = _kda_scan(*ins)
    mx.eval(o_r, sall_r)
    for name, fn in variants:
        o_m, sall_m = fn()
        mx.eval(o_m, sall_m)
        d_o = ((o_m - o_r).abs().max() / (o_r.abs().max() + 1e-12)).item()
        d_s = ((sall_m - sall_r).abs().max() / (sall_r.abs().max() + 1e-12)).item()
        status = "OK" if max(d_o, d_s) <= 1e-4 else "FAIL"
        print(f"{name:<22} rel o {d_o:.2e}  Sall {d_s:.2e}  {status}")

    # 同进程交替 A/B（fwd only），两轮抵热漂移
    gf = 2 * B * H * NC * (3 * C * D * DV + C * C * DV) / 1e9
    arms = [("标量(默认几何)", lambda: kda_scan_metal(*ins)[0])] + [
        (n, (lambda f=f: f()[0])) for n, f in variants
    ]
    for rnd in range(2):
        for name, fn in arms:
            t = timed(fn)
            print(f"[r{rnd}] {name:<22} fwd {t:6.2f}ms   {gf / t:6.2f} GF/ms")

    # ---- bwd：梯度对拍（ZSC 语义）+ f+b 计时 ----
    print()
    mma_op = make_mma_op()
    cot = mx.random.normal((B, H, NC, C, DV)).astype(mx.float32)
    mx.eval(cot)

    def loss_mma(*a):
        return (mma_op(*a)[0] * cot).sum()

    def loss_ref(*a):
        return (_kda_scan(*a)[0] * cot).sum()

    def loss_cur(*a):
        return (kda_scan_metal(*a, zsc=True)[0] * cot).sum()

    lg, gg = mx.value_and_grad(loss_mma, argnums=tuple(range(7)))(*ins)
    lr, gr = mx.value_and_grad(loss_ref, argnums=tuple(range(7)))(*ins)
    mx.eval(lg, lr, *gg, *gr)
    names7 = ["dqe", "dw", "du", "dAqk", "dkd", "degl", "dS0"]
    worst = 0.0
    for nm, a, b_ in zip(names7, gg, gr):
        rel = ((a - b_).abs().max() / (b_.abs().max() + 1e-12)).item()
        worst = max(worst, rel)
        print(f"  grad {nm:<5} rel {rel:.2e}")
    print(f"  loss {abs(lg.item() - lr.item()):.3e}  worst rel {worst:.2e}")

    vg_mma = mx.compile(mx.value_and_grad(loss_mma, argnums=tuple(range(7))))
    vg_cur = mx.compile(mx.value_and_grad(loss_cur, argnums=tuple(range(7))))
    f_mma = mx.compile(lambda *a: mma_op(*a)[0])
    f_cur = mx.compile(lambda *a: kda_scan_metal(*a, zsc=True)[0])
    for rnd in range(2):
        tf_c = timed(lambda: f_cur(*ins))
        tfb_c = timed(lambda: vg_cur(*ins))
        tf_m = timed(lambda: f_mma(*ins))
        tfb_m = timed(lambda: vg_mma(*ins))
        print(
            f"[r{rnd}] 标量zsc fwd {tf_c:6.2f} f+b {tfb_c:6.2f} bwd {tfb_c - tf_c:6.2f}"
            f"   MMA fwd {tf_m:6.2f} f+b {tfb_m:6.2f} bwd {tfb_m - tf_m:6.2f}"
        )


if __name__ == "__main__":
    main()
