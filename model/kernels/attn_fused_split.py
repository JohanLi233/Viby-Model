"""Split-D Metal kernels for the backward flash attention path.

These kernels are the v3 backend for ``attn_fused.py``.  The key change over the
previous backend is that the D dimension of the MMA is split across pairs of
simdgroups instead of being kept in one simdgroup:

- two simdgroups cooperate on the same 8 query/key rows, each owning one half
  of D_k / D_v;
- partial S (QKᵀ) and partial dP (dO·Vᵀ) are stored side by side in
  threadgroup memory and combined in the elementwise pass;
- each simdgroup then accumulates only its half of the dQ/dK/dV output, so the
  accumulator count is halved (D=128: 8 instead of 16 matrices, D_v=96: 6).

The halved register pressure removes the AccHi threadgroup accumulator and lets
NT=256 keep RES=32 rows with ~16KB threadgroup memory.  In practice dq goes
7.4ms -> 4.6ms and dkv 10.1ms -> 6.1ms on M4 Max at B=12 H=8 T=1024.
"""

import os
import re

import mlx.core as mx

_RE_K = re.compile(r"\bKs\b")
_RE_Q = re.compile(r"\bQs\b")

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

PAD_B = 0 if os.environ.get("VIBY_FLASH_NOPAD") == "1" else 8
PAD_F = 0 if os.environ.get("VIBY_FLASH_NOPAD") == "1" else 4

STR_LSE = int(os.environ.get("VIBY_FLASH_STR_LSE", 32))
_TGMEM_LIMIT = 32 * 1024


def split_ok(Dk, Dv, nt):
    """Whether the split-D decomposition applies.

    Two simdgroups form a row-pair, so nt must contain an even number of
    simdgroups, and each simdgroup owns D/2 which must be a whole number of
    8-wide MMA tiles.
    """
    return (
        nt >= 64
        and nt % 32 == 0
        and (nt // 32) % 2 == 0
        and Dk % 16 == 0
        and Dv % 16 == 0
    )


def res_split(nt):
    """Rows produced by one threadgroup (2 simdgroups -> 8 rows)."""
    return nt // 8


def lse_str(nt, strb):
    """Tile height used by the lse kernel.  It has no accumulators, so a taller
    stream block (32) is faster; dq/dkv keep STR=16."""
    del nt, strb
    return STR_LSE


def _mask_expr(has_mask):
    if not has_mask:
        return "0.0f"
    return "float(maskb[((size_t)(bh / NH) * TQ + gi) * TK + gj])"


def _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb):
    return f"""
        #define DK {Dk}
        #define DV {Dv}
        #define TQ {Tq}
        #define TK {Tk}
        #define NH {nh}
        #define NT {nt}
        #define RES {res}
        #define STR {strb}
        #define SK {Dk + PAD_B}
        #define SV {Dv + PAD_B}
        #define SF {strb + PAD_F}
        #define SP {strb + PAD_B}
        #define SCALE {scale!r}f
        #define HDK {Dk // 2}
        #define HDV {Dv // 2}
        #define NHDK {Dk // 2 // 8}
        #define NHDV {Dv // 2 // 8}
        #define MT {mt}
        #define EXP(x) fast::exp2((x) * 1.4426950408889634f)
    """


def tgmem(kind, Dk, Dv, nt, strb, strl=None):
    """Static threadgroup usage of a split kernel (bytes)."""
    res = res_split(nt)
    if kind == "lse":
        s = strl if strl is not None else strb
        ks = s * (Dk + PAD_B) * 2
        ss = 2 * res * (s + PAD_F) * 4  # Ss0 + Ss1
        red = 2 * res * 4  # ls + delta partial
        return ks + ss + red
    ks = strb * (Dk + PAD_B) * 2
    ss = 2 * res * (strb + PAD_F) * 4  # Ss0 + Ss1
    dp = 2 * res * (strb + PAD_F) * 4  # DPs0 + DPs1
    ps = 2 * res * (strb + PAD_B) * 2  # Ps + dSs（P 与 dS 各一份）
    red = 2 * res * 4  # Ls + Ds
    return ks + ss + dp + ps + red


def tgmem_combined(Dk, Dv, nt, strb):
    """dq+dkv 合成 kernel 的 threadgroup 内存（stream/S/P 共享一份）。"""
    res = res_split(nt)
    stream = strb * (Dk + PAD_B) * 2
    ss = 2 * res * (strb + PAD_F) * 4  # Ss0 + Ss1
    dp = ss  # DPs0 + DPs1
    ps = 2 * res * (strb + PAD_B) * 2  # Ps + dSs
    red = 2 * res * 4  # Ls + Ds（取 max(RES, STR)）
    return stream + ss + dp + ps + red


def tgmem_fwd(Dk, Dv, nt, strb):
    """flash 前向（O+LSE）threadgroup 用量。K/V 进 TG，P 复用 Ss0。"""
    res = res_split(nt)
    ks = strb * (Dk + PAD_B) * 2
    vs = strb * (Dv + PAD_B) * 2
    ss = 2 * res * (strb + PAD_F) * 4
    tmp = (nt // 32) * 64 * 4
    red = res * 4
    return ks + vs + ss + tmp + red


def tile_ok(Dk, Dv, nt, strb, strl=None):
    if not split_ok(Dk, Dv, nt):
        return False
    if strb % 8:
        return False
    strl = strl if strl is not None else lse_str(nt, strb)
    if strl % 8:
        return False
    return (
        all(
            tgmem(k, Dk, Dv, nt, strb, strl) <= _TGMEM_LIMIT
            for k in ("lse", "dq", "dkv")
        )
        and tgmem_combined(Dk, Dv, nt, strb) <= _TGMEM_LIMIT
    )


def fwd_str(Dk, Dv, nt):
    """前向流过块高：优先 32，K/V 进 TG 塞不下时退到 16/8。"""
    for s in (STR_LSE, 16, 8):
        if s % 8 == 0 and tgmem_fwd(Dk, Dv, nt, s) <= _TGMEM_LIMIT:
            return s
    return None


def _build_lse(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    res = res_split(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint pair = sg >> 1;
        uint side = sg & 1;
        uint qs  = qb * RES;
        uint row0 = qs + pair * 8;
        // 8 lanes cooperate on one row: 8-way shuffle reduction, STR/8 exp per lane
        uint row = tid / 8;
        uint chunk = tid % 8;

        threadgroup MT Ks[STR * SK];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float ls[RES];
        threadgroup float ds[RES];
        for (uint i = tid; i < RES; i += NT) ls[i] = 0.0f;

        // delta = rowsum(dO * O), folded into the lse launch (same resident rows)
        float pd = 0.0f;
        for (uint d = chunk * (DV / 8); d < (chunk + 1) * (DV / 8); d++) {{
            size_t off = ((size_t)bh * TQ + qs + row) * DV + d;
            pd += float(dout[off]) * float(o[off]);
        }}
        pd += simd_shuffle_xor(pd, 1);
        pd += simd_shuffle_xor(pd, 2);
        pd += simd_shuffle_xor(pd, 4);
        if (chunk == 0) delta[(size_t)bh * TQ + qs + row] = pd;

        simdgroup_matrix<MT, 8, 8> Qf[NHDK];
        const device MT* qp = q + ((size_t)bh * TQ + row0) * DK + side * HDK;
        for (uint d = 0; d < NHDK; d++) simdgroup_load(Qf[d], qp + d * 8, DK);

        uint nkb = (qs + RES + STR - 1) / STR;
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Ks[(i / DK) * SK + i % DK] =
                    k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> Sf[STR / 8];
            for (uint c = 0; c < STR / 8; c++)
                Sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < NHDK; d++)
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + side * HDK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}
            threadgroup float* Ss = (side == 0) ? Ss0 : Ss1;
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Sf[c], Ss + (size_t)pair * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float p = 0.0f;
            uint gi = qs + row;
            for (uint j = chunk * (STR / 8); j < (chunk + 1) * (STR / 8); j++) {{
                uint gj = ks + j;
                float s = (Ss0[row * SF + j] + Ss1[row * SF + j]) * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                p += EXP(s);
            }}
            p += simd_shuffle_xor(p, 1);
            p += simd_shuffle_xor(p, 2);
            p += simd_shuffle_xor(p, 4);
            if (chunk == 0) ls[row] += p;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RES; i += NT) {{
            // fully masked rows (padding) fall through as lse=0
            float l = ls[i];
            lse[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? metal::log(l) : 0.0f;
        }}
    """
    )
    names = ["q", "k", "dout", "o"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_lse_split_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["lse", "delta"],
        source=src,
        header=_HEADER,
    )


def _build_fwd(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """Flash 前向：一次扫 K 写出 O 与 LSE。

    不走在线缩放（每块把 O 卸到 TG 再乘 α 太贵）。P=exp(S) 留在 f32
    Ss 里做 MMA，末尾再除以 Z；scale=1/sqrt(128) 下 exp(S) 不爆 f32。
    K/V 一起进 TG。LSE 与旧 lse kernel 同口径：log(sum(exp(S)))。
    """
    res = res_split(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint pair = sg >> 1;
        uint side = sg & 1;
        uint qs  = qb * RES;
        uint row0 = qs + pair * 8;
        uint row = tid / 8;
        uint chunk = tid % 8;
        uint lane = tid % 32;

        threadgroup MT Ks[STR * SK];
        threadgroup MT Vs[STR * SV];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float ls[RES];
        threadgroup float Tmp[NT / 32 * 64];
        for (uint i = tid; i < RES; i += NT) ls[i] = 0.0f;

        simdgroup_matrix<MT, 8, 8> Qf[NHDK];
        const device MT* qp = q + ((size_t)bh * TQ + row0) * DK + side * HDK;
        for (uint d = 0; d < NHDK; d++) simdgroup_load(Qf[d], qp + d * 8, DK);

        simdgroup_matrix<float, 8, 8> accO[NHDV];
        for (uint e = 0; e < NHDV; e++)
            accO[e] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        uint nkb = (qs + RES + STR - 1) / STR;
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Ks[(i / DK) * SK + i % DK] =
                    k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];
            for (uint i = tid; i < STR * DV; i += NT)
                Vs[(i / DV) * SV + i % DV] =
                    v[((size_t)bh * TK + ks + i / DV) * DV + i % DV];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> Sf[STR / 8];
            for (uint c = 0; c < STR / 8; c++)
                Sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < NHDK; d++)
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + side * HDK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}
            threadgroup float* Ss = (side == 0) ? Ss0 : Ss1;
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Sf[c], Ss + (size_t)pair * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float psum = 0.0f;
            uint gi = qs + row;
            for (uint j = chunk * (STR / 8); j < (chunk + 1) * (STR / 8); j++) {{
                uint gj = ks + j;
                float s = (Ss0[row * SF + j] + Ss1[row * SF + j]) * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                float p = EXP(s);
                Ss0[row * SF + j] = p;
                psum += p;
            }}
            psum += simd_shuffle_xor(psum, 1);
            psum += simd_shuffle_xor(psum, 2);
            psum += simd_shuffle_xor(psum, 4);
            if (chunk == 0) ls[row] += psum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<float, 8, 8> Pf;
                simdgroup_load(Pf, Ss0 + (size_t)pair * 8 * SF + c * 8, SF);
                for (uint e = 0; e < NHDV; e++) {{
                    simdgroup_matrix<MT, 8, 8> Vf;
                    simdgroup_load(Vf, Vs + (size_t)c * 8 * SV + side * HDV + e * 8, SV);
                    simdgroup_multiply_accumulate(accO[e], Pf, Vf, accO[e]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RES; i += NT)
            lse[(size_t)bh * TQ + qs + i] = (ls[i] > 0.0f) ? metal::log(ls[i]) : 0.0f;
        for (uint e = 0; e < NHDV; e++) {{
            simdgroup_store(accO[e], Tmp + (size_t)sg * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64; i += 32) {{
                float z = ls[pair * 8 + i / 8];
                float val = Tmp[sg * 64 + i] * ((z > 0.0f) ? (1.0f / z) : 0.0f);
                o[((size_t)bh * TQ + qs + pair * 8 + i / 8) * DV
                  + side * HDV + e * 8 + (i % 8)] = MT(val);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    )
    names = ["q", "k", "v"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_fwd_v2_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["o", "lse"],
        source=src,
        header=_HEADER,
    )


def _dq_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    res = res_split(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint pair = sg >> 1;
        uint side = sg & 1;
        uint qs  = qb * RES;
        uint row0 = qs + pair * 8;

        threadgroup MT Ks[STR * SK];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float DPs0[RES * SF];
        threadgroup float DPs1[RES * SF];
        threadgroup MT Ps[RES * SP];
        threadgroup float Ls[RES];
        threadgroup float Ds[RES];
        for (uint i = tid; i < RES; i += NT) {{
            Ls[i] = lse[(size_t)bh * TQ + qs + i];
            Ds[i] = delta[(size_t)bh * TQ + qs + i];
        }}

        // Each simdgroup accumulates exactly its half of dQ.  No AccHi needed.
        simdgroup_matrix<float, 8, 8> accLo[NHDK];
        for (uint dc = 0; dc < NHDK; dc++)
            accLo[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        simdgroup_matrix<MT, 8, 8> Qf[NHDK];
        const device MT* qp = q + ((size_t)bh * TQ + row0) * DK + side * HDK;
        for (uint d = 0; d < NHDK; d++) simdgroup_load(Qf[d], qp + d * 8, DK);

        const device MT* op = dout + ((size_t)bh * TQ + row0) * DV + side * HDV;
        device float* dqp = dq + ((size_t)bh * TQ + row0) * DK + side * HDK;

        uint nkb = (qs + RES + STR - 1) / STR;
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Ks[(i / DK) * SK + i % DK] =
                    k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // partial S = Q @ Kᵀ (each side does one half of the K dim)
            simdgroup_matrix<float, 8, 8> Sf[STR / 8];
            simdgroup_matrix<float, 8, 8> Pf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                Sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                Pf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            for (uint d = 0; d < NHDK; d++)
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + side * HDK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}

            // partial dP = dO @ Vᵀ (each side does one half of the V dim)
            const device MT* vp = v + ((size_t)bh * TK + ks) * DV + side * HDV;
            for (uint e = 0; e < NHDV; e++) {{
                simdgroup_matrix<MT, 8, 8> dOf;
                simdgroup_load(dOf, op + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Vf;
                    simdgroup_load(Vf, vp + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Pf[c], dOf, Vf, Pf[c]);
                }}
            }}

            threadgroup float* Ss = (side == 0) ? Ss0 : Ss1;
            threadgroup float* DPs = (side == 0) ? DPs0 : DPs1;
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(Sf[c], Ss + (size_t)pair * 8 * SF + c * 8, SF);
                simdgroup_store(Pf[c], DPs + (size_t)pair * 8 * SF + c * 8, SF);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P and dS in one pass; exp result stays f32 until the final store
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint i = t / STR, j = t % STR;
                uint gi = qs + i, gj = ks + j;
                float s = (Ss0[i * SF + j] + Ss1[i * SF + j]) * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                float dp = DPs0[i * SF + j] + DPs1[i * SF + j];
                Ps[i * SP + j] = MT(SCALE * EXP(s - Ls[i]) * (dp - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<MT, 8, 8> dSf;
                simdgroup_load(dSf, Ps + (size_t)pair * 8 * SP + c * 8, SP);
                for (uint dc = 0; dc < NHDK; dc++) {{
                    simdgroup_matrix<MT, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + side * HDK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accLo[dc], dSf, Kf, accLo[dc]);
                }}
            }}
        }}
        for (uint dc = 0; dc < NHDK; dc++)
            simdgroup_store(accLo[dc], dqp + dc * 8, DK);
    """
    )
    return src


def _build_dq(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    src = _dq_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dq_split_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dq"],
        source=src,
        header=_HEADER,
    )


def _dkv_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    res = res_split(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint kbi = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint pair = sg >> 1;
        uint side = sg & 1;
        uint kss = kbi * RES;
        uint row0 = kss + pair * 8;

        threadgroup MT Qs[STR * SK];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float DPs0[RES * SF];
        threadgroup float DPs1[RES * SF];
        threadgroup MT Ps[RES * SP];
        threadgroup MT dSs[RES * SP];
        threadgroup float Ls[STR];
        threadgroup float Ds[STR];

        // One pass over queries; each side keeps only its half of dK and dV.
        simdgroup_matrix<float, 8, 8> accK[NHDK];
        simdgroup_matrix<float, 8, 8> accV[NHDV];
        for (uint dc = 0; dc < NHDK; dc++)
            accK[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint ec = 0; ec < NHDV; ec++)
            accV[ec] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        const device MT* kp = k + ((size_t)bh * TK + row0) * DK + side * HDK;
        const device MT* vp = v + ((size_t)bh * TK + row0) * DV + side * HDV;
        device float* dkp = dk + ((size_t)bh * TK + row0) * DK + side * HDK;
        device float* dvp = dv + ((size_t)bh * TK + row0) * DV + side * HDV;
        uint q0 = (kss / STR) * STR;

        for (uint qs = q0; qs < TQ; qs += STR) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Qs[(i / DK) * SK + i % DK] =
                    q[((size_t)bh * TQ + qs + i / DK) * DK + i % DK];
            for (uint i = tid; i < STR; i += NT) {{
                Ls[i] = lse[(size_t)bh * TQ + qs + i];
                Ds[i] = delta[(size_t)bh * TQ + qs + i];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // partial S = K @ Qᵀ and partial dP = V @ dOᵀ
            simdgroup_matrix<float, 8, 8> STf[STR / 8];
            simdgroup_matrix<float, 8, 8> PTf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                STf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                PTf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            for (uint d = 0; d < NHDK; d++) {{
                simdgroup_matrix<MT, 8, 8> Kf;
                simdgroup_load(Kf, kp + d * 8, DK);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Qt;
                    simdgroup_load(Qt, Qs + (size_t)c * 8 * SK + side * HDK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(STf[c], Kf, Qt, STf[c]);
                }}
            }}
            const device MT* op = dout + ((size_t)bh * TQ + qs) * DV;
            for (uint e = 0; e < NHDV; e++) {{
                simdgroup_matrix<MT, 8, 8> Vf;
                simdgroup_load(Vf, vp + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Ot;
                    simdgroup_load(Ot, op + (size_t)c * 8 * DV + side * HDV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(PTf[c], Vf, Ot, PTf[c]);
                }}
            }}

            threadgroup float* Ss = (side == 0) ? Ss0 : Ss1;
            threadgroup float* DPs = (side == 0) ? DPs0 : DPs1;
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(STf[c], Ss + (size_t)pair * 8 * SF + c * 8, SF);
                simdgroup_store(PTf[c], DPs + (size_t)pair * 8 * SF + c * 8, SF);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P 与 dS 一次算完，各落一块 buffer；dV/dK 之间不再需要 barrier
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = (Ss0[j * SF + i] + Ss1[j * SF + i]) * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                float p = EXP(s - Ls[i]);
                Ps[j * SP + i] = MT(p);
                dSs[j * SP + i] =
                    MT(SCALE * p * (DPs0[j * SF + i] + DPs1[j * SF + i] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dV = P @ dO, dK = dS @ Q
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<MT, 8, 8> PT;
                simdgroup_matrix<MT, 8, 8> GT;
                simdgroup_load(PT, Ps + (size_t)pair * 8 * SP + c * 8, SP);
                simdgroup_load(GT, dSs + (size_t)pair * 8 * SP + c * 8, SP);
                for (uint ec = 0; ec < NHDV; ec++) {{
                    simdgroup_matrix<MT, 8, 8> Of;
                    simdgroup_load(Of, op + (size_t)c * 8 * DV + side * HDV + ec * 8, DV);
                    simdgroup_multiply_accumulate(accV[ec], PT, Of, accV[ec]);
                }}
                for (uint dc = 0; dc < NHDK; dc++) {{
                    simdgroup_matrix<MT, 8, 8> Qf2;
                    simdgroup_load(Qf2, Qs + (size_t)c * 8 * SK + side * HDK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accK[dc], GT, Qf2, accK[dc]);
                }}
            }}
        }}
        for (uint ec = 0; ec < NHDV; ec++)
            simdgroup_store(accV[ec], dvp + ec * 8, DV);
        for (uint dc = 0; dc < NHDK; dc++)
            simdgroup_store(accK[dc], dkp + dc * 8, DK);
    """
    )
    return src


def _build_dkv(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    src = _dkv_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dkv_split_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dk", "dv"],
        source=src,
        header=_HEADER,
    )


def _build_dqkv(B, Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dq 与 dkv 合成一个 kernel。

    grid.z 前半是 query 块（dq），后半是 key 块（dkv）。两个 kernel 都没有
    跨 threadgroup 依赖，合成后 GPU 可以在两种线程组之间自由调度，互相填充
    延迟；同时两个路径共享同一块 stream/S/P threadgroup 内存。
    """
    res = res_split(nt)

    def _branch(src, is_dq):
        src = src[src.index("        uint tid") :]
        src = src.replace("        uint tid = thread_position_in_grid.x;\n", "")
        if is_dq:
            src = src.replace(
                "        uint qb  = thread_position_in_grid.y;\n",
                "        uint qb = y;\n",
            )
            start = src.index("        threadgroup MT Ks[STR * SK];\n")
            end = src.index("        for (uint i = tid; i < RES; i += NT) {", start)
            src = src[:start] + src[end:]
            src = _RE_K.sub("stream", src)
        else:
            src = src.replace(
                "        uint kbi = thread_position_in_grid.y;\n",
                "        uint kbi = y;\n",
            )
            start = src.index("        threadgroup MT Qs[STR * SK];\n")
            end = src.index("        simdgroup_matrix<float, 8, 8> accK[NHDK];", start)
            src = src[:start] + src[end:]
            src = _RE_Q.sub("stream", src)
        return src.replace("        uint bh  = thread_position_in_grid.z;\n", "")

    common = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        #define BH {B * nh}

        uint tid = thread_position_in_grid.x;
        uint y = thread_position_in_grid.y;
        uint z = thread_position_in_grid.z;
        uint mode = z % 2;
        uint bh = z / 2;

        // dq 用 Ks、dkv 用 Qs，二者尺寸相同，合并成一块 stream
        threadgroup MT stream[STR * SK];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float DPs0[RES * SF];
        threadgroup float DPs1[RES * SF];
        threadgroup MT Ps[RES * SP];
        threadgroup MT dSs[RES * SP];
        threadgroup float Ls[RES];
        threadgroup float Ds[RES];

        if (mode == 0) {{
{_branch(_dq_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb), True)}
        }} else {{
{_branch(_dkv_src(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb), False)}
        }}
    """
    )
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dqkv_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dq", "dk", "dv"],
        source=common,
        header=_HEADER,
    )


def _build_lse_dp(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """lse + delta without the forward output.

    delta = rowsum(dO * O) = rowsum(P * (dO @ V^T))。这样 custom VJP 不再
    依赖前向输出，value_and_grad 就不会为了喂 outputs 再跑一遍 mlx 前向。
    """
    res = res_split(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, mt, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint pair = sg >> 1;
        uint side = sg & 1;
        uint qs  = qb * RES;
        uint row0 = qs + pair * 8;
        uint row = tid / 8;
        uint chunk = tid % 8;

        threadgroup MT Ks[STR * SK];
        threadgroup float Ss0[RES * SF];
        threadgroup float Ss1[RES * SF];
        threadgroup float DPs0[RES * SF];
        threadgroup float DPs1[RES * SF];
        threadgroup float ls[RES];
        threadgroup float ds[RES];
        for (uint i = tid; i < RES; i += NT) {{ ls[i] = 0.0f; ds[i] = 0.0f; }}

        simdgroup_matrix<MT, 8, 8> Qf[NHDK];
        const device MT* qp = q + ((size_t)bh * TQ + row0) * DK + side * HDK;
        for (uint d = 0; d < NHDK; d++) simdgroup_load(Qf[d], qp + d * 8, DK);
        const device MT* op = dout + ((size_t)bh * TQ + row0) * DV + side * HDV;

        uint nkb = (qs + RES + STR - 1) / STR;
        for (uint kb = 0; kb < nkb; kb++) {{
            uint ks = kb * STR;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Ks[(i / DK) * SK + i % DK] =
                    k[((size_t)bh * TK + ks + i / DK) * DK + i % DK];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> Sf[STR / 8];
            simdgroup_matrix<float, 8, 8> Pf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                Sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                Pf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            for (uint d = 0; d < NHDK; d++)
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + side * HDK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}
            const device MT* vp = v + ((size_t)bh * TK + ks) * DV + side * HDV;
            for (uint e = 0; e < NHDV; e++) {{
                simdgroup_matrix<MT, 8, 8> dOf;
                simdgroup_load(dOf, op + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<MT, 8, 8> Vf;
                    simdgroup_load(Vf, vp + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Pf[c], dOf, Vf, Pf[c]);
                }}
            }}
            threadgroup float* Ss = (side == 0) ? Ss0 : Ss1;
            threadgroup float* DPs = (side == 0) ? DPs0 : DPs1;
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(Sf[c], Ss + (size_t)pair * 8 * SF + c * 8, SF);
                simdgroup_store(Pf[c], DPs + (size_t)pair * 8 * SF + c * 8, SF);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float psum = 0.0f;
            float dsum = 0.0f;
            uint gi = qs + row;
            for (uint j = chunk * (STR / 8); j < (chunk + 1) * (STR / 8); j++) {{
                uint gj = ks + j;
                float s = (Ss0[row * SF + j] + Ss1[row * SF + j]) * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                float p = EXP(s);
                psum += p;
                dsum += p * (DPs0[row * SF + j] + DPs1[row * SF + j]);
            }}
            psum += simd_shuffle_xor(psum, 1);
            psum += simd_shuffle_xor(psum, 2);
            psum += simd_shuffle_xor(psum, 4);
            dsum += simd_shuffle_xor(dsum, 1);
            dsum += simd_shuffle_xor(dsum, 2);
            dsum += simd_shuffle_xor(dsum, 4);
            if (chunk == 0) {{ ls[row] += psum; ds[row] += dsum; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RES; i += NT) {{
            float l = ls[i];
            lse[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? metal::log(l) : 0.0f;
            delta[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? (ds[i] / l) : 0.0f;
        }}
    """
    )
    names = ["q", "k", "dout", "v"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_lse_dp_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["lse", "delta"],
        source=src,
        header=_HEADER,
    )


BUILDERS = {"lse": _build_lse, "dq": _build_dq, "dkv": _build_dkv}
