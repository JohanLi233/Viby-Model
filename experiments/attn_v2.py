"""flash 反向 kernel v2：attn_fused.py 的优化变体，供 A/B 基准与对拍。

相对 v1（model/attn_fused.py）的改动：
1. fast_exp：exp → fast::exp2(x·log2e)（硬件 SFU，相对误差 ~2^-21，
   远小于 P 落 bf16 的 2^-9 量化误差）。
2. lse_nomax：行 logsumexp 不再做在线 max（s 值域 ~[-40,10]，f32 直接
   sum 不溢出），去掉 m 维护与 rescale；每行 4 线程并行分段求部分和 +
   simd_shuffle 归约，替代原来 32/128 线程单线程串行 16 次 exp 的链。
3. lse_delta：Δ = rowsum(dO∘O) 折叠进 lse kernel（同网格同 resident 行），
   消掉单独的 mlx 逐元素+归约 kernel（基线 ~0.74ms）。
4. dq 融合 elementwise：dP 不再复用 Ss 缓冲，单独 DPs；P 与 dS 合并为
   一遍 exp 循环（原来 P 先落 bf16 再读回，现在 exp 结果直接 f32 参与
   dS，少一遍 Ps 往返、少一个 barrier，精度还更高）。Q/dO 常驻块 hoist
   到寄存器（原来每个 key 块从 device 重复载入）。
5. dkv_interleave：D=128 的两遍扫描合并为一遍（原来 S 算两遍、Q 拷两遍、
   exp 两遍、8 个 barrier/块）。accK/accV 低 64 维都常驻寄存器，高维走
   threadgroup；每 query 块 5 个 barrier。K 常驻块 hoist 到寄存器。
6. dkv_lo：D≤64 路径同样融合 elementwise。

cfg 开关（默认全开）：
    fast_exp, lse_nomax, lse_delta, dq_fused, dq_hoist, dq_qcache,
    dkv_interleave, dkv_khoist, dkv_pad
"""

import math
import os

import mlx.core as mx

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

NTHREADS = int(os.environ.get("VIBY_FLASH_NT", 128))
STR = int(os.environ.get("VIBY_FLASH_STR", 16))
_TGMEM_LIMIT = 32 * 1024
_DC = 8

V2_DEFAULT = {
    "fast_exp": True,
    "lse_nomax": True,
    "lse_delta": True,
    "dq_fused": True,
    "dq_hoist": True,
    "dq_qcache": False,
    "dkv_interleave": True,
    "dkv_khoist": True,
    "dkv_pad": True,
}

_kernel_cache: dict = {}


def _cfg_key(cfg):
    return tuple(sorted((cfg or {}).items()))


def _res(nt):
    return nt // 32 * 8


def _mask_expr(has_mask):
    if not has_mask:
        return "0.0f"
    return "float(maskb[((size_t)(bh / NH) * TQ + gi) * TK + gj])"


PAD_B = 8
PAD_F = 4


def _dlo(d):
    return min(_DC, d // 8)


def _dhi(d):
    return d // 8 - _dlo(d)


def _ahi(d):
    return max(d - _dlo(d) * 8, 8) + PAD_F


def _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb, cfg):
    if cfg.get("fast_exp"):
        expm = "#define EXP(x) fast::exp2((x) * 1.4426950408889634f)"
    else:
        expm = "#define EXP(x) metal::exp(x)"
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
        #define SF0 {strb}
        #define SP0 {strb}
        #define SCALE {scale!r}f
        #define DLO {_dlo(Dk)}
        #define DHI {_dhi(Dk)}
        #define AHI {_ahi(Dk)}
        #define ELO {_dlo(Dv)}
        #define EHI {_dhi(Dv)}
        #define AVH {_ahi(Dv)}
        #define AMAX {max(_ahi(Dk), _ahi(Dv))}
        #define HSTEP 4
        {expm}
    """


def _tgmem(kind, Dk, Dv, res, strb, cfg):
    """静态 threadgroup 用量（字节）。"""
    sk = (Dk + PAD_B) * 2
    sf, sp = (strb + PAD_F) * 4, (strb + PAD_B) * 2
    acc_k = res * (_ahi(Dk) if _dhi(Dk) else 0) * 4
    if kind == "lse":
        return strb * sk + res * sf + res * 8
    if kind == "dq":
        qcache = res * sk if cfg.get("dq_qcache") else 0
        return strb * sk + res * sf + res * sf + res * sp + res * 8 + acc_k + qcache
    # dkv：Ss/Ps/DPs 带行距填充（dkv_pad），AccVHi + AccKHi 并存
    if cfg.get("dkv_pad"):
        ss, sp_np, dp = res * sf, res * sp, res * sf
    else:
        ss, sp_np, dp = res * strb * 4, res * strb * 2, res * strb * 4
    acc_v = res * (_ahi(Dv) if _dhi(Dv) else 0) * 4
    return strb * sk + ss + sp_np + dp + strb * 8 + acc_k + acc_v


def _build_lse(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg):
    res = _res(nt)
    nomax = cfg.get("lse_nomax", True)
    delta = cfg.get("lse_delta", True)
    if nomax:
        ew = f"""
            float p = 0.0f;
            uint gi = qs + row;
            for (uint j = chunk * (STR / 4); j < (chunk + 1) * (STR / 4); j++) {{
                uint gj = ks + j;
                float s = Ss[row * SF + j] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                p += EXP(s);
            }}
            p += simd_shuffle_xor(p, 1);
            p += simd_shuffle_xor(p, 2);
            if (chunk == 0) ls[row] += p;
        """
        head = """
        uint row = tid / 4;
        uint chunk = tid % 4;
        threadgroup float ls[RES];
        for (uint i = tid; i < RES; i += NT) ls[i] = 0.0f;
        """
        epi = """
        for (uint i = tid; i < RES; i += NT) {
            // 整行全屏蔽（padding 区）时 l=0：落 0 而非 -inf，避免下游 nan
            float l = ls[i];
            lse[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? metal::log(l) : 0.0f;
        }
        """
    else:
        ew = f"""
            for (uint i = tid; i < RES; i += NT) {{
                uint gi = qs + i;
                float mprev = ms[i];
                float mcur = mprev;
                for (uint j = 0; j < STR; j++) {{
                    uint gj = ks + j;
                    float s = Ss[i * SF + j] * SCALE + {_mask_expr(has_mask)};
                    s = (gj <= gi) ? s : -INFINITY;
                    Ss[i * SF + j] = s;
                    mcur = metal::max(mcur, s);
                }}
                if (mcur > -INFINITY) {{
                    float sum = 0.0f;
                    for (uint j = 0; j < STR; j++)
                        sum += EXP(Ss[i * SF + j] - mcur);
                    ls[i] = ls[i] * EXP(mprev - mcur) + sum;
                    ms[i] = mcur;
                }}
            }}
        """
        head = """
        uint row = tid / 4;
        uint chunk = tid % 4;
        threadgroup float ms[RES];
        threadgroup float ls[RES];
        for (uint i = tid; i < RES; i += NT) {{ ms[i] = -INFINITY; ls[i] = 0.0f; }}
        """
        epi = """
        for (uint i = tid; i < RES; i += NT) {
            float l = ls[i];
            lse[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? (ms[i] + metal::log(l)) : 0.0f;
        }
        """

    delta_src = ""
    if delta:
        delta_src = """
        float pd = 0.0f;
        for (uint d = chunk * (DV / 4); d < (chunk + 1) * (DV / 4); d++) {
            size_t off = ((size_t)bh * TQ + qs + row) * DV + d;
            pd += float(dout[off]) * float(o[off]);
        }
        pd += simd_shuffle_xor(pd, 1);
        pd += simd_shuffle_xor(pd, 2);
        if (chunk == 0) delta[(size_t)bh * TQ + qs + row] = pd;
        """
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb, cfg)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint qs  = qb * RES;
        {head}
        threadgroup {mt} Ks[STR * SK];
        threadgroup float Ss[RES * SF];

        {delta_src}

        simdgroup_matrix<{mt}, 8, 8> Qf[DK / 8];
        const device {mt}* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
        for (uint d = 0; d < DK / 8; d++) simdgroup_load(Qf[d], qp + d * 8, DK);

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
            for (uint d = 0; d < DK / 8; d++)
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Sf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            {ew}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {epi}
    """
    )
    names = (
        ["q", "k"] + (["dout", "o"] if delta else []) + (["maskb"] if has_mask else [])
    )
    outs = ["lse", "delta"] if delta else ["lse"]
    return mx.fast.metal_kernel(
        name=f"flash_lse_v3_{Dk}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}_"
        f"{int(nomax)}{int(delta)}",
        input_names=names,
        output_names=outs,
        source=src,
        header=_HEADER,
    )


def _build_dq(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg):
    res = _res(nt)
    hoist = cfg.get("dq_hoist", True)
    qcache = cfg.get("dq_qcache", False)
    if qcache:
        qpre = f"""
        threadgroup {mt} Qr[RES * SK];
        for (uint i = tid; i < RES * DK; i += NT)
            Qr[(i / DK) * SK + i % DK] =
                q[((size_t)bh * TQ + qs + i / DK) * DK + i % DK];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        qmma = f"""
                simdgroup_matrix<{mt}, 8, 8> QfL;
                simdgroup_load(QfL, Qr + (size_t)sg * 8 * SK + d * 8, SK);
        """
        qop = "QfL"
    elif hoist:
        qpre = f"""
        simdgroup_matrix<{mt}, 8, 8> Qf[DK / 8];
        const device {mt}* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
        for (uint d = 0; d < DK / 8; d++) simdgroup_load(Qf[d], qp + d * 8, DK);
        """
        qmma = ""
        qop = "Qf[d]"
    else:
        qpre = f"""
        const device {mt}* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
        """
        qmma = f"""
                simdgroup_matrix<{mt}, 8, 8> QfL;
                simdgroup_load(QfL, qp + d * 8, DK);
        """
        qop = "QfL"
    dhoist = f"""
        simdgroup_matrix<{mt}, 8, 8> dOf[DV / 8];
        const device {mt}* op0 = dout + ((size_t)bh * TQ + qs + sg * 8) * DV;
        for (uint e = 0; e < DV / 8; e++) simdgroup_load(dOf[e], op0 + e * 8, DV);
    """
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb, cfg)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint qs  = qb * RES;

        threadgroup {mt} Ks[STR * SK];
        threadgroup float Ss[RES * SF];
        threadgroup float DPs[RES * SF];
        threadgroup {mt} Ps[RES * SP];
        threadgroup float Ls[RES];
        threadgroup float Ds[RES];
        threadgroup float AccHi[RES * AHI];
        for (uint i = tid; i < RES; i += NT) {{
            Ls[i] = lse[(size_t)bh * TQ + qs + i];
            Ds[i] = delta[(size_t)bh * TQ + qs + i];
        }}
        for (uint i = tid; i < RES * AHI; i += NT) AccHi[i] = 0.0f;

        simdgroup_matrix<float, 8, 8> accLo[DLO];
        for (uint dc = 0; dc < DLO; dc++)
            accLo[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        {qpre}
        {dhoist}
        const device {mt}* op = dout + ((size_t)bh * TQ + qs + sg * 8) * DV;
        device float* dqp = dq + ((size_t)bh * TQ + qs + sg * 8) * DK;
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
            for (uint d = 0; d < DK / 8; d++) {{
                {qmma}
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], {qop}, Kf, Sf[c]);
                }}
            }}
            const device {mt}* vp = v + ((size_t)bh * TK + ks) * DV;
            for (uint e = 0; e < DV / 8; e++) {{
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Vf;
                    simdgroup_load(Vf, vp + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Pf[c], dOf[e], Vf, Pf[c]);
                }}
            }}
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(Sf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
                simdgroup_store(Pf[c], DPs + (size_t)sg * 8 * SF + c * 8, SF);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 融合：P 与 dS 一遍算完，exp 结果以 f32 直接参与 dS
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint i = t / STR, j = t % STR;
                uint gi = qs + i, gj = ks + j;
                float s = Ss[i * SF + j] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[i * SP + j] =
                    {mt}(SCALE * EXP(s - Ls[i]) * (DPs[i * SF + j] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<{mt}, 8, 8> dSf[STR / 8];
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_load(dSf[c], Ps + (size_t)sg * 8 * SP + c * 8, SP);
            for (uint c = 0; c < STR / 8; c++) {{
                for (uint dc = 0; dc < DLO; dc++) {{
                    simdgroup_matrix<{mt}, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accLo[dc], dSf[c], Kf, accLo[dc]);
                }}
            }}
            for (uint h0 = 0; h0 < DHI; h0 += HSTEP) {{
                uint nh = metal::min(uint(HSTEP), DHI - h0);
                simdgroup_matrix<float, 8, 8> accH[HSTEP];
                for (uint h = 0; h < nh; h++)
                    simdgroup_load(accH[h], AccHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
                for (uint c = 0; c < STR / 8; c++) {{
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Kf;
                        simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + (DLO + h0 + h) * 8, SK);
                        simdgroup_multiply_accumulate(accH[h], dSf[c], Kf, accH[h]);
                    }}
                }}
                for (uint h = 0; h < nh; h++)
                    simdgroup_store(accH[h], AccHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dc = 0; dc < DLO; dc++)
            simdgroup_store(accLo[dc], dqp + dc * 8, DK);
        for (uint h = 0; h < DHI; h++) {{
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, AccHi + (size_t)sg * 8 * AHI + h * 8, AHI);
            simdgroup_store(acc, dqp + (DLO + h) * 8, DK);
        }}
    """
    )
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dq_v5_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}_"
        f"{int(hoist)}{int(qcache)}",
        input_names=names,
        output_names=["dq"],
        source=src,
        header=_HEADER,
    )


def _build_dkv(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg):
    if _dhi(Dk) == 0 and _dhi(Dv) == 0:
        return _build_dkv_lo(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg)
    return _build_dkv_hi(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg)


def _build_dkv_lo(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg):
    """D<=64：累加器全部常驻寄存器，单遍；elementwise 融合。"""
    res = _res(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb, cfg)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint kbi = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint kss = kbi * RES;

        threadgroup {mt} Qs[STR * SK];
        threadgroup float Ss[RES * SF];
        threadgroup float DPs[RES * SF];
        threadgroup {mt} Ps[RES * SP];
        threadgroup float Ls[STR];
        threadgroup float Ds[STR];

        simdgroup_matrix<float, 8, 8> accK[DLO];
        simdgroup_matrix<float, 8, 8> accV[ELO];
        for (uint dc = 0; dc < DLO; dc++)
            accK[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint ec = 0; ec < ELO; ec++)
            accV[ec] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        const device {mt}* kp = k + ((size_t)bh * TK + kss + sg * 8) * DK;
        const device {mt}* vp = v + ((size_t)bh * TK + kss + sg * 8) * DV;
        device float* dkp = dk + ((size_t)bh * TK + kss + sg * 8) * DK;
        device float* dvp = dv + ((size_t)bh * TK + kss + sg * 8) * DV;
        for (uint qs = (kss / STR) * STR; qs < TQ; qs += STR) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Qs[(i / DK) * SK + i % DK] =
                    q[((size_t)bh * TQ + qs + i / DK) * DK + i % DK];
            for (uint i = tid; i < STR; i += NT) {{
                Ls[i] = lse[(size_t)bh * TQ + qs + i];
                Ds[i] = delta[(size_t)bh * TQ + qs + i];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> STf[STR / 8];
            simdgroup_matrix<float, 8, 8> PTf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                STf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                PTf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            for (uint d = 0; d < DK / 8; d++) {{
                simdgroup_matrix<{mt}, 8, 8> Kf;
                simdgroup_load(Kf, kp + d * 8, DK);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Qt;
                    simdgroup_load(Qt, Qs + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(STf[c], Kf, Qt, STf[c]);
                }}
            }}
            const device {mt}* op = dout + ((size_t)bh * TQ + qs) * DV;
            for (uint e = 0; e < DV / 8; e++) {{
                simdgroup_matrix<{mt}, 8, 8> Vf;
                simdgroup_load(Vf, vp + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Ot;
                    simdgroup_load(Ot, op + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(PTf[c], Vf, Ot, PTf[c]);
                }}
            }}
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(STf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
                simdgroup_store(PTf[c], DPs + (size_t)sg * 8 * SF + c * 8, SF);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = Ss[j * SF + i] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[j * SP + i] = {mt}(EXP(s - Ls[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> PT;
                simdgroup_load(PT, Ps + (size_t)sg * 8 * SP + c * 8, SP);
                for (uint ec = 0; ec < ELO; ec++) {{
                    simdgroup_matrix<{mt}, 8, 8> Of;
                    simdgroup_load(Of, op + (size_t)c * 8 * DV + ec * 8, DV);
                    simdgroup_multiply_accumulate(accV[ec], PT, Of, accV[ec]);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                Ps[j * SP + i] =
                    {mt}(SCALE * float(Ps[j * SP + i]) * (DPs[j * SF + i] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> GT;
                simdgroup_load(GT, Ps + (size_t)sg * 8 * SP + c * 8, SP);
                for (uint dc = 0; dc < DLO; dc++) {{
                    simdgroup_matrix<{mt}, 8, 8> Qf2;
                    simdgroup_load(Qf2, Qs + (size_t)c * 8 * SK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accK[dc], GT, Qf2, accK[dc]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dc = 0; dc < DLO; dc++)
            simdgroup_store(accK[dc], dkp + dc * 8, DK);
        for (uint ec = 0; ec < ELO; ec++)
            simdgroup_store(accV[ec], dvp + ec * 8, DV);
    """
    )
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dkv_lo_v2_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dk", "dv"],
        source=src,
        header=_HEADER,
    )


def _build_dkv_hi(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb, cfg):
    """D=128：单遍扫 query，dV/dK 的 64 维低段各占一套寄存器累加器，
    高维走 threadgroup（AccVHi/AccKHi）。S、Qs 拷贝、P 的 exp 都只做一遍。
    """
    res = _res(nt)
    khoist = cfg.get("dkv_khoist", True)
    pad = cfg.get("dkv_pad", True)
    sf, sp = ("SF", "SP") if pad else ("SF0", "SP0")
    kdecl = ""
    if khoist:
        kdecl = f"""
        simdgroup_matrix<{mt}, 8, 8> Kf[DK / 8];
        for (uint d = 0; d < DK / 8; d++) simdgroup_load(Kf[d], kp + d * 8, DK);
        """
        kload = "Kf[d]"
    else:
        kdecl = f"simdgroup_matrix<{mt}, 8, 8> Kf;"
        kload = "Kf"
    kld = "simdgroup_load(Kf, kp + d * 8, DK);" if not khoist else ""
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb, cfg)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint kbi = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint kss = kbi * RES;

        threadgroup {mt} Qs[STR * SK];
        threadgroup float Ss[RES * {sf}];
        threadgroup float DPs[RES * {sf}];
        threadgroup {mt} Ps[RES * {sp}];
        threadgroup float Ls[STR];
        threadgroup float Ds[STR];
        threadgroup float AccVHi[RES * AVH];
        threadgroup float AccKHi[RES * AHI];

        const device {mt}* kp = k + ((size_t)bh * TK + kss + sg * 8) * DK;
        const device {mt}* vp = v + ((size_t)bh * TK + kss + sg * 8) * DV;
        device float* dkp = dk + ((size_t)bh * TK + kss + sg * 8) * DK;
        device float* dvp = dv + ((size_t)bh * TK + kss + sg * 8) * DV;
        uint q0 = (kss / STR) * STR;

        for (uint i = tid; i < RES * AVH; i += NT) AccVHi[i] = 0.0f;
        for (uint i = tid; i < RES * AHI; i += NT) AccKHi[i] = 0.0f;
        simdgroup_matrix<float, 8, 8> accVLo[ELO];
        simdgroup_matrix<float, 8, 8> accKLo[DLO];
        for (uint ec = 0; ec < ELO; ec++)
            accVLo[ec] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dc = 0; dc < DLO; dc++)
            accKLo[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        {kdecl}
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

            simdgroup_matrix<float, 8, 8> STf[STR / 8];
            simdgroup_matrix<float, 8, 8> PTf[STR / 8];
            for (uint c = 0; c < STR / 8; c++) {{
                STf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                PTf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }}
            for (uint d = 0; d < DK / 8; d++) {{
                {kld}
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Qt;
                    simdgroup_load(Qt, Qs + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(STf[c], {kload}, Qt, STf[c]);
                }}
            }}
            const device {mt}* op = dout + ((size_t)bh * TQ + qs) * DV;
            for (uint e = 0; e < DV / 8; e++) {{
                simdgroup_matrix<{mt}, 8, 8> Vf;
                simdgroup_load(Vf, vp + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Ot;
                    simdgroup_load(Ot, op + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(PTf[c], Vf, Ot, PTf[c]);
                }}
            }}
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_store(STf[c], Ss + (size_t)sg * 8 * {sf} + c * 8, {sf});
                simdgroup_store(PTf[c], DPs + (size_t)sg * 8 * {sf} + c * 8, {sf});
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P = exp(S - L)
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = Ss[j * {sf} + i] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[j * {sp} + i] = {mt}(EXP(s - Ls[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dV = P @ dO
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> PT;
                simdgroup_load(PT, Ps + (size_t)sg * 8 * {sp} + c * 8, {sp});
                for (uint ec = 0; ec < ELO; ec++) {{
                    simdgroup_matrix<{mt}, 8, 8> Of;
                    simdgroup_load(Of, op + (size_t)c * 8 * DV + ec * 8, DV);
                    simdgroup_multiply_accumulate(accVLo[ec], PT, Of, accVLo[ec]);
                }}
            }}
            for (uint h0 = 0; h0 < EHI; h0 += HSTEP) {{
                uint nh = metal::min(uint(HSTEP), EHI - h0);
                simdgroup_matrix<float, 8, 8> accH[HSTEP];
                for (uint h = 0; h < nh; h++)
                    simdgroup_load(accH[h], AccVHi + (size_t)sg * 8 * AVH + (h0 + h) * 8, AVH);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> PT;
                    simdgroup_load(PT, Ps + (size_t)sg * 8 * {sp} + c * 8, {sp});
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Of;
                        simdgroup_load(Of, op + (size_t)c * 8 * DV + (ELO + h0 + h) * 8, DV);
                        simdgroup_multiply_accumulate(accH[h], PT, Of, accH[h]);
                    }}
                }}
                for (uint h = 0; h < nh; h++)
                    simdgroup_store(accH[h], AccVHi + (size_t)sg * 8 * AVH + (h0 + h) * 8, AVH);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dS = SCALE * P * (dP - Delta)
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                Ps[j * {sp} + i] =
                    {mt}(SCALE * float(Ps[j * {sp} + i]) * (DPs[j * {sf} + i] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dK = dS @ Q
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> GT;
                simdgroup_load(GT, Ps + (size_t)sg * 8 * {sp} + c * 8, {sp});
                for (uint dc = 0; dc < DLO; dc++) {{
                    simdgroup_matrix<{mt}, 8, 8> Qf2;
                    simdgroup_load(Qf2, Qs + (size_t)c * 8 * SK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accKLo[dc], GT, Qf2, accKLo[dc]);
                }}
            }}
            for (uint h0 = 0; h0 < DHI; h0 += HSTEP) {{
                uint nh = metal::min(uint(HSTEP), DHI - h0);
                simdgroup_matrix<float, 8, 8> accH[HSTEP];
                for (uint h = 0; h < nh; h++)
                    simdgroup_load(accH[h], AccKHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> GT;
                    simdgroup_load(GT, Ps + (size_t)sg * 8 * {sp} + c * 8, {sp});
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Qf2;
                        simdgroup_load(Qf2, Qs + (size_t)c * 8 * SK + (DLO + h0 + h) * 8, SK);
                        simdgroup_multiply_accumulate(accH[h], GT, Qf2, accH[h]);
                    }}
                }}
                for (uint h = 0; h < nh; h++)
                    simdgroup_store(accH[h], AccKHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ec = 0; ec < ELO; ec++)
            simdgroup_store(accVLo[ec], dvp + ec * 8, DV);
        for (uint h = 0; h < EHI; h++) {{
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, AccVHi + (size_t)sg * 8 * AVH + h * 8, AVH);
            simdgroup_store(acc, dvp + (ELO + h) * 8, DV);
        }}
        for (uint dc = 0; dc < DLO; dc++)
            simdgroup_store(accKLo[dc], dkp + dc * 8, DK);
        for (uint h = 0; h < DHI; h++) {{
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, AccKHi + (size_t)sg * 8 * AHI + h * 8, AHI);
            simdgroup_store(acc, dkp + (DLO + h) * 8, DK);
        }}
    """
    )
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dkv_v6_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}_"
        f"{int(khoist)}{int(pad)}",
        input_names=names,
        output_names=["dk", "dv"],
        source=src,
        header=_HEADER,
    )


_BUILDERS = {"lse": _build_lse, "dq": _build_dq, "dkv": _build_dkv}


def _get(kind, *key, cfg=None):
    ck = _cfg_key(cfg)
    kern = _kernel_cache.get((kind,) + key + (ck,))
    if kern is None:
        kern = _BUILDERS[kind](*key, dict(ck) if ck else {})
        _kernel_cache[(kind,) + key + (ck,)] = kern
    return kern


def tile_ok(Dk, Dv, nt, strb, cfg=None):
    cfg = cfg or {}
    res = _res(nt)
    if nt % 32 or strb % 8 or Dk % 8 or Dv % 8:
        return False
    return all(
        _tgmem(kind, Dk, Dv, res, strb, cfg) <= _TGMEM_LIMIT for kind in _BUILDERS
    )


def _align(nt, strb):
    return math.lcm(_res(nt), strb)


def _pad_t(x, t_pad, axis):
    t = x.shape[axis]
    if t == t_pad:
        return x
    shape = list(x.shape)
    shape[axis] = t_pad - t
    return mx.concatenate([x, mx.zeros(tuple(shape), dtype=x.dtype)], axis=axis)


def _pad_mask(mask, tq_pad, tk_pad):
    tq, tk = mask.shape[-2], mask.shape[-1]
    if tk != tk_pad:
        mask = _pad_t(mask, tk_pad, axis=-1)
    if tq != tq_pad:
        mask = _pad_t(mask, tq_pad, axis=-2)
    return mask


def flash_backward(q, k, v, o, dout, scale, mask=None, nt=None, strb=None, cfg=None):
    """因果注意力反向 (dq, dk, dv)。cfg 见模块 docstring。"""
    cfg = {**V2_DEFAULT, **(cfg or {})}
    nt = nt or NTHREADS
    strb = strb or STR
    B, H, Tq, Dk = q.shape
    Tk, Dv = k.shape[2], v.shape[3]
    if not tile_ok(Dk, Dv, nt, strb, cfg):
        raise ValueError(f"分块超 threadgroup 内存上限：nt={nt} str={strb}")
    align = _align(nt, strb)
    tq_pad = (Tq + align - 1) // align * align
    tk_pad = (Tk + align - 1) // align * align
    if tq_pad != Tq:
        q, o, dout = _pad_t(q, tq_pad, 2), _pad_t(o, tq_pad, 2), _pad_t(dout, tq_pad, 2)
    if tk_pad != Tk:
        k, v = _pad_t(k, tk_pad, 2), _pad_t(v, tk_pad, 2)
    if mask is not None:
        mask = _pad_mask(mask, tq_pad, tk_pad)
    res = _res(nt)
    mt = _METAL_TYPE[q.dtype]
    hm = mask is not None
    key = (Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strb)

    if cfg.get("lse_delta", True):
        lse_in = [q, k, dout, o] + ([mask] if hm else [])
        lse, delta = _get("lse", *key, cfg=cfg)(
            inputs=lse_in,
            output_shapes=[(B, H, tq_pad), (B, H, tq_pad)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(nt, tq_pad // res, B * H),
            threadgroup=(nt, 1, 1),
        )
    else:
        delta = (dout.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)
        lse = _get("lse", *key, cfg=cfg)(
            inputs=[q, k] + ([mask] if hm else []),
            output_shapes=[(B, H, tq_pad)],
            output_dtypes=[mx.float32],
            grid=(nt, tq_pad // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0]

    base = [q, k, v, dout, lse, delta] + ([mask] if hm else [])
    dq = _get("dq", *key, cfg=cfg)(
        inputs=base,
        output_shapes=[(B, H, tq_pad, Dk)],
        output_dtypes=[mx.float32],
        grid=(nt, tq_pad // res, B * H),
        threadgroup=(nt, 1, 1),
    )[0]
    dk, dv = _get("dkv", *key, cfg=cfg)(
        inputs=base,
        output_shapes=[(B, H, tk_pad, Dk), (B, H, tk_pad, Dv)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(nt, tk_pad // res, B * H),
        threadgroup=(nt, 1, 1),
    )
    if tq_pad != Tq:
        dq = dq[:, :, :Tq]
    if tk_pad != Tk:
        dk, dv = dk[:, :, :Tk], dv[:, :, :Tk]
    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
