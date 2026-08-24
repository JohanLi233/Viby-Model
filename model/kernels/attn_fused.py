"""因果注意力的手写 Metal flash（前向 O+LSE + 反向）。

mlx 的 mx.fast.scaled_dot_product_attention 只有前向 kernel。本模块：
- 前向：split-D 在线 softmax，一次扫 K 同时写出 O 与 LSE（不等宽 V 无需 pad）。
- 反向：复用前向 LSE，只跑 dq+dkv 合成 kernel，不再重算 S。

B=12 H=8 T=1024 d_qk=128 d_v=96 doc_mask 公平 vg（含前向）：
手写 16.5ms vs mlx autodiff 18.0ms（约 1.10×）。
"""

import math
import os

import mlx.core as mx

from . import attn_fused_split as _split

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

# 默认分块。D 维按 simdgroup 对半拆分（见 attn_fused_split.py），寄存器
# 累加器减半后 NT=256 可以常驻 RES=32 行，TG 内存约 16KB；lse 没有累加器，
# 单独用更高的 STR=32。旧「低维常驻寄存器、高维走 TG」路径保留为 fallback。
NTHREADS = int(os.environ.get("VIBY_FLASH_NT", 256))
STR = int(os.environ.get("VIBY_FLASH_STR", 16))
_COMBINED = os.environ.get("VIBY_FLASH_COMBINED", "1") != "0"
# 手写 flash 前向同时吐出 LSE，反向跳过 lse kernel。关了就回退 mlx SDPA。
_USE_FWD = os.environ.get("VIBY_FLASH_FWD", "1") != "0"
_TGMEM_LIMIT = 32 * 1024
# 寄存器里常驻的 D 片段数。8 片 = 64 维，与 D=64 快路径同压。
_DC = 8

_kernel_cache: dict = {}


def _res(nt):
    """旧路径（fallback）的常驻块高。"""
    return nt // 32 * 8


def _res_split(nt):
    """split-D 路径的常驻块高（2 个 simdgroup 一组 -> 8 行）。"""
    return nt // 8


def _split_ok(Dk, Dv, nt):
    return _split.split_ok(Dk, Dv, nt)


def _lse_str(nt, strb):
    """lse kernel 单独的分块高（无累加器，STR 可以更大）。"""
    return _split.lse_str(nt, strb)


def _mask_expr(has_mask):
    """数组 mask 的加性偏置（(B,1,Tq,Tk)，跨 head 广播）。"""
    if not has_mask:
        return "0.0f"
    return "float(maskb[((size_t)(bh / NH) * TQ + gi) * TK + gj])"


# threadgroup 内存行距填充，消除 bank 冲突。
# Apple GPU 的 threadgroup 内存是 32 个 4 字节 bank。bf16 行距 DK=128 时
# 一行 256 字节 = 64 bank，回绕后每行都从 bank 0 起——simdgroup_load 转置读
# 一个 8×8 片段要跨 8 行，8 路全撞同一 bank，串行化。行距 +8 个 bf16 让相邻
# 行错开 4 个 bank（8 行落在 0/4/…/28，互不相同）；f32 缓冲同理 +4。
# 代价只有每块几百字节。（MoE 的 decode kernel 同样按此规避，见 moe_decode.py）
PAD_B = 8
PAD_F = 4


def _dlo(d):
    """寄存器常驻的 8×8 片段数（最多 _DC 片 = 64 维）。"""
    return min(_DC, d // 8)


def _dhi(d):
    return d // 8 - _dlo(d)


def _ahi(d):
    """高维累加器 threadgroup 行距。DHI=0 时给 8 占位，避免 0 长度数组。"""
    return max(d - _dlo(d) * 8, 8) + PAD_F


def _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb):
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
    """


def _tgmem(kind, Dk, Dv, res, strb):
    """静态 threadgroup 用量（字节），用于提前拒绝超 32KB 的分块。"""
    sk = (Dk + PAD_B) * 2
    sf, sp = (strb + PAD_F) * 4, (strb + PAD_B) * 2
    acc_k = res * (_ahi(Dk) if _dhi(Dk) else 0) * 4
    if kind == "lse":
        return strb * sk + res * sf + res * 8
    if kind == "dq":
        # V 从 device 现载，腾出 AccHi。
        return strb * sk + res * sf + res * sp + res * 8 + acc_k
    # dkv：两遍扫描共用一块 AccHi（max(AHI,AVH)）。
    ss, sp_np = strb * 4, strb * 2
    acc_hi = (
        res
        * max(
            _ahi(Dk) if _dhi(Dk) else 0,
            _ahi(Dv) if _dhi(Dv) else 0,
        )
        * 4
    )
    return strb * sk + res * ss + res * sp_np + strb * 8 + acc_hi


def _build_lse_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    res = _res(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint qs  = qb * RES;

        threadgroup {mt} Ks[STR * SK];
        threadgroup float Ss[RES * SF];
        threadgroup float ms[RES];
        threadgroup float ls[RES];
        for (uint i = tid; i < RES; i += NT) {{ ms[i] = -INFINITY; ls[i] = 0.0f; }}

        simdgroup_matrix<{mt}, 8, 8> Qf[DK / 8];
        const device {mt}* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
        for (uint d = 0; d < DK / 8; d++) simdgroup_load(Qf[d], qp + d * 8, DK);

        // 因果性：起点超过本 query 块最大下标的整个 key 块直接不算
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
                    // 转置载入：Kf[dd][jj] = K[ks+8c+jj][8d+dd] ⇒ S = Q·Kᵀ
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf[d], Kf, Sf[c]);
                }}
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Sf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
                // 整块对本行全屏蔽时 mcur 仍是 -inf，alpha=exp(+inf) 会出
                // inf/nan，必须整体跳过而不是照常更新
                if (mcur > -INFINITY) {{
                    float sum = 0.0f;
                    for (uint j = 0; j < STR; j++)
                        sum += metal::exp(Ss[i * SF + j] - mcur);
                    ls[i] = ls[i] * metal::exp(mprev - mcur) + sum;
                    ms[i] = mcur;
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RES; i += NT) {{
            // 整行全屏蔽（padding 区）时 l=0：落 0 而非 -inf，避免下游
            // exp(S-L) 出 nan；这些行的梯度贡献本来就该是 0
            float l = ls[i];
            lse[(size_t)bh * TQ + qs + i] = (l > 0.0f) ? (ms[i] + metal::log(l)) : 0.0f;
        }}
    """
    )
    names = ["q", "k"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_lse_v2_{Dk}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["lse"],
        source=src,
        header=_HEADER,
    )


def _build_dq_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dQ = scale·dS·K。RES 是 query 块（常驻），STR 是 key 块（流过）。

    低 DLO 片（64 维）累加器常驻寄存器，跨 kb 复用；高维放 threadgroup，
    每次只取出 HSTEP 片做 MMA，避免 D=128 时 16 个片段把寄存器撑爆。
    V 从 device 转置载入，不占 TG。
    """
    res = _res(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint qb  = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint qs  = qb * RES;

        threadgroup {mt} Ks[STR * SK];
        threadgroup float Ss[RES * SF];
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

        const device {mt}* qp = q + ((size_t)bh * TQ + qs + sg * 8) * DK;
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
                simdgroup_matrix<{mt}, 8, 8> Qf;
                simdgroup_load(Qf, qp + d * 8, DK);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + d * 8, SK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Sf[c], Qf, Kf, Sf[c]);
                }}
            }}
            const device {mt}* vp = v + ((size_t)bh * TK + ks) * DV;
            for (uint e = 0; e < DV / 8; e++) {{
                simdgroup_matrix<{mt}, 8, 8> dOf;
                simdgroup_load(dOf, op + e * 8, DV);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> Vf;
                    simdgroup_load(Vf, vp + (size_t)c * 8 * DV + e * 8, DV,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(Pf[c], dOf, Vf, Pf[c]);
                }}
            }}
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Sf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint i = t / STR, j = t % STR;
                uint gi = qs + i, gj = ks + j;
                float s = Ss[i * SF + j] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[i * SP + j] = {mt}(metal::exp(s - Ls[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(Pf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint i = t / STR, j = t % STR;
                Ps[i * SP + j] =
                    {mt}(SCALE * float(Ps[i * SP + j]) * (Ss[i * SF + j] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> dSf;
                simdgroup_load(dSf, Ps + (size_t)sg * 8 * SP + c * 8, SP);
                for (uint dc = 0; dc < DLO; dc++) {{
                    simdgroup_matrix<{mt}, 8, 8> Kf;
                    simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + dc * 8, SK);
                    simdgroup_multiply_accumulate(accLo[dc], dSf, Kf, accLo[dc]);
                }}
            }}
            for (uint h0 = 0; h0 < DHI; h0 += HSTEP) {{
                uint nh = metal::min(uint(HSTEP), DHI - h0);
                simdgroup_matrix<float, 8, 8> accH[HSTEP];
                for (uint h = 0; h < nh; h++)
                    simdgroup_load(accH[h], AccHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> dSf;
                    simdgroup_load(dSf, Ps + (size_t)sg * 8 * SP + c * 8, SP);
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Kf;
                        simdgroup_load(Kf, Ks + (size_t)c * 8 * SK + (DLO + h0 + h) * 8, SK);
                        simdgroup_multiply_accumulate(accH[h], dSf, Kf, accH[h]);
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
        name=f"flash_dq_v4_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dq"],
        source=src,
        header=_HEADER,
    )


def _build_dkv_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dV = Pᵀ·dO、dK = scale·dSᵀ·Q。

    D<=64：dK/dV 累加器都常驻寄存器，一遍扫完。
    D=128：两遍扫 query，每遍只留一套 64 维累加器，高维走 AccHi。
    """
    if _dhi(Dk) == 0 and _dhi(Dv) == 0:
        return _build_dkv_lo(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)
    return _build_dkv_hi(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)


def _build_dkv_lo(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """D<=64：累加器全部常驻，单遍。"""
    res = _res(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint kbi = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint kss = kbi * RES;

        threadgroup {mt} Qs[STR * SK];
        threadgroup float Ss[RES * SF];
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
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(STf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = Ss[j * SF + i] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[j * SP + i] = {mt}(metal::exp(s - Ls[i]));
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
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(PTf[c], Ss + (size_t)sg * 8 * SF + c * 8, SF);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                Ps[j * SP + i] =
                    {mt}(SCALE * float(Ps[j * SP + i]) * (Ss[j * SF + i] - Ds[i]));
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
        name=f"flash_dkv_lo_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dk", "dv"],
        source=src,
        header=_HEADER,
    )


def _build_dkv_hi(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dV = Pᵀ·dO、dK = scale·dSᵀ·Q。

    两遍扫 query：第一遍只累加 dV，第二遍只累加 dK。寄存器里始终只有一套
    64 维累加器（与 D=64 快路径同压）。S 算两遍，比两套累加器把寄存器
    撑爆更便宜。第一遍不算 dP（dV 只用 P）。dO 从 device 现载。
    """
    res = _res(nt)
    src = (
        _consts(Dk, Dv, Tq, Tk, nh, scale, nt, res, strb)
        + f"""
        uint tid = thread_position_in_grid.x;
        uint kbi = thread_position_in_grid.y;
        uint bh  = thread_position_in_grid.z;
        uint sg  = tid / 32;
        uint kss = kbi * RES;

        threadgroup {mt} Qs[STR * SK];
        threadgroup float Ss[RES * SF0];
        threadgroup {mt} Ps[RES * SP0];
        threadgroup float Ls[STR];
        threadgroup float Ds[STR];
        threadgroup float AccHi[RES * AMAX];

        const device {mt}* kp = k + ((size_t)bh * TK + kss + sg * 8) * DK;
        const device {mt}* vp = v + ((size_t)bh * TK + kss + sg * 8) * DV;
        device float* dkp = dk + ((size_t)bh * TK + kss + sg * 8) * DK;
        device float* dvp = dv + ((size_t)bh * TK + kss + sg * 8) * DV;
        uint q0 = (kss / STR) * STR;

        // ---- pass 1: dV = P @ dO（不算 dP）----
        for (uint i = tid; i < RES * AVH; i += NT) AccHi[i] = 0.0f;
        simdgroup_matrix<float, 8, 8> accVLo[ELO];
        for (uint ec = 0; ec < ELO; ec++)
            accVLo[ec] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint qs = q0; qs < TQ; qs += STR) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < STR * DK; i += NT)
                Qs[(i / DK) * SK + i % DK] =
                    q[((size_t)bh * TQ + qs + i / DK) * DK + i % DK];
            for (uint i = tid; i < STR; i += NT)
                Ls[i] = lse[(size_t)bh * TQ + qs + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> STf[STR / 8];
            for (uint c = 0; c < STR / 8; c++)
                STf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(STf[c], Ss + (size_t)sg * 8 * SF0 + c * 8, SF0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = Ss[j * SF0 + i] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[j * SP0 + i] = {mt}(metal::exp(s - Ls[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const device {mt}* op = dout + ((size_t)bh * TQ + qs) * DV;
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> PT;
                simdgroup_load(PT, Ps + (size_t)sg * 8 * SP0 + c * 8, SP0);
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
                    simdgroup_load(accH[h], AccHi + (size_t)sg * 8 * AVH + (h0 + h) * 8, AVH);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> PT;
                    simdgroup_load(PT, Ps + (size_t)sg * 8 * SP0 + c * 8, SP0);
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Of;
                        simdgroup_load(Of, op + (size_t)c * 8 * DV + (ELO + h0 + h) * 8, DV);
                        simdgroup_multiply_accumulate(accH[h], PT, Of, accH[h]);
                    }}
                }}
                for (uint h = 0; h < nh; h++)
                    simdgroup_store(accH[h], AccHi + (size_t)sg * 8 * AVH + (h0 + h) * 8, AVH);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ec = 0; ec < ELO; ec++)
            simdgroup_store(accVLo[ec], dvp + ec * 8, DV);
        for (uint h = 0; h < EHI; h++) {{
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, AccHi + (size_t)sg * 8 * AVH + h * 8, AVH);
            simdgroup_store(acc, dvp + (ELO + h) * 8, DV);
        }}

        // ---- pass 2: dK = dS @ Q ----
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RES * AHI; i += NT) AccHi[i] = 0.0f;
        simdgroup_matrix<float, 8, 8> accKLo[DLO];
        for (uint dc = 0; dc < DLO; dc++)
            accKLo[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(STf[c], Ss + (size_t)sg * 8 * SF0 + c * 8, SF0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                uint gj = kss + j, gi = qs + i;
                float s = Ss[j * SF0 + i] * SCALE + {_mask_expr(has_mask)};
                s = (gj <= gi) ? s : -INFINITY;
                Ps[j * SP0 + i] = {mt}(metal::exp(s - Ls[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < STR / 8; c++)
                simdgroup_store(PTf[c], Ss + (size_t)sg * 8 * SF0 + c * 8, SF0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = tid; t < RES * STR; t += NT) {{
                uint j = t / STR, i = t % STR;
                Ps[j * SP0 + i] =
                    {mt}(SCALE * float(Ps[j * SP0 + i]) * (Ss[j * SF0 + i] - Ds[i]));
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < STR / 8; c++) {{
                simdgroup_matrix<{mt}, 8, 8> GT;
                simdgroup_load(GT, Ps + (size_t)sg * 8 * SP0 + c * 8, SP0);
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
                    simdgroup_load(accH[h], AccHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
                for (uint c = 0; c < STR / 8; c++) {{
                    simdgroup_matrix<{mt}, 8, 8> GT;
                    simdgroup_load(GT, Ps + (size_t)sg * 8 * SP0 + c * 8, SP0);
                    for (uint h = 0; h < nh; h++) {{
                        simdgroup_matrix<{mt}, 8, 8> Qf2;
                        simdgroup_load(Qf2, Qs + (size_t)c * 8 * SK + (DLO + h0 + h) * 8, SK);
                        simdgroup_multiply_accumulate(accH[h], GT, Qf2, accH[h]);
                    }}
                }}
                for (uint h = 0; h < nh; h++)
                    simdgroup_store(accH[h], AccHi + (size_t)sg * 8 * AHI + (h0 + h) * 8, AHI);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dc = 0; dc < DLO; dc++)
            simdgroup_store(accKLo[dc], dkp + dc * 8, DK);
        for (uint h = 0; h < DHI; h++) {{
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, AccHi + (size_t)sg * 8 * AHI + h * 8, AHI);
            simdgroup_store(acc, dkp + (DLO + h) * 8, DK);
        }}
    """
    )
    names = ["q", "k", "v", "dout", "lse", "delta"] + (["maskb"] if has_mask else [])
    return mx.fast.metal_kernel(
        name=f"flash_dkv_v5_{Dk}_{Dv}_{Tq}_{Tk}_{nh}_{int(has_mask)}_{mt}_{nt}_{strb}",
        input_names=names,
        output_names=["dk", "dv"],
        source=src,
        header=_HEADER,
    )


def _build_lse(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """lse kernel 工厂（兼容旧调用：输入 q/k、输出 lse）。

    split lse 额外融合了 delta，输入/输出签名不同，由 flash_backward 通过
    _get_lse_fused 单独走，不让它破坏既有脚本对 _get("lse") 的调用约定。
    """
    return _build_lse_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)


def _build_dq(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dq kernel 工厂：split-D 快路径优先。"""
    if _split_ok(Dk, Dv, nt):
        return _split._build_dq(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)
    return _build_dq_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)


def _build_dkv(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb):
    """dkv kernel 工厂：split-D 快路径优先。"""
    if _split_ok(Dk, Dv, nt):
        return _split._build_dkv(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)
    return _build_dkv_v1(Dk, Dv, Tq, Tk, nh, scale, has_mask, mt, nt, strb)


_BUILDERS = {"lse": _build_lse, "dq": _build_dq, "dkv": _build_dkv}


def _get(kind, *key):
    kern = _kernel_cache.get((kind,) + key)
    if kern is None:
        kern = _BUILDERS[kind](*key)
        _kernel_cache[(kind,) + key] = kern
    return kern


def _get_lse_fused(*key):
    """split-D lse kernel（lse + delta 一次输出）的缓存入口。"""
    ck = ("lse_fused",) + key
    kern = _kernel_cache.get(ck)
    if kern is None:
        kern = _split._build_lse(*key)
        _kernel_cache[ck] = kern
    return kern


def _get_lse_dp(*key):
    """不依赖前向 O 的 lse+delta kernel（delta 用 P⊙(dO·Vᵀ) 行和重算）。"""
    ck = ("lse_dp",) + key
    kern = _kernel_cache.get(ck)
    if kern is None:
        kern = _split._build_lse_dp(*key)
        _kernel_cache[ck] = kern
    return kern


def _get_fwd(*key):
    """flash 前向（O+LSE）缓存入口。"""
    ck = ("fwd",) + key
    kern = _kernel_cache.get(ck)
    if kern is None:
        kern = _split._build_fwd(*key)
        _kernel_cache[ck] = kern
    return kern


def _get_dqkv_combined(*key):
    """dq+dkv 合成 kernel（grid.z 前半 dq、后半 dkv）的缓存入口。"""
    ck = ("dqkv",) + key
    kern = _kernel_cache.get(ck)
    if kern is None:
        kern = _split._build_dqkv(*key)
        _kernel_cache[ck] = kern
    return kern


def tile_ok(Dk, Dv, nt, strb):
    """分块是否满足 threadgroup 内存与整除约束。"""
    if nt % 32 or strb % 8 or Dk % 8 or Dv % 8:
        return False
    if _split_ok(Dk, Dv, nt):
        return _split.tile_ok(Dk, Dv, nt, strb, _lse_str(nt, strb))
    res = _res(nt)
    return all(_tgmem(kind, Dk, Dv, res, strb) <= _TGMEM_LIMIT for kind in _BUILDERS)


def _align(nt, strb, split=False):
    """序列 padding 对齐到 RES 与 lse/dq STR 的公倍数。"""
    if split and (nt // 32) % 2 == 0:
        return math.lcm(_res_split(nt), strb, _lse_str(nt, strb))
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


def flash_forward(q, k, v, scale, mask=None, nt=None, strb=None):
    """因果注意力前向，返回 (o, lse)。lse 给 flash_backward 复用。"""
    nt = nt or NTHREADS
    B, H, Tq, Dk = q.shape
    Tk, Dv = k.shape[2], v.shape[3]
    strb = strb or _split.fwd_str(Dk, Dv, nt)
    if strb is None or not (_split_ok(Dk, Dv, nt) and tile_ok(Dk, Dv, nt, STR)):
        raise ValueError(f"flash 前向分块不支持：nt={nt} Dk={Dk} Dv={Dv}")
    align = _align(nt, STR, True)
    tq_pad = (Tq + align - 1) // align * align
    tk_pad = (Tk + align - 1) // align * align
    if tq_pad != Tq:
        q = _pad_t(q, tq_pad, 2)
    if tk_pad != Tk:
        k, v = _pad_t(k, tk_pad, 2), _pad_t(v, tk_pad, 2)
    if mask is not None:
        mask = _pad_mask(mask, tq_pad, tk_pad)
    res = _res_split(nt)
    mt = _METAL_TYPE[q.dtype]
    hm = mask is not None
    key = (Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strb)
    o, lse = _get_fwd(*key)(
        inputs=[q, k, v] + ([mask] if hm else []),
        output_shapes=[(B, H, tq_pad, Dv), (B, H, tq_pad)],
        output_dtypes=[q.dtype, mx.float32],
        grid=(nt, tq_pad // res, B * H),
        threadgroup=(nt, 1, 1),
    )
    if tq_pad != Tq:
        o, lse = o[:, :, :Tq], lse[:, :, :Tq]
    return o, lse


def flash_backward(q, k, v, o, dout, scale, mask=None, nt=None, strb=None, lse=None):
    """因果注意力反向，返回 (dq, dk, dv)，dtype 与各自 primal 一致。

    o 是前向输出（用来算 Δ）；传 None 时 split 路径用
    Δ=rowsum(P⊙(dO·Vᵀ)) 在 lse kernel 里重算。
    lse 若已由 flash_forward 给出则跳过 lse kernel。
    mask 为 None 表示纯 causal，否则是 (B,1,Tq,Tk) 的加性偏置。
    T 不是分块倍数时在序列尾部零填充，算完切回。
    """
    nt = nt or NTHREADS
    strb = strb or STR
    B, H, Tq, Dk = q.shape
    Tk, Dv = k.shape[2], v.shape[3]
    if not tile_ok(Dk, Dv, nt, strb):
        raise ValueError(f"分块超 threadgroup 内存上限：nt={nt} str={strb}")
    split = _split_ok(Dk, Dv, nt)
    align = _align(nt, strb, split)
    tq_pad = (Tq + align - 1) // align * align
    tk_pad = (Tk + align - 1) // align * align
    if tq_pad != Tq:
        q = _pad_t(q, tq_pad, 2)
        dout = _pad_t(dout, tq_pad, 2)
        if o is not None:
            o = _pad_t(o, tq_pad, 2)
        if lse is not None:
            lse = _pad_t(lse, tq_pad, 2)
    if tk_pad != Tk:
        k, v = _pad_t(k, tk_pad, 2), _pad_t(v, tk_pad, 2)
    if mask is not None:
        mask = _pad_mask(mask, tq_pad, tk_pad)
    if dout.dtype != q.dtype:
        # 训练图的 cotangent 常为 f32，而 MMA 的两个操作数必须同精度
        # （K/V 是 bf16）；把 dout 先压回 q.dtype 再进 kernel。
        dout = dout.astype(q.dtype)
    res = _res_split(nt) if split else _res(nt)
    mt = _METAL_TYPE[q.dtype]
    hm = mask is not None
    key = (Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strb)

    saved_lse = lse
    if saved_lse is not None and o is not None:
        delta = (dout.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)
        lse = saved_lse
    elif split:
        if o is None:
            key_lse = (Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strb)
            lse, delta = _get_lse_dp(*key_lse)(
                inputs=[q, k, dout, v] + ([mask] if hm else []),
                output_shapes=[(B, H, tq_pad), (B, H, tq_pad)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, tq_pad // res, B * H),
                threadgroup=(nt, 1, 1),
            )
        else:
            strl = _lse_str(nt, strb)
            key_lse = (Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strl)
            lse, delta = _get_lse_fused(*key_lse)(
                inputs=[q, k, dout, o] + ([mask] if hm else []),
                output_shapes=[(B, H, tq_pad), (B, H, tq_pad)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(nt, tq_pad // res, B * H),
                threadgroup=(nt, 1, 1),
            )
    else:
        if o is None:
            raise ValueError("非 split 路径需要前向输出 o 来计算 Δ")
        delta = (dout.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1)
        lse = _get("lse", *key)(
            inputs=[q, k] + ([mask] if hm else []),
            output_shapes=[(B, H, tq_pad)],
            output_dtypes=[mx.float32],
            grid=(nt, tq_pad // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0]

    base = [q, k, v, dout, lse, delta] + ([mask] if hm else [])
    if split and tq_pad == tk_pad and _COMBINED:
        key_comb = (B, Dk, Dv, tq_pad, tk_pad, H, scale, hm, mt, nt, strb)
        dq, dk, dv = _get_dqkv_combined(*key_comb)(
            inputs=base,
            output_shapes=[(B, H, tq_pad, Dk), (B, H, tk_pad, Dk), (B, H, tk_pad, Dv)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(nt, tq_pad // res, B * H * 2),
            threadgroup=(nt, 1, 1),
        )
    else:
        dq = _get("dq", *key)(
            inputs=base,
            output_shapes=[(B, H, tq_pad, Dk)],
            output_dtypes=[mx.float32],
            grid=(nt, tq_pad // res, B * H),
            threadgroup=(nt, 1, 1),
        )[0]
        dk, dv = _get("dkv", *key)(
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


_FUSED_DISABLED = os.environ.get("VIBY_ATTN_FUSED", "1") != "1"
_flash_fn_cache: dict = {}


def _flash_fn(scale, masked, v_dim=None, use_fwd=False):
    """按 (scale, 是否数组 mask, v_dim, use_fwd) 缓存 custom_function。

    use_fwd：手写前向同时吐出 LSE，反向跳过 lse kernel。
    否则前向走 mlx SDPA（V 零填到 QK 维命中等宽快路径），反向重算 LSE。
    """
    key = (float(scale), bool(masked), v_dim, bool(use_fwd))
    hit = _flash_fn_cache.get(key)
    if hit is not None:
        return hit
    mlx_sdpa = mx.fast.scaled_dot_product_attention

    def _mlx_fwd(q, k, v, mask):
        qdim = q.shape[-1]
        vp = v
        if v_dim is not None and v.shape[-1] < qdim:
            vp = mx.concatenate(
                [v, mx.zeros(v.shape[:-1] + (qdim - v.shape[-1],), dtype=v.dtype)],
                axis=-1,
            )
        out = mlx_sdpa(q, k, vp, scale=scale, mask=mask)
        return out[..., :v_dim] if v_dim is not None and v_dim < out.shape[-1] else out

    if use_fwd:
        if masked:

            @mx.custom_function
            def _core(q, k, v, mask):
                return flash_forward(q, k, v, scale, mask)

            def _vjp(primals, cotangents, outputs):
                q, k, v, mask = primals
                o, lse = outputs[0], outputs[1]
                do = (
                    cotangents[0]
                    if isinstance(cotangents, (list, tuple))
                    else cotangents
                )
                dq, dk, dv = flash_backward(q, k, v, o, do, scale, mask, lse=lse)
                return dq, dk, dv, None

        else:

            @mx.custom_function
            def _core(q, k, v):
                return flash_forward(q, k, v, scale, None)

            def _vjp(primals, cotangents, outputs):
                q, k, v = primals
                o, lse = outputs[0], outputs[1]
                do = (
                    cotangents[0]
                    if isinstance(cotangents, (list, tuple))
                    else cotangents
                )
                return flash_backward(q, k, v, o, do, scale, None, lse=lse)

        _core.vjp(_vjp)

        def _fwd(*args):
            return _core(*args)[0]

    elif masked:

        @mx.custom_function
        def _fwd(q, k, v, mask):
            return _mlx_fwd(q, k, v, mask)

        def _vjp(primals, cotangents, outputs):
            q, k, v, mask = primals
            o = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            do = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
            dq, dk, dv = flash_backward(q, k, v, o, do, scale, mask)
            return dq, dk, dv, None

        _fwd.vjp(_vjp)
    else:

        @mx.custom_function
        def _fwd(q, k, v):
            return _mlx_fwd(q, k, v, "causal")

        def _vjp(primals, cotangents, outputs):
            q, k, v = primals
            o = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            do = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
            return flash_backward(q, k, v, o, do, scale, None)

        _fwd.vjp(_vjp)

    _flash_fn_cache[key] = _fwd
    return _fwd


def flash_sdpa(q, k, v, scale, mask="causal", v_dim=None):
    """训练用 SDPA：手写 Metal 前向（O+LSE）+ 手写反向；失败则回退 mlx。

    默认开启（VIBY_ATTN_FUSED=1；设 0 回退 mx 自带的 SDPA autodiff）。
    VIBY_FLASH_FWD=0 时前向仍走 mlx SDPA，反向重算 LSE。
    """
    mlx_sdpa = mx.fast.scaled_dot_product_attention
    qdim = q.shape[-1]
    if v_dim is None:
        v_dim = v.shape[-1]

    def _plain_forward():
        vp = v
        if v.shape[-1] < qdim:
            vp = mx.concatenate(
                [v, mx.zeros(v.shape[:-1] + (qdim - v.shape[-1],), dtype=v.dtype)],
                axis=-1,
            )
        out = mlx_sdpa(q, k, vp, scale=scale, mask=mask)
        return out[..., :v_dim] if v_dim < out.shape[-1] else out

    if _FUSED_DISABLED:
        return _plain_forward()
    if (
        q.ndim != 4
        or q.shape[2] != k.shape[2]
        or q.dtype not in _METAL_TYPE
        or q.dtype == mx.float32
        # f32 输入的 Ks/Vs TG 占用翻倍（D=96 时 38KB > 32KB 上限），且
        # fused 对 f32 无收益；训练主路径是 bf16，f32 一律回退 mlx SDPA
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or v_dim != v.shape[-1]
        or not tile_ok(q.shape[-1], v_dim, NTHREADS, STR)
    ):
        return _plain_forward()
    has_arr = mask is not None and not isinstance(mask, str)
    use_fwd = (
        _USE_FWD
        and _split_ok(q.shape[-1], v_dim, NTHREADS)
        and _split.fwd_str(q.shape[-1], v_dim, NTHREADS) is not None
    )
    try:
        fn = _flash_fn(scale, has_arr, v_dim, use_fwd)
        return fn(q, k, v, mask) if has_arr else fn(q, k, v)
    except Exception:
        return _plain_forward()


def reference_lse(q, k, scale, mask=None):
    """mlx 算子参考实现（f32），用于对拍。"""
    Tq, Tk = q.shape[2], k.shape[2]
    s = (q.astype(mx.float32) @ mx.swapaxes(k, -1, -2).astype(mx.float32)) * scale
    if mask is not None:
        s = s + mask.astype(mx.float32)
    return mx.logsumexp(s + mx.triu(mx.full((Tq, Tk), -mx.inf), k=1), axis=-1)


def reference_attention(q, k, v, scale, mask=None):
    """f32 朴素前向，给对拍提供 o 与梯度基准。"""
    Tq, Tk = q.shape[2], k.shape[2]
    s = (q.astype(mx.float32) @ mx.swapaxes(k, -1, -2).astype(mx.float32)) * scale
    if mask is not None:
        s = s + mask.astype(mx.float32)
    s = s + mx.triu(mx.full((Tq, Tk), -mx.inf), k=1)
    return mx.softmax(s, axis=-1) @ v.astype(mx.float32)
