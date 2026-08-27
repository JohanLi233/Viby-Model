"""EigenGate 融合进 KDA 跨 chunk 扫描：一次 Metal 发射。

普通 kda_scan 把 S 放 threadgroup、按 Dv 分片；谱门控需要完整 SᵀS，
分片之间无法 device barrier。这里 dv_split=1，S/Gram/polar 走 global，
vt 留 TG。每隔 K 个 chunk 在 write 之后做与 Python ste_weight 相同的
cubic+STE：S ← W S，W=(1−λ)I + λ (PᵀP)/u²，P=polar(S) 不反传。

反向：门控处 dS ← Wᵀ dS，其余与 kda_scan 同。训练路径仍可 ZSC。
"""

from __future__ import annotations

import os

import mlx.core as mx

from ..eigengate import U_PEAK, cubic_coeffs

_KERNELS: dict = {}
_OPS: dict = {}

_NT_FWD = 256
_NT_BWD = 256
_DISABLED = (
    os.environ.get("VIBY_KDA_SCAN_EG", "1") != "1"
    or os.environ.get("VIBY_KDA_SCAN", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)


def _gidx_list(gate_tuple: tuple[int, ...]) -> list[int]:
    out = []
    gi = 0
    for g in gate_tuple:
        if g:
            out.append(gi)
            gi += 1
        else:
            out.append(-1)
    return out


def _supported(
    NC: int,
    C: int,
    D: int,
    DV: int,
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
) -> bool:
    """形状守卫：S 在 global，TG 只放 vt（fwd）或 vt+dvt（bwd）。"""
    if _DISABLED or NC <= 0 or C == 0:
        return False
    if D != DV:
        return False
    if nt_fwd % C != 0 or nt_bwd % C != 0:
        return False
    if DV % (nt_fwd // C) != 0:
        return False
    nb = nt_bwd // C
    if DV % nb != 0 or D % nb != 0:
        return False
    tg_fwd = (C * DV + nt_fwd // 32) * 4
    tg_bwd = 2 * C * DV * 4
    return tg_fwd <= 30720 and tg_bwd <= 30720


def _cubic_block(nsteps: int, ca_src: str, cb_src: str) -> str:
    """写完 S 后：X=polar(S)，W=(1-λ)I+λ(XXᵀ)/u²，S←W@S，W 写入 Wg[gi]。"""
    return f"""
            int gi = GIDX[c];
            if (gi >= 0) {{
                size_t xb = sb;
                size_t ab = (size_t)bh * D * D;
                size_t wb = ((size_t)bh * NG + (size_t)gi) * D * D;
                threadgroup float sh[NT / 32];
                float loc = 0.0f;
                for (uint i = tid; i < D * DV; i += NT) {{
                    float v = Sw[sb + i];
                    loc += v * v;
                }}
                loc = simd_sum(loc);
                if (tid % 32 == 0) sh[tid / 32] = loc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {{
                    float s = 0.0f;
                    for (uint k = 0; k < NT / 32; k++) s += sh[k];
                    sh[0] = s;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float invn = 1.0f / (sqrt(sh[0]) + 1e-7f);
                for (uint i = tid; i < D * DV; i += NT)
                    Xb[xb + i] = Sw[sb + i] * invn;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                constexpr float CA[{nsteps}] = {{ {ca_src} }};
                constexpr float CB[{nsteps}] = {{ {cb_src} }};
                for (uint step = 0; step < {nsteps}; step++) {{
                    for (uint idx = tid; idx < D * D; idx += NT) {{
                        uint i = idx / D, j = idx % D;
                        float acc = 0.0f;
                        for (uint p = 0; p < DV; p++)
                            acc += Xb[xb + i * DV + p] * Xb[xb + j * DV + p];
                        Ab[ab + i * D + j] = acc;
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint idx = tid; idx < D * DV; idx += NT) {{
                        uint i = idx / DV, p = idx % DV;
                        float acc = 0.0f;
                        for (uint j = 0; j < D; j++)
                            acc += Ab[ab + i * D + j] * Xb[xb + j * DV + p];
                        Tmp[sb + i * DV + p] = acc;
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float a = CA[step], b = CB[step];
                    for (uint i = tid; i < D * DV; i += NT)
                        Xb[xb + i] = a * Xb[xb + i] + b * Tmp[sb + i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}
                // A = P Pᵀ；W = (1-λ)I + λ A / u²
                for (uint idx = tid; idx < D * D; idx += NT) {{
                    uint i = idx / D, j = idx % D;
                    float acc = 0.0f;
                    for (uint p = 0; p < DV; p++)
                        acc += Xb[xb + i * DV + p] * Xb[xb + j * DV + p];
                    Ab[ab + i * D + j] = acc;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint idx = tid; idx < D * D; idx += NT) {{
                    uint i = idx / D, j = idx % D;
                    float wij = LAM * Ab[ab + idx] / U2;
                    if (i == j) wij += 1.0f - LAM;
                    Wg[wb + idx] = wij;
                    Ab[ab + idx] = wij;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint idx = tid; idx < D * DV; idx += NT) {{
                    uint i = idx / DV, p = idx % DV;
                    float acc = 0.0f;
                    for (uint j = 0; j < D; j++)
                        acc += Ab[ab + i * D + j] * Sw[sb + j * DV + p];
                    Tmp[sb + i * DV + p] = acc;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < D * DV; i += NT)
                    Sw[sb + i] = Tmp[sb + i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
"""


def _wt_ds_block() -> str:
    """dS ← Wᵀ dS（STE）。"""
    return """
            int gi = GIDX[c];
            if (gi >= 0) {
                size_t wb = ((size_t)bh * NG + (size_t)gi) * D * D;
                for (uint idx = tid; idx < D * DV; idx += NT) {
                    uint i = idx / DV, p = idx % DV;
                    float acc = 0.0f;
                    for (uint j = 0; j < D; j++)
                        acc += Wg[wb + j * D + i] * dSw[sb + j * DV + p];
                    dStmp[sb + i * DV + p] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < D * DV; i += NT)
                    dSw[sb + i] = dStmp[sb + i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
"""


def _build(
    NC: int,
    C: int,
    D: int,
    DV: int,
    gate_tuple: tuple[int, ...],
    lam: float,
    coeffs: list[tuple[float, float]],
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
    zsc: bool = False,
):
    gidx = _gidx_list(gate_tuple)
    NG = max(1, sum(1 for g in gate_tuple if g))
    gidx_src = ", ".join(str(x) for x in gidx)
    ca_src = ", ".join(f"{a:.8f}f" for a, _ in coeffs)
    cb_src = ", ".join(f"{b:.8f}f" for _, b in coeffs)
    nsteps = len(coeffs)
    u2 = float(U_PEAK * U_PEAK)
    key = (NC, C, D, DV, gate_tuple, float(lam), tuple(coeffs), nt_fwd, nt_bwd, zsc)
    if key in _KERNELS:
        return _KERNELS[key]

    DVH = DV
    JBF = DVH // (nt_fwd // C)
    NBF = DVH // JBF
    JBB = DV // (nt_bwd // C)
    NBB = DV // JBB
    DBB = D // (nt_bwd // C)

    if zsc:
        p0_src = f"""
            if (c == {NC} - 1) {{
                for (uint idx = tid; idx < D * DV; idx += NT)
                    dSw[sb + idx] = 0.0f;
            }}
        """
        ds0_src = """
        for (uint idx = tid; idx < D * DV; idx += NT)
            dS0[sb + idx] = dSw[sb + idx];
        """
    else:
        p0_src = f"""
            for (uint idx = tid; idx < D * DV; idx += NT) {{
                float ct = cot_Sall[sab + ((size_t)c + 1) * D * DV + idx];
                dSw[sb + idx] = (c == {NC} - 1) ? ct : dSw[sb + idx] + ct;
            }}
        """
        ds0_src = """
        for (uint idx = tid; idx < D * DV; idx += NT)
            dS0[sb + idx] = dSw[sb + idx] + cot_Sall[sab + idx];
        """

    cubic = _cubic_block(nsteps, ca_src, cb_src)
    wt_ds = _wt_ds_block()

    fwd_src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint JB = {JBF};
        constexpr uint NB = {NBF};
        constexpr uint NT = {nt_fwd};
        constexpr uint NG = {NG};
        constexpr float LAM = {float(lam):.8f}f;
        constexpr float U2 = {u2:.8f}f;
        constexpr int GIDX[{NC}] = {{ {gidx_src} }};
        size_t cb = (size_t)bh * {NC};
        size_t sb = (size_t)bh * D * DV;
        size_t sab = (size_t)bh * ({NC} + 1) * D * DV;

        threadgroup float vt[C * DV];

        for (uint idx = tid; idx < D * DV; idx += NT)
            Sw[sb + idx] = S0[sb + idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < {NC}; c++) {{
            size_t cbh = cb + c;
            for (uint idx = tid; idx < D * DV; idx += NT)
                Sall[sab + (size_t)c * D * DV + idx] = Sw[sb + idx];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++)
                    acc[jj] = u[(cbh * C + i) * DV + jb + jj];
                for (uint d = 0; d < D; d++) {{
                    float sv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] -= sv * Sw[sb + d * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    vt[i * DV + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = 0.0f;
                for (uint d = 0; d < D; d++) {{
                    float sv = qe[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += sv * Sw[sb + d * DV + jb + jj];
                }}
                for (uint l = 0; l < C; l++) {{
                    float av = Aqk[(cbh * C + i) * C + l];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += av * vt[l * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    o[(cbh * C + i) * DV + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u_ = tid; u_ < D * NB; u_ += NT) {{
                uint d = u_ / NB, jb = (u_ % NB) * JB;
                float ev = egl[cbh * D + d];
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++)
                    acc[jj] = Sw[sb + d * DV + jb + jj] * ev;
                for (uint l = 0; l < C; l++) {{
                    float kv = kd[(cbh * C + l) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += kv * vt[l * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    Sw[sb + d * DV + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
{cubic}
        }}
        for (uint idx = tid; idx < D * DV; idx += NT)
            Sall[sab + (size_t){NC} * D * DV + idx] = Sw[sb + idx];
    """

    bwd_src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint JB = {JBB};
        constexpr uint NB = {NBB};
        constexpr uint DB = {DBB};
        constexpr uint NDB = D / DB;
        constexpr uint NT = {nt_bwd};
        constexpr uint NG = {NG};
        constexpr int GIDX[{NC}] = {{ {gidx_src} }};
        size_t cb = (size_t)bh * {NC};
        size_t sb = (size_t)bh * D * DV;
        size_t sab = (size_t)bh * ({NC} + 1) * D * DV;

        threadgroup float vt[C * DV];
        threadgroup float dvt[C * DV];

        for (int c = {NC} - 1; c >= 0; c--) {{
            size_t cbh = cb + c;
{p0_src}
            threadgroup_barrier(mem_flags::mem_threadgroup);
{wt_ds}
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++)
                    acc[jj] = u[(cbh * C + i) * DV + jb + jj];
                for (uint d = 0; d < D; d++) {{
                    float sv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] -= sv * Sall[sab + (size_t)c * D * DV + d * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    vt[i * DV + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = 0.0f;
                for (uint l = 0; l < C; l++) {{
                    float av = Aqk[(cbh * C + l) * C + i];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += av * cot_o[(cbh * C + l) * DV + jb + jj];
                }}
                for (uint d = 0; d < D; d++) {{
                    float kv = kd[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += kv * dSw[sb + d * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    dvt[i * DV + jb + jj] = acc[jj];
            }}
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint i = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float dv_ = cot_o[(cbh * C + i) * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += dv_ * Sall[sab + (size_t)c * D * DV + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++)
                    dqe[(cbh * C + i) * D + db + dd] = acc[dd];
            }}
            for (uint u_ = tid; u_ < C * C; u_ += NT) {{
                uint i = u_ / C, l = u_ % C;
                float acc = 0.0f;
                for (uint j = 0; j < DV; j++)
                    acc += cot_o[(cbh * C + i) * DV + j] * vt[l * DV + j];
                dAqk[(cbh * C + i) * C + l] = acc;
            }}
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint l = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float vv = vt[l * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += vv * dSw[sb + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++)
                    dkd[(cbh * C + l) * D + db + dd] = acc[dd];
            }}
            for (uint d = tid; d < D; d += NT) {{
                float acc = 0.0f;
                for (uint j = 0; j < DV; j++)
                    acc += Sall[sab + (size_t)c * D * DV + d * DV + j] * dSw[sb + d * DV + j];
                degl[cbh * D + d] = acc;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint u_ = tid; u_ < D * NB; u_ += NT) {{
                uint d = u_ / NB, jb = (u_ % NB) * JB;
                float ev = egl[cbh * D + d];
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++)
                    acc[jj] = dSw[sb + d * DV + jb + jj] * ev;
                for (uint i = 0; i < C; i++) {{
                    float qv = qe[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] += qv * cot_o[(cbh * C + i) * DV + jb + jj];
                }}
                for (uint i = 0; i < C; i++) {{
                    float wv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] -= wv * dvt[i * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++)
                    dSw[sb + d * DV + jb + jj] = acc[jj];
            }}
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint i = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float tv = dvt[i * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += tv * Sall[sab + (size_t)c * D * DV + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++)
                    dw[(cbh * C + i) * D + db + dd] = -acc[dd];
            }}
            for (uint idx = tid; idx < C * DV; idx += NT)
                du[cbh * C * DV + idx] = dvt[idx];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
{ds0_src}
    """

    bits = 0
    for i, g in enumerate(gate_tuple):
        if g:
            bits |= 1 << i
    k_fwd = mx.fast.metal_kernel(
        name=f"kda_scan_eg_fwd_{NC}_{C}_{D}_{DV}_{bits}_{nt_fwd}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall", "Wg", "Sw", "Xb", "Ab", "Tmp"],
        source=fwd_src,
    )
    bwd_inputs = ["qe", "w", "u", "Aqk", "kd", "egl", "Sall", "cot_o", "Wg"]
    if not zsc:
        bwd_inputs.append("cot_Sall")
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_scan_eg_bwd_{NC}_{C}_{D}_{DV}_{bits}_{nt_bwd}{'_zsc' if zsc else ''}",
        input_names=bwd_inputs,
        output_names=[
            "dqe",
            "dw",
            "du",
            "dAqk",
            "dkd",
            "degl",
            "dS0",
            "dSw",
            "dStmp",
        ],
        source=bwd_src,
    )
    _KERNELS[key] = (k_fwd, k_bwd, NG)
    return _KERNELS[key]


def _scan_op_factory(
    NC: int,
    C: int,
    D: int,
    DV: int,
    gate_tuple: tuple[int, ...],
    lam: float,
    coeffs: list[tuple[float, float]],
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
    zsc: bool = False,
):
    k_fwd, k_bwd, NG = _build(
        NC, C, D, DV, gate_tuple, lam, coeffs, nt_fwd, nt_bwd, zsc
    )

    @mx.custom_function
    def _op(qe, w, u, Aqk, kd, egl, S0):
        B, H = qe.shape[0], qe.shape[1]
        o, Sall, Wg, *_scratch = k_fwd(
            inputs=[qe, w, u, Aqk, kd, egl, S0],
            output_shapes=[
                (B, H, NC, C, DV),
                (B, H, NC + 1, D, DV),
                (B, H, NG, D, D),
                (B, H, D, DV),
                (B, H, D, DV),
                (B, H, D, D),
                (B, H, D, DV),
            ],
            output_dtypes=[mx.float32] * 7,
            grid=(nt_fwd, B * H, 1),
            threadgroup=(nt_fwd, 1, 1),
        )
        return o, Sall, Wg

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, w, u, Aqk, kd, egl, _S0 = primals
        cot_o = cotangent[0]
        B, H = qe.shape[0], qe.shape[1]
        Wg = output[2]
        ins = [qe, w, u, Aqk, kd, egl, output[1], cot_o, Wg]
        if not zsc:
            ins.append(cotangent[1])
        dqe, dw, du, dAqk, dkd, degl, dS0, _dSw, _dt = k_bwd(
            inputs=ins,
            output_shapes=[
                (B, H, NC, C, D),
                (B, H, NC, C, D),
                (B, H, NC, C, DV),
                (B, H, NC, C, C),
                (B, H, NC, C, D),
                (B, H, NC, D),
                (B, H, D, DV),
                (B, H, D, DV),
                (B, H, D, DV),
            ],
            output_dtypes=[mx.float32] * 9,
            grid=(nt_bwd, B * H, 1),
            threadgroup=(nt_bwd, 1, 1),
        )
        return [dqe, dw, du, dAqk, dkd, degl, dS0]

    return _op


def kda_scan_eg_metal(
    qe,
    w,
    u,
    Aqk,
    kd,
    egl,
    S0,
    gate_tuple: tuple[int, ...],
    lam: float = 1.0,
    zsc: bool = False,
):
    """门控融合扫描。返回 (o, Sall)；Wg 留在 custom_function tape 里供 VJP。"""
    NC, C, D = int(qe.shape[2]), int(qe.shape[3]), int(qe.shape[4])
    DV = int(u.shape[-1])
    coeffs = cubic_coeffs()
    key = (NC, C, D, DV, gate_tuple, float(lam), tuple(coeffs), zsc)
    op = _OPS.get(key)
    if op is None:
        op = _scan_op_factory(
            NC, C, D, DV, gate_tuple, float(lam), coeffs, zsc=zsc
        )
        _OPS[key] = op
    o, Sall, _Wg = op(qe, w, u, Aqk, kd, egl, S0)
    return o, Sall
