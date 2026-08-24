"""KDA 跨 chunk 状态扫描的融合 Metal kernel（fwd + 手写 VJP）。

替换 `_kda_scan`（model/kda.py）的 NC 步 Python 循环（每 chunk ~7 个图
节点、NC=64 时数百次 kernel 发射，实测 fwd 6.5ms / fwd+bwd ~27ms 每层）。
融合后 fwd/bwd 各 1 次发射，顺序依赖收进单个 threadgroup 内的循环。

数学（与 eager 参考逐项一致，全部 f32）：
  fwd  每 chunk：vt = u − w@S；o = qe@S + Aqk@vt；S = egl⊙S + kdᵀ@vt；
       Sall[c] 记录 chunk 入态（供 VJP 复用，免重算）。
  bwd  逆时间：vt 重算；dvt = Aqkᵀ@do + kd@dS；dqe = do@Scᵀ；
       dAqk = do@vtᵀ；dw = −dvt@Scᵀ；du = dvt；dkd = (dS@vtᵀ)ᵀ；
       degl = Σ_dv Sc⊙dS；dS = qeᵀ@do − wᵀ@dvt + egl⊙dS。

性能要点（朴素"每线程一个输出元素"版本与 eager 同速的教训）：
- 寄存器分块：每线程一次算 JB=6 个相邻输出，输入元素（w/qe/kd/Aqk/…）
  复用 6×，把 L2 重读流量压到接近 compulsory；分块数按 NT/C 对齐，
  使所有线程恰好满载（units == NT 或其整数倍）。
- fwd：threadgroup 负责 (bh, Dv 半片)，S 半片 96×48 f32 = 18KB 常驻
  threadgroup（Dv 全片 36KB 超 32KB 上限）；o/Sall 按半片写，无跨片归约。
- bwd：dqe/dw/dkd/degl 沿 Dv 全维归约，分片会引入跨组 partial，故
  threadgroup 负责整个 bh；dS 放 global scratch（L2 常驻），TG 只留
  vt/dvt 两片 12KB；Sc(Sall[c]) 从 global/L2 读。

首调用按形状键做 fwd+bwd 在线校验（对照 eager 参考），失败永久回退
eager（mx.compile trace 内无 host sync，需由调用方 compile 前 prewarm）。
"""

import mlx.core as mx

_KERNELS: dict = {}
_OPS: dict = {}
_VERIFIED: set = set()
_DISABLED = False

# 发射几何默认值（由 sweep_kda_scan.py 在真实形状上实测选定）。这三个数
# 只改变工作在线程/线程组间的分配，不改变任何输出元素的累加顺序，因此
# 不同取值之间逐位等价。
_DV_SPLIT = 2
_NT_FWD = 256
_NT_BWD = 256


def _build(
    NC: int,
    C: int,
    D: int,
    DV: int,
    dv_split: int = _DV_SPLIT,
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
):
    """按 (NC,C,D,DV,dv_split,nt_fwd,nt_bwd) 构建 fwd/bwd kernel。

    要求：DV % dv_split == 0；DVH % (nt_fwd//C) == 0、D % (nt_bwd//C) == 0、
    DV % (nt_bwd//C) == 0（寄存器分块对齐，由调度器守卫）。dv_split 越大
    并行组数越多、每核驻留组数越多（TG 占用随分片减小），代价是
    w/qe/kd/Aqk 等输入被多分片重复读取。"""
    key = (NC, C, D, DV, dv_split, nt_fwd, nt_bwd)
    if key in _KERNELS:
        return _KERNELS[key]
    DVH = DV // dv_split
    JBF = DVH // (nt_fwd // C)  # fwd 寄存器分块宽
    NBF = DVH // JBF  # fwd j 块数
    JBB = DV // (nt_bwd // C)  # bwd 寄存器分块宽
    NBB = DV // JBB  # bwd j 块数
    DBB = D // (nt_bwd // C)  # bwd d 分块宽

    fwd_src = f"""
        uint tid = thread_position_in_grid.x;
        uint bh  = thread_position_in_grid.y;
        uint dvh = thread_position_in_grid.z;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint DV = {DV};
        constexpr uint DVH = {DVH};
        constexpr uint JB = {JBF};
        constexpr uint NB = {NBF};
        constexpr uint NT = {nt_fwd};
        size_t cb = (size_t)bh * {NC};
        size_t sb = (size_t)bh * D * DV;
        size_t sab = (size_t)bh * ({NC} + 1) * D * DV;
        uint j0 = dvh * DVH;

        threadgroup float S[D * DVH];
        threadgroup float vt[C * DVH];

        for (uint idx = tid; idx < D * DVH; idx += NT)
            S[idx] = S0[sb + (idx / DVH) * DV + j0 + (idx % DVH)];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < {NC}; c++) {{
            size_t cbh = cb + c;
            // Sall[c] = 入态
            for (uint idx = tid; idx < D * DVH; idx += NT)
                Sall[sab + ((size_t)c * D + idx / DVH) * DV + j0 + (idx % DVH)] = S[idx];
            // P1: vt = u − w@S（(i, j块) 寄存器分块，w 复用 JB×）
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = u[(cbh * C + i) * DV + j0 + jb + jj];
                for (uint d = 0; d < D; d++) {{
                    float sv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] -= sv * S[d * DVH + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) vt[i * DVH + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P2: o = qe@S + Aqk@vt
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = 0.0f;
                for (uint d = 0; d < D; d++) {{
                    float sv = qe[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += sv * S[d * DVH + jb + jj];
                }}
                for (uint l = 0; l < C; l++) {{
                    float av = Aqk[(cbh * C + i) * C + l];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += av * vt[l * DVH + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) o[(cbh * C + i) * DV + j0 + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3: S = egl⊙S + kdᵀ@vt（(d, j块) 分块，kd 复用 JB×）
            for (uint u_ = tid; u_ < D * NB; u_ += NT) {{
                uint d = u_ / NB, jb = (u_ % NB) * JB;
                float ev = egl[cbh * D + d];
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = S[d * DVH + jb + jj] * ev;
                for (uint l = 0; l < C; l++) {{
                    float kv = kd[(cbh * C + l) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += kv * vt[l * DVH + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) S[d * DVH + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint idx = tid; idx < D * DVH; idx += NT)
            Sall[sab + ((size_t){NC} * D + idx / DVH) * DV + j0 + (idx % DVH)] = S[idx];
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
        size_t cb = (size_t)bh * {NC};
        size_t sb = (size_t)bh * D * DV;
        size_t sab = (size_t)bh * ({NC} + 1) * D * DV;

        threadgroup float vt[C * DV];
        threadgroup float dvt[C * DV];

        for (int c = {NC} - 1; c >= 0; c--) {{
            size_t cbh = cb + c;
            // P0: dS += cot_Sall[c+1]（首轮直接载入，免零初始化）
            for (uint idx = tid; idx < D * DV; idx += NT) {{
                float ct = cot_Sall[sab + ((size_t)c + 1) * D * DV + idx];
                dSw[sb + idx] = (c == {NC} - 1) ? ct : dSw[sb + idx] + ct;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P1: vt = u − w@Sc（重算；Sc = Sall[c]，(i, j块) 分块）
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = u[(cbh * C + i) * DV + jb + jj];
                for (uint d = 0; d < D; d++) {{
                    float sv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++)
                        acc[jj] -= sv * Sall[sab + (size_t)c * D * DV + d * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) vt[i * DV + jb + jj] = acc[jj];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P2a: dvt = Aqkᵀ@do + kd@dS
            for (uint u_ = tid; u_ < C * NB; u_ += NT) {{
                uint i = u_ / NB, jb = (u_ % NB) * JB;
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = 0.0f;
                for (uint l = 0; l < C; l++) {{
                    float av = Aqk[(cbh * C + l) * C + i];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += av * cot_o[(cbh * C + l) * DV + jb + jj];
                }}
                for (uint d = 0; d < D; d++) {{
                    float kv = kd[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += kv * dSw[sb + d * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) dvt[i * DV + jb + jj] = acc[jj];
            }}
            // P2b: dqe = do@Scᵀ（(i, d块) 分块）
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint i = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float dv_ = cot_o[(cbh * C + i) * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += dv_ * Sall[sab + (size_t)c * D * DV + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++) dqe[(cbh * C + i) * D + db + dd] = acc[dd];
            }}
            // P2c: dAqk = do@vtᵀ
            for (uint u_ = tid; u_ < C * C; u_ += NT) {{
                uint i = u_ / C, l = u_ % C;
                float acc = 0.0f;
                for (uint j = 0; j < DV; j++)
                    acc += cot_o[(cbh * C + i) * DV + j] * vt[l * DV + j];
                dAqk[(cbh * C + i) * C + l] = acc;
            }}
            // P2d: dkd = (dS@vtᵀ)ᵀ → dkd[l,d] = Σ_j dS[d,j]·vt[l,j]
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint l = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float vv = vt[l * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += vv * dSw[sb + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++) dkd[(cbh * C + l) * D + db + dd] = acc[dd];
            }}
            // P2e: degl[d] = Σ_j Sc[d,j]·dS[d,j]
            for (uint d = tid; d < D; d += NT) {{
                float acc = 0.0f;
                for (uint j = 0; j < DV; j++)
                    acc += Sall[sab + (size_t)c * D * DV + d * DV + j] * dSw[sb + d * DV + j];
                degl[cbh * D + d] = acc;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3a: dS = qeᵀ@do − wᵀ@dvt + egl⊙dS（(d, j块) 分块）
            for (uint u_ = tid; u_ < D * NB; u_ += NT) {{
                uint d = u_ / NB, jb = (u_ % NB) * JB;
                float ev = egl[cbh * D + d];
                float acc[JB];
                for (uint jj = 0; jj < JB; jj++) acc[jj] = dSw[sb + d * DV + jb + jj] * ev;
                for (uint i = 0; i < C; i++) {{
                    float qv = qe[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] += qv * cot_o[(cbh * C + i) * DV + jb + jj];
                }}
                for (uint i = 0; i < C; i++) {{
                    float wv = w[(cbh * C + i) * D + d];
                    for (uint jj = 0; jj < JB; jj++) acc[jj] -= wv * dvt[i * DV + jb + jj];
                }}
                for (uint jj = 0; jj < JB; jj++) dSw[sb + d * DV + jb + jj] = acc[jj];
            }}
            // P3b: dw = −dvt@Scᵀ（(i, d块) 分块）
            for (uint u_ = tid; u_ < C * NDB; u_ += NT) {{
                uint i = u_ / NDB, db = (u_ % NDB) * DB;
                float acc[DB];
                for (uint dd = 0; dd < DB; dd++) acc[dd] = 0.0f;
                for (uint j = 0; j < DV; j++) {{
                    float tv = dvt[i * DV + j];
                    for (uint dd = 0; dd < DB; dd++)
                        acc[dd] += tv * Sall[sab + (size_t)c * D * DV + (db + dd) * DV + j];
                }}
                for (uint dd = 0; dd < DB; dd++) dw[(cbh * C + i) * D + db + dd] = -acc[dd];
            }}
            // P3c: du = dvt
            for (uint idx = tid; idx < C * DV; idx += NT)
                du[cbh * C * DV + idx] = dvt[idx];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        // dS0 = dS + cot_Sall[0]
        for (uint idx = tid; idx < D * DV; idx += NT)
            dS0[sb + idx] = dSw[sb + idx] + cot_Sall[sab + idx];
    """

    k_fwd = mx.fast.metal_kernel(
        name=f"kda_scan_fwd_{NC}_{C}_{D}_{DV}_{dv_split}_{nt_fwd}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall"],
        source=fwd_src,
    )
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_scan_bwd_{NC}_{C}_{D}_{DV}_{nt_bwd}",
        input_names=[
            "qe",
            "w",
            "u",
            "Aqk",
            "kd",
            "egl",
            "Sall",
            "cot_o",
            "cot_Sall",
        ],
        output_names=["dqe", "dw", "du", "dAqk", "dkd", "degl", "dS0", "dSw"],
        source=bwd_src,
    )
    _KERNELS[key] = (k_fwd, k_bwd)
    return _KERNELS[key]


def _scan_op_factory(
    NC: int,
    C: int,
    D: int,
    DV: int,
    dv_split: int = _DV_SPLIT,
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
):
    k_fwd, k_bwd = _build(NC, C, D, DV, dv_split, nt_fwd, nt_bwd)

    @mx.custom_function
    def _op(qe, w, u, Aqk, kd, egl, S0):
        B, H = qe.shape[0], qe.shape[1]
        o, Sall = k_fwd(
            inputs=[qe, w, u, Aqk, kd, egl, S0],
            output_shapes=[
                (B, H, NC, C, DV),
                (B, H, NC + 1, D, DV),
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=(nt_fwd, B * H, dv_split),
            threadgroup=(nt_fwd, 1, 1),
        )
        return o, Sall

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, w, u, Aqk, kd, egl, _S0 = primals
        cot_o, cot_Sall = cotangent
        B, H = qe.shape[0], qe.shape[1]
        dqe, dw, du, dAqk, dkd, degl, dS0, _dSw = k_bwd(
            inputs=[qe, w, u, Aqk, kd, egl, output[1], cot_o, cot_Sall],
            output_shapes=[
                (B, H, NC, C, D),
                (B, H, NC, C, D),
                (B, H, NC, C, DV),
                (B, H, NC, C, C),
                (B, H, NC, C, D),
                (B, H, NC, D),
                (B, H, D, DV),
                (B, H, D, DV),
            ],
            output_dtypes=[mx.float32] * 8,
            grid=(nt_bwd, B * H, 1),
            threadgroup=(nt_bwd, 1, 1),
        )
        return [dqe, dw, du, dAqk, dkd, degl, dS0]

    return _op


def _supported(
    NC: int,
    C: int,
    D: int,
    DV: int,
    dv_split: int = _DV_SPLIT,
    nt_fwd: int = _NT_FWD,
    nt_bwd: int = _NT_BWD,
) -> bool:
    """形状守卫：分片/寄存器分块对齐与 TG 上限。"""
    if C == 0 or nt_fwd % C != 0 or nt_bwd % C != 0:
        return False
    if DV % dv_split != 0:
        return False
    DVH = DV // dv_split
    if DVH % (nt_fwd // C) != 0:
        return False
    nb = nt_bwd // C
    if DV % nb != 0 or D % nb != 0:
        return False
    tg_fwd = (D * DVH + C * DVH) * 4
    tg_bwd = 2 * C * DV * 4
    return tg_fwd <= 30720 and tg_bwd <= 30720


def kda_scan_metal(
    qe,
    w,
    u,
    Aqk,
    kd,
    egl,
    S0,
    dv_split: int = None,
    nt_fwd: int = None,
    nt_bwd: int = None,
):
    """融合扫描入口。仅接受 f32 且满足 _supported；由 _chunk_kda 侧的
    调度器（含在线校验与 eager 回退）调用。

    发射几何默认取模块级 _DV_SPLIT/_NT_FWD/_NT_BWD（sweep 脚本改这三个
    模块变量即可切换，无需改调用点）。"""
    dv_split = _DV_SPLIT if dv_split is None else dv_split
    nt_fwd = _NT_FWD if nt_fwd is None else nt_fwd
    nt_bwd = _NT_BWD if nt_bwd is None else nt_bwd
    NC, C, D = qe.shape[2], qe.shape[3], qe.shape[4]
    DV = u.shape[-1]
    key = (NC, C, D, DV, dv_split, nt_fwd, nt_bwd)
    op = _OPS.get(key)
    if op is None:
        op = _scan_op_factory(NC, C, D, DV, dv_split, nt_fwd, nt_bwd)
        _OPS[key] = op
    return op(qe, w, u, Aqk, kd, egl, S0)
