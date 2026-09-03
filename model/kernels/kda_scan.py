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

MMA 变体（默认，VIBY_KDA_SCAN_MMA=0 回退标量版）：标量「JB 寄存器分块」
版 fwd/bwd 实测都卡在 ~1.5 TF/s 的标量 FMA 天花板（H=16 口径 fwd 6.9ms /
bwd 17.0ms，几何 sweep 只有 2.9% 空间）。chunk 步的矩阵乘全部换成
simdgroup MMA（§5.5：操作数从 TG 载入 11-12 TF/s）后：
  fwd  每 TG 一个 (bh, Dv 半片)，S 半片 96×48 常驻 TG；NT=4·D → 12 个
       simdgroup，P1/P2 每 SG 一个 (M,N) tile 组合，P3 每 SG 一个 M tile
       且 acc 直写 S 与 Sall[c+1]；只 stage −w（P1 的取负），qe/Aqk
       device 直读作 A，kd 转置 device 载入并按 SG hoist。
  bwd  同样按 Dv 半片分 TG，dS 半片常驻 TG（旧版放 global scratch，
       内层循环两操作数都从 device 读，§5.5 最慢形态）。递推链只需
       dS/dvt（j 可分）；j 全维归约的 dqe/dw hoist 到 VJP 里的 MLX
       batched GEMM（do/du 拼接后单 GEMM，Sall 439MB 只流读一遍），
       dAqk/dkd/degl 按半片写部分和、MLX 求和。
实测（B=12 H=16 T=1024，probe_kda_scan_mma）：fwd 6.9→3.3ms、
bwd(zsc) 15.1→8.7ms、f+b 22.0→12.0ms（1.83×）；梯度对拍 rel ≤ 1e-6。
"""

import os

import mlx.core as mx

_KERNELS: dict = {}
_OPS: dict = {}
_VERIFIED: set = set()
_DISABLED = False

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# VIBY_KDA_SCAN_MMA=0：回退标量 JB 分块版（A/B 对照用）
_MMA_DISABLED = os.environ.get("VIBY_KDA_SCAN_MMA", "1") != "1"

# 发射几何默认值（由 sweep_kda_scan.py 在真实形状上实测选定）。这三个数
# 只改变工作在线程/线程组间的分配，不改变任何输出元素的累加顺序，因此
# 不同取值之间逐位等价。仅标量版使用。
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
    zsc: bool = False,
):
    """按 (NC,C,D,DV,dv_split,nt_fwd,nt_bwd,zsc) 构建 fwd/bwd kernel。

    要求：DV % dv_split == 0；DVH % (nt_fwd//C) == 0、D % (nt_bwd//C) == 0、
    DV % (nt_bwd//C) == 0（寄存器分块对齐，由调度器守卫）。dv_split 越大
    并行组数越多、每核驻留组数越多（TG 占用随分片减小），代价是
    w/qe/kd/Aqk 等输入被多分片重复读取。zsc=True 构建「状态 cotangent 恒零」
    的 bwd 变体（不读 cot_Sall；与通用版在 cot_Sall=0 时逐位等价）。"""
    key = (NC, C, D, DV, dv_split, nt_fwd, nt_bwd, zsc)
    if key in _KERNELS:
        return _KERNELS[key]
    DVH = DV // dv_split
    JBF = DVH // (nt_fwd // C)  # fwd 寄存器分块宽
    NBF = DVH // JBF  # fwd j 块数
    JBB = DV // (nt_bwd // C)  # bwd 寄存器分块宽
    NBB = DV // JBB  # bwd j 块数
    DBB = D // (nt_bwd // C)  # bwd d 分块宽

    # ZSC（zero state-cotangent）bwd 变体：训练图里 Sall 的 cotangent 恒为零
    # （_chunk_kda 只经 Sall[:, :, NC] 暴露末态，训练时末态不被下游消费），
    # 省掉 cot_Sall 零张量的物化与逐 chunk 切片读。通用版此处做 dSw += ct，
    # ct=0 时首轮等价于零初始化、其余迭代为 +0.0f 恒等（x+0.0f 逐位等于 x），
    # 故两变体在 cot_Sall=0 下逐位等价。
    if zsc:
        p0_src = f"""
            // P0: cot_Sall 恒零（ZSC）：首轮零初始化 dSw，其余迭代跳过
            if (c == {NC} - 1) {{
                for (uint idx = tid; idx < D * DV; idx += NT)
                    dSw[sb + idx] = 0.0f;
            }}
        """
        ds0_src = """
        // dS0 = dS（cot_Sall[0] 恒零）
        for (uint idx = tid; idx < D * DV; idx += NT)
            dS0[sb + idx] = dSw[sb + idx];
        """
    else:
        p0_src = f"""
            // P0: dS += cot_Sall[c+1]（首轮直接载入，免零初始化）
            for (uint idx = tid; idx < D * DV; idx += NT) {{
                float ct = cot_Sall[sab + ((size_t)c + 1) * D * DV + idx];
                dSw[sb + idx] = (c == {NC} - 1) ? ct : dSw[sb + idx] + ct;
            }}
        """
        ds0_src = """
        // dS0 = dS + cot_Sall[0]
        for (uint idx = tid; idx < D * DV; idx += NT)
            dS0[sb + idx] = dSw[sb + idx] + cot_Sall[sab + idx];
        """

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
{p0_src}
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
{ds0_src}
    """

    k_fwd = mx.fast.metal_kernel(
        name=f"kda_scan_fwd_{NC}_{C}_{D}_{DV}_{dv_split}_{nt_fwd}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall"],
        source=fwd_src,
    )
    bwd_inputs = ["qe", "w", "u", "Aqk", "kd", "egl", "Sall", "cot_o"]
    if not zsc:
        bwd_inputs.append("cot_Sall")
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_scan_bwd_{NC}_{C}_{D}_{DV}_{nt_bwd}{'_zsc' if zsc else ''}",
        input_names=bwd_inputs,
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
    zsc: bool = False,
):
    k_fwd, k_bwd = _build(NC, C, D, DV, dv_split, nt_fwd, nt_bwd, zsc)

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
        cot_o = cotangent[0]
        B, H = qe.shape[0], qe.shape[1]
        # zsc：cotangent[1]（Sall 的协梯度）按约定恒零，不进 kernel 输入表；
        # compile 下该 lazy zeros 被 DCE 掉，连同其物化一起消除。
        ins = [qe, w, u, Aqk, kd, egl, output[1], cot_o]
        if not zsc:
            ins.append(cotangent[1])
        dqe, dw, du, dAqk, dkd, degl, dS0, _dSw = k_bwd(
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
            ],
            output_dtypes=[mx.float32] * 8,
            grid=(nt_bwd, B * H, 1),
            threadgroup=(nt_bwd, 1, 1),
        )
        return [dqe, dw, du, dAqk, dkd, degl, dS0]

    return _op


def _mma_ok(NC: int, C: int, D: int, DV: int) -> bool:
    """MMA 变体形状守卫。不满足时走标量 JB 分块版（数值同一容差）。

    结构假设：C=16（2 个 M tile，A tile 按 SG hoist 成对展开）；D=DV 且
    D%16=0（NT=4·D → NSG=D/8 个 simdgroup，P1/P2 的 (m,n) 映射要求
    NSG = 2·(DVH/8)）；TG 预算 fwd/bwd 各 ≤ 30720B。"""
    if _MMA_DISABLED or NC <= 0:
        return False
    if C != 16 or D != DV or D % 16 != 0:
        return False
    nt = 4 * D
    if nt > 1024:
        return False
    dvh = DV // 2
    tg_fwd = (D * dvh + C * dvh + C * (D + 4) + D) * 4
    tg_bwd = (D * (dvh + 4) + C * (dvh + 4) + C * dvh + D) * 4
    return tg_fwd <= 30720 and tg_bwd <= 30720


def _build_mma(NC: int, C: int, D: int, DV: int, zsc: bool):
    """simdgroup MMA 版 fwd/bwd（设计与实测见模块 docstring）。"""
    key = ("mma", NC, C, D, DV, zsc)
    if key in _KERNELS:
        return _KERNELS[key]
    DVH = DV // 2
    NT = 4 * D
    SD = D + 4  # −w staging 的行距（转置载入 kd 时 8 行错开 4 bank）
    SW = DVH + 4  # bwd dS/vt 行距（转置载入 dSᵀ/vtᵀ）

    fwd_src = f"""
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

        threadgroup float S[D * DVH];
        threadgroup float vt[C * DVH];
        threadgroup float Wst[C * SD];
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

            for (uint i = tid; i < C * D; i += NT)
                Wst[(i / D) * SD + (i % D)] = -w_c[i];
            if (tid < D) egls[tid] = egl[cbh * D + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // P1: vt = u + (−w)@S（acc 以 u tile 初始化，SG → (m,n)）
            {{
                uint m = sg & 1, n = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                simdgroup_load(acc, u_c + m * 8 * DV + j0 + n * 8, DV);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + n * 8, DVH);
                    simdgroup_load(Af, Wst + m * 8 * SD + kk * 8, SD);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, vt + m * 8 * DVH + n * 8, DVH);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P2: o = qe@S + Aqk@vt（qe/Aqk device 直读作 A）
            {{
                uint m = sg & 1, n = sg >> 1;
                simdgroup_matrix<float, 8, 8> acc, Af, Bf;
                acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kk = 0; kk < D / 8; kk++) {{
                    simdgroup_load(Bf, S + kk * 8 * DVH + n * 8, DVH);
                    simdgroup_load(Af, qe_c + m * 8 * D + kk * 8, D);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                for (uint kk = 0; kk < C / 8; kk++) {{
                    simdgroup_load(Bf, vt + kk * 8 * DVH + n * 8, DVH);
                    simdgroup_load(Af, Aqk_c + m * 8 * C + kk * 8, C);
                    simdgroup_multiply_accumulate(acc, Af, Bf, acc);
                }}
                simdgroup_store(acc, o_c + m * 8 * DV + j0 + n * 8, DV);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // S ⊙= egl（NT=4·D：每 4 线程一行，免热路径除法）
            {{
                uint d = tid >> 2;
                float ev = egls[d];
                uint jb = (tid & 3) * (DVH / 4);
                for (uint j = 0; j < DVH / 4; j++)
                    S[d * DVH + jb + j] *= ev;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P3: S += kdᵀ@vt（SG → M tile；kdᵀ 转置 device 载入并 hoist；
            // acc 直写 S 与 Sall[c+1]，省下一 chunk 头部的 TG 重读）
            {{
                simdgroup_matrix<float, 8, 8> acc, A0, A1, Bf;
                simdgroup_load(A0, kd_c + 0 * D + sg * 8, D, ulong2(0, 0), true);
                simdgroup_load(A1, kd_c + 8 * D + sg * 8, D, ulong2(0, 0), true);
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

    if zsc:
        ds_init = """
        for (uint i = tid; i < D * DVH; i += NT)
            dS[(i / DVH) * SW + (i % DVH)] = 0.0f;
        """
        ds_add = ""
        ds0_src = """
        for (uint i = tid; i < D * DVH; i += NT)
            dS0[(size_t)bh * D * DV + (i / DVH) * DV + j0 + (i % DVH)] =
                dS[(i / DVH) * SW + (i % DVH)];
        """
    else:
        ds_init = f"""
        for (uint i = tid; i < D * DVH; i += NT)
            dS[(i / DVH) * SW + (i % DVH)] =
                cot_Sall[sab + (size_t){NC} * D * DV + (i / DVH) * DV + j0 + (i % DVH)];
        """
        # 放在 P1b 相（与 vt 修正并行，写 dS / 写 vt 互不相干；上一轮
        # P3a 的尾 barrier 与本轮 P1b 后的 barrier 共同保证可见性）
        ds_add = f"""
            if (c != {NC} - 1)
                for (uint i = tid; i < D * DVH; i += NT)
                    dS[(i / DVH) * SW + (i % DVH)] +=
                        cot_Sall[sab + ((size_t)c + 1) * D * DV + (i / DVH) * DV + j0 + (i % DVH)];
        """
        ds0_src = """
        for (uint i = tid; i < D * DVH; i += NT)
            dS0[(size_t)bh * D * DV + (i / DVH) * DV + j0 + (i % DVH)] =
                dS[(i / DVH) * SW + (i % DVH)]
                + cot_Sall[sab + (size_t)(i / DVH) * DV + j0 + (i % DVH)];
        """

    bwd_src = f"""
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
        constexpr uint SW = {SW};
        uint j0 = dvh * DVH;
        size_t cb = (size_t)bh * NC;
        size_t sab = (size_t)bh * (NC + 1) * D * DV;
        size_t pb2 = ((size_t)bh * 2 + dvh) * NC;

        threadgroup float dS[D * SW];
        threadgroup float vt[C * SW];
        threadgroup float ndvt[C * DVH];
        threadgroup float egls[D];

{ds_init}
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

            // P1a: vt ← w@Sc
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
            // P1b: vt = u − vt；（非 ZSC）dS += cot_Sall[c+1]
            for (uint i = tid; i < C * DVH; i += NT) {{
                uint r = i / DVH, col = i % DVH;
                vt[r * SW + col] = u_c[r * DV + j0 + col] - vt[r * SW + col];
            }}
{ds_add}
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
            // P2c: dAqk_p = do@vtᵀ（M=N=C：4 tile，sg<4）
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
            // P2d: dkd_p = vt@dSᵀ（每 SG 一个 N tile，M 两片）
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
            // P2e: degl_p[d] = Σ_{{j∈半片}} Sc[d,j]·dS[d,j]
            if (tid < D) {{
                float acc = 0.0f;
                for (uint j = 0; j < DVH; j++)
                    acc += sc[tid * DV + j0 + j] * dS[tid * SW + j];
                degl_p[(pb2 + c) * D + tid] = acc;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // EL: du 写出 + dvt 取负（P3a 的减法折进数据）+ dS ⊙= egl
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
            // P3a: dS += qeᵀ@do + wᵀ@(−dvt)（A 转置 device 载入按 SG hoist）
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
{ds0_src}
    """

    k_fwd = mx.fast.metal_kernel(
        name=f"kda_scan_mma_fwd_{NC}_{C}_{D}_{DV}_{NT}",
        input_names=["qe", "w", "u", "Aqk", "kd", "egl", "S0"],
        output_names=["o", "Sall"],
        source=fwd_src,
        header=_HEADER,
    )
    bwd_inputs = ["qe", "w", "u", "Aqk", "kd", "egl", "Sall", "cot_o"]
    if not zsc:
        bwd_inputs.append("cot_Sall")
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_scan_mma_bwd_{NC}_{C}_{D}_{DV}_{NT}{'_zsc' if zsc else ''}",
        input_names=bwd_inputs,
        output_names=["du", "dAqk_p", "dkd_p", "degl_p", "dS0"],
        source=bwd_src,
        header=_HEADER,
    )
    _KERNELS[key] = (k_fwd, k_bwd)
    return _KERNELS[key]


def _mma_op_factory(NC: int, C: int, D: int, DV: int, zsc: bool = False):
    k_fwd, k_bwd = _build_mma(NC, C, D, DV, zsc)
    NT = 4 * D

    @mx.custom_function
    def _op(qe, w, u, Aqk, kd, egl, S0):
        B, H = qe.shape[0], qe.shape[1]
        o, Sall = k_fwd(
            inputs=[qe, w, u, Aqk, kd, egl, S0],
            output_shapes=[(B, H, NC, C, DV), (B, H, NC + 1, D, DV)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(NT, B * H, 2),
            threadgroup=(NT, 1, 1),
        )
        return o, Sall

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, w, u, Aqk, kd, egl, _S0 = primals
        cot_o = cotangent[0]
        Sall = output[1]
        B, H = qe.shape[0], qe.shape[1]
        ins = [qe, w, u, Aqk, kd, egl, Sall, cot_o]
        if not zsc:
            ins.append(cotangent[1])
        du, dAqk_p, dkd_p, degl_p, dS0 = k_bwd(
            inputs=ins,
            output_shapes=[
                (B, H, NC, C, DV),
                (B, H, 2, NC, C, C),
                (B, H, 2, NC, C, D),
                (B, H, 2, NC, D),
                (B, H, D, DV),
            ],
            output_dtypes=[mx.float32] * 5,
            grid=(NT, B * H, 2),
            threadgroup=(NT, 1, 1),
        )
        # dqe/dw 不依赖递推链，hoist 成 batched GEMM；do/du 拼接后单 GEMM
        # 使 Sall（最大单份流量）只流读一遍。
        ScT = mx.swapaxes(Sall[:, :, :NC], -1, -2)
        dodu = mx.concatenate([cot_o, du], axis=3) @ ScT
        dqe = dodu[..., :C, :]
        dw = -dodu[..., C:, :]
        dAqk = dAqk_p[:, :, 0] + dAqk_p[:, :, 1]
        dkd = dkd_p[:, :, 0] + dkd_p[:, :, 1]
        degl = degl_p[:, :, 0] + degl_p[:, :, 1]
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
    zsc: bool = False,
):
    """融合扫描入口。仅接受 f32 且满足 _supported；由 _chunk_kda 侧的
    调度器（含在线校验与 eager 回退）调用。zsc=True 走「状态 cotangent
    恒零」bwd 变体（调用方须保证该条件，训练路径由 _chunk_kda 断言）。

    形状满足 _mma_ok 且未显式指定发射几何时走 MMA 变体（与标量版数值
    非逐位一致——GEMM 累加顺序不同，但同在 eager 对照容差内）；显式传
    dv_split/nt_* 强制标量版（sweep 脚本用）。"""
    explicit_geom = dv_split is not None or nt_fwd is not None or nt_bwd is not None
    NC, C, D = qe.shape[2], qe.shape[3], qe.shape[4]
    DV = u.shape[-1]
    if not explicit_geom and _mma_ok(NC, C, D, DV):
        key = ("mma", NC, C, D, DV, zsc)
        op = _OPS.get(key)
        if op is None:
            op = _mma_op_factory(NC, C, D, DV, zsc)
            _OPS[key] = op
        return op(qe, w, u, Aqk, kd, egl, S0)
    dv_split = _DV_SPLIT if dv_split is None else dv_split
    nt_fwd = _NT_FWD if nt_fwd is None else nt_fwd
    nt_bwd = _NT_BWD if nt_bwd is None else nt_bwd
    key = (NC, C, D, DV, dv_split, nt_fwd, nt_bwd, zsc)
    op = _OPS.get(key)
    if op is None:
        op = _scan_op_factory(NC, C, D, DV, dv_split, nt_fwd, nt_bwd, zsc)
        _OPS[key] = op
    return op(qe, w, u, Aqk, kd, egl, S0)
