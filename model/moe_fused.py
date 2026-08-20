"""MoE 训练融合 kernel（block-sparse tiled experts，前向融合 + 手写 VJP）。

把稀疏桶路径（MoEFeedForward._sparse_forward）的专家计算区——padded 桶上
的 SwiGLU 前向——从每组 ~6 个 MLX 算子 × n_groups 收敛为 2 个手写 Metal
kernel：

1. gu_swiglu：一个 threadgroup（256 线程）负责某专家桶内 64 行 × 全部 I 列
   的 h 输出块；K 维（D）按 64 分块循环，x 块（8KB）与 W 块（gate 或 up，
   I×64×2B，I=104 时 13.3KB）载入 threadgroup memory，f32 累加；gate/up
   两遍复用同一 x 块。padding 行（实际计数 < 桶容量）必须写 0——反向 dW
   的转置 matmul 会读到这些行，dy 为 0 但 0×垃圾 = NaN。
2. down：threadgroup 负责 64 行 × 64 列的 y 输出块，I 维按 13 分块。
   padding 行的 y 永不读回，跳过不写。grid z 维多挂一个 threadgroup
   （z==E）把 trash 行（溢出 pair 的落点）清零，省掉一次整Buffer concat。

溢出语义与旧路径一致：桶只容纳每专家前 cap_e 个 pair，kernel 内
cnt = min(counts[e], cap_e)（counts 含溢出 pair，不钳制会读到下一个
专家的桶）；溢出 pair 输出落 trash 行 = 0。

反向（Stage A）暂用 padded batched GEMM 重建（与原稀疏路径反向同构、
成本持平），经 mx.custom_function 的 vjp 接入 autodiff。容量表是逐微批
变化的 host int（value_and_grad 建图期不能同步取数，否则钉住整张前向
图），而 vjp 建反向图需要本批的 host 侧切片边界 ⇒ 每微批每个调用点由
make_fused_experts 新建一个轻量 custom_function 实例，vjp 闭包捕获本批
caps/starts；kernel 本体按 (D,I,E,dtype) 全局缓存，实例只是 python 包装。

网格全部由 host 侧容量表算出 ⇒ 前向保持无 host sync。权重直接用
_StackedExperts 原始布局（gate_up_w (E,2I,D)、down_w (E,D,I)），与旧路径
和既有 checkpoint 完全兼容。custom_function 边界只包专家计算区：外面的
argsort/scatter（xb 构造）与 gather/加权 scatter-add（输出收回）保持
MLX autodiff（线性算子），x 与 w 的梯度白拿。
"""

import mlx.core as mx
import mlx.nn as nn

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_kernel_cache: dict = {}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _build_train_kernels(D, I, E, dtype):
    """构建 2 个训练前向 kernel。按 (D,I,E,dtype) 缓存。

    约束（不满足由调用方回退旧稀疏路径）：D % 64 == 0、I % 13 == 0、
    桶容量 64 对齐（_SPARSE_ALIGN=128 保证）。E 烧进 kernel 源（trash 行
    判断用），E 变化时重新编译。

    threadgroup 内存预算（32KB 上限）：f32 元素 4B，K 分块减半（32），
    xs 64×32×4=8KB + wst 32×I×4=13.3KB（I=104）；bf16 时 K 分块 64，
    xs 64×64×2=8KB + wst 64×I×2=13.3KB。

    W 以 kk-major（转置）staging：计算内层相邻 lane 读连续 i 地址，
    消除 32 路 bank conflict（行主存 128B stride 时全部 lane 撞同一
    bank——首版实测慢 3× 的根因）。"""
    if D % 64 != 0 or I % 13 != 0:
        raise ValueError(f"fused MoE kernel 要求 D%64==0 且 I%13==0，got D={D} I={I}")
    key = (D, I, E, dtype)
    if key in _kernel_cache:
        return _kernel_cache[key]
    mt = _METAL_TYPE[dtype]
    isz = 4 if dtype == mx.float32 else 2
    CHK = 32 if isz == 4 else 64  # K 分块
    col_tiles = D // 64
    # 每线程输出数 / staging 趟数（256 线程）
    nout = _ceil_div(64 * I, 256)
    nws = _ceil_div(I * CHK, 256)
    out_guard = "true" if 64 * I % 256 == 0 else f"f < {64 * I}"
    ws_guard = "true" if I * CHK % 256 == 0 else f"f < {I * CHK}"
    nx = 64 * CHK // 256  # x staging 趟数（恒整除）

    # h 块：64 行 × I 列。accg/accu 各 nout 个 f32 寄存器。
    gu_src = f"""
        uint tid = thread_position_in_grid.x;
        uint tr = thread_position_in_grid.y;
        uint e = thread_position_in_grid.z;
        uint cap_e = uint(cap_tab[e]);
        uint r0 = tr * 64;
        if (r0 >= cap_e) return;
        uint cnt = metal::min(uint(counts[e]), cap_e);
        size_t row0 = (size_t)base[e] + r0;
        if (r0 >= cnt) {{
            // 全 padding tile：不算只写 0（反向 dW 的转置 matmul 会读 h，
            // 必须为有限值）；容量远超计数时（集中态 3.5× padding）省掉
            // 整块 64×I×D 的 MAC。
            for (uint j = 0; j < {nout}; j++) {{
                uint f = tid + j * 256;
                if ({out_guard}) {{
                    h[(row0 + f / {I}) * {I} + f % {I}] = {mt}(0.0f);
                }}
            }}
            return;
        }}
        threadgroup {mt} xs[64 * {CHK}];
        threadgroup {mt} wst[{CHK} * {I}];
        float accg[{nout}];
        float accu[{nout}];
        for (uint j = 0; j < {nout}; j++) {{ accg[j] = 0.0f; accu[j] = 0.0f; }}
        for (uint c = 0; c < {D // CHK}; c++) {{
            for (uint t = 0; t < {nx}; t++) {{
                uint f = tid + t * 256;
                xs[f] = xb[(row0 + f / {CHK}) * {D} + c * {CHK} + f % {CHK}];
            }}
            for (uint t = 0; t < {nws}; t++) {{
                uint f = tid + t * 256;
                if ({ws_guard}) {{
                    uint ii = f / {CHK}; uint kk = f % {CHK};
                    wst[kk * {I} + ii] = gu_w[((size_t)e * {2 * I} + ii) * {D} + c * {CHK} + kk];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < {nout}; j++) {{
                uint f = tid + j * 256;
                if ({out_guard}) {{
                    uint r = f / {I}; uint i = f % {I};
                    float a = 0.0f;
                    for (uint kk = 0; kk < {CHK}; kk++) a += float(xs[r * {CHK} + kk]) * float(wst[kk * {I} + i]);
                    accg[j] += a;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < {nws}; t++) {{
                uint f = tid + t * 256;
                if ({ws_guard}) {{
                    uint ii = f / {CHK}; uint kk = f % {CHK};
                    wst[kk * {I} + ii] = gu_w[((size_t)e * {2 * I} + {I} + ii) * {D} + c * {CHK} + kk];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < {nout}; j++) {{
                uint f = tid + j * 256;
                if ({out_guard}) {{
                    uint r = f / {I}; uint i = f % {I};
                    float a = 0.0f;
                    for (uint kk = 0; kk < {CHK}; kk++) a += float(xs[r * {CHK} + kk]) * float(wst[kk * {I} + i]);
                    accu[j] += a;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint j = 0; j < {nout}; j++) {{
            uint f = tid + j * 256;
            if ({out_guard}) {{
                uint r = f / {I}; uint i = f % {I};
                float g = accg[j];
                float u = accu[j];
                float hv = g / (1.0f + metal::exp(-g)) * u;
                h[(row0 + r) * {I} + i] = (r0 + r < cnt) ? {mt}(hv) : {mt}(0.0f);
            }}
        }}
    """
    # y 块：64 行 × 64 列。z==E 的 threadgroup 清零 trash 行（R = 末专家
    # 桶末尾，桶按 EG 均匀分配时 base[E-1]+cap = R）。dw_w 同样转置
    # staging（ii-major）消除 bank conflict。
    down_src = f"""
        uint tid = thread_position_in_grid.x;
        uint ty = thread_position_in_grid.y;
        uint e = thread_position_in_grid.z;
        if (e == {E}) {{
            if (ty == 0) {{
                size_t R = (size_t)base[{E} - 1] + (size_t)cap_tab[{E} - 1];
                for (uint d = tid; d < {D}; d += 256) y[R * {D} + d] = {mt}(0.0f);
            }}
            return;
        }}
        uint cap_e = uint(cap_tab[e]);
        uint tr = ty / {col_tiles};
        uint tc = ty % {col_tiles};
        uint r0 = tr * 64;
        if (r0 >= cap_e) return;
        uint cnt = metal::min(uint(counts[e]), cap_e);
        // 全 padding tile 直接退出（y 的 padding 行永不读回，无需写 0）
        if (r0 >= cnt) return;
        size_t row0 = (size_t)base[e] + r0;
        threadgroup {mt} hs[64 * 13];
        threadgroup {mt} wst[13 * 64];
        float acc[16];
        for (uint j = 0; j < 16; j++) acc[j] = 0.0f;
        for (uint c = 0; c < {I // 13}; c++) {{
            for (uint t = 0; t < 4; t++) {{
                uint f = tid + t * 256;
                if (f < 832) hs[f] = h[(row0 + f / 13) * {I} + c * 13 + f % 13];
            }}
            for (uint t = 0; t < 4; t++) {{
                uint f = tid + t * 256;
                if (f < 832) {{
                    uint dd = f / 13; uint ii = f % 13;
                    wst[ii * 64 + dd] = dw_w[((size_t)e * {D} + tc * 64 + dd) * {I} + c * 13 + ii];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < 16; j++) {{
                uint f = tid + j * 256;
                uint r = f / 64; uint dd = f % 64;
                float a = 0.0f;
                for (uint ii = 0; ii < 13; ii++) a += float(hs[r * 13 + ii]) * float(wst[ii * 64 + dd]);
                acc[j] += a;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        for (uint j = 0; j < 16; j++) {{
            uint f = tid + j * 256;
            uint r = f / 64; uint dd = f % 64;
            if (r0 + r < cnt) y[(row0 + r) * {D} + tc * 64 + dd] = {mt}(acc[j]);
        }}
    """
    k_gu = mx.fast.metal_kernel(
        name=f"moe_tr_gu_{D}_{I}_{E}_{mt}",
        input_names=["xb", "gu_w", "base", "counts", "cap_tab"],
        output_names=["h"],
        source=gu_src,
    )
    k_down = mx.fast.metal_kernel(
        name=f"moe_tr_down_{D}_{I}_{E}_{mt}",
        input_names=["h", "dw_w", "base", "counts", "cap_tab"],
        output_names=["y"],
        source=down_src,
    )
    _kernel_cache[key] = (k_gu, k_down)
    return k_gu, k_down


def make_fused_experts(D, I, E, EG, caps, starts, dtype):
    """为本微批创建一次性 custom_function 实例（fwd=2 个融合 kernel，
    vjp 闭包捕获本批 host 容量表，用 padded batched GEMM 建反向图）。

    caps/starts：host int 列表（_sparse_forward 的容量表与前缀和）；
    EG：稀疏分组大小；要求 E % EG == 0 且桶按 EG 均匀布局（trash 行
    定位依赖 base[E-1]+cap == R）。调用签名：
        fused(xb, gate_up_w, down_w, base_e, counts, cap_e) -> y_flat
    xb (R+1,D)、y_flat (R+1,D)（末行为 trash，恒 0）；base_e/cap_e (E,)
    int32 为各专家桶起始行/容量（device）；counts (E,) int32 实测计数。
    """
    if E % EG != 0:
        raise ValueError(f"fused MoE kernel 要求 E%EG==0，got E={E} EG={EG}")
    k_gu, k_down = _build_train_kernels(D, I, E, dtype)
    R = int(starts[-1])
    n_groups = len(caps)
    row_tiles = _ceil_div(max(caps), 64)
    col_tiles = D // 64

    @mx.custom_function
    def _fused(xb, gu_w, dw_w, base_e, counts, cap_e):
        h = k_gu(
            inputs=[xb, gu_w, base_e, counts, cap_e],
            output_shapes=[(R, I)],
            output_dtypes=[dtype],
            grid=(256, row_tiles, E),
            threadgroup=(256, 1, 1),
        )[0]
        y = k_down(
            inputs=[h, dw_w, base_e, counts, cap_e],
            output_shapes=[(R + 1, D)],
            output_dtypes=[dtype],
            grid=(256, row_tiles * col_tiles, E + 1),
            threadgroup=(256, 1, 1),
        )[0]
        return y

    def _vjp(primals, cotangents, outputs):
        # Stage A：padded batched GEMM 反向（与原稀疏路径 autodiff 同构）。
        # 桶内 padding 行：前向 h 恒 0、xb 恒 0、dy 恒 0 ⇒ 各 dW/dxb 贡献
        # 自动为 0；trash 行的 cotangent 不落任何分组切片，自然丢弃（溢出
        # pair 前向输出为 0，梯度就该是 0）。
        xb, gu_w, dw_w, _base_e, _counts, _cap_e = primals
        dy_full = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        dy_full = dy_full.astype(xb.dtype)
        gu_t = gu_w.swapaxes(-1, -2)  # (E,D,2I)
        dxb_parts, dgu_parts, ddw_parts = [], [], []
        for gi in range(n_groups):
            e0, e1 = gi * EG, min(gi * EG + EG, E)
            eg = e1 - e0
            Cg = int(caps[gi])
            sl = slice(starts[gi], starts[gi] + eg * Cg)
            xg = xb[sl].reshape(eg, Cg, D)
            dyg = dy_full[sl].reshape(eg, Cg, D)
            gu = xg @ gu_t[e0:e1]  # (eg,Cg,2I)，重算前向激活
            g, u = mx.split(gu, 2, axis=-1)
            h_g = nn.silu(g) * u  # (eg,Cg,I)
            dh = dyg @ dw_w[e0:e1]  # (eg,Cg,I)
            ddw_parts.append(mx.matmul(dyg.swapaxes(-1, -2), h_g))  # (eg,D,I)
            sg = mx.sigmoid(g)
            dg = dh * u * (sg * (1.0 + g * (1.0 - sg)))  # dh⊙u⊙silu'(g)
            du = dh * (g * sg)  # dh⊙silu(g)
            dgu = mx.concatenate([dg, du], axis=-1)  # (eg,Cg,2I)
            dgu_parts.append(mx.matmul(dgu.swapaxes(-1, -2), xg))  # (eg,2I,D)
            dxb_parts.append((dgu @ gu_w[e0:e1]).reshape(eg * Cg, D))
        dxb = mx.concatenate(dxb_parts + [mx.zeros((1, D), dtype=xb.dtype)], axis=0)
        d_gu = mx.concatenate(dgu_parts, axis=0)  # (E,2I,D)
        d_dw = mx.concatenate(ddw_parts, axis=0)  # (E,D,I)
        return dxb, d_gu, d_dw, None, None, None

    _fused.vjp(_vjp)
    return _fused


def prewarm_fused(moe) -> bool:
    """在模块 __init__（任何 value_and_grad 建图之外）编译并端到端验证
    kernel（含 vjp 建图）。返回 False 表示不可用，调用方应永久回退。

    mlx metal_kernel 是 lazy 编译（首次 mx.eval 才触发），若把验证推迟到
    训练首个微批，编译失败/建图错误会在 value_and_grad 内部抛出且无法
    干净回退，故在此用微型 dummy 输入预编译。"""
    D = moe.experts.gate_up_w.shape[-1]
    I = moe.moe_in
    E = moe.n_routed
    EG = min(moe._SPARSE_GROUP, E)
    if D % 64 != 0 or I % 13 != 0 or E % EG != 0:
        return False
    n_groups = E // EG
    caps = [128] * n_groups
    starts = [0] * (n_groups + 1)
    for gi in range(n_groups):
        starts[gi + 1] = starts[gi] + EG * caps[gi]
    R = starts[-1]
    exp_ids = mx.arange(E, dtype=mx.int32)
    grp = exp_ids // EG
    caps_dev = mx.array(caps, dtype=mx.int32)
    start_dev = mx.array(starts[:-1], dtype=mx.int32)
    base_e = start_dev[grp] + (exp_ids - grp * EG) * caps_dev[grp]
    cap_e = caps_dev[grp]
    counts = mx.zeros((E,), dtype=mx.int32)
    dtypes = {mx.bfloat16, moe.experts.gate_up_w.dtype}
    for dt in dtypes:
        if dt not in _METAL_TYPE:
            continue
        fused = make_fused_experts(D, I, E, EG, caps, starts, dt)
        xb = mx.zeros((R + 1, D), dtype=dt)
        gu = moe.experts.gate_up_w.astype(dt)
        dw = moe.experts.down_w.astype(dt)

        def _loss(xb_):
            return fused(xb_, gu, dw, base_e, counts, cap_e).sum()

        val, g = mx.value_and_grad(_loss)(xb)
        mx.eval(val, g)
    return True
