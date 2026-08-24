"""融合因果 depthwise 短卷积 Metal kernel（fwd + 手写 VJP）。

服务两类调用点（数学一致，仅是否带 SiLU 不同）：
- KDA 的 q/k/v 短程混合 `_KDAConv`（随机初始化 + SiLU）；
- block 级 `ShortConv`（identity-init，无 SiLU）。

y[b,t,c] = act( Σ_j w[j,c] · x[b,t−j,c] · m_j )，m_j 为 segment 掩码
（seg[t]==seg[t−j]，无 seg 时恒 1），t−j<0 视为 0。内部一律 f32 累加。

eager 参考口径每 conv 是 4 次 shift + 逐 tap 掩码/乘加共 ~15 个图节点
（KDA 每层 3 个 conv，block 每分支出口各 1 个），fwd+bwd 实测 ~20ms/层；
融合后 fwd 1 个 kernel、bwd 2 个 kernel（dx+dz 同核，dw 分片归约另核），
内存流量接近下界（x/dy 各读一遍）。

反向解析式：dz = dy ⊙ act′(pre)；dx[t] = Σ_j w_j·dz[t+j]·m；
dw[j,c] = Σ_{b,t} dz·x。m 是离散掩码无梯度。act=silu 时 pre 在 bwd
里用 x/w 重算（省一次 fwd 物化）；act=identity 时 dz = dy。
dw 核按 T 轴切 8 片写部分和 (8,K,C)，由调用方一次 sum 归约
（避开 atomic 输出与零初始化问题，归约张量仅 96KB）。

K 固定为 4（KDA_CONV_KERNEL / ShortConv 默认值），其余 K 回退 eager。
JIT 编译失败或在线校验不通过时永久回退 eager 参考（与 attn_res_fused
同一约定）；mx.compile trace 内不做校验（无 host sync），由调用方在
compile 前 prewarm。
"""

import os

import mlx.core as mx

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_KERNELS: dict = {}
_OPS: dict = {}
_VERIFIED: set = set()
# VIBY_CONV_FUSED=0 或 VIBY_FUSED_KERNELS=0：训练走 eager conv（run 81 口径）
_DISABLED = (
    os.environ.get("VIBY_CONV_FUSED", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)

# dw 归约核沿 T 轴的切片数（并行度与部分和大小的折中）
_DW_T_CHUNKS = 128
# fwd/dx 核沿 T 轴的分块长度。K=4 因果卷积在 t 轴上无递推依赖（每个输出
# 只看 4 个相邻输入），所以不必"每线程扫完整个 T"——那样 grid 只有 C×B
# 个线程（C=768/B=12 时 9216），只占 M4 Max 并发能力的 ~20%，带宽跑不满。
# 按 TB 分块后并行度 ×ceil(T/TB)，块首多读 3 个历史元素（占比 3/TB）。
_T_BLOCK = 128
# 线程组沿通道轴的宽度（同一线程组内 c 连续，保证合并访存）
_TG_W = 128


def _build(C: int, dtype, has_seg: bool, has_silu: bool, tblk: int = _T_BLOCK):
    """按 (C,dtype,has_seg,has_silu,tblk) 构建 fwd/dx/dw 三个 kernel（K=4）。"""
    key = (C, dtype, has_seg, has_silu, tblk)
    if key in _KERNELS:
        return _KERNELS[key]
    mt = _METAL_TYPE[dtype]
    hs = 1 if has_seg else 0
    silu_fwd = "acc = acc / (1.0f + metal::exp(-acc));" if has_silu else ""

    # 块划分前缀：把 grid.y 拆成 (b, t 分块)，算出本线程负责的 [t0, t1)
    blk_head = f"""
        uint c = thread_position_in_grid.x;
        uint gy = thread_position_in_grid.y;
        constexpr uint C = {C};
        constexpr uint TB = {tblk};
        constexpr bool HAS_SEG = {hs};
        uint nblk = (T + TB - 1) / TB;
        uint b = gy / nblk;
        uint t0 = (gy % nblk) * TB;
        uint t1 = metal::min(t0 + TB, (uint)T);
        size_t base = (size_t)b * T * C + c;
        float w0 = float(w[c]);
        float w1 = float(w[C + c]);
        float w2 = float(w[2 * C + c]);
        float w3 = float(w[3 * C + c]);
    """

    # pre[tt] = Σ_j w_j·x[tt−j]·m_j（掩码 seg[tt]==seg[tt−j]）。直接索引，
    # 用于块首历史与 bwd 重算；块内主循环仍走寄存器滑窗避免重复读。
    calc_pre = """
        #define CALC_PRE(tt, av) \\
            av = float(x[base + (size_t)(tt) * C]) * w0; \\
            if ((tt) >= 1 && (!HAS_SEG || seg[b * T + (tt)] == seg[b * T + (tt) - 1])) \\
                av += float(x[base + (size_t)((tt) - 1) * C]) * w1; \\
            if ((tt) >= 2 && (!HAS_SEG || seg[b * T + (tt)] == seg[b * T + (tt) - 2])) \\
                av += float(x[base + (size_t)((tt) - 2) * C]) * w2; \\
            if ((tt) >= 3 && (!HAS_SEG || seg[b * T + (tt)] == seg[b * T + (tt) - 3])) \\
                av += float(x[base + (size_t)((tt) - 3) * C]) * w3;
    """
    # dz[tt] = dy[tt] ⊙ act′(pre[tt])；silu′(a) = σ(a)·(1 + a·(1−σ(a)))
    calc_dz = (
        """
        #define CALC_DZ(tt, dzv) { \\
            float pa_; CALC_PRE(tt, pa_) \\
            float sg_ = 1.0f / (1.0f + metal::exp(-pa_)); \\
            dzv = float(dy[base + (size_t)(tt) * C]) * sg_ * (1.0f + pa_ * (1.0f - sg_)); \\
        }
        """
        if has_silu
        else """
        #define CALC_DZ(tt, dzv) { dzv = float(dy[base + (size_t)(tt) * C]); }
        """
    )

    fwd_src = f"""
        {blk_head}
        // 块首历史窗（t0−1..t0−3 从 global 读，之后块内滑窗复用）
        float x1 = (t0 >= 1) ? float(x[base + (size_t)(t0 - 1) * C]) : 0.0f;
        float x2 = (t0 >= 2) ? float(x[base + (size_t)(t0 - 2) * C]) : 0.0f;
        float x3 = (t0 >= 3) ? float(x[base + (size_t)(t0 - 3) * C]) : 0.0f;
        int s0 = 0, s1 = -1, s2 = -2, s3 = -3;
        if (HAS_SEG) {{
            if (t0 >= 1) s1 = seg[b * T + t0 - 1];
            if (t0 >= 2) s2 = seg[b * T + t0 - 2];
            if (t0 >= 3) s3 = seg[b * T + t0 - 3];
        }}
        for (uint t = t0; t < t1; t++) {{
            float x0 = float(x[base + (size_t)t * C]);
            if (HAS_SEG) s0 = seg[b * T + t];
            float acc = x0 * w0;
            if (t >= 1 && (!HAS_SEG || s0 == s1)) acc += x1 * w1;
            if (t >= 2 && (!HAS_SEG || s0 == s2)) acc += x2 * w2;
            if (t >= 3 && (!HAS_SEG || s0 == s3)) acc += x3 * w3;
            {silu_fwd}
            y[base + (size_t)t * C] = {mt}(acc);
            x3 = x2; x2 = x1; x1 = x0;
            s3 = s2; s2 = s1; s1 = s0;
        }}
    """

    # dx 核：块内 t 降序，dz[t+1..t+3] 寄存器滑窗；块尾的 3 个未来 dz 从
    # 邻块范围重算（不写出，避免与邻块的 dz 写重叠）。dz 顺带物化（f32）
    # 供 dw 核复用，免其再重算一次 pre。
    dx_src = f"""
        {calc_pre}
        {calc_dz}
        {blk_head}
        float dz1 = 0.0f, dz2 = 0.0f, dz3 = 0.0f;
        if (t1 + 0 < (uint)T) {{ float v_; CALC_DZ(t1 + 0, v_) dz1 = v_; }}
        if (t1 + 1 < (uint)T) {{ float v_; CALC_DZ(t1 + 1, v_) dz2 = v_; }}
        if (t1 + 2 < (uint)T) {{ float v_; CALC_DZ(t1 + 2, v_) dz3 = v_; }}
        for (int t = (int)t1 - 1; t >= (int)t0; t--) {{
            float dzv; CALC_DZ((uint)t, dzv)
            dz[base + (size_t)t * C] = dzv;
            float d = dzv * w0;
            if (t + 1 < (int)T && (!HAS_SEG || seg[b * T + t] == seg[b * T + t + 1])) d += dz1 * w1;
            if (t + 2 < (int)T && (!HAS_SEG || seg[b * T + t] == seg[b * T + t + 2])) d += dz2 * w2;
            if (t + 3 < (int)T && (!HAS_SEG || seg[b * T + t] == seg[b * T + t + 3])) d += dz3 * w3;
            dx[base + (size_t)t * C] = {mt}(d);
            dz3 = dz2; dz2 = dz1; dz1 = dzv;
        }}
    """

    # dw 核：thread (c, t-切片)，逐 b 升序扫，x/seg 直接索引历史窗；
    # 每线程 4 个 f32 累加器，写出部分和 (TC,4,C)，host 侧一次 sum。
    dw_src = f"""
        uint c = thread_position_in_grid.x;
        uint tc = thread_position_in_grid.y;
        constexpr uint C = {C};
        constexpr bool HAS_SEG = {hs};
        // NC 由调用方按 min(_DW_T_CHUNKS, T) 传入（标量而非编译期常量，
        // 免得 T < 切片数时还要为每个 T 重编译一份 kernel）
        uint t_begin = (T * tc) / NC;
        uint t_end = (T * (tc + 1)) / NC;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (uint b = 0; b < B; b++) {{
            size_t base = (size_t)b * T * C + c;
            for (uint t = t_begin; t < t_end; t++) {{
                float z = dz[base + t * C];
                acc0 += z * float(x[base + t * C]);
                if (t >= 1 && (!HAS_SEG || seg[b * T + t] == seg[b * T + t - 1]))
                    acc1 += z * float(x[base + (t - 1) * C]);
                if (t >= 2 && (!HAS_SEG || seg[b * T + t] == seg[b * T + t - 2]))
                    acc2 += z * float(x[base + (t - 2) * C]);
                if (t >= 3 && (!HAS_SEG || seg[b * T + t] == seg[b * T + t - 3]))
                    acc3 += z * float(x[base + (t - 3) * C]);
            }}
        }}
        size_t o = ((size_t)tc * 4) * C + c;
        dw[o] = acc0;
        dw[o + C] = acc1;
        dw[o + 2 * C] = acc2;
        dw[o + 3 * C] = acc3;
    """

    sfx = f"{C}_{mt}_{hs}_{int(has_silu)}_{tblk}"
    k_fwd = mx.fast.metal_kernel(
        name=f"conv_fwd_{sfx}",
        input_names=["x", "w", "seg", "T"],
        output_names=["y"],
        source=fwd_src,
    )
    k_dx = mx.fast.metal_kernel(
        name=f"conv_bwd_dx_{sfx}",
        input_names=["x", "w", "dy", "seg", "T"],
        output_names=["dx", "dz"],
        source=dx_src,
    )
    k_dw = mx.fast.metal_kernel(
        name=f"conv_bwd_dw_{sfx}",
        input_names=["x", "dz", "seg", "T", "B", "NC"],
        output_names=["dw"],
        source=dw_src,
    )
    _KERNELS[key] = (k_fwd, k_dx, k_dw)
    return _KERNELS[key]


def _conv_eager(x, w, seg, has_silu: bool):
    """eager 参考口径（校验与回退用）。f32 累加，与 kernel 数学一致。"""
    K = w.shape[0]
    xf = x.astype(mx.float32)
    y = xf * w[0].astype(mx.float32)
    for j in range(1, K):
        shifted = mx.concatenate([mx.zeros_like(xf[:, :j]), xf[:, :-j]], axis=1)
        if seg is not None:
            sj = mx.concatenate([mx.zeros_like(seg[:, :j]) - 1, seg[:, :-j]], axis=1)
            same = (seg == sj).astype(mx.float32)[..., None]
            shifted = shifted * same
        y = y + shifted * w[j].astype(mx.float32)
    if has_silu:
        y = y * mx.sigmoid(y)
    return y.astype(x.dtype)


def _op_factory(C: int, dtype, has_seg: bool, has_silu: bool):
    tblk = _T_BLOCK
    dwc = _DW_T_CHUNKS
    k_fwd, k_dx, k_dw = _build(C, dtype, has_seg, has_silu, tblk)
    tg = (min(C, _TG_W), 1, 1)

    @mx.custom_function
    def _op(x, w, seg):
        B, T, _ = x.shape
        y = k_fwd(
            inputs=[x, w, seg, T],
            output_shapes=[(B, T, C)],
            output_dtypes=[dtype],
            grid=(C, B * ((T + tblk - 1) // tblk), 1),
            threadgroup=tg,
        )
        return y[0]

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        x, w, seg = primals
        B, T, _ = x.shape
        dx, dz = k_dx(
            inputs=[x, w, cotangent, seg, T],
            output_shapes=[(B, T, C), (B, T, C)],
            output_dtypes=[dtype, mx.float32],
            grid=(C, B * ((T + tblk - 1) // tblk), 1),
            threadgroup=tg,
        )
        # T 轴切片数不超过 T（每片至少 1 个 t，否则空片白跑）
        nc = min(dwc, max(1, T))
        dw_part = k_dw(
            inputs=[x, dz, seg, T, B, nc],
            output_shapes=[(nc, 4, C)],
            output_dtypes=[mx.float32],
            grid=(C, nc, 1),
            threadgroup=tg,
        )[0]
        return [dx, mx.sum(dw_part, axis=0).astype(w.dtype), None]

    return _op


def causal_conv(x, w, seg=None, silu: bool = True):
    """因果 depthwise 卷积（K=w.shape[0] 须为 4）。x (B,T,C)，w (4,C)，
    seg (B,T) int32 或 None。JIT/校验失败或不支持形状时回退 eager。"""
    global _DISABLED
    B, T, C = x.shape
    if _DISABLED or x.dtype not in _METAL_TYPE or w.shape[0] != 4:
        return _conv_eager(x, w, seg, silu)
    key = (C, x.dtype, seg is not None, silu)
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(C, x.dtype, seg is not None, silu)
            _OPS[key] = op
        y = op(x, w, seg if seg is not None else _seg_dummy(x))
        if key not in _VERIFIED:
            ref = _conv_eager(x, w, seg, silu)
            mx.eval(y, ref)  # 触发 JIT 编译；失败走 except 永久回退
            d = (y.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
            tol = 1e-5 if x.dtype == mx.float32 else 2e-2
            if d > tol:
                raise RuntimeError(f"conv fused 校验失败 |Δ|={d:.2e} (key={key})")
            _VERIFIED.add(key)
        return y
    except Exception:
        _DISABLED = True
        return _conv_eager(x, w, seg, silu)


_SEG_DUMMY = None


def _seg_dummy(x):
    global _SEG_DUMMY
    if _SEG_DUMMY is None:
        _SEG_DUMMY = mx.zeros((1,), dtype=mx.int32)
    return _SEG_DUMMY


def prewarm(C: int, dtype, has_seg: bool, has_silu: bool) -> bool:
    """compile 前预编译 + fwd/bwd 在线校验（梯度对照 eager 参考）。
    返回 False 表示已回退 eager。K 固定 4。"""
    global _DISABLED
    if _DISABLED:
        return False
    key = (C, dtype, has_seg, has_silu)
    if key in _VERIFIED:
        return True
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(C, dtype, has_seg, has_silu)
            _OPS[key] = op
        B, T = 2, 37  # 奇数 T 覆盖 tap 边界
        x = (mx.random.normal((B, T, C)) * 0.5).astype(dtype)
        w = (mx.random.normal((4, C)) * 0.3).astype(dtype)
        seg = (
            mx.cumsum((mx.random.uniform(shape=(B, T)) < 0.2).astype(mx.int32), axis=1)
            if has_seg
            else None
        )

        def fk(x_, w_):
            y = op(x_, w_, seg if seg is not None else _seg_dummy(x))
            return (y.astype(mx.float32) ** 2).sum()

        def fe(x_, w_):
            return (_conv_eager(x_, w_, seg, has_silu).astype(mx.float32) ** 2).sum()

        lg, gg = mx.value_and_grad(fk, argnums=[0, 1])(x, w)
        lr, gr = mx.value_and_grad(fe, argnums=[0, 1])(x, w)
        mx.eval(lg, lr, *gg, *gr)
        tol = 1e-4 if dtype == mx.float32 else 5e-2
        if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"conv prewarm loss 不一致 {lg.item()} vs {lr.item()}")
        for a, b in zip(gg, gr):
            d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
            rel = d / (b.astype(mx.float32).abs().max().item() + 1e-12)
            if rel > tol * 10:
                raise RuntimeError(f"conv prewarm 梯度不一致 rel={rel:.2e}")
        _VERIFIED.add(key)
        return True
    except Exception:
        _DISABLED = True
        return False
