"""KDA 逐 token decode 步的融合 Metal kernel（T=1，无前向梯度需求）。

把 decode 单步从 ~28 个小 kernel 收成 1 个（外加无法折叠的 f_b_proj /
o_proj 两个 GEMM；满秩 W_g 在 Python 侧 σ 后点乘）：
  q/k 逐 head 无参 RMS 单位化 + scale 折叠（bf16 双舍入口径照抄
    _rms_unit + 标量乘）、逐通道 log 衰减 log_g = g_min·σ(e^{A} z)
    （g_min=−5，z=a+dt_bias）、β=σ(bl)、SSM 状态递推
    S' = S⊙e^{log_g} + k̂⊗β(v−k̂ᵀS)、输出 out = q̂ᵀS'。

数学上与 KDAAttention.__call__ 的 T=1 eager 链逐项一致（含 bf16 中间
舍入），唯一差异是 f32 归约的求和顺序（serial vs tree），bf16 输出下
不可分辨。S 状态全程 f32。

布局：threadgroup 负责一个 (b,h)；q̂/k̂/e^{log_g} 驻 shared（D 各一份），
S 按列 (Dv) 分到线程，d 向串行点积，S 读两遍（点积 / 更新）写一遍，
全程 f32。S_in/S_out 是独立 buffer（MLX kernel 输出语义），调用方
直接替换 cache 里的 state。

仅服务 T==1 decode（含 MTP 逐步 draft，trace 由调用方在 kernel 外照旧
追加 S 快照）；T>1 verify/prefill 仍走 _chunk_kda / _recurrent_kda。
JIT 编译失败或在线校验不通过时永久回退（返回 None），与 conv.py /
attn_fused.py 同一约定。
"""

import mlx.core as mx

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_KERNELS: dict = {}
_VERIFIED: set = set()
_DISABLED = False

_NT = 128  # threadgroup 线程数（≥D 即可，Dv 按 NT 跨步）


def _build(D: int, Dv: int, H: int, dtype, scale: float, eps: float):
    key = (D, Dv, H, dtype, scale, eps)
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    mt = _METAL_TYPE[dtype]
    # eager 里 `scale * bf16张量` 会先把 python float 舍到 bf16 再乘（实测
    # 与显式 bf16 标量逐位一致）；kernel 用舍入后的值做 f32 乘再舍 bf16，
    # 与 bf16×bf16→bf16 的硬件语义一致
    scale_r = float(mx.array(scale).astype(dtype).astype(mx.float32))
    scale2_r = float(mx.array(scale * scale).astype(dtype).astype(mx.float32))
    kern = mx.fast.metal_kernel(
        name=f"kda_decode_{D}_{Dv}_{H}_{mt}",
        # q/k/v/a: (B,H,D) x.dtype；bl: (B,H)；A_log: (H,) f32 语义无dtype假定
        # （按 x.dtype 读）；dt_bias: (H*D,)；S: (B,H,D,Dv) f32
        input_names=["q", "k", "v", "a", "bl", "A_log", "dt_bias", "S"],
        output_names=["out", "Snew"],
        source=f"""
        constexpr uint D = {D};
        constexpr uint DV = {Dv};
        constexpr uint H = {H};
        constexpr uint NT = {_NT};
        constexpr float SCALE = {scale_r!r}f;
        constexpr float SCALE2 = {scale2_r!r}f;
        constexpr float EPS = {eps!r}f;
        constexpr float G_MIN = -5.0f;
        threadgroup float sh_qh[D];
        threadgroup float sh_kh[D];
        threadgroup float sh_egl[D];
        threadgroup float red[NT / 32];
        threadgroup float redk[NT / 32];
        threadgroup float sh_beta[1];
        uint tid = thread_position_in_threadgroup.x;
        uint bh = threadgroup_position_in_grid.y;
        uint h = bh % H;
        size_t row = (size_t)bh * D;

        // ---- q/k RMS 单位化（bf16 双舍入照抄 eager）+ log_g/beta ----
        float qv = 0.0f, kvv = 0.0f;
        if (tid < D) {{
            qv = float(q[row + tid]);
            kvv = float(k[row + tid]);
        }}
        float sq = simd_sum(qv * qv);
        float sk = simd_sum(kvv * kvv);
        if (tid % 32 == 0) {{ red[tid / 32] = sq; redk[tid / 32] = sk; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float s0 = 0.0f, s1 = 0.0f;
            for (uint i = 0; i < NT / 32; i++) {{ s0 += red[i]; s1 += redk[i]; }}
            red[0] = metal::rsqrt(s0 / D + EPS);
            redk[0] = metal::rsqrt(s1 / D + EPS);
            sh_beta[0] = 1.0f / (1.0f + metal::exp(-float(bl[bh])));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rq = red[0], rk = redk[0];
        if (tid < D) {{
            // eager：rms_norm 输出 bf16 → 乘 python float（弱类型→bf16）→
            // 进递推前 astype(f32)；两次 bf16 舍入都在
            {mt} qh = ({mt})(SCALE2 * (float)(({mt})(qv * rq)));
            {mt} kh = ({mt})(SCALE * (float)(({mt})(kvv * rk)));
            sh_qh[tid] = float(qh);
            sh_kh[tid] = float(kh);
            float av = float(a[row + tid]) + float(dt_bias[h * D + tid]);
            // K3: g = g_min * sigmoid(exp(A_log) * (a + dt_bias)); exp(A_log) stored dtype
            float eg = float(({mt})(metal::exp(float(A_log[h]))));
            float lg = G_MIN / (1.0f + metal::exp(-eg * av));
            sh_egl[tid] = metal::exp(lg);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float beta = sh_beta[0];

        // ---- SSM 步：按列（Dv）分到线程 ----
        // 注意 kv 用的是「衰减后」的 S（照抄 _recurrent_kda：先 S 乘 egl
        // 再 k̂ᵀS）——kv 点积用 sh_kh[dd]*sh_egl[dd] 作为系数
        for (uint j = tid; j < DV; j += NT) {{
            float kvd = 0.0f;
            for (uint dd = 0; dd < D; dd++) {{
                kvd += (sh_kh[dd] * sh_egl[dd]) * S[(row + dd) * DV + j];
            }}
            float dl = beta * (float(v[row + j]) - kvd);
            float o = 0.0f;
            for (uint dd = 0; dd < D; dd++) {{
                float s = S[(row + dd) * DV + j] * sh_egl[dd] + sh_kh[dd] * dl;
                Snew[(row + dd) * DV + j] = s;
                o += sh_qh[dd] * s;
            }}
            out[row + j] = ({mt})o;
        }}
        """,
    )
    _KERNELS[key] = kern
    return kern


def _eager_ref(q, k, v, a, bl, A_log, dt_bias, S, scale, eps):
    """eager 参考：照抄 KDAAttention.__call__ 的 T=1 链（rms 单位化 +
    scale 折叠 → log_g/β → _recurrent_kda 单步）。"""
    from ..norms import _rms_unit

    B, H, D = q.shape
    qh = (scale**2) * _rms_unit(q, eps)
    kh = scale * _rms_unit(k, eps)
    z = a.astype(mx.float32) + dt_bias.reshape(H, D)[None]
    log_g = -5.0 * mx.sigmoid(mx.exp(A_log)[None, :, None] * z)
    beta = mx.sigmoid(bl.astype(mx.float32))
    qf = qh.astype(mx.float32)
    kf = kh.astype(mx.float32)
    vf = v.astype(mx.float32)
    S0 = S.astype(mx.float32)
    S1 = S0 * mx.exp(log_g)[..., None]
    kv = (S1 * kf[..., None]).sum(axis=-2)
    delta = beta[..., None] * (vf - kv)
    S1 = S1 + kf[..., None] * delta[..., None, :]
    out = (S1 * qf[..., None]).sum(axis=-2)
    return out.astype(q.dtype), S1


def kda_decode_step(q, k, v, a, bl, A_log, dt_bias, S, *, scale, eps=1e-6):
    """融合 T=1 decode 步。q/k/v/a (B,H,D) x.dtype，bl (B,H)，
    A_log (H,)，dt_bias (H*D,)，S (B,H,D,Dv) f32。
    返回 (out (B,H,Dv) x.dtype, S_new f32)；不可用/校验失败返回 None。
    """
    global _DISABLED
    if _DISABLED or q.dtype not in _METAL_TYPE:
        return None
    B, H, D = q.shape
    Dv = v.shape[-1]
    if D > _NT:
        return None
    key = (D, Dv, H, q.dtype, scale, eps)
    try:
        kern = _build(D, Dv, H, q.dtype, scale, eps)
        out, Snew = kern(
            inputs=[q, k, v, a, bl, A_log, dt_bias, S],
            output_shapes=[(B, H, Dv), (B, H, D, Dv)],
            output_dtypes=[q.dtype, mx.float32],
            grid=(_NT, B * H, 1),
            threadgroup=(_NT, 1, 1),
        )
        if key not in _VERIFIED:
            ref_o, ref_s = _eager_ref(q, k, v, a, bl, A_log, dt_bias, S, scale, eps)
            mx.eval(out, Snew, ref_o, ref_s)  # 触发 JIT；失败走 except 永久回退
            do = (out.astype(mx.float32) - ref_o.astype(mx.float32)).abs().max().item()
            ds = (Snew - ref_s).abs().max().item()
            st = float(mx.abs(ref_s).max().item())
            if do > 2e-2 or ds > 2e-2 * max(1.0, st):
                raise RuntimeError(
                    f"kda_decode 校验失败 |Δout|={do:.2e} |ΔS|={ds:.2e} (key={key})"
                )
            _VERIFIED.add(key)
        return out, Snew
    except Exception:
        _DISABLED = True
        return None


def prewarm(D: int, Dv: int, H: int, dtype, scale: float, eps: float = 1e-6) -> bool:
    """生成循环前的 JIT 预编译 + 在线校验。返回 False 表示已回退。"""
    global _DISABLED
    if _DISABLED or dtype not in _METAL_TYPE:
        return False
    try:
        B = 2
        q = (mx.random.normal((B, H, D)) * 0.5).astype(dtype)
        k = (mx.random.normal((B, H, D)) * 0.5).astype(dtype)
        v = (mx.random.normal((B, H, Dv)) * 0.5).astype(dtype)
        a = (mx.random.normal((B, H, D)) * 0.3).astype(dtype)
        bl = (mx.random.normal((B, H)) * 0.3).astype(dtype)
        A_log = mx.log(mx.random.uniform(1.0, 16.0, (H,))).astype(dtype)
        dt_bias = mx.log(mx.expm1(mx.random.uniform(0.001, 0.1, (H * D,)))).astype(
            dtype
        )
        S = mx.random.normal((B, H, D, Dv)).astype(mx.float32) * 0.1
        r = kda_decode_step(q, k, v, a, bl, A_log, dt_bias, S, scale=scale, eps=eps)
        return r is not None
    except Exception:
        _DISABLED = True
        return False
