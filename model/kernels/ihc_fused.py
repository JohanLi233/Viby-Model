"""iHC 读/写融合 Metal kernel（fwd + 手写 VJP）。

eager 每个 sublayer 是 RMSNorm(MD) + 瘦 GEMM(2M) + sigmoid + 4D 加权
读 + 跳轴广播写。B12 T1024 D768 M=4 下 16 次 compile 链 ~247ms，
带宽下界 ~16ms。本模块每 (b,t) 一个 threadgroup：

  mix  fwd：R → x̃、h_post（顺带存 h_pre / rstd / dots 给 VJP）
  mix  bwd：dx、dh_post → dR、dlogits；dW / dα / dbias 在 MLX 侧 GEMM
  write fwd：R' = R + h_post ⊙ Δ
  write bwd：dR = cot；ddelta / dh_post 一次归约（不写回 4D dR）

失败按 (M,D,dtype) 隔离回退 eager。"""

import os

import mlx.core as mx

from ..norms import _rms_unit

_METAL = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_NT = 256
_SG = _NT // 32
_MAX_TG = 30720
_DISABLED = (
    os.environ.get("VIBY_IHC_FUSED", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)
_MIX_KERNELS: dict = {}
_WR_KERNELS: dict = {}
_MIX_OPS: dict = {}
_WR_OPS: dict = {}
_MIX_VERIFIED: set = set()
_MIX_FAILED: set = set()
_WR_VERIFIED: set = set()
_WR_FAILED: set = set()


def _tg_ok(m: int, d: int) -> bool:
    md = m * d
    # rs[MD] + red[SG] + logits/hpre/hpost
    nbytes = (md + _SG + 4 * m) * 4
    return 2 <= m <= 16 and d >= 1 and nbytes <= _MAX_TG


def _mix_eager(r, w, alpha, bpre, bpost, eps: float):
    """与核内 f32 归约同口径（预热对照）。不要用 mx.fast.rms_norm。"""
    b, t, m, d = r.shape
    vf = r.reshape(b, t, m * d).astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(vf * vf, axis=-1, keepdims=True) + eps)
    n = vf * rstd
    dots = n @ w.astype(mx.float32).T
    logits = alpha.astype(mx.float32) * dots
    pre, post = mx.split(logits, 2, axis=-1)
    h_pre = mx.sigmoid(pre + bpre.astype(mx.float32))
    h_post = 2.0 * mx.sigmoid(post + bpost.astype(mx.float32))
    x = (h_pre[..., None] * vf.reshape(b, t, m, d)).sum(axis=2)
    return (
        x.astype(r.dtype),
        h_post.astype(r.dtype),
        h_pre,
        rstd.reshape(b, t),
        dots,
    )


def _mix_python(r, w, alpha, bpre, bpost, eps: float):
    """生产回退：与 IHCGate.gates+read 相同（fast rms_norm）。"""
    b, t, m, d = r.shape
    n = _rms_unit(r.reshape(b, t, m * d), eps)
    logits = alpha.astype(r.dtype) * (n @ w.T)
    pre, post = mx.split(logits, 2, axis=-1)
    h_pre = mx.sigmoid((pre + bpre.astype(r.dtype)).astype(mx.float32)).astype(r.dtype)
    h_post = (
        2.0 * mx.sigmoid((post + bpost.astype(r.dtype)).astype(mx.float32))
    ).astype(r.dtype)
    x = (h_pre[..., None] * r).sum(axis=2)
    return x, h_post


def _write_python(r, delta, h_post):
    return r + h_post[..., None] * delta[..., None, :]


def _build_mix(m: int, d: int, dtype, eps: float):
    key = (m, d, dtype, eps)
    if key in _MIX_KERNELS:
        return _MIX_KERNELS[key]
    mt = _METAL[dtype]
    md = m * d
    om = 2 * m
    fwd = mx.fast.metal_kernel(
        name=f"ihc_mix_fwd_{m}_{d}_{mt}_v2",
        input_names=["R", "W", "alpha", "bpre", "bpost"],
        output_names=["x", "hpost", "hpre", "rstd", "dots"],
        source=f"""
        constexpr uint M = {m};
        constexpr uint D = {d};
        constexpr uint MD = {md};
        constexpr uint OM = {om};
        constexpr uint NT = {_NT};
        constexpr uint SG = {_SG};
        constexpr float EPS = {eps!r}f;
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        uint lane = tid & 31u;
        uint sg = tid >> 5;
        threadgroup float rs[MD];
        threadgroup float red[SG];
        threadgroup float logt[OM];
        threadgroup float hpr[M];
        threadgroup float hpo[M];
        size_t base = (size_t)row * MD;
        for (uint i = tid; i < MD; i += NT) {{
            rs[i] = float(R[base + i]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0.0f;
        for (uint i = tid; i < MD; i += NT) acc += rs[i] * rs[i];
        acc = simd_sum(acc);
        if (lane == 0) red[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float s = 0.0f;
            for (uint i = 0; i < SG; i++) s += red[i];
            red[0] = metal::rsqrt(s / float(MD) + EPS);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rstd_v = red[0];
        if (tid == 0) rstd[row] = rstd_v;
        float al = float(alpha[0]);
        float acck[OM];
        for (uint k = 0; k < OM; k++) acck[k] = 0.0f;
        for (uint i = tid; i < MD; i += NT) {{
            float ni = rs[i] * rstd_v;
            for (uint k = 0; k < OM; k++) {{
                acck[k] += ni * float(W[(size_t)k * MD + i]);
            }}
        }}
        for (uint k = 0; k < OM; k++) {{
            float a = simd_sum(acck[k]);
            if (lane == 0) red[sg] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {{
                float s = 0.0f;
                for (uint i = 0; i < SG; i++) s += red[i];
                logt[k] = s;
                dots[(size_t)row * OM + k] = s;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid < M) {{
            float p = 1.0f / (1.0f + metal::exp(-(al * logt[tid] + float(bpre[tid]))));
            hpr[tid] = p;
            hpre[(size_t)row * M + tid] = p;
            float z = 1.0f / (1.0f + metal::exp(-(al * logt[M + tid] + float(bpost[tid]))));
            hpo[tid] = 2.0f * z;
            hpost[(size_t)row * M + tid] = {mt}(hpo[tid]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint di = tid; di < D; di += NT) {{
            float a = 0.0f;
            for (uint mi = 0; mi < M; mi++) a += hpr[mi] * rs[mi * D + di];
            x[(size_t)row * D + di] = {mt}(a);
        }}
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"ihc_mix_bwd_{m}_{d}_{mt}_v2",
        input_names=["R", "W", "alpha", "hpre", "hpost", "rstd", "dx", "dhpost"],
        output_names=["dR", "dlogits"],
        source=f"""
        constexpr uint M = {m};
        constexpr uint D = {d};
        constexpr uint MD = {md};
        constexpr uint OM = {om};
        constexpr uint NT = {_NT};
        constexpr uint SG = {_SG};
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        uint lane = tid & 31u;
        uint sg = tid >> 5;
        threadgroup float rs[MD];
        threadgroup float red[SG];
        threadgroup float hpr[M];
        threadgroup float dlog[OM];
        threadgroup float coeff_s[1];
        size_t base = (size_t)row * MD;
        for (uint i = tid; i < MD; i += NT) rs[i] = float(R[base + i]);
        if (tid < M) hpr[tid] = float(hpre[(size_t)row * M + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rstd_v = float(rstd[row]);
        float al = float(alpha[0]);
        for (uint mi = 0; mi < M; mi++) {{
            float a = 0.0f;
            for (uint di = tid; di < D; di += NT) {{
                a += rs[mi * D + di] * float(dx[(size_t)row * D + di]);
            }}
            a = simd_sum(a);
            if (lane == 0) red[sg] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {{
                float s = 0.0f;
                for (uint i = 0; i < SG; i++) s += red[i];
                float hp = hpr[mi];
                dlog[mi] = s * hp * (1.0f - hp);
                float hpo = float(hpost[(size_t)row * M + mi]);
                float dhp = float(dhpost[(size_t)row * M + mi]);
                dlog[M + mi] = dhp * hpo * (1.0f - 0.5f * hpo);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float acc = 0.0f;
        for (uint i = tid; i < MD; i += NT) {{
            float dn = 0.0f;
            for (uint k = 0; k < OM; k++) {{
                dn += dlog[k] * al * float(W[(size_t)k * MD + i]);
            }}
            acc += rs[i] * dn;
        }}
        acc = simd_sum(acc);
        if (lane == 0) red[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float s = 0.0f;
            for (uint i = 0; i < SG; i++) s += red[i];
            coeff_s[0] = (rstd_v * rstd_v * rstd_v) / float(MD) * s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float coeff = coeff_s[0];
        for (uint di = tid; di < D; di += NT) {{
            float dxv = float(dx[(size_t)row * D + di]);
            for (uint mi = 0; mi < M; mi++) {{
                uint i = mi * D + di;
                float dn = 0.0f;
                for (uint k = 0; k < OM; k++) {{
                    dn += dlog[k] * al * float(W[(size_t)k * MD + i]);
                }}
                float dv = rstd_v * dn - coeff * rs[i] + hpr[mi] * dxv;
                dR[base + i] = {mt}(dv);
            }}
        }}
        if (tid < OM) {{
            dlogits[(size_t)row * OM + tid] = dlog[tid];
        }}
        """,
    )
    _MIX_KERNELS[key] = (fwd, bwd)
    return fwd, bwd


def _build_write(m: int, d: int, dtype):
    key = (m, d, dtype)
    if key in _WR_KERNELS:
        return _WR_KERNELS[key]
    mt = _METAL[dtype]
    fwd = mx.fast.metal_kernel(
        name=f"ihc_wr_fwd_{m}_{d}_{mt}",
        input_names=["R", "delta", "hpost"],
        output_names=["out"],
        source=f"""
        constexpr uint M = {m};
        constexpr uint D = {d};
        constexpr uint NT = {_NT};
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        for (uint di = tid; di < D; di += NT) {{
            float dv = float(delta[(size_t)row * D + di]);
            for (uint mi = 0; mi < M; mi++) {{
                size_t i = ((size_t)row * M + mi) * D + di;
                float hp = float(hpost[(size_t)row * M + mi]);
                out[i] = {mt}(float(R[i]) + hp * dv);
            }}
        }}
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"ihc_wr_bwd_{m}_{d}_{mt}",
        input_names=["cot", "delta", "hpost"],
        output_names=["ddelta", "dhpost"],
        source=f"""
        constexpr uint M = {m};
        constexpr uint D = {d};
        constexpr uint NT = {_NT};
        constexpr uint SG = {_SG};
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        uint lane = tid & 31u;
        uint sg = tid >> 5;
        threadgroup float red[SG];
        threadgroup float hp[M];
        if (tid < M) hp[tid] = float(hpost[(size_t)row * M + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint di = tid; di < D; di += NT) {{
            float acc = 0.0f;
            for (uint mi = 0; mi < M; mi++) {{
                float c = float(cot[((size_t)row * M + mi) * D + di]);
                acc += c * hp[mi];
            }}
            ddelta[(size_t)row * D + di] = {mt}(acc);
        }}
        for (uint mi = 0; mi < M; mi++) {{
            float a = 0.0f;
            for (uint di = tid; di < D; di += NT) {{
                float c = float(cot[((size_t)row * M + mi) * D + di]);
                a += c * float(delta[(size_t)row * D + di]);
            }}
            a = simd_sum(a);
            if (lane == 0) red[sg] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {{
                float s = 0.0f;
                for (uint i = 0; i < SG; i++) s += red[i];
                dhpost[(size_t)row * M + mi] = {mt}(s);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        """,
    )
    _WR_KERNELS[key] = (fwd, bwd)
    return fwd, bwd


def _mix_op_factory(m: int, d: int, dtype, eps: float):
    k_fwd, k_bwd = _build_mix(m, d, dtype, eps)

    @mx.custom_function
    def _op(r, w, alpha, bpre, bpost):
        b, t = int(r.shape[0]), int(r.shape[1])
        bt = b * t
        om = 2 * m
        x, hpost, hpre, rstd, dots = k_fwd(
            inputs=[
                r.reshape(bt, m * d),
                w,
                alpha.reshape((1,)),
                bpre,
                bpost,
            ],
            output_shapes=[
                (bt, d),
                (bt, m),
                (bt, m),
                (bt,),
                (bt, om),
            ],
            output_dtypes=[r.dtype, r.dtype, mx.float32, mx.float32, mx.float32],
            grid=(_NT, bt, 1),
            threadgroup=(_NT, 1, 1),
        )
        return (
            x.reshape(b, t, d),
            hpost.reshape(b, t, m),
            hpre.reshape(b, t, m),
            rstd.reshape(b, t),
            dots.reshape(b, t, om),
        )

    @_op.vjp
    def _op_vjp(primals, cots, outputs):
        r, w, alpha, bpre, bpost = primals
        dx = cots[0]
        dhpost = cots[1]
        if dx is None:
            dx = mx.zeros(r.shape[:2] + (d,), dtype=r.dtype)
        if dhpost is None:
            dhpost = mx.zeros(r.shape[:2] + (m,), dtype=r.dtype)
        _x, _hp, hpre, rstd, dots = outputs
        b, t = int(r.shape[0]), int(r.shape[1])
        bt = b * t
        md = m * d
        om = 2 * m
        dR, dlogits = k_bwd(
            inputs=[
                r.reshape(bt, md),
                w,
                alpha.reshape((1,)),
                hpre.reshape(bt, m),
                _hp.reshape(bt, m),
                rstd.reshape((bt,)),
                dx.reshape(bt, d),
                dhpost.reshape(bt, m),
            ],
            output_shapes=[(bt, md), (bt, om)],
            output_dtypes=[r.dtype, mx.float32],
            grid=(_NT, bt, 1),
            threadgroup=(_NT, 1, 1),
        )
        dlogits = dlogits.reshape(b, t, om)
        n = (rstd[..., None] * r.reshape(b, t, md).astype(mx.float32)).reshape(bt, md)
        ds = (alpha.astype(mx.float32) * dlogits).reshape(bt, om)
        dW = (ds.T @ n).astype(w.dtype)
        dalpha = (dlogits * dots).sum().reshape(alpha.shape).astype(alpha.dtype)
        dbpre = dlogits[..., :m].sum(axis=(0, 1)).astype(bpre.dtype)
        dbpost = dlogits[..., m:].sum(axis=(0, 1)).astype(bpost.dtype)
        return (
            dR.reshape(r.shape),
            dW,
            dalpha,
            dbpre,
            dbpost,
        )

    return _op


def _write_op_factory(m: int, d: int, dtype):
    k_fwd, k_bwd = _build_write(m, d, dtype)

    @mx.custom_function
    def _op(r, delta, hpost):
        b, t = int(r.shape[0]), int(r.shape[1])
        bt = b * t
        (out,) = k_fwd(
            inputs=[
                r.reshape(bt, m * d),
                delta.reshape(bt, d),
                hpost.reshape(bt, m),
            ],
            output_shapes=[(bt, m * d)],
            output_dtypes=[r.dtype],
            grid=(_NT, bt, 1),
            threadgroup=(_NT, 1, 1),
        )
        return out.reshape(r.shape)

    @_op.vjp
    def _op_vjp(primals, cot, outputs):
        r, delta, hpost = primals
        c = cot[0] if isinstance(cot, (list, tuple)) else cot
        b, t = int(r.shape[0]), int(r.shape[1])
        bt = b * t
        ddelta, dh = k_bwd(
            inputs=[
                c.reshape(bt, m * d),
                delta.reshape(bt, d),
                hpost.reshape(bt, m),
            ],
            output_shapes=[(bt, d), (bt, m)],
            output_dtypes=[delta.dtype, hpost.dtype],
            grid=(_NT, bt, 1),
            threadgroup=(_NT, 1, 1),
        )
        return c, ddelta.reshape(delta.shape), dh.reshape(hpost.shape)

    return _op


def mix(r, weight, alpha, bias_pre, bias_post, eps: float = 1e-6):
    """R (B,T,M,D) → (x̃ (B,T,D), h_post (B,T,M))。失败返回 None。"""
    if _DISABLED or r.ndim != 4 or r.dtype not in _METAL:
        return None
    b, t, m, d = (int(x) for x in r.shape)
    if not _tg_ok(m, d):
        return None
    if weight.shape != (2 * m, m * d):
        return None
    key = (m, d, r.dtype, float(eps))
    if key in _MIX_FAILED:
        return None
    try:
        op = _MIX_OPS.get(key)
        if op is None:
            op = _mix_op_factory(m, d, r.dtype, float(eps))
            _MIX_OPS[key] = op
        outs = op(r, weight, alpha, bias_pre, bias_post)
        if key not in _MIX_VERIFIED:
            x, hp = outs[0], outs[1]
            xr, hpr, *_rest = _mix_eager(r, weight, alpha, bias_pre, bias_post, eps)
            mx.eval(x, hp, xr, hpr)
            tol = 2e-4 if r.dtype == mx.float32 else 5e-2
            dx = (x.astype(mx.float32) - xr.astype(mx.float32)).abs().max().item()
            dh = (hp.astype(mx.float32) - hpr.astype(mx.float32)).abs().max().item()
            if not (dx <= tol and dh <= tol):
                raise RuntimeError(f"ihc mix fwd |Δx|={dx:.2e} |Δh|={dh:.2e}")
            _MIX_VERIFIED.add(key)
        return outs[0], outs[1]
    except Exception:
        _MIX_FAILED.add(key)
        return None


def write(r, delta, h_post):
    """R (B,T,M,D) + Δ (B,T,D) → R'。失败返回 None。"""
    if _DISABLED or r.ndim != 4 or r.dtype not in _METAL:
        return None
    b, t, m, d = (int(x) for x in r.shape)
    if not _tg_ok(m, d):
        return None
    if tuple(delta.shape) != (b, t, d) or tuple(h_post.shape) != (b, t, m):
        return None
    key = (m, d, r.dtype)
    if key in _WR_FAILED:
        return None
    try:
        op = _WR_OPS.get(key)
        if op is None:
            op = _write_op_factory(m, d, r.dtype)
            _WR_OPS[key] = op
        out = op(r, delta, h_post)
        if key not in _WR_VERIFIED:
            ref = _write_python(r, delta, h_post)
            mx.eval(out, ref)
            tol = 1e-5 if r.dtype == mx.float32 else 2e-2
            dd = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
            if not (dd <= tol):
                raise RuntimeError(f"ihc write fwd |Δ|={dd:.2e}")
            _WR_VERIFIED.add(key)
        return out
    except Exception:
        _WR_FAILED.add(key)
        return None


def prewarm(m: int, d: int, dtype, eps: float = 1e-6) -> bool:
    if _DISABLED or dtype not in _METAL or not _tg_ok(m, d):
        return False
    mk = (m, d, dtype, float(eps))
    wk = (m, d, dtype)
    if mk in _MIX_FAILED or wk in _WR_FAILED:
        return False
    if mk in _MIX_VERIFIED and wk in _WR_VERIFIED:
        return True
    try:
        r = (mx.random.normal((2, 4, m, d)) * 0.5).astype(dtype)
        w = (mx.random.normal((2 * m, m * d)) * 0.02).astype(dtype)
        alpha = mx.array([0.01], dtype=mx.float32)
        import math

        logit = math.log(1.0 / (m - 1))
        bpre = mx.full((m,), logit)
        bpost = mx.zeros((m,))
        delta = (mx.random.normal((2, 4, d)) * 0.5).astype(dtype)
        mx.eval(r, w, alpha, bpre, bpost, delta)

        def loss_mix(r_, w_, a_, bp, bq):
            x, hp = mix(r_, w_, a_, bp, bq, eps)
            return (
                x.astype(mx.float32).square().sum()
                + hp.astype(mx.float32).square().sum()
            ) / 2

        def loss_mix_e(r_, w_, a_, bp, bq):
            x, hp, *_ = _mix_eager(r_, w_, a_, bp, bq, eps)
            return (
                x.astype(mx.float32).square().sum()
                + hp.astype(mx.float32).square().sum()
            ) / 2

        lg, gg = mx.value_and_grad(loss_mix, argnums=range(5))(r, w, alpha, bpre, bpost)
        lr, gr = mx.value_and_grad(loss_mix_e, argnums=range(5))(
            r, w, alpha, bpre, bpost
        )
        mx.eval(lg, lr, *gg, *gr)
        tol = 2e-4 if dtype == mx.float32 else 5e-2
        if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"ihc mix prewarm loss {lg.item()} vs {lr.item()}")
        glim = 5e-3 if dtype == mx.float32 else 0.5
        for a, b in zip(gg, gr):
            dd = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
            rel = dd / (b.astype(mx.float32).abs().max().item() + 1e-12)
            if rel > glim:
                raise RuntimeError(f"ihc mix grad rel={rel:.2e}")

        def loss_w(r_, dlt, hp):
            return write(r_, dlt, hp).astype(mx.float32).square().sum() / 2

        def loss_we(r_, dlt, hp):
            return _write_python(r_, dlt, hp).astype(mx.float32).square().sum() / 2

        hp0 = (2.0 * mx.sigmoid(mx.random.normal((2, 4, m)))).astype(dtype)
        mx.eval(hp0)
        lg, gg = mx.value_and_grad(loss_w, argnums=range(3))(r, delta, hp0)
        lr, gr = mx.value_and_grad(loss_we, argnums=range(3))(r, delta, hp0)
        mx.eval(lg, lr, *gg, *gr)
        if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"ihc write prewarm loss {lg.item()} vs {lr.item()}")
        for a, b in zip(gg, gr):
            dd = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
            rel = dd / (b.astype(mx.float32).abs().max().item() + 1e-12)
            if rel > glim:
                raise RuntimeError(f"ihc write grad rel={rel:.2e}")
        _MIX_VERIFIED.add(mk)
        _WR_VERIFIED.add(wk)
        return True
    except Exception:
        _MIX_FAILED.add(mk)
        _WR_FAILED.add(wk)
        if os.environ.get("VIBY_IHC_DEBUG"):
            raise
        return False


def prewarm_from_model(model, dtype) -> bool:
    if _DISABLED:
        return True
    cfg = getattr(model, "config", None)
    if cfg is None or not bool(getattr(cfg, "ihc", False)):
        return True
    m = int(getattr(cfg, "ihc_streams", 0) or 0)
    d = int(cfg.hidden_size)
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    if m < 2:
        return True
    return prewarm(m, d, dtype, eps)
