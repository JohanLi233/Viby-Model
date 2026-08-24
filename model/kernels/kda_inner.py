"""KDA chunk 内线性代数：L / (I+L)⁻¹ / Aqk / w / u。

prep 已经把 qe/ke/ki/kd/egl 融合成 1 个 kernel。剩下这串 16×16 /
16×96 batched GEMM 在 autodiff 下会穿透 3 轮倍增求逆（每轮 2 个 C³），
反向图比前向厚一截。本模块：

  前向：与原 _chunk_kda 同一套 GEMM / 倍倍增，额外物化 X、Afb、ke_b。
  反向：闭式 dL = −Xᵀ dX Xᵀ，不再对倍增循环做 autodiff。

失败/非 f32 回退 eager 原式。
"""

import math
import os

import mlx.core as mx

_OPS: dict = {}
_DISABLED = os.environ.get("VIBY_KDA_INNER", "1") != "1"
_METAL_DISABLED = os.environ.get("VIBY_KDA_INNER_METAL", "1") != "1"
_METAL_OPS: dict = {}
_EYE: dict = {}
_TRIL: dict = {}
_TRIL_S: dict = {}
_NT = 256


def _metal_ok(C, D, Dv):
    return (
        not _METAL_DISABLED
        and C >= 4
        and C <= 16
        and D <= 128
        and Dv == D
        and (C & (C - 1)) == 0
    )


def _eye(C: int) -> mx.array:
    e = _EYE.get(C)
    if e is None:
        e = mx.eye(C, dtype=mx.float32)
        _EYE[C] = e
    return e


def _tril(C: int, strict: bool) -> mx.array:
    cache = _TRIL_S if strict else _TRIL
    m = cache.get(C)
    if m is None:
        m = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1 if strict else 0)
        cache[C] = m
    return m


def _inv_I_plus_L(L: mx.array, C: int) -> mx.array:
    """(I+L)⁻¹，L 严格下三角幂零：倍倍增 log2(C)−1 轮精确。"""
    P = -L
    X = _eye(C) + P
    p = P
    for _ in range(int(math.log2(C)) - 1):
        p = p @ p
        X = X + p @ X
    return X


def _inner_eager(qe, ke, ki, v, beta):
    """与 _chunk_kda 原式同数学：先 L/Aqk 分算再倍增，供对照。"""
    C = ke.shape[-2]
    ke_b = ke * beta[..., None]
    L = mx.where(_tril(C, True), ke_b @ mx.swapaxes(ki, -1, -2), 0)
    X = _inv_I_plus_L(L, C)
    Afb = X * beta[..., None, :]
    w = Afb @ ke
    u = Afb @ v
    Aqk = mx.where(_tril(C, False), qe @ mx.swapaxes(ki, -1, -2), 0)
    return w, u, Aqk


def _inner_fwd(qe, ke, ki, v, beta):
    """与 eager 同 GEMM 切分，额外物化 X/Afb/ke_b 供 VJP。"""
    C = ke.shape[-2]
    ke_b = ke * beta[..., None]
    L = mx.where(_tril(C, True), ke_b @ mx.swapaxes(ki, -1, -2), 0)
    Aqk = mx.where(_tril(C, False), qe @ mx.swapaxes(ki, -1, -2), 0)
    X = _inv_I_plus_L(L, C)
    Afb = X * beta[..., None, :]
    w = Afb @ ke
    u = Afb @ v
    return w, u, Aqk, X, Afb, ke_b


def _inner_vjp(qe, ke, ki, v, beta, dw, du, dAqk, X, Afb, ke_b):
    """闭式 VJP，复用前向物化的 X / Afb / ke_b。"""
    C = ke.shape[-2]

    # w=Afb@ke, u=Afb@v
    dAfb = dw @ mx.swapaxes(ke, -1, -2) + du @ mx.swapaxes(v, -1, -2)
    dke = mx.swapaxes(Afb, -1, -2) @ dw
    dv = mx.swapaxes(Afb, -1, -2) @ du

    # Afb = X * beta[col]
    dX = dAfb * beta[..., None, :]
    dbeta = (dAfb * X).sum(axis=-2)

    # X = (I+L)⁻¹ → dL = −Xᵀ dX Xᵀ，且 L 只吃严格下三角
    Xt = mx.swapaxes(X, -1, -2)
    dL = mx.where(_tril(C, True), -(Xt @ dX @ Xt), 0)

    # L = tril_strict(ke_b @ kiᵀ)：dke_b = dL@ki，dki = dLᵀ@ke_b
    dke_b = dL @ ki
    dki = mx.swapaxes(dL, -1, -2) @ ke_b
    dke = dke + dke_b * beta[..., None]
    dbeta = dbeta + (dke_b * ke).sum(axis=-1)

    # Aqk = tril(qe @ kiᵀ)
    dA = mx.where(_tril(C, False), dAqk, 0)
    dqe = dA @ ki
    dki = dki + mx.swapaxes(dA, -1, -2) @ qe
    return dqe, dke, dki, dv, dbeta


def _metal_build(C: int, D: int):
    """每 (b,h,nc) 一个 threadgroup：C×C / C×D 全在 shared 里做完。"""
    ninv = int(math.log2(C)) - 1
    nt = _NT
    fwd_src = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint NT = {nt};
        size_t bcd = (size_t)row * C * D;
        size_t bcc = (size_t)row * C * C;
        size_t bc = (size_t)row * C;

        threadgroup float ke[C * D];
        threadgroup float ki[C * D];
        threadgroup float qe[C * D];
        threadgroup float vv[C * D];
        threadgroup float beta[C];
        threadgroup float X[C * C];
        threadgroup float P[C * C];
        threadgroup float T[C * C];

        for (uint i = tid; i < C * D; i += NT) {{
            ke[i] = ke_g[bcd + i];
            ki[i] = ki_g[bcd + i];
            qe[i] = qe_g[bcd + i];
            vv[i] = v_g[bcd + i];
        }}
        if (tid < C) beta[tid] = beta_g[bc + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ij = tid; ij < C * C; ij += NT) {{
            uint i = ij / C, j = ij % C;
            float acc = 0.0f;
            if (i > j) {{
                float bi = beta[i];
                for (uint d = 0; d < D; d++)
                    acc += ke[i * D + d] * bi * ki[j * D + d];
            }}
            P[ij] = -acc;
            X[ij] = (i == j ? 1.0f : 0.0f) + (-acc);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < {ninv}; r++) {{
            for (uint ij = tid; ij < C * C; ij += NT) {{
                uint i = ij / C, j = ij % C;
                float pp = 0.0f;
                for (uint k = 0; k < C; k++)
                    pp += P[i * C + k] * P[k * C + j];
                T[ij] = pp;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ij = tid; ij < C * C; ij += NT) P[ij] = T[ij];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ij = tid; ij < C * C; ij += NT) {{
                uint i = ij / C, j = ij % C;
                float px = 0.0f;
                for (uint k = 0; k < C; k++)
                    px += P[i * C + k] * X[k * C + j];
                T[ij] = X[ij] + px;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ij = tid; ij < C * C; ij += NT) X[ij] = T[ij];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Afb = X * beta[col]；Aqk = tril(qe @ kiᵀ)。写出 X/Afb（各 16×16）
        // 供 VJP；ke_b 不落 HBM，反向用 ke·β 现算。
        for (uint ij = tid; ij < C * C; ij += NT) {{
            uint i = ij / C, j = ij % C;
            float afb = X[ij] * beta[j];
            T[ij] = afb;
            Xo[bcc + ij] = X[ij];
            Afbo[bcc + ij] = afb;
            float aqk = 0.0f;
            if (i >= j) {{
                for (uint d = 0; d < D; d++)
                    aqk += qe[i * D + d] * ki[j * D + d];
            }}
            Aqk[bcc + ij] = aqk;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < C * D; i += NT) {{
            uint ci = i / D, d = i % D;
            float tw = 0.0f, tu = 0.0f;
            for (uint j = 0; j < C; j++) {{
                float a = T[ci * C + j];
                tw += a * ke[j * D + d];
                tu += a * vv[j * D + d];
            }}
            w[bcd + i] = tw;
            u[bcd + i] = tu;
        }}
    """
    bwd_src = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint NT = {nt};
        size_t bcd = (size_t)row * C * D;
        size_t bcc = (size_t)row * C * C;
        size_t bc = (size_t)row * C;

        threadgroup float X[C * C];
        threadgroup float Afb[C * C];
        threadgroup float dAfb[C * C];
        threadgroup float dX[C * C];
        threadgroup float dL[C * C];
        threadgroup float dA[C * C];
        threadgroup float beta[C];
        threadgroup float dbeta[C];

        for (uint i = tid; i < C * C; i += NT) {{
            X[i] = Xg[bcc + i];
            Afb[i] = Afbg[bcc + i];
            dA[i] = dAqkg[bcc + i];
        }}
        if (tid < C) beta[tid] = beta_g[bc + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dAfb = dw@keᵀ + du@vᵀ
        for (uint ij = tid; ij < C * C; ij += NT) {{
            uint i = ij / C, j = ij % C;
            float acc = 0.0f;
            for (uint d = 0; d < D; d++)
                acc += dw_g[bcd + i * D + d] * ke_g[bcd + j * D + d]
                    + du_g[bcd + i * D + d] * v_g[bcd + j * D + d];
            dAfb[ij] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dke = Afbᵀ@dw；dv = Afbᵀ@du
        for (uint i = tid; i < C * D; i += NT) {{
            uint j = i / D, d = i % D;
            float acc = 0.0f, accv = 0.0f;
            for (uint k = 0; k < C; k++) {{
                float a = Afb[k * C + j];
                acc += a * dw_g[bcd + k * D + d];
                accv += a * du_g[bcd + k * D + d];
            }}
            dke_g[bcd + i] = acc;
            dv_g[bcd + i] = accv;
        }}
        if (tid < C) {{
            float s = 0.0f;
            for (uint i = 0; i < C; i++) s += dAfb[i * C + tid] * X[i * C + tid];
            dbeta[tid] = s;
        }}
        for (uint ij = tid; ij < C * C; ij += NT)
            dX[ij] = dAfb[ij] * beta[ij % C];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dL = tril_strict(−Xᵀ dX Xᵀ)
        for (uint ij = tid; ij < C * C; ij += NT) {{
            uint i = ij / C, j = ij % C;
            float acc = 0.0f;
            if (i > j) {{
                for (uint p = 0; p < C; p++) {{
                    float t = 0.0f;
                    for (uint q = 0; q < C; q++) t += dX[p * C + q] * X[j * C + q];
                    acc += X[p * C + i] * t;
                }}
                acc = -acc;
            }}
            dL[ij] = acc;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dke += (dL@ki)*β；dki = dLᵀ@ke_b；dβ += (dke_b·ke)
        for (uint i = tid; i < C * D; i += NT) {{
            uint ci = i / D, d = i % D;
            float acc = 0.0f, acci = 0.0f;
            for (uint j = 0; j < C; j++) {{
                acc += dL[ci * C + j] * ki_g[bcd + j * D + d];
                acci += dL[j * C + ci] * (ke_g[bcd + j * D + d] * beta[j]);
            }}
            dke_g[bcd + i] += acc * beta[ci];
            dki_g[bcd + i] = acci;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < C) {{
            float s = 0.0f;
            for (uint d = 0; d < D; d++) {{
                float acc = 0.0f;
                for (uint j = 0; j < C; j++)
                    acc += dL[tid * C + j] * ki_g[bcd + j * D + d];
                s += acc * ke_g[bcd + tid * D + d];
            }}
            dbeta_g[bc + tid] = dbeta[tid] + s;
        }}
        // dA = tril(dAqk)；dqe = dA@ki；dki += dAᵀ@qe
        for (uint ij = tid; ij < C * C; ij += NT) {{
            uint i = ij / C, j = ij % C;
            if (i < j) dA[ij] = 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < C * D; i += NT) {{
            uint ci = i / D, d = i % D;
            float acc = 0.0f, acci = 0.0f;
            for (uint j = 0; j < C; j++) {{
                acc += dA[ci * C + j] * ki_g[bcd + j * D + d];
                acci += dA[j * C + ci] * qe_g[bcd + j * D + d];
            }}
            dqe_g[bcd + i] = acc;
            dki_g[bcd + i] += acci;
        }}
    """
    k_fwd = mx.fast.metal_kernel(
        name=f"kda_inner_fwd5_{C}_{D}",
        input_names=["qe_g", "ke_g", "ki_g", "v_g", "beta_g"],
        output_names=["w", "u", "Aqk", "Xo", "Afbo"],
        source=fwd_src,
    )
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_inner_bwd5_{C}_{D}",
        input_names=[
            "qe_g",
            "ke_g",
            "ki_g",
            "v_g",
            "beta_g",
            "dw_g",
            "du_g",
            "dAqkg",
            "Xg",
            "Afbg",
        ],
        output_names=["dqe_g", "dke_g", "dki_g", "dv_g", "dbeta_g"],
        source=bwd_src,
    )
    return k_fwd, k_bwd


def _metal_op_factory(C: int, D: int):
    k_fwd, k_bwd = _metal_build(C, D)

    @mx.custom_function
    def _op(qe, ke, ki, v, beta):
        B, H, NC = qe.shape[0], qe.shape[1], qe.shape[2]
        rows = B * H * NC
        w, u, Aqk, X, Afb = k_fwd(
            inputs=[
                qe.reshape(rows, C, D),
                ke.reshape(rows, C, D),
                ki.reshape(rows, C, D),
                v.reshape(rows, C, D),
                beta.reshape(rows, C),
            ],
            output_shapes=[
                (rows, C, D),
                (rows, C, D),
                (rows, C, C),
                (rows, C, C),
                (rows, C, C),
            ],
            output_dtypes=[mx.float32] * 5,
            grid=(_NT * rows, 1, 1),
            threadgroup=(_NT, 1, 1),
        )
        return (
            w.reshape(B, H, NC, C, D),
            u.reshape(B, H, NC, C, D),
            Aqk.reshape(B, H, NC, C, C),
            X.reshape(B, H, NC, C, C),
            Afb.reshape(B, H, NC, C, C),
        )

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, ke, ki, v, beta = primals
        dw, du, dAqk = cotangent[0], cotangent[1], cotangent[2]
        X, Afb = output[3], output[4]
        B, H, NC = qe.shape[0], qe.shape[1], qe.shape[2]
        rows = B * H * NC
        dqe, dke, dki, dv, dbeta = k_bwd(
            inputs=[
                qe.reshape(rows, C, D),
                ke.reshape(rows, C, D),
                ki.reshape(rows, C, D),
                v.reshape(rows, C, D),
                beta.reshape(rows, C),
                dw.reshape(rows, C, D),
                du.reshape(rows, C, D),
                dAqk.reshape(rows, C, C),
                X.reshape(rows, C, C),
                Afb.reshape(rows, C, C),
            ],
            output_shapes=[
                (rows, C, D),
                (rows, C, D),
                (rows, C, D),
                (rows, C, D),
                (rows, C),
            ],
            output_dtypes=[mx.float32] * 5,
            grid=(_NT * rows, 1, 1),
            threadgroup=(_NT, 1, 1),
        )
        return (
            dqe.reshape(B, H, NC, C, D),
            dke.reshape(B, H, NC, C, D),
            dki.reshape(B, H, NC, C, D),
            dv.reshape(B, H, NC, C, D),
            dbeta.reshape(B, H, NC, C),
        )

    return _op


def _op_factory():
    @mx.custom_function
    def _op(qe, ke, ki, v, beta):
        w, u, Aqk, _X, _Afb, _keb = _inner_fwd(qe, ke, ki, v, beta)
        return w, u, Aqk

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        qe, ke, ki, v, beta = primals
        dw, du, dAqk = cotangent[0], cotangent[1], cotangent[2]
        _w, _u, _a, X, Afb, ke_b = _inner_fwd(qe, ke, ki, v, beta)
        return _inner_vjp(qe, ke, ki, v, beta, dw, du, dAqk, X, Afb, ke_b)

    return _op


_OP = None


def prewarm(C: int = 16, D: int = 96) -> bool:
    """compile 前编译并校验 Metal inner。失败只关 Metal，Python 闭式 VJP 仍可用。"""
    global _METAL_DISABLED
    if _DISABLED:
        return False
    if _METAL_DISABLED or not _metal_ok(C, D, D):
        return True
    key = (C, D)
    try:
        op = _METAL_OPS.get(key)
        if op is None:
            op = _metal_op_factory(C, D)
            _METAL_OPS[key] = op
        B, H, NC = 1, 2, 2
        mx.random.seed(0)
        qe = mx.random.normal((B, H, NC, C, D)) * 0.1
        ke = mx.random.normal((B, H, NC, C, D)) * 0.1
        ki = mx.random.normal((B, H, NC, C, D)) * 0.1
        v = mx.random.normal((B, H, NC, C, D)) * 0.3
        beta = mx.random.uniform(0, 1, (B, H, NC, C))

        def f_op(*a):
            w, u, A = op(*a)[:3]
            return (w**2).sum() + (u**2).sum() + (A**2).sum()

        def f_ref(*a):
            w, u, A = _inner_eager(*a)
            return (w**2).sum() + (u**2).sum() + (A**2).sum()

        lg, gg = mx.value_and_grad(f_op, argnums=(0, 1, 2, 3, 4))(qe, ke, ki, v, beta)
        lr, gr = mx.value_and_grad(f_ref, argnums=(0, 1, 2, 3, 4))(qe, ke, ki, v, beta)
        mx.eval(lg, lr, *gg, *gr)
        if abs(lg.item() - lr.item()) > 1e-3:
            raise RuntimeError(f"kda_inner prewarm loss {lg.item()} vs {lr.item()}")
        for a, b in zip(gg, gr):
            rel = (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)
            if rel > 2e-4:
                raise RuntimeError(f"kda_inner prewarm grad rel={rel:.2e}")
        return True
    except Exception:
        _METAL_DISABLED = True
        return False


def kda_inner(qe, ke, ki, v, beta):
    """返回 (w, u, Aqk)，形状 (B,H,NC,C,D) / (B,H,NC,C,Dv) / (B,H,NC,C,C)。"""
    global _DISABLED, _OP, _METAL_DISABLED
    if _DISABLED or qe.dtype != mx.float32:
        return _inner_eager(qe, ke, ki, v, beta)
    C, D, Dv = ke.shape[-2], ke.shape[-1], v.shape[-1]
    if _metal_ok(C, D, Dv):
        key = (C, D)
        try:
            op = _METAL_OPS.get(key)
            if op is None:
                op = _metal_op_factory(C, D)
                _METAL_OPS[key] = op
            outs = op(qe, ke, ki, v, beta)
            return outs[0], outs[1], outs[2]
        except Exception:
            _METAL_DISABLED = True
    if _OP is None:
        _OP = _op_factory()
    outs = _OP(qe, ke, ki, v, beta)
    return outs[0], outs[1], outs[2]
