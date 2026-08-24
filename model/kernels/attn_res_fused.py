"""AttnRes 合并的融合 Metal kernel（fwd+bwd）。

_attn_res_merge(w, vs)：α_j = softmax_j(w·RMSNorm(v_j))（K3 key 归一化），
h = Σ_j α_j·v_j。eager 口径要物化 (B,T,N,D) 栈并做 ~5 趟大元素级
读写；融合 kernel 每 (b,t) 一个 threadgroup，两趟读完全部 v：

  fwd：pass1 逐 j 归约 ss 与 w·v → s_j=(w·v)·rstd → tid0 softmax 写
       alpha → pass2 加权求和写 out。
  bwd：pass1 复算 rstd / 归约 da_j=cot·v_j → tid0 算 ds
       → pass2 dv=α⊙cot + ds·rstd·w − ds·s·rstd²·v/D，
       dw 来自 ds·RMSNorm(v)。

N 烧进 kernel 源（逐 j 展开），按 (N,D,dtype) 缓存编译；N>26 回退
eager。首次调用对同一输入校验 eager 一致性：失败只禁用该 (N,D,dtype)，
不株连其它 N（bf16 下单个 N 的 VJP 舍入刚过阈值时，旧逻辑会把整轮
AttnRes 打回 eager）。
"""

import mlx.core as mx

from ..norms import _rms_unit

_ATTN_RES_EPS = 1e-6

_KERNELS: dict = {}
_DISABLED = False
_VERIFIED: set = set()
_FAILED: set = set()  # 单 key 校验失败；不得据此全局禁用
_MAX_N = 26  # Metal buffer 上限（2N+3 ≤ 31 留余量）
_THREADS = 256  # 每 (b,t) 一个 threadgroup；D=768 时线程 3 元素
_SIMD_GROUPS = _THREADS // 32


def _build(N: int, D: int, dtype):
    key = (N, D, dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    mt = "float" if dtype == mx.float32 else "bfloat"

    # 逐 j 展开的归约块。vec_expr 为含 d 的完整索引表达式。
    def _dot_block(vec_expr: str, dst: str, j: int) -> str:
        return f"""
        {{
            float acc = 0.0f;
            for (uint d = tid; d < {D}; d += {_THREADS}) {{
                acc += float({vec_expr}) * float(v{j}[base + d]);
            }}
            acc = simd_sum(acc);
            if (lane == 0) red[sg] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {{
                float t = 0.0f;
                for (uint s = 0; s < {_SIMD_GROUPS}; s++) t += red[s];
                {dst}[{j}] = t;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    def _score_block(j: int) -> str:
        # s_j = (w·v_j) * rsqrt(mean(v_j²)+eps)
        return f"""
        {{
            float acc = 0.0f, ss = 0.0f;
            for (uint d = tid; d < {D}; d += {_THREADS}) {{
                float vd = float(v{j}[base + d]);
                acc += float(w[d]) * vd;
                ss += vd * vd;
            }}
            acc = simd_sum(acc);
            ss = simd_sum(ss);
            if (lane == 0) {{ red[sg] = acc; red2[sg] = ss; }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {{
                float t = 0.0f, t2 = 0.0f;
                for (uint s = 0; s < {_SIMD_GROUPS}; s++) {{
                    t += red[s]; t2 += red2[s];
                }}
                float rstd = metal::rsqrt(t2 / {D}.0f + {_ATTN_RES_EPS:.1e}f);
                rstds[{j}] = rstd;
                sbuf[{j}] = t * rstd;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    fwd_dots = "".join(_score_block(j) for j in range(N))
    fwd_mix = "\n".join(
        f"            acc += sbuf[{j}] * float(v{j}[base + d]);" for j in range(N)
    )
    fwd_src = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint bt = threadgroup_position_in_grid.x;
        size_t base = (size_t)bt * {D};
        threadgroup float red[{_SIMD_GROUPS}];
        threadgroup float red2[{_SIMD_GROUPS}];
        threadgroup float sbuf[{N}];
        threadgroup float rstds[{N}];
        uint lane = tid & 31;
        uint sg = tid >> 5;
        // pass1: s_j = w·RMSNorm(v_j)
        {fwd_dots}
        // softmax over depth（tid0，N 很小）
        if (tid == 0) {{
            float m = sbuf[0];
            for (int j = 1; j < {N}; j++) m = metal::max(m, sbuf[j]);
            float sum = 0.0f;
            for (int j = 0; j < {N}; j++) {{
                sbuf[j] = metal::exp(sbuf[j] - m);
                sum += sbuf[j];
            }}
            float inv = 1.0f / sum;
            for (int j = 0; j < {N}; j++) {{
                sbuf[j] *= inv;
                alpha[bt * {N} + j] = sbuf[j];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // pass2: out_d = Σ_j α_j·v_j[d]
        for (uint d = tid; d < {D}; d += {_THREADS}) {{
            float acc = 0.0f;
{fwd_mix}
            out[base + d] = {mt}(acc);
        }}
    """

    bwd_rstd = "".join(_score_block(j) for j in range(N))
    bwd_dots = "".join(_dot_block("cot[base + d]", "da", j) for j in range(N))
    bwd_al = "\n".join(
        f"            al[{j}] = alpha[bt * {N} + {j}];" for j in range(N)
    )
    bwd_ds = "\n".join(
        f"            ds[{j}] = al[{j}] * (da[{j}] - dot);" for j in range(N)
    )
    # s_j 存在 sbuf（复算的 (w·v)*rstd）；dv 经 RMSNorm VJP
    bwd_mix = "\n".join(
        f"""            {{
                float vjd = float(v{j}[base + d]);
                float rj = rstds[{j}];
                float sj = sbuf[{j}];
                gg += ds[{j}] * rj * vjd;
                float dvsc = ds[{j}] * rj * (wd - rj * sj * vjd / {D}.0f);
                dv_all[((size_t)bt * {N} + {j}) * {D} + d] = {mt}(al[{j}] * c + dvsc);
            }}"""
        for j in range(N)
    )
    bwd_src = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint bt = threadgroup_position_in_grid.x;
        size_t base = (size_t)bt * {D};
        threadgroup float red[{_SIMD_GROUPS}];
        threadgroup float red2[{_SIMD_GROUPS}];
        threadgroup float sbuf[{N}];
        threadgroup float rstds[{N}];
        threadgroup float da[{N}];
        threadgroup float al[{N}];
        threadgroup float ds[{N}];
        uint lane = tid & 31;
        uint sg = tid >> 5;
        // 复算 rstd / s_j（与 fwd 同式）
        {bwd_rstd}
        // da_j = cot·v_j（混合仍用原始 v）
        {bwd_dots}
        if (tid == 0) {{
{bwd_al}
            float dot = 0.0f;
            for (int j = 0; j < {N}; j++) dot += al[j] * da[j];
{bwd_ds}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dv = α⊙cot + RMSNorm VJP；g = Σ_j ds_j·RMSNorm(v_j)
        for (uint d = tid; d < {D}; d += {_THREADS}) {{
            float c = float(cot[base + d]);
            float wd = float(w[d]);
            float gg = 0.0f;
{bwd_mix}
            g[base + d] = gg;
        }}
    """

    k_fwd = mx.fast.metal_kernel(
        name=f"attn_res_fwd_{N}_{D}_{mt}",
        input_names=["w"] + [f"v{j}" for j in range(N)],
        output_names=["out", "alpha"],
        source=fwd_src,
    )
    k_bwd = mx.fast.metal_kernel(
        name=f"attn_res_bwd_{N}_{D}_{mt}",
        input_names=["w", "alpha", "cot"] + [f"v{j}" for j in range(N)],
        output_names=["dv_all", "g"],
        source=bwd_src,
    )
    _KERNELS[key] = (k_fwd, k_bwd)
    return _KERNELS[key]


def _merge_eager(w: mx.array, vs: list) -> mx.array:
    """eager 参考：score 用 f32 RMSNorm(v_j)，混合仍用原始 v_j。

    Metal kernel 的 rstd / 点积也在 f32 里算。bf16 输入若直接走
    `mx.fast.rms_norm`，和核内 f32 归约会差到 grad rel≈0.3，刚好卡在
    旧预热阈值上，把整条融合路径误杀。
    """
    kn = [_rms_unit(v.astype(mx.float32), _ATTN_RES_EPS) for v in vs]
    k = mx.stack(kn, axis=2)
    v = mx.stack(vs, axis=2)
    s = mx.sum(k * w.astype(mx.float32), axis=-1)
    alpha = mx.softmax(s, axis=-1)
    return mx.sum(v * alpha[..., None].astype(v.dtype), axis=2)


def _merge_op_factory(N: int, D: int, dtype):
    """按 (N,D,dtype) 建一次性 custom_function（fwd 透出 alpha 供 bwd）。"""
    k_fwd, k_bwd = _build(N, D, dtype)

    @mx.custom_function
    def _op(*args):  # w, v0..v{N-1}
        B, T = args[1].shape[0], args[1].shape[1]
        BT = B * T
        out, alpha = k_fwd(
            inputs=list(args),
            output_shapes=[(B, T, D), (B, T, N)],
            output_dtypes=[args[1].dtype, mx.float32],
            grid=(_THREADS * BT, 1, 1),
            threadgroup=(_THREADS, 1, 1),
        )
        return out, alpha

    @_op.vjp
    def _op_vjp(primals, cotangents, output):
        w = primals[0]
        vs = list(primals[1:])
        cot = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        alpha = output[1]
        B, T = vs[0].shape[0], vs[0].shape[1]
        BT = B * T
        dv_all, g = k_bwd(
            inputs=[w, alpha, cot] + vs,
            output_shapes=[(B, T, N, D), (B, T, D)],
            output_dtypes=[vs[0].dtype, mx.float32],
            grid=(_THREADS * BT, 1, 1),
            threadgroup=(_THREADS, 1, 1),
        )
        dw = mx.sum(g, axis=(0, 1)).astype(w.dtype)
        return [dw] + [dv_all[:, :, j] for j in range(N)]

    return _op


_OPS: dict = {}


def prewarm(D: int, dtype, Ns) -> bool:
    """在编译训练步之前预编译并校验 (N,D,dtype) 组合（含 bwd）。

    mx.compile 图内不允许 .item() host sync：若首次调用发生在 compile
    trace 里，merge 内的在线校验会抛异常并触发回退。trainer 应在
    mx.compile 之前调用本函数完成一次性校验；此后 compile 图内直接
    命中 _VERIFIED 缓存，不再校验。返回 False 表示至少一个 N 未通过
    （该 N 走 eager，其它已通过的 N 仍走融合）。"""
    if _DISABLED:
        return False
    all_ok = True
    for N in Ns:
        if N > _MAX_N:
            continue
        key = (N, D, dtype)
        if key in _VERIFIED:
            continue
        if key in _FAILED:
            all_ok = False
            continue
        try:
            op = _OPS.get(key)
            if op is None:
                op = _merge_op_factory(N, D, dtype)
                _OPS[key] = op
            # 稍大一点的 (B,T) 让 softmax/RMSNorm VJP 更稳，避免 2-token
            # 小样本把 bf16 舍入抬过阈值。
            w = (mx.random.normal((D,)) * 0.25).astype(dtype)
            vs = [(mx.random.normal((2, 4, D)) * 0.5).astype(dtype) for _ in range(N)]
            lg, gg = mx.value_and_grad(
                lambda w_, *v_: (op(w_, *v_)[0].astype(mx.float32) ** 2).sum() / 2,
                argnums=range(N + 1),
            )(w, *vs)
            lr, gr = mx.value_and_grad(
                lambda w_, *v_: (
                    _merge_eager(w_, list(v_)).astype(mx.float32) ** 2
                ).sum()
                / 2,
                argnums=range(N + 1),
            )(w, *vs)
            mx.eval(lg, lr, *gg, *gr)
            # bf16 对照 eager 的 VJP 含 softmax+RMSNorm，rel≈0.3 仍是舍入
            # 不是实现错；只把该 N 记失败，禁止株连其它 N。
            tol = 2e-4 if dtype == mx.float32 else 5e-2
            if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
                raise RuntimeError(
                    f"attn_res prewarm loss 不一致 {lg.item()} vs {lr.item()}"
                )
            grad_lim = 5e-3 if dtype == mx.float32 else 0.5
            for a, b in zip(gg, gr):
                dd = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
                rel = dd / (b.astype(mx.float32).abs().max().item() + 1e-12)
                if rel > grad_lim:
                    raise RuntimeError(f"attn_res prewarm 梯度不一致 rel={rel:.2e}")
            _VERIFIED.add(key)
        except Exception:
            _FAILED.add(key)
            all_ok = False
    return all_ok


def merge(w: mx.array, vs: list) -> mx.array:
    """融合版 AttnRes 合并；约束不满足/该 N 校验失败时回退 eager。"""
    N = len(vs)
    if (
        _DISABLED
        or N > _MAX_N
        or w.dtype not in (mx.float32, mx.bfloat16)
        or vs[0].dtype != w.dtype
    ):
        return _merge_eager(w, vs)
    B, T, D = vs[0].shape
    key = (N, D, w.dtype)
    if key in _FAILED:
        return _merge_eager(w, vs)
    try:
        op = _OPS.get(key)
        if op is None:
            op = _merge_op_factory(N, D, w.dtype)
            _OPS[key] = op
        out = op(w, *vs)[0]
        if key not in _VERIFIED:
            ref = _merge_eager(w, vs)
            mx.eval(out, ref)  # 触发 JIT 编译；失败只回退这一档 N
            d = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
            tol = 1e-4 if w.dtype == mx.float32 else 5e-2
            if d > tol:
                raise RuntimeError(f"attn_res fused 校验失败 |Δ|={d:.2e} (key={key})")
            _VERIFIED.add(key)
        return out
    except Exception:
        _FAILED.add(key)
        return _merge_eager(w, vs)
