"""KDA chunk 分解段的 elementwise 融合 Metal kernel（fwd + 手写 VJP）。

替换 `_chunk_kda` 里这段图（每层 fwd 约 12 个 kernel、f+b 实测 6.8ms）：

    gc = cumsum(log_g, axis=-2);  eg = exp(gc)
    qe = q·eg;  ke = k·eg;  ki = k·exp(−gc)
    gl = gc[..., -1, :];  kd = k·exp(gl − gc);  egl = exp(gl)

慢的根因不是算术而是 cumsum 沿倒数第二维（stride = D）：非连续归约 +
每个中间量（gc/eg/qe/ke/ki/kd）都要往返 HBM 一次，(B,H,NC,C,D) f32 单份
36MB，实测 2.05ms fwd 对 0.63ms 的 compulsory 下界。

融合后一个线程负责一个 (bh, nc, d) 列的全部 C 个时间步：累加链留在寄存器
里（C 是编译期常量，acc[C] 完全展开），只读 q/k/log_g 各一次、只写
qe/ke/ki/kd/egl 各一次。相邻线程 d 相邻，读写全部合并。

反向同构（逆序 cumsum 收 dlog_g）：
    dq[t]  = dqe[t]·e[t]
    dk[t]  = dke[t]·e[t] + dki[t]·ie[t] + dkd[t]·e^{gl−acc[t]}
    g[t]   = dqe[t]·qe[t] + dke[t]·ke[t] − dki[t]·ki[t] − dkd[t]·kd[t]
    g[C−1] += Σ_t dkd[t]·kd[t] + degl·e^{gl}      （gl ≡ acc[C−1]）
    dlog_g[t] = Σ_{s≥t} g[s]

首调用按形状键在线校验 fwd+bwd（对照 eager 参考），失败永久回退 eager
（mx.compile trace 内不能 host sync，需调用方在 compile 前 prewarm）。
"""

import os

import mlx.core as mx

_KERNELS: dict = {}
_OPS: dict = {}
_VERIFIED: set = set()
# VIBY_KDA_PREP=0 或 VIBY_FUSED_KERNELS=0：分解段走 eager
_DISABLED = (
    os.environ.get("VIBY_KDA_PREP", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)

_NT = 256


def _build(C: int, D: int, NC: int, nt: int):
    key = (C, D, NC, nt)
    if key in _KERNELS:
        return _KERNELS[key]

    # 每线程一列 (bh, nc, d)：gx = d + D·nc，gy = bh。
    # base 指向该列 t=0 的元素，步长 D。
    head = f"""
        uint gx = thread_position_in_grid.x;
        uint bh = thread_position_in_grid.y;
        constexpr uint C = {C};
        constexpr uint D = {D};
        constexpr uint NC = {NC};
        if (gx >= D * NC) return;
        uint d = gx % D;
        uint nc = gx / D;
        size_t cbh = (size_t)bh * NC + nc;
        size_t base = cbh * C * D + d;
    """

    fwd_src = (
        head
        + """
        float acc[C];
        float a = 0.0f;
        for (uint t = 0; t < C; t++) {
            size_t off = base + (size_t)t * D;
            a += log_g[off];
            acc[t] = a;
            float e = fast::exp(a);
            float ie = fast::exp(-a);
            float kv = k[off];
            qe[off] = q[off] * e;
            ke[off] = kv * e;
            ki[off] = kv * ie;
        }
        float gl = acc[C - 1];
        egl[cbh * D + d] = fast::exp(gl);
        for (uint t = 0; t < C; t++) {
            size_t off = base + (size_t)t * D;
            kd[off] = k[off] * fast::exp(gl - acc[t]);
        }
    """
    )

    bwd_src = (
        head
        + """
        float acc[C];
        float a = 0.0f;
        for (uint t = 0; t < C; t++) {
            a += log_g[base + (size_t)t * D];
            acc[t] = a;
        }
        float gl = acc[C - 1];
        float g[C];
        // gl ≡ acc[C−1]：kd 的 +gl 依赖与 egl 的贡献都落到最后一步
        float gl_extra = degl[cbh * D + d] * fast::exp(gl);
        for (uint t = 0; t < C; t++) {
            size_t off = base + (size_t)t * D;
            float e = fast::exp(acc[t]);
            float ie = fast::exp(-acc[t]);
            float ed = fast::exp(gl - acc[t]);
            float kv = k[off];
            float dqe_v = dqe[off], dke_v = dke[off];
            float dki_v = dki[off], dkd_v = dkd[off];
            float kd_v = kv * ed;
            dq[off] = dqe_v * e;
            dk[off] = dke_v * e + dki_v * ie + dkd_v * ed;
            g[t] = dqe_v * (q[off] * e) + dke_v * (kv * e)
                 - dki_v * (kv * ie) - dkd_v * kd_v;
            gl_extra += dkd_v * kd_v;
        }
        g[C - 1] += gl_extra;
        float s = 0.0f;
        for (int t = (int)C - 1; t >= 0; t--) {
            s += g[t];
            dlog_g[base + (size_t)t * D] = s;
        }
    """
    )

    k_fwd = mx.fast.metal_kernel(
        name=f"kda_prep_fwd_{C}_{D}_{NC}_{nt}",
        input_names=["q", "k", "log_g"],
        output_names=["qe", "ke", "ki", "kd", "egl"],
        source=fwd_src,
    )
    k_bwd = mx.fast.metal_kernel(
        name=f"kda_prep_bwd_{C}_{D}_{NC}_{nt}",
        input_names=["q", "k", "log_g", "dqe", "dke", "dki", "dkd", "degl"],
        output_names=["dq", "dk", "dlog_g"],
        source=bwd_src,
    )
    _KERNELS[key] = (k_fwd, k_bwd)
    return _KERNELS[key]


def _prep_eager(q, k, log_g):
    """eager 参考 / 回退路径（与 _chunk_kda 原式逐项一致）。"""
    gc = mx.cumsum(log_g, axis=-2)
    eg = mx.exp(gc)
    gl = gc[:, :, :, -1, :]
    return (
        q * eg,
        k * eg,
        k * mx.exp(-gc),
        k * mx.exp(gl[:, :, :, None, :] - gc),
        mx.exp(gl),
    )


def _prep_op_factory(C: int, D: int, NC: int, nt: int):
    k_fwd, k_bwd = _build(C, D, NC, nt)
    grid_x = ((D * NC + nt - 1) // nt) * nt

    @mx.custom_function
    def _op(q, k, log_g):
        B, H = q.shape[0], q.shape[1]
        return k_fwd(
            inputs=[q, k, log_g],
            output_shapes=[
                (B, H, NC, C, D),
                (B, H, NC, C, D),
                (B, H, NC, C, D),
                (B, H, NC, C, D),
                (B, H, NC, D),
            ],
            output_dtypes=[mx.float32] * 5,
            grid=(grid_x, B * H, 1),
            threadgroup=(nt, 1, 1),
        )

    @_op.vjp
    def _op_vjp(primals, cotangent, output):
        q, k, log_g = primals
        dqe, dke, dki, dkd, degl = cotangent
        B, H = q.shape[0], q.shape[1]
        return k_bwd(
            inputs=[q, k, log_g, dqe, dke, dki, dkd, degl],
            output_shapes=[(B, H, NC, C, D)] * 3,
            output_dtypes=[mx.float32] * 3,
            grid=(grid_x, B * H, 1),
            threadgroup=(nt, 1, 1),
        )

    return _op


def _supported(C: int, D: int, NC: int) -> bool:
    # acc[C]/g[C] 是每线程寄存器数组，C 过大会溢出到 thread-local memory
    return 0 < C <= 32 and D > 0 and NC > 0


def _verify(C: int, D: int, NC: int, nt: int) -> bool:
    """对照 eager 参考校验 fwd 与 bwd（一次性，按形状键缓存）。"""
    key = (C, D, NC, nt)
    if key in _VERIFIED:
        return True
    mx.random.seed(0)
    B, H = 2, 2
    q = mx.random.normal((B, H, NC, C, D)) * 0.3
    k = mx.random.normal((B, H, NC, C, D)) * 0.3
    lg = -mx.random.uniform(0.001, 2.0, (B, H, NC, C, D))
    mx.eval(q, k, lg)
    op = _prep_op_factory(C, D, NC, nt)
    cots = [mx.random.normal((B, H, NC, C, D)) for _ in range(4)]
    cots.append(mx.random.normal((B, H, NC, D)))
    mx.eval(cots)

    def mk(fn):
        def f(a, b, c):
            outs = fn(a, b, c)
            return sum((o * ct).sum() for o, ct in zip(outs, cots))

        return f

    tol = 2e-5
    for got, ref in zip(op(q, k, lg), _prep_eager(q, k, lg)):
        if mx.abs(got - ref).max().item() > tol * max(1.0, mx.abs(ref).max().item()):
            return False
    g_got = mx.grad(mk(op), argnums=(0, 1, 2))(q, k, lg)
    g_ref = mx.grad(mk(_prep_eager), argnums=(0, 1, 2))(q, k, lg)
    for got, ref in zip(g_got, g_ref):
        if mx.abs(got - ref).max().item() > tol * max(1.0, mx.abs(ref).max().item()):
            return False
    _VERIFIED.add(key)
    return True


def prewarm(C: int, D: int, NC: int, nt: int = _NT) -> bool:
    """compile 前调用：跑一次在线校验并缓存结果（含 host sync）。"""
    global _DISABLED
    if _DISABLED or not _supported(C, D, NC):
        return False
    try:
        if not _verify(C, D, NC, nt):
            _DISABLED = True
            return False
    except Exception:
        _DISABLED = True
        return False
    return True


def kda_prep(q, k, log_g, nt: int = _NT):
    """融合分解段入口。返回 (qe, ke, ki, kd, egl)。

    未经 prewarm 时首次调用做一次 fwd 在线校验（含 host sync，故 compile
    trace 内必须先 prewarm，否则异常被吞掉后永久回退 eager）。
    """
    global _DISABLED
    C, D, NC = q.shape[3], q.shape[4], q.shape[2]
    if _DISABLED or q.dtype != mx.float32 or not _supported(C, D, NC):
        return _prep_eager(q, k, log_g)
    key = (C, D, NC, nt)
    op = _OPS.get(key)
    if op is None:
        op = _prep_op_factory(C, D, NC, nt)
        _OPS[key] = op
    try:
        outs = op(q, k, log_g)
        if key not in _VERIFIED:
            refs = _prep_eager(q, k, log_g)
            mx.eval(outs, refs)
            worst = max(
                ((g - r).abs().max() / (r.abs().max() + 1e-12)).item()
                for g, r in zip(outs, refs)
            )
            if worst > 1e-4:
                raise RuntimeError(f"kda_prep 校验失败 rel={worst:.2e}")
            _VERIFIED.add(key)
        return outs
    except Exception:
        _DISABLED = True
        return _prep_eager(q, k, log_g)
