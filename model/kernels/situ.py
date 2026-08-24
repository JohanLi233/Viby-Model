"""SiTU-GLU 训练融合 Metal kernel（fwd + 手写 VJP）。

eager 是 5+ 个逐元素核（tanh/sigmoid/mul），compile 下仍约 5.4ms
（G=98304×I=384）。本模块一次读 g/u、一次写 y；反向重算 tanh/σ
不物化中间量。失败回退 `acts.situ_glu` 公式。

两个入口：
- `situ_glu(g, u)`：两个独立张量。
- `situ_glu_packed(h)`：h 的最后一维是 [g | u] 拼接（gate/up 合并成
  一次 GEMM 的自然输出布局）。调用方切出的 `h[..., :I]` / `h[..., I:]`
  是跨步视图，两参数入口里的 `reshape(-1)` 会各物化一份连续副本，反向
  还要把 dg/du 拼回 (..., 2I)。打包入口在核内直接寻址两半，省掉
  fwd 两次、bwd 两次拷贝与一次拼接（MoE 路由路径 G=98304×I=384 下
  单层 f+b 实测 8.6ms → 1.7ms）。逐元素算式与两参数核逐位相同。
"""

import os

import mlx.core as mx

from ..acts import SITU_BETA1, SITU_BETA2, situ_glu_eager

_DISABLED = os.environ.get("VIBY_SITU_FUSED", "1") != "1"
_VERIFIED = False
_OP = None
_METAL = {mx.float32: "float", mx.bfloat16: "bfloat"}


def _build(dtype):
    mt = _METAL[dtype]
    b1, b2 = float(SITU_BETA1), float(SITU_BETA2)
    fwd = mx.fast.metal_kernel(
        name=f"situ_fwd_{mt}",
        input_names=["g", "u"],
        output_names=["y"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint n = g_shape[0];
        if (i >= n) return;
        float gv = float(g[i]);
        float uv = float(u[i]);
        float t1 = metal::tanh(gv / {b1}f);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float t2 = metal::tanh(uv / {b2}f);
        y[i] = {mt}(({b1}f * t1 * s) * ({b2}f * t2));
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"situ_bwd_{mt}",
        input_names=["g", "u", "cot"],
        output_names=["dg", "du"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint n = g_shape[0];
        if (i >= n) return;
        float gv = float(g[i]);
        float uv = float(u[i]);
        float c = float(cot[i]);
        float t1 = metal::tanh(gv / {b1}f);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float t2 = metal::tanh(uv / {b2}f);
        float gate = {b1}f * t1 * s;
        float up = {b2}f * t2;
        float dgate = c * up;
        float dup = c * gate;
        float ds = s * (1.0f - s);
        float dt1 = 1.0f - t1 * t1;
        dg[i] = {mt}(dgate * (dt1 * s + {b1}f * t1 * ds));
        du[i] = {mt}(dup * (1.0f - t2 * t2));
        """,
    )
    return fwd, bwd


def _tg_width(half: int) -> int:
    """线程组沿通道轴的宽度：half 的最大 32 倍数因子（≤256）。

    二维 grid 用 (c, row) 而不是展平的一维索引，是为了避开每线程一次
    `i / half` 整数除法——GPU 无硬件整除，37.7M 个元素上实测这一项就
    把 fwd 从访存下界抬高 5 倍。
    """
    for d in (256, 192, 128, 96, 64, 32):
        if half % d == 0:
            return d
    return min(half, 32)


def _build_packed(dtype, half: int):
    """打包布局：g = h[..., :half]、u = h[..., half:]，逐元素算式同 _build。"""
    mt = _METAL[dtype]
    b1, b2 = float(SITU_BETA1), float(SITU_BETA2)
    # thread (c, row)：行首偏移 row*2*half，两半相距 half；相邻线程 c 相邻
    # ⇒ 两半的读与写全部合并访存。
    addr = f"""
        uint c = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        constexpr uint HALF = {half};
        if (c >= HALF) return;
        size_t hb = (size_t)row * (2 * HALF);
        size_t yb = (size_t)row * HALF;
        float gv = float(h[hb + c]);
        float uv = float(h[hb + HALF + c]);
    """
    fwd = mx.fast.metal_kernel(
        name=f"situ_pk_fwd_{mt}_{half}",
        input_names=["h"],
        output_names=["y"],
        source=f"""
        {addr}
        float t1 = metal::tanh(gv / {b1}f);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float t2 = metal::tanh(uv / {b2}f);
        y[yb + c] = {mt}(({b1}f * t1 * s) * ({b2}f * t2));
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"situ_pk_bwd_{mt}_{half}",
        input_names=["h", "cot"],
        output_names=["dh"],
        source=f"""
        {addr}
        float c_ = float(cot[yb + c]);
        float t1 = metal::tanh(gv / {b1}f);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float t2 = metal::tanh(uv / {b2}f);
        float gate = {b1}f * t1 * s;
        float up = {b2}f * t2;
        float dgate = c_ * up;
        float dup = c_ * gate;
        float ds = s * (1.0f - s);
        float dt1 = 1.0f - t1 * t1;
        dh[hb + c] = {mt}(dgate * (dt1 * s + {b1}f * t1 * ds));
        dh[hb + HALF + c] = {mt}(dup * (1.0f - t2 * t2));
        """,
    )
    return fwd, bwd


def _packed_op_factory(dtype, half: int):
    k_fwd, k_bwd = _build_packed(dtype, half)
    tgx = _tg_width(half)
    gx = (half + tgx - 1) // tgx * tgx

    @mx.custom_function
    def _op(h):
        rows = h.size // (2 * half)
        (y,) = k_fwd(
            inputs=[h],
            output_shapes=[(rows * half,)],
            output_dtypes=[h.dtype],
            grid=(gx, rows, 1),
            threadgroup=(tgx, 1, 1),
        )
        return y.reshape(*h.shape[:-1], half)

    @_op.vjp
    def _op_vjp(primals, cot, output):
        # 单入参时 mx.custom_function 直接传数组本身，多入参才传元组
        h = primals[0] if isinstance(primals, (list, tuple)) else primals
        c = cot[0] if isinstance(cot, (list, tuple)) else cot
        rows = h.size // (2 * half)
        (dh,) = k_bwd(
            inputs=[h, c.reshape(-1)],
            output_shapes=[(h.size,)],
            output_dtypes=[h.dtype],
            grid=(gx, rows, 1),
            threadgroup=(tgx, 1, 1),
        )
        return dh.reshape(h.shape)

    return _op


def _op_factory(dtype):
    k_fwd, k_bwd = _build(dtype)

    @mx.custom_function
    def _op(g, u):
        n = g.size
        tg = 256
        (y,) = k_fwd(
            inputs=[g.reshape(-1), u.reshape(-1)],
            output_shapes=[(n,)],
            output_dtypes=[g.dtype],
            grid=((n + tg - 1) // tg * tg, 1, 1),
            threadgroup=(tg, 1, 1),
        )
        return y.reshape(g.shape)

    @_op.vjp
    def _op_vjp(primals, cot, output):
        g, u = primals
        c = cot[0] if isinstance(cot, (list, tuple)) else cot
        n = g.size
        tg = 256
        dg, du = k_bwd(
            inputs=[g.reshape(-1), u.reshape(-1), c.reshape(-1)],
            output_shapes=[(n,), (n,)],
            output_dtypes=[g.dtype, u.dtype],
            grid=((n + tg - 1) // tg * tg, 1, 1),
            threadgroup=(tg, 1, 1),
        )
        return dg.reshape(g.shape), du.reshape(u.shape)

    return _op


_OPS: dict = {}
_PACKED_OPS: dict = {}


def situ_glu_packed(h: mx.array):
    """h (..., 2I) → SiTU-GLU(h[..., :I], h[..., I:])，形状 (..., I)。

    不支持的 dtype / kernel 失败时回退 eager 公式（此时才切片）。
    最后一维为奇数是调用方的布局错误，直接抛 ValueError。
    """
    global _DISABLED
    W = h.shape[-1]
    if W % 2 != 0:
        raise ValueError(f"situ_glu_packed 需要偶数末维（[g|u] 拼接），得到 {W}")
    half = W // 2
    if _DISABLED or h.dtype not in _METAL:
        return situ_glu_eager(h[..., :half], h[..., half:])
    key = (h.dtype, half)
    try:
        op = _PACKED_OPS.get(key)
        if op is None:
            op = _packed_op_factory(*key)
            _PACKED_OPS[key] = op
        return op(h)
    except Exception:
        _DISABLED = True
        return situ_glu_eager(h[..., :half], h[..., half:])


def situ_glu(g: mx.array, u: mx.array):
    """融合 SiTU-GLU；不支持的 dtype / 未校验 / 失败时回退 eager 公式。"""
    global _DISABLED
    if _DISABLED or g.dtype not in _METAL or u.dtype != g.dtype:
        return situ_glu_eager(g, u)
    key = g.dtype
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(key)
            _OPS[key] = op
        y = op(g, u)
        if not _VERIFIED:
            return y  # prewarm 负责首次校验；训练图内已 prewarm
        return y
    except Exception:
        _DISABLED = True
        return situ_glu_eager(g, u)


def prewarm(dtype=mx.bfloat16, n: int = 4096) -> bool:
    """compile 前校验 fwd+bwd 对照 eager。"""
    global _DISABLED, _VERIFIED
    if _DISABLED:
        return False
    if _VERIFIED:
        return True
    try:
        op = _OPS.get(dtype)
        if op is None:
            op = _op_factory(dtype)
            _OPS[dtype] = op
        g = (mx.random.normal((n,)) * 0.7).astype(dtype)
        u = (mx.random.normal((n,)) * 0.7).astype(dtype)

        def f_op(g_, u_):
            return (op(g_, u_).astype(mx.float32) ** 2).sum()

        def f_ref(g_, u_):
            return (situ_glu_eager(g_, u_).astype(mx.float32) ** 2).sum()

        lg, (gg, ug) = mx.value_and_grad(f_op, argnums=(0, 1))(g, u)
        lr, (gr, ur) = mx.value_and_grad(f_ref, argnums=(0, 1))(g, u)
        mx.eval(lg, lr, gg, ug, gr, ur)
        tol = 2e-4 if dtype == mx.float32 else 5e-2
        if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"situ prewarm loss {lg.item()} vs {lr.item()}")
        for a, b in ((gg, gr), (ug, ur)):
            rel = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item() / (
                b.astype(mx.float32).abs().max().item() + 1e-12
            )
            if rel > (5e-3 if dtype == mx.float32 else 0.5):
                raise RuntimeError(f"situ prewarm grad rel={rel:.2e}")
        _VERIFIED = True
        return True
    except Exception:
        _DISABLED = True
        return False


def prewarm_packed(dtype, halves) -> bool:
    """compile 前按 half 宽度预编译打包核并对照两参数核（应逐位一致）。

    halves 是实际会用到的 I 值集合（每个 I 一份 kernel）。任一失败即整体
    关闭融合 situ（与两参数入口共用 _DISABLED，回退路径一致）。
    """
    global _DISABLED
    if _DISABLED or dtype not in _METAL:
        return False
    try:
        for half in sorted({int(v) for v in halves if int(v) > 0}):
            key = (dtype, half)
            op = _PACKED_OPS.get(key)
            if op is None:
                op = _packed_op_factory(dtype, half)
                _PACKED_OPS[key] = op
            h = (mx.random.normal((7, 2 * half)) * 0.7).astype(dtype)
            cot = (mx.random.normal((7, half))).astype(dtype)

            def f_pk(h_):
                return (op(h_).astype(mx.float32) * cot.astype(mx.float32)).sum()

            def f_ref(h_):
                y = situ_glu_eager(h_[..., :half], h_[..., half:])
                return (y.astype(mx.float32) * cot.astype(mx.float32)).sum()

            lp, dp = mx.value_and_grad(f_pk)(h)
            lr, dr = mx.value_and_grad(f_ref)(h)
            mx.eval(lp, lr, dp, dr)
            tol = 2e-4 if dtype == mx.float32 else 5e-2
            if abs(lp.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
                raise RuntimeError(
                    f"situ packed prewarm loss {lp.item()} vs {lr.item()} (I={half})"
                )
            rel = (dp.astype(mx.float32) - dr.astype(mx.float32)).abs().max().item() / (
                dr.astype(mx.float32).abs().max().item() + 1e-12
            )
            if rel > (5e-3 if dtype == mx.float32 else 0.5):
                raise RuntimeError(f"situ packed prewarm grad rel={rel:.2e} (I={half})")
        return True
    except Exception:
        _DISABLED = True
        return False
