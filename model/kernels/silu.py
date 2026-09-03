"""SiLU-GLU（SwiGLU）训练融合 Metal kernel（fwd + 手写 VJP）。

eager 是 2+ 个逐元素核（sigmoid/mul），compile 下要多次读写 g/u/y。
本模块一次读 g/u、一次写 y；反向重算 σ 不物化中间量。失败回退
`acts.silu_glu_eager` 公式，与 `situ.py` 等价：默认 `hidden_act=silu`
时也走高温融合核，而非回退 eager。

两个入口：
- `silu_glu(g, u)`：两个独立张量。
- `silu_glu_packed(h)`：h 的最后一维是 [g | u] 拼接（gate/up 合并成
  一次 GEMM 的自然输出布局）。调用方切出的 `h[..., :I]` / `h[..., I:]`
  是跨步视图，两参数入口里的 `reshape(-1)` 会各物化一份连续副本，反向
  还要把 dg/du 拼回 (..., 2I)。打包入口在核内直接寻址两半，省掉
  fwd 两次、bwd 两次拷贝与一次拼接。逐元素算式与两参数核逐位相同。
"""

import os

import mlx.core as mx

from ..acts import silu_glu_eager

_DISABLED = os.environ.get("VIBY_SILU_FUSED", "1") != "1"
_VERIFIED = False
_OP = None
_METAL = {mx.float32: "float", mx.bfloat16: "bfloat"}
_PACKED_VERIFIED: set = set()
_PACKED_FAILED: set = set()  # (dtype, half)；不得据此全局禁用两参数核


def _build(dtype):
    mt = _METAL[dtype]
    fwd = mx.fast.metal_kernel(
        name=f"silu_fwd_{mt}",
        input_names=["g", "u"],
        output_names=["y"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint n = g_shape[0];
        if (i >= n) return;
        float gv = float(g[i]);
        float uv = float(u[i]);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float silu = gv * s;
        y[i] = {mt}(silu * uv);
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"silu_bwd_{mt}",
        input_names=["g", "u", "cot"],
        output_names=["dg", "du"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint n = g_shape[0];
        if (i >= n) return;
        float gv = float(g[i]);
        float uv = float(u[i]);
        float c = float(cot[i]);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float ds = s * (1.0f - s);
        float silu = gv * s;
        dg[i] = {mt}(c * uv * (s + gv * ds));
        du[i] = {mt}(c * silu);
        """,
    )
    return fwd, bwd


def _tg_width(half: int) -> int:
    """线程组沿通道轴的宽度：half 的最大 32 倍数因子（≤256）。

    二维 grid 用 (c, row) 而不是展平的一维索引，是为了避开每线程一次
    `i / half` 整数除法——GPU 无硬件整除，大元素量上实测这一项就把
    fwd 从访存下界抬高数倍。与 situ.py 共用同一选取策略。
    """
    for d in (256, 192, 128, 96, 64, 32):
        if half % d == 0:
            return d
    return min(half, 32)


def _build_packed(dtype, half: int):
    """打包布局：g = h[..., :half]、u = h[..., half:]，逐元素算式同 _build。"""
    mt = _METAL[dtype]
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
        name=f"silu_pk_fwd_{mt}_{half}",
        input_names=["h"],
        output_names=["y"],
        source=f"""
        {addr}
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float silu = gv * s;
        y[yb + c] = {mt}(silu * uv);
        """,
    )
    bwd = mx.fast.metal_kernel(
        name=f"silu_pk_bwd_{mt}_{half}",
        input_names=["h", "cot"],
        output_names=["dh"],
        source=f"""
        {addr}
        float c_ = float(cot[yb + c]);
        float s = 1.0f / (1.0f + metal::exp(-gv));
        float ds = s * (1.0f - s);
        float silu = gv * s;
        dh[hb + c] = {mt}(c_ * uv * (s + gv * ds));
        dh[hb + HALF + c] = {mt}(c_ * silu);
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


def silu_glu_packed(h: mx.array):
    """h (..., 2I) → SiLU-GLU(h[..., :I], h[..., I:])，形状 (..., I)。

    不支持的 dtype / kernel 失败时回退 eager 公式（此时才切片）。最后一维
    为奇数是调用方的布局错误，直接抛 ValueError。
    """
    global _DISABLED
    W = h.shape[-1]
    if W % 2 != 0:
        raise ValueError(f"silu_glu_packed 需要偶数末维（[g|u] 拼接），得到 {W}")
    half = W // 2
    key = (h.dtype, half)
    if _DISABLED or h.dtype not in _METAL or key in _PACKED_FAILED:
        return silu_glu(h[..., :half], h[..., half:])
    try:
        op = _PACKED_OPS.get(key)
        if op is None:
            op = _packed_op_factory(*key)
            _PACKED_OPS[key] = op
        return op(h)
    except Exception:
        _PACKED_FAILED.add(key)
        return silu_glu(h[..., :half], h[..., half:])


def silu_glu(g: mx.array, u: mx.array):
    """融合 SiLU-GLU；不支持的 dtype / 未校验 / 失败时回退 eager 公式。"""
    global _DISABLED
    if _DISABLED or g.dtype not in _METAL or u.dtype != g.dtype:
        return silu_glu_eager(g, u)
    key = g.dtype
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(key)
            _OPS[key] = op
        return op(g, u)
    except Exception:
        _DISABLED = True
        return silu_glu_eager(g, u)


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
            return (silu_glu_eager(g_, u_).astype(mx.float32) ** 2).sum()

        lg, (gg, ug) = mx.value_and_grad(f_op, argnums=(0, 1))(g, u)
        lr, (gr, ur) = mx.value_and_grad(f_ref, argnums=(0, 1))(g, u)
        mx.eval(lg, lr, gg, ug, gr, ur)
        tol = 2e-4 if dtype == mx.float32 else 5e-2
        if abs(lg.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"silu prewarm loss {lg.item()} vs {lr.item()}")
        for a, b in ((gg, gr), (ug, ur)):
            rel = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item() / (
                b.astype(mx.float32).abs().max().item() + 1e-12
            )
            if rel > (5e-3 if dtype == mx.float32 else 0.5):
                raise RuntimeError(f"silu prewarm grad rel={rel:.2e}")
        _VERIFIED = True
        return True
    except Exception:
        _DISABLED = True
        return False


def prewarm_packed(dtype, halves) -> bool:
    """compile 前按 half 宽度预编译打包核并对照 eager 公式。

    halves 是实际会用到的 I 值集合（每个 I 一份 kernel）。单个 I 失败
    只禁用该 (dtype, I)，两参数核与其它 I 不受株连。loss 用 y² 之和
    （与 ``prewarm`` 相同），避免 (y·cot).sum() 正负相消后 |L|~O(1)
    把 bf16 舍入误判成失败。
    """
    if _DISABLED or dtype not in _METAL:
        return False
    all_ok = True
    for half in sorted({int(v) for v in halves if int(v) > 0}):
        key = (dtype, half)
        if key in _PACKED_VERIFIED:
            continue
        if key in _PACKED_FAILED:
            all_ok = False
            continue
        try:
            op = _PACKED_OPS.get(key)
            if op is None:
                op = _packed_op_factory(dtype, half)
                _PACKED_OPS[key] = op
            # 行数让元素量接近 unp unpacked prewarm 的 n=4096，bf16 归约更稳。
            rows = max(8, (4096 + half - 1) // half)
            h = (mx.random.normal((rows, 2 * half)) * 0.7).astype(dtype)

            def f_pk(h_):
                return (op(h_).astype(mx.float32) ** 2).sum()

            def f_ref(h_):
                y = silu_glu_eager(h_[..., :half], h_[..., half:])
                return (y.astype(mx.float32) ** 2).sum()

            lp, dp = mx.value_and_grad(f_pk)(h)
            lr, dr = mx.value_and_grad(f_ref)(h)
            mx.eval(lp, lr, dp, dr)
            tol = 2e-4 if dtype == mx.float32 else 5e-2
            if abs(lp.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
                raise RuntimeError(
                    f"silu packed prewarm loss {lp.item()} vs {lr.item()} (I={half})"
                )
            rel = (dp.astype(mx.float32) - dr.astype(mx.float32)).abs().max().item() / (
                dr.astype(mx.float32).abs().max().item() + 1e-12
            )
            if rel > (5e-3 if dtype == mx.float32 else 0.5):
                raise RuntimeError(f"silu packed prewarm grad rel={rel:.2e} (I={half})")
            _PACKED_VERIFIED.add(key)
        except Exception:
            _PACKED_FAILED.add(key)
            all_ok = False
    return all_ok
