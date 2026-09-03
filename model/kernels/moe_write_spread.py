"""LatentMoE 写出基扩展的融合 Metal kernel（fwd + 手写 VJP）。

eager 要物化 (M,K,hidden) 的 tile(h)*s*α 再对 K 求和。本模块：

  fwd：每 (m, i) 沿 K 累加，不物化 tiled 激活；
  bwd：dy/dw 同一核沿 hidden 扫一遍；ds 按 token 分片写部分和，
       调用方一次 sum（与 conv 的 dw 分片同策略，避开 atomic）。

不把 latent 混合并进本核：那会拆掉 gather_mm 与 (G,d) scatter 的融合。
JIT / 校验失败时回退 eager，不改变数学。
"""

import os

import mlx.core as mx

_METAL = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_KERNELS: dict = {}
_OPS: dict = {}
_VERIFIED: set = set()
_FAILED: set = set()
_DS_CHUNKS = 32
_DISABLED = (
    os.environ.get("VIBY_SPREAD_FUSED", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)


def _tgx(n: int) -> int:
    for t in (256, 128, 64, 32):
        if n % t == 0:
            return t
    return 1


def _eager(y, w, scale, idx):
    """y (M,K,d) × scale[idx] ⊙ tile(y) × w → (M, hidden)。"""
    from ..moe import _tile_to_hidden

    hid = int(scale.shape[-1])
    s = scale[idx].astype(y.dtype)
    tiled = _tile_to_hidden(y, hid)
    return (tiled * s * w.astype(y.dtype)[..., None]).sum(axis=1)


def _build(dtype, d: int, hid: int, k: int, e: int):
    key = (dtype, d, hid, k, e)
    if key in _KERNELS:
        return _KERNELS[key]
    mt = _METAL[dtype]
    nc = _DS_CHUNKS
    fwd = mx.fast.metal_kernel(
        name=f"moe_spread_fwd_{mt}_{d}_{hid}_{k}_{e}",
        input_names=["y", "idx", "w", "scale"],
        output_names=["out"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        constexpr uint DD = {hid};
        constexpr uint DL = {d};
        constexpr uint KK = {k};
        uint M = y_shape[0] / (KK * DL);
        if (i >= DD || m >= M) return;
        uint lat = i % DL;
        float acc = 0.0f;
        for (uint t = 0; t < KK; t++) {{
            uint pair = m * KK + t;
            int ex = idx[pair];
            float wv = float(w[pair]);
            float sv = float(scale[(size_t)ex * DD + i]);
            float hv = float(y[(size_t)pair * DL + lat]);
            acc += wv * sv * hv;
        }}
        out[(size_t)m * DD + i] = {mt}(acc);
        """,
    )
    bwd_act = mx.fast.metal_kernel(
        name=f"moe_spread_bwd_act_{mt}_{d}_{hid}_{k}_{e}",
        input_names=["y", "idx", "w", "scale", "cot"],
        output_names=["dy", "dw"],
        source=f"""
        uint m = thread_position_in_grid.x;
        uint t = thread_position_in_grid.y;
        constexpr uint DD = {hid};
        constexpr uint DL = {d};
        constexpr uint KK = {k};
        uint M = y_shape[0] / (KK * DL);
        if (m >= M || t >= KK) return;
        uint pair = m * KK + t;
        int ex = idx[pair];
        float wv = float(w[pair]);
        float dw_acc = 0.0f;
        for (uint j = 0; j < DL; j++) {{
            float dy_acc = 0.0f;
            float hv = float(y[(size_t)pair * DL + j]);
            for (uint i = j; i < DD; i += DL) {{
                float c = float(cot[(size_t)m * DD + i]);
                float sv = float(scale[(size_t)ex * DD + i]);
                dw_acc += c * sv * hv;
                dy_acc += c * wv * sv;
            }}
            dy[(size_t)pair * DL + j] = {mt}(dy_acc);
        }}
        dw[pair] = {mt}(dw_acc);
        """,
    )
    bwd_ds = mx.fast.metal_kernel(
        name=f"moe_spread_bwd_ds_{mt}_{d}_{hid}_{k}_{e}",
        input_names=["y", "idx", "w", "cot"],
        output_names=["part"],
        source=f"""
        uint i = thread_position_in_grid.x;
        uint chunk = thread_position_in_grid.y;
        constexpr uint DD = {hid};
        constexpr uint DL = {d};
        constexpr uint KK = {k};
        constexpr uint EE = {e};
        constexpr uint NC = {nc};
        uint M = y_shape[0] / (KK * DL);
        if (i >= DD || chunk >= NC) return;
        uint m_per = (M + NC - 1) / NC;
        uint m0 = chunk * m_per;
        uint m1 = metal::min(m0 + m_per, M);
        float acc[EE];
        for (uint ex = 0; ex < EE; ex++) acc[ex] = 0.0f;
        uint lat = i % DL;
        for (uint m = m0; m < m1; m++) {{
            float c = float(cot[(size_t)m * DD + i]);
            for (uint t = 0; t < KK; t++) {{
                uint pair = m * KK + t;
                int ex = idx[pair];
                float hv = float(y[(size_t)pair * DL + lat]);
                acc[ex] += c * float(w[pair]) * hv;
            }}
        }}
        for (uint ex = 0; ex < EE; ex++) {{
            part[((size_t)chunk * EE + ex) * DD + i] = acc[ex];
        }}
        """,
    )
    _KERNELS[key] = (fwd, bwd_act, bwd_ds)
    return _KERNELS[key]


def _op_factory(dtype, d: int, hid: int, k: int, e: int):
    k_fwd, k_act, k_ds = _build(dtype, d, hid, k, e)
    nc = _DS_CHUNKS
    tg_d = (_tgx(hid), 1, 1)
    gx_d = (hid + tg_d[0] - 1) // tg_d[0] * tg_d[0]

    @mx.custom_function
    def _op(y, w, scale, idx):
        M = y.shape[0]
        yf = y.reshape(-1)
        wf = w.reshape(-1)
        sf = scale.reshape(-1)
        idf = idx.reshape(-1)
        (out,) = k_fwd(
            inputs=[yf, idf, wf, sf],
            output_shapes=[(M * hid,)],
            output_dtypes=[dtype],
            grid=(gx_d, M, 1),
            threadgroup=tg_d,
        )
        return out.reshape(M, hid)

    @_op.vjp
    def _op_vjp(primals, cot, output):
        y, w, scale, idx = primals
        c = cot[0] if isinstance(cot, (list, tuple)) else cot
        M = y.shape[0]
        yf = y.reshape(-1)
        wf = w.reshape(-1)
        sf = scale.reshape(-1)
        idf = idx.reshape(-1)
        cf = c.reshape(-1)
        gx_m = (M + 31) // 32 * 32
        dy, dw = k_act(
            inputs=[yf, idf, wf, sf, cf],
            output_shapes=[(M * k * d,), (M * k,)],
            output_dtypes=[dtype, dtype],
            grid=(gx_m, k, 1),
            threadgroup=(32, 1, 1),
        )
        part = k_ds(
            inputs=[yf, idf, wf, cf],
            output_shapes=[(nc * e * hid,)],
            output_dtypes=[mx.float32],
            grid=(gx_d, nc, 1),
            threadgroup=tg_d,
        )[0]
        ds = mx.sum(part.reshape(nc, e, hid), axis=0).astype(scale.dtype)
        return [dy.reshape(y.shape), dw.reshape(w.shape), ds, None]

    return _op


def write_spread(y, w, scale, idx):
    """融合写出扩展。失败返回 None，调用方走 eager。"""
    if _DISABLED or y.dtype not in _METAL:
        return None
    d = int(y.shape[-1])
    hid = int(scale.shape[-1])
    k = int(y.shape[1])
    e = int(scale.shape[0])
    key = (y.dtype, d, hid, k, e)
    if key in _FAILED:
        return None
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(*key)
            _OPS[key] = op
        return op(y, w.astype(y.dtype), scale, idx.astype(mx.int32))
    except Exception:
        _FAILED.add(key)
        return None


def prewarm(dtype, d: int, hid: int, k: int, e: int, m: int = 64) -> bool:
    """compile 前校验 fwd+bwd 对照 eager。"""
    if _DISABLED or dtype not in _METAL:
        return False
    key = (dtype, d, hid, k, e)
    if key in _VERIFIED:
        return True
    if key in _FAILED:
        return False
    try:
        op = _OPS.get(key)
        if op is None:
            op = _op_factory(*key)
            _OPS[key] = op
        mx.random.seed(0)
        y = (mx.random.normal((m, k, d)) * 0.5).astype(dtype)
        w = (mx.random.normal((m, k)) * 0.5).astype(dtype)
        scale = (mx.random.normal((e, hid)) * 0.1).astype(dtype)
        idx = mx.random.randint(0, e, (m, k)).astype(mx.int32)

        def f_op(y_, w_, s_):
            return (op(y_, w_, s_, idx).astype(mx.float32) ** 2).sum()

        def f_ref(y_, w_, s_):
            return (_eager(y_, w_, s_, idx).astype(mx.float32) ** 2).sum()

        lo, go = mx.value_and_grad(f_op, argnums=(0, 1, 2))(y, w, scale)
        lr, gr = mx.value_and_grad(f_ref, argnums=(0, 1, 2))(y, w, scale)
        mx.eval(lo, lr, *go, *gr)
        tol = 2e-4 if dtype == mx.float32 else 5e-2
        if abs(lo.item() - lr.item()) > tol * max(1.0, abs(lr.item())):
            raise RuntimeError(f"spread prewarm loss {lo.item()} vs {lr.item()}")
        grad_lim = 5e-3 if dtype == mx.float32 else 0.5
        for a, b in zip(go, gr):
            rel = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item() / (
                b.astype(mx.float32).abs().max().item() + 1e-12
            )
            if rel > grad_lim:
                raise RuntimeError(f"spread prewarm grad rel={rel:.2e}")
        _VERIFIED.add(key)
        return True
    except Exception:
        _FAILED.add(key)
        return False


def prewarm_from_model(model, dtype) -> bool:
    """扫 write_scale 模块，预热实际 (d, hidden, K, E, dtype)。无则 True。"""
    keys = set()
    try:
        mods = model.named_modules()
    except Exception:
        return True
    for _, mod in mods:
        ws = getattr(mod, "write_scale", None)
        if ws is None:
            continue
        d = int(getattr(mod, "latent_dim", 0) or 0)
        kk = int(getattr(mod, "top_k", 0) or 0)
        if d <= 0 or kk <= 0 or getattr(ws, "ndim", 0) != 2:
            continue
        keys.add((dtype, d, int(ws.shape[1]), kk, int(ws.shape[0])))
    if not keys:
        return True
    ok = True
    for key in sorted(keys):
        if not prewarm(*key):
            ok = False
    return ok
