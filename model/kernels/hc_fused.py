"""mHC 的融合 Metal kernel（hc_post 前向 + 反向）。

    out[b,t,m,d] = post[b,t,m]·x[b,t,d] + Σ_j comb[b,t,m,j]·res[b,t,j,d]

现状是"广播乘 + [4,4]×[4,D] 的 4096 个小批量 GEMM + 加"三趟，每层子层要
调用一次、每次前向 24 次（2 个子层 × 12 层）。融合成一个 kernel 之后：

- 一个线程负责一个 (b,t,d)，把 hc 条流的输出**全部**算出来 —— x 只读一次、
  res 的每个元素只读一次、out 只写一遍，总共 ~74 MB 访存
  （x 8MB + res 33MB + out 33MB，B4/T1024/hc4/D1024），理想 ~0.19 ms；
- 反向同样一个 kernel 出 dx 与 dres[j]，dpost/dcomb 是沿 D 的归约，
  用一次批量 matmul 算（输出只有 [BT,hc] / [BT,hc,hc]）。

精度：post/comb 是 fp32，kernel 内全程 fp32 累加、只在写回时降到激活 dtype，
比原先"先降到 bf16 再乘"更准。

开关：`VIBY_HC_KERNEL=0` 回退纯 MLX 实现（数值对拍 / 排障用）。
"""

import os

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_HC_KERNEL", "1") != "0"

_FWD_CACHE = {}
_BWD_CACHE = {}

_FWD_TEMPLATE = """
    uint idx = thread_position_in_grid.x;
    const uint D = (uint)dims[0];
    const uint HC = (uint)dims[1];
    const uint N = (uint)dims[2];   // = B*T
    if (idx >= N * D) return;
    uint d = idx % D;
    uint bt = idx / D;
    T xv = x[bt * D + d];
    float acc[{HC}];
    #pragma unroll
    for (uint m = 0; m < {HC}; ++m) acc[m] = post[bt * {HC} + m] * (float)xv;
    #pragma unroll
    for (uint j = 0; j < {HC}; ++j) {{
        float r = (float)res[(bt * {HC} + j) * D + d];
        #pragma unroll
        for (uint m = 0; m < {HC}; ++m)
            acc[m] += comb[bt * {HC} * {HC} + m * {HC} + j] * r;
    }}
    #pragma unroll
    for (uint m = 0; m < {HC}; ++m) out[(bt * {HC} + m) * D + d] = (T)acc[m];
"""

_BWD_TEMPLATE = """
    uint idx = thread_position_in_grid.x;
    const uint D = (uint)dims[0];
    const uint HC = (uint)dims[1];
    const uint N = (uint)dims[2];   // = B*T
    if (idx >= N * D) return;
    uint d = idx % D;
    uint bt = idx / D;
    float gv[{HC}];
    #pragma unroll
    for (uint m = 0; m < {HC}; ++m) gv[m] = (float)g[(bt * {HC} + m) * D + d];
    float ax = 0.0f;
    #pragma unroll
    for (uint m = 0; m < {HC}; ++m) ax += gv[m] * post[bt * {HC} + m];
    dx[bt * D + d] = (T)ax;
    #pragma unroll
    for (uint j = 0; j < {HC}; ++j) {{
        float a = 0.0f;
        #pragma unroll
        for (uint m = 0; m < {HC}; ++m)
            a += gv[m] * comb[bt * {HC} * {HC} + m * {HC} + j];
        dres[(bt * {HC} + j) * D + d] = (T)a;
    }}
"""


def _fwd_kernel(hc: int):
    k = _FWD_CACHE.get(hc)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"hc_post_fwd_hc{hc}",
            input_names=["x", "res", "post", "comb", "dims"],
            output_names=["out"],
            source=_FWD_TEMPLATE.format(HC=hc),
        )
        _FWD_CACHE[hc] = k
    return k


def _bwd_kernel(hc: int):
    k = _BWD_CACHE.get(hc)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"hc_post_bwd_hc{hc}",
            input_names=["g", "res", "post", "comb", "dims"],
            output_names=["dx", "dres"],
            source=_BWD_TEMPLATE.format(HC=hc),
        )
        _BWD_CACHE[hc] = k
    return k


def _hc_post_ref(x, residual, post, comb):
    """纯 MLX 参考实现（与 model/hc.py 的原实现一致）。"""
    dt = x.dtype
    return (
        post.astype(dt)[..., None] * x[..., None, :]
        + mx.matmul(comb.astype(dt), residual)
    ).astype(dt)


@mx.custom_function
def _hc_post_kernel(x, residual, post, comb, dims):
    hc = residual.shape[-2]
    bt = residual.size // (hc * residual.shape[-1])
    grid = bt * residual.shape[-1]
    (out,) = _fwd_kernel(hc)(
        inputs=[x, residual, post, comb, dims],
        template=[("T", x.dtype)],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )
    return out


@_hc_post_kernel.vjp
def _hc_post_kernel_vjp(primals, cotangent, output):
    x, residual, post, comb, dims = primals
    g = cotangent
    hc = residual.shape[-2]
    d = residual.shape[-1]
    bt = residual.size // (hc * d)

    def _as2d(a):  # [BT, ...]
        return a.reshape(bt, -1)

    g2 = g.reshape(bt, hc, d)
    res2 = residual.reshape(bt, hc, d)
    x2 = x.reshape(bt, d)
    post2 = post.reshape(bt, hc)
    comb2 = comb.reshape(bt, hc, hc)

    (dx2, dres2) = _bwd_kernel(hc)(
        inputs=[g2, res2, post2, comb2, dims],
        template=[("T", g.dtype)],
        grid=(bt * d, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x2.shape, res2.shape],
        output_dtypes=[g.dtype, g.dtype],
    )
    # post/comb 的梯度是沿 D 的归约：一次批量 matmul（输出很小）
    dpost = mx.sum(g2 * x2[:, None, :], axis=-1).astype(post.dtype)
    dcomb = mx.matmul(g2, res2.swapaxes(-1, -2)).astype(comb.dtype)

    def _as_orig(a, like):
        return a.reshape(like.shape)

    return (
        _as_orig(dx2, x),
        _as_orig(dres2, residual),
        _as_orig(dpost, post),
        _as_orig(dcomb, comb),
        None,
    )


def hc_post_fused(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array):
    """融合版 hc_post；kernel 不可用时自动回退纯 MLX 实现。"""
    global _ENABLED
    if not _ENABLED or x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return _hc_post_ref(x, residual, post, comb)
    try:
        dims = mx.array(
            [residual.shape[-1], residual.shape[-2], x.size // x.shape[-1]],
            dtype=mx.int32,
        )
        return _hc_post_kernel(x, residual, post, comb, dims)
    except Exception as exc:  # noqa: BLE001
        _ENABLED = False
        print(f"[hc_fused] kernel 不可用，回退纯 MLX 实现：{type(exc).__name__}: {exc}")
        return _hc_post_ref(x, residual, post, comb)


def prewarm_hc_post(hc: int = 4, dtype=mx.bfloat16, dim: int = 8):
    """在 eager 上下文里先把 kernel 编出来。

    必须在 `mx.compile` 的 trace 之前调用：Metal 库是首次 dispatch 时编译的，
    如果这次编译落在 compile 的 trace 里，host 侧的同步会出问题（旧实现的
    model/kernels/__init__.py 记过这个坑）。这里用 1×dim 的哑输入预热，
    kernel 只按 (HC, dtype) 特化，与真实形状无关。
    """
    if not _ENABLED:
        return False
    try:
        x = mx.zeros((1, dim), dtype=dtype)
        res = mx.zeros((1, hc, dim), dtype=dtype)
        post = mx.zeros((1, hc), dtype=mx.float32)
        comb = mx.zeros((1, hc, hc), dtype=mx.float32)
        dims = mx.array([dim, hc, 1], dtype=mx.int32)
        mx.eval(_hc_post_kernel(x, res, post, comb, dims))
        if dtype != mx.float32:  # fp32 训练路径也预热一份，避免首调用落在 compile trace 里
            xf = x.astype(mx.float32)
            resf = res.astype(mx.float32)
            mx.eval(_hc_post_kernel(xf, resf, post, comb, dims))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[hc_fused] 预热失败，回退纯 MLX 实现：{type(exc).__name__}: {exc}")
        return False
