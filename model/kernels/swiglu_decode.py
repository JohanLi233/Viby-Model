"""稠密 SiTU-GLU 的 decode 融合 Metal kernel（仅推理前向，小批量）。

h = situ_glu(g, u) → out = h @ W_downᵀ 的两段合成 1 个 kernel。
函数名 silu_mul_down_decode 保留以免改调用点。

布局：threadgroup 负责一个 token 行；h 驻 shared（I 个 f32）。W_down
(D, I) 每线程一行连续读。不可微——training=True 或行数 >
_MAX_ROWS 时调用方回退 eager 链。JIT/校验失败永久回退。
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

_NT = 256
_MAX_ROWS = 512


def _build(D: int, I: int, dtype):  # noqa: E741
    key = (D, I, dtype)
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    mt = _METAL_TYPE[dtype]
    kern = mx.fast.metal_kernel(
        name=f"swiglu_down_{D}_{I}_{mt}",
        input_names=["gu", "wd"],
        output_names=["out"],
        source=f"""
        constexpr uint D = {D};
        constexpr uint I = {I};
        constexpr uint NT = {_NT};
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        threadgroup float h_sh[I];
        size_t gbase = (size_t)row * 2 * I;
        size_t obase = (size_t)row * D;

        // h = SiTU-GLU(g,u)：β1=4, β2=25
        for (uint j = tid; j < I; j += NT) {{
            float g = float(gu[gbase + j]);
            float u = float(gu[gbase + I + j]);
            float sg = 1.0f / (1.0f + metal::exp(-g));
            float gate = 4.0f * metal::tanh(g / 4.0f) * sg;
            float up = 25.0f * metal::tanh(u / 25.0f);
            {mt} hb = ({mt})(float(({mt})(gate * up)));
            h_sh[j] = float(hb);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // out = h @ W_downᵀ（W_down (D,I)，行 i 连续读，f32 累加）
        for (uint i = tid; i < D; i += NT) {{
            float acc = 0.0f;
            size_t wbase = (size_t)i * I;
            for (uint j = 0; j < I; j++) {{
                acc += h_sh[j] * float(wd[wbase + j]);
            }}
            out[obase + i] = ({mt})acc;
        }}
        """,
    )
    _KERNELS[key] = kern
    return kern


def silu_mul_down_decode(gu, wd):
    """gu (M, 2I) bf16（gate|up 拼接 GEMM 输出），wd (D, I) down 权重。
    返回 (M, D) 或 None（回退 eager）。"""
    global _DISABLED
    if _DISABLED or gu.dtype not in _METAL_TYPE or wd.dtype != gu.dtype:
        return None
    M, I2 = gu.shape
    if I2 % 2:
        return None
    I = I2 // 2  # noqa: E741
    D = wd.shape[0]
    if wd.shape != (D, I) or M > _MAX_ROWS:
        return None
    key = (D, I, gu.dtype)
    try:
        kern = _build(D, I, gu.dtype)
        (out,) = kern(
            inputs=[gu, wd],
            output_shapes=[(M, D)],
            output_dtypes=[gu.dtype],
            grid=(_NT, M, 1),
            threadgroup=(_NT, 1, 1),
        )
        if key not in _VERIFIED:
            from ..acts import situ_glu

            g, u = mx.split(gu, 2, axis=-1)
            ref = situ_glu(g, u) @ wd.T
            mx.eval(out, ref)  # 触发 JIT；失败走 except 永久回退
            d = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
            tol = 1e-5 if gu.dtype == mx.float32 else 2e-2
            if d > tol:
                raise RuntimeError(f"swiglu_down 校验失败 |Δ|={d:.2e} (key={key})")
            _VERIFIED.add(key)
        return out
    except Exception:
        _DISABLED = True
        return None
