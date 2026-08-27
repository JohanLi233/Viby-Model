"""GatedNorm 的 decode 融合 Metal kernel（仅推理前向，小批量）。

y = RMSNorm(x)·w；g = 2·sigmoid(silu(y @ W_down) @ W_up)；out = y * g。
eager 链每调用 ~7 个 kernel（rms + 2 GEMM + silu/sigmoid/两次乘），
decode 每 token 19 处调用（block rms_attn/rms_mlp ×2、embed、final），
是 launch-bound 开销大头。融合后 1 个 kernel。

每一级的 bf16 中间舍入照抄 eager：rms 输出 bf16 → GEMM f32 累加 bf16 舍
→ silu bf16 → GEMM bf16 → sigmoid bf16 → ×2 bf16 → 相乘 bf16。kernel 用
显式 (bfloat16_t)  cast 复现同一条舍入链。

布局：threadgroup 负责一个 token 行；y/ h 驻 shared（D + RANK 个 f32）。
W_down (D,RANK) / W_up (RANK,D) 各读一遍（coalesced）。不可微——
training=True 或行数 > _MAX_ROWS 时调用方回退 eager 链。

JIT 失败/在线校验不通过永久回退（与 conv.py 同一约定）。
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
_MAX_ROWS = 512  # decode / MTP verify；更大走 eager（训练/preill 可微链）


def _build(D: int, rank: int, dtype, eps: float):
    key = (D, rank, dtype, eps)
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    mt = _METAL_TYPE[dtype]
    kern = mx.fast.metal_kernel(
        name=f"gated_norm_{D}_{rank}_{mt}",
        input_names=["x", "wn", "wd", "wu"],
        output_names=["out"],
        source=f"""
        constexpr uint D = {D};
        constexpr uint RANK = {rank};
        constexpr uint NT = {_NT};
        constexpr float EPS = {eps!r}f;
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.y;
        threadgroup float y_sh[D];
        threadgroup float h_sh[RANK];
        threadgroup float red[NT / 32];
        size_t base = (size_t)row * D;

        // ---- RMSNorm（f32 归约，bf16 输出照抄 mx.fast.rms_norm）----
        float acc = 0.0f;
        for (uint i = tid; i < D; i += NT) {{
            float t = float(x[base + i]);
            acc += t * t;
        }}
        acc = simd_sum(acc);
        if (tid % 32 == 0) red[tid / 32] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            float s = 0.0f;
            for (uint i = 0; i < NT / 32; i++) s += red[i];
            red[0] = metal::rsqrt(s / D + EPS);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rms = red[0];
        for (uint i = tid; i < D; i += NT) {{
            y_sh[i] = float(({mt})(float(x[base + i]) * rms * float(wn[i])));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- gate 第一 GEMM：y @ W_down → silu（bf16 中间舍入）----
        if (tid < RANK) {{
            float a2 = 0.0f;
            for (uint d = 0; d < D; d++) {{
                a2 += y_sh[d] * float(wd[d * RANK + tid]);
            }}
            float hb = float(({mt})a2);
            float sg = 1.0f / (1.0f + metal::exp(-hb));
            h_sh[tid] = float(({mt})(hb * sg));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 第二 GEMM → sigmoid → ×2 → 与 y 相乘 ----
        for (uint i = tid; i < D; i += NT) {{
            float a3 = 0.0f;
            for (uint j = 0; j < RANK; j++) {{
                a3 += h_sh[j] * float(wu[j * D + i]);
            }}
            float gb = float(({mt})a3);
            float g2 = float(({mt})(2.0f / (1.0f + metal::exp(-gb))));
            out[base + i] = ({mt})(y_sh[i] * g2);
        }}
        """,
    )
    _KERNELS[key] = kern
    return kern


def _eager_ref(x, wn, wd, wu, eps):
    y = mx.fast.rms_norm(x, wn, eps)
    import mlx.nn as nn

    g = 2.0 * mx.sigmoid(nn.silu(y @ wd) @ wu)
    return y * g.astype(y.dtype)


def gated_norm_decode(x, wn, wd, wu, eps: float):
    """x (..., D)；wn (D,)；wd (D, rank)；wu (rank, D)。
    全部同 dtype 且行数 ≤ _MAX_ROWS 时走融合 kernel，否则/失败返回 None。
    """
    global _DISABLED
    if _DISABLED or x.dtype not in _METAL_TYPE:
        return None
    D = x.shape[-1]
    if wn.shape[0] != D:
        return None
    rank = wd.shape[1]
    if wd.shape != (D, rank) or wu.shape != (rank, D):
        return None
    # kernel 结构限制：第一 GEMM 只有 tid < RANK 的线程写 h_sh（rank > NT
    # 会留未初始化段）；threadgroup 内存 y_sh/h_sh/red 不得超过 32KB。
    if rank > _NT or (D + rank + _NT // 32) * 4 > 32768:
        return None
    if not (wn.dtype == wd.dtype == wu.dtype == x.dtype):
        return None
    R = x.size // D
    if R > _MAX_ROWS:
        return None
    key = (D, rank, x.dtype, eps)
    try:
        kern = _build(D, rank, x.dtype, eps)
        (out,) = kern(
            inputs=[x.reshape(R, D), wn, wd, wu],
            output_shapes=[(R, D)],
            output_dtypes=[x.dtype],
            grid=(_NT, R, 1),
            threadgroup=(_NT, 1, 1),
        )
        out = out.reshape(x.shape)
        if key not in _VERIFIED:
            ref = _eager_ref(x, wn, wd, wu, eps)
            mx.eval(out, ref)  # 触发 JIT；失败走 except 永久回退
            d = (out.astype(mx.float32) - ref.astype(mx.float32)).abs().max().item()
            tol = 1e-5 if x.dtype == mx.float32 else 2e-2
            # 不用 d > tol：NaN 经该比较恒为 False 会误过校验
            if not (d <= tol):
                raise RuntimeError(f"gated_norm fused 校验失败 |Δ|={d:.2e} (key={key})")
            _VERIFIED.add(key)
        return out
    except Exception:
        _DISABLED = True
        return None
