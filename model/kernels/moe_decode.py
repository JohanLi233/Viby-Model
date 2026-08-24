# ---------------------------------------------------------------------------
# 融合 Metal kernel：decode/极小批量 MoE 路由专家前向（仅推理，无 autodiff）
# ---------------------------------------------------------------------------
# decode（T=1）时 _dense_forward 的全专家广播 matmul 要读全部 E 个专家权重
# （100M 配置 ~15.3MB/层）且 router+专家每层 ~18 个 kernel 发射；decode 的
# 主瓶颈是 GPU 侧大量小 kernel 的固定调度开销。这里把整个路由 FFN 融合成
# 3 个 kernel，且只读 top-k 命中专家的权重（~2.9MB/层）：
#   moe_router: scores=sigmoid(x·W^T)+expert_bias，lane 分专家算分，
#               threadgroup barrier 后 lane0 串行 top-k，归一化×scaling；
#   moe_up:     h[m,k,i] = SiTU-GLU(x·W_g, x·W_u)  (β1=4, β2=25)
#   moe_down:   out[m,d] = Σ_k w[m,k] · (h[m,k,:] · down_w[e,d,:])
# 内存访问按 simdgroup（32 lane）协作布局：连续 lane 读连续地址（合并访问），
# 点积经 simd_sum 归约；D/I/K 编译期注入为常量（循环可 unroll）。
# 调用约定（mlx 0.32 实测）：
# - 不传 template（host 下发 0.65us vs 3.11us），Metal 类型名直接注入源码；
# - 小输入会被放 constant 地址空间（随尺寸变化），body 内一律直接下标索引，
#   不声明局部 device/constant 指针；
# - JIT 编译是 lazy 的，首次调用后需 mx.eval 触发，失败则整体回退稠密路径。

import mlx.core as mx

_MOE_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_moe_decode_kernel_cache: dict = {}


def _build_moe_decode_kernels(
    D,
    moe_in,
    K,
    E,
    dtype,
    norm_topk,
    scaling,
    logit_norm=False,
    logit_temp=1.0,
    latent_dim=0,
):
    """构建 3 个融合 kernel：router（打分+top-k 选择）、up（SiTU-GLU 前半）、
    down（加权合并）。按 (D,moe_in,K,E,dtype,norm,scaling,logit_norm,
    logit_temp,latent_dim) 缓存。

    latent_dim>0（Latent MoE）时 router kernel 仍在全维 D 上打分，
    up/down kernel 在 latent 维 DE 上计算（专家权重 in/out 维为 DE）；
    投影 GEMM 由调用方在 kernel 外完成。"""
    DE = latent_dim if latent_dim > 0 else D
    key = (
        D,
        moe_in,
        K,
        E,
        dtype,
        norm_topk,
        scaling,
        logit_norm,
        logit_temp,
        latent_dim,
    )
    if key in _moe_decode_kernel_cache:
        return _moe_decode_kernel_cache[key]
    mt = _MOE_METAL_TYPE[dtype]
    norm_code = (
        f"float scale = {scaling}f / metal::max(wsum, 1e-9f);"
        if norm_topk
        else f"float scale = {scaling}f;"
    )
    if logit_norm:
        logit_norm_code = f"""
        float lmean = 0.0f;
        for (uint e = 0; e < {E}; e++) lmean += lg[e];
        lmean /= {E}.0f;
        float lvar = 0.0f;
        for (uint e = 0; e < {E}; e++) {{
            float d = lg[e] - lmean;
            lvar += d * d;
        }}
        lvar /= {E}.0f;
        float lscale = {logit_temp}f / metal::sqrt(lvar + 1e-6f);
        for (uint e = 0; e < {E}; e++) {{
            lg[e] = (lg[e] - lmean) * lscale;
        }}"""
    else:
        logit_norm_code = ""
    # 每 token 一个 simdgroup：lane 分专家算分进 threadgroup 数组，
    # barrier 后 lane0 做 delta clamp、sigmoid 与 K 轮 argmax
    # （E≤1024；选择分连续浮点 tie 概率 0，tie 时与 argpartition 的
    # 选择可能不同——数学上等价的合法 top-k）。
    router_src = f"""
        uint lane = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        threadgroup float sel[{E}];
        threadgroup float scr[{E}];
        threadgroup float lg[{E}];
        size_t xb = (size_t)m * {D};
        for (uint e = lane; e < {E}; e += 32) {{
            size_t wb = (size_t)e * {D};
            float base = 0.0f;
            for (uint d = 0; d < {D}; d++) {{
                base += float(x[xb + d]) * float(weight[wb + d]);
            }}
            lg[e] = base;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {{
            {logit_norm_code}
            for (uint e = 0; e < {E}; e++) {{
                float s = 1.0f / (1.0f + metal::exp(-lg[e]));
                scr[e] = s;
                sel[e] = s + float(bias[e]);
            }}
            float wsum = 0.0f;
            float wk[{K}];
            int ik[{K}];
            for (uint k = 0; k < {K}; k++) {{
                int best = 0;
                float bv = sel[0];
                for (uint e = 1; e < {E}; e++) {{
                    if (sel[e] > bv) {{ bv = sel[e]; best = e; }}
                }}
                sel[best] = -1.0f;
                ik[k] = best;
                wk[k] = scr[best];
                wsum += scr[best];
            }}
            {norm_code}
            for (uint k = 0; k < {K}; k++) {{
                idx[m * {K} + k] = ik[k];
                w[m * {K} + k] = {mt}(wk[k] * scale);
            }}
        }}
    """
    up_src = f"""
        uint lane = thread_position_in_grid.x;
        uint ki = thread_position_in_grid.y;
        uint m = thread_position_in_grid.z;
        uint k = ki / {moe_in};
        uint i = ki % {moe_in};
        int e = idx[m * {K} + k];
        size_t xb = (size_t)m * {DE};
        size_t eb = (size_t)e * {2 * moe_in * DE} + (size_t)i * {DE};
        float g = 0.0f, u = 0.0f;
        for (uint d = lane; d < {DE}; d += 32) {{
            float xv = float(x[xb + d]);
            g += xv * float(gate_up_w[eb + d]);
            u += xv * float(gate_up_w[eb + {moe_in * DE} + d]);
        }}
        g = metal::simd_sum(g);
        u = metal::simd_sum(u);
        if (lane == 0) {{
            float sg = 1.0f / (1.0f + metal::exp(-g));
            float gate = 4.0f * metal::tanh(g / 4.0f) * sg;
            float up = 25.0f * metal::tanh(u / 25.0f);
            h[m * {K * moe_in} + k * {moe_in} + i] = {mt}(gate * up);
        }}
    """
    down_src = f"""
        uint lane = thread_position_in_grid.x;
        uint d = thread_position_in_grid.y;
        uint m = thread_position_in_grid.z;
        float acc = 0.0f;
        for (uint k = 0; k < {K}; k++) {{
            int e = idx[m * {K} + k];
            size_t db = (size_t)e * {DE * moe_in} + (size_t)d * {moe_in};
            size_t hb = (size_t)m * {K * moe_in} + k * {moe_in};
            float inner = 0.0f;
            for (uint i = lane; i < {moe_in}; i += 32) {{
                inner += float(h[hb + i]) * float(down_w[db + i]);
            }}
            acc += float(w[m * {K} + k]) * metal::simd_sum(inner);
        }}
        if (lane == 0) {{
            out[m * {DE} + d] = {mt}(acc);
        }}
    """
    router = mx.fast.metal_kernel(
        name=(
            f"moe_router_{D}_{E}_{K}_"
            f"{int(logit_norm)}_"
            f"{str(logit_temp).replace('.', '_')}_{mt}"
        ),
        input_names=["x", "weight", "bias"],
        output_names=["idx", "w"],
        source=router_src,
    )
    up = mx.fast.metal_kernel(
        name=f"moe_up_{DE}_{moe_in}_{K}_{mt}",
        input_names=["x", "gate_up_w", "idx"],
        output_names=["h"],
        source=up_src,
    )
    down = mx.fast.metal_kernel(
        name=f"moe_down_{DE}_{moe_in}_{K}_{mt}",
        input_names=["h", "down_w", "w", "idx"],
        output_names=["out"],
        source=down_src,
    )
    _moe_decode_kernel_cache[key] = (router, up, down)
    return router, up, down
