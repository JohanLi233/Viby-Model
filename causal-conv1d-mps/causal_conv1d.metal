#include <metal_stdlib>
using namespace metal;

kernel void causal_conv1d_fwd_kernel(
    device const float *input [[buffer(0)]],           // 输入张量 (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],          // 权重 (dim, width)
    device const float *bias [[buffer(2)]],            // 偏置 (dim) - 可为 nullptr
    device float *output [[buffer(3)]],                // 输出张量 (batch, dim, seqlen)
    
    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &silu_activation [[buffer(8)]],      // 是否启用 SiLU 激活
    
    // Strides (以元素为单位)
    constant uint &x_batch_stride [[buffer(9)]],
    constant uint &x_c_stride [[buffer(10)]],
    constant uint &x_l_stride [[buffer(11)]],
    constant uint &weight_c_stride [[buffer(12)]],
    constant uint &weight_width_stride [[buffer(13)]],
    constant uint &out_batch_stride [[buffer(14)]],
    constant uint &out_c_stride [[buffer(15)]],
    constant uint &out_l_stride [[buffer(16)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
)
{
    // 线程组织：每个 threadgroup 处理一个 (batch_id, channel_id) 对
    const uint batch_id = threadgroup_position_in_grid.x;
    const uint channel_id = threadgroup_position_in_grid.y;
    const uint thread_id = thread_position_in_grid.x % threads_per_threadgroup.x;
    
    // 边界检查
    if (batch_id >= batch_size || channel_id >= dim) {
        return;
    }
    
    // 计算数据指针偏移
    device const float *x = input + batch_id * x_batch_stride + channel_id * x_c_stride;
    device const float *w = weight + channel_id * weight_c_stride;
    device float *out = output + batch_id * out_batch_stride + channel_id * out_c_stride;
    
    // 获取偏置值
    float bias_val = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // 预加载权重值
    float weight_vals[4];  // 固定 width=4 的简化版本
    for (uint i = 0; i < width && i < 4; i++) {
        weight_vals[i] = w[i * weight_width_stride];
    }
    
    // 每个线程处理多个序列位置
    const uint elements_per_thread = 4;
    const uint total_threads = threads_per_threadgroup.x;
    const uint chunk_size = total_threads * elements_per_thread;
    const uint num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    
    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        const uint chunk_start = chunk * chunk_size;
        const uint thread_start = chunk_start + thread_id * elements_per_thread;
        
        // 处理当前线程的元素
        for (uint elem = 0; elem < elements_per_thread; elem++) {
            uint pos = thread_start + elem;
            if (pos >= seqlen) break;
            
            float result = bias_val;
            
            // 因果卷积：只使用当前和之前的输入
            for (uint w_idx = 0; w_idx < width; w_idx++) {
                int input_pos = (int)pos - (int)(width - 1 - w_idx);
                if (input_pos >= 0) {
                    float input_val = x[input_pos * x_l_stride];
                    result += weight_vals[w_idx] * input_val;
                }
            }
            
            // 可选的 SiLU 激活函数
            if (silu_activation) {
                result = result / (1.0f + exp(-result));
            }
            
            // 存储结果
            out[pos * out_l_stride] = result;
        }
    }
}

// SiLU 激活函数的辅助函数
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// BF16 <-> FP32 转换辅助函数
inline float bf16_to_float(ushort h) {
    uint u = (uint)h << 16;
    return as_type<float>(u);
}

inline ushort float_to_bf16(float f) {
    uint u = as_type<uint>(f);
    // round-to-nearest-even
    uint lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (ushort)(u >> 16);
}

// 简化版本：固定 width=4，不使用状态管理
kernel void causal_conv1d_simple_kernel(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    
    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    // 每个线程处理一个输出位置
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    // 边界检查
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    // 计算线性索引
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4;  // width=4
    const uint output_idx = input_base + seq_pos;
    
    // 获取偏置
    float result = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // 因果卷积：width=4
    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = input[input_base + input_pos];
            float weight_val = weight[weight_base + w];
            result += weight_val * input_val;
        }
    }
    
    // 可选的 SiLU 激活
    if (silu_activation) {
        result = silu(result);
    }
    
    // 存储结果
    output[output_idx] = result;
}

// 简化版本 (float16)：固定 width=4
kernel void causal_conv1d_simple_kernel_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],

    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],

    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;

    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }

    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4; // width=4
    const uint output_idx = input_base + seq_pos;

    float result = (bias != nullptr) ? (float)bias[channel_id] : 0.0f;

    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = (float)input[input_base + input_pos];
            float weight_val = (float)weight[weight_base + w];
            result += weight_val * input_val;
        }
    }

    if (silu_activation) {
        result = silu(result);
    }

    output[output_idx] = (half)result;
}

// 简化版本 (bfloat16)：固定 width=4
kernel void causal_conv1d_simple_kernel_bf16(
    device const ushort *input [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device ushort *output [[buffer(3)]],

    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant bool &silu_activation [[buffer(7)]],

    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;

    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }

    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4; // width=4
    const uint output_idx = input_base + seq_pos;

    float result = (bias != nullptr) ? bf16_to_float(bias[channel_id]) : 0.0f;

    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = bf16_to_float(input[input_base + input_pos]);
            float weight_val = bf16_to_float(weight[weight_base + w]);
            result += weight_val * input_val;
        }
    }

    if (silu_activation) {
        result = silu(result);
    }

    output[output_idx] = float_to_bf16(result);
}

kernel void short_conv_fused_btd_kernel(
    // Inputs
    device const float *input [[buffer(0)]],      // (B, T, D) 原始输入 (用于残差)
    device const float *weight [[buffer(1)]],     // (D, W=4) 权重
    device const float *bias [[buffer(2)]],       // (D) 偏置 (可选)
    device const float *mask [[buffer(3)]],       // (B, T) Attention mask (可选)
    device float *output [[buffer(4)]],           // (B, T, D) 输出张量

    // 参数
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant bool &use_silu [[buffer(8)]],
    constant bool &use_residual [[buffer(9)]],

    // 线程位置，网格组织为 (B, T, D)
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;

    // 边界检查
    if (b >= B || t >= T || d >= D) return;

    const uint W = 4; // 固定 width=4
    const uint TD = T * D;

    // 线性索引 (BTD 布局步长: T*D, D, 1)
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;

    // 1. 初始化 (读取偏置)
    float result = (bias != nullptr) ? bias[d] : 0.0f;

    // 2. 因果卷积 + 融合 Masking
    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = input[input_idx];

            // 融合 Masking: 在卷积前动态应用 mask (0 或 1)
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= mask[mask_idx];
            }

            float weight_val = weight[weight_base + w];
            result += weight_val * input_val;
        }
    }

    // 3. 可选 SiLU 激活
    if (use_silu) {
        result = silu(result);
    }

    // 4. 残差连接
    if (use_residual) {
        result += input[output_idx];
    }

    // 5. 写回输出
    output[output_idx] = result;
}

// Fused ShortConvolution (float16 版本)
kernel void short_conv_fused_btd_kernel_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device const half *mask [[buffer(3)]],
    device half *output [[buffer(4)]],

    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant bool &use_silu [[buffer(8)]],
    constant bool &use_residual [[buffer(9)]],

    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;

    if (b >= B || t >= T || d >= D) return;

    const uint W = 4;
    const uint TD = T * D;
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;

    float result = (bias != nullptr) ? (float)bias[d] : 0.0f;

    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = (float)input[input_idx];
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= (float)mask[mask_idx];
            }
            float weight_val = (float)weight[weight_base + w];
            result += weight_val * input_val;
        }
    }

    if (use_silu) {
        result = silu(result);
    }
    if (use_residual) {
        result += (float)input[output_idx];
    }
    output[output_idx] = (half)result;
}

// Fused ShortConvolution (bfloat16 版本)
kernel void short_conv_fused_btd_kernel_bf16(
    device const ushort *input [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device const ushort *mask [[buffer(3)]],
    device ushort *output [[buffer(4)]],

    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant bool &use_silu [[buffer(8)]],
    constant bool &use_residual [[buffer(9)]],

    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;

    if (b >= B || t >= T || d >= D) return;

    const uint W = 4;
    const uint TD = T * D;
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;

    float result = (bias != nullptr) ? bf16_to_float(bias[d]) : 0.0f;

    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = bf16_to_float(input[input_idx]);
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= bf16_to_float(mask[mask_idx]);
            }
            float weight_val = bf16_to_float(weight[weight_base + w]);
            result += weight_val * input_val;
        }
    }

    if (use_silu) {
        result = silu(result);
    }
    if (use_residual) {
        result += bf16_to_float(input[output_idx]);
    }
    output[output_idx] = float_to_bf16(result);
}

// ====================================================================================
// Single-token Update Kernels (for efficient inference)
// ====================================================================================

kernel void short_conv_update_kernel(
    device const float *x [[buffer(0)]],                // 单步输入 (B, D) - 新的 token
    device float *conv_state [[buffer(1)]],             // 卷积状态 (B, D, STATE_LEN) - 就地更新
    device const float *weight [[buffer(2)]],           // 权重 (D, W)
    device const float *bias [[buffer(3)]],             // 偏置 (D) - 可选
    device const int *cache_seqlens [[buffer(4)]],      // 各 batch 的当前序列长度 (B,)
    device float *output [[buffer(5)]],                 // 单步输出 (B, D)
    
    constant uint &B [[buffer(6)]],                     // batch_size
    constant uint &D [[buffer(7)]],                     // hidden_dim
    constant uint &W [[buffer(8)]],                     // kernel_width (固定为4)
    constant uint &STATE_LEN [[buffer(9)]],             // 状态缓冲区长度
    constant bool &use_silu [[buffer(10)]],
    constant bool &use_residual [[buffer(11)]],
    
    uint2 gid [[thread_position_in_grid]]               // (B, D)
)
{
    const uint b = gid.x;  // batch index
    const uint d = gid.y;  // dimension index
    
    // 边界检查
    if (b >= B || d >= D) return;
    
    // 获取当前序列长度
    int current_seq_len = cache_seqlens[b];
    
    // 计算在循环缓冲区中的写入位置
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    // 计算线性索引
    const uint x_idx = b * D + d;
    const uint output_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    // 读取当前输入
    float current_input = x[x_idx];
    
    // 初始化结果为偏置
    float result = (bias != nullptr) ? bias[d] : 0.0f;
    
    // 执行因果卷积：需要读取过去 W-1 个状态 + 当前输入
    for (uint w = 0; w < W; w++) {
        float input_val;
        
        if (w == W - 1) {
            // 最后一个权重对应当前输入
            input_val = current_input;
        } else {
            // 从循环缓冲区读取历史数据
            // 位置计算：(write_pos - (W - 1 - w)) % STATE_LEN
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = conv_state[state_idx];
        }
        
        float weight_val = weight[weight_base + w];
        result += weight_val * input_val;
    }
    
    // 应用激活函数
    if (use_silu) {
        result = silu(result);
    }
    
    // 应用残差连接
    if (use_residual) {
        result += current_input;
    }
    
    // 更新状态：将当前输入写入循环缓冲区
    uint state_write_idx = state_base + write_pos;
    conv_state[state_write_idx] = current_input;
    
    // 写入输出
    output[output_idx] = result;
}

// Float16 版本
kernel void short_conv_update_kernel_f16(
    device const half *x [[buffer(0)]],
    device half *conv_state [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device const int *cache_seqlens [[buffer(4)]],
    device half *output [[buffer(5)]],
    
    constant uint &B [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant uint &W [[buffer(8)]],
    constant uint &STATE_LEN [[buffer(9)]],
    constant bool &use_silu [[buffer(10)]],
    constant bool &use_residual [[buffer(11)]],
    
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint d = gid.y;
    
    if (b >= B || d >= D) return;
    
    int current_seq_len = cache_seqlens[b];
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    const uint x_idx = b * D + d;
    const uint output_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    float current_input = (float)x[x_idx];
    float result = (bias != nullptr) ? (float)bias[d] : 0.0f;
    
    for (uint w = 0; w < W; w++) {
        float input_val;
        
        if (w == W - 1) {
            input_val = current_input;
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = (float)conv_state[state_idx];
        }
        
        float weight_val = (float)weight[weight_base + w];
        result += weight_val * input_val;
    }
    
    if (use_silu) {
        result = silu(result);
    }
    
    if (use_residual) {
        result += current_input;
    }
    
    uint state_write_idx = state_base + write_pos;
    conv_state[state_write_idx] = (half)current_input;
    
    output[output_idx] = (half)result;
}

// BFloat16 版本  
kernel void short_conv_update_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device ushort *conv_state [[buffer(1)]],
    device const ushort *weight [[buffer(2)]],
    device const ushort *bias [[buffer(3)]],
    device const int *cache_seqlens [[buffer(4)]],
    device ushort *output [[buffer(5)]],
    
    constant uint &B [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant uint &W [[buffer(8)]],
    constant uint &STATE_LEN [[buffer(9)]],
    constant bool &use_silu [[buffer(10)]],
    constant bool &use_residual [[buffer(11)]],
    
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint d = gid.y;
    
    if (b >= B || d >= D) return;
    
    int current_seq_len = cache_seqlens[b];
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    const uint x_idx = b * D + d;
    const uint output_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    float current_input = bf16_to_float(x[x_idx]);
    float result = (bias != nullptr) ? bf16_to_float(bias[d]) : 0.0f;
    
    for (uint w = 0; w < W; w++) {
        float input_val;
        
        if (w == W - 1) {
            input_val = current_input;
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = bf16_to_float(conv_state[state_idx]);
        }
        
        float weight_val = bf16_to_float(weight[weight_base + w]);
        result += weight_val * input_val;
    }
    
    if (use_silu) {
        result = silu(result);
    }
    
    if (use_residual) {
        result += current_input;
    }
    
    uint state_write_idx = state_base + write_pos;
    conv_state[state_write_idx] = float_to_bf16(current_input);
    
    output[output_idx] = float_to_bf16(result);
}

// =====================================================================================
// BACKWARD PASS KERNELS
// =====================================================================================

// Backward pass for basic causal conv1d (float32)
kernel void causal_conv1d_bwd_kernel(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    device const float *grad_output [[buffer(2)]], // Gradient w.r.t. output (batch, dim, seqlen)
    device float *grad_x [[buffer(3)]],            // Gradient w.r.t. input (batch, dim, seqlen)
    device float *grad_weight [[buffer(4)]],       // Gradient w.r.t. weight (dim, width) - accumulated
    device const float *bias [[buffer(5)]],        // Forward bias (dim) - may be nullptr
    device float *grad_bias [[buffer(6)]],         // Gradient w.r.t. bias (dim) - accumulated (optional)
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4; // Fixed width
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    // Get gradient from output
    float grad_out = grad_output[output_idx];
    
    // If SiLU activation was used, we need to apply its derivative
    if (silu_activation) {
        // Recompute forward pass to get the pre-activation value
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                float weight_val = weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        // Add bias contribution if available
        if (bias != nullptr) {
            pre_activation += bias[channel_id];
        }
        
        // Apply SiLU derivative: d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute gradient w.r.t. input (dx)
    // For causal conv: dx[t] = sum_w(weight[w] * grad_output[t + (width - 1 - w)])
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = grad_output[input_base + future_pos];
            
            // Apply activation derivative if needed for future position
            if (silu_activation) {
                // Recompute forward for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = x[input_base + future_input_pos];
                        float future_weight_val = weight[weight_base + fw];
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                // Add bias contribution if available
                if (bias != nullptr) {
                    future_pre_activation += bias[channel_id];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            
            dx_val += weight[weight_base + w] * future_grad;
        }
    }
    grad_x[output_idx] = dx_val;
    
    // Accumulate gradient w.r.t. weight (dweight) - using atomic operations
    // dweight[w] += x[t-(width-1-w)] * grad_output[t]
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = x[input_base + input_pos];
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    // Accumulate gradient w.r.t. bias (dbias) - using atomic operations
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for basic causal conv1d (float16)
kernel void causal_conv1d_bwd_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *grad_output [[buffer(2)]],
    device half *grad_x [[buffer(3)]],
    device float *grad_weight [[buffer(4)]],     // Use float32 for accumulation
    device const half *bias [[buffer(5)]],       // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(6)]],       // Use float32 for accumulation
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = (float)grad_output[output_idx];
    
    // Apply SiLU derivative if needed
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = (float)x[input_base + input_pos];
                float weight_val = (float)weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += (float)bias[channel_id];
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute dx
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = (float)grad_output[input_base + future_pos];
            if (silu_activation) {
                // Apply activation derivative for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = (float)x[input_base + future_input_pos];
                        float future_weight_val = (float)weight[weight_base + fw];
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_pre_activation += (float)bias[channel_id];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            dx_val += (float)weight[weight_base + w] * future_grad;
        }
    }
    grad_x[output_idx] = (half)dx_val;
    
    // Accumulate dweight and dbias in float32
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = (float)x[input_base + input_pos];
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for basic causal conv1d (bfloat16)
kernel void causal_conv1d_bwd_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *grad_output [[buffer(2)]],
    device ushort *grad_x [[buffer(3)]],
    device float *grad_weight [[buffer(4)]],     // Use float32 for accumulation
    device const ushort *bias [[buffer(5)]],     // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(6)]],       // Use float32 for accumulation
    
    constant uint &batch_size [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seqlen [[buffer(9)]],
    constant bool &silu_activation [[buffer(10)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]]
)
{
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    const uint width = 4;
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * width;
    const uint output_idx = input_base + seq_pos;
    
    float grad_out = bf16_to_float(grad_output[output_idx]);
    
    // Apply SiLU derivative if needed
    if (silu_activation) {
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = bf16_to_float(x[input_base + input_pos]);
                float weight_val = bf16_to_float(weight[weight_base + w]);
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += bf16_to_float(bias[channel_id]);
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    // Compute dx
    float dx_val = 0.0f;
    for (uint w = 0; w < width; w++) {
        int future_pos = (int)seq_pos + (int)(width - 1 - w);
        if (future_pos < (int)seqlen) {
            float future_grad = bf16_to_float(grad_output[input_base + future_pos]);
            if (silu_activation) {
                // Apply activation derivative for future position
                float future_pre_activation = 0.0f;
                for (uint fw = 0; fw < width; fw++) {
                    int future_input_pos = future_pos - (int)(width - 1 - fw);
                    if (future_input_pos >= 0) {
                        float future_input_val = bf16_to_float(x[input_base + future_input_pos]);
                        float future_weight_val = bf16_to_float(weight[weight_base + fw]);
                        future_pre_activation += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_pre_activation += bf16_to_float(bias[channel_id]);
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_pre_activation));
                future_grad *= future_sigmoid * (1.0f + future_pre_activation * (1.0f - future_sigmoid));
            }
            dx_val += bf16_to_float(weight[weight_base + w]) * future_grad;
        }
    }
    grad_x[output_idx] = float_to_bf16(dx_val);
    
    // Accumulate dweight and dbias in float32
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = bf16_to_float(x[input_base + input_pos]);
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[channel_id],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for fused short convolution (float32)
kernel void short_conv_fused_btd_bwd_kernel(
    device const float *x [[buffer(0)]],           // Original input (B, T, D)
    device const float *weight [[buffer(1)]],      // Original weight (D, W)
    device const float *mask [[buffer(2)]],        // Attention mask (B, T)
    device const float *grad_output [[buffer(3)]], // Gradient w.r.t. output (B, T, D)
    device float *grad_x [[buffer(4)]],            // Gradient w.r.t. input (B, T, D)
    device float *grad_weight [[buffer(5)]],       // Gradient w.r.t. weight (D, W) - accumulated
    device const float *bias [[buffer(6)]],        // Forward bias (D) may be nullptr
    device float *grad_bias [[buffer(7)]],         // Gradient w.r.t. bias (D) - accumulated
    
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant bool &use_activation [[buffer(11)]],
    constant bool &use_residual [[buffer(12)]],
    
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;
    
    if (b >= B || t >= T || d >= D) return;
    
    const uint W = 4;
    const uint TD = T * D;
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;
    
    float grad_out = grad_output[output_idx];
    
    // If residual was used, part of gradient flows directly to input
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Compute forward pass result for activation derivative
    float conv_result = 0.0f;
    if (use_activation) {
        for (uint w = 0; w < W; w++) {
            int tt = (int)t - (int)(W - 1 - w);
            if (tt >= 0) {
                const uint input_idx = b * TD + (uint)tt * D + d;
                float input_val = x[input_idx];
                if (mask != nullptr) {
                    const uint mask_idx = b * T + (uint)tt;
                    input_val *= mask[mask_idx];
                }
                float weight_val = weight[weight_base + w];
                conv_result += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            conv_result += bias[d];
        }
        // Apply SiLU derivative
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute gradient w.r.t. input from convolution
    float dx_conv = 0.0f;
    for (uint w = 0; w < W; w++) {
        int future_t = (int)t + (int)(W - 1 - w);
        if (future_t < (int)T) {
            const uint future_idx = b * TD + (uint)future_t * D + d;
            float future_grad = grad_output[future_idx];
            
            // Apply activation derivative for future position if needed
            if (use_activation) {
                // Recompute forward for future position
                float future_conv_result = 0.0f;
                for (uint fw = 0; fw < W; fw++) {
                    int future_input_t = future_t - (int)(W - 1 - fw);
                    if (future_input_t >= 0) {
                        const uint future_input_idx = b * TD + (uint)future_input_t * D + d;
                        float future_input_val = x[future_input_idx];
                        if (mask != nullptr) {
                            const uint future_mask_idx = b * T + (uint)future_input_t;
                            future_input_val *= mask[future_mask_idx];
                        }
                        float future_weight_val = weight[weight_base + fw];
                        future_conv_result += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_conv_result += bias[d];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_conv_result));
                future_grad *= future_sigmoid * (1.0f + future_conv_result * (1.0f - future_sigmoid));
            }
            
            // Apply mask to weight gradient
            float weight_val = weight[weight_base + w];
            if (mask != nullptr) {
                const uint mask_idx = b * T + t;
                weight_val *= mask[mask_idx];
            }
            dx_conv += weight_val * future_grad;
        }
    }
    
    grad_x[output_idx] = dx_residual + dx_conv;
    
    // Accumulate gradient w.r.t. weight
    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = x[input_idx];
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= mask[mask_idx];
            }
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    // Accumulate gradient w.r.t. bias
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for fused short convolution (float16)
kernel void short_conv_fused_btd_bwd_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *mask [[buffer(2)]],
    device const half *grad_output [[buffer(3)]],
    device half *grad_x [[buffer(4)]],
    device float *grad_weight [[buffer(5)]],       // Use float32 for accumulation
    device const half *bias [[buffer(6)]],         // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(7)]],         // Use float32 for accumulation
    
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant bool &use_activation [[buffer(11)]],
    constant bool &use_residual [[buffer(12)]],
    
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;
    
    if (b >= B || t >= T || d >= D) return;
    
    const uint W = 4;
    const uint TD = T * D;
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;
    
    float grad_out = (float)grad_output[output_idx];
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Apply activation derivative if needed
    if (use_activation) {
        float conv_result = 0.0f;
        for (uint w = 0; w < W; w++) {
            int tt = (int)t - (int)(W - 1 - w);
            if (tt >= 0) {
                const uint input_idx = b * TD + (uint)tt * D + d;
                float input_val = (float)x[input_idx];
                if (mask != nullptr) {
                    const uint mask_idx = b * T + (uint)tt;
                    input_val *= (float)mask[mask_idx];
                }
                float weight_val = (float)weight[weight_base + w];
                conv_result += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            conv_result += (float)bias[d];
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute dx from convolution
    float dx_conv = 0.0f;
    for (uint w = 0; w < W; w++) {
        int future_t = (int)t + (int)(W - 1 - w);
        if (future_t < (int)T) {
            const uint future_idx = b * TD + (uint)future_t * D + d;
            float future_grad = (float)grad_output[future_idx];
            
            if (use_activation) {
                // Apply activation derivative for future position
                float future_conv_result = 0.0f;
                for (uint fw = 0; fw < W; fw++) {
                    int future_input_t = future_t - (int)(W - 1 - fw);
                    if (future_input_t >= 0) {
                        const uint future_input_idx = b * TD + (uint)future_input_t * D + d;
                        float future_input_val = (float)x[future_input_idx];
                        if (mask != nullptr) {
                            const uint future_mask_idx = b * T + (uint)future_input_t;
                            future_input_val *= (float)mask[future_mask_idx];
                        }
                        float future_weight_val = (float)weight[weight_base + fw];
                        future_conv_result += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_conv_result += (float)bias[d];
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_conv_result));
                future_grad *= future_sigmoid * (1.0f + future_conv_result * (1.0f - future_sigmoid));
            }
            
            float weight_val = (float)weight[weight_base + w];
            if (mask != nullptr) {
                const uint mask_idx = b * T + t;
                weight_val *= (float)mask[mask_idx];
            }
            dx_conv += weight_val * future_grad;
        }
    }
    
    grad_x[output_idx] = (half)(dx_residual + dx_conv);
    
    // Accumulate gradients in float32
    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = (float)x[input_idx];
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= (float)mask[mask_idx];
            }
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for fused short convolution (bfloat16)
kernel void short_conv_fused_btd_bwd_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *mask [[buffer(2)]],
    device const ushort *grad_output [[buffer(3)]],
    device ushort *grad_x [[buffer(4)]],
    device float *grad_weight [[buffer(5)]],       // Use float32 for accumulation
    device const ushort *bias [[buffer(6)]],       // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(7)]],         // Use float32 for accumulation
    
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant bool &use_activation [[buffer(11)]],
    constant bool &use_residual [[buffer(12)]],
    
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;
    
    if (b >= B || t >= T || d >= D) return;
    
    const uint W = 4;
    const uint TD = T * D;
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;
    
    float grad_out = bf16_to_float(grad_output[output_idx]);
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Apply activation derivative if needed
    if (use_activation) {
        float conv_result = 0.0f;
        for (uint w = 0; w < W; w++) {
            int tt = (int)t - (int)(W - 1 - w);
            if (tt >= 0) {
                const uint input_idx = b * TD + (uint)tt * D + d;
                float input_val = bf16_to_float(x[input_idx]);
                if (mask != nullptr) {
                    const uint mask_idx = b * T + (uint)tt;
                    input_val *= bf16_to_float(mask[mask_idx]);
                }
                float weight_val = bf16_to_float(weight[weight_base + w]);
                conv_result += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            conv_result += bf16_to_float(bias[d]);
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute dx from convolution
    float dx_conv = 0.0f;
    for (uint w = 0; w < W; w++) {
        int future_t = (int)t + (int)(W - 1 - w);
        if (future_t < (int)T) {
            const uint future_idx = b * TD + (uint)future_t * D + d;
            float future_grad = bf16_to_float(grad_output[future_idx]);
            
            if (use_activation) {
                // Apply activation derivative for future position
                float future_conv_result = 0.0f;
                for (uint fw = 0; fw < W; fw++) {
                    int future_input_t = future_t - (int)(W - 1 - fw);
                    if (future_input_t >= 0) {
                        const uint future_input_idx = b * TD + (uint)future_input_t * D + d;
                        float future_input_val = bf16_to_float(x[future_input_idx]);
                        if (mask != nullptr) {
                            const uint future_mask_idx = b * T + (uint)future_input_t;
                            future_input_val *= bf16_to_float(mask[future_mask_idx]);
                        }
                        float future_weight_val = bf16_to_float(weight[weight_base + fw]);
                        future_conv_result += future_weight_val * future_input_val;
                    }
                }
                if (bias != nullptr) {
                    future_conv_result += bf16_to_float(bias[d]);
                }
                float future_sigmoid = 1.0f / (1.0f + exp(-future_conv_result));
                future_grad *= future_sigmoid * (1.0f + future_conv_result * (1.0f - future_sigmoid));
            }
            
            float weight_val = bf16_to_float(weight[weight_base + w]);
            if (mask != nullptr) {
                const uint mask_idx = b * T + t;
                weight_val *= bf16_to_float(mask[mask_idx]);
            }
            dx_conv += weight_val * future_grad;
        }
    }
    
    grad_x[output_idx] = float_to_bf16(dx_residual + dx_conv);
    
    // Accumulate gradients in float32
    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = bf16_to_float(x[input_idx]);
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= bf16_to_float(mask[mask_idx]);
            }
            atomic_fetch_add_explicit(
                (device atomic<float>*)&grad_weight[weight_base + w],
                input_val * grad_out,
                memory_order_relaxed
            );
        }
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for convolution update (float32)
kernel void short_conv_update_bwd_kernel(
    device const float *x [[buffer(0)]],           // Original input (B, D)
    device const float *conv_state [[buffer(1)]],  // Original conv state (B, D, STATE_LEN)
    device const float *weight [[buffer(2)]],      // Original weight (D, W)
    device const int *cache_seqlens [[buffer(3)]], // Sequence lengths (B)
    device const float *grad_output [[buffer(4)]], // Gradient w.r.t. output (B, D)
    device float *grad_x [[buffer(5)]],            // Gradient w.r.t. input (B, D)
    device float *grad_conv_state [[buffer(6)]],   // Gradient w.r.t. conv state (B, D, STATE_LEN)
    device float *grad_weight [[buffer(7)]],       // Gradient w.r.t. weight (D, W) - accumulated
    device const float *bias [[buffer(8)]],        // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(9)]],         // Gradient w.r.t. bias (D) - accumulated
    
    constant uint &B [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant uint &W [[buffer(12)]],
    constant uint &STATE_LEN [[buffer(13)]],
    constant bool &use_activation [[buffer(14)]],
    constant bool &use_residual [[buffer(15)]],
    
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint d = gid.y;
    
    if (b >= B || d >= D) return;
    
    const uint x_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    int current_seq_len = cache_seqlens[b];
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    float current_input = x[x_idx];
    float grad_out = grad_output[x_idx];
    
    // If residual was used, part of gradient flows directly to input
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Compute forward result for activation derivative
    float conv_result = 0.0f;
    if (use_activation) {
        for (uint w = 0; w < W; w++) {
            float input_val;
            if (w == W - 1) {
                input_val = current_input;
            } else {
                int hist_offset = (int)(W - 1 - w);
                int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
                uint state_idx = state_base + (uint)hist_pos;
                input_val = conv_state[state_idx];
            }
            float weight_val = weight[weight_base + w];
            conv_result += weight_val * input_val;
        }
        if (bias != nullptr) {
            conv_result += bias[d];
        }
        // Apply SiLU derivative
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute gradient w.r.t. input and conv_state
    for (uint w = 0; w < W; w++) {
        float weight_val = weight[weight_base + w];
        
        if (w == W - 1) {
            // Gradient w.r.t. current input
            grad_x[x_idx] = dx_residual + weight_val * grad_out;
        } else {
            // Gradient w.r.t. conv_state
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            grad_conv_state[state_idx] = weight_val * grad_out;
        }
    }
    
    // Accumulate gradient w.r.t. weight
    for (uint w = 0; w < W; w++) {
        float input_val;
        if (w == W - 1) {
            input_val = current_input;
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = conv_state[state_idx];
        }
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_weight[weight_base + w],
            input_val * grad_out,
            memory_order_relaxed
        );
    }
    
    // Accumulate gradient w.r.t. bias
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for convolution update (float16)
kernel void short_conv_update_bwd_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *conv_state [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device const int *cache_seqlens [[buffer(3)]],
    device const half *grad_output [[buffer(4)]],
    device half *grad_x [[buffer(5)]],
    device half *grad_conv_state [[buffer(6)]],
    device float *grad_weight [[buffer(7)]],       // Use float32 for accumulation
    device const half *bias [[buffer(8)]],         // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(9)]],         // Use float32 for accumulation
    
    constant uint &B [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant uint &W [[buffer(12)]],
    constant uint &STATE_LEN [[buffer(13)]],
    constant bool &use_activation [[buffer(14)]],
    constant bool &use_residual [[buffer(15)]],
    
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint d = gid.y;
    
    if (b >= B || d >= D) return;
    
    const uint x_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    int current_seq_len = cache_seqlens[b];
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    float current_input = (float)x[x_idx];
    float grad_out = (float)grad_output[x_idx];
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Apply activation derivative if needed
    if (use_activation) {
        float conv_result = 0.0f;
        for (uint w = 0; w < W; w++) {
            float input_val;
            if (w == W - 1) {
                input_val = current_input;
            } else {
                int hist_offset = (int)(W - 1 - w);
                int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
                uint state_idx = state_base + (uint)hist_pos;
                input_val = (float)conv_state[state_idx];
            }
            float weight_val = (float)weight[weight_base + w];
            conv_result += weight_val * input_val;
        }
        if (bias != nullptr) {
            conv_result += (float)bias[d];
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute gradients
    for (uint w = 0; w < W; w++) {
        float weight_val = (float)weight[weight_base + w];
        
        if (w == W - 1) {
            grad_x[x_idx] = (half)(dx_residual + weight_val * grad_out);
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            grad_conv_state[state_idx] = (half)(weight_val * grad_out);
        }
    }
    
    // Accumulate weight and bias gradients in float32
    for (uint w = 0; w < W; w++) {
        float input_val;
        if (w == W - 1) {
            input_val = current_input;
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = (float)conv_state[state_idx];
        }
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_weight[weight_base + w],
            input_val * grad_out,
            memory_order_relaxed
        );
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}

// Backward pass for convolution update (bfloat16)
kernel void short_conv_update_bwd_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *conv_state [[buffer(1)]],
    device const ushort *weight [[buffer(2)]],
    device const int *cache_seqlens [[buffer(3)]],
    device const ushort *grad_output [[buffer(4)]],
    device ushort *grad_x [[buffer(5)]],
    device ushort *grad_conv_state [[buffer(6)]],
    device float *grad_weight [[buffer(7)]],       // Use float32 for accumulation
    device const ushort *bias [[buffer(8)]],       // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(9)]],         // Use float32 for accumulation
    
    constant uint &B [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant uint &W [[buffer(12)]],
    constant uint &STATE_LEN [[buffer(13)]],
    constant bool &use_activation [[buffer(14)]],
    constant bool &use_residual [[buffer(15)]],
    
    uint2 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint d = gid.y;
    
    if (b >= B || d >= D) return;
    
    const uint x_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    int current_seq_len = cache_seqlens[b];
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    float current_input = bf16_to_float(x[x_idx]);
    float grad_out = bf16_to_float(grad_output[x_idx]);
    float dx_residual = use_residual ? grad_out : 0.0f;
    
    // Apply activation derivative if needed
    if (use_activation) {
        float conv_result = 0.0f;
        for (uint w = 0; w < W; w++) {
            float input_val;
            if (w == W - 1) {
                input_val = current_input;
            } else {
                int hist_offset = (int)(W - 1 - w);
                int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
                uint state_idx = state_base + (uint)hist_pos;
                input_val = bf16_to_float(conv_state[state_idx]);
            }
            float weight_val = bf16_to_float(weight[weight_base + w]);
            conv_result += weight_val * input_val;
        }
        if (bias != nullptr) {
            conv_result += bf16_to_float(bias[d]);
        }
        float sigmoid_val = 1.0f / (1.0f + exp(-conv_result));
        grad_out *= sigmoid_val * (1.0f + conv_result * (1.0f - sigmoid_val));
    }
    
    // Compute gradients
    for (uint w = 0; w < W; w++) {
        float weight_val = bf16_to_float(weight[weight_base + w]);
        
        if (w == W - 1) {
            grad_x[x_idx] = float_to_bf16(dx_residual + weight_val * grad_out);
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            grad_conv_state[state_idx] = float_to_bf16(weight_val * grad_out);
        }
    }
    
    // Accumulate weight and bias gradients in float32
    for (uint w = 0; w < W; w++) {
        float input_val;
        if (w == W - 1) {
            input_val = current_input;
        } else {
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = bf16_to_float(conv_state[state_idx]);
        }
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_weight[weight_base + w],
            input_val * grad_out,
            memory_order_relaxed
        );
    }
    
    if (grad_bias != nullptr) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_bias[d],
            grad_out,
            memory_order_relaxed
        );
    }
}