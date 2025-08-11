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