#include <metal_stdlib>
using namespace metal;

kernel void causal_conv1d_fwd_kernel(
    device const float *input [[buffer(0)]],            // input tensor (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],           // weights (dim, width)
    device const float *bias [[buffer(2)]],             // bias (dim) - can be nullptr
    device float *output [[buffer(3)]],                 // output tensor (batch, dim, seqlen)
    
    constant uint &batch_size [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &seqlen [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &silu_activation [[buffer(8)]],       // whether to use SiLU activation
    
    // Strides (in elements)
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
    // Thread organization: each threadgroup handles a (batch_id, channel_id) pair
    const uint batch_id = threadgroup_position_in_grid.x;
    const uint channel_id = threadgroup_position_in_grid.y;
    const uint thread_id = thread_position_in_grid.x % threads_per_threadgroup.x;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim) {
        return;
    }
    
    // Calculate data pointer offsets
    device const float *x = input + batch_id * x_batch_stride + channel_id * x_c_stride;
    device const float *w = weight + channel_id * weight_c_stride;
    device float *out = output + batch_id * out_batch_stride + channel_id * out_c_stride;
    
    // Get bias value
    float bias_val = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // Preload weight values
    float weight_vals[4];  // fixed width=4 simplified version
    // Use min to ensure no out-of-bounds reads
    uint effective_width = min(width, (uint)4);
    for (uint i = 0; i < effective_width; i++) {
        weight_vals[i] = w[i * weight_width_stride];
    }
    
    // Each thread handles multiple sequence positions
    const uint elements_per_thread = 4;
    const uint total_threads = threads_per_threadgroup.x;
    const uint chunk_size = total_threads * elements_per_thread;
    const uint num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    
    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        const uint chunk_start = chunk * chunk_size;
        const uint thread_start = chunk_start + thread_id * elements_per_thread;
        
        // Process elements for the current thread
        for (uint elem = 0; elem < elements_per_thread; elem++) {
            uint pos = thread_start + elem;
            if (pos >= seqlen) break;
            
            float result = bias_val;
            
            // Causal convolution: only use current and previous inputs (Convention B)
            for (uint w_idx = 0; w_idx < effective_width; w_idx++) {
                int input_pos = (int)pos - (int)(effective_width - 1 - w_idx);
                if (input_pos >= 0) {
                    float input_val = x[input_pos * x_l_stride];
                    result += weight_vals[w_idx] * input_val;
                }
            }
            
            // Optional SiLU activation function
            if (silu_activation) {
                result = result / (1.0f + exp(-result));
            }
            
            // Store result
            out[pos * out_l_stride] = result;
        }
    }
}

// SiLU activation function helper
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// BF16 <-> FP32 conversion helper
inline float bf16_to_float(ushort h) {
    uint u = (uint)h << 16;
    return as_type<float>(u);
}

inline ushort float_to_bf16(float f) {
    uint u = as_type<uint>(f);
    
    // Extract sign bit and move to BF16 position (bit 15)
    ushort sign_bit = (ushort)((u >> 16) & 0x8000);

    // Add robustness: handle NaN/Inf
    // Check if exponent is all 1 (0xFF)
    if ((u & 0x7F800000) == 0x7F800000) {
        if ((u & 0x007FFFFF) != 0) {
            // NaN (mantissa non-zero) - use Quiet NaN (0x7FC0)
            return sign_bit | 0x7FC0;
        } else {
            // Infinity (mantissa all zero)
            return sign_bit | 0x7F80;
        }
    }

    // Round-to-nearest-even (RNE) for finite numbers
    uint lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (ushort)(u >> 16);
}

// Simplified version: fixed width=4, no state management
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
    // Each thread handles one output position
    const uint batch_id = thread_position_in_grid.x;
    const uint channel_id = thread_position_in_grid.y;
    const uint seq_pos = thread_position_in_grid.z;
    
    // Boundary check
    if (batch_id >= batch_size || channel_id >= dim || seq_pos >= seqlen) {
        return;
    }
    
    // Calculate linear index
    const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
    const uint weight_base = channel_id * 4;  // width=4
    const uint output_idx = input_base + seq_pos;
    
    // Get bias
    float result = (bias != nullptr) ? bias[channel_id] : 0.0f;
    
    // Causal convolution: width=4
    const uint width = 4;
    for (uint w = 0; w < width; w++) {
        int input_pos = (int)seq_pos - (int)(width - 1 - w);
        if (input_pos >= 0) {
            float input_val = input[input_base + input_pos];
            float weight_val = weight[weight_base + w];
            result += weight_val * input_val;
        }
    }
    
    // Optional SiLU activation
    if (silu_activation) {
        result = silu(result);
    }
    
    // Store result
    output[output_idx] = result;
}

// Simplified version (float16): fixed width=4
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

// Simplified version (bfloat16): fixed width=4
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
    device const float *input [[buffer(0)]],     // (B, T, D) original input (for residual)
    device const float *weight [[buffer(1)]],    // (D, W=4) weights
    device const float *bias [[buffer(2)]],      // (D) bias (optional)
    device const float *mask [[buffer(3)]],      // (B, T) Attention mask (optional)
    device float *output [[buffer(4)]],          // (B, T, D) output tensor

    // Parameters
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant bool &use_silu [[buffer(8)]],
    constant bool &use_residual [[buffer(9)]],

    // Thread position, grid organized as (B, T, D)
    uint3 gid [[thread_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;

    // Boundary check
    if (b >= B || t >= T || d >= D) return;

    const uint W = 4; // fixed width=4
    const uint TD = T * D;

    // Linear index (BTD layout stride: T*D, D, 1)
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;

    // 1. Initialize (read bias)
    float result = (bias != nullptr) ? bias[d] : 0.0f;

    // 2. Causal convolution + fused Masking
    for (uint w = 0; w < W; w++) {
        int tt = (int)t - (int)(W - 1 - w);
        if (tt >= 0) {
            const uint input_idx = b * TD + (uint)tt * D + d;
            float input_val = input[input_idx];

            // Fused Masking: apply mask (0 or 1) dynamically before convolution
            if (mask != nullptr) {
                const uint mask_idx = b * T + (uint)tt;
                input_val *= mask[mask_idx];
            }

            float weight_val = weight[weight_base + w];
            result += weight_val * input_val;
        }
    }

    // 3. Optional SiLU activation
    if (use_silu) {
        result = silu(result);
    }

    // 4. Residual connection
    if (use_residual) {
        result += input[output_idx];
    }

    // 5. Write back output
    output[output_idx] = result;
}

// Threadgroup memory tiled version of short_conv_fused_btd_kernel (float32)
// **OPTIMIZED**: Simplified indexing and removed global memory fallback.
kernel void short_conv_fused_btd_kernel_tiled(
    // Inputs  
    device const float *input [[buffer(0)]],    // (B, T, D) original input
    device const float *weight [[buffer(1)]],   // (D, W=4) weights
    device const float *bias [[buffer(2)]],     // (D) bias (optional)
    device const float *mask [[buffer(3)]],     // (B, T) Attention mask (optional)
    device float *output [[buffer(4)]],         // (B, T, D) output tensor

    // Parameters
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant bool &use_silu [[buffer(8)]],
    constant bool &use_residual [[buffer(9)]],

    // Thread organization and position information
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
)
{
    const uint b = gid.x;
    const uint t = gid.y;
    const uint d = gid.z;
    
    const uint W = 4; // fixed width=4
    const uint TD = T * D;
    
    // Threadgroup dimensions and tiling parameters
    const uint tile_t_size = threads_per_threadgroup.y;
    const uint tile_d_size = threads_per_threadgroup.z;
    
    // Threadgroup memory for input tiles (include W-1 padding)
    // NOTE: Assuming max tile size 32x32. This must match the host configuration.
    threadgroup float shared_input[32 * 32 + 3 * 32]; // Tile + W-1=3 padding rows
    threadgroup float shared_mask[32 * 32 + 3 * 32];  // Mask tile
    
    // Thread position within threadgroup
    const uint local_t = tid.y;
    const uint local_d = tid.z;
    
    // Threadgroup position in the grid
    const uint tg_t_start = threadgroup_position_in_grid.y * tile_t_size;
    const uint tg_d_start = threadgroup_position_in_grid.z * tile_d_size;
    
    // --- 1. Collaborative Loading ---
    // (Loading logic remains the same as it correctly loads the padded window)
    const uint total_threads_in_group = tile_t_size * tile_d_size;
    const uint thread_id_in_group = local_t * tile_d_size + local_d;
    const uint padded_tile_size = (tile_t_size + W - 1) * tile_d_size;
    const uint elements_per_thread = (padded_tile_size + total_threads_in_group - 1) / total_threads_in_group;
    
    for (uint elem = 0; elem < elements_per_thread; elem++) {
        uint elem_idx = thread_id_in_group * elements_per_thread + elem;
        if (elem_idx < padded_tile_size) {
            // Convert linear index to (t_offset, d_offset) within the padded tile
            uint t_offset = elem_idx / tile_d_size;
            uint d_offset = elem_idx % tile_d_size;
            
            // Global coordinates (t_offset includes padding, so subtract W-1)
            int global_t = (int)tg_t_start + (int)t_offset - (int)(W - 1);
            uint global_d = tg_d_start + d_offset;
            
            // Load input data if within bounds
            float input_val = 0.0f;
            float mask_val = 1.0f;
            
            // Check global bounds. Crucially, check global_t >= 0 for causality.
            if (b < B && global_t >= 0 && (uint)global_t < T && global_d < D) {
                uint input_idx = b * TD + (uint)global_t * D + global_d;
                input_val = input[input_idx];
                
                if (mask != nullptr) {
                    uint mask_idx = b * T + (uint)global_t;
                    mask_val = mask[mask_idx];
                }
            }
            
            shared_input[elem_idx] = input_val;
            shared_mask[elem_idx] = mask_val;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // --- 2. Convolution Computation (Optimized) ---
    
    // Check if the current thread's output element is within global bounds
    if (b >= B || t >= T || d >= D) return;
    
    const uint output_idx = b * TD + t * D + d;
    const uint weight_base = d * W;
    
    // Initialize result with bias
    float result = (bias != nullptr) ? bias[d] : 0.0f;
    
    // Causal convolution using shared memory (SSM Convention B)
    for (uint w = 0; w < W; w++) {
        // Optimized Index Calculation:
        // We are calculating Y[t] using Convention B.
        // We need Input[t - (W-1-w)] and Weight[w].
        // The index in shared memory corresponding to Input[t - (W-1-w)] is simply: local_t + w.
        uint shared_t_idx = local_t + w;
        uint shared_idx = shared_t_idx * tile_d_size + local_d;

        // Load from shared memory. The collaborative load ensures boundary conditions (e.g. t<0) are handled (loaded as 0.0f).
        float input_val = shared_input[shared_idx];
        float mask_val = shared_mask[shared_idx];
            
        // Apply mask
        input_val *= mask_val;
            
        float weight_val = weight[weight_base + w];
        result += weight_val * input_val;
    }
    
    // Apply SiLU activation
    if (use_silu) {
        result = silu(result);
    }
    
    // Apply residual connection
    if (use_residual) {
        // Residual connection uses the original input at the current time step (t, d)
        result += input[output_idx];
    }
    
    // Write output
    output[output_idx] = result;
}

// Fused ShortConvolution (float16 version)
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

// Fused ShortConvolution (bfloat16 version)
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
    device const float *x [[buffer(0)]],              // single step input (B, D) - new token
    device float *conv_state [[buffer(1)]],           // convolution state (B, D, STATE_LEN) - in-place update
    device const float *weight [[buffer(2)]],         // weights (D, W)
    device const float *bias [[buffer(3)]],           // bias (D) - optional
    device const int *cache_seqlens [[buffer(4)]],    // current sequence length for each batch (B,)
    device float *output [[buffer(5)]],               // single step output (B, D)
    
    constant uint &B [[buffer(6)]],                   // batch_size
    constant uint &D [[buffer(7)]],                   // hidden_dim
    constant uint &W [[buffer(8)]],                   // kernel_width (fixed to 4)
    constant uint &STATE_LEN [[buffer(9)]],           // state buffer length
    constant bool &use_silu [[buffer(10)]],
    constant bool &use_residual [[buffer(11)]],
    
    uint2 gid [[thread_position_in_grid]]             // (B, D)
)
{
    const uint b = gid.x;  // batch index
    const uint d = gid.y;  // dimension index
    
    // Boundary check
    if (b >= B || d >= D) return;
    
    // Get current sequence length
    int current_seq_len = cache_seqlens[b];
    
    // Calculate write position in circular buffer
    uint write_pos = (uint)current_seq_len % STATE_LEN;
    
    // Calculate linear index
    const uint x_idx = b * D + d;
    const uint output_idx = b * D + d;
    const uint weight_base = d * W;
    const uint state_base = b * D * STATE_LEN + d * STATE_LEN;
    
    // Read current input
    float current_input = x[x_idx];
    
    // Initialize result to bias
    float result = (bias != nullptr) ? bias[d] : 0.0f;
    
    // Execute causal convolution: need to read past W-1 states + current input
    for (uint w = 0; w < W; w++) {
        float input_val;
        
        if (w == W - 1) {
            // Last weight corresponds to current input
            input_val = current_input;
        } else {
            // Read history data from circular buffer
            // Position calculation: (write_pos - (W - 1 - w)) % STATE_LEN
            int hist_offset = (int)(W - 1 - w);
            int hist_pos = ((int)write_pos - hist_offset + (int)STATE_LEN) % (int)STATE_LEN;
            uint state_idx = state_base + (uint)hist_pos;
            input_val = conv_state[state_idx];
        }
        
        float weight_val = weight[weight_base + w];
        result += weight_val * input_val;
    }
    
    // Apply activation function
    if (use_silu) {
        result = silu(result);
    }
    
    // Apply residual connection
    if (use_residual) {
        result += current_input;
    }
    
    // Update state: write current input to circular buffer
    uint state_write_idx = state_base + write_pos;
    conv_state[state_write_idx] = current_input;
    
    // Write output
    output[output_idx] = result;
}

// Float16 version
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

// BFloat16 version
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
// BACKWARD PASS KERNELS - OPTIMIZED FOR O(W) COMPLEXITY
// =====================================================================================

// Pre-activation gradient computation kernel (float32)
// This kernel computes d_preact = grad_output * silu_derivative once per element
// avoiding the O(W²) complexity in the original backward pass
kernel void causal_conv1d_preact_grad_kernel(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    device const float *bias [[buffer(2)]],        // Forward bias (dim) - may be nullptr
    device const float *grad_output [[buffer(3)]], // Gradient w.r.t. output (batch, dim, seqlen)
    device float *d_preact [[buffer(4)]],          // Output: d_preact (batch, dim, seqlen)
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
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
    
    float grad_out = grad_output[output_idx];
    
    if (silu_activation) {
        // Recompute forward pass once to get pre-activation value
        float pre_activation = 0.0f;
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                float weight_val = weight[weight_base + w];
                pre_activation += weight_val * input_val;
            }
        }
        if (bias != nullptr) {
            pre_activation += bias[channel_id];
        }
        
        // Apply SiLU derivative: d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float sigmoid_val = 1.0f / (1.0f + exp(-pre_activation));
        grad_out *= sigmoid_val * (1.0f + pre_activation * (1.0f - sigmoid_val));
    }
    
    d_preact[output_idx] = grad_out;
}

// Pre-activation gradient computation kernel (float16)
kernel void causal_conv1d_preact_grad_kernel_f16(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device const half *grad_output [[buffer(3)]],
    device float *d_preact [[buffer(4)]],          // Always use float32 for d_preact
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
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
    
    d_preact[output_idx] = grad_out;
}

// Pre-activation gradient computation kernel (bfloat16)
kernel void causal_conv1d_preact_grad_kernel_bf16(
    device const ushort *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device const ushort *grad_output [[buffer(3)]],
    device float *d_preact [[buffer(4)]],          // Always use float32 for d_preact
    
    constant uint &batch_size [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &seqlen [[buffer(7)]],
    constant bool &silu_activation [[buffer(8)]],
    
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
    
    d_preact[output_idx] = grad_out;
}

// ORIGINAL backward pass for basic causal conv1d (float32) - restored for compatibility
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

// Hierarchical reduction optimized backward kernel (float32)
// Uses threadgroup memory for local accumulation to reduce atomic contention
kernel void causal_conv1d_bwd_kernel_hierarchical(
    device const float *x [[buffer(0)]],           // Original input (batch, dim, seqlen)
    device const float *weight [[buffer(1)]],      // Original weight (dim, width)
    // NOTE: This now takes d_preact, not grad_output
    device const float *d_preact [[buffer(2)]],    // Pre-computed activation gradients
    device float *grad_x [[buffer(3)]],            // Gradient w.r.t. input
    device float *grad_weight [[buffer(4)]],       // Gradient w.r.t. weight - accumulated
    device float *grad_bias [[buffer(5)]],         // Gradient w.r.t. bias - accumulated
    
    constant uint &batch_size [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &seqlen [[buffer(8)]],
    
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
)
{
    const uint batch_id = threadgroup_position_in_grid.x;
    const uint channel_id = threadgroup_position_in_grid.y;
    
    // Each threadgroup handles one channel. Threads iterate over seqlen.
    if (batch_id >= batch_size || channel_id >= dim) {
        return;
    }
    
    const uint width = 4;
    
    // Each threadgroup handles one channel. Threads iterate over seqlen.
    for (uint seq_pos = thread_index_in_threadgroup; seq_pos < seqlen; seq_pos += threads_per_threadgroup.x) {
        
        const uint input_base = batch_id * dim * seqlen + channel_id * seqlen;
        const uint weight_base = channel_id * width;
        const uint output_idx = input_base + seq_pos;
        
        // Use pre-computed activation gradient
        float grad_out = d_preact[output_idx];
        
        // --- 1. Compute gradient w.r.t. input (dx) ---
        // For causal convolution backward pass:
        // dx[t] = sum_{w=0..W-1} weight[w] * d_preact[t + (W-1-w)]
        // This accounts for how input x[t] affects all future outputs
        float dx_val = 0.0f;
        for (uint w = 0; w < width; w++) {
            int future_pos = (int)seq_pos + (int)(width - 1 - w);
            if (future_pos < (int)seqlen) {
                float future_grad = d_preact[input_base + future_pos];
                dx_val += weight[weight_base + w] * future_grad;
            }
        }
        grad_x[output_idx] = dx_val;
        
        // --- 2. Accumulate gradients for weight and bias ---
        // Use atomic operations since multiple threads may write to the same location
        // dweight[w] += x[t - (W-1-w)] * d_preact[t]
        for (uint w = 0; w < width; w++) {
            int input_pos = (int)seq_pos - (int)(width - 1 - w);
            if (input_pos >= 0) {
                float input_val = x[input_base + input_pos];
                atomic_fetch_add_explicit((device atomic<float>*)&grad_weight[weight_base + w], input_val * grad_out, memory_order_relaxed);
            }
        }
        
        if (grad_bias != nullptr) {
            atomic_fetch_add_explicit((device atomic<float>*)&grad_bias[channel_id], grad_out, memory_order_relaxed);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // No need for final reduction since we're writing directly to global memory
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
    device float *grad_weight [[buffer(5)]],      // Use float32 for accumulation
    device const half *bias [[buffer(6)]],        // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(7)]],        // Use float32 for accumulation
    
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
    device float *grad_weight [[buffer(5)]],      // Use float32 for accumulation
    device const ushort *bias [[buffer(6)]],      // Forward bias (may be nullptr)
    device float *grad_bias [[buffer(7)]],        // Use float32 for accumulation
    
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
