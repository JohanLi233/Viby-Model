#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <string>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <vector>
#include <cstring>
#include <unordered_map>

// Cache Metal device, queue, and pipelines to avoid repeated initialization overhead
static id<MTLDevice> g_device = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

// Convolution parameter struct
struct ConvParams {
    int batch_size;
    int dim;
    int seqlen;
    int width;
    bool silu_activation;
    
    // Strides
    int x_batch_stride;
    int x_c_stride;
    int x_l_stride;
    int weight_c_stride;
    int weight_width_stride;
    int out_batch_stride;
    int out_c_stride;
    int out_l_stride;
};

static void ensure_metal_pipeline_initialized(id<MTLDevice> device) {
  if (g_device != nil && !g_pipelines.empty()) {
    return;
  }

  g_device = device;
  TORCH_CHECK(g_device, "Failed to get default MTLDevice");

  // Load and compile Metal source
  std::ifstream file("causal_conv1d.metal");
  if (!file.good()) {
    const char *alt = std::getenv("CAUSAL_CONV1D_METAL_PATH");
    TORCH_CHECK(alt != nullptr,
                "Cannot open causal_conv1d.metal. Set env CAUSAL_CONV1D_METAL_PATH or run from the directory containing causal_conv1d.metal");
    file = std::ifstream(alt);
    TORCH_CHECK(file.good(), "Cannot open causal_conv1d.metal from CAUSAL_CONV1D_METAL_PATH");
  }
  std::string source((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  NSString *librarySource = [NSString stringWithUTF8String:source.c_str()];

  NSError *error = nil;
  id<MTLLibrary> library = [g_device newLibraryWithSource:librarySource
                                                  options:nil
                                                    error:&error];
  TORCH_CHECK(library, "Failed to compile Metal library: ",
              [[error localizedDescription] UTF8String]);

  // Create compute pipelines
  std::vector<std::string> function_names = {
    "causal_conv1d_fwd_kernel",
    "causal_conv1d_simple_kernel",
    "causal_conv1d_simple_kernel_f16",
    "causal_conv1d_simple_kernel_bf16",
    "causal_conv1d_bwd_kernel",
    "causal_conv1d_bwd_kernel_f16",
    "causal_conv1d_bwd_kernel_bf16",
    // New optimized kernels
    "causal_conv1d_preact_grad_kernel",
    "causal_conv1d_preact_grad_kernel_f16", 
    "causal_conv1d_preact_grad_kernel_bf16",
    "causal_conv1d_bwd_kernel_hierarchical",
    "short_conv_fused_btd_kernel",
    "short_conv_fused_btd_kernel_f16",
    "short_conv_fused_btd_kernel_bf16",
    "short_conv_fused_btd_kernel_tiled",  // New tiled version
    "short_conv_fused_btd_bwd_kernel",
    "short_conv_fused_btd_bwd_kernel_f16",
    "short_conv_fused_btd_bwd_kernel_bf16",
    "short_conv_update_kernel",
    "short_conv_update_kernel_f16",
    "short_conv_update_kernel_bf16",
    "short_conv_update_bwd_kernel",
    "short_conv_update_bwd_kernel_f16",
    "short_conv_update_bwd_kernel_bf16"
  };
  
  for (const auto& func_name : function_names) {
    NSString *functionName = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    TORCH_CHECK(function, "Failed to find Metal function '", func_name, "'");
    
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pipeline, "Failed to create compute pipeline state for '", func_name, "': ",
                [[error localizedDescription] UTF8String]);
    
    g_pipelines[func_name] = pipeline;
  }
}

// Setup convolution parameters
static ConvParams setup_conv_params(
    const torch::Tensor &x, 
    const torch::Tensor &weight,
    const torch::Tensor &out,
    bool silu_activation
) {
    ConvParams params;
    
    // Basic dimensions
    params.batch_size = x.size(0);
    params.dim = x.size(1);
    params.seqlen = x.size(2);
    params.width = weight.size(1);
    params.silu_activation = silu_activation;
    
    // Strides
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);
    
    return params;
}

// C++ wrapper for MPS causal conv1d kernel
torch::Tensor causal_conv1d_fwd_mps(
    const torch::Tensor &x, 
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool silu_activation = false
) {
  // Input validation
  TORCH_CHECK(x.device().is_mps(), "Tensor 'x' must be a MPS tensor");
  TORCH_CHECK(weight.device().is_mps(), "Tensor 'weight' must be a MPS tensor");
  TORCH_CHECK(x.is_contiguous(), "Tensor 'x' must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Tensor 'weight' must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
          x.scalar_type() == torch::kBFloat16,
      "Only float32/float16/bfloat16 supported for x");
  TORCH_CHECK(
      weight.scalar_type() == x.scalar_type(),
      "weight dtype must match x dtype");
  
  // Shape checks
  TORCH_CHECK(x.dim() == 3, "Input tensor must have shape (batch, dim, seqlen)");
  TORCH_CHECK(weight.dim() == 2, "Weight tensor must have shape (dim, width)");
  TORCH_CHECK(x.size(1) == weight.size(0), "Input dim must match weight dim");
  
  if (bias.defined() && bias.numel() > 0) {
    TORCH_CHECK(bias.device().is_mps(), "Tensor 'bias' must be a MPS tensor");
    TORCH_CHECK(bias.is_contiguous(), "Tensor 'bias' must be contiguous");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x dtype");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(1), "Bias dim must match input dim");
  }
  TORCH_CHECK(weight.size(1) == 4, "Simple kernel supports width=4");

  const int64_t batch_size = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t seqlen = x.size(2);
  
  if (seqlen == 0) {
    return torch::empty_like(x);
  }

  // Use PyTorch current MPS stream
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = (id<MTLDevice>)stream->device();
  ensure_metal_pipeline_initialized(device);

  // Create output tensor
  auto result_tensor = torch::empty_like(x);
  
  // Get Metal buffers
  id<MTLBuffer> x_buffer = (id<MTLBuffer>)x.storage().data();
  id<MTLBuffer> weight_buffer = (id<MTLBuffer>)weight.storage().data();
  id<MTLBuffer> bias_buffer = (bias.defined() && bias.numel() > 0) ? (id<MTLBuffer>)bias.storage().data() : nil;
  id<MTLBuffer> result_buffer = (id<MTLBuffer>)result_tensor.storage().data();
  
  // Compute byte offsets
  const NSUInteger x_offset = (NSUInteger)(x.storage_offset() * x.element_size());
  const NSUInteger weight_offset = (NSUInteger)(weight.storage_offset() * weight.element_size());
  const NSUInteger bias_offset = (bias.defined() && bias.numel() > 0) ? (NSUInteger)(bias.storage_offset() * bias.element_size()) : 0;
  const NSUInteger result_offset = (NSUInteger)(result_tensor.storage_offset() * result_tensor.element_size());

  // Note: removed unused params as we use a simplified kernel
  // ConvParams params = setup_conv_params(x, weight, result_tensor, silu_activation);

  // Select kernel by dtype
  id<MTLComputePipelineState> pipeline = nil;
  if (x.scalar_type() == torch::kFloat) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel"];
  } else if (x.scalar_type() == torch::kHalf) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel_f16"];
  } else if (x.scalar_type() == torch::kBFloat16) {
    pipeline = g_pipelines["causal_conv1d_simple_kernel_bf16"];
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  
  // Encode compute command
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  [encoder setComputePipelineState:pipeline];
  
  // Set buffers
  [encoder setBuffer:x_buffer offset:x_offset atIndex:0];
  [encoder setBuffer:weight_buffer offset:weight_offset atIndex:1];
  if (bias_buffer != nil) {
    [encoder setBuffer:bias_buffer offset:bias_offset atIndex:2];
  } else {
    [encoder setBuffer:nil offset:0 atIndex:2];
  }
  [encoder setBuffer:result_buffer offset:result_offset atIndex:3];
  
  // Set parameters
  uint32_t batch_u32 = (uint32_t)batch_size;
  uint32_t dim_u32 = (uint32_t)dim;
  uint32_t seqlen_u32 = (uint32_t)seqlen;
  bool silu_flag = silu_activation;
  
  [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:4];
  [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:5];
  [encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:6];
  [encoder setBytes:&silu_flag length:sizeof(bool) atIndex:7];

  // Compute threadgroup size
  // Use 3D grid: (batch_size, dim, seqlen)
  MTLSize gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, (NSUInteger)seqlen);
  
  // Choose threadgroup size (conservative)
  NSUInteger maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
  NSUInteger threadsPerGroup = MIN(256, maxThreadsPerGroup);
  MTLSize threadgroupSize = MTLSizeMake(1, 1, threadsPerGroup);
  
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  return result_tensor;
}

// Simplified interface function to match PyTorch autograd function expectations
torch::Tensor causal_conv1d_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &seq_idx,
    const torch::Tensor &initial_states,
    const torch::Tensor &final_states_out,
    bool silu_activation
) {
    // Only basic functionality is implemented; advanced parameters are ignored
    return causal_conv1d_fwd_mps(x, weight, bias, silu_activation);
}

// Helper to safely get MTLBuffer and offset
static inline std::pair<id<MTLBuffer>, NSUInteger> getBufferAndOffset(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) return {nil, 0};
    id<MTLBuffer> buffer = (id<MTLBuffer>)t.storage().data();
    NSUInteger offset = (NSUInteger)(t.storage_offset() * t.element_size());
    return {buffer, offset};
}

// Fused ShortConvolution (B, T, D): Mask + Conv (W=4) + SiLU + Residual
torch::Tensor short_conv_fused_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &mask,
    bool silu_activation,
    bool residual
) {
  TORCH_CHECK(x.device().is_mps(), "Input 'x' must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "'weight' must be on MPS");
  TORCH_CHECK(!bias.defined() || bias.device().is_mps(), "'bias' must be on MPS if provided");
  TORCH_CHECK(!mask.defined() || mask.device().is_mps(), "'mask' must be on MPS if provided");

  TORCH_CHECK(
      x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
          x.scalar_type() == torch::kBFloat16,
      "Only float32/float16/bfloat16 supported for x");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
  // Allow bias to be undefined or empty as a marker of no-bias
  TORCH_CHECK(!bias.defined() || bias.numel() == 0 || bias.scalar_type() == x.scalar_type(), "bias dtype must match x");
  // If mask is undefined or empty, skip dtype check
  TORCH_CHECK(
      !mask.defined() || mask.numel() == 0 || mask.scalar_type() == x.scalar_type(),
      "mask dtype must match x");

  TORCH_CHECK(x.dim() == 3, "Expected x shape (B, T, D)");
  TORCH_CHECK(weight.dim() == 2, "Expected weight shape (D, W)");
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t D = x.size(2);
  TORCH_CHECK(weight.size(0) == D, "weight dim must match D");
  const int64_t W = weight.size(1);
  TORCH_CHECK(W == 4, "Fused kernel supports width=4 only");

  const bool has_bias = bias.defined() && bias.numel() > 0;
  if (has_bias) {
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == D, "bias must be (D)");
  }
  if (mask.defined() && mask.numel() > 0) {
    if (mask.dim() == 1) {
      TORCH_CHECK(mask.numel() == B * T || mask.numel() == T || mask.numel() == B,
                  "mask must be length T or B or B*T when 1D");
    } else {
      TORCH_CHECK(mask.dim() == 2 && ((mask.size(0) == B && mask.size(1) == T) ||
                                      (mask.size(0) == T && mask.size(1) == B) ||
                                      (mask.size(0) == 1 && mask.size(1) == T) ||
                                      (mask.size(0) == B && mask.size(1) == 1)),
                  "mask must be (B,T) or (T,B) or (1,T) or (B,1)");
    }
  }

  auto x_contig = x.contiguous();
  auto w_contig = weight.contiguous();
  auto b_contig = has_bias ? bias.contiguous() : bias;
  auto m_contig = mask.defined() ? mask.contiguous() : mask;

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = (id<MTLDevice>)stream->device();
  ensure_metal_pipeline_initialized(device);

  auto out = torch::empty_like(x_contig);

  auto [x_buf, x_off] = getBufferAndOffset(x_contig);
  auto [w_buf, w_off] = getBufferAndOffset(w_contig);
  auto [b_buf, b_off] = getBufferAndOffset(b_contig);
  auto [m_buf, m_off] = getBufferAndOffset(m_contig);
  auto [o_buf, o_off] = getBufferAndOffset(out);

  id<MTLComputePipelineState> pipeline = nil;
  bool use_tiled_kernel = (T >= 64 && D >= 64);  // Use tiled kernel for larger problem sizes
  
  if (x.scalar_type() == torch::kFloat) {
    pipeline = use_tiled_kernel ? 
      g_pipelines["short_conv_fused_btd_kernel_tiled"] : 
      g_pipelines["short_conv_fused_btd_kernel"];
  } else if (x.scalar_type() == torch::kHalf) {
    pipeline = g_pipelines["short_conv_fused_btd_kernel_f16"];  // Tiled version not yet implemented for f16
  } else if (x.scalar_type() == torch::kBFloat16) {
    pipeline = g_pipelines["short_conv_fused_btd_kernel_bf16"];  // Tiled version not yet implemented for bf16
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  [encoder setComputePipelineState:pipeline];

  // Buffers 0..4
  [encoder setBuffer:x_buf offset:x_off atIndex:0];
  [encoder setBuffer:w_buf offset:w_off atIndex:1];
  [encoder setBuffer:b_buf offset:b_off atIndex:2];
  [encoder setBuffer:m_buf offset:m_off atIndex:3];
  [encoder setBuffer:o_buf offset:o_off atIndex:4];

  // Scalars 5..9
  uint32_t B_u32 = (uint32_t)B;
  uint32_t T_u32 = (uint32_t)T;
  uint32_t D_u32 = (uint32_t)D;
  bool use_silu = silu_activation;
  bool use_resid = residual;

  [encoder setBytes:&B_u32 length:sizeof(uint32_t) atIndex:5];
  [encoder setBytes:&T_u32 length:sizeof(uint32_t) atIndex:6];
  [encoder setBytes:&D_u32 length:sizeof(uint32_t) atIndex:7];
  [encoder setBytes:&use_silu length:sizeof(bool) atIndex:8];
  [encoder setBytes:&use_resid length:sizeof(bool) atIndex:9];

  MTLSize gridSize = MTLSizeMake((NSUInteger)B, (NSUInteger)T, (NSUInteger)D);
  NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
  
  MTLSize tgSize;
  if (use_tiled_kernel) {
    // Configure threadgroup size for tiled kernel for optimal memory usage
    NSUInteger tg_t = MIN(MIN((NSUInteger)T, 32u), maxThreads / 32);
    NSUInteger tg_d = MIN(MIN((NSUInteger)D, 32u), maxThreads / tg_t);
    tgSize = MTLSizeMake(1, tg_t, tg_d);
  } else {
    // Original threadgroup configuration
    NSUInteger tz = MIN((NSUInteger)256, maxThreads);
    if (tz == 0) tz = 1;
    tgSize = MTLSizeMake(1, 1, tz);
  }
  
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

  return out;
}

// Stateful update function for single-token inference
torch::Tensor short_conv_update_mps(
    const torch::Tensor &x,
    const torch::Tensor &conv_state,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &cache_seqlens,
    bool silu_activation,
    bool residual
) {
  TORCH_CHECK(x.device().is_mps(), "Input 'x' must be on MPS");
  TORCH_CHECK(conv_state.device().is_mps(), "'conv_state' must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "'weight' must be on MPS");
  TORCH_CHECK(!bias.defined() || bias.device().is_mps(), "'bias' must be on MPS if provided");
  TORCH_CHECK(cache_seqlens.device().is_mps(), "'cache_seqlens' must be on MPS");

  TORCH_CHECK(
      x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
          x.scalar_type() == torch::kBFloat16,
      "Only float32/float16/bfloat16 supported for x");
  TORCH_CHECK(conv_state.scalar_type() == x.scalar_type(), "conv_state dtype must match x");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
  // Allow bias to be undefined or empty
  TORCH_CHECK(!bias.defined() || bias.numel() == 0 || bias.scalar_type() == x.scalar_type(), "bias dtype must match x");
  TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt, "cache_seqlens must be int32");

  TORCH_CHECK(x.dim() == 2, "Expected x shape (B, D)");
  TORCH_CHECK(conv_state.dim() == 3, "Expected conv_state shape (B, D, STATE_LEN)");
  TORCH_CHECK(weight.dim() == 2, "Expected weight shape (D, W)");
  TORCH_CHECK(cache_seqlens.dim() == 1, "Expected cache_seqlens shape (B,)");
  
  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t STATE_LEN = conv_state.size(2);
  const int64_t W = weight.size(1);
  
  TORCH_CHECK(weight.size(0) == D, "weight dim must match D");
  TORCH_CHECK(conv_state.size(0) == B && conv_state.size(1) == D, "conv_state shape mismatch");
  TORCH_CHECK(cache_seqlens.size(0) == B, "cache_seqlens batch size mismatch");
  TORCH_CHECK(W == 4, "Update kernel supports width=4 only");

  const bool has_bias_update = bias.defined() && bias.numel() > 0;
  if (has_bias_update) {
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == D, "bias must be (D)");
  }

  auto x_contig = x.contiguous();
  auto conv_state_contig = conv_state.contiguous();
  auto weight_contig = weight.contiguous();
  auto bias_contig = has_bias_update ? bias.contiguous() : bias;
  auto cache_seqlens_contig = cache_seqlens.contiguous();

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  id<MTLDevice> device = (id<MTLDevice>)stream->device();
  ensure_metal_pipeline_initialized(device);

  auto out = torch::empty_like(x_contig);

  auto [x_buf, x_off] = getBufferAndOffset(x_contig);
  auto [cs_buf, cs_off] = getBufferAndOffset(conv_state_contig);
  auto [w_buf, w_off] = getBufferAndOffset(weight_contig);
  auto [b_buf, b_off] = getBufferAndOffset(bias_contig);
  auto [cl_buf, cl_off] = getBufferAndOffset(cache_seqlens_contig);
  auto [o_buf, o_off] = getBufferAndOffset(out);

  id<MTLComputePipelineState> pipeline = nil;
  if (x.scalar_type() == torch::kFloat) {
    pipeline = g_pipelines["short_conv_update_kernel"];
  } else if (x.scalar_type() == torch::kHalf) {
    pipeline = g_pipelines["short_conv_update_kernel_f16"];
  } else if (x.scalar_type() == torch::kBFloat16) {
    pipeline = g_pipelines["short_conv_update_kernel_bf16"];
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }

  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  [encoder setComputePipelineState:pipeline];

  // Buffers 0..5
  [encoder setBuffer:x_buf offset:x_off atIndex:0];
  [encoder setBuffer:cs_buf offset:cs_off atIndex:1];
  [encoder setBuffer:w_buf offset:w_off atIndex:2];
  [encoder setBuffer:b_buf offset:b_off atIndex:3];
  [encoder setBuffer:cl_buf offset:cl_off atIndex:4];
  [encoder setBuffer:o_buf offset:o_off atIndex:5];

  // Scalars 6..11
  uint32_t B_u32 = (uint32_t)B;
  uint32_t D_u32 = (uint32_t)D;
  uint32_t W_u32 = (uint32_t)W;
  uint32_t STATE_LEN_u32 = (uint32_t)STATE_LEN;
  bool use_silu = silu_activation;
  bool use_resid = residual;

  [encoder setBytes:&B_u32 length:sizeof(uint32_t) atIndex:6];
  [encoder setBytes:&D_u32 length:sizeof(uint32_t) atIndex:7];
  [encoder setBytes:&W_u32 length:sizeof(uint32_t) atIndex:8];
  [encoder setBytes:&STATE_LEN_u32 length:sizeof(uint32_t) atIndex:9];
  [encoder setBytes:&use_silu length:sizeof(bool) atIndex:10];
  [encoder setBytes:&use_resid length:sizeof(bool) atIndex:11];

  MTLSize gridSize = MTLSizeMake((NSUInteger)B, (NSUInteger)D, 1);
  NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
  NSUInteger tg = MIN((NSUInteger)256, maxThreads);
  if (tg == 0) tg = 1;
  MTLSize tgSize = MTLSizeMake(1, tg, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

  return out;
}

// =====================================================================================
// BACKWARD PASS IMPLEMENTATIONS
// =====================================================================================

// Optimized backward pass for basic causal conv1d using two-pass approach
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d_bwd_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &grad_output,
    bool silu_activation
) {
    // Input validation
    TORCH_CHECK(x.device().is_mps(), "Tensor 'x' must be a MPS tensor");
    TORCH_CHECK(weight.device().is_mps(), "Tensor 'weight' must be a MPS tensor");
    TORCH_CHECK(grad_output.device().is_mps(), "Tensor 'grad_output' must be a MPS tensor");
    TORCH_CHECK(x.is_contiguous(), "Tensor 'x' must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Tensor 'weight' must be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "Tensor 'grad_output' must be contiguous");
    
    const int64_t batch_size = x.size(0);
    const int64_t dim = x.size(1);
    const int64_t seqlen = x.size(2);
    const int64_t width = weight.size(1);
    
    TORCH_CHECK(width == 4, "Only width=4 supported in backward kernel");
    
    if (seqlen == 0) {
        auto dx = torch::empty_like(x);
        auto dweight = torch::zeros_like(weight);
        auto dbias = bias.defined() && bias.numel() > 0 ? torch::zeros_like(bias) : torch::tensor({});
        return std::make_tuple(dx, dweight, dbias);
    }

    // Use PyTorch's current MPS stream
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    ensure_metal_pipeline_initialized(device);

    // Create output tensors
    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros_like(weight, torch::dtype(torch::kFloat32));
    auto dbias = torch::Tensor();
    if (bias.defined() && bias.numel() > 0) {
        dbias = torch::zeros_like(bias, torch::dtype(torch::kFloat32));
    }

    // Two-pass optimization for SiLU backward pass:
    // Pass 1: Compute pre-activation gradients (O(W) complexity)
    // Pass 2: Use pre-computed gradients in backward pass (O(W) complexity)
    // Total: O(W) instead of O(W²)
    
    // Get Metal buffers and offsets
    auto [x_buf, x_off] = getBufferAndOffset(x);
    auto [w_buf, w_off] = getBufferAndOffset(weight);
    auto [grad_out_buf, grad_out_off] = getBufferAndOffset(grad_output);
    auto [dx_buf, dx_off] = getBufferAndOffset(dx);
    auto [dw_buf, dw_off] = getBufferAndOffset(dweight);
    
    // Forward bias buffer
    id<MTLBuffer> fwd_bias_buf = nil;
    NSUInteger fwd_bias_off = 0;
    if (bias.defined() && bias.numel() > 0) {
        auto [b_buf, b_off] = getBufferAndOffset(bias);
        fwd_bias_buf = b_buf;
        fwd_bias_off = b_off;
    }
    
    id<MTLBuffer> db_buf = nil;
    NSUInteger db_off = 0;
    if (dbias.defined()) {
        auto [bias_buf, bias_off] = getBufferAndOffset(dbias);
        db_buf = bias_buf;
        db_off = bias_off;
    }

    // Two-pass optimization: First pass computes pre-activation gradients
    auto d_preact = torch::empty_like(grad_output, torch::dtype(torch::kFloat32));
    
    // Set up parameters for both passes
    uint32_t batch_u32 = (uint32_t)batch_size;
    uint32_t dim_u32 = (uint32_t)dim;
    uint32_t seqlen_u32 = (uint32_t)seqlen;
    bool silu_flag = silu_activation;
    
    // Set up dispatch parameters
    MTLSize gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, (NSUInteger)seqlen);
    NSUInteger maxThreadsPerGroup = [g_pipelines["causal_conv1d_preact_grad_kernel"] maxTotalThreadsPerThreadgroup];
    NSUInteger threadsPerGroup = MIN(256, maxThreadsPerGroup);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, threadsPerGroup);
    
    // Pass 1: Compute pre-activation gradients using optimized kernel
    id<MTLComputePipelineState> preact_pipeline = nil;
    if (x.scalar_type() == torch::kFloat) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel"];
    } else if (x.scalar_type() == torch::kHalf) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel_f16"];
    } else if (x.scalar_type() == torch::kBFloat16) {
        preact_pipeline = g_pipelines["causal_conv1d_preact_grad_kernel_bf16"];
    } else {
        TORCH_CHECK(false, "Unsupported dtype for backward pass");
    }
    
    auto [d_preact_buf, d_preact_off] = getBufferAndOffset(d_preact);
    
    // Execute pre-activation gradient kernel
    id<MTLComputeCommandEncoder> preact_encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [preact_encoder setComputePipelineState:preact_pipeline];
    
    [preact_encoder setBuffer:x_buf offset:x_off atIndex:0];
    [preact_encoder setBuffer:w_buf offset:w_off atIndex:1];
    [preact_encoder setBuffer:fwd_bias_buf offset:fwd_bias_off atIndex:2];
    [preact_encoder setBuffer:grad_out_buf offset:grad_out_off atIndex:3];
    [preact_encoder setBuffer:d_preact_buf offset:d_preact_off atIndex:4];
    
    [preact_encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:5];
    [preact_encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:6];
    [preact_encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:7];
    [preact_encoder setBytes:&silu_flag length:sizeof(bool) atIndex:8];
    
    [preact_encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    // Pass 2: Use hierarchical backward kernel with pre-computed gradients
    id<MTLComputePipelineState> pipeline = g_pipelines["causal_conv1d_bwd_kernel_hierarchical"];

    // Execute optimized backward kernel using pre-computed gradients
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    // Set buffers for hierarchical backward kernel
    [encoder setBuffer:x_buf offset:x_off atIndex:0];
    [encoder setBuffer:w_buf offset:w_off atIndex:1];
    [encoder setBuffer:d_preact_buf offset:d_preact_off atIndex:2];  // Pre-computed activation gradients
    [encoder setBuffer:dx_buf offset:dx_off atIndex:3];
    [encoder setBuffer:dw_buf offset:dw_off atIndex:4];
    [encoder setBuffer:db_buf offset:db_off atIndex:5];

    // Set parameters for hierarchical backward kernel
    [encoder setBytes:&batch_u32 length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&dim_u32 length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&seqlen_u32 length:sizeof(uint32_t) atIndex:8];

    // Use threadgroup-based dispatch for hierarchical kernel
    MTLSize hierarchical_gridSize = MTLSizeMake((NSUInteger)batch_size, (NSUInteger)dim, 1);
    NSUInteger hierarchical_maxThreadsPerGroup = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger hierarchical_threadsPerGroup = MIN(256, hierarchical_maxThreadsPerGroup);
    MTLSize hierarchical_threadgroupSize = MTLSizeMake(hierarchical_threadsPerGroup, 1, 1);
    
    [encoder dispatchThreadgroups:hierarchical_gridSize threadsPerThreadgroup:hierarchical_threadgroupSize];

    // Convert accumulation dtypes back to match input dtypes
    dweight = dweight.to(weight.dtype());
    if (dbias.defined()) {
        dbias = dbias.to(bias.dtype());
    }

    return std::make_tuple(dx, dweight, dbias);
}

// Backward pass for fused short convolution
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> short_conv_fused_bwd_mps(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &attention_mask,
    const torch::Tensor &grad_output,
    bool activation,
    bool residual
) {
    TORCH_CHECK(x.device().is_mps(), "Input 'x' must be on MPS");
    TORCH_CHECK(weight.device().is_mps(), "'weight' must be on MPS");
    TORCH_CHECK(grad_output.device().is_mps(), "'grad_output' must be on MPS");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t D = x.size(2);
    const int64_t W = weight.size(1);

    TORCH_CHECK(W == 4, "Backward fused kernel supports width=4 only");

    if (T == 0) {
        auto dx = torch::empty_like(x);
        auto dweight = torch::zeros_like(weight);
        auto dbias = bias.defined() && bias.numel() > 0 ? torch::zeros_like(bias) : torch::tensor({});
        return std::make_tuple(dx, dweight, dbias);
    }

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    ensure_metal_pipeline_initialized(device);

    auto dx = torch::empty_like(x);
    auto dweight = torch::zeros_like(weight, torch::dtype(torch::kFloat32));
    auto dbias = torch::Tensor();
    if (bias.defined() && bias.numel() > 0) {
        dbias = torch::zeros_like(bias, torch::dtype(torch::kFloat32));
    }

    auto [x_buf, x_off] = getBufferAndOffset(x);
    auto [w_buf, w_off] = getBufferAndOffset(weight);
    auto [mask_buf, mask_off] = getBufferAndOffset(attention_mask);
    auto [grad_out_buf, grad_out_off] = getBufferAndOffset(grad_output);
    auto [dx_buf, dx_off] = getBufferAndOffset(dx);
    auto [dw_buf, dw_off] = getBufferAndOffset(dweight);
    // forward bias for activation derivative
    auto [fb_buf, fb_off] = getBufferAndOffset(bias);
    
    id<MTLBuffer> db_buf = nil;
    NSUInteger db_off = 0;
    if (dbias.defined()) {
        auto [bias_buf, bias_off] = getBufferAndOffset(dbias);
        db_buf = bias_buf;
        db_off = bias_off;
    }

    id<MTLComputePipelineState> pipeline = nil;
    if (x.scalar_type() == torch::kFloat) {
        pipeline = g_pipelines["short_conv_fused_btd_bwd_kernel"];
    } else if (x.scalar_type() == torch::kHalf) {
        pipeline = g_pipelines["short_conv_fused_btd_bwd_kernel_f16"];
    } else if (x.scalar_type() == torch::kBFloat16) {
        pipeline = g_pipelines["short_conv_fused_btd_bwd_kernel_bf16"];
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused backward pass");
    }

    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    // Set buffers
    [encoder setBuffer:x_buf offset:x_off atIndex:0];
    [encoder setBuffer:w_buf offset:w_off atIndex:1];
    [encoder setBuffer:mask_buf offset:mask_off atIndex:2];
    [encoder setBuffer:grad_out_buf offset:grad_out_off atIndex:3];
    [encoder setBuffer:dx_buf offset:dx_off atIndex:4];
    [encoder setBuffer:dw_buf offset:dw_off atIndex:5];
    [encoder setBuffer:fb_buf offset:fb_off atIndex:6];
    [encoder setBuffer:db_buf offset:db_off atIndex:7];

    // Set parameters
    uint32_t B_u32 = (uint32_t)B;
    uint32_t T_u32 = (uint32_t)T;
    uint32_t D_u32 = (uint32_t)D;
    bool use_activation = activation;
    bool use_residual = residual;

    [encoder setBytes:&B_u32 length:sizeof(uint32_t) atIndex:8];
    [encoder setBytes:&T_u32 length:sizeof(uint32_t) atIndex:9];
    [encoder setBytes:&D_u32 length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&use_activation length:sizeof(bool) atIndex:11];
    [encoder setBytes:&use_residual length:sizeof(bool) atIndex:12];

    MTLSize gridSize = MTLSizeMake((NSUInteger)B, (NSUInteger)T, (NSUInteger)D);
    NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger tz = MIN((NSUInteger)256, maxThreads);
    if (tz == 0) tz = 1;
    MTLSize tgSize = MTLSizeMake(1, 1, tz);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    // Convert accumulation dtypes back to match input dtypes
    dweight = dweight.to(weight.dtype());
    if (dbias.defined()) {
        dbias = dbias.to(bias.dtype());
    }

    return std::make_tuple(dx, dweight, dbias);
}

// Backward pass for convolution update (inference)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> short_conv_update_bwd_mps(
    const torch::Tensor &x,
    const torch::Tensor &conv_state,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    const torch::Tensor &cache_seqlens,
    const torch::Tensor &grad_output,
    bool activation,
    bool residual
) {
    TORCH_CHECK(x.device().is_mps(), "Input 'x' must be on MPS");
    TORCH_CHECK(conv_state.device().is_mps(), "'conv_state' must be on MPS");
    TORCH_CHECK(weight.device().is_mps(), "'weight' must be on MPS");
    TORCH_CHECK(grad_output.device().is_mps(), "'grad_output' must be on MPS");

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t STATE_LEN = conv_state.size(2);
    const int64_t W = weight.size(1);

    TORCH_CHECK(W == 4, "Update backward kernel supports width=4 only");

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    ensure_metal_pipeline_initialized(device);

    auto dx = torch::empty_like(x);
    auto dconv_state = torch::zeros_like(conv_state);
    auto dweight = torch::zeros_like(weight, torch::dtype(torch::kFloat32));
    auto dbias = torch::Tensor();
    if (bias.defined() && bias.numel() > 0) {
        dbias = torch::zeros_like(bias, torch::dtype(torch::kFloat32));
    }

    auto [x_buf, x_off] = getBufferAndOffset(x);
    auto [cs_buf, cs_off] = getBufferAndOffset(conv_state);
    auto [w_buf, w_off] = getBufferAndOffset(weight);
    auto [cl_buf, cl_off] = getBufferAndOffset(cache_seqlens);
    auto [grad_out_buf, grad_out_off] = getBufferAndOffset(grad_output);
    auto [dx_buf, dx_off] = getBufferAndOffset(dx);
    auto [dcs_buf, dcs_off] = getBufferAndOffset(dconv_state);
    auto [dw_buf, dw_off] = getBufferAndOffset(dweight);
    auto [fb_buf, fb_off] = getBufferAndOffset(bias);
    
    id<MTLBuffer> db_buf = nil;
    NSUInteger db_off = 0;
    if (dbias.defined()) {
        auto [bias_buf, bias_off] = getBufferAndOffset(dbias);
        db_buf = bias_buf;
        db_off = bias_off;
    }

    id<MTLComputePipelineState> pipeline = nil;
    if (x.scalar_type() == torch::kFloat) {
        pipeline = g_pipelines["short_conv_update_bwd_kernel"];
    } else if (x.scalar_type() == torch::kHalf) {
        pipeline = g_pipelines["short_conv_update_bwd_kernel_f16"];
    } else if (x.scalar_type() == torch::kBFloat16) {
        pipeline = g_pipelines["short_conv_update_bwd_kernel_bf16"];
    } else {
        TORCH_CHECK(false, "Unsupported dtype for update backward pass");
    }

    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    // Set buffers
    [encoder setBuffer:x_buf offset:x_off atIndex:0];
    [encoder setBuffer:cs_buf offset:cs_off atIndex:1];
    [encoder setBuffer:w_buf offset:w_off atIndex:2];
    [encoder setBuffer:cl_buf offset:cl_off atIndex:3];
    [encoder setBuffer:grad_out_buf offset:grad_out_off atIndex:4];
    [encoder setBuffer:dx_buf offset:dx_off atIndex:5];
    [encoder setBuffer:dcs_buf offset:dcs_off atIndex:6];
    [encoder setBuffer:dw_buf offset:dw_off atIndex:7];
    [encoder setBuffer:fb_buf offset:fb_off atIndex:8];
    [encoder setBuffer:db_buf offset:db_off atIndex:9];

    // Set parameters
    uint32_t B_u32 = (uint32_t)B;
    uint32_t D_u32 = (uint32_t)D;
    uint32_t W_u32 = (uint32_t)W;
    uint32_t STATE_LEN_u32 = (uint32_t)STATE_LEN;
    bool use_activation = activation;
    bool use_residual = residual;

    [encoder setBytes:&B_u32 length:sizeof(uint32_t) atIndex:10];
    [encoder setBytes:&D_u32 length:sizeof(uint32_t) atIndex:11];
    [encoder setBytes:&W_u32 length:sizeof(uint32_t) atIndex:12];
    [encoder setBytes:&STATE_LEN_u32 length:sizeof(uint32_t) atIndex:13];
    [encoder setBytes:&use_activation length:sizeof(bool) atIndex:14];
    [encoder setBytes:&use_residual length:sizeof(bool) atIndex:15];

    MTLSize gridSize = MTLSizeMake((NSUInteger)B, (NSUInteger)D, 1);
    NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger tg = MIN((NSUInteger)256, maxThreads);
    if (tg == 0) tg = 1;
    MTLSize tgSize = MTLSizeMake(1, tg, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

    // Convert accumulation dtypes back to match input dtypes
    dweight = dweight.to(weight.dtype());
    if (dbias.defined()) {
        dbias = dbias.to(bias.dtype());
    }

    return std::make_tuple(dx, dconv_state, dweight, dbias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Forward pass functions
  m.def("causal_conv1d_fwd", &causal_conv1d_fwd_mps, 
        "Causal Conv1D forward pass using Metal compute kernel (MPS)");
  m.def("causal_conv1d", &causal_conv1d_mps,
        "Causal Conv1D with full interface compatibility");
  m.def("short_conv_fused", &short_conv_fused_mps,
        "Fused ShortConvolution (Mask+Conv+SiLU+Residual) on MPS (BTD layout)");
  m.def("short_conv_update", &short_conv_update_mps,
        "Single-token causal convolution update for efficient inference");
  
  // Backward pass functions
  m.def("causal_conv1d_bwd", &causal_conv1d_bwd_mps,
        "Causal Conv1D backward pass using Metal compute kernel (MPS)");
  m.def("short_conv_fused_bwd", &short_conv_fused_bwd_mps,
        "Fused ShortConvolution backward pass on MPS (BTD layout)");
  m.def("short_conv_update_bwd", &short_conv_update_bwd_mps,
        "Single-token causal convolution update backward pass for efficient inference");
}