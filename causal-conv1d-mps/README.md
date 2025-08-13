## causal-conv1d-mps

High-performance CausalConv1D for PyTorch on Apple Silicon using Metal/MPS. It provides optimized forward/backward kernels, a fused short-convolution kernel for common (B, T, D) usage, and stateful single-token update for inference.

- Backend: Apple Metal Performance Shaders (MPS)
- Languages: Python + C++/Objective-C++ + Metal Shaders
- Precisions: fp32 / fp16 / bf16
- Kernel constraint: current convolution width is fixed to 4 (width=4)


### Features
- High-performance CausalConv1D: MPS backend with native Metal compute, optimized for both (B, D, T) and (B, T, D) layouts.
- Fused short-convolution kernel: applies masking, convolution, SiLU activation, and residual in one pass for common short-convolution paths.
- Autograd support: provided autograd functions with an optimized two-pass backward for the basic convolution (O(W) complexity).
- Single-token update: stateful short-convolution update `short_conv_update` that updates the cache in-place for inference.


### Requirements
- macOS (Apple Silicon, macOS 13+ recommended)
- Python ≥ 3.10
- PyTorch ≥ 2.1 (with MPS support)
- Clang/Xcode toolchain (to build the extension)
- Ninja ≥ 1.11 (faster builds)

If `torch.backends.mps.is_available()` returns False, please enable MPS per PyTorch documentation.


### Installation
Run at the repository root:

```bash
uv pip install -e .
```

Notes:
- The extension loads `causal_conv1d.metal` at runtime. It first checks the `CAUSAL_CONV1D_METAL_PATH` environment variable; if not set, it tries to find the file at the repo root or the current working directory.
- If using the package from a non-source directory, set:

```bash
export CAUSAL_CONV1D_METAL_PATH=/absolute/path/to/causal-conv1d-mps/causal_conv1d.metal
```


### Quick Start

#### Basic causal 1D convolution (with autograd)
```python
import torch
import causal_conv1d_mps as ccmps

B, D, T, W = 2, 128, 256, 4
x = torch.randn(B, D, T, device='mps', dtype=torch.bfloat16)
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)

# activation: None / 'silu' / 'swish'
y = ccmps.causal_conv1d(x, weight, bias, activation='silu')
loss = y.mean()
loss.backward()
```

Shapes and device constraints:
- `x`: (B, D, T) on MPS device
- `weight`: (D, W=4), same dtype as `x`, on MPS
- `bias` (optional): (D), same dtype as `x`, on MPS
- Returns: `(B, D, T)`


#### Fused short-convolution ((B, T, D) path)
```python
import torch
import causal_conv1d_mps as ccmps

B, T, D, W = 2, 256, 768, 4
x = torch.randn(B, T, D, device='mps', dtype=torch.float32)
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)
mask = torch.ones(B, T, device='mps', dtype=x.dtype)  # optional

y = ccmps.short_conv_fused(
    x, weight, bias, mask,
    activation=True,   # SiLU
    residual=True      # y += x
)
```

Mask is normalized to `(B, T)`. Supported shapes: `(B, T)`, `(1, T)`, `(B, 1)`, `(T, B)`, or 1D of length `T/B/(B*T)`.


#### Single-token incremental update (inference)
```python
import torch
import causal_conv1d_mps as ccmps

B, D, W, STATE = 2, 512, 4, 8
x = torch.randn(B, D, device='mps', dtype=torch.float32)
conv_state = torch.zeros(B, D, STATE, device='mps', dtype=x.dtype)  # updated in-place
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)
cache_seqlens = torch.zeros(B, device='mps', dtype=torch.int32)  # track current position per batch

y = ccmps.short_conv_update(
    x, conv_state, weight, bias, cache_seqlens,
    activation=True, residual=True
)
# increment cache_seqlens after use
cache_seqlens += 1
```

Requirements:
- `x`: (B, D); `conv_state`: (B, D, STATE_LEN); `weight`: (D, 4); `cache_seqlens`: (B,), int32
- Returns: (B, D). `conv_state` is updated in place.


### Python API Overview
- Autograd-enabled (recommended):
  - `causal_conv1d(x, weight, bias=None, activation=None) -> (B, D, T)`
    - `activation ∈ {None, 'silu', 'swish'}`
  - `short_conv_fused(x, weight, bias=None, attention_mask=None, activation=True, residual=True) -> (B, T, D)`
  - `short_conv_update(x, conv_state, weight, bias=None, cache_seqlens=None, activation=True, residual=True) -> (B, D)`
- Forward-only (low-level wrappers):
  - `causal_conv1d_fwd(x, weight, bias=None, silu_activation=False) -> (B, D, T)`
  - `short_conv_fused_fwd_only(...)`, `short_conv_update_fwd_only(...)` (forward-only aliases)

Common constraints:
- All tensors must be on the MPS device and share the same dtype
- Only width `W=4` is supported currently
- Tensors must be contiguous
- Some advanced parameters (e.g., `seq_idx`, `initial_states`, `return_final_states`) are not implemented on MPS


### Benchmarks
Run the basic and common scenarios:

```bash
python bench.py --dtype bf16
```

The script does warmup and multiple measured runs; it may take a while. Outputs include MPS/reference times, speedup, and stability statistics per configuration.

Example results:

```text
python bench.py
🚀 Causal Conv1D MPS performance test
Config               MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
--------------------------------------------------------------------------------
1×64×128×4           0.00       0.03         8.83       17.64           ✅
2×128×256×4          0.00       0.04         8.18       18.71           ✅
4×256×512×4          0.01       0.03         3.42       21.24           ✅
1×512×1024×4         0.01       0.04         3.33       13.73           ✅
8×64×128×4           0.00       0.03         7.06       72.24           ✅

🔥 SiLU activation function performance test
Config               MPS+SiLU(ms)   PyTorch+SiLU(ms)  Speedup    MPS_StdDev(%)
------------------------------------------------------------------------------------------
2×128×256×4          0.00           0.05              9.58       53.09

📊 Performance test completed!
💡 Tip: Speedup > 1.0 means MPS is faster. StdDev(%) smaller means more stable test results.

🧪 Canon scene (B,T,D interface) benchmark
Config                   MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
------------------------------------------------------------------------------------------------
B1 T128 D512 W4          0.01       0.05         4.39       9.64            ✅
B2 T256 D768 W4          0.02       0.05         2.16       3.29            ✅
B4 T512 D1024 W4         0.09       0.14         1.59       1.65            ✅

🧪 Scene (Optimized Fused): CanonA/C (B,T,D + Fused Kernel)
Config                       MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
--------------------------------------------------------------------------------------------------------
B2 T256 D768 W4              0.02       0.07         3.98       6.45            ✅
B4 T512 D1024 W4             0.06       0.32         5.29       1.62            ✅

🧪 Scene (Optimized Fused): CanonB (QKV concat + Fused Kernel)
Config                                   MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
----------------------------------------------------------------------------------------------------------------------
B2 T256 H12 KV4 hd64 W4                  0.03       0.11         3.96       2.21            ✅
B2 T512 H16 KV8 hd64 W4                  0.07       0.32         4.81       2.02            ✅

🧪 Scene (Optimized Fused): CanonD (MLP Gate&Up concat + Fused Kernel)
Config                              MPS(ms)    PyTorch(ms)  Speedup    MPS_StdDev(%)   Correct
-----------------------------------------------------------------------------------------------------------------
B2 T256 H768 I2048 W4               0.06       0.35         6.06       3.33            ✅
B2 T512 H1024 I4096 W4              0.20       1.20         6.01       3.63            ✅
```


### Tests
```bash
python test.py
```


### License
MIT


