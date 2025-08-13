## causal-conv1d-mps

Apple Silicon 上基于 Metal/MPS 的 PyTorch 因果一维卷积（CausalConv1D）实现，提供高性能前向/反向计算与通用短卷积场景的融合核（Fused ShortConvolution），并支持单步增量更新（single-token update）。

- **后端**: Apple Metal Performance Shaders (MPS)
- **语言**: Python + C++/Objective-C++ + Metal Shaders
- **支持精度**: fp32 / fp16 / bf16
- **内核限制**: 当前卷积核宽度固定为 4（width=4）


### 特性
- **高性能 CausalConv1D**: MPS 后端，原生 Metal 计算，针对 (B,D,T)/(B,T,D) 两种布局做了优化。
- **融合短卷积内核**: 支持一次性完成 Mask 应用、卷积、SiLU 激活与残差连接，适配常见 Canon/ShortConvolution 典型路径。
- **自动求导**: 提供 autograd 函数封装，支持反向传播；基础卷积提供经优化的两段式反向（O(W) 复杂度）。
- **单步更新内核**: 面向推理的状态化短卷积更新 `short_conv_update`，支持就地更新缓存状态。


### 环境要求
- macOS（Apple Silicon，推荐 macOS 13+）
- Python ≥ 3.10
- PyTorch ≥ 2.1（需支持 MPS）
- Clang/Xcode 工具链（用于编译扩展）
- Ninja ≥ 1.11（构建加速）

如果 `torch.backends.mps.is_available()` 为 False，请参考 PyTorch 文档开启 MPS 支持。


### 安装
在项目根目录执行：

```bash
uv pip install -e .


> 说明
> - 扩展在运行时需要加载 `causal_conv1d.metal` 源文件。默认会优先使用环境变量 `CAUSAL_CONV1D_METAL_PATH` 指定的路径；若未设置，会尝试从仓库根目录或当前工作目录查找同名文件。
> - 若在非源码目录中使用本包，请确保设置：
>
> ```bash
> export CAUSAL_CONV1D_METAL_PATH=/absolute/path/to/causal-conv1d-mps/causal_conv1d.metal
> ```


### 快速开始

#### 基础因果一维卷积（带自动求导）
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

#### 融合短卷积（Canon/ShortConvolution 路径）
```python
import torch
import causal_conv1d_mps as ccmps

B, T, D, W = 2, 256, 768, 4
x = torch.randn(B, T, D, device='mps', dtype=torch.float32)
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)
mask = torch.ones(B, T, device='mps', dtype=x.dtype)  # 可选

y = ccmps.short_conv_fused(
    x, weight, bias, mask,
    activation=True,   # SiLU
    residual=True      # 残差: y += x
)
```

掩码将被标准化为 `(B, T)`，支持以下形状：`(B, T)`, `(1, T)`, `(B, 1)`, `(T, B)`，或长度为 `T/B/(B*T)` 的 1D 张量。


#### 单步增量更新（推理）
```python
import torch
import causal_conv1d_mps as ccmps

B, D, W, STATE = 2, 512, 4, 8
x = torch.randn(B, D, device='mps', dtype=torch.float32)
conv_state = torch.zeros(B, D, STATE, device='mps', dtype=x.dtype)  # 将被就地更新
weight = torch.randn(D, W, device='mps', dtype=x.dtype)
bias = torch.randn(D, device='mps', dtype=x.dtype)
cache_seqlens = torch.zeros(B, device='mps', dtype=torch.int32)  # 追踪各 batch 的当前位置

y = ccmps.short_conv_update(
    x, conv_state, weight, bias, cache_seqlens,
    activation=True, residual=True
)
# 使用后请自行递增 cache_seqlens
cache_seqlens += 1
```

要求：
- `x`: (B, D)；`conv_state`: (B, D, STATE_LEN)；`weight`: (D, 4)；`cache_seqlens`: (B,), int32；
- 返回：(B, D)。`conv_state` 将被就地更新为新状态。


### Python API 概览
- 自动求导（推荐）：
  - `causal_conv1d(x, weight, bias=None, activation=None) -> (B, D, T)`
    - `activation ∈ {None, 'silu', 'swish'}`
  - `short_conv_fused(x, weight, bias=None, attention_mask=None, activation=True, residual=True) -> (B, T, D)`
  - `short_conv_update(x, conv_state, weight, bias=None, cache_seqlens=None, activation=True, residual=True) -> (B, D)`
- 仅前向（低层封装）：
  - `causal_conv1d_fwd(x, weight, bias=None, silu_activation=False) -> (B, D, T)`
  - `short_conv_fused_fwd_only(...)`、`short_conv_update_fwd_only(...)`（等价前向别名）

通用约束：
- 所有参与计算的张量需位于 MPS 设备，且 dtype 对齐；
- 当前仅支持卷积核宽度 `W=4`；
- 张量需为 contiguous；
- 某些高级参数（如 `seq_idx`、`initial_states`、`return_final_states` 等）在 MPS 版本中未实现。


### 基准测试
运行基础与常见场景基准：

```bash
python bench.py --dtype bf16
```

脚本会进行较充分的预热与多轮统计，可能耗时较长。输出包含每配置的 MPS/参考耗时、加速比与方差指标。

示例结果：

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


### 测试

```bash
python test.py
```


### 许可证
MIT
