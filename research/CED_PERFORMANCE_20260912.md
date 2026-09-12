# 2026-09-12：循环 CED 的 mask / 元数据性能优化

本轮保持 `k=4,q=3`、相同参数、完整 token 级 evidence KV、原读取预算和 NTP/MoE
目标。比较对象是**本轮修改前的循环 CED 实现**，不是 token baseline，也不是旧 PSR。
没有长期训练或质量结论。

## 实现与回退

- `Attention._recurrent_fast` 在一次 latent 栈内缓存固定 query/memory 的位置、doc、pad
  与 reach。首次采样和每次 reindex 后校验选择，reuse 层复用校验结果。metadata 以
  实际数组身份绑定，query/memory/candidate 改变时重建。RoPE 仅在同一层跨轮复用，
  不假设不同层的 checkpoint RoPE 表相同。
- 候选块 IDs/lengths 先采样为 A 个 anchor，再展开候选 mask；不再先构造完整 T 行
  的候选 mask。最终 token 仍保持锚点选择，无锚点处回退 boundary。
- `indexed_attention` 新增原 token `query_positions` 与 `token_window_size`，前向与
  各反向共享真实跨度、doc/PAD 判断。紧凑选择的宽度 K 与 memory 长度 N 分开，
  避免每轮填充 `[B*A,N]` 选择。完整 evidence 及其梯度保留。
- packed、尾部 padding、内部 padding 间隔可以进入位置感知 Metal 路径，避免显式
  KV gather 及其反向开销。原 token 路径未改；循环的旧实现保留。key-owned 和
  parallel-backward 实验不支持真实位置邻接，位置模式明确采用 sharded 后端。
- masked MoE 负载计数使用整数 histogram/reduction，代替每次调用 k 次 scatter。
  每轮负载与 QB 采样、原辅助损失分母和跨轮平均保持原定义。分数 mask 仍走旧实现。
- 优化训练路径不再先 `.item()` 判断 batch 是否无需 mask；位置感知路径直接处理
  实际元数据。旧路径保留该判断，作为本轮前版本对照。

默认启用；保留本轮前的循环路径：

```bash
VIBY_CED_OPTIMIZED=0 VIBY_CED_MASKED_COUNTS=0 <原命令>
```

`VIBY_CED_RECURRENT_SPARSE=0` 仍可用于 SDPA 数学参考。模型参数、k/q、目标或读取
预算没有借此改变；metadata 缓存不跨 forward，latent `(round,layer)` self-cache
仍各自独立。

## 实际检查

```bash
.venv/bin/python -m pytest tests/test_ced_position_kernel.py -q
# 11 passed in 1.31s
.venv/bin/python -m pytest tests/test_ced_masked_moe_counts.py tests/test_ced_optimized.py \
  tests/test_ced_recurrent_fused.py tests/test_ced_recurrent_runtime.py -x -q
# 32 passed in 29.57s
```

包括 D64/128、Q≠N、紧凑 K、真实 token 间隔、文档/PAD、FP16/BF16 的
eager/compiled VJP、9 个 array primal 的梯度叶子、旧规则的常规位置对拍、
mask 计数/辅助损失梯度与 QB、模型损失/梯度、缓存与 checkpoint。

计数和独立 aux 梯度严格相同；native MoE 的浮点 scatter 存在 A/A 最末位波动，
专家输出与全网络梯度使用单独的 A/A / dtype 容差，不能用它放松整数统计断言。

**数值边界：**masked 路径从 native SDPA 接到已有 BF16-MMA 数值口径，不能声称
逐元素等价或相同学习轨迹。B1 packed 的 scaled loss 差为 0，完整梯度 relative-L2
约 0.0266；两微批更新后参数 relative-L2 约 0.000544，最大绝对差 0.00177（含 router
bias）。QB bias relative-L2 约 0.00874。所有权重、梯度、optimizer/bias 有限，但部分
专家选择/负载和 QB 样本不同。保留旧路径供更严格的训练质量对照。

## 计时边界与结果

脚本：[bench_ced_performance.py](../experiments/bench_ced_performance.py)。Apple M4 Max、
MLX 0.32.2、默认宽度/12 层、BF16（router FP32）、T=1024，MTP/PSR/Engram 关闭。
固定权重、输入、LR、编译模式、MoE bias 和实际 Muon/Sinkhorn/AdamW optimizer。
每次采样前在计时外恢复同一份参数及 optimizer 容器/数组快照。

- f+b：包含 evaluated loss、所有权重梯度、MoE/QB 统计。
- window：包含两次微批、梯度累积、真实 optimizer 更新和下一窗口 QB bias 更新。
- 不含数据生成、snapshot 恢复、数据加载、保存和生成 cache；不宣称测得实际 FLOPs。
- 2 次预热、3 组 ABBA、每槽 3 次，另有首尾 A/A。百分比使用每组配对比率的中位数；
  表中绝对时间为全部样本中位数，二者不混算。

有效的 B=1 结果：

| 输入/计时 | 修改前 | 修改后 | 配对耗时下降 | 首尾 A/A 漂移 |
|---|---:|---:|---:|---:|
| packed，f+b | 173.39 ms | 137.47 ms | 20.66% | 2.79% |
| packed，完整两微批 window | 541.35 ms | 448.18 ms | 17.09% | 0.83% |
| padded，f+b | 179.47 ms | 139.07 ms | 23.34% | 2.62% |

packed f+b 的峰值 allocated memory 为 9.8003 → 8.3657 GB；完整 window 峰值
16.2014 GB，基本不变。普通无 mask 输入约为持平，且有槽间漂移，未确认额外收益。
padded 完整 window 首尾漂移 7.36%，不采用其约 16% 的表面改善作为确认结果。

数据：[default_b1/results.json](../research_runs/ced_performance_20260912/default_b1/results.json)。
源码整体集合检查标为 false：运行中另一任务新增了 `trainer/sinkhorn_pairs.py`；
紧接 B1 运行后的核对显示原文件哈希均未变化，该新增文件不在这次已加载路径中。
原始标志保留，复核记于 [review.json](../research_runs/ced_performance_20260912/review.json)。

## 不采用的测量

- B4 / cache8：首尾 A/A 漂移 37.6%，主动中止。不能拿接近减半的表面 f+b 比率作结论。
- B4 / cache1：f+b 完成后另一 GPU benchmark 启动；完整 window 漂移 170.6%，
  源文件同时发生外部修改。进程在尝试停止前已结束，原始记录保留，**本轮没有
  确认 B4 的完整 window 收益**。

原始目录：[default_b4](../research_runs/ced_performance_20260912/default_b4)、
[default_b4_lowcache](../research_runs/ced_performance_20260912/default_b4_lowcache)。
未停止其他任务，也未把 GPU/内存竞争计为算法收益。

## 复现

GPU 空闲时运行，单独输出目录：

```bash
.venv/bin/python experiments/bench_ced_performance.py --batches 1 --lengths 1024 \
  --layouts plain packed padded --modes fb window --optimizer muon \
  --cache-limit-gb 8 --warmup 2 --blocks 3 --block-iters 3 --aa-iters 2 \
  --run-dir research_runs/ced_performance_reproduce
```

修改前源码及 dirty/untracked 哈希保存在
`research_runs/ced_performance_20260912/before/`、`before_manifest.json`；各运行目录另有
配置、输入哈希、源码哈希、tracked diff 和原始槽位样本。不同优化器、batch、packing
或 dtype 需单独测量。上述性能结果没有验证 CE 或多步推理能力提升。
