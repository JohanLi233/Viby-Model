# 研究导航与证据范围

本目录同时保存当前机制契约、特定运行的结果和旧架构研究。
开始工作时用 [AGENTS.md](../AGENTS.md) 定位源码，用
[开发指南](../docs/DEVELOPMENT.md) 选择检查，用
[实验协议](EXPERIMENT_PROTOCOL.md) 记录新实验。
本文是导航；当前开关、路由与兼容性以源码和实际解析配置为准。

## 架构与训练

| 文档 | 用途 | 阅读边界 |
| --- | --- | --- |
| [受保护 PSR](VIBY_PSR.md) | 输出修正、梯度隔离、训练与 engine 状态 | 机制与历史验收不证明验证集提升 |
| [循环 CED](CED_RECURRENT.md) | 残差提升、锚点读取、分轮缓存、对照 | 可选主干变体，engine 兼容性需查当前入口 |
| [QB 路由](MOE_QB_ROUTING.md) | 路由偏置、跨调用统计、累积窗口更新 | 改变训练配方，不是等价 kernel 优化 |
| [TailSFT](TAIL_SFT.md) | SFT 默认损失、初始策略缓存、筛选和恢复契约 | 机制验证不证明覆盖率或后续 RL 收益 |
| [DSpark 引擎](DSPARK_ENGINE.md) | 草稿、主干验证、拒绝修正、回滚 | 需要 MTP 权重；不保证吞吐收益 |

模型库默认值与训练入口默认值分开读取：[model/config.py](../model/config.py)、
[trainer/config.py](../trainer/config.py)、[trainer/utils.py](../trainer/utils.py)。
测试文件存在仅表示有检查入口；旧实验被移除后，相关测试也可能需要随实现迁移。

## Kernel 与性能

| 文档 | 用途与时效 |
| --- | --- |
| [2026-09-12 kernel 工作](KERNEL_SPEED_20260912.md) | 对应日期、配置和运行目录的优化结果；默认配方变化后不能直接外推 |
| [80% MFU 续轮收尾](KERNEL_GOAL_20260912.md) | 默认 PSR 下的候选、基准状态修复与未达成项；新增候选默认关闭 |
| [循环 CED 性能优化](CED_PERFORMANCE_20260912.md) | mask/位置感知融合、元数据复用与完整训练窗口；明确 BF16 差异及无效 B4 测量 |
| [循环 CED 第二轮尝试](CED_PASS2_20260912.md) | 三个默认关闭候选、106 项检查；组件计时受漂移/竞争影响，未确认新增提速 |
| [MoE / 训练状态修复](MOE_TRAINING_REPAIR.md) | optimizer snapshot、A/A 分组和后续测量修正；先读再引用旧窗口结果 |
| [CSA2 验收](CSA2_KERNEL_ACCEPTANCE.md) | 2026-09-11 的完整栈记录；窗口结果受后续 snapshot 修复的证据限制 |
| [MoE 数据流](MOE_DATAFLOW_PR.md) | host / Metal 分层验证及独立开关；后续更正见上行修复记录 |
| [Key-owned dKV](KEY_OWNED_DKV.md) | 独立反向实现和当时的数值/性能边界 |
| [稀疏 VJP 崩溃](SPARSE_ATTN_SIGSEGV.md) | custom-VJP array 叶子槽位问题及历史复现 |
| [Sinkhorn fusion](SINKHORN_KERNEL.md) | 融合实现与当时测试/测量，不能当作当前全仓验收 |
| [MLX 性能经验](MLX_PERF.md) | 惰性求值、编译、内存与旧优化经验；硬件数值和路径需重核 |
| [旧 kernel 算法规格](KERNEL_OPTIMIZATION_PLAN_LUNA.md) | 文件名保留兼容旧引用，正文为特定任务的算法设计规格；不作为当前执行规则 |

## 历史研究

以下资料用于解释旧实验，不作为当前训练启动模板：

- [旧 optimizer 实验协议](archive/OPTIMIZER_EXPERIMENT_PROTOCOL.md)：保留原命令、
  参数和失败教训；其中 KDA / latent MoE 参数与部分脚本属于旧架构。
- [优化器研究汇总](OPTIMIZER_RESEARCH.md)、[谱理论](SPECTRAL_THEORY.md)、
  [论文](PAPER.md)、[实验表](experiments.tsv)：按 run、版本、数据和步数口径读取。
- [HRM / MoE](HRM_MOE.md)：历史架构研究。
- [autoresearch-mlx](../autoresearch-mlx)：独立研究目录，使用自身环境与约定。

新增结果文档应写日期、代码状态、完整配置、执行命令、产物路径、对照方式与
未验证部分，再加入本索引。不要把历史单次最优值覆盖成新默认基线。
