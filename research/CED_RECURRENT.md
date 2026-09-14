# 残差提升式循环 CED：MLX 实验契约

后续实现与测量见 [2026-09-12 性能优化](CED_PERFORMANCE_20260912.md)：位置感知
融合路径现已覆盖 packed/PAD，并缓存跨轮元数据。下文首次移植与计时记录保留为
历史阶段，后续性能比较以其单独记录的源码/配置为准。

这是基于 `ffa4ba877236fc30f244110e57b33c8cf50f5beb` 研究提案的可选主干路径。
它把中间 decoder 的工作重新分配给锚点；既有 protected PSR 的 detached
logits residual 仍是另一种实验。两者不能同时开启。此路径没有新增可训练模块、
词表头、VQ、teacher、未来表征 MSE 或停止器。机制实现和结构测试不证明 CE、
推理能力或真实训练吞吐改善。

## 变换与状态

默认 12 层保持完整 token 层 0–6 和 11，在层 6 执行之后，取每个连续文档
已完成四 token 组的最后一个位置，携带完整 mHC hidden、`pre_mix` 与稀疏选择。
复用层 7–10 的参数执行三轮，最后只提升一次：

```text
Z0 = D(H)
Zq = F^q(Z0; M)
H_new = H + U(Zq - Z0)
```

`U` 向锚点及之后三个位置保持广播，在下个文档或 padding 处截断；首个完整
锚点之前保持 boundary hidden。`D U = I`，所以 hidden 的 `Q=I-UD` 部分保留，
组内 token 与锚点的差分保留。恒等式不保证整个模型信息无损、循环收敛或准确率。
`pre_mix` 直接广播最终锚点值，无锚点处用 boundary 值；不对混合系数作残差相加。

层 6 的投影 evidence KV 与 index K 保持完整 token 分辨率，同次前向内循环只读，
并从 NTP 正常反传到 evidence 投影。MLX 注意力显式区分 query/memory 的位置、
文档和 pad。循环 RoPE 使用原 token 锚点位置；latent SWA 使用实际 token 距离，
窗口条件为 `0 <= query_position - key_position < window_size`。

中段 reuse 层继承选择，reindex 层按既有固定 candidate pool 重新打分。跨轮保留
更新后的选择，输出 token 使用对应锚点最终选择；无锚点处使用 boundary 选择。
所有选择再次受原 token 因果位置、文档和 padding 约束。每次读取保持固定 top-k；
不能把每层全历史 dense 检索作为提升来源。

共享参数不共享 latent self-KV：cache 按 `(round, layer)` 分开，输出层有自己的
token cache，原始 evidence 共享。缓存的执行模式和 stride/rounds 必须一致；切换
模式或回退跨越循环历史时重新 prefill。首次实现的 cached generation 仅支持
无 padding、每行一个文档、相同进度的 batch；packed 文档训练/评估使用无 cache
路径。continuous-batch engine 不支持该缓存，必须明确拒绝，不能静默退回旧路径。

## 配置和对照

```python
cfg = VibyConfig(
    ced_recurrent_enabled=True,
    ced_recurrent_stride=4,
    ced_recurrent_rounds=3,
    n_mtp_layers=0,
)
model = VibyForCausalLM(cfg)
recurrent = model(ids, labels=labels, use_mtp=False)
baseline = model(ids, labels=labels, use_mtp=False, use_ced_recurrent=False)
```

当前 CLI（2026-09-14 起 PSR 研究线已整体移除，无需再显式关闭）：

```bash
.venv/bin/python trainer/train_pretrain.py --ced-recurrent --mtp_depth 0 \
  --ced-recurrent-stride 4 --ced-recurrent-rounds 3 \
  --data_path /path/to/train.jsonl --out_dir out/ced_recurrent
# 配对基线使用同样其余参数，并将 --ced-recurrent 换为 --no-ced-recurrent。
```

这是运行入口，不是已完成的质量实验；本轮没有启动长期预训练。
旧 checkpoint 转换需显式 `--resume /path/to/checkpoint.safetensors --reset_optimizer`，
并选择新的输出目录。相同循环配置可以自动恢复；baseline、stride、rounds 或版本
发生变化时不能静默继承 optimizer/进度。step 权重快照也保存 execution sidecar。

默认关闭。`k=1, q=1` 与 `use_ced_recurrent=False` 都直接执行旧中段路径，避免
`H+(F(H)-H)` 的浮点舍入破坏逐元素回退。权重结构不变；checkpoint 配置必须保留
循环模式元数据。对照同时关闭 MTP，不能只在新方法删除 MTP 后计算节省量。当前
实现拒绝循环与 MTP/PSR 同开，以及不符合共享完整 evidence 假设的层布局。

MoE 的每次物理调用都计入统计。路由负载按有效 query 跨轮求和；每轮序列辅助损失
按有效 query 归一化、忽略全空行，然后对同一个逻辑层的所有轮取平均，再按原层级
系数求和。这保留每层的原辅助损失系数，避免共享三轮把该系数放大三倍。
QB 采样限额在 accumulation 和 rounds 间分配，所有轮样本拼接。不同层样本长度
以全 NaN 行表示缺席；仅循环路径的 QB 更新过滤这些显式 padding 行，真实非有限
观测仍拒绝更新。无锚点中层负载/辅助损失为零，不沿用上一批副作用。

## 预算与测量

有效 token 总数 `T`，锚点数 `A=sum_doc floor(length_doc/k)`。原中段逻辑更新
`4T`，新中段 `4qA`，`k=4,q=3` 时不超过 75%。MLX 静态 shape 为各行保留
`floor(sequence_length/k)` anchor 槽；packed 短文档产生无效槽时，padding 仍可能
消耗矩阵工作。因此有效锚点数、分配容量和真实耗时必须分开报告。

验收条件仍是：

```text
q * C_middle(A, full_memory, fixed_read_budget) + C_lift_and_extra
    <= C_middle(T, full_memory, original_read_budget)
```

不能从逻辑更新减少推断实际 FLOPs 或 wall-clock 减少。真实训练还包括反向、
共享参数梯度累加、MoE 统计/更新、mask、cache 和内存搬运。锚点会集中执行循环，
严格逐 token 最坏延迟不增加也不是本方案的性质。

`experiments/ced_recurrent_probe.py` 提供同一模型、同 token/label、同进程的 ABBA
计时。预热后每次 `mx.eval()`，分别报告 forward 和 forward+backward，保留 MoE
图输出防止编译器删除统计。两边关闭 MTP/PSR/Engram，且不更新参数；计时
包含实际 loss 图、mask、路由和提升，排除 optimizer/data/distributed/cache。
它不等同于完整训练 step benchmark，更不建立同 FLOPs 质量优势。

```bash
.venv/bin/python -m pytest tests/test_ced_recurrent.py -q
.venv/bin/python experiments/ced_recurrent_probe.py --compile --lengths 16 64 --output /tmp/ced-recurrent.json
# 大配置有明显内存与耗时要求；按显式需要运行：
.venv/bin/python experiments/ced_recurrent_probe.py --preset default --compile --lengths 1024 --output /tmp/ced-recurrent-default.json
```

## 固定候选池否证与质量实验

probe 的随机指针诊断使用随机排列和随机标签，比较全 memory oracle 与固定有限
候选池的完整指针链覆盖率。起始键保证在候选池，其余键独立随机；后继证据不在
池内时，重复读取同池不能恢复它。这是结构性瓶颈诊断，**不是已训练 Viby 的
路由准确率或多步推理分数**。

实际 candidate IDs 诊断另外由 `experiments/ced_pointer_pool.py` 提供：用 MLX
边界的真实候选池，随机排列指针事实与独立随机标签，覆盖四种 query 相位。
它只统计必要的指针值/标签位置是否在固定候选池，惰性执行只求所需路由输出，
没有用它计时或声称等 FLOPs。encoder、其他 latent 与最终 token 层还能传递信息，
因此该覆盖率不是整个模型准确率的上界。已训练模型的预测正确率仍须另测。

质量验收使用相同初始化、token 顺序、packing、实际 LR 与日程、相同 optimizer
和 MoE 设置，在独立验证集比较固定 token 和固定实际 FLOPs 的 CE 曲线。还需
随机多跳检索/程序执行任务，并在已训练模型上替换中间 latent、冻结后续 read
query、改变跨轮状态传递，观察因果变化；不能把随机初始模型或有限梯度测试当作
有效 reasoning 证明。等 FLOPs 干预要确认无用计算未被编译器删除。

频繁代码分支、符号运算、极短文档与关键信息在锚点后出现是明确风险。不得把
问题末尾额外对齐到锚点后宣称无调度开销；状态范数、latent MSE、早期单轮 loss
都不是成功条件。

## 实际验证记录

2026-09-12，在当前未提交工作区、Apple M4 Max 上实际执行：

```bash
.venv/bin/python -m pytest tests/test_ced_recurrent.py tests/test_ced_recurrent_trainer.py \
  tests/test_ced_recurrent_runtime.py tests/test_ced_recurrent_fused.py \
  tests/test_training_flops.py tests/test_moe_qb.py \
  tests/test_checkpoint_save.py -x -q
# 62 passed in 8.75s
```

包括代数与精确旧路径、三步 AdamW 回退轨迹、文档/padding/所有长度前缀、
FP32/BF16 eager/compiled 梯度、完整 evidence 投影梯度、各共享层梯度、独立阶段
cache 与 B=2 的公开 prefill/decode 接口，以及 K>N 的短前缀。
真实 `BaseTrainer` 的 BF16 compiled 检查执行 4 个微批、2 次 Muon 更新，校验 QB
采样预算、有限参数/偏置和 checkpoint 保存重载后的精确 logits；这是机制验证。

首次 SDPA 路径：同进程同权重 ABBA，BF16（router 保留 FP32），默认宽度/12 层、B=1、MTP/PSR/
Engram 均关闭；2 次预热，3 组 ABBA，每个图 6 次取中位数：

| T | 基线 forward | 循环 forward | 基线 forward+backward | 循环 forward+backward |
|---|---:|---:|---:|---:|
| 256 | 17.84 ms | 23.85 ms | 51.75 ms | 79.06 ms |
| 1024 | 44.96 ms | 53.77 ms | 126.51 ms | 176.22 ms |

原始结果及代码哈希：[final_default_probe.json](../research_runs/ced_recurrent/final_default_probe.json)。
这一阶段的耗时验收失败：T=1024 的前向慢约 19.6%，前向加反向慢约 39.3%。
不能把 75% 中段逻辑更新宣称为硬件收益。复用层已改为只校验选中证据的元数据，
latent SWA 直接枚举实际跨度内的至多 ceil(W/k) 个历史槽，避免反复构造/排序 Q×Q
可见矩阵；共享栈更多的串行物理调用及 gather/反向成本仍须继续优化。
这一阶段的 B=4 追加测量遇到另一组并行内核 benchmark，主动中止，没有生成验收结果。

随后接入仓库已有 Metal indexed attention / fused VJP，复用它支持 Q≠N 的接口，
不改动该 kernel 的可见性或数学运算。每轮仅在 reindex 时重建稀疏元数据，后续
reuse 层直接消费。新 `VIBY_CED_RECURRENT_SPARSE=0` 可回到显式 gather/SDPA。
该快路只用于无 padding 的单文档整段前向；显式文档/padding 元数据和缓存仍使用
实际位置检查路径。训练器在编译函数之外检查全有效且每行单文档的批次，才移除
冗余 attention 元数据（保留 loss mask）；这次 batch 级同步属于训练器额外开销。

最终融合路径的同权重 ABBA 结果如下。B=1 预热 2 次、3 组 ABBA；B=4 预热
1 次、2 组 ABBA。测量期间对应源码哈希保持一致，统计不含 optimizer/data/cache：

| B/T | 基线 forward | 循环 forward | 基线 forward+backward | 循环 forward+backward |
|---|---:|---:|---:|---:|
| 1/256 | 17.09 ms | 22.16 ms | 50.61 ms | 70.14 ms |
| 1/1024 | 43.23 ms | 47.54 ms | 122.34 ms | 138.06 ms |
| 4/1024 | 135.07 ms | 132.43 ms | 409.65 ms | 393.00 ms |

结果：[fused_default_probe.json](../research_runs/ced_recurrent/fused_default_probe.json)、
[fused_b4_probe.json](../research_runs/ced_recurrent/fused_b4_probe.json)。
B=4/T=1024 前向约少 2.0%，前向加反向约少 4.1%；B=1/T=1024 仍慢 12.8%。
这支持特定矩阵工作量下的小幅收益，不能宣称所有 batch、更长序列或完整训练均加速。
融合专项对 D=64/128、BF16 eager/compiled 的 logits、evidence/共享层权重梯度做了
SDPA 对照；并验证内部 padding 仍走严格位置路径。

长程指针诊断用 512 节点、1537–1540 token、64 个随机问题/相位，保留默认
top-k=64 与候选块 64×8。实际平均候选 token 数 511；能看到 query 的锚点相位
中，1/2/4/8 跳完整必要位置覆盖率为 14.06%/4.69%/1.56%/0%。
结果：[actual_pointer_pool_long.json](../research_runs/ced_recurrent/actual_pointer_pool_long.json)。
这是未训练模型的候选域瓶颈证据；没有验证 held-out CE 或多步推理能力提升。

不得把上述结果与提案中的 NumPy 17 项检查混为一谈。
