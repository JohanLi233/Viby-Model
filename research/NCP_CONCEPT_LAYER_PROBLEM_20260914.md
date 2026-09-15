# NCP × CED 概念层质量问题：问题陈述与求解请求（2026-09-14）

> 用途：发给外部协作者（GPT Pro）的完整问题包。阅读者不需要本仓库的任何先验上下文。
> 仓库证据：`research/NCP_CED.md`（契约）、`model/ncp.py`、`model/model.py`、
> `experiments/eval_ncp_contribution.py`。训练曲线为 W&B 面板读数（微批级，非窗口均值）。

## 1. 我要做什么

在自研 MLX 语言模型（Viby，mHC/CED 主干，类 DeepSeek-V4.1-Flash 的压缩路由结构）上，
结合 **NCP（Next-Concept-Prediction）** 思想（接口参照 NCP-ArchPreview, arXiv 2609.10715，
**非严格复现**）构建一个概念层：模型每 4 个 token 形成一个"概念"，预测下一个概念，
并把预测/状态反馈回主干，期望概念层成为主干的高层次上下文来源（论文动机是 token
efficiency；本仓库**未主张**任何 efficiency 结论）。

我的判断：**概念层没有做好——它正在退化成"静态上下文向量"而不是演化的概念**。
已有诊断数据支持（见 §4）。请帮忙回答 §7 的问题。

## 2. 架构现状（代码可核对）

### 2.1 概念产生（两版共用，未变过）

- token 流按文档边界切连续段，每 `ncp_stride=4` 个 token 一组；PAD/跨文档重置分组。
- 概念 = **4 个 token 的 encoder 输出做均值池化（FP32）→ LayerNorm（`pool_norm`）**。
  不完整尾组不产概念。概念宽度 = 模型 dim（池化前后同宽）。
- 这就是概念的全部"出生方式"：**无保序、无组内结构、可学习部分仅 LayerNorm gain**。

### 2.2 概念加工与预测（`model/ncp.py`）

- 概念 → `memory_project`(dim→128) → RMSNorm → RoPE，得到 128 维 K=V 概念记忆。
- `ncp_layers=2` 个 ConceptBlock：pre-norm 概念注意力（query 按 head RMSNorm、反向旋转 RoPE）
  + 宽度 2·dim 的 SwiGLU。
- PQ 预测头：第二层输出 → RMSNorm → 线性 → 8 个 segment 各自 softmax over 256 码字 →
  与 codebook（8×256×dim/8）加权混合 → 连续预测向量。
- 反馈：预测 → RMSNorm → 线性 → **乘零初始化标量 `feedback_gate`** → 在组结束位置
  立即可用（k=4 时作用于位置 3 的 logit，保持到位置 6，逐段常数）→ 按
  `signal / sum(pre_mix)` 等量提升到各 mHC stream。
- 注入只发生在 encoder 边界层（12 层模型默认索引 6）。CED 全局 KV 用注入前的干净输入，
  概念**不能**改写 CED 全局 KV；主干 indexer、深度路由、SWA 均不消费概念。

### 2.3 训练目标

- `L = L_NTP + L_MoE + λ·L_NCP + β·L_VQ`，λ=β=1.0（**未经尺度校准的起始值**）。
- L_NCP = MSE（预测_t, 概念_{t+1}.detach())，仅同文档相邻组；**目标永远 detach**。
- L_VQ = 概念到最近码字距离；选择 detach，codebook 由 VQ 和 PQ-softmax 混合两条路收梯度。
- 门控全零时：NTP 到概念分支的梯度被阻断（门自身有梯度），但**辅助损失仍训练
  encoder/概念分支**——即初始前向与"无 NCP"等价，后续轨迹不被保护。

### 2.4 v1 → v2（最近一次"优化"）

| | v1 `ced_shared_kv_v1`（旧 run） | v2 `ced_state_v2`（新 run，当前工作区） |
|---|---|---|
| 层间记忆 | 两层读**同一份原始概念 KV** | 第二层改读 `memory_updates` 从第一层输出**新学的** 128 维刷新记忆（推理时拼接累积） |
| 注入 | 仅边界 PQ 反馈 × `feedback_gate` | 增加状态通路：每层 `state_norm→state_projects`、深度路由（softmax over 2 层）、零初始化 `state_gates`/`prediction_gates`；边界层 + 2/3 深度层两个接入点 |
| 缓存/token | 32 元素 | 64 元素 + 投影状态 |

**v2 没有改变概念的产生方式和任何损失**。62 项定向测试通过（含梯度边界、缓存/engine），
非实现 bug。

## 3. 两个观测 run

- **Run A（新，v2）**：~2.3k 步，进行中。W&B 名 `viby41_n...0.001717`。
- **Run B（旧，v1）**：5.5k 步。W&B 名 `viby41_nc...R0.001717`。
- 两 run 前 ~2k 步在概念侧指标上完全重合；run B 的 gate 在 ~1.5k 步后起飞，
  run A 到 2.3k 步 gate 仍为 ~0。
- 由 pairs≈4000 反推：batch ≈ 16 行 × 每行 ~250 对 ≈ 每行 ~1000 token。

## 4. 诊断证据（症状）

指标口径：`previous_mse` = 用当前概念复制下一概念的 MSE（copy 基线）；
`target_energy`/`zero_mse` = 零预测 MSE（同一量，代码里两项填同一值）；
`relative_mse_zero` = 预测MSE/zero_mse；`relative_mse_previous` = 预测MSE/copy_mse；
`concept_variance` = 概念逐通道方差（去均值后）；均为 detach 的微批统计。

| 症状 | Run A（新, ~2.2k 步） | Run B（旧, 5.5k 步） | 读法 |
|---|---|---|---|
| `concept_variance`/概念能量 | AC 占比 ~0.53 | **~0.20**（variance 0.14） | 概念 80% 能量是跨样本共享的均值分量 |
| `previous_mse`（copy 基线） | ~0.70（从 0.95 缓降） | **0.19** | 相邻概念差异极小，概念"僵" |
| `zero_mse` | ~0.94 | 0.70 | 概念能量缓降（LayerNorm gain 被 weight decay 拉） |
| `relative_mse_zero` | 0.69，**仍在降** | 0.70，**~2k 步后不再改善** | 预测胜过零预测 30%，但早已平台 |
| `relative_mse_previous` | ~1.4–1.5 | 平滑分量算 ~2.6；微批比值 13–20（重尾，分母 0.19 太小） | **预测从未跑赢 copy 基线** |
| `ncp/gate`（feedback_gate） | ~0 | 0 →（1.5k 后）0.135 单调爬升 | PQ 反馈通路开得很晚、很小 |
| `ncp/loss` | 0.98 → 0.65 稳定下降 | →0.44@1.3k 触底**回升**至 0.49 | 旧 run 的回升与 gate 打开/概念变僵同期 |
| `ncp/vq_loss` | 0.65 | 0.44→0.49 | codebook 拟合也变慢/变差 |
| pairs / valid_samples | ~4000 / 16 稳定 | 同 | 监督群体本身健康 |

时间线耦合（Run B）：~2k 步概念变僵 → 预测停止改善 → gate 打开 → aux loss 回升。
整个分支收敛到"一个逐段常数、持续注入的文档均值向量" regime。

## 5. 我的根因分析（请批判性验证）

1. **概念原生表示太弱**：均值池化丢组内顺序/结构，概念天生只是局部嵌入的平滑；
   后续加工救不了原料。
2. **MSE 回归奔向均值**：MSE 最优解是条件均值 E[c_{t+1}|历史]；概念一旦被共享均值
   主导，预测静态化是**损失函数定义内**的结果，不只是训练不足。这解释了
   "胜过零预测却永远输给 copy"的组合。
3. **没有任何梯度奖励多样性**：NCP/VQ/NTP 三个损失全都偏好稳定、低熵、可预测的概念；
   无方差/协方差正则、无重建约束、无判别式目标。变僵是所有梯度的合力。
4. **主干从不"需要"概念**：CED 的压缩成立是因为主路径被迫消费压缩态；这里 indexer、
   深度路由、SWA 都不消费概念，注入是 单标量门 × 线性 × 逐段常数。概念分支是阑尾，
   LM 可以绕过它。v2 加的状态通路读的仍是同一批概念层状态——天花板没变。

## 6. 已排除项

- 非日志 bug：指标映射逐 index 核对过，v1/v2 一致；绿点数值勾稽闭合。
- 非梯度断裂：62 项定向测试覆盖 NCP/VQ 梯度边界、状态单独 NTP 梯度、编译反向。
- v2 的 aux loss 降得慢是**架构变化的预期代价**（第二层输入从零学起，绝对值不可跨架构比）。
- 症状不是尺度错觉：概念 RMS≈LayerNorm gain，能量下降是 weight decay 正常体现。

## 7. 请回答的问题（按优先级）

**Q1 概念表示的出生方式。** 用什么替换"4-token 均值池化"才能既保留组内信息又保持
概念频率的压缩意义？候选：保序 group encoder（4 token + RoPE 的 1–2 层注意力压缩读出）、
多 stride 金字塔、VQ 式硬量化概念。各自的代价与预期信号？

**Q2 预测目标的参数化。** 如何把"下一概念 MSE"换成不奔向均值的目标？候选：
InfoNCE（同文档下一概念为正）、copy-residual 参数化（显式建模变化量 c_{t+1}−c_t）、
更长 horizon（跳一组预测）。哪个最能直接打破"输给 copy 基线"的现状？注意目标必须
保持因果（预测只能用已观察概念）且实现便宜（MLX，每微批预算有限）。

**Q3 信息保持与多样性压力。** 在 NTP 联合训练下，给概念加"重建组内 token"的解码路径
（VQ-VAE 式）和/或 VICReg 式方差-协方差正则，会不会与 NTP 抢容量、把概念拉成
另一种塌缩？权重量级该怎么起步？有没有更便宜的防塌缩信号（例如 per-group 中心化、
码本使用率监控）？

**Q4 让主干被迫使用概念。** 在不破坏 CED 干净 KV 契约的前提下，怎么让 LM 对概念
形成依赖（CED 哲学）？候选：indexer query/深度路由消费概念状态、FiLM 式乘性注入
替代加性残差、gate 的 warmup 日程（aux 先成熟再放行 NTP 梯度）。训练稳定性风险各是什么？

**Q5 最小验证路径。** 上面每个改动，应该先动哪个面板/指标来判定有效？我的现有判据：
`previous_mse`（动态性）、`relative_mse_previous`（>1→<1）、`concept_variance` AC 占比、
最终 `eval_ncp_contribution.py` 的 normal/off/swap NTP CE。这个诊断体系还缺什么
（我已知缺 state_gates/prediction_gates 日志）？

## 8. 硬约束（建议必须能落地）

- 框架 MLX（Apple Silicon），custom VJP 要求每个 primal 对应一个梯度叶子；
  编译/eager、prefill/decode、BF16/FP32 路径并存。
- engine 契约：概念 KV 按行搬运、prefix snapshot、连续 batching、DSpark 拒绝恢复；
  结构签名不匹配必须在主干执行前报错。
- 架构/损失变更 = 显式研究变体：新增 config 开关、定向测试、新 run 显式 reset
  optimizer；不允许把配方差异表述为等价优化。
- gate 零初始化是设计（初始前向 ≡ 无 NCP），改动注入语义需保持可对照。
- 训练信号周期短：几千步内指标必须可分辨；诊断是微批级统计，受 packing 组成影响。
- 概念注意力随概念长度二次增长；每微批算力预算现实。

## 9. 一句话总结

概念层拥有弱原料（均值池化）、均值回归目标（MSE）、零多样性压力、以及与主干
零依赖的注入通路——四面合力把它推向静态上下文向量。目标是：在保持因果、缓存
契约和 MLX 预算的前提下，重新设计"概念是什么、被预测成什么、主干为什么需要它"。
