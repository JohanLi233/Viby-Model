# HRM 候选架构（2026-08-16 新方向）

## 结论摘要

放弃"训练省/推理 10x"的循环展开路线。新的主候选是 **HRM 双状态层次循环**，
与 HuggingFace transformers 5.15 内置的 HrmText（Sapient AI，论文
`HRM-Text: Efficient Pretraining Beyond Scaling`, arXiv 2605.20613）对齐。

该架构满足用户的硬约束：**训练前向 = 推理前向，推理不额外展开**。

## 机制（参考 HF modeling_hrm_text.py 实现）

语义澄清（2026-08-16 review）：
- z_L 在每个 token 的前向开始时从零初始化，z_H 从该 token embedding 出发；
  二者都是 depth-state，不在序列维传播。跨 token 记忆全部由每个循环调用
  独立的 KV cache slot 承载。因此更准确的描述是：每个 token 做一次带注意
  力记忆的定点迭代，而不是经典 sequence-level RNN。
- z_L 在同一个前向内跨 H cycle 继承、不重置；层次性只有在 H>=2 时才出现。
- HF embedding_scale=50 的前提是 embedding 初始化 std=0.02；MLX 的
  nn.Embedding 初始化 std=1/sqrt(hidden)，因此等价 scale = sqrt(hidden)。
  scale=1.0 是错误操作点，只能作为消融臂。

- 两个独立的 transformer stack：
  - L_module：快状态 z_L
  - H_module：慢状态 z_H
- 每个 token 的前向：
  - z_L <- L_stack(z_L + z_H)，连续 L_cycles 次
  - z_H <- H_stack(z_H + z_L)，每个 H cycle 一次
- 每个 stack 的真实层数为 P（= VibyConfig.num_hidden_layers）。
- 每 token 层求值次数 = H_cycles * (L_cycles + 1) * P，训练/推理完全一致。
- 每次循环调用的 attention K/V 有独立 cache slot（flat past_key_values）。
- L_bp_cycles 梯度路由：每个 H cycle 只有尾部若干 L cycle 回传梯度，
  与 HF 默认一致；小尺度筛查暂用全部回传。
- 参考实现还有：无参数 RMSNorm、attention 输出 sigmoid gate、
  embedding_scale = 1/initializer_range。Viby 移植第一版保留加权 RMSNorm
  与现有 MLA/attn_gate/value_res/MTP 配方，仅新增 H/L 双状态循环。

## 已完成的移植

- `VibyConfig`：`hrm_H_cycles` / `hrm_L_cycles` / `hrm_bp_cycles` / `hrm_emb_scale`
- `model/model.py`：`VibyStack`；`VibyModel` 在 hrm_H_cycles>0 时构建
  L_module/H_module，并按 H/L 循环执行；支持 prefill/chunk/decode cache、
  doc_mask、padding、stop_gradient 梯度路由。
- CLI：`--hrm_H_cycles --hrm_L_cycles --hrm_bp_cycles --hrm_emb_scale`
- 测试：test_consistency.py 6/6 PASS（含 HRM 因果性/分段/decode/padding）。
- 小模型梯度 smoke：loss 0.944 -> 0.008（20 步 Adam），compile 通过。

## 实验

| run | 配置 | 层求值/token | N | 对照 | 状态 |
|---|---|---|---|---|---|
| r040 | HRM 576 P=2 H=1 L=2 @0.1B | 6 | 26.2M | r030 dense 512x6（6 evals, 28.5M）：iso-FLOPs, iso-N-ish | 训练中 |
| r041 | HRM 512 P=3 H=1 L=2 @0.1B | 9 | 28.5M | r030：严格 iso-N，1.5x per-token FLOPs | 排队 |
| r042 | HRM 768 P=1 H=2 L=3 @0.1B, emb_scale=27.71, bp=2 | 8 | 28.46M | 忠实 HF 操作点；iso-N，~3.0x per-token FLOPs | 排队 |
| r043 | HRM 592 P=2 H=2 L=1 @0.1B, emb_scale=24.33, bp=1 | 8 | 27.63M | 最小层次/快速探针；~1.78x per-token FLOPs | 排队 |

判据：holdout CE vs r030=3.5819；差 >0.05 nats 采信。
注意 r040/r041（H=1 或 scale=1.0）是退化/消融臂，不是 HRM 主判据。

## 最终判定（2026-08-16 22:50）

r042 忠实操作点在 step1900 被人工终止：
- same-D step1900 train main：r042 2.469 vs dense r030 2.434（+0.035 nats）
- r042 per-token FLOPs = 3.0x dense
- holdout CE @58M tokens：3.7889（dense 满 0.1B 为 3.5819，且总 FLOPs 更少）
- 同总 FLOPs 口径差距更大（约 +0.6~0.75 nats）

结论：HRM 路线在 28M/0.1B 尺度关闭。停止 r043。转向 ARCH-SEARCH 协议。
