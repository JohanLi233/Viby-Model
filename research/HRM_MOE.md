# HRM-Text-MoE 设计（r070，2026-08-17）

## 动机

r042 已在 28M/0.1B 尺度关闭纯 dense HRM 路线（同 D 追平 dense 但 3× per-token
FLOPs）。但那次失败有一个结构性原因没被攻击：**循环引理**
（PROPOSAL §3.3：`F_looped(k,N) ⊆ F_untied(k·N)`）——同一组权重迭代 k 次，
表达能力严格不超出 k 层非共享网。r042 的 8 次层求值只是"同一个 1 层网
原地转 8 圈"，循环没有买到有效深度。

MoE 恰好拆掉这个上界：循环第 c 步路由到的专家子集与第 c' 步不同 →
**每次迭代的有效权重并不同一**，looped net 不再是 untied net 的真子集。
同时 MoE 把参数量与 per-token 计算解耦：专家池做大（参数到 100M）不花
吞吐，循环次数（有效深度）也不花参数。这是 HRM × MoE 的互补性：
**循环提供深度，MoE 提供"每步不同的权重"**。

## 架构（r070 基线臂）

- 主干：h=768，8 头 MLA（kv_lora 192，qk_rope 32），与 r060 同配方。
- HRM：P=1（num_hidden_layers=1），H=2，L=3 → 每 token 8 次 stack 求值
  （r042 的忠实 HF 操作点），bp_cycles=2（尾部 2 个 L cycle 回传），
  emb_scale=√768=27.71。z_L/z_H 双状态、KV cache 每循环独立 slot，
  全部复用现有 `VibyStack`/HRM 前向（已过因果/prefill/decode/padding 测试）。
- MoE：L-module、H-module、MTP 块全部为 MoE 层（n_dense_layers=0）；
  64 路由专家 top-6 + 1 共享，moe_in 104，scaling 2.5（参数量打印后
  用专家数校准到 ~100M）。
- Engram：挂 L-module 唯一位点（layer 0），每循环重读；MTP depth 1 保留。
- 注意力：稠密 MLA。

## 创新点（全部零初始化 ⇒ 初始严格等价现有 HRM）

1. **CycleRouter（核心）**：每个 router 增加 `cycle_emb (n_cycles, hidden)`，
   router 输入 = x + cycle_emb[c]（第 c 次迭代的嵌入）。专家按迭代特化——
   "草稿专家/打磨专家"分化有可解释性。零初始化 ⇒ 初始严格等价；嵌入
   形式经 sigmoid 分到 w 回传梯度（logit 偏置形式因离散选择+stop_gradient
   不可微，已否决）。参数 8×768/gate，进 router 的 AdamW 小 lr 组。
2. **CycleFiLM**：每次 stack 调用前对注入状态做 per-cycle FiLM
   （scale/shift 零初始化），给每个 cycle 显式"时间身份"，打破迭代同质性
   （loop_k 路径已有同机制，HRM 路径补上）。
3. **跨循环负载统计**：router 的 last_load 从"末次调用覆盖"改为
   "整次前向累加"，bias 均衡项看到全部 8 次迭代的真实负载。

## 吞吐预算

每 token 8 次层求值 ≈ r060 的 8 层；专家 64+
→ 均衡桶 C 更小。bp=2 使反向驻留 < r060。预期 ≥ r060 的 11.3K tok/s；
不达标的调节旋钮：L=2（6 次求值）、专家数、moe_in、bp_cycles。

## 对照口径

与 r060 同数据（pan/text 全量、packed+docmask、seq2048）、同优化器、
同 bs6×accum2。判据：holdout CE vs r060；吞吐 ≥ 11.3K tok/s。

锚点存档（r060/r061 代码与 checkpoint 已于 2026-08-18 精简删除）：
- r060 V4-MoE+DSA（99.54M/33.5M 激活，全程 12,022 tok/s）：
  holdout CE **1.9738** / PPL 7.1977（完整 epoch 27800 步）
- r070 结果：CE 2.1957 / PPL 8.9865（提前收尾 @step21999，77% epoch，
  lr 未衰减到底；同 step 训练 main loss 2.266 vs r060 2.388 领先 -0.12，
  但 holdout 落后 +0.22 —— HRM-MoE 在此尺度 iso-参数/iso-激活下
  未展现优势，与 28M 时代 r042 结论方向一致）。

## 消融臂（排队）

- r071: hrm_cycle_router=0（无 CycleRouter，纯 HRM+MoE）
- r072: H=2 L=2（6 次求值，更快档）

## 实现状态（2026-08-17）

已落地：`VibyConfig.hrm_cycle_router/hrm_cycle_film`；`MoEGate.cycle_emb`
（router 输入叠加，可微）；`VibyModel.hrm_film_scale/shift`（stack 输入
per-cycle FiLM）；负载统计跨循环累加 + 每次前向开始重置；MTP 块 router
复用末位 cycle 槽位（保梯度通路）；融合 decode kernel 在 cycle 模式下
自动旁路（与训练路径一致）；CLI `--hrm_cycle_router/--hrm_cycle_film`；
muon.py 排除 `hrm_film`（cycle_emb 随 ".router." 进 0.05× AdamW 组）。
回归 test_consistency 9/9（新增 test_hrm_moe_cycle：参数审计/零初始化
等价/梯度可达/通路接线/负载累加五项）。

基线臂定稿：**E=112（总参 100.78M），H2 L3 P1，bp=2，emb_scale=27.7128**，
激活 ≈33M/token ≈ r060 同口径。queue：experiments/queue_hrm_moe.sh。
等 r060 完成后跑 bench_hrm_moe.py 对齐吞吐再启动。

## 路由均衡事故与修复（2026-08-17 探针 A-D）

**事故**：首个真实数据 soak（bias=0.001 + router_lr=0.05×，即 r060@E=32
的稳定点）吞吐 14.8K→3K 单调衰减、swap 持续增长。埋点
（trainer VIBY_DEBUG_MEM=1：active/cache/peak + 分 gate 负载 + 逐次
调用桶容量 `_c_max_seen`）实锤：**E=112 细粒度路由下 L/H/MTP 三个
router 的 top-1 桶容量 C 全部单调塌缩**（6.7K→22K+/160 步），
(E,C,D) 稀疏桶缓冲（`_sparse_forward`，C=最大桶对齐 512）随 C 膨胀，
单步峰值 29.6G→52G 超过 48G 物理内存 → swap 拖垮。非泄漏（步间
active 稳定 ~0.8G）、非缓存问题（cache 钉在 20G 上限正常工作）、
**与 CycleRouter 无关**（探针 B 关 cycle_emb 对照臂同样塌缩，含单调用
的 MTP gate——证明是 E=112 + 原均衡强度不足，不是 HRM 循环或新机制）。

**修复**（探针 C/D 验证）：`--moe_bias_update_rate 0.005` +
`MOE_ROUTER_LR_MULT=0.01`。结果：峰值钉死 38.74G 全程不动，稳态
C≈3-4K（偶发 10K 尖峰会被 bias 控制器拉回），瞬时吞吐 ~12.1K tok/s
持平 r060 真实数据口径，swap 不增。结论：专家数 32→112（均值负载
2304→658）后，符号式 bias 的固定校正带宽 0.001/步 压不住路由漂移，
需同步放大；router 权重 lr 也要同步放慢。r070 已按此配置启动。

**经验**：MoE 的均衡超参（bias 更新率、router lr）随专家数/每专家
token 均值变化，换 E 必须重新标定；`_c_max_seen` + VIBY_DEBUG_MEM
埋点保留在代码里供后续排查。
