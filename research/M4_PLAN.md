# M4 Max 执行计划（v1，执行中）

本文件是 `PROPOSAL.md` / `SPEC.md` 在本机（Apple M4 Max 48GB，MLX，现有 Viby 代码库）
上的可行子集。原则：**不假装能跑 E0 完整版，用 15M–73M × 0.1–0.34B tokens 的本地
网格回答同样的问题，并诚实标注外推边界。**

## 0. 硬件与数据事实

- 硬件：M4 Max 48GB 统一内存；单训练进程实测 ~19k tokens/s @ 73M（bs48×seq640）。
- 数据：`/Volumes/pan/text/pretrain_t2t_mini.jsonl`，1.27M 篇文档，约 343M tokens
  （新切分后 train=343.2M，holdout=尾部 2000 篇=0.8M）。
- 已建立干净切分：
  - train：`/Volumes/pan/text/pretrain_train_mini.jsonl`
  - holdout：`/Volumes/pan/text/pretrain_holdout_mini.jsonl`（训练永不可见）
- 缓存：packed/packedsegs 已生成于 `.cache/`，后续实验启动不再等待。

## 1. 本地可回答 / 不可回答

| 问题 | M4 能否回答 |
|---|---|
| 本地语料的 (E,A,α,β) 在 15–73M、0.1–0.34B 窗口内的估计 | 能（有限精度） |
| 每个架构的 α 是否随 N 变化 | 弱，需要更多 N |
| 乘子方向 (μ_N, μ_D)：循环/深度/宽度的实测 | 能（同 FLOPs 对比） |
| 60M@0.1C 是否追平 600M@12B（论文口径） | **不能直接验证**，只能外推并报告不确定性 |
| B1/B3/B4 探针 | 能（≤10M 模型） |

**本地成功判据改为**：任何新架构在 60M 级、同训练 FLOPs 下，holdout CE 的改善
是否超过基线 seed 噪声（<0.05 nats 不采信），并估计其 (μ_N, μ_D) 投影点。

## 2. 夜间队列（已启动）

| run | 配置 | N | D | 状态 |
|---|---|---|---|---|
| r000 | 768×8 MLA，bs48×640，当前最佳配方 | 73.2M | 343M（全 epoch） | 训练中 |
| r001 | 576×12 MLA | 61.3M | 343M | 排队 |
| r002 | 512×6 MLA | 28.5M | 343M | 排队 |
| r003 | 320×8 MLA | 15.2M | 343M | 排队 |

- 所有 run：packed + doc_mask + value_res + attn_gate + MTP1 + Muon 混合优化器，
  lr=0.01，cosine 到一个 epoch；`save_interval=3255`（=0.1B tokens/检查点，
  bs48×seq640=30720 tokens/step）。
- 结果：训练结束后自动 holdout PPL 评估，写入 `research/experiments.tsv`。
- 中间检查点（0.1/0.2/0.3B）将用于本地 (N,D) 网格拟合；注意 cosine 未对齐
  D 终点造成的调度混杂，最终拟合时标注。

## 3. 已完成的代码改动

- `model/model.py`：新增 `loop_k`（整栈循环）与 per-step FiLM（scale/shift 零
  初始化，loop_k=1 时不创建参数，严格向后兼容）。
- `trainer/config.py`：新增 `--loop_k`。
- `trainer/train_pretrain.py`：把 `loop_k` 传入模型配置。
- `trainer/utils.py`：`loop_k` 纳入 checkpoint 继承白名单。
- 验证：`test_consistency.py` 4/4 PASS（loop_k=1 路径）；loop_k=2 前向/掩码
  路径 smoke test PASS，`step_scale/step_shift` 进入可训练参数。

### ΔW-Loop（步条件低秩权重再生，2026-08-16）

- 机制：loop_k>1 且 dw_rank>0 时，主循环块内 9 个 Linear（注意力 6 + FFN 3）
  获得跨步共享低秩基 U/V 与每步系数表 g∈R^{k×r}：
  W_eff(step) = W + U·diag(g_step)·V，用激活空间等价形式
  `y = xWᵀ + ((xVᵀ)·g_step)Uᵀ` 实现（不重建大权重，推理每 token 增量
  ~2r(in+out) FLOPs，r=8 时 <2%）。V 零初始化 → 初始严格等价基线；
  g 全 1 初始化 → 各步从"共享 LoRA"出发逐步分化。
- 定位：把循环的每步多样性从激活空间（FiLM/TMLT 式）提升到权重空间，
  直接检验"循环引理的间隙有多少可由低秩步特化回收"；FiLM 是其 r→0 退化。
- 参数代价：≈ r·(in+out)+k·r 每矩阵（768×8 配置、r=8、k=8 全量约 +1.1M，
  在 SPEC §6.1 的 ≈3M 调制预算线内）；仅主循环块启用，MTP 块不受影响。
- 改动文件：`model/model.py`（`StepDeltaLinear` + `_linear` 辅助 + 逐层
  step_idx 透传，`dw_rank` config 键）；`trainer/config.py`（`--dw_rank`）；
  `trainer/train_pretrain.py`（传入）；`trainer/utils.py`（继承白名单）。
  eval 两侧经 sidecar `VibyConfig.from_dict` 自动继承，无需改动。
- 验证：`test_consistency.py` 5/5 PASS（新增 `loop2+dw8` 变体跑因果性/
  cache/padding 回归；ΔW 专项覆盖参数审计、V=0 初始化等价（任意 g 不改
  输出）、delta 通路接线确认、LoRA 式启动动力学——V=0 时 dw_u/dw_g 梯度
  恰为 0、仅 dw_v 收梯度）。
- 训练 smoke（320×8、loop_k=2、dw_rank=8、packed+doc_mask+value_res+
  attn_gate+mtp1、compile 开，2 分钟 614 步）：loss 5.96→5.17 正常下降，
  grad_norm 稳定 ~0.65，吞吐 ~21k tokens/s（与同尺度基线持平，ΔW 开销
  不可见）；checkpoint strict 加载往返通过；614 步后 dw_v 已从 0 移到
  max|·|=0.32，dw_g 从全 1 分化到 [-0.64, 2.02]——机制在真实训练中存活。
- 事故与修复（2026-08-16 晚）：r032 首跑（bs48×seq640 真实尺度）爬行卡死。
  微基准定位：ΔW 路径对整个 (B,T,d) 激活做 f32 上采样，长序列大 batch 下
  激活显存推高数倍。修复：大 tensor 全程保持 x.dtype，仅 (B,T,r) 瓶颈过
  f32。修复后真实尺度基准 dw8 vs dw0：每步 +13%、峰值内存持平；
  r032 重跑稳态 17.7k tok/s（FiLM-only 的 0.73×）。
- 并行扩展（用户）：`ws_loop`（W-Scale-Loop，步条件对角缩放
  W_eff=diag(s_out)·W·diag(s_in)，与 ΔW 正交可叠加），同样经
  `StepDeltaLinear` 承载；`test_consistency.py` 6/6 PASS（含 test_ws_loop）。
- eval 侧：`eval_ppl.py --loop_k_override` + `load_model_weights(
  allow_dim0_slice=True)` 支持 H4 推理展开扫描（per-step 参数按第 0 维
  前缀切片/保留初始化），已 smoke 验证。

## 4. 后续队列（按优先级，等 r003 结束）

### 4.0 ΔW-Loop 筛查（进行中，2026-08-16 晚）

~~`experiments/queue_dw.sh`~~ 已跑完 r030/r031；r032 首跑因上述 f32 显存
事故被杀并重跑（修复后吞吐 17.7k tok/s，eta ~90min）。当前执行序列：
r032（重跑）→ `experiments/queue_loop_search.sh`（H4 推理展开扫描：
r031 k=1/2/3/4/6、r032 k=1/2/4 → r033 W-Scale-only → r034 ΔW+W-Scale）。

| run | 配置 | 状态 |
|---|---|---|
| r030 | dense 512×6 @0.1B | done：holdout CE 3.582 |
| r031 | loop ×2 FiLM @0.1B | done：holdout CE 3.493（**比 dense 好 0.089 nats**——同 N 同 D 下循环买了数据效率，但花了 2× 算力） |
| r032 | loop ×2 + dw8 @0.1B | 重跑中 |
| r033 | loop ×2 + ws @0.1B | 排队 |
| r034 | loop ×2 + dw8 + ws @0.1B | 排队 |

判据：r032/r033/r034 vs r031 的 holdout CE 差 > 0.05 nats（seed 噪声线）
才采信；H4 扫描看 k_inf > k_train=2 是否继续降 CE（迭代泛化假设）。
iso-FLOPs 对照：dense r030 全量 ≈ loop 系列 step 1600 处。

### 4.1 R1：循环引理实测（最关键）

| run | 配置 | N | D/FLOPs | 目的 |
|---|---|---|---|---|
| r010 | 基线 576×12（=r001 的 0.2B 检查点或重跑 0.2B） | 61.3M | 0.2B | 对照 |
| r011 | loop：288×12 × loop_k=2 | 61.3M？ | 同 wall/FLOPs | 测 k=2 是否兑现 μ_N |
| r012 | loop：576×6 × loop_k=2 | ~36M | 同 FLOPs？ | 参数更少但同有效深度 12 |

- 控制变量：有效深度一致（12 层）、每 token FLOPs 一致，仅参数量不同。
- 判据：loop 版 holdout CE 落在基线 seed 噪声内 → 循环兑现为“显存收益”；
  若显著更差 → 循环引理的小尺度证据；若更好 → 需要解释的归纳偏置。
- 附加 H4 测试：r011/r012 训练结束后用 eval_ppl 测 loop_k=1..4 的推理展开
  （需要 eval 支持，届时加 `--loop_k_override`）。

### 4.2 R2：宽度-深度分配（E1 的 F 因子）

| run | 配置 | N | 备注 |
|---|---|---|---|
| r020 | 576×12 | 61.3M | r001 复用 |
| r021 | 704×8 | 62.6M | 浅宽，同族 |
| r022 | 512×16 | 63.8M | 深窄 |

三者 N 接近、FLOPs 接近，全 epoch 或 0.2B 预算二选一（看时间）。

### 4.3 R3：E1-lite 乘子方向（30M，D=0.2B）

因子：loop（k=1/2）、MTP（0/1）、packing（on/off）、geometry（512×6 vs 384×12）。
用 2^4 全因子=16 组；每组 30M×0.2B ≈ 1h。只跑与 r002 同一数据前缀，固定 seed。
响应：holdout CE -> (μ_N^D-proj, μ_D^N-proj)。

### 4.4 R4：B1/B3/B4 探针（≤10M）

- P1：合成 key-value 事实注入，拟合 bit/param。
- P2：迭代复合（无条件深度分离）+ S5 状态追踪（条件性），dense vs loop。
- P3：精确复制/联想召回，ctx 512–4096，测 MLA 全注意力与候选 SSM/窗口的边界。

## 5. 账本与记录

- 所有实验记录：`research/experiments.tsv`（run_exp.sh 自动写）。
- PPL 记录：`experiments/ppl_results.tsv`（holdout 口径单独标注，不与旧 HQ
  时代记录混用）。
- FLOP 估算：\(C=6ND\) 主口径 + 训练日志 tokens/s、墙钟时长留档。

## 6. 风险

- 48GB 峰值占用接近上限（r000 峰值 ~50GB），**严禁并行训练**；串行队列已固化。
- 343M tokens 只有一个 epoch，D 轴上界受限；D 方向证据只能到 0.34B。
- 旧 checkpoint（round*）训练过 holdout，所有旧记录与新的干净 holdout 不可直接比。
- 外推到 600M 只作为假设性参考，不作为结论。
