# HRM-Text-MoE 设计（r070，2026-08-17）

> **已归档**：本文记录的 HRM 双栈循环、MLA、CycleRouter/CycleFiLM、
> Engram、value-residual 与 V3 rate 控制器已从代码库删除。当前架构见
> README.md（KDA 3:1 + NoPE GQA、SiTU-GLU、独立共享专家、LatentMoE+QB、
> AttnRes key RMSNorm、z-loss = mean(lse²)）。下文是历史研究记录，
> 不要当现行实现读。

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

## CycleRouter v2 / Engram 位置优化（2026-08-18 后）

- **CycleDeltaRouter**（`hrm_cycle_router_rank > 0`）：旧 `cycle_emb` 等价于
  给 router 输入加一个固定偏置，实测对 top-k 选择影响很小（同输入切换
  cycle 的 top-6 重叠 0.96~1.0）。v2 改为 per-cycle 低秩路由增量：
  `W_router(c) = W_router + U·diag(g_c)·V`，V 零初始化保持初始严格等价。
  优化器新增独立 cycle-router AdamW 组（默认 `MOE_CYCLE_ROUTER_LR_MULT=0.1`；
  r070 的 base router lr=0.01× 时相当于 10×），避免旧实现被 router 稳定性小 lr 压住。
- **Engram 初始注入**（`engram_inject_every_cycle=0`，新默认）：HRM 下不再
  每个 L cycle 重复注入同一记忆（r070 为 6 次/token），而是在 token 进入
  循环前注入初始 z_H 一次。记忆对全部 cycle 可见，且在 `hrm_bp_cycles`
  截断下仍处于 z_H 的梯度通路上（不依赖 value_res 泄漏）。
  `from_dict` 对旧 HRM+engram checkpoint 自动设置
  `engram_inject_every_cycle=1` 保持可复现。

## 吞吐塌缩的终极根因与无 host-sync 稀疏桶（2026-08-19）

**事故再现**：r073 正式跑在 ~200 步处吞吐 13.2K→7.6K tok/s 持续劣化
（复现 r070 模式）。上轮修复（bias=0.005 + router_lr=0.01 + 分组稀疏桶 +
合并 eval 拆分）只压住部分：合成基准单步仍 46GB/4.0s。

**根因链**（逐 cycle 埋点实锤，active 从 6.2G 线性涨到 42.8G、cache≈0）：
`_sparse_forward` 的 `counts.tolist()` host sync 迫使前向在
`value_and_grad` **建图期**分段执行——此时 vjp 图尚未构建，所有中间
张量被未完成的惰性图引用钉住，峰值 ≈ 全部前向中间总和（8 cycle ×
~5.2G + 尾部 ≈ 46GB），48GB 物理机进 swap、且桶越大钉得越多（路由倾斜
→ 进一步变慢的正反馈）。旁证：无 tape 大 eval 的 intra-eval 回收正常
（20×600MB 链式临时峰值仅 3.6G）；`mx.checkpoint` 无效（闭包参数无梯度；
显式传参后反向把"重算前向+反向"合并成一次大 eval，峰值反而更高）；
cache_limit 调 4/8/20G 对峰值均无影响。

**修复**（model.py `_sparse_forward` 重写 + trainer 接线）：
- 桶容量不再取本批实测最大值，改用**滚动容量表**：上一微批实测组内
  最大桶 ×5/4+64 余量、128 对齐，按调用槽位（step_idx）存于模块
  （`_cap_table`）；首微批/评估路径用 4× 均值默认。
- 排名/散布/收回全 device 化：row = 组起始 + 组内专家×Cg + 桶内序号，
  一次 scatter 进平铺桶 (ΣEg·Cg+1, D)，GEMM 按 host int 边界切片，
  收回用同一 row gather——整个前向零 host sync。
- 溢出（某专家单步计数超容量，正常余量下不会发生）导入 trash 行、
  输出恒 0：错误有界不崩溃，`update_capacity_table()` 检测后抬升容量。
- trainer 每个微批 `mx.eval(grads)` 后调用各 MoE 模块的
  `update_capacity_table()`（counts 已物化，tolist 不钉图）。

**效果**（合成 bs6×2048 基准，同口径对比）：单步 3.3s→**1.31s**
（9373 tok/s），峰值 46GB→**17.6GB**；fwd(tape) 段 2331→505ms。
塌缩场景的 MoE 单次调用从"swap 悬崖"变为 157ms 的有界 slowdown。
A/B 验证：稀疏 vs 稠密 rel diff ~5e-3（bf16 累加噪声级）；塌缩暖表后
精确一致；容量表随路由恢复自动回落。回归 13/13。

**附带**：decode 侧 CycleDeltaRouter 曾使融合 kernel 整体旁路（每 MoE
调用读全部 112 专家权重）。router kernel 现原生支持低秩增量（host 预算
cz=(x·V^T⊙g_c)，kernel 内 acc += cu·cz），cycle 开/关 decode 差 <4%；
kernel 路径改由 `model.training` 门控（不可微，训练一律旁路——此前靠
`has_cycle`  incidental 保护，cycle 进 kernel 后该保护失效，必须用
training 标志）。旧 cycle_emb 形式仍旁路稠密路径。

## 偏置符号更新的棘轮事故与比例-截断修复（2026-08-19）

**事故**：r073 正式跑（无 host-sync 稀疏桶修复后重启）全程几乎每微批
报桶容量溢出（均值 ~3K 对、峰值 24K 对 ≈ 33% pair 被置零），吞吐
12.9K→6.5K tok/s（step 2500 后企稳但远低于 r070 的 11.3K），大溢出
微批对应 loss 尖刺（3.0→12.1）。容量表按设计每微批抬升到
实测×1.25+64，理论上稳态分布下不可能连续溢出——说明热专家计数
逐批跳动 >25%，即**热点在专家间快速轮换**。

**根因**（旧 checkpoint 权重实锤）：V3 纯符号更新
`b ← b − u·sign(load − mean)` 在 E=112 重尾负载下，低于均值的专家
占多数（~80%）→ 每步多数专家 +u 同向齐步走：
- 均值棘轮无界上涨：h_gate bias 均值 5.7K 优化步后达 **+17.4**
  （sigmoid 分数域仅 (0,1)，共模本身是 gauge 自由度，纯属浪费）；
- 冷专家 bias 近持平（std 仅 0.45）→ 控制器每压下一个热点，负载
  整体跳到另一个 near-tie 专家 → winner-take-most 轮换，容量表永远
  追不上；bias 差分随 √t 扩散逐渐压倒 router 分数，路由质量劣化。

**修复**（model.py `update_moe_biases` + 容量表 + 分组）：
- bias 更新改**比例-截断**：`b ← b − u·clip((load−mean)/mean, ±1)`，
  大失衡时力度与符号式相同，近均衡时校正量→0 自收敛，打破齐步走；
  更新后**零均值投影**（top-k 对共模平移不变，gauge 自由度逐步投影，
  阻止棘轮）。rate 仍 0.005（r070 标定带宽不变）。
- 容量表衰减 0.9→**0.995**（半衰期 ~139 微批）：0.9 时容量追不上
  复发周期 >4 微批的尖峰；慢衰减覆盖复发尖峰，路由集中期容量本来
  就顶在高位，不显著增加期望 padding。
- `_SPARSE_GROUP` 8→**4**：热点专家的连带 padding 减半。
- 偏置更新的负载统计改为**累积窗口内逐微批累加**（原只用最后一个
  微批，噪声大）；溢出改为日志点聚合上报 + swanlab
  `moe/overflow_pairs` 趋势。

旧运行 checkpoint 留存于 `research_runs/r073_hrm_moe_cycledelta_bad1/`
供复查。队列脚本新增 `USE_SWANLAB=0`（本地探针不上报）。

**补记（同日探针复盘）**：上述修复（比例-截断+零均值+慢衰减+组4）的
400 步探针显示吞吐仍逐百步下滑（7098→5000 tok/s），callC 300 步内
涨到 15.2K。对账发现更深的结构性根因：**聚合 gate_max 仅 ~3× 均值
（11.6K/6 calls）而单次调用 callC 达 23×——不同 cycle 的热点专家
不同（per-slot 专用化是 CycleDeltaRouter 的预期行为），聚合到共享
bias 后被稀释掩盖**。共享 (E,) 偏置结构性无解：压下某专家等于在所
有 cycle 同时压它，正确动作（只压该 cycle）不存在；控制器每压下
一个槽位热点，该槽位负载整体迁移到下一个 near-tie 专家，容量表
永远追不上轮换。

**修复 2**：CycleDeltaRouter 下 `expert_bias` 按 cycle 槽位拆分为
(n_cycles, E)——每个 cycle 独立做比例-截断+零均值均衡（不抹平
cycle 间路由差异，只摊平每个 cycle 内部的负载）。配套：collect_stats
按槽位累积 (n_cycles,E) 负载行（未服务槽位行恒 0、err_n 恒 0 自动
无操作）；decode kernel 路由输入取 bias 对应行；from_pretrained 兼容
旧 (E,) 格式自动广播。旧 cycle_emb / 无 cycle 形式保持共享 (E,)。

**基准与动力学探针（同日后续，阴性结果同样重要）**：
- `_SPARSE_GROUP` 扫描（_bad1 集中态真实权重+真实数据）：group 8/4/1
  = 2.06s/2.35s/4.69s 每步。组越小 padding 越少但 GEMM 启动开销主导，
  **维持 8**。
- 轻量 A/B 证伪"单桶更快"假设：均衡态单桶(EG=112) 6.3ms vs 分组(EG=8)
  4.7ms——Metal 对 batch=112 小矩阵 batched GEMM 利用率反而差。
  r070 的 13K 不是 GEMM 结构红利，而是它稳态 callC 3-4K（5-6× 均值）
  vs r073 cycle-delta 的 23×——差距来自集中程度本身。
- 动力学探针（冻结权重、从 _bad1 集中态出发、只跑控制器）：0.995 慢
  衰减使溢出 100 微批内收敛到 0；零均值投影立即修正 +17.4 共模；但
  热点身份每 ~10 微批轮换，控制器带宽（0.005 或 0.02）只能跟不能压。
  **注意此探针是"坏盆地救援"场景，不代表从随机初始化起步的全新
  训练**——新 run 的 router 在均衡器正常工作下成长，不进入坏盆地。

## r073 正式 run 复盘（step3000 检查点）：确定性 top-k 仍失衡 + 三个附带修复

**现象**：r073 正式跑（CycleDelta rank8 + per-slot bias + rate0.02）几乎
每个微批仍报桶容量溢出（数十~数千 pair 被置零），吞吐从 15.5K 持续
下滑到 11.8K；H 槽位与 MTP 槽位的 max/mean 负载比常达 12~18×。

**根因**：per-slot bias 解决了"槽位间热点不同"的聚合稀释，但 bias-only
控制器有一个结构性盲区——**当大量 token 的 top-k 分数排序相同或 near-tie
时，确定性 argpartition 会把整批 token 压给同一小撮专家**；控制器只能让
热点整体轮换（winner-take-most），无法把同一批 token 摊给不同专家。
在 r073 step3000 检查点上冻结权重、只跑 bias 控制器，rate 0.02~1.0 均
不能把 max/mean 压到 1.5× 以下，上限恰好逼近 E/K=18.7×（每 token 都选
同一组专家的退化态）。

**修复 3（训练期 router 抖动）**：`moe_router_noise`——仅训练期在
选择分 `sigmoid(logits)+bias` 上加高斯噪声再 argpartition；路由权重 w
仍取原 sigmoid 分，噪声只决定选谁。同检查点探针：noise=0.1 时
max/mean 从 ~15× 降到 ~1.5-4×，bias rate0.02 下 60 微批累计溢出从
每步数千降为 0（首微批由 4× 默认容量吸收）。评估/decode 不经过噪声，
保持确定。queue 脚本 `ROUTER_NOISE=0.1`。

**附带修复**：
1. **首微批桶容量默认 1.5×→4× 均值**：1.5× 只覆盖近泊松负载，路由
   未热身前首微批必溢出；4× 只贵第一个微批，update_capacity_table
   拿到实测后立即收敛。
2. **engram 余弦门控分母**：旧实现误传 attention qk_dim(128) 而不是
   hidden_size(768)，sigmoid 长期接近饱和。新增 `engram_gate_dim`，
   新训练默认 hidden_size；旧 checkpoint sidecar 无此键时 from_dict
   自动补 128 保持逐位兼容。
3. **MTP 辅助 loss 的 doc_mask 泄漏**：`--doc_mask` 只掩了主 loss 的
   注意力；MTP block 的自注意力原来会跨文档。现在 MTP 也按源位置
   segment id 建 causal+doc 融合掩码。
4. **tied embedding × hrm_emb_scale 的初始 CE 偏移**：绑定 lm_head 复用
   被放大 27.7× 的输入表，初始 main CE≈27（正常 lnV≈8.8）。新增
   `scale_logits_by_emb_scale`，tied 时 logits 乘 1/hrm_emb_scale
   （小模型探针 CE 27.1→8.8）。新 run 默认开启（CLI 默认 True，写入
   sidecar 后 SFT/DPO/评估自动继承）；旧 checkpoint 的 sidecar 无此键
   时保持关闭，行为不变。A/B 旧行为用
   `--no-scale_logits_by_emb_scale`（queue 脚本 `SCALE_LOGITS=0`）。

## 代码清理与默认值（r073 后续）

- `moe_router_noise` 默认 **0.05**；`moe_aux_loss_weight` 默认 **0.001**
  （soft balance loss，只在可回传 cycle 收集）。`cycle_delta_max` 保留为
  实验开关但默认 **0**：r073 检查点探针中 RMS clamp 反而让 MTP 槽位退化
  为 base-router winner-take-most，噪声+aux 已能压到 2~4× 且零溢出。
- `scale_logits_by_emb_scale` 默认 **关闭**：开启虽把初始 CE 拉回 ln V，
  但等比例缩小 CE 对 hidden/embedding 的梯度，toy probe 中早期下降明显
  变慢（40 步 4.87→3.97 vs 关闭 11.06→1.11），保留为诊断开关。
- 噪声 0.1 的探针显示早期 loss 明显变慢（toy 40 步 0.44→0.58），
  0.05 是稳定性与早期收敛的折中；溢出主要发生在首微批，容量表更新后
  后续微批为零。
- 删除旧兼容分支：`cycle_emb` v1 路径、`engram_gate_dim=128` 兼容、
  `from_dict` 的旧 engram 注入/门控补丁、旧 `(E,)` expert_bias 广播。
  新代码要求 `--hrm_cycle_router` 同时给 `--hrm_cycle_router_rank > 0`。
- `BaseTrainer` 捕获 Ctrl-C，在最后完成微批位置保存 checkpoint 后退出。

## r073 终检后的架构修复（下一轮生效）

终检发现并修复：
1. **CycleDelta 改为 per-cycle V_c**：`W_router(c)=W+U·V_c`，删除从未训练的
   `cycle_g`（bf16@1.0 更新被舍入）。V_c 形状 `(n_cycles, rank, D)`，
   ~+49K 参数/gate，decode kernel 仍复用 cu·cz 口径。
2. **HRM input skip**：`hrm_input_skip=0.1`，每个 cycle 注入状态加一份
   RMS 归一化的初始 token embedding，抵抗后期 cycle 的 router_x res/common
   塌缩。
3. **Engram 独立 AdamW 组**：engram table/key/value/taps 不再走 Muon；
   `--engram_lr_mult`（默认 1.0）+ `--engram_scale`（默认 1.0）。
4. **MTP P=1 的 v_res_lambda 死参数关闭**；P>1 时 MTP block 使用自己的
   value-residual list。

## r074 中检后的第二轮架构优化（下一步 run）

针对 r074 step4999 仍存在的后期 cycle 塌缩与 delta 饱和：
1. **HRM state norm + 门控 token memory**：
   - `hrm_state_norm=True`：混合前对 z_L/z_H 分别 RMS 归一化；
   - `hrm_token_gate_scale=0.1`：`z <- (1-g)z + g·rms(embed(x))`；
   - 旧的加性 `hrm_input_skip` 默认 0，仅留作消融。
2. **Router logit 标准化**：`moe_router_logit_norm=True`，逐 token 把
   logits 标准化到 temp=1.0，训练与 decode kernel 同口径，防止 sigmoid
   饱和。
3. **Token 多样性正则**：`moe_diversity_loss_weight=0.01`，
   `mean_b log1p(common²/residual²)`，只在 hrm_bp_cycles 可回传 cycle
   收集。健康 res/common≈1 时 loss≈0，塌缩到 0.01 时≈9.2。

## 外部借鉴：Marin hero run scaling ladder（marin#8435，2026-08-23 读）

Marin 535B-A23B MoE 英雄 run 前的方法论。对本项目有用的部分：

**流程（制度化我们已踩过的坑）**：
- Scaling ladder 花 ~1% 算力，作用是抓 bug 而非只外推：他们曾在放大
  token horizon 时抓到 grad norm 涨到 4+，定位到缺 logit z-loss（不加
  会在高 batch 下中途爆掉）。对应我们 r070→r073 三次事故全是"换了
  一个结构维度（E=32→112）后旧超参失效、正式 run 才暴露"——教训同构：
  **改 E / seq / horizon / cycle 数任一维度，先跑百步级 ladder 点验证
  动力学再上全量**。
- 固化"健康曲线形状"作为不干预/干预判据：他们的参照是 grad norm 前
  40% 涨后随 LR 衰减回落；我们的等价信号是 `moe/overflow_pairs`、
  max/mean 负载比（健康 2~4×）、容量表水位。新 run 偏离形状立即停，
  不跑到 step3000 复盘。（r080 中检修正：细粒度 E=288/均值负载 256
  下溢出是"低水平慢性 + 缓慢下降"而不收敛到 0——近泊松尾必然偶发
  超余量，判据改为"不增长、无 loss 尖刺关联、占比 <0.1% pair"。）
- Early cooldown 分支做零污染实验：主 run 中途切短 cooldown 分支出
  可对比评估点（r070 提前 kill @77% 导致口径不干净的规范替代）。
- seq len 与均衡负相关：他们为此从 8k 退回 4k 预训练（每 batch 序列
  数翻倍）。我们 bs6×2048 仅 6 条序列，扩 seq 前需预期溢出恶化，
  ladder 需含 seq 维度。

**架构（见下文候选改动）**：
- 共享专家做抗 dropping 主干：他们 2 个半宽共享 ≈ 1/3 神经元来自
  shared，在 40% dropping 下仍无 loss 尖刺。我们 n_shared=1 全宽，
  共享占比仅 ~1/7。
- Latent MoE：active 专家前 2× 压缩，等效同 FLOPs 下 routed 神经元
  翻倍。
- Logit z-loss 压 grad norm 随 horizon 增长。
- 长上下文扩程：qk_mult 按 mscale=X·ln(new/old)+1（X≈0.1）调整；
  扩程是 dropping 的主要诱因（4k→65k：7%→40%），届时再启用。

## Latent MoE 落地（2026-08-23，`--moe_latent_dim`）

实现：`MoEFeedForward` 加共享 `lat_down (D→d)` / `lat_up (d→D)` 投影
包住路由专家（`moe_latent_dim=d`），router 仍在全维打分、共享专家保持
全宽。三路径全部复用：稀疏桶/kernel 均形状参数化，传 d 维输入即可，
bucket gather/scatter 流量减半；decode kernel 加 `latent_dim` 参数
（router kernel 全维、up/down kernel latent 维）。lat_down/lat_up 为
2D 矩阵自动进 Muon 组。无零初始化等价（函数类从 step 0 改变），只用于
新 run；`moe_latent_dim=0` 默认关，旧 checkpoint/sidecar 零影响。
回归 18/18（新增 test_moe_latent：结构审计/三路径逐专家参考等价/
梯度可达/sidecar 往返）。

吞吐 A/B（experiments/bench_latent_moe.py，r080 口径 bs12×1024
E=288 K=6 D=768 bf16，r080 在跑有背景竞争，三臂轮转计时取相对值）：
- baseline I=128：fwd+bwd 29.7ms（1.00×）
- latent d384/I=128：18.7ms（**0.63×**）——账面 FLOPs 仅 −16%，实测
  收益远大于此：bucket/权重读取减半，该路径是访存受限
- latent d384/I=256（等专家参数、神经元翻倍）：26.8ms（**0.90×**）——
  账面 +33% FLOPs 但比 baseline 还快

结论：d=hidden/2 时 latent MoE 在 Metal 上是"白拿容量"——I 翻倍
（routed 神经元 ×2）MoE 成本反而降 10%。A/B 臂：r081 = r080 配置
+ `--moe_latent_dim 384 --moe_intermediate_size 256`。

## 优化器借鉴：Marin 的 MuonH / AdamH / Adam 三分（同 issue 模型 spec，2026-08-23）

三组是三种更新规则：MuonH = NS 正交化方向 + **Frobenius 范数球投影**
（scale-invariant：方向由优化器学、范数冻结，fp32 算范数；专家 4D 栈
逐专家 NS 不 gather 矩阵维）；AdamH = Adam 方向 + 同样范数球投影
（只用于 lm_head，lr 用 MuonH 的）；Adam = embed/router/bias/gate/
1-D norm/小 conv kernel 的普通 Adam。

与现状差异：我们 Muon 无范数球投影（只有 Moonlight √max(1,r/c) 缩放）；
专家堆叠整体排除 Muon（reshape (E,out·in) 跨专家耦合）走 AdamW。

**已落地（2026-08-23，`--muonh`，默认关）**：
- `BatchedMuon(hyperball=True)`：groups/seg_groups/stack_groups 三分支
  更新后逐矩阵 rescale 回更新前 Frobenius 范数（fp32 范数）。
- 堆叠专家（ndim>2）以 axis0 为 batch 进 `_ns5` 逐专家 NS（批量维
  无耦合），从 AdamW 移入 MuonH；`VIBY_MUONH_EXPERTS=0` 可只开投影
  不移专家（消融/省 NS 开销）。
- `AdamH(FusedAdamW)`：lm_head（非 tied 才存在）Adam 方向 + 同一
  范数球投影，lr=基础 lr，wd=0（径向分量投影后无操作）。
- 注意：开关改变优化器分组，**不能续跑旧 checkpoint**（optimizer
  state 分组对不上），只用于新 run；r081 latent A/B 不开，避免混淆。
- 验证：experiments/test_muonh.py 11 项全过（范数冻结/对照组确实
  变化/堆叠逐专家==2D 逐片（含动量两步）/AdamH 与 Adam 共线且范数
  保持/端到端分组+一步更新范数审计）；muonh=False 分组与旧行为一致；
  真实数据 6 步冒烟通过。
- 成本提示：E=288 逐专家 NS 约 18 个 (288,r,c) 批量 NS/步，估计
  整步开销 +10~30%，正式 run 前用 queue 探针实测；太贵就
  VIBY_MUONH_EXPERTS=0 只留范数球投影。
- **实测（experiments/bench_muonh.py，r081 口径，r080 背景竞争下相对
  值有效）**：2D Muon base 36ms → +范数球投影 38ms（≈免费）；+专家
  逐专家 NS5 1221ms（+34×，~12 TFLOP/步）；NS bf16 无效（训练本
  就是 bf16，NS 一直在 bf16 跑）；NS 3 步 855ms；NS3+每 4 步重算
  摊薄 331ms（缓存命中步仍有 ~115ms 堆叠/动量/norm 固定流量）。
  ## Temporal MuonH / Cache Q（2026-08-23 研究原型）

专家必须留在 MuonH、又要压单机 NS 墙钟时，走 CacheMuon 口径：

- 刷新步：`_ns5_gram`（与标准 NS5 cos≈0.999）并缓存左变换 `Q`
- 命中步：`D = Q @ normalize(U)`，新动量仍进更新（对比复用旧极因子 D）
- 开关：`VIBY_MUONH_CACHE_Q=1` + `VIBY_MUONH_STACK_NS_EVERY=4|8`
- 验证：`experiments/test_muonh_cache.py`；墙钟：`bench_muonh.py` 的
  `exp_cache_q_e*`

**实测（r081，M4 Max）**：每步 NS5 ≈1277ms；Cache Q every=8 摊薄
先约 468ms。后续优化：刷新改为标准 NS5 顺带累积 Q（不再走更贵的
Gram 刷新）、动量/投影 mx.compile、命中残差触发
（`VIBY_MUONH_CACHE_Q_RES`，默认 0.15）。`--muonh` 默认
CACHE_Q=1 + EVERY=8。
