# 研究规格说明书 SPEC v1.1

## 60M @ 0.1C 追平 600M：判定实验、乘子雅可比与架构搜索

| 字段 | 值 |
|---|---|
| 版本 | v1.1（draft for review） |
| 关联提案 | `research/PROPOSAL.md` |
| 状态 | 待锁定：语料版本、参考 600M checkpoint、硬件 |
| 变更规则 | 本文件作为预注册文档；任何变更记为 amendment（理由、对预算的影响） |
| 数值约定 | 本文所有数值可复算；若与本文件冲突，以 `research/` 下复算脚本输出为准 |

---

## 1. 目标与约束（正式版）

### 1.1 主目标

固定任务分布 \(\mathcal P\)（定义见 §3.5）与参考模型 \(T\)（600M，定义见 §5），检验是否存在架构 \(A\) 与训练算法，使：

\[
N_{\text{mem}}(A)\le 60\text{M},\quad
C_{\text{train}}(A)\le 0.1\,C_{\text{ref}},\quad
c_{\text{inf}}(A)\le c_{\text{ref}},\quad
L_{\text{BPB}}(A)\le T
\]

其中 \(T\in\{T_{\text{opt}}=2.7296,\ T_{\text{pes}}=2.6716\}\)（nats/token，BPB 等价见 §3.2）。

### 1.2 三个赛道

| 赛道 | 闭卷 | 账本口径 | 教师/过滤器 | 存储 |
|---|---|---|---|---|
| A（对照） | 是 | 总账 | 禁止 | \(N_{\text{mem}}\le60\text{M}\) |
| B（主线） | 是 | 摊销 | 允许（FLOPs 登记） | \(N_{\text{mem}}\le60\text{M}\) |
| C（退路） | 否（允许非参数检索） | 总账 | 允许 | 权重 ≤60M；检索存储字节单独计入 \(\mathcal R\) |

### 1.3 资源向量

\[
\mathcal R=(N_{\text{mem}}, C_{\text{train}}, c_{\text{inf}}, M_{\text{kv}}, d_{\text{ser}})
\]

- \(N_{\text{mem}}\)：权重字节。主赛道含 embedding、输出头（若 tied 只计一次）、所有可训练标量。
- \(C_{\text{train}}\)：训练总 FLOPs（口径见 §3.3）。
- \(c_{\text{inf}}\)：每 token 推理 FLOPs（profiler 实测，口径见 §3.3）。
- \(M_{\text{kv}}\)：推理时的 KV/状态字节（ctx=4096 下报告）。
- \(d_{\text{ser}}\)：串行深度（一次前向的串行层数；循环模型按展开次数计）。

---

## 2. 三个赛道之外的约束

1. 主赛道锁 \(N_{\text{mem}}, C_{\text{train}}, c_{\text{inf}}\) 三项；MoE（总参数>60M）与无限循环（\(c_{\text{inf}}\) 超限）都被排除。
2. 双账本强制：所有 FLOPs 同时登记**总账**与**摊销**两个口径（§9）。
3. 禁止在 eval 集上做任何选择；HP 搜索、架构搜索全部计入账本。
4. 60M 处 embedding 预算 ≤6M（本仓库 6.4k×768 tied ≈4.9M，合规）。

---

## 3. 定义与口径

### 3.1 参数口径

- \(N_{\text{total}}\)：所有可训练标量（含 embedding、norm、门控标量、MTP 头）。
- \(N_{\text{nonemb}}\)：去掉词表嵌入与输出头（若 tied 则只减一次）。
- 报告两者。参考目标默认用 \(N_{\text{total}}\)。

### 3.2 主度量：bits-per-byte（BPB）

\[
\text{BPB}=\text{nats/token}\times(\text{tokens/byte})\times\log_2 e
\]

- tokens/byte 在锁定 eval 集上用目标模型自己的 tokenizer 实测；BPB 是 tokenizer 不变量（分割效应除外）。
- 次级度量：分域 BPB（web/code/math/dialogue）、P1–P4 probe 分数、MMLU-5shot（仅用于解释，不作主判据）。
- 报告任何总 loss 差值时必须附分域分解表。

### 3.3 FLOP 口径

- 训练：默认同时报 (i) 近似口径 \(C=6ND\)（D=参与 loss 的有效 token 数，backward=2×forward），与 (ii) profiler/解析口径（含注意力 \(4L_{\text{ctx}}d\)/层/token、embedding、MTP 头，逐项列出）。
- 上下文长度 \(L_{\text{ctx}}=4096\) 写死。\(L_{\text{ctx}}\) 是自由变量，不锁定则所有比较无效。
- 推理：\(c_{\text{inf}}\) 用 profiler 在 batch=1、ctx=4096 下测前向 FLOPs/token。
- 参考 \(c_{\text{ref}}\)：公开 600M checkpoint 同法实测；若无法实测，用其解析式并显式标注。

### 3.4 计算账本（compute ledger）

必须登记：模型训练、HP 搜索、架构搜索、教师前向、数据过滤模型前向、合成数据生成、消融/复现。两种口径：

- **总账**：一切计入 \(C_A\)（科学问题视角）。
- **摊销**：教师预训练、一次性过滤模型作为基础设施摊销（部署视角）。

### 3.5 语料与任务分布

- 主分布：C4 子集 + code + math（锁定版本/过滤管线，预注册）。
- eval：每域 1–2B tokens held-out；拼接后测 BPB 与分域 BPB。
- 下游：MMLU-5shot（子集可）仅作解释变量。
- 在语料未锁定前，任何 E0 之后的实验不允许启动。

---

## 4. 数学参考值（复算通过，写入 v1.1）

### 4.1 拟合常数

Hoffmann（Chinchilla）：

\[
E=1.69,\ A=406.4,\ B=410.7,\ \alpha=0.34,\ \beta=0.28
\]

Besiroglu et al. (2024)：

\[
E=1.82,\ A=482.0,\ B=2085.4,\ \alpha=0.348,\ \beta=0.366
\]

参考点 \(N_t=600\text{M},\ D_t=12\text{B},\ C_t=6N_tD_t=4.32\times10^{19}\)。

### 4.2 闭式分解

\[
\Delta L=A N_t^{-\alpha}(10^{\alpha}-1)+B D_t^{-\beta}\big((10\gamma)^{-\beta}-1\big)
\]

- 参数项固定缺口 \(=0.5002\) nats；
- \(\gamma=0.1\)：\(\Delta L=0.5002\)（数据项恰好为 0）；
- \(\gamma=1.0\)：\(\Delta L=0.2062\)；
- breakeven \(\gamma^*=36.7\)，即 60M 需 \(4.41\times10^{12}\) tokens（Hoffmann）；
- 过训练不变性：目标 \(D_t=s\times12\text{B}\)、预算按同一 \(s\) 缩放时，\(\Delta L=0.5002\) 对 \(s=1,5,20\) 逐位不变。

### 4.3 等值线

目标 excess \(T=A N_t^{-\alpha}+B D_t^{-\beta}=1.0396\)。对任意 \(\mu_N\)：

\[
\mu_D^{\min}(\mu_N)=\left(\frac{B}{(T-A(\mu_N\cdot60\text{M})^{-\alpha})\,D_t^{\beta}}\right)^{1/\beta}
\]

| \(\mu_N\) | 1.00 | 1.60 | 2.00 | 2.70 | 3.00 | 5.00 | 7.26 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| \(\mu_D^{\min}\) | 367.2 | 23.9 | 11.6 | 5.57 | 4.52 | 2.04 | 1.34 | 1.00 |
| \(\mu_C=\mu_N\mu_D\) | 367 | 38.2 | 23.1 | 15.0 | 13.6 | 10.2 | **9.71** | 10.0 |

- 最小有效计算乘子 \(\mu_C^{\min}=9.71\) @ \((\mu_N,\mu_D)=(7.26,1.34)\)。
- 教训写入：乘积 10.3 但方向为 (1.60, 6.47) 时 excess=1.153 > 1.0396，**不达标**。方向与乘积同等重要。
- Besiroglu 下即使 \(D\to\infty\)，仍需 \(\mu_N\ge1.36\) 才能追平 \(T_{\text{pes}}\)。

### 4.4 循环引理

\[
\mathcal F_{\text{looped}(k,N)}\subseteq \mathcal F_{\text{untied}(k\cdot N)}
\]

推论：近似误差主导且优化充分时，looped 最优损失 \(\ge\) untied(kN) 最优损失；循环只可能通过归纳偏置（优化/泛化项）赢，属二阶效应，只能实测。

训练侧最优展开（\(C=0.1C_t,\ N=60\text{M}\)）：

\[
k^*=\left(\frac{\alpha A(60\text{M})^{-\alpha}}{\beta B D_t^{-\beta}}\right)^{1/(\alpha+\beta)}=2.60
\]

- 收益 \(0.0658\) nats = 所需量的 13.2%；
- \(k=12\)：\(E+B(10^9)^{-\beta}=2.9303>2.7296\)，即使参数项为 0 也不达标。

### 4.5 四条界

| 界 | 陈述 | 实验 |
|---|---|---|
| B1 知识容量 | 2 bit/param 量级：60M→120 Mbit≈15MB 事实；600M→150MB。闭卷 10× 知识追平被禁止；总 loss 追平不被禁止 | P1 |
| B2 时间 Kolmogorov | \(Kt(x)=\min_p |p|+\log t(p)\)；循环在 \(|p|\) 与 \(\log t\) 间交换，固定计算无免费空间 | 分析 |
| B3 深度 | 常数深度 log-precision Transformer ⊆ 均匀 \(\mathsf{TC}^0\)；S5/迭代群乘法分离为条件性（\(\mathsf{TC}^0\ne\mathsf{NC}^1\) 开放），必须带条件引用或改用无条件分离（迭代复合类） | P2 |
| B4 状态检索 | 精确复制/联想召回需状态 \(s=\Omega(L\log V)\)；SSM 通道必丢精确检索，需注意力通路 | P3 |

---

## 5. 参考基线与目标值

### 5.1 参考点

- 约定参考：600M@12B（Hoffmann \(L=2.7296\)；Besiroglu \(L=2.6716\)）。
- 精确最优注记：600M@12B 不是 Hoffmann 拟合的精确计算最优点（精确最优点为 441M@16.3B，\(L=2.7250\)）。论文中二者都要报，主判据用约定点。
- 第二参考：公开的 600M 级 checkpoint，实测 BPB 与 \(c_{\text{inf}}\)；其训练 FLOPs 登记入摊销账本（若公开值缺失，用 \(4.32\times10^{19}\) 标注为估计）。
- **不自己训练 600M**：除非 E0 通过后单独批准，600M@12B 自训成本 4.32e19 FLOPs 已超出总预算的 70%，不在默认计划内。

### 5.2 目标值

\[
T_{\text{opt}}=2.7296,\qquad T_{\text{pes}}=2.6716
\]

成功分级：S1 = BPB ≤ \(T_{\text{opt}}\)；S2 = BPB ≤ \(T_{\text{pes}}\)。

---

## 6. 架构假说 v1.1（候选，可被 E1 淘汰）

### 6.1 参数预算（60M 总额）

| 组件 | 参数 | 备注 |
|---|---|---|
| 词表嵌入 tied | ≈4.9M | 6.4k×768，沿用本仓库 tokenizer |
| 循环块 \(T_\theta\)（含块内 norm） | ≈52M | 8L×768，MLA+SwiGLU+RMSNorm+QK-norm，即 round11 类配置 |
| per-step 调制 + MTP 头（若开）+ 其他 | ≈3M | FiLM scale/shift 打破层同质性；总额以 60M 审计为准 |


### 6.2 展开策略

- \(k_{\text{train}}\)：每步从分布中采样，均值 ≤2.6，偶发 \(k=10\)（支持 H4）；报告采样分布。
- \(k_{\text{inf}}\)：上界由 \(c_{\text{inf}}\le c_{\text{ref}}\) 决定，预期 4–8（用 profiler 实测，**不得用 2N 近似**）。
- H4 判定实验：训练完成后扫 \(k_{\text{inf}}\in\{1,\dots,k_{\max}\}\) 测 BPB 曲线。若 \(k_{\text{inf}}>k_{\text{train}}\) 无收益且不退化，H4 成立；若退化，H4 为假，循环降级。

### 6.3 组件规则

1. 注意力：保留 MLA 全注意力（ctx 4096），作为 B4 要求的精确检索通路。
2. SSM 通道：仅在 P3 显示窗口注意力+SSM 的失效点可接受时加入；其状态字节计入 \(M_{\text{kv}}\)。
3. DEQ：仅允许作为显存优化；不宣称省训练计算（反向 Krylov 迭代仍需 \(O(k)\) 计算）。
4. 超网络：先做预验证——公开 600M checkpoint 逐层 PCA/低秩+稀疏分解，若保持 99% 损失所需内在维度 > 40M（60M 的 2/3），该路线终止，不进入 E3。

---

## 7. 实验协议

### P0：本仓库 MLX pilot（与 E0 并行）

- 目的：验证 BPB 口径、账本格式、H4 采样训练代码在 Viby 代码库上可行。
- 配置：30M、D=0.6B、ctx=4096（或当前可行的最大 ctx）、1 seed。
- 预算：≤3×10^17 FLOPs。
- 通过条件：产出可复算的 BPB、profiler FLOPs、ledger 行、checkpoint hash。

### E0：O2-lite + M0 重拟合（最先的判定实验）

**E0a 网格（FLOPs 已按 6ND 列出）：**

| N | D/N 比值 | D | seeds | FLOPs |
|---|---|---|---|---|
| 30M | 10 / 40 / 160 | 0.3 / 1.2 / 4.8B | 2 | 2.27×10^18 |
| 60M | 10 / 40 / 160 / 200 | 0.6 / 2.4 / 9.6 / 12B | 3 | 1.98×10^19 |
| 100M | 10 / 40 | 1 / 4B | 1 | 3.0×10^18 |
| 150M | 10 / 40 | 1.5 / 6B | 1 | 6.75×10^18 |
| 合计 | | | | **3.18×10^19** |

- O2-lite 即 60M 在 D=2.4/9.6/12B 的锚点序列，用 Hoffmann/Besiroglu 双先验贝叶斯外推 \(D\to\infty\) 地板。
- 拟合输出：五参数 \((E,A,\alpha,B,\beta)\) + bootstrap CI；加性形式 vs 交互项模型

\[
L=E+A/N^{\alpha}+B/D^{\beta}+C/N^{\alpha/2}D^{\beta/2}
\]

  的模型比较（AIC/BIC、C 的 CI）。
- 敏感性：Huber-δ；窗口敏感性（去掉最小/最大 N 重拟合）；非嵌入参数口径重算。
- **Kill gate G1**：
  - 若 60M 地板 CI 完全高于 \(T_{\text{pes}}\) → A/B 不成立，发表否定结果，转 C；
  - 若 CI 完全低于 \(T_{\text{opt}}\) → 继续，单目标 S1；
  - 若跨两个目标 → 继续，双目标（E3 同时判 S1/S2）。
- O2-full（60M@24B/48B）仅当 G1 straddle 且另批预算时才执行；它一次 8.6–17.3×10^18 FLOPs，不包含在 E0 默认预算内。

### E1：乘子雅可比（部分因子设计）

- 设计：\(2^{6-2}_{\text{IV}}\)，16 runs，@30M、D=0.6B、ctx=4096。
- 因子（预注册，+1 水平见括号）：

| 代号 | 因子 | -1 | +1 |
|---|---|---|---|
| A | 数据 | dedup 基座 | 质量过滤（≤30M 过滤器，FLOPs 记账） |
| B | 优化器 | AdamW | Muon+AdamW 混合 |
| C | 循环 | dense 8L | loop 4L×2（同有效深度、同每 token FLOPs） |
| D | 辅助目标 | 无 | MTP depth=1 |
| E=ABC | 打包 | pad | pack+docmask |
| F=ABD | 几何 | 浅宽：有效 8L×768 | 深窄：有效 16L×~544（N_total 等配 ±2%；loop 运行时为 8L 块 × k=2） |

- 设计矩阵见附录 A；该设计为主效应清晰（aliased 到 3FI），部分 2FI 可估（按 alias 组报告）。
- 响应变量（每 run）：
  1. \(R_0\) = BPB；
  2. \(R_1=\log \mu_N^{\text{D-proj}}\)（把全部增益投影到 N 轴，固定 D=0.6B）；
  3. \(R_2=\log \mu_D^{\text{N-proj}}\)（投影到 D 轴，固定 N=30M）；
  4. profiler FLOPs、wall time（作协变量）。

- 投影公式（\(\hat{}\) 为 E0 后验均值，CI 用 bootstrap 传播）：

\[
T_i=L_i-\hat E,\quad
\mu_N^{\text{D-proj}}=\left(\frac{\hat A}{(T_i-\hat B/D^{\hat\beta})(30M)^{\hat\alpha}}\right)^{1/\hat\alpha},\quad
\mu_D^{\text{N-proj}}=\left(\frac{\hat B}{(T_i-\hat A/(30M)^{\hat\alpha})D^{\hat\beta}}\right)^{1/\hat\beta}
\]
- 分析：对 \(R_1,R_2\) 做主效应与可估 2FI 的 contrast 分析 + bootstrap；输出**乘子雅可比矩阵**（每杠杆一行：\((\Delta\log\mu_N,\Delta\log\mu_D)\)）。
- 预算：16×1.08×10^17 + 2 个复现点 ≈ **2.0×10^18 FLOPs**。
- **Kill gate G2**：
  - 由雅可比构造可达集（含交互的保守椭球）；
  - 若可达集与等值线（Hoffmann 及 Besiroglu 两版）均不相交 → A 终止；
  - 若仅在摊销口径相交 → 只保留 B；
  - 若总账口径相交 → A、B 都保留。

### E2：探针 P1–P4（≤10M 模型）

| Probe | 对应界 | 协议 | 预算 |
|---|---|---|---|
| P1 知识容量 | B1 | 注入 \(Q\) 条可控 bit 数的事实（随机 key-value），扫重复次数与 Q；拟合 2 bit/param 常数；另用闭卷 QA 做知识/技能拆分 | ~3×10^16 |
| P2 深度 | B3 | S5 状态追踪（条件性）、迭代复合（无条件）、多步算术；对比 dense 与 loop，测参数节约倍数 | ~2×10^16 |
| P3 检索 | B4 | ctx 512–4096 精确复制与联想召回；测 SSM/窗口注意力失效点 | ~3×10^16 |
| P4 分域 BPB | — | web/code/math/dialogue held-out | eval-only |
| 合计 | | | **≤1×10^17** |

淘汰线：单机制兑现 <3× 即不进入 E3。

### E3：60M 全量判定

1. 组合验证：E1 前 2 名组合 @60M、D=12B、ctx=4096（≈4.32×10^18 each）。
2. O3 蒸馏（B 赛道）：公开 600M 教师；教师前向预算 ≤2B tokens（≈2.4×10^18），学生训练计入账本；教师预训练 FLOPs 4.32×10^19 只进摊销账本。
3. O4 无教师（A 赛道）：60M@12B 从头训练（4.32×10^18）。
4. 输出：S1/S2 判定 + P1–P4 + 分域 BPB 分解。
5. 预算合计 ≈1.6×10^19 FLOPs。

### 预算总计

| 阶段 | FLOPs |
|---|---|
| P0 | ≤3×10^17 |
| E0 | 3.18×10^19 |
| E1 | 2.0×10^18 |
| E2 | 1×10^17 |
| E3 | 1.6×10^19 |
| Buffer（HP/架构搜索，记账） | ≤1.5×10^18 |
| **总上限** | **≤6×10^19** |

≈42 H100-小时（40% MFU）；8×H100 约一晚。若用本仓库 MLX 环境，仅 P0 可行（≈10 小时），E0 起需要租用 GPU。

---

## 8. 统计规范

1. 关键 60M 点 3 seeds；其余 ≥1 seed；任何 <0.05 nats 的宣称必须带 CI。
2. 超参迁移用 μP：在 10M proxy 上调 HP，固定后迁移到 30M/60M/100M/150M。
3. 等调优预算：候选架构与基线消耗的 HP 搜索 FLOPs 必须相等并登记。
4. 基线是前沿不是单点：至少 3 宽度 × 3 深度 @60M，报 Pareto。
5. 所有缩放律拟合报 bootstrap CI；报告 Huber-δ 与拟合窗口敏感性。
6. 预注册：本文件 + 因子水平 + eval 集 commit hash，先于 E0 数据拟合冻结。

---

## 9. 计算账本 schema

每条记录：

```yaml
run_id: E0_N60M_r200_seed1
stage: E0
account: model_training | hp_search | arch_search | teacher_forward |
         data_filter | synthetic_gen | ablation | reproduction
ledger_view: total | amortized      # 该条 FLOPs 计入哪些口径
model: viby_8L768_mla
N_total: 60967000
N_nonemb: 56000000
D_effective_tokens: 12e9
ctx: 4096
FLOPs_6ND: 4.32e18
FLOPs_profiled: 5.0e18        # 含注意力，实测
device: H100
mfut: 0.42
wall_hours: 3.1
seed: 1
parent_run: null               # HP/消融时指向被服务的主 run
checkpoint_sha: abc123
eval_set_sha: def456
notes: ""
```

- 摊销口径默认：教师预训练、一次性数据过滤器训练/前向、tokenizer 训练（若用）不计入 \(C_A\)；其余全部计入。
- 总账口径：所有行求和。
- 每个论文表格必须注明用的是哪个口径。

---

## 10. 交付物与文件结构

```
research/
  PROPOSAL.md          # 本提案
  SPEC.md              # 本规格
  compute_isoquant.py  # 等值线/预算/设计矩阵复算脚本（待建）
  ledger_template.yaml # 账本模板（待建）
  fit_scaling_law.py   # 双先验 bootstrap 拟合（待建）
  e0_configs.yaml      # E0 网格与 seed 分配（待建，预注册）
  e1_design.yaml       # E1 16-run 设计（待建，预注册）
```

报告必须包含：目标定义、账本口径、CI、分域 BPB、知识/技能拆分、失败/成功的 kill gate 路径。

---

## 11. 终止判据汇总

| Gate | 触发条件 | 动作 |
|---|---|---|
| G1 | 60M 地板 CI 高于 \(T_{\text{pes}}\) | 发表否定结果；A/B 终止；转 C |
| G2 | 乘子可达集与两版等值线均不相交 | A 终止；视摊销口径决定 B；或转 C |
| G3 | 单机制 <3× | 该机制不进入 E3 |
| G4 | E3 未达 S1 且未达 S2 | 发表否定结果 + 缺口分解（知识/技能/分域） |
| H4 | \(k_{\text{inf}}\) 外推退化 | 循环降级为 13% 收益组件 |

---

## 附录 A：E1 设计矩阵（\(2^{6-2}_{IV}\)，E=ABC，F=ABD）

| run | A data | B opt | C loop | D mtp | E pack | F geom |
|---|---|---|---|---|---|---|
| 1 | -1 | -1 | -1 | -1 | -1 | -1 |
| 2 | -1 | -1 | -1 | +1 | -1 | +1 |
| 3 | -1 | -1 | +1 | -1 | +1 | -1 |
| 4 | -1 | -1 | +1 | +1 | +1 | +1 |
| 5 | -1 | +1 | -1 | -1 | +1 | +1 |
| 6 | -1 | +1 | -1 | +1 | +1 | -1 |
| 7 | -1 | +1 | +1 | -1 | -1 | +1 |
| 8 | -1 | +1 | +1 | +1 | -1 | -1 |
| 9 | +1 | -1 | -1 | -1 | +1 | +1 |
| 10 | +1 | -1 | -1 | +1 | +1 | -1 |
| 11 | +1 | -1 | +1 | -1 | -1 | +1 |
| 12 | +1 | -1 | +1 | +1 | -1 | -1 |
| 13 | +1 | +1 | -1 | -1 | -1 | -1 |
| 14 | +1 | +1 | -1 | +1 | -1 | +1 |
| 15 | +1 | +1 | +1 | -1 | +1 | -1 |
| 16 | +1 | +1 | +1 | +1 | +1 | +1 |

说明：E 与 F 是生成元列，不是可自由选择的两个额外因子；若需更换因子，只能整列替换并作为 amendment 预注册。分辨率 IV 下主效应清晰，2FI 按别名组报告。

---

## 附录 B：FLOP 预算推导摘要

- E0：\(\sum_{\text{grid}} 6ND\times \text{seeds}=3.18\times10^{19}\)（见 §7 表）。
- E1：\(16\times 6\times30\text{M}\times0.6\text{B}=1.73\times10^{18}\)，加复现点取 2.0e18。
- E3：2 组合 × 4.32e18 + O3 教师 2.4e18 + O3 学生 0.72e18 + O4 4.32e18 ≈ 1.6e19。
- 总上限 6e19 FLOPs ≈ 42 H100-小时 @40% MFU（H100 BF16 按 989 TFLOPS 计）。
