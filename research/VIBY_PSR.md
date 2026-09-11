# Viby-PSR 科学修订：受保护的输出修正器

日期：2026-09-11。修订基点为 `52d17a67f3012af1052c99ff48f4eccade6dcd7e`。
本文描述当前实现及验收边界，不是验证集收益或效率优势声明。旧 PSR 实现、旧配方和实验历史
保留在该 commit；已有 checkpoint 未被覆盖。旧的 Decoder 多层注入与辅助分类/自蒸馏/value
训练路径已从当前主线移除。

## 计算与梯度契约

原 Viby 完整计算 `h0, z0`，保留原 CED/mHC `pre_mix`、RoPE、CSA2、MoE、Engram 和 MTP
目标。PSR 仅截取 CED 的原始 KV 和真实边界 attention input，不修改任何 Decoder 层。

```text
原始主干 ── h0、z0 ─────────────────────────── 原主干 loss → 原优化器
       └── 原 CED KV / anchor 特征（detach）
                  ↓
           独立工作槽 → R 轮稠密读取 + 共享更新器
                  ↓
     workspace attention(query=detach(h0), offset=t-anchor)
                  ↓
         零初始化词表输出头 A → Δz
                  ↓
              detach(z0) + g·Δz → 同位置 NTP CE → 独立 AdamW
```

memory、anchor、读出 query hidden、base logits 四条输入都切断梯度。主干 loss 保留原 LM CE、
原 MoE 辅助项以及启用时的 MTP/z-loss。没有 future-offset 分类、自身读取蒸馏、价值或 rank loss。

`A` 严格为零、`g` 初始为 1。全局初始化器豁免 `psr.output.weight`，构造末尾再次清零。
第一步输出头通常有梯度、Reasoner 无纠错梯度；头更新后，上游获得梯度。零头只保证有限
数值前向的初始化等价，不保证训练后的预测永远更好。

`trainer/psr_optim.py` 提供分离的参数视图。主干优化器分组、LR 自动推导、裁剪、weight decay、
MoE bias 更新保持原规则。侧路使用独立 optimizer state、LR、裁剪、NaN 处理；侧路坏梯度不会
让主干跳过更新。`--psr_freeze_base` 可只训练侧路。端到端共适应不在当前实现中暗中启用。

新侧路在所有主干初始化结束后构造，并恢复原 MLX RNG key。MLX 0.32.2 的 random.state
是只读的线程状态 sentinel；实现通过原始二字 key 恢复到公开 seed API。anchor 抽样用显式
独立 PRNG key（seed+microstep），作为编译函数的运行时输入，不消耗主干或 shuffle RNG。
参见 [MLX 随机数说明](https://ml-explore.github.io/mlx/build/html/python/random.html) 与
[0.32.2 随机状态实现](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/random.cpp)。
即使初始化流一致，配对实验仍必须逐 tensor 复制/核对公共权重、optimizer state 和 MoE bias，
不能用“相同 seed”替代这一检查。

## 模式与局部时间对齐

- `off`：完全跳过新模块；忽略遗留的 thinking options/targets；返回原始 logits。
- `state_only`：构造 S0，不执行循环，但读出与词表修正仍存在。
- `recurrent`：明确执行 R≥1。R=0 报错，不再兼任开关。

模型库仍显式使用 `VibyConfig(psr_enabled=True)`；`train_pretrain.py` 默认开启受保护路径，
`--no-psr` 关闭。默认 **H=16、R=1、8 槽、dim=256、2 个共享 dense block**。
每个文档从文档首位置开始划分长度 H 的块；anchor b 只能看 `x<=b`，只修正 `[b,b+H)`，
并继续受同文档/PAD 限制。对应 logits 始终预测 `x[t+1]`，读出 query 来自合法的实际 `h0_t`。
R=1/2/4 的目标、anchor、horizon 和 mask 不变。

训练每行均匀抽取最多 2 个候选块，包含短文档，不再只选最长文档。多个 anchor 在 B×A 维
并行。纠错 CE 采用块抽样逆包含概率权重，并除以全部有效标签数，估计完整固定块策略的
目标。日志中的 corrected CE 则是不加这种训练估计权重的真实当前预测 CE；抽样覆盖收益
不能冒充全覆盖收益。评估使用 `compact_anchor_plan` 的完整固定策略，布局整理在编译/计时外。

显式 anchor 若重叠，采用最新合法 anchor 唯一负责，不把多个修正相加。`-1` 表示填充槽。
无有效标签时 LM/纠错项为零，原主干既有辅助项保持其原语义。

## 使用方式

```python
cfg = VibyConfig(psr_enabled=True, psr_rounds=1, psr_horizon=16)
model = VibyForCausalLM(cfg)
out = model(
    input_ids, labels=labels, loss_mask=loss_mask,
    attention_mask=pad_mask, segment_ids=segment_ids,
    psr_mode="recurrent", psr_anchors=anchors,  # int32 [B,A]
    thinking_options={"rounds": 1},
    psr_gate=mx.array(1.0),                    # 运行时可变，不由标签决定
    return_metrics=True,
)
# out.loss = 原主干全部目标 + 隔离纠错目标
# out.lm_loss = 原始主干纯 CE
# out.corrected_lm_loss = 实际修正后、token 加权纯 CE
```

`prefill(prompt, psr_mode=...)` 在问题末尾启动第一块；普通逐 token decode 每 H 个位置重新
从已观察的原始 CED cache 启动下一块，继续沿用当前模式、R 和 gate。该修正不反馈入 token KV；
同一输入 token 序列的原始 cache 不变。采样 token 可以因输出分布改变而不同。
`cache.psr_phases` 记录发起的工作区阶段，不能当成 GPU profiler 计数。不同条件使用全新 cache。
多文档 packed 数据支持无缓存训练/评估；多文档 cached 生成明确拒绝。连续 batch engine 和
DSpark speculative 生成尚未搬运该工作区，仍拒绝 PSR 配置；MTP 的原主干训练目标可以保留。

普通预训练命令可以继续使用，默认配方已改为受保护 R=1；可选参数包括：

```text
--psr_horizon 16 --psr_rounds 1 --psr_train_anchors 2
--psr_learning_rate 0.0001 --psr_grad_clip 1 --psr_weight_decay 0
--psr_training_mode off|state_only|recurrent
--psr_freeze_base
```

`--no_save` 仍按原语义关闭 checkpoint、SwanLab 和 auto-resume；不会为了实验记录偷偷写入。
普通 baseline checkpoint 加载进新模型时，严格检查/复制公共参数，新增侧路保持初始化，
**保留主干 optimizer 和训练进度**。新侧路 optimizer 存在单独文件。
旧版 PSR checkpoint 会明确拒绝，避免将不兼容的多层桥接权重误映射成新结构。
新 checkpoint 记录组参数名、可序列化超参数、公共权重/主干 optimizer SHA256、源码/tokenizer hash、MLX/Python RNG、当前 epoch shuffle 起始状态
及独立 anchor microstep。旧 checkpoint 缺少 RNG/数据位置记录时，不能声称精确续训轨迹已复现。

## 评估、诊断与校准

`experiments/psr_evaluate.py` 接受预先冻结的 NPZ：`input_ids, labels, loss_mask` [N,T]，
可选 `attention_mask, segment_ids`；文档 bootstrap 必须提供全局 `document_ids` [N,T]。
缺少 document_ids 时只做 row bootstrap，并明确标注，不称为文档级置信区间。

```bash
.venv/bin/python experiments/psr_evaluate.py \
  --checkpoint RUN/model.safetensors --data validation.npz \
  --baseline-checkpoint BASE/model.safetensors --output evaluation.json
```

所有条件使用同一数据、完整 anchor policy 和全新无缓存前向，报告 off/state_only/recurrent、
`nll_sum / valid_label_count`、直接覆盖区/补集、实际覆盖率、逐文档配对记录及 bootstrap。
有独立 baseline checkpoint 时计算 `D_total = D_injection + D_history`；这是即时干预分解，
不是完整训练因果结论。初始化零头的 on/off 相等也不是模块学到算法状态的证据。

训练 `psr_metrics.jsonl` 保存可加和的 NLL/count；分组依次为 all、covered、complement、prefix、
offset 0–3/4–7/8–15/16–31。不得对不等长 batch 的均值再无权平均。`microstep` 与记录时已完成的
`optimizer_step` 分开，`phase=pre_update`；记录 consumed input token 和有效监督数。
覆盖率是结构上的合法修正区域，零头/g=0 时仍可非零。

`calibrate_gate(base_logits, delta, labels, mask)` 实现固定方向上 g∈[0,1] 的凸校准。
它只能用于独立校准集；推理只传已固定的 gate。参数改变或数据参与过校准后，不得把同一批
数据的结果叫独立验证。若校准选 g=0，记录“没有可验证修正收益”，不扩展 R 来掩盖结果。

## 稀疏化与成本边界

默认仅稠密工作区读取，原 CSA2 不变。没有训练中的自身教师，也没有自动停机。
`read_mode="fixed"` 需要显式固定 indices [B,A,m,k]，直接 gather 并重算当前权重，
**不执行全局扫描**，供固定证据消融。这个控制不是精确稀疏优化。
`read_mode="sparse_diagnostic"` 在同一 state/query/value 下测 rho、局部读出误差和
`2*Vmax*(1-rho)`；默认 rho<0.95 或非有限结果即报错，不返回该近似预测。降低阈值仅用于
明确的诊断实验，不能作为通过质量门槛的证据。诊断路径包含稠密教师和 host 质量检查，不能
当成部署加速路径或固定图训练路径。

当前不训练 Indexer，不执行旧 Top-K 稳定性证书跳过，也不添加 value/rank loss。E4 必须等
稠密 E2/E3 在独立验证有信号后再做固定教师蒸馏。

`--profile` 可另外输出同步 eager 的 baseline/Reasoner/workspace readout/词表头分段计时；
这是诊断 profile，强制中间输出且不等于融合路径吞吐。`profile_smoke.json` 已跑通该流程。
评估计时包含完整前向及首次编译，记录进程峰值内存；logical scan 是代码调用计数，非 GPU
profiler 结果。训练 FLOPs 是名义估算，包含循环、桥接、额外词表头及 detached base logits；
不是带宽/排序/优化器时间。E3 的等计算前馈、显式 CoT、多种子及真实 GPU profile 仍需独立
实验，不能从这次实现短跑推出速度或统计显著性。

## 本次验收与尚缺证据

`tests/test_psr.py` 覆盖 T01–T18 中可执行的模型/训练边界，包括：零头、共享权重逐 tensor
核对、隔离梯度、十次共同更新、独立 NaN 处理、因果与 packed 隔离、相同 horizon、明确模式、
完整前向/缓存解码、运行时 gate/anchor key、固定读取无扫描及同状态稀疏误差诊断。
十次更新检查覆盖 tiny FP32/BF16 eager/compiled AdamW，并补了真实 Muon/Sinkhorn 分组；
预先采用公共参数/状态绝对容差 2e-6，FP32 前向/缓存比较 2e-5。不是默认 1B/full-shape kernel
的验收。T16 的真实 GPU profiler 对账尚未完成，逻辑阶段计数不替代它。

本地只识别到一份不含 PSR 张量的基线 checkpoint（epoch 0、step 14721）。其 metadata 与
文件 SHA256 已只读导出到 `research_runs/psr_revision_audit/baseline_inventory.json`。
缺少对应旧 PSR checkpoint 和图中两条 run 的匹配记录，E0 分解尚未完成，不能归因曲线退化。

`research_runs/psr_revision_smoke/e2_smoke/` 完成 3 个 tiny、冻结基座的编译更新，确认公共参数
未改变；`evaluation_smoke.json` 只验证评估流程，使用同一实现 fixture，不是独立验证集。
尚未执行真实基线上的 E2 效果确认、E3 循环优势实验、E4 稀疏蒸馏、E5 深度外推或大预算重训。

```bash
# 最便宜的冻结基座机制入口；数据需预先固定，另备独立 validation.npz
.venv/bin/python -m trainer.train_psr \
  --checkpoint BASE/model.safetensors --data train.npz \
  --out-dir research_runs/psr_e2_new --steps 32 --rounds 1 --horizon 16 --compile
```

E2 若只支持 g=0 或独立验证无收益，应停止扩大计算预算。固定槽不能无条件替代任意 CoT；
本实现不保证每个样本、每个 R 或相同训练成本下必然获益。

本次最终相关回归：97 passed（13.33s）。随后源码哈希/optimizer 超参恢复及稀疏索引范围
检查分别通过 checkpoint 专项与相关 PSR 专项。原 `train_pretrain.py` 的一个两微批
Muon 累积窗口已完成，侧路 optimizer_step=1、梯度有限；日志在
`research_runs/psr_revision_smoke/cli_console.txt`。
其中发现并修复了全 mask 填充 anchor 的 BF16 反向非有限问题：空分布使用分母 1，
有效分布使用自身概率和，避免 1e-30 分母的反向下溢。失败日志保留，没有将失败当作通过。
