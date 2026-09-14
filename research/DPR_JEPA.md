# DPR-JEPA：分布预测残差

日期：2026-09-13。基于 e610110cb6d41ffde0c87a0809d58708b163838f 工作区实现。
状态：已接通模型、预训练、检查点与机制测试；训练效果、配对 token 效率及墙钟收益未建立。

## 数据流与契约

默认关闭，`--dpr --no-psr --no-ced-recurrent --mtp_depth 0` 启用。
全部 token 层保留，在 `n_encoder_layers`（默认下标 6）执行后加入残差。
边界的 shared evidence 已由原层构建；后续层及注意力候选池保持原调用路径。
读出为 `apply_hc_pre_norm(h, pre_mix, next_layer.attn_norm)`：复用下一层规范化权重，
不取 stream mean，不更新或重算已有 `pre_mix`。残差只在 hc stream 轴广播。

`model/dpr.py` 实现：

- 单个带 bias 线性头输出 M 个 `(logit, r-dimensional latent)` 槽。
  `pi=softmax(logits)`，`z=sqrt(3)*tanh(raw_z)`，默认 M=4、r=32。
- 槽位拼接 `[pi_m, pi_m*z_m]`，经无 bias、严格零初始化的 `output` 回写。
  核评分对槽位置换不敏感，拼接接口具有可学习槽身份；不宣称置换不变。
- future-only 编码器用 k=4 个按相对位置区分的 `d→w` 投影（w=128），
  求和加 bias，再 GELU、`w→r`、有界 tanh。仅 embedding 输入 stop-gradient；
  目标编码器接收辅助评分及正则的梯度。复用本次原始 embedding，不取 contextual hidden。
- 未来 target 只进入辅助 loss，任何模式都不进入当前 logits。
- `use_dpr=False` 直接跳过头和回写；推理不运行目标网络，无新增跨 token latent 状态。
  prefill/decode 保持开关；中途变更缓存的 DPR 条件会报错。
- 新模块在主干初始化完成后以独立 `dpr_seed`（20260913）初始化，再恢复主 RNG。
  覆盖真实初始化与 skip-init 的逐参数及后续随机数一致性测试。

默认 MuonH 固定矩阵范数，零矩阵会被困在零范数。因此仅
`model.dpr.output.weight` 使用现有 AdamW 无衰减组；其余 DPR 矩阵沿现有分组。
这属于研究分支配方，A/B/C/D 的公共主干优化器不改。

## 目标和统计

高斯核 `K(u,v)=exp(-||u-v||²/(2r))`，使用全部对角项：

```text
S = sum_ij pi_i pi_j K(z_i,z_j) - 2 sum_i pi_i K(z_i,y) + 1
R = ||mean(y)||²/r + ||Cov(y)-I||²_F/r
L = NTP + lambda * (mean_valid(S) + R) + existing MoE auxiliary loss
```

所有 latent 距离、核评分、均值和协方差均用 FP32。协方差分母固定 1/N。
目标只使用输入序列内完整未来窗口，因此最后 k 个输入位置没有 latent 标签，
不从最后一个 NTP label 额外补窗口。检查当前位置和每个未来位置的 PAD、segment，
并禁止窗口跨过 EOS（EOS 可以是目标窗口最后一个 token）。packed 训练要求 `--doc_mask`。
缺少完整窗口不会删除 NTP 标签。

N=0：辅助项为零，同时输出 N、覆盖率和秩不足标记；N>=1：继续计算评分与正则。
N<=r 时明确记录 `cov_rank_deficient=1`，不假称协方差能达到满秩。
`target_capacity=B*T`，coverage=N/(B*T)，包含尾窗、短文档和 padding 的损失。
统计范围为当前微批，不是整个累积窗口或独立样本集合。

lambda 以已消费的有效 NTP 标签数升温，smoothstep 从 0 到 0.05。
预算优先使用 `token_budget`；未指定时用 LR horizon 的微步数 × B × T（计划容量）。
默认升温预算为计划容量的 1%，padding/边界较多时有效标签计数到达该预算更慢。
计数包含已消费但因 NaN 被跳过更新的批次。计数及升温预算写入 checkpoint 并恢复。
模型独立 `labels=` 调用默认使用完整 lambda，训练器传入动态 FP32 lambda，避免 compile 常量冻结。

SwanLab 与本地 `dpr_metrics.jsonl` 分开记录：`lm_loss`、`total_training_loss`、
`dpr/latent_kernel_loss`、`dpr/latent_cov_loss`、MSE、N、coverage、variance、
particle spread、概率熵、lambda 与有效 NTP token 计数。console 的常规 loss 是总优化目标。
B（lambda=0）完全不运行目标与辅助评分，仅记录 CE/总损失/计数。

## 成本与边界

d=1024、M=4、r=32、k=4、w=128：新增 799,012 参数，推理调用其中 270,468。
目标参数仍保存在 checkpoint，推理时不执行。
推理主要 GEMM 为 540,672 FLOPs/token；训练完整窗口近似 3,743,744。
实际静态目标长度 T-k：训练估计为 `12*d*M*(r+1)+(4*k*d*w+6*w*r)*(T-k)/T`。
不物化 `[B,T,k*d]` 拼接。FLOPs 估计不含核评分、协方差、规范化、激活、mask、优化器与 launch；
不能解释成实际耗时或硬件利用率收益。统计 `gemm_active_params` 将 DPR 单独排除再按上述公式加入。

固定目标空间下，期望核评分等于 MMD² 加与预测无关的常数；有限四粒子仅给族内近似。
联合训练目标映射会改变该常数，所以不提供语义最优或 CE 改善保证。
零初始化只保持初始主函数；辅助梯度改变后续主干训练轨迹。
白化、非零 rank、低核 loss、概率熵都不证明有用的未来状态。

## 受控消融与否证

| 组 | 额外参数/接口 | 辅助监督 |
| --- | --- | --- |
| A | `--no-dpr` | 无 |
| B | `--dpr --dpr_loss_weight 0` | 无，目标网络不运行 |
| C | `--dpr --dpr_objective mse` | 加权均值与目标的维度归一化 MSE + R |
| D | `--dpr --dpr_objective kernel` | 核分布评分 + R |

四组均关闭 PSR/recurrent/MTP。C/D 需预注册等梯度尺度规则或同等小范围 lambda 网格；
当前 .05 是实验起点，未完成校准。至少三组 paired seeds，固定公共权重、数据顺序、
packing、优化器和日程，比较验证 CE 的等 tokens、等 FLOPs、等完整训练时间结果。
本轮按用户要求只启动 D 的 seed=1337，不能据此宣称优于 A/B/C。

`model(..., dpr_intervention="mean")` 保留槽位数量但收缩为加权均值；
`"shuffle"` 沿 batch 循环移位 pi/z，要求至少两个匹配长度/文档布局的样本。
这些接口用于后续机制干预，不是本轮已完成的质量实验。
无条件替换、结构化多跳/程序任务与验证 CE 干预评估尚未实施。

## 验证与运行

```bash
python3 scripts/check_repo.py test dpr
python3 scripts/check_repo.py
```

测试覆盖固定双峰评分、对角项、窗口掩码、future-only/梯度隔离、
原权重与 RNG、零初始化/关闭回退、非零残差下的因果性与 packed 隔离、
FP32/BF16 prefill/decode、eager/compiled 有限梯度、动态辅助权重、
真实 compiled trainer 的累积/QB/零输出矩阵更新与 checkpoint/optimizer 恢复。
本轮 DPR 与相关配置/FLOPs/检查点/mHC/训练日志回归合计 **64 passed**；
检查日志在正式运行目录的 `evidence/acceptance.log`。
这些是小模型机制验证，非语言能力证据；continuous-batch engine 搬运未单独验收。

正式运行使用独立目录 `research_runs/viby_dpr_jepa_v1`，完整命令、启动配置、源码快照、
检查日志、数据标识和运行状态记录在该目录。计划命令：

```bash
uv run trainer/train_pretrain.py \
  --data_path /Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl \
  --out_dir research_runs/viby_dpr_jepa_v1 \
  --pack_sequences --doc_mask \
  --batch_size 16 --accumulation_steps 2 --max_seq_len 1024 \
  --log_interval 1 --seed 1337 --use_swanlab --auto_resume \
  --no-psr --no-ced-recurrent --mtp_depth 0 --dpr
```
