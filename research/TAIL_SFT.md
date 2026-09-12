# TailSFT 接入契约（2026-09-12）

依据 [TailSFT: Filtered Fine-Tuning Improves Post-Training Performance](https://arxiv.org/html/2608.25756v1)
的 Algorithm 1 与 Appendix D.1。此页说明 Viby 的实现和适用范围；论文中的
OLMo/GRPO 收益不是 Viby 的实验结果。

## 默认入口和算法

`trainer/train_full_sft.py` 默认 `--sft_algorithm tail`，原路径可用
`--sft_algorithm standard`。默认过滤比例 0.5、static 日程是本仓库的起始配方，
尚未在 Viby 上调参。其他学习率、优化器默认值沿用现有 SFT。

对每条样本，先用初始模型算 assistant 监督位置的平均 CE `l0`。训练时同一次
前向得到 `lt`，按 `lt - l0` 从小到大过滤已进步最多的序列。仅分数按序列长度
归一化；保留序列的 CE 总和除以保留的监督 token 总数，形成训练目标。
参考损失与筛选 mask 不接收梯度。词表投影分块进行，不产生完整 B×T×V logits。

筛选批次为 MLX 单设备的一次前向微批，大小由 `--batch_size` 决定；累积窗口
的各微批独立筛选并平均梯度。论文多设备的 selection batch 对应所有 rank 的
同一前向批次，这里没有数据并行 rank。计数使用 round-half-up；同分按批内次序，
过滤比例范围 `[0,1)`，至少保留一条有效序列。单样本微批不会被过滤。
最后不足 `accumulation_steps` 的窗口按实际微批数平均并执行更新。

`--tail_sft_schedule static` 固定比例；`ramp` 从首微步的 0 升至计划末微步的
目标值，计划长度取 `min(epochs × 每轮微批数, max_steps)`，恢复后按原全局微步继续。
只有一步的 ramp 保持 0。`max_steps` 仍在优化器窗口边界检查。

## 数据与其他目标

TailSFT 固定每条样本的随机预处理：用 `seed + 原始行号` 的独立 RNG，初始评分、
shuffle 和后续 epoch 始终采用相同 token/监督 mask，不修改全局 Python RNG。
保留现有模板、截断规则和多轮对话所有 assistant 内容（含结束标记）的监督；
这与论文仅监督最终 assistant 回复的实验设置不同。一行对话作为一个筛选单位。
超长样本仍按 `max_seq_len` 截断，筛选的是该截断后的样本；没有监督 token 的
行在评分后排除，打印数量。最后不足 batch_size 的批次仍参与训练。

流式打包会把长回复拆成多个块，并在同一行混合不同样本，故 TailSFT 显式拒绝
`--pack_sequences` 和 `--doc_mask`。普通 SFT 的打包接口保留。

模型的 z-loss 和 MTP（如果基座包含 MTP）使用保留序列的 loss mask。原有 MoE
序列均衡辅助项和每次物理调用的路由统计仍覆盖实际前向 token；QB 仍仅在
累积窗口末尾更新。这些辅助项、优化器与多轮监督属于 Viby 配方；需要纯论文
CE 对照时显式用 `--z_loss_weight 0 --aux_balance_loss_weight 0 --mtp_depth 0`，
并匹配基座结构。SFT 的 PSR 仍为 off；拒绝冻结主干的专门训练模式。

## 缓存和恢复

默认缓存 `.cache/tailsft_<identity>.npz`，可指定 `--tail_sft_cache PATH`。
身份包含基座权重与数据 SHA-256、tokenizer backend/模板/特殊 token、长度、seed、
dtype、最终模型配置、相关源码哈希和 VIBY 环境开关。缓存仅包含逐序列初始损失、
监督 token 数与身份，不持有第二份模型。新缓存先写临时文件再原子替换。
`--no_save` 禁止训练产物，仍允许数据/参考损失缓存。

初始评分发生在 BaseTrainer 恢复 SFT 权重之前；缓存缺失时从原始基座重算，
不会用当前训练模型重新定义 `l0`。缓存错误或非有限初始损失会直接失败。
sidecar 的 `args.tail_sft_reference` 保存参考身份细节，`args.tail_sft_state` 保存
缓存数组哈希、筛选和批次日程契约，`rng` 保存恢复 shuffle 所需状态。
继续训练仍需提供原始 `--pretrain_checkpoint` 及相同数据/配方。

```bash
# 新 TailSFT 运行（pretrain_checkpoint 推荐使用绝对路径）
.venv/bin/python trainer/train_full_sft.py \
  --pretrain_checkpoint /path/to/pretrain_1024.safetensors \
  --data_path /path/to/sft.jsonl --out_dir out/tail_sft \
  --batch_size 16 --tail_sft_filter_fraction 0.5 --tail_sft_schedule static

# 同一运行恢复：其余数据、基座、epochs、batch/accum 和 filter 参数保持原值
.venv/bin/python trainer/train_full_sft.py \
  --pretrain_checkpoint /path/to/pretrain_1024.safetensors \
  --data_path /path/to/sft.jsonl --out_dir out/tail_sft --auto_resume

# 把旧 SFT 作为新 TailSFT 的初始策略；重置优化器/进度，重新评分
.venv/bin/python trainer/train_full_sft.py \
  --resume /path/to/full_sft_1024.safetensors --reset_optimizer \
  --data_path /path/to/sft.jsonl --out_dir out/tail_from_sft
```

改变算法或参考/筛选契约不能静默继承优化器；显式 `--resume ... --reset_optimizer`
建立新运行。TailSFT 遇 Ctrl-C 保留最近已经落盘的 checkpoint，不将未完成窗口
或更新中的状态覆盖到该 checkpoint。保存名仍为 `full_sft_<dim>`，DPO 无需改入口。

## 验证与证据边界

针对性入口：`python3 scripts/check_repo.py test sft`。
测试检查 FP32/BF16、eager/compile 的 loss 与梯度、筛选方向与 ties、空目标、
缓存身份/复用、预处理确定性，以及真实模型的编译训练和尾部累积窗口。
日志报告 `tail/sequences`、`tail/retained_sequences`、`tail/retained_tokens`、
`tail/unfiltered_ce`、`tail/mean_offset` 与计划过滤比例；筛选后的 CE 不能直接与
普通 SFT 的全样本 CE 比较，`unfiltered_ce` 只表示同一次前向的全有效样本口径。

机制通过不代表 pass@K、DPO、RL、吞吐或最终能力提升。未运行长训练或性能实验。
后续质量实验应以 `--sft_algorithm tail --tail_sft_filter_fraction 0` 为无过滤对照，
匹配初始权重、固定预处理、数据、优化器和预算，再独立评估 pass@1 / pass@K。
