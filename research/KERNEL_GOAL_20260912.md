# 2026-09-12：80% MFU 续轮收尾记录

用户要求尽快收尾。本轮测试、基准和调试构建均已结束；**80% MFU 未达到**。
上一轮默认启用的优化继续保留。本轮新增的性能候选均保持默认关闭，不能把
单项改善或不稳定结果当作整个目标已经完成。

## 当前配置与比较边界

本轮开始时工作树已由其他任务移除 NCP、恢复预训练默认 PSR，同时新增循环 CED。
本轮基准使用当时实际解析的默认 PSR 配方：M4 Max 40 GPU cores、48 GB、MLX 0.32.2，
BF16、12 层、dim=1024、96 experts/top-6、FP32 router、QB、Engram；循环 CED 关闭。
它与 [上一轮 NCP 结果](KERNEL_SPEED_20260912.md) 的绝对数字不能直接比较。

训练主比较为 B=4、T=1024、accum=2、packed 文档均长 200、cache limit=8 GiB。
MFU 仍使用有效 attention 长度与既有 FLOPs 估算，分母 13.5 TFLOPS 为声明的历史
GEMM 实测上限，并非本次重新标定或 Apple 官方峰值。

## 实际结果

| 候选 | 有效证据 | 结论 |
|---|---|---|
| Sinkhorn 列/行成对融合，并复用第一行范数 | 实际表形状 1966704×64、K=11；37.23→29.56 ms，配对耗时下降 20.44%，参数和动量 max-abs=0，A/A 漂移 2.82% | 单项改善确认 |
| 同一 Sinkhorn 候选的完整训练窗口 | 配对耗时下降 1.38%，A/A 漂移 1.81%；三个块比率为 1.0038、0.9729、0.9862 | 收益小，继续 opt-in |
| mHC decode split/Sinkhorn 融合 | 完整 decode logits 逐位一致，配对耗时增加 1.57% | 不启用 |
| partial RoPE decode 融合 | logits 逐位一致；单项漂移 3.92%，组合慢 2.33% | 无确认收益，不启用 |
| Window key-owned 反向 | 17 项相关数值检查通过；整模型 f+b 慢 2.28%，A/A 漂移 1.46% | 不启用 |
| 只融合 Sinkhorn 行归一化 | 完整 K=11 更新逐位一致，但慢 3.42% | 不启用 |
| AdamW 标量校正前移 | 参数/动量逐位一致，耗时比 1.0055 | 不集成 |
| AdamW 梯度布局转换、MoE gather/count 数据流 | 结果不足以确认收益，完整窗口漂移分别约 9.3% / 17.8% | 保持关闭 |
| 更大微批、相同有效 batch=24 | B4/acc6、B8/acc3、B12/acc2 的筛选 MFU 约 45.3%、43.3%、44.5% | 扩大微批未解决瓶颈，不改训练默认 |
| Metal buffer/synchronization 调度参数 | 1024 MB / 100 ops / fast-sync 的窗口 MFU 约 37.4%～37.7% | 未达到目标，不改运行默认 |

完整窗口的 Sinkhorn 对照 MFU 估算为 36.64%→37.40%，而单独 f+b 的约 52% 不能
充当完整训练 MFU。所有百分比来自各 ABBA block 配对比率的中位数；不混用两臂
总样本中位数之比。数值检查不证明长程训练质量或逐位学习轨迹一致。

## 本轮保留的修复与代码

- 基准完整恢复 PSR 独立优化器、采样微步计数与 lookahead 状态；预热通过真实
  `_optimizer_step` 路径，避免把 PSR 梯度错误交给主干优化器。
- attention FLOPs 审计显式使用 `psr_mode="off"` 统计未被 PSR 改写的主干可见集合；
  PSR 仍由配置对应的独立公式估算，没有通过停用实际训练 PSR 改善计时。
- `VIBY_BENCH_EXCLUSIVE=1` 在计时边界检查其他实验进程；遇到竞争时中止本任务的
  该轮测量，不停止别的任务。按用户选择等待 CED 基准退出后再测。
- 新候选位于 `trainer/sinkhorn_rows.py`、`trainer/sinkhorn_pairs.py`、
  `model/kernels/hc_decode.py`、`model/kernels/rope_decode.py` 和
  `model/kernels/window_attention_backward.py`。窗口独占后端仅接普通连续 token
  路径，位置感知循环 CED 保留原后端。
- 下列新增候选开关全部默认 0：`VIBY_SINKHORN_FUSED_PAIRS`、
  `VIBY_SINKHORN_FUSED_ROWS`、`VIBY_ADAM_CONTIG_GRADS`、
  `VIBY_HC_DECODE_FUSION`、`VIBY_ROPE_DECODE_FUSION`、`VIBY_WINDOW_KEY_BWD`。

未更改模型结构、读取预算、精度或优化器配方来制造达标数字。其他任务的未提交
和未跟踪文件均保留，没有提交或推送 Git。

## 检查与产物

运行目录：`research_runs/kernel_goal_20260912/`。

- `sinkhorn_pairs.jsonl`：单项原始 ABBA。
- `pairs_exclusive/results.jsonl`：PSR 完整状态恢复后的窗口比较。
- `window_owned_fb/results.jsonl`、`decode_candidates/results.jsonl`：未采用候选。
- `geometry/results.jsonl`、`runtime_1024_fast/results.jsonl`：执行配置筛选。
- `profile_b4/profile.json`：带分组同步的耗时归因，不能替代真实窗口计时。
- `start.patch`、`start-status.txt`、`base-commit.txt`：续轮起点。
- 数值检查分别记录于 `hc_decode_validation.log`（早期失败保留）、
  `rows_rope_validation.log`（21 passed）、`pairs_validation.log`（15 passed）、
  `window_hybrid_validation.log`（17 passed）、`host_validation.log`（29 passed）。
  最后定向检查结果另见 `wrapup_validation.log`。

Metal System Trace 曾受其他 GPU 实验干扰；shader timeline 尝试报告 counter profile
不受支持，未得到可用于定量归因的 shader 时间。相关 trace 保留，仅作诊断记录。
按 [MLX 官方调试方法](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
构建的 MLX 0.32.2 调试包放在本运行目录的 `mlx-debug/`，原 `.venv` 未被替换；
该包尚未用于有效的逐算子时间线分析。

后续若继续，优先使用隔离调试包定位真实原生算子耗时，再决定更大范围的改动。
此次按用户要求结束工作，不再追加新实验。
