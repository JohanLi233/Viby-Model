# 2026-09-12 kernel speed work

以下基准记录于移除 NCP 之前。当前预训练已默认启用 PSR，旧结果不代表当前 PSR 配方的性能。

本次直接优化现有 MLX/Metal 路径。80% MFU **尚未达到**。结果只代表下面的同权重、同输入基准；不是训练质量或所有形状上的速度保证。

运行环境：Apple M4 Max，40 GPU cores，48 GB unified memory，MLX 0.32.2。训练配方为 dim=1024、12 层、16×128 attention、96 experts / top-6、BF16、NCP 开启、PSR 关闭、QB / FP32 router；完整配置保存在每个最终基准的 `protocol.resolved_config`。

## 默认启用的实现

- `model/kernels/sparse_attention.py`：把 dQ、dKV 的 QK / dOK / P / Ds 计算合入同一 key-tile 遍历；按 `(head,key)` 分摊标量计算；Q/G staging 与之后的梯度 scratch 共用存储，允许 BK=32。D=128 时 threadgroup 存储 27,076 bytes。前向保持 BK=16，梯度仍返回每个 primal 对应的 array 叶子。
- `model/kernels/decode_metadata.py`：K≤256 的已选下标在单 SIMD group 的寄存器中完成 bitonic 排序。保留原 `argpartition` 的集合、不可达哨兵和 offset；更大 K 保留旧后处理。
- `model/kernels/sinkhorn_fused.py`：最多八个 4×4 矩阵时，16 lanes 各自处理一个元素，通过 shuffle 执行原来的四项顺序求和。epsilon、40 次行/列归一化及反向算法不变。
- `model/kernels/moe_decode.py`：验证后默认开启已有的纯 MoE 编译区域。权重全部是显式输入，继续使用 native `gather_mm`，沿用原有 dtype 和 expert-axis combine。这个编译实现早于本次，本次贡献是验证和默认接通。

所有开关在进程启动前读取，回退命令：

```bash
VIBY_SPARSE_ATTN_FUSED_BWD=0 \
VIBY_DECODE_SIMD_POST=0 \
VIBY_SINKHORN_SIMD_DECODE=0 \
VIBY_MOE_DECODE_COMPILE=0 <原启动命令>
```

`VIBY_SPARSE_ATTN_FUSED_BWD_TILE=16` 保留小 tile 对照；key-owned dKV 的显式开关优先于融合 VJP。单独 parallel-bwd 试验只带来约 1.6% f+b 收益，仍默认关闭。

## 测量口径与限制

有效结果如下。时间是所有样本的中位数；提速使用各 ABBA block 的配对比率中位数，两种聚合方式不混算。

| 测量 | A 中位时间 | B 中位时间 | 配对耗时下降 | 配对吞吐提升 | A/A 漂移 |
|---|---:|---:|---:|---:|---:|
| 完整训练窗口，8192 tokens | 1.4090 s | 1.3543 s | 2.97% | 3.07% | 2.61% |
| Decode，B=1，context=1024，连续 8 步 | 68.90 ms | 55.03 ms | 15.72% | 18.65% | 2.38% |

训练窗口 MFU 估算从 36.43% 到 **37.90%**，峰值 allocated memory 为 20.546 GB，两臂相同。窗口更新后 router bias 差值为零，参数最大绝对差 0.0009765625、相对 L2 差 0.0003134。Decode 全部输出 logits 的 max-abs 与 relative-L2 差都为零；峰值 allocated memory 约 2.618 GB。

证据：`research_runs/kernel_20260912/fused32_window/results.jsonl` 和 `research_runs/kernel_20260912/simd_decode_cached/results.jsonl` 中的 `variant=simd_decode`。分项 `simd_sinkhorn` 的漂移 14.59%，不单独报告它的收益。f+b 重复出现 >3% 漂移，约 4% 的初步配对收益不作独立确认。

最后复测期间发现并行运行的 `experiments/ced_recurrent_probe.py --preset default --compile --lengths 1024 --batch-size 4`，两项训练任务争用 GPU/内存，系统 swap 使用量约 15.8 GiB。只终止了本任务的 `final_window` 与后续 benchmark 调度，保留了另一实验。`final_fb` 漂移 6.87%、`final_window` 未完成、`final_decode` 未执行，均不作为验收结果。最后一个减少 metadata barrier 的额外变体虽通过数值检查，但整窗复测受干扰，因此已撤回；交付的融合 VJP 对应上述稳定 window 版本。

训练 B=4、T=1024、accum=2、平均文档长度 200、Metal cache limit=8 GiB。同进程 A/B/B/A，预热两臂，各臂来自相同参数与优化器快照；window 包含两次 f+b、梯度累加、范数/有限值检查、优化器、QB bias 更新，不包含数据读取和 checkpoint。推理使用相同 prompt、逐 token 输入和完整还原的 cache；NCP 为**已有的增量 cache 路径**，不是每步重算完整前缀的 reference 路径。

修复了旧推理基准只保存 token-cache、遗漏 NCPState 的问题：pending、concept K/V、latest prediction、filled/n_concepts 现在一并复制、恢复和求值。两臂都使用同一种增量 cache，不能把增量缓存相对全前缀重算的收益记成本次 kernel 收益。

MFU 使用仓库的 `6N + 实测有效 attention occurrences + NCP` 近似 FLOPs；分母 13.5 TFLOPS 是声明使用的历史 dense-GEMM 实测上限，**不是本次重新标定或 Apple 官方 BF16 峰值**。optimizer FLOPs 不计入 useful model FLOPs，但其时间计入 window。分母由 `--peak-tflops` 显式记录。Apple 官方提供该机型的 [546 GB/s 内存带宽规格](https://support.apple.com/en-ie/121554)，并未在该规格页提供这里使用的 13.5 TFLOPS。

MLX `atomic_outputs` 把全部输出声明为 atomic，见 [官方 custom Metal kernel 文档](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst)。融合反向因此使用一次唯一拥有的 FP32 `atomic_store` 写 dQ，再 cast 到激活 dtype；pool 的 shard reduction 与现有接口兼容。

原始样本、配对比率、首尾 A/A 漂移、逐槽 peak memory 与数值比较均在 `research_runs/kernel_20260912/*/results.jsonl`。A/A 漂移超过 3% 的记录明确标为 inconclusive，不用于确认该分项的提速幅度。不能把单独 f+b 的 MFU 写成完整训练 MFU，也不能为单 token decode 使用计算密集训练的 80% 目标。

## 复现

窗口与推理复测命令（与有效测量相同的 8 步 decode；应在 GPU 没有其他实验时运行）：

```bash
.venv/bin/python experiments/bench_csa2_plan.py \
  --run-dir research_runs/kernel_20260912/reproduce_window \
  --reference current --variants fused_bwd --mode window \
  --warmup 5 --blocks 3 --block-iters 3

.venv/bin/python experiments/bench_csa2_inference.py \
  --run-dir research_runs/kernel_20260912/reproduce_decode \
  --reference combined --variants simd_decode --mode decode --steps 8 \
  --warmup 5 --blocks 3 --block-iters 5
```

其中 `current` 是本次修改前已优化过的训练栈，`combined` 是已使用 selected-expert `gather_mm`、融合 metadata 与 Indexer 的推理栈；两者都不是“关闭全部 kernel”的弱对照。

定向检查覆盖 FP16/BF16、D=64/128、空压缩池、部分 tile、文档/pad 掩码、重复 occurrence 与共享 primal、非连续 Sinkhorn 输入、极值/零值、K=1/256/>256、不可达 Top-K、动态权重及多步 cache。最后一次聚焦相关模块的验证日志为 `research_runs/kernel_20260912/validation.log`：130 passed，3 个无关的 native-sorted MoE VJP 项未运行；重复 occurrence 的两项另行通过。

训练允许现有 BF16/MMA 舍入与 atomic 到达顺序差异，不承诺 bitwise 学习轨迹。实测 paired f+b loss 一致；窗口更新参数的误差与 router bias 检查见原始记录。没有执行长程质量训练，也没有把上述数值检查当作质量结论。

80% 所需完整窗口时间约为 **0.642 s / 8192 tokens**（本次 FLOPs 与 13.5T 分母）。当前窗口仍明显更慢；现有证据不支持声称达标。初始分段基准显示 optimizer 单次约 0.23 s，且融合后两次 f+b 的时间已高于整个 80% 窗口预算；继续接近目标需要进一步降低整体计算和训练状态处理开销。
