# 任务锚定的短块潜在推断：冻结 CED 否证入口

日期：2026-09-13。实现版本 `task_latent_v1`。这是潜在变量语言模型实验，
不是 DPR/JEPA 几何辅助目标。没有修改预训练默认算法，没有启动完整主干续训。

## 本轮规格与实现

[分支与原生状态](../model/latent_inference.py)、
[配对运行入口](../experiments/latent_frozen_probe.py)、
[机制检查](../tests/test_latent_inference.py)。

- 保留全部 CED 层及 token 分辨率。`model.model(...,
  return_latent_features=True, use_dpr=False, use_ced_recurrent=False)` 只导出特征。
  `a` 是 CED 边界执行后的下一层 mHC 规范化读出；evidence 是边界注意力
  真正接收的 mHC 读出。导出不写残差，不改变已有 pre_mix、选择或 MoE。
- `k=4, M=4, r=64, R=2`，两个共享权重的 attention + SwiGLU 更新。
  使用两个 d→r K/V 投影，保持完整 token evidence。候选为边界原有 CSA2
  选择的最多 64 个 token 加最近 64 个 token，排序去重并重新验证因果、
  同文档、EOS 与 PAD。上限 128；不是将 CSA2 的原 top-k 改成 128。
  第二次只在此集合重新加权，不能声称全历史自适应检索。
- 本探针不新增 latent RoPE/位置嵌入；位置结构来自 contextual evidence。
  原 CED RoPE、组完成时间及原索引语义保持。这里边界 ratio 必须是 1。
- 输出是 `sum_i w_i softmax(base_logits + U(sigmoid(Wg d)*z_i))`。
  无 gain RMS 规范化 z；U 初始每行范数为 `epsilon/sqrt(r)`，epsilon=1e-3，
  slots 独立非对称初始化。输出分量共享 U 和 gate，没有拼接粒子残差。
- 唯一优化目标是 token 似然。A 使用 exclusive cumulative log likelihood
  的因果 filtering；总 token NLL 精确等于块 logsumexp NLL。
  B 每个 token 重置为 prior。不存在额外 kernel/KL/白化/熵损失，也不 detach
  posterior 梯度。主干冻结时原 MoE 辅助项为常数，不参与分支优化。
- 每个有效 NTP label 只计一次。块不跨无效标签、EOS、文档或 context 尾部，
  尾块照常计分。上下文片段最多 1024 个输入 token，长文档续片重新 prefill；
  bootstrap 仍按原始文档聚合这些片段。
- 分支 FP32；冻结主干及 head 保留 checkpoint dtype，head matmul 后的
  softmax、似然与累计均 FP32。每次最多 64 块（256 token）计算词表分量；
  用 MLX checkpoint 重算 softmax 的反向中间激活，而非仅靠 Python 分块。
- `LatentRuntime` 支持单序列原生 prefill/decode：缓存投影 K/V、slots、
  log weights、已观察计数以及待观察 token 的分量概率。prompt 结束不重置；
  新观察先更新权重，满块刷新，新文档清空 CED/Engram/latent 缓存。
  返回归一化 log probabilities。未接入 continuous-batch engine、投机回退、
  prefix cache 共享或通用预训练 CLI；本轮入口专门训练冻结分支。

## 固定的否证协议

正式目录：`research_runs/task_latent_frozen_20260913/`。

基座：`research_runs/viby_dpr_jepa_v2/pretrain_1024.safetensors`，保存于 epoch 0 /
microstep 20036。加载原权重后显式关闭旧 DPR、PSR、recurrent 与 MTP。
它是曾受 DPR 训练的 checkpoint 的 ordinary-CED 路径，不是从未接触 DPR 的
另一个独立训练基线；因此结论仅适用于这一冻结表示。

数据：从 `/Volumes/pan/text/pretrain_t2t_dedup.jsonl` 依次选择新文档，
先排除 checkpoint 训练过的 `pretrain_t2t_mini_dedup.jsonl` 的 1,269,983 个
规范化文档 SHA256，再排除新片段内重复。规范化仅合并 Unicode 空白。
这不构成全互联网或近重复/子串去重证明。文档哈希首 64 bits 模 11 等于 0
进入验证，其余训练。恰好 1,000,000 / 100,000 个有效 label，5,158 篇文档。
原始源文件读到 line 6174 / byte offset 4729379；记录原始行字节范围、
raw/text SHA256、tokenizer 哈希、排除语料哈希、source stat 和精确 token 数。

Engram 的 rolling hash 本身不使用 segment_ids。为满足文档隔离，缓存时将
每个文档片段放入独立 padded batch 行，并把已计算特征拼回磁盘 shard。
没有修改全仓 Engram 实现。PAD 不可见；A/B/CED 共享相同隔离后的特征。

只进行一次单遍 A/B 配对：相同逐参数初始化、相同 token 顺序，每个 shard
更新一次 AdamW，FP32 m/v，常数 LR=1e-3，betas=(.9,.999)，weight_decay=0，
无 clipping、无搜索。奇偶 shard 交替 A/B 更新顺序。正式运行 eager，词表重算
启用；主干冻结且不属于两个优化器。正式运行与 GPU 检查串行。

完成后在同一独立验证集统计 A、B、冻结 CED、A 的第二次 query 固定为第一次
四个 CE，保存逐文档 NLL/count。2000 次固定 seed 文档配对 bootstrap；
最低统计门槛是 A−CED 与 A−B 的 95% 区间上界都 <0。报告同步墙钟、缓存时间、
峰值内存，不将 FLOPs 解析估计当墙钟结果。通过统计门槛也只说明冻结接口收益，
后续共同主干续训需另行固定数据、optimizer、LR 和总预算，不自动升级为长期训练。

未通过则停止本路线本轮投入，不调粒子数、秩、循环或增加熵正则补救。
第二次 query 干预仅检查贡献，分布外干预下降不是优越性证明。

## 运行、证据和限制

```bash
python3 scripts/check_repo.py test latent
.venv/bin/python experiments/latent_frozen_probe.py \
  --run-dir research_runs/task_latent_frozen_20260913 \
  --checkpoint research_runs/viby_dpr_jepa_v2/pretrain_1024.safetensors \
  --data /Volumes/pan/text/pretrain_t2t_dedup.jsonl \
  --exclude-data /Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl
```

查看 `status.json`、`train.log`、`training_metrics.jsonl`；结束后结果在
`results.json` / `validation_documents.jsonl`。`data/*.npz` 是精确 token 样本，
`data/manifest.json` 是来源清单；`features/*.safetensors` 是实际训练/验证特征，
没有缓存全词表 logits。`resolved_config.json` 含最终结构、dtype、优化器、
checkpoint SHA256、Python/MLX/硬件。A/B 权重与各自优化器每 100 步保存，
本入口拒绝覆盖已开始的 run，不支持静默恢复。异常保存 `failed` 及 traceback。

`task_latent_smoke_20260913` 为真实 checkpoint 上 2048/1024 token 的接线
试跑，完成缓存、A/B 更新、保存和文档统计；其小预算 CE 与自动统计字段不能
用于上述正式门槛。随后 head dtype 对齐到原 checkpoint 路径，正式结果以最终
源码快照为准。没有根据 smoke CE 调整优化器或模型规模。

FP32 原生 prefill/decode 做严格数值对拍。BF16 小模型在不同 prefix shape 下
主干自身有不同舍入路径（首次探针最大 log-prob 差约 0.16）；测试用同状态
ordinary-CED 对照隔离新增分支差异，未声称整模型 BF16 严格前缀一致。
eager/compiled 有限非零梯度、精确边际/过滤梯度、重算梯度、文档/未来隔离、
尾块计数、初始化界、原权重输出不变、状态保存与 EOS 重置均有针对性测试。

本实现与测试不证明新 latent 优于 CED、不证明预训练 token efficiency，
也不提供完整主干训练或 M4 Max 推理加速结论。正式结论以同目录结果为准。

## 本轮完成结果：2026-09-13 18:39 +08:00

正式进程 PID 18657 已退出并完成 982 个 shard；A/B 各训练 1,000,000 个有效
token，在 470 篇独立文档的 100,000 token 上验证。24 项针对性机制/回归测试
通过（2.84 秒），默认静态检查、diff whitespace、数据计数和来源清单核对通过。
运行时源码与 `source_hashes.json` 一致；本节是运行后新增记录。

| 条件 | 独立验证 CE |
| --- | ---: |
| 冻结 CED，关闭旧 DPR | 1.975206659 |
| B：静态输出混合 | 1.950571467 |
| A：块内 posterior filtering | 1.950327330 |
| A：第二次 query 固定为第一次 | 1.950327867 |

按同一文档配对 bootstrap，A−CED 为 −0.024879329，95% CI
[−0.026459228, −0.023351854]；A−B 为 −0.000244137，95% CI
[−0.000427494, −0.000065944]。本轮最低统计门槛通过，但 **A 相对 B 的增量
很小，大部分增益也出现在静态输出混合中**。不能把全部 −0.02488 归因于状态维护。

固定第二次 query 的 CE 变化为 +0.000000537，95% CI
[−0.000008480, +0.000009623]，没有测出自适应再读取的可检测贡献。
不能据此宣称第二轮完成了有用推演。

分支实际参数数 704,832。A/B 同步分支训练时间分别 10.400 / 10.177 秒，
特征生成及写盘/训练前重载共 88.365 秒，最终验证 1.146 秒；峰值 Metal
内存 5.975 GiB。A/B 分支训练时间比约 1.022，仅描述该冻结分支工作量，
不构成完整 CED 训练墙钟增量测量。总运行起止与核对在 `completion_audit.json`。

保存的 `A_filter.safetensors`、`B_static.safetensors` 及各自 optimizer 是完成
1M token 的版本。`training_checkpoint.json` 记录 step 982，所有验证文档 NLL
重算后与 `results.json` 一致。原 checkpoint、旧 DPR 运行及预训练默认值均保留。

结论限于这一冻结表示、这批新文档与一个 seed：输出接口有增益，filtering
有小幅增量；独立再读取尚无贡献证据，主干学习的 token / compute efficiency
未测。本轮没有自动启动长期训练，也没有用新增 sweep 或正则追求更大差异。
