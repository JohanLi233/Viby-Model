# CED 边界思考态：两阶段流水线（2026-09-14）

> 后续修订：本页保留 `ced_pipeline_v1` 历史实现和结果。当前新模型显式
> `--thinking` 选择 [共享写回迭代版](TRANSITION_THINKING_20260914.md)，
> 原sidecar及显式 `--thinking_arch ced_pipeline_v1` 保留旧执行方式。


用户明确终止 NCP v1/v3 路线，授权更换机制并处理梯度不稳定。本轮保留旧源码、
检查点和研究记录，新增独立 `ced_pipeline_v1`。pilot 未过质量门槛，最终默认为
纯 CED，显式 `--thinking` 启用研究路径；历史 sidecar 不自动改架构。未建立质量/效率收益。

## 机制与预算

参考 [ByteDance PLT](https://arxiv.org/html/2510.24824v1) 的右移状态依赖，
只在 encoder/decoder 边界放一个共享小块。不是 PLT 全模型复现，也不是 RLT
无限跨 token 递归。原 CED/mHC/MoE 全 token 路径保留。默认 dim128、固定两阶段。

```text
l_t = RMS(Input(RMS(e_t)))
M_t = RoPE(l_t, t)                         # compact shared K=V, observed encoder only
s_t^1 = Block(l_t, M_<=t)
s_t^2 = Block((l_t + s_(t-1)^1)/sqrt(2), M_<=t)
signal_t = 0.1 * sqrt(mean(e_t^2)+eps) * RMS(Output(s_t^2))
```

Block 为 Q-normalized 单头 attention + SwiGLU，FP32 softmax / residual，输出
无参数 RMSNorm。状态直接服务 NTP；没有 NCP/VQ/MSE、额外词表头或 detached
未来目标。分支从首次更新即有 NTP 梯度，避免零门阻断整个思考模块的冷启动。
信号 RMS 上界为配置 scale × encoder RMS（含 eps），约束注入幅度；不宣称
限制整个模型 Jacobian 或保证梯度稳定。scale 固定默认0.1，不在本轮调参。

训练：先批量算 stage1，再按同文档右移后批量算 stage2，最后原 decoder；
没有 T 次 Python token 循环，没有 Jacobi 近似或 TBPTT。
增量：当前 l_t 与历史 stage1 均已就绪，两 stage 的 query 合在一次 block 调用中；
保留两种执行方式的逐位置数值对照。最新输入不能在同一步接受两次串行依赖的精炼。

额外 compact K=V **单独从 clean encoder 投影**，不是零成本复用原 CED compressor
张量；两 stage 共用这份额外 memory。当前实现 dense masked QK/AV，训练注意力
是 O(T²d_think)，不是 CSA2 kernel 优化，也不是固定 Top-K 等价路径。
矩阵参数额外 `2*d*w+8*w²`，每 token block 用两次，projection 各一次；FLOPs
已计入 trainer/flops.py，实际 GPU 完整窗口速度需单独实测。

## 状态与兼容性

- 未注入的边界输入仍传给原 CED compressor/index K，思考分支不改全局事实 KV。
- 注入按原 `signal/sum(pre_mix)` 提升到 mHC streams，不重算或丢弃 pre_mix。
- packed 相邻文档变化、重复文档 ID、PAD 间断均重置移位状态和可见历史。
  query/memory 用原 token RoPE 位置；纯 PAD 行输出、梯度为零。
- 新 `ThinkingCache` 持有额外 compact memory、最后 stage1 和时钟/签名；原生
  prefill/decode、chunked prefill、连续 batch 不同 clocks、prefix snapshot 支持。
  回滚使用完整 snapshot，直接 rewind 拒绝以防只回 token clock。
- `--thinking` / `--no-thinking`、`--thinking_dim`、`--thinking_scale` 经 parser、
  config、sidecar 贯通。显式旧式 `VibyConfig(ncp_enabled=False)` 保留纯 CED 语义；
  若同传 `thinking_enabled=True` 则启用新模块。新调用需要显式 thinking_enabled=True / --thinking。
- 切换 NCP → thinking 必须显式 `--resume ... --reset_optimizer`：严格保留共同
  主干权重，初始化 thinking，只允许丢弃旧 `model.ncp.*`；不继承 optimizer/步数。
  新训练建议独立输出目录；转换没有消除旧主干已受 NCP 训练影响的历史。

## Adam 根因修复

实际 v1/v2/v3 sidecar 的 beta2=0.9997499061952749，eps≈9.05934e−16，muonh=False。
当前源码此前把 m/v 及 EMA 算术留在 BF16。恒定 g=1、5000 步的解析值
`v=1-beta2^5000=0.713674332`，旧 BF16 路径停在 0.125；这不是噪声或日志尺度问题。

修复 `FusedAdamW` 的初始化、EMA、bias correction、lr 和更新算术为 FP32，最终
参数仍转换回原 dtype；覆盖单张量、同形状堆叠、shapeful 和 AdamH 继承路径。
checkpoint 新增 `optimizer_math=adam_fp32_v1`；旧语义恢复要求 reset，直接 cast
旧矩不能恢复已经丢失的历史。没有启用 grad clipping，也没有改变 Muon/路由配方。
FP32 矩增加 optimizer 内存，不能声称仍具有旧 BF16 内存成本。

该缺陷是已证实的数值问题，尚不能解释所有旧 grad norm 尖峰；本轮没有读取出
完整历史 grad norm 时间序列，因此不把所有 NCP 质量问题归因于它。

## 真实文本 pilot 预注册（首次运行前）

[thinking_preflight.py](../experiments/thinking_preflight.py) 固定 seed1337、256 updates、
batch4×length128、accum1，每条件131,072有效NTP标签；验证256独立文档前缀、32,768标签。
同一原始数据源 `/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl` 和本仓库 tokenizer；
按源行号模5划分 train/eval，去除重复 token 前缀，保存所选原文/行号、tokens、顺序。

6层 dim128 MoE小模型、thinking_dim32；不启用 Engram，不 packing，全部有效长文档
前缀。使用实际 BaseTrainer 编译 loss、混合 optimizer、QB 统计与窗口更新；
BF16 GPU，FP32 Adam，warmup32，Adam lr0.001、Muon lr13/3×0.001，beta2/eps同旧运行，
关闭 MuonH 和 clipping。它不是用户 dim1024/batch16/accum2 原规模配方。

三条件从逐张量相等的共同主干初始化开始、相同数据次序：正常思考态、每步 reset
思考传递（同参数/逻辑运算）、纯 CED（参数和计算较少）。reset 是仅 probe 中的
子类，不作为新的生产架构；其权重不可冒充普通 thinking checkpoint 恢复。
正常模型另做 reset/swap 干预。只采用最终 checkpoint，不按验证集选步数。

门槛：所有更新/指标有限，grad max/median<=20；正常减纯 CED 的验证 NLL<=−0.01、
正常减 reset<=−0.005，且两者配对 bootstrap 95% CI 上界<0。通过仅允许独立种子
确认及进一步小模型验证；未通过停止长预训练，不据此宣布 token 效率改善。

## 已知数值验证边界

新模块在固定 BF16 encoder 输入下，整段与增量本轮观测一致；完整 FP32模型也一致。
但完整 BF16主干已有形状/离散路由敏感的 prefill-decode 差异：固定 tiny 权重下，
纯 CED最大 logit差约0.2314、新路径约0.2344；均值分别约0.00755/0.00483。
不能将此写成 BF16完整 logits逐元素相等。测试对新模块做严格检查，对完整BF16路径
与同权重纯 CED 对照比较误差；实际生成分布一致性仍是限制。

## Pilot 结果与最终默认决定

运行目录：`research_runs/ced_thinking_20260914/pilot_seed1337/`。
每条件均实际完成256次 optimizer updates，131,072 NTP标签；三条件共393,216。
没有追加种子或调整超参重跑同一验证集。

| 条件 | 验证 NLL | grad norm 中位数 | grad norm 最大值 |
| --- | ---: | ---: | ---: |
| 新思考态 | 6.687040 | 0.940953 | 2.711617 |
| 从头训练的 reset 对照 | 6.687406 | 0.949856 | 2.746305 |
| 纯 CED | 6.688504 | 0.950179 | 2.812016 |

新机制减纯 CED NLL=−0.001464，95% CI [−0.005295,+0.002410]；
减训练 reset 对照=−0.000366，95% CI [−0.004546,+0.003693]，均未过门槛。
在正常模型上，评估时 reset / swap 使 NLL 分别增加0.005536/0.007538，CI均不跨零。
这支持当前模型利用了该状态，但不能证明该结构比单独训练的对照更好。

结论为 `stable=true, passed=false, stop_no_large_pretraining`。在实现阶段曾临时
将新机制设为默认；根据上述结果，**最终默认是修复后的纯 CED**，NCP不再自动启用，
新机制保留 `--thinking`，不把新的未证实路线自动交给下一轮昂贵预训练。

配置见 manifest.json，原文和源行号见 selected_documents.jsonl，token数组见data.npz，
训练顺序见train_order.npy，每文档评估见per_document.npz，各条件step日志和权重分别保存。
各条件 elapsed_seconds 含首次编译，运行顺序固定，没有A/A或ABBA验证，不作性能结论。

## 原规模检查点数值门槛

使用 `research_runs/viby41_ncp3/pretrain_1024.safetensors` 的dim1024/12层原主干，
去除NCP并随机初始化新thinking，读取pilot的同一个B1/T128文档；没有更新任何权重。
检查脚本：[thinking_checkpoint_check.py](../experiments/thinking_checkpoint_check.py)。

| 路径 | direct/VJP primal loss差 | 梯度 | 判定 |
| --- | ---: | --- | --- |
| BF16 GPU，thinking | 0.001708984 | 有限，主干norm5.058829、thinking0.366232 | 未过1e−5门槛 |
| BF16 GPU，纯CED同检查点 | 0.001220703 | 有限，主干norm5.084301 | 同样未过 |
| BF16 GPU，关闭sparse attention自定义核 | 0.001708984 | 有限 | 仍未通过，不能单独归因此核 |
| FP32 GPU，thinking | 0 | 有限，主干norm5.069674、thinking0.366076 | 通过该微批数值门槛 |

原始结果分别为运行根目录 `full_width_numerical.json`、`full_width_ced_numerical.json`、
`full_width_dense_reference.json`、`full_width_fp32_reference.json`，每份有配置/环境和日志。
FP32峰值约9.31GiB，BF16约5.97GiB，均不是包含Adam矩的训练峰值。
全部源检查点mtime/size保持不变。这些检查只覆盖短B1微批，不覆盖原B16/T1024累积窗口。

原规模BF16问题存在于无thinking的对照，关闭单个sparse核未消除，FP32该样本无差异；
尚未定位单一算子，不能宣称所有梯度不稳定已根治。因质量gate与此数值gate均未通过，
没有启动原规模预训练，也没有把FP32参考结果当作BF16验收。

## 实际检查与限制

- `python3 scripts/check_repo.py doctor/list` 已执行；默认静态检查和 `git diff --check`通过。
- 定向回归89项通过，另1项SFT sidecar迁移检查通过，涵盖thinking因果/梯度/状态快照/连续batch/DSpark恢复、旧NCP、
  FP32 Adam解析EMA、checkpoint、FLOPs和配置。日志：`final_regression_with_resume_guard.log`；SFT迁移单项另执行通过。
- 新文件 Ruff 检查通过；扩大到已有 trainer/muon.py 的lint仍报告两处原有 `O` 变量名
  E741（HEAD中已存在），未作为本次无关修改整理。
- 单独Metal测试最初因测试函数未恢复compile捕获的模型参数占位报错；修正测试的
  参数恢复后通过，保留metal_failure.log。不得把这次测试夹具错误归因于模型算法。
- 大模型审计脚本首次因导入路径拼写错误退出；修正后上述实际检查全部完成。
- Adam矩从BF16升FP32的常驻开销，按原v3检查点482个m/v张量测算约增加3.526GiB；
  不含临时分配和移除NCP后的差额，见adam_precision_audit.json。历史BF16 optimizer
  不能靠cast恢复丢失的EMA；必须显式reset。
