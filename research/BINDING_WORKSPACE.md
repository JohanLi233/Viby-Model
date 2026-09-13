# 可改写绑定记忆与两步组合读取

版本 `binding_workspace_v1`，2026-09-13 实现，运行标识/seed `20260914`。
基于用户提供的 GPT Pro 规格；研究范围是合成关系组合，不是自然中文效率证明。

## 机制与审查结论

从 CED 边界真实 attention 输入 e 写 delta 记忆，从该层执行后的下一层
mHC 规范化读出 a 构造 query。两个 r=64 的 FP32 记忆银行，共享 key/value
坐标；每 token 只写一次，再在同一 S_t 上做两次读取。第二步用 q1 作为地址；
控制组改用 q0，但保留依赖 q1 的银行控制和两份输出。没有额外 FFN 或词表头。

固定 QR 正交桥 P 将 concat(q1,q2) 投影到 d=1024；注入下标
`n_encoder_layers+1` 层的 attention 输出，随后走原 `hc_post`、FFN 与后续
全部 token 层。原 CSA2 KV 源、pre_mix、位置与索引路径保留。只用原 NTP +
原 MoE 项，所有主干与新参数正常更新，不增加辅助目标或冻结 decoder。

写入 S'=S+(beta*(v-Sk))k^T；key/value/query/读取结果投影到单位球。
参数量 202,886；固定 P 131,072 个 FP32 数，不进入优化器。
新头采用正常 fan-in 初始化，独立 NumPy seed；不消耗主干随机流。

这比末端 logits 适配具有明确的组合地址接口，但有两个解释限制：

- e_t 本身已经包含当前问题上下文，写入器可以在问题期间改写记忆。
  不能从两跳准确率或非零 q1→q2 效应直接推断“事实记录被忠实绑定”。
- 完整 CED 和 q1 的直接桥接都可以处理部分组合问题。完整组只赢 baseline
  而不赢固定地址控制，不支持第二步组合地址是增益来源。

用户指定的数据任务/三臂不因此改成另一种架构。干预仅用同一事实阶段中
**较早**问题的 q1 作为第二次地址；原 q1 交给 decoder 的路径保持。
更晚问题的状态不作为 donor，避免把未来状态混入这个因果干预。

## 扫描、梯度与缓存

前向用工作量 O(T H r^3) 的 reduce/scan/reconstruct affine scan，非逐 token
Python 循环，也非 Hillis–Steele 的 O(T log T) 工作量。保留顺序参考实现。
遇新文档设置 A=0，PAD 设置有效 beta=0。状态与扫描全程 FP32。

本机 MLX 0.32.2 对递归交错扫描直接自动微分，出现前向一致、梯度不一致；
例如最小测试 k[0,6,0] 自动梯度 −0.680743，而顺序/有限差分约 −0.768571。
已采用显式解析 VJP：用逆序 affine scan 求伴随，再求 k/v/beta/初态梯度。
每个 array primal 返回一个 array 梯度，包括布尔 metadata 的零占位。
这不是靠放宽梯度容差掩盖问题；所有浮点输入梯度与顺序参考对拍。
高阶导数不属于当前验收范围。

原生 prefill/decode 保存最终 S 与上一 token，改变 full/fixed 条件拒绝复用
缓存；EOS 后整份原生 CED/Engram/记忆缓存重置。文档混合 cached prefill、
rewind 和 continuous-batch engine 明确拒绝。独立实验的 loader 重建工作区；
没有改变通用预训练 CLI 的默认架构或伪装成 DPR checkpoint 自动恢复。

## 固定数据与预算

运行目录：`research_runs/binding_workspace_20260914/`。
生成器：[binding_data.py](../experiments/binding_data.py)。使用原 tokenizer
核验的单 token A–H（IDs 35–42），八个实体、两个随机映射；事实顺序随机，
两阶段各六个问题，中间一次覆盖写入。训练有 f/g/ff/fg/gg，确认集只有 gf
且世界重新生成。单 hop 是普通计费文本，不给内部地址或银行标签。

三臂：baseline、full、fixed。每臂 **524,288** 个唯一有效 NTP label，
global batch **2048**，**256** 次 optimizer 更新。训练 1217 个世界；开发
16 个、确认 128 个。世界最长 456 个输入 token，小于指定 1024 上限。
逐文档独立 padded batch 行，避免 Engram 跨文档；物理 batch=1。

更新边界切过文档时，该文档会被重算，但 loss mask 不相交，每个 label
只计一次；重复的物理上下文和 PAD 算量另记。数据、world/question 元信息、
tokenizer 哈希、逐窗口 label 起止游标在 `data/`。旧自然中文 100k 验证集不使用。

原 checkpoint 为 DPR v2 microstep 20036。每臂从同一文件加载所有原权重、
optimizer、半径、组时钟、MoE bias/RNG，并核对原参数分组列表。DPR/PSR/
recurrent/MTP 在前向显式关闭；休眠 DPR 参数保持原分组，防止按数组位置
错误搬动原 optimizer state。新数据游标独立从 0 开始，不重放旧数据。

主干各组 LR 固定保留源 checkpoint 当前值，不重构 epoch LR horizon；
新参数独立 FP32 AdamW，LR=.001，betas=(.9,.999)，无 weight decay。
P 不训练，无新增全局裁剪；坏 loss/梯度立即失败。所有微批先按有效标签数
加权，完整窗口才更新 optimizer 和 MoE QB。只有一个联合反向。

## 资源门槛与结果口径

启动训练前使用真实 checkpoint、相同权重/optimizer/输入窗口做 ABBA，
warmup=3，每槽5次，3组。每次窗口测量之前恢复参数与 optimizer 容器，
不做 FP32 参数副本。基线首尾漂移>3% 时不接受性能结论；相对基线训练
窗口开销>20% 或峰值 Metal>=40 GiB 时停止。不是用多余 baseline 计算凑平。

三臂 GPU 串行运行；单臂40分钟安全上限，三臂总上限两小时。
保存 step 128/256 权重，最终 optimizer，0/128/256 开发评分；只在训练完成
后读确认集。按 baseline 预算保存 full/fixed 的时间与算术预算前缀。
FLOPs 是依据实际物理长度的名义 GEMM/attention/scan 账，明确不包含索引、
标量 kernel、optimizer、I/O；不能冒充 profiler 测得的完整 FLOPs。
匹配墙钟独立比较；代码编译和保存耗时须与训练同步时间分开解释。

主门槛：full 对两个对照均降低答案 NLL 至少 **0.05 nats**，并提高答案
准确率至少 **5 个百分点**。报告2000次按世界配对 bootstrap；不等于 seed
区间。结果不达门槛就停止该变体，不增大数据、维度或事后改确认集。
通过也仅说明该合成关系生成任务上的机制/有限样本表现，不证明自然中文
CE、全面推理能力或从零预训练效率。

## 入口与证据

```bash
python3 scripts/check_repo.py test binding
.venv/bin/python -u experiments/binding_probe.py \
  --run-dir research_runs/binding_workspace_20260914 \
  --checkpoint research_runs/viby_dpr_jepa_v2/pretrain_1024.safetensors
```

`--phase benchmark/train/summary` 为独立入口；全流程仅在资源门槛通过后串行
启动三臂。每臂新目录必须不存在；没有静默续跑或覆盖其他实验。
`status.json` / `train.log` 查看阶段；`benchmark.json` 保存原始时间；
各臂 `metrics.jsonl`、`training.json`、`confirmation.json` 及模型/优化器是
原始证据，最终 `results.json` 汇总门槛。源码、补丁与哈希随启动存档。

原始机制来源：[Fast Weight Programmers](https://arxiv.org/abs/2102.11174)、
[DeltaNet 并行化](https://arxiv.org/abs/2406.06484)。核对的
[MLX-LM 版本](https://github.com/ml-explore/mlx-lm/blob/dcbcf786c0cf56f9a12fabe9468c887781431ae2/mlx_lm/models/gated_delta.py)
提供顺序参考/Metal 路径，但本实验没有把它冒充为已经实现双次读取的高效训练器。

## 完成结果：2026-09-13 22:24:42 +08:00

三臂均正常完成256次更新，各524,288个有效标签、657,696个实际输入位置
（含PAD和跨更新边界重新计算的上下文）。全部记录中的loss/梯度范数有限，
未触发数值或资源停止条件；进程已退出。35项针对性检查通过。
运行后的逐世界计数/NLL核对通过，核对时源码哈希与启动快照一致。

确认集为128个新世界、1536个未见操作顺序gf的答案：

| 条件 | 答案 NLL | 答案准确率 |
| --- | ---: | ---: |
| 普通 CED | 1.866547878 | 30.4036% |
| 完整绑定工作区 | 1.864273726 | 30.2734% |
| 第二次地址固定控制 | 1.870186408 | 30.6641% |

完整组减普通CED：NLL −0.002274151，95%世界配对区间
[−0.012740414, +0.008398542]；准确率 −0.1302 个百分点。
完整组减固定地址控制：NLL −0.005912681，区间
[−0.015017401, +0.002749142]；准确率 −0.3906 个百分点。
两组NLL差异区间均跨零，准确率也没有提高。预注册的NLL至少降低0.05、
准确率至少提高5个百分点门槛未通过，结论为
`stop_this_variant_no_automatic_extension`。

成本方面，同状态完整窗口ABBA测得配对增加2.86%，基线首尾漂移0.535%，
测量峰值23.09 GiB。三臂实际同步训练分别324.153、336.684、336.021秒；
完整组全程比baseline多3.87%，其余加载/评分/保存时间不包括在该训练计时内。
正式训练峰值分别16.710、16.855、16.855 GiB；benchmark持有用于状态恢复的
额外引用，不能把两个峰值之差当作优化收益。

按基线训练时间选出的完整组前缀为step246 / 503,808有效token，NLL
1.849889308，准确率29.6875%；固定地址控制为step247 / 505,856token，
NLL1.845558423，准确率30.4036%。完整组前缀虽比最终baseline的NLL低，
但准确率没有提高，而且未优于固定地址控制，不能把它归因于组合地址机制。
名义算术预算的两工作区前缀均为step252 / 516,096token：完整组NLL
1.860018670，准确率29.6224%；固定组NLL1.858391651，准确率29.7526%。
这些前缀由预算决定，并未按确认集loss选择。名义FLOPs不包含完整硬件工作。

无重训地址干预覆盖1211个答案位置。替换成同阶段较早问题的q1后，
相应donor后继答案的平均log概率变化为−0.000873037，95%世界配对区间
[−0.002859067,+0.001160099]。没有测出预期的向donor后继移动的正信号；
这不是“工作区完全没被使用”或“所有绑定记忆都无效”的证明。

原始汇总在运行目录的`results.json`；逐世界记录在各臂`confirmation.json`，
完成核对与预算比较补充在`completion_audit.json`和`matched_budget_audit.json`。
源码、两种工作区、权重和optimizer保留，不自动追加训练或调整任务难度。
当前只否证这份具体规格在当前基座、预算和生成任务上的研究收益，未检验
自然中文预训练效率，也没有证明整个fast-weight/latent-thinking方向无效。
