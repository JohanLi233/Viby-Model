# 当前查询驱动的 latent 迭代：机制筛查（2026-09-14）

状态：预注册、独立实验；不改变 NCP v1/v2/v3、生产配置或已有训练。
用户观察 v3 似乎不如 v1；本轮未取得匹配 checkpoint / held-out 数据的 v1-v3
比较，不能把该观察写成已确认的因果结论。

## 为什么换问题

v3 当前源码的预测为 `c_t + r_theta(s_t)`，辅助目标为
`||r_theta(s_t) - sg(c_{t+1}-c_t)||²`。它解决了初始预测不能 copy 的参数化问题，
没有改变平方误差的条件均值最优解：给定历史 H，最优残差仍是
`E[c_{t+1}|H] - c_t`。目标比 copy 好不蕴含 NTP 好，反之亦然。

代码中的差分目标 detach，不意味着预测器上游永久无梯度：残差输出矩阵离开
零初始化后，NCP 梯度仍经残差头、概念层和输入进入 encoder。因此现有文档中
“v3 只更新残差头”的表述过窄。VQ 目标的概念和选择 detach，只训练 codebook，
也不能据此说 VQ 正在把 encoder 拉向均值。这里记录源码审阅结论，未改写旧问题包。

此外，encoder 输出本身含因果位置与上下文，均值池化并不等于完全没有顺序信息；
它确实损失组内显式分辨率，但不能仅从“均值”推导整个模型对 token 排列不变。

## 最小机制

把 latent 定义为**当前问题尚未完成的计算状态**，让它直接服务当前答案：

```text
M_t = observed causal evidence through position t
z_t^0 = current token hidden
z_t^(r+1) = F_theta(z_t^r; M_t)
p(x_(t+1)) = Decoder(z_t^R)
L = next-token cross entropy (+ unchanged backbone routing objective in a later integration)
```

F 的 cross-attention query 来自 z^r；前次结果改变下次读取，所有轮参数共享。
M 在内循环中只读；不把预测的未来当作已观察证据。不增加 PQ、未来 MSE、
辅助词表头、多个门控、teacher 或可学习停止器。训练完整反传全部轮，推理执行同一递推。

这属于已有的多跳记忆 / recurrent-depth 路线，不主张发明新的基本算法：
[End-To-End Memory Networks](https://arxiv.org/abs/1503.08895) 使用端到端训练的多次记忆读取；
[Recurrent Depth](https://arxiv.org/abs/2502.05171) 探索通过循环深度增加内部计算；
[Coconut](https://arxiv.org/abs/2412.06769) 把隐状态作为后续连续输入。
本文选择的是复用因果 evidence 的查询递推，尚未取得 Viby 收益。

相对于仓库已有循环 CED，本候选不做四 token 锚点保持广播：未来集成让每个当前
query 有自己的状态，避免不同问题共享逐段常数修正。**这会增加计算，未解决成本。**
不能把多次读取完整 toy memory 推广为固定 candidate pool 的 CSA2 效果；真实池若缺少
后继证据，改变 query 也救不了。若要刷新 candidate pool，必须另立改变读取预算的变体。

## 预注册筛查（在首次训练前固定）

脚本：[latent_query_probe.py](../experiments/latent_query_probe.py)。
仅复用现有 `ConceptBlock` 的 K=V 注意力、残差和 FFN，64 维、2 heads；
每个 fact 独立编码为 source/destination embedding 的拼接，再 RMSNorm。
这是明确提供关系结构的强归纳偏置，无自然语言 encoder，没有组压缩。
全部位置设为零以保持 fact 排列不变；这不是生产 RoPE 验证。

数据：16 个实体的独立随机单环，打乱 fact 次序，随机查询 q，答案 f(f(q))。
每世界一个答案，禁止中间节点监督、CoT、辅助 MSE、课程或冻结预训练主干。
训练/评估使用独立 RNG seed；保留完整原始数组。最后答案是唯一 CE 监督。

固定配置：seed=0，512 步 × batch64 = **32,768 个答案标签/条件**，
2,048 个 held-out 世界，Adam lr=0.002，FP32 eager CPU，三个条件从逐张量相等的
初始化开始，读取完全相同的训练样本顺序。初筛结果若不通过，停止本候选的集成；
不在同一 held-out 上调整超参再报确认通过。若通过，仅允许另做 seed1/2 复核，
全部通过仍只表示值得做真实 CED 小模型实验，不授权从此结论直接启动长预训练。

| 条件 | 读取 | 第二轮 Q | 参数 |
| --- | --- | --- | --- |
| recurrent | 2 | 更新后的状态 | 同一共享 block |
| fixed_query | 2 | 初始 query，残差/FFN 仍迭代 | 与 recurrent 相同 |
| one_read | 1 | 无 | 与 recurrent 相同，计算较少 |

fixed_query 控制新增 FFN 深度与两次逻辑读取；没有编译但相同 Q 仍可能存在
运行时复用差异，不能称为实测等 FLOPs / 等时间。one_read 是较少计算的参考。

只对 recurrent 做额外评估干预：reset 第一轮状态为初始 query；donor 将第一轮
增量 `z1-z0` 换成下个独立世界的增量并匹配范数，保留自己的初始 query 和全部 memory。
每 128 世界循环 donor，原始行号和算法足够复原配对。donor 是分布外干预，必须与
从头训练的 fixed_query 对照共同解释，不单凭损坏后变差宣称推理成立。

先检查 direct/VJP primal 差 <=1e-5、第一轮状态的最终 CE 梯度有限且 norm>1e-8、
随机方向有限差分误差 <=2e-4。每步检查所有梯度/loss 有限。
最终固定 gate（逐世界配对 bootstrap 2,000 次，95% CI）：

- recurrent held-out accuracy >=80%；
- 相对 fixed_query accuracy >=+10pp、NLL <=−0.05，NLL 差 CI 上界 <0；
- own 相对 donor NLL <=−0.05，差 CI 上界 <0。

one_read/reset 只做诊断，不用于更换主要比较。保存全部条件与失败记录。
梯度 gate 仅检查 recurrence 的学习路径；参数梯度的数值健康不保证它能学会组合。

## 后续真正要证明的量

1. 训练 token 效率：在相同 tokenizer、原始数据、有效标签口径下，达到共同独立
   validation NLL 所需 token 数，至少多个随机种子；不能用本 toy accuracy 代替。
2. 输出 token 效率：在同任务正确率下生成多少文字/推理 token，同时报告 latent 轮数。
3. 计算效率：同 FLOPs / 完整训练窗口 wall time 的质量前沿。两次内部运算不是免费。
4. CED 接入：每 token query、共享只读 causal KV、原 token RoPE、packed/PAD 隔离、
   mHC pre_mix、每次 MoE 统计及原路由损失必须明确；无 latent self-KV 的版本可只缓存
   evidence，但当前 probe 没有实现上述接口，不能当作 engine/DSpark 接通。

## 首次结果：停止集成（2026-09-14）

命令：

```bash
.venv/bin/python -u experiments/latent_query_probe.py --run-dir research_runs/latent_query_20260914/seed0
```

三个条件均完成 512 步，每条件 32,768 个答案，共 98,304 个训练答案；
不是 98,304 个自然语言 token。每条件另读 1,081,344 个符号 prompt ID。
每个条件有 44,864 个参数、同一份初始化、相同数据顺序。

| 条件 | held-out NLL | accuracy |
| --- | ---: | ---: |
| 两轮动态查询 | 2.765427 | 7.3730% |
| 两轮固定查询 | 2.762321 | 6.2988% |
| 单次读取 | 2.763666 | 5.5664% |
| 动态查询，换 donor 增量 | 2.783244 | 6.1035% |
| 动态查询，reset 中间状态 | 2.766742 | 7.2266% |

动态减固定 NLL = +0.003106，95% CI [−0.003215,+0.009738]；
own 减 donor = −0.017818，95% CI [−0.025879,−0.009738]，未达 −0.05 门槛。
无上下文均匀分类参考 1/16=6.25%；单环还排除了 query 本身作为答案，
所以仅排除该 ID 就有 1/15≈6.67%，这些结果不支持已学到两跳计算。

direct/VJP primal gap=0；中间状态梯度 norm=0.588139，随机方向有限差分误差
1.064e−5。3 项定向 CPU 检查通过。梯度存在但学习 gate 失败；不追加 seed1/2、
不扩大步数、不据此修改 Viby 默认。只限制该模型/数据/预算，不否定所有查询递推。

原始产物：`research_runs/latent_query_20260914/seed0/` 下的 `manifest.json`、
`train.npz`、`heldout.npz`、`per_world.npz`、各条件训练 JSONL/权重、初始化数组、
`results.json` 和源码快照；运行 stdout/stderr 另存 `run.log`。
记录中的 CPU 时间仅是本脚本执行观察，不用来推断 MLX GPU 或完整模型吞吐。
其中 `train_seconds` 字段实际还包含最终 own 条件评估，未作为训练速度比较。
`completion_audit.json` 复核三条件各 512 次更新、最终权重有限、13 个参数张量
确实更新，以及训练/验证之间完整图与 query 组合的交集为零。

## RLT 带来的下一候选：跨 token 保留计算修正

用户随后给出 [RLT 仓库](https://github.com/yifanzhang-pro/recurrent-looped-tranformer)。
本轮核实 HEAD `9c40dec78a6d7da73e1226310fd6c8edaa32577e`，读取 README 与论文
第 2 节，并渲染核对第 4 页。原始论文保存在 `research_runs/rlt_review_20260914/RLT.pdf`。
此前记忆中的“只有设计、没有实验”已不完整：当前 README 新增约 79K 参数、3 seeds
的独立合成状态追踪结果，32 operations 训练，128 operations 测试；参数/数据匹配，
FLOPs 未匹配。长序列五状态任务 20.7% 接近 20% 随机水平。
这支持小规模方向探索，不构成语言模型 token 效率证明。

RLT 的必要契约是 `H_t=(s_t,C_t^D)`：上一 token 最终隐状态与每层 SWA KV
一起延续，prompt/response 不重置，训练完整 BPTT。它不是同 token 的多轮读取。
论文具体 Merge 仍有投影与 sigmoid gate；“小非零 feedback 初值”也只写成候选。

从它得到的**待验证假设**不是加更多循环，而是改变所携带状态的语义。
在一个共同 d 维残差坐标系里，定义输入 e_t、持久修正 r_t、输出 z_t：

```text
r_0 = 0
z_t = D_theta(e_t + r_(t-1); M_<=t)
r_t = z_t - e_t
p(x_(t+1)) = readout(z_t)
```

若 `D_theta(u;M)=u+F_theta(u;M)`，实现可直接使用

```text
r_t = r_(t-1) + F_theta(e_t + r_(t-1); M_<=t)
z_t = e_t + r_t
```

避免数值上先加大向量再相减。**当 F=0，r_t=r_(t-1)**：不产生新计算时保留
旧状态，同时不会仅因 identity path 把每个输入不断累加进记忆。
相比 `z_t=D(e_t+z_(t-1))` 的整状态加性反馈，这是一个可检验的结构差异。
不声称比论文的 gated Merge 更强；也不主张 residual recurrence 本身首次出现。

r 不是下一概念预测、不是预测误差目标；它是任务 CE 自行学出的持久工作状态。
无需额外 MSE/VQ，所有位置在推理时更新。理论直觉是任务相关状态能跨 token
继续加工；是否有用仍取决于数据与优化，不能从残差恒等式推导有效推理。

边界与否证：

- 不能对不同归一化/投影坐标随意做 `decoder_hidden-encoder_hidden`。Viby 的
  mHC `pre_mix` 会随块变化，因此候选首先需要独立的共同坐标局部残差块；
  不能直接把上述 d 维式子当作当前整个 mHC decoder 的等价重写。
- `dr_t/dr_(t-1)=I+J_F` 提供 identity path，但不保证谱半径、状态有界或长期梯度。
  RMSNorm 分支输入也不约束被累加状态。若 norm/梯度随长度爆炸，判机制失败，
  不用 clipping 掩盖根因；非零任务梯度同样不证明泛化。
- `M` 若只含 clean encoder evidence且局部块无 latent self-attention，跨 token
  新状态仅需 r；若保留 decoder SWA，完整状态仍必须包含每层 KV，不能只复制 r。
- 下一项应是 residual feedback / RLT-style full feedback / token-only 的同参数
  小实验，使用**非交换**状态转换、匹配操作计数但改变顺序的样本及 no-op 干扰，
  再做状态 donor/reset 与长度外推；parity 单项可被交换统计解决，证据不够。
  它是新的候选，不是本次两跳探针的结果，也未在本轮运行。
- 真正逐 token 非线性更新仍要求 decoder 序列执行。只并行 encoder、使用有限轮
  Jacobi、detach 或 block 化分别改变吞吐或梯度/状态契约，不可冒充无代价的精确 RLT。

本轮结论：下一步的研究重点是**可持续更新、直接服务任务的状态**，但没有证据
证明该 residual feedback 候选优于 v1/v3 或比 RLT 原版更好。
