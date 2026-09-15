# 让读取结果成为下一次查询：机制筛查（2026-09-14）

## 研究问题与本轮边界

用户要求寻找具有有用中间计算的 latent thinking。旧两阶段流水线的显式状态链
只从前一 token 的 stage1 到当前 stage2，stage2 不再进入思考模块；先前同 token
ConceptBlock 两跳探针也未学会任务。本轮不增加另一个未经检验的默认架构。

发现用户的 `research_runs/viby41_thinking` GPU训练正在运行（PID88776，命令含
`--thinking --batch_size 12 --accumulation_steps 2 --max_seq_len 1024`）。本轮先做
独立CPU/FP32实验，保留该任务与其加载的生产模型实现。MuonH默认关闭继续保持。

## 一个具体的改动

令连续状态表示下一次读取所需的查询，反复执行：

```text
q_0 = query representation
p_r = softmax(8 * normalize(Wq q_r) @ normalize(K).T)
q_(r+1) = p_r @ V
```

K/V分开表示“以什么条件取出”与“取出了什么”。输出的value直接成为下一轮查询，
不在每轮叠加旧query，不通过另一个FFN把新地址混回旧地址。这是覆盖式状态更新，
而非保持所有旧信息的累加。固定证据与固定轮数下，所有token的查询可并行，
只需沿计算轮次顺序运行。训练和推理执行同一个计算图，不用Jacobi或TBPTT。

这是多跳注意力的研究实例，非基础算法首创：
[Memory Networks](https://arxiv.org/abs/1503.08895) 研究多次记忆读取，
[Universal Transformers](https://arxiv.org/abs/1807.03819) 研究token并行的共享参数深度递推。
计算轮次带来真实FLOPs与解码串行代价；它不保留PLT的跨轮并行解码优势。

## 可学习接口，避免把答案写进初始化

16个实体，每世界随机单环、随机facts顺序、随机query。每个fact的输入是两个实体
embedding拼接；独立且随机初始化的Linear学K/V，随机Linear学Q，最终分类与实体
embedding共享坐标。没有oracle初始化、直接查表、hard argmax中间状态或路径损失。
显式facts边界与共享实体词表仍是很强的接口先验，不代表语言encoder已学会提取关系。

训练只监督问题的最终答案，先128 updates的一跳问题，再512 updates的两跳问题，
batch64、dim64、Adam lr0.002、FP32 CPU；共40,960答案，其中32,768为两跳。
一跳阶段是显式课程，训练样本成本完整计入，不能宣称“仅靠两跳监督自发发现算法”。
warm与composition阶段使用不同世界。正常递推和fixed-query对照使用相同初始化、
数据、课程、参数和逻辑读取次数；fixed-query始终用初始query，但仍执行各次读取。
重复读取可能被运行时复用，因此不把逻辑读取次数当作实测等FLOPs。

seed0做初筛；如果通过，再用独立seed1/2确认。每个seed最终评估2,048新世界，
只读一次固定的最终checkpoint，不在确认集上调整超参。另测1/4/8跳，4/8不参与
主训练，预算按实际执行跳数计，不作为“免费泛化”。

## 强因果检验

记录每轮attention选中的fact，比较真实中间节点的对应key；这些路径标签仅用于
最终诊断，绝不进入梯度。

把世界i第一轮的连续状态换成世界i+1第一轮状态，保留世界i的全部K/V。
若状态编码的是donor世界的中间实体m，而非现成答案，下一步应输出 **f_i(m)**，
不是donor的最终答案。这项可预测的反事实正确率比“swap使原答案变差”更有区分力。
保存所有原始世界、donor行号、路径、原答案及反事实答案。

首次训练前固定gate：初始化两跳accuracy<25%（排除接口已经直接解题）；
最终两跳accuracy>=90%，相对训练fixed-query>=20pp且配对NLL差95%CI上界<0；
每一步寻址accuracy>=90%；跨世界patch的反事实答案accuracy>=80%；
全部训练loss/梯度有限，direct/VJP primal差<=1e-5，有限差分<=2e-4。
初筛失败则停止该参数化，不启动Viby预训练。通过三个seed也只建立结构化关系任务的
机制证据，后续仍需在真实encoder/decoder接口、因果缓存和同预算语言模型上验证。

数值准备记录：第一次在训练前停止，FD eps=0.01产生2.757e−4误差。固定相同
参数的eps收敛检查中，0.003/0.001的误差均约4.94e−5；后续要求这两个步长
均通过原2e−4门槛，不放宽容差。第一次失败保留在seed0目录，实际训练另用seed0_checked。

seed1的eps=0.003检查在训练前仍有2.208e−4截断误差，eps=0.001为1.764e−5。
为避免FP32差分和步长选择影响门槛，最终审计改用同一权重、同一中间状态的
NumPy FP64算术参考，固定eps=1e−4/1e−5，仍要求两者误差<=2e−4。网络、训练
预算、任务、学习率和所有效果门槛均未改变；seed1初次未执行任何更新。

## 首轮三种子结果与第二个参数化（运行前登记）

独立seed0/1/2中，0/2的两跳、4/8跳及反事实均100%；seed1两跳仅6.74%，
一跳寻址0%，未通过。没有挑选成功种子作为普遍结论；全学习K/Q参数化停止集成。
这个失败表明“一跳接口首先要可学稳”仍是关键，不证明所有问题都由K/Q坐标导致。

第二个参数化 `--addressing tied`：keys=normalize(A*E(source))，queries使用同一A；
value仍由随机Linear从完整fact拼接学习，最终分类与E共享。这样初始化就偏好
匹配同一实体，**匹配先验由设计提供**；value到可再次查询状态的转换仍需学习，
初始化二跳正确率必须<25%。本轮不声称从无结构文本中自行发现实体寻址。

固定与第一轮相同的dim、步数、lr、课程和gate，使用全新seed3/4/5以及对应的新
训练/测试世界。减少一个独立key投影，参数量降低；正常与fixed-query仍完全同参数。
三个新seed必须全部通过才推进下一层接口验证；不把最差seed延长训练后覆盖旧结果。

## tied版本确认与CED接口验证计划（在语言pilot之前）

全新seed3/4/5均通过：2/4/8跳、每跳寻址、反事实正确率均100%；训练fixed-query
两跳accuracy为6.54%/7.13%/7.23%。初始化最终答案仍接近随机，说明value到可读取状态
的转换确实经过学习；初始化第一跳匹配先验来自共享Q/K，不计作学出的能力。

已提炼 [IterativeEvidenceReader](../model/iterative_thinking.py)，并用任意权重对拍
证明它与tied探针的运算一致。CED适配版 `ced_iterative_v2`：

```text
u_t = Input(RMS(e_t))
K_t = normalize(Address(u_t))
V_t = Value(RMS(e_t))
z_t^0 = u_t
z_t^(r+1) = softmax(8 * normalize(Address(z_t^r)) @ K_<=t.T) @ V_<=t
signal_t = scale * RMS_amplitude(e_t) * RMS(Output(z_t^R))
```

每一步完整覆盖状态，后一步使用更新后的查询。状态是同token内的工作状态，
不声称跨token无限持久递归；计算轮数由配置给定，没有学停止策略。
在关系任务里，4/8跳测试给出了对应计算步数；不证明对同一两跳问题多算到8步更好。

K/V来自额外的encoder投影，两者独立；Q/K共享Address。没有新的FFN、预测头、
MSE或码本。迭代中的状态、寻址和softmax为FP32，额外KV缓存也为FP32（每token
2*w元素）；encoder和decoder工作dtype照旧。原CED全局KV、RoPE及mHC保持原契约。
新支路本身不再添加RoPE，以维持状态/键的共同寻址坐标；输入encoder本身含位置信息，
因果、文档、PAD仍显式屏蔽。这是新架构语义，不是旧支路的等价kernel优化。

训练在每个round沿token并行，round串行；解码也要执行R次小模块读取，无法宣称
PLT式几乎零延迟。新增矩阵参数 `3*d*w+w*w`，逻辑前反向矩阵工作按
`6*(3*d*w+(R+1)*w*w)`/token计，另加dense QK/AV工作。首次Q与新K可能被编译器
复用，因此逻辑计数不等于实际执行指令数。

纯CED与MuonH关闭的默认不变。显式`--thinking`的新模型选择v2；历史sidecar保留v1。
`--thinking_arch ced_pipeline_v1`可选旧模式；v2用`--thinking_steps`指定预算，改变
版本或步数要fresh cache，恢复训练需显式reset_optimizer。同维度迁移保留input/output
投影及共同主干，新reader/value随机初始化，丢弃旧block；不声称是从头训练对照。

语言pilot：使用新的source_offset=4000、seed1338，FP32 GPU reference，256 updates，
其余维度/预算/optimizer/学习率沿用上一轮pilot；正常、fixed-query(reset)、纯CED
三条件同主干初值与数据。沿用−0.01/−0.005 NLL与CI上界<0的门槛，不从结果调参。
记录与旧pilot所选token前缀的交集，不把同语料源当独立语料。新的关系测试成功不能
代替该语言接口gate。检查点旧路径目前不存在，本轮不复做旧dim1024权重的数值检查。

## CED首轮结果与共享写回的接口修订（第二轮运行前）

独立写回版v2的FP32语言pilot：正常6.664124，fixed-query6.670580，纯CED6.659585。
正常相对fixed-query为−0.006456，95%CI[−0.012441,−0.000758]；相对纯CED为+0.004539，
CI[−0.001060,+0.009653]。迭代有作用，但总体收益gate失败，不宣称语言token效率提升。

与成功原型相比，适配器还多了一个独立Output投影；原型则用同一实体embedding做
状态输入和最终分类。新增显式版本 `ced_iterative_tied_v1`，将写回改为Input矩阵的
转置：若u=P*e，则decoder接收P^T*z。这是同一坐标的伴随映射，**不是P的逆**，
也不保证重构原encoder表示；它去掉了一个独立坐标变换及d*w参数。原v2保留加载。

新模型显式`--thinking`默认选择共享写回版；全局thinking开关仍默认关闭。
相同维度下新模块参数为`2*d*w+w*w`；P在输入/输出各使用一次，逻辑FLOPs并未因
少存一份矩阵而消失。Q/K共享Address、value覆盖state的闭环不变。

第二轮预算固定为2048 updates×batch4×length128=1,048,576 labels/条件，
四条件共4,194,304 labels，仍是3.3M小模型；source_offset=12000，seed1340，FP32 GPU，
其余lr/optimizer/warmup沿用前次，不在同一held-out上选超参。只评估最终checkpoint。
四条件：共享写回循环、共享写回fixed-query、纯CED、独立写回循环。共同参数逐张量
同初值，独立写回条件仅额外初始化其独有Output矩阵；所有条件同数据、同更新数。
原−0.01/−0.005 NLL与CI gate仍保留，独立写回对照另报配对差以检验接口解释。
扩大的是预注册的小模型接口检验，不是启动原dim1024预训练。若失败，停止扩大此轮。

## 最终结果与解释边界

共享写回版的小模型实验（FP32，1,048,576 NTP labels/条件）：

| 条件 | 验证NLL | 相对正常循环的区别 |
| --- | ---: | --- |
| 共享写回循环 | 4.962601 | 当前候选 |
| 共享写回fixed-query | 4.963722 | 不用上一步读取结果生成下一查询 |
| 纯CED | 4.982090 | 无新增模块 |
| 独立写回循环 | 4.981054 | 额外独立初始化/训练Output矩阵 |

正常−CED = −0.019489，95%CI[−0.032290,−0.007427]；正常−独立写回 = −0.018453，
CI[−0.030608,−0.005844]。但正常−fixed-query = −0.001121，CI[−0.013060,+0.010255]，
未达到预先规定的循环贡献门槛。正常模型评估时reset使NLL增加0.016108，swap增加
0.004010：模型依赖了它，不代表从头训练的无循环模型达不到相同性能。

所有条件梯度有限；正常grad norm中位数1.274958、最大3.284926。总gate仍为
`passed=false, stop_no_large_pretraining`，停止扩大本轮。共享写回的整体改动有
单种子语言收益信号，但初始化、参数约束与优化耦合均随之改变，不把它解释成
已经证实的某个单独几何机制；也不把总体NLL收益归因于thinking循环。

固定query控制的前几轮readout不进入最终loss，惰性执行可以将重复计算裁掉。
它是同参数的机制消融，不是严格等FLOPs的性能基线；本轮没有计算效率结论。
elapsed_seconds含编译，运行顺序固定，没有A/A或ABBA，不能据此报告加速。

关系任务方面，三个全新seed共6,144个测试世界均通过；4/8跳复用相同世界但改变
目标，不算成新的独立世界。反事实审计另外排除原答案与donor原答案偶合的样本，
剩余1774/1810/1776世界仍全部输出预期反事实答案，且第二轮确实读取donor中间实体。
这支持在该结构化任务中执行了可复用的中间关系运算，而非通用语言推理已得到验证。
原型提供了typed source字段、共享实体词表和一跳课程；CED适配器必须自己从上下文
学出地址/值，不能省略这一层难度。P^T是伴随映射，不等于P的逆或无损解码。

## 当前接口与验证

- 默认仍为纯CED、MuonH关闭。新模型显式`--thinking`选择`ced_iterative_tied_v1`，
  `--thinking_steps 2`指定内部计算预算；没有额外文字token、NCP或VQ损失。
- `ced_pipeline_v1`和`ced_iterative_v2`保留历史加载；版本/steps进入execution和cache
  signature。旧pipeline固定两阶段，不能用steps参数假装改变其深度。
- 输入/共享寻址/值投影/伴随写回均在CED边界执行，完整token分辨率、原CED证据KV、
  原token位置、文档/PAD隔离与mHC提升保持。新增支路没有单独RoPE，见前述版本说明。
- `tests/test_iterative_thinking.py`验证闭环组合、masked/PAD梯度、凸组合幅度边界、
  整段/分块/增量一致性、Metal FP32/BF16编译梯度、缓存/DSpark恢复与迁移。
  本轮定向99项通过，日志为`integration/final_all_modes.log`。这不是全仓或原规模验收。
- 大模型审计工具新增显式`--thinking-arch`/`--thinking-steps`；其默认保留旧pipeline
  以复现历史命令。当前旧权重路径不存在，因此本轮未复做原dim1024 checkpoint检查。
- 本轮最初观察到的用户GPU进程后来已不在进程表中；没有向它发送停止信号，生产
  模型代码修改在确认该进程不再运行后进行。没有启动新的原规模训练。

产物根目录：`research_runs/transition_thinking_20260914/`。
`seed0_checked/seed1_checked/seed2`保留全学习寻址的成功与失败；
`tied_seed3/4/5`保存共享寻址确认，`mechanism_audit.json`含反事实与组件等价核对，
`tied_confirmation.json`汇总。`language_seed1338`与`language_tied_seed1340`分别保存
两次语言pilot的原文、行号、token数组、参数初值、训练日志、每文档loss和最终权重。
这些结果已建立一个有限任务的latent计算机制实例；尚未建立通用LM的循环收益、
训练token效率倍率或实际解码加速。
