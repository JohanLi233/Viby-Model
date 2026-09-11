# Viby-PSR 实现与运行契约

2026-09-11。当前交付是可训练、可验证的机制实现；尚未证明准确率或端到端效率提升。

## 数据流

```text
token embedding → causal encoder → CED 边界原始 mHC attention input
                                        │
                      原 Compressor / Indexer 生成 KV、index K
                                        │ 只读前缀视图
                       S0 → [选址、读取或纯计算 → 两个 dense block] × R
                                        │
                              独立 ThinkingState
                                        │
                   Decoder 每层输出后的门控 workspace attention → logits
```

边界输入仍是 `apply_hc_pre_norm(h, pre_mix, layer.attn_norm)`。边界层先完成原注意力和
FFN，生成原始 CED KV，随后执行 PSR，并在该层输出后首次注入工作区。后续 Decoder 层
继续使用原 CSA2 状态；PSR 不写 `SharedAttnState` 的选择、候选池或 token cache。

`model/psr.py` 定义独立的 `EvidenceMemory`、`ThinkingState`、循环更新器、Decoder
桥接、辅助损失和有限预算 Bellman 标签生成。记忆在一段思考中不变化，但训练梯度可以
回传到产生它的编码器；“只读”不表示冻结编码器参数。

模型库配置默认 `psr_enabled=False`，普通预训练 CLI 默认开启（见下节）。启用后的默认形状为 8 个槽、宽度 256、2 个跨轮共享
dense block、每槽最多 16 个精确选中位置，默认 4 轮、最大 8 轮。更新包含槽间自注意力、
SwiGLU，以及有界 `tanh` 残差。没有共享大 MoE 的逐轮重跑，也没有思考 token。

索引使用与 CSA2 相同的多头 ReLU 加权公式、CED 的旋转后 index K，以及独立可训练的
PSR query/weight 投影。内部所有轮次的 RoPE 查询位置均为问题末位置，轮次不递增位置。
原始分数稳定排序，ties 按较小地址；不用会改变近邻排序的 epsilon。对选中 KV 每轮使用
当前查询重算注意力，不复用旧注意力输出。

## 前向与生成

```python
import mlx.core as mx
from model.config import VibyConfig
from model.model import VibyForCausalLM

model = VibyForCausalLM(VibyConfig(psr_enabled=True))
# 加载并训练新模块后使用；随机初始化不具备已验证的推理能力。
out = model(
    input_ids,                         # [B,T]，可以含 teacher-forced 答案
    labels=next_token_labels,
    loss_mask=answer_loss_mask,
    thinking_prefix_lengths=prefix_lengths,  # int 或 int32[B]，含问题末 token
    thinking_options={"rounds": 4},
    thinking_targets=targets,
    use_mtp=False,
    return_thinking=True,
)
loss = out.loss
slots = out.thinking_state.slots
trace = out.thinking_trace

model.eval()
logits, cache = model.prefill(prompt_ids, thinking_options={"rounds": 4})
next_logits, cache = model.decode_step(next_token_ids, cache)
# 或 model.generate(prompt_ids, thinking_options={"rounds": 4})
```

低层 `model(...)` 不推断问题边界。高层 `prefill` / `generate` 把整个输入视为问题，
启用 PSR 时自动执行一次思考；decode 只读取已经完成的工作区。

`prefix_lengths=L` 表示允许读取位置 `0..L-1`，首次注入位置是 `L-1`，该位置的 logit
预测第一个答案 token。检索还同时受 `segment_ids` 和 `attention_mask` 限制，桥接也
只能影响锚点所属文档的合法位置。一行只定义一个问题/回答工作区；packed 行里的其他
文档不会各自启动工作区。数组边界越界或锚点为 padding 时该行关闭桥接、清空可见集，
以便编译路径无需 GPU→CPU 同步；标量边界越界直接报错。

普通无 PSR 调用保持既有参数与执行路径。已有模型升级时，可用原 config 加
`psr_enabled=True` 构造新模型，再用 `load_weights(base_weights, strict=False)` 载入
原权重，显式核对未匹配键仅为新增模块。默认桥接 gate=0；先训练地址/行为目标，再打开
桥接联合训练。零门时答案 CE 不会给工作区有效梯度。`use_thinking=False` 可以在新
prefill/无缓存前向中绕过 PSR；已经含工作区影响的 cache 不能中途关闭 PSR。

`ThinkingState` 在 cache 中独立保存，不改变 `start_pos`、Engram 历史或压缩器填充数。
cache 仅保留 terminal slots/anchor/segment/budget，不保留完整训练 trace。回退越过问题
边界会报错，要求重新 prefill。显式传 `thinking_state=` 可做无缓存状态干预，但调用方
负责状态与问题的对应关系；不要把另一个请求的工作区传入已有 cache。

首版支持整段问题 prefill、固定预算 batch、普通自回归 decode。**尚不支持 chunked
问题 prefill、连续 batch 状态迁移、前缀池工作区复用和 PSR+DSpark 联合路径**。
`VibyEngine` 对 PSR 配置明确报错，避免静默丢失工作区；PSR 训练使用 `use_mtp=False`。
原来的非 PSR engine / DSpark 路径保留。

## 监督契约

`thinking_targets` 仅在循环结束后的损失函数消费，不是 Reasoner 的输入。

| 键 | 形状 | 含义 |
|---|---|---|
| `address` | `[B,R,m]` | 每轮读取前的正确物理证据位置；`-1` 忽略 |
| `tests` | `[B,R+1,P]` | 行为测试 U 的 ID，包括 S0 和终态 |
| `results` | `[B,R+1,P]` | 行为测试的离散结果 Z；`-1` 忽略 |
| `values` | `[B,R+1,3]` | STOP/COMPUTE/READ 的后续损失和未来成本；NaN 标记未测动作 |
| `terminal_tests` / `terminal_results` | `[B,P]` | 仅终态的行为测试与标签，与全状态 tests/results 二选一 |

行为预测头对 U 查询工作槽后输出分类分布；优化交叉熵，不做教师 hidden-state MSE。
地址 CE 直接训练全局选择分数，因此不依赖 hard Top-K 的离散地址梯度。
纯计算轮没有地址 CE。不同证据对应不同槽时可给每槽不同目标。

总损失为原答案 CE/原辅助项，加权地址 CE、行为 CE、价值 MSE 与执行成本。
成本单位由 `psr_read_cost`、`psr_compute_cost` 指定；READ 成本包含完整索引、读取和更新，
COMPUTE 包含更新。它们默认是**抽象单位**，不是实测毫秒。Python 轮数被 stop-gradient；
成本常数本身不训练离散决策，价值监督承担这个职责。

`counterfactual_value_targets` 从模型自己的 `trace.states` 出发，以纯前向 `advance`
枚举 READ/COMPUTE 动作组以及 STOP，执行有限预算 Bellman backup。终端损失由调用方
提供，可以是只读 S 的正确答案 CE。value target 包含后续最优损失与未来成本，排除
当前动作组成本；调度器加上当前组成本再比较。标签可以利用正确答案，状态转移不接收
正确答案。这是每样本的实测终端损失 backup，跨样本拟合后才近似期望价值。

枚举随组数指数增长，函数限制为最多 8 轮，仅适合起步可验证任务。价值拟合不自动授予
校准标志，也不保证未见任务上的停机策略可靠。

## 计算控制与诊断

固定预算默认每轮 READ。`thinking_options={"actions": (READ, COMPUTE, ...)}` 可指定
不依赖教师未来的固定动作日程；COMPUTE 实际不调用 Indexer 或 memory read。
固定预算完整前向/反向可 `mx.compile`，使用几个固定 R 分别缓存编译函数。
这与 [MLX 的编译及状态捕获契约](https://ml-explore.github.io/mlx/build/html/usage/compile.html) 一致。

自适应执行需 `eval()`、batch=1，并显式声明已有独立校准证据：

```python
options = {"rounds": 8, "mode": "adaptive", "policy_calibrated": True}
```

每 `psr_group_size=2` 轮在组边界同步一次价值决策。当前状态和剩余预算选择 READ、
COMPUTE 或 STOP；同一组复用动作类型，READ 组内仍每轮自适应重选地址。STOP 真正退出
循环，COMPUTE 真正跳过扫描。组控制器不能包进固定图编译；它不使用答案熵、状态距离或
连续预测相同作为停机规则。

`selection_mode="fixed"` 是固定证据消融：首次集合保持不变，但重新计算当前 attention。
目前为了保留相同地址监督路径，它仍算全量 scores 和精确排序，不能把这个消融叫作
优化后的固定读取时延基线。`--no-predictive` 用于行为目标消融。

`trace` 保留 S0..SR、分数、地址、动作价值、实际扫描数、执行成本及证书诊断。
证书用最近一次**实际全量索引**的 query/weights 和 Top-K 边界间隔，计算多头 ReLU
变化上界。ties、无有限边界或非有限上界均不通过。证书仅在同一不可变记忆/可见集内
评估；调用间不复用。**没有 FP32/Metal 舍入误差包络证明，因此永远不据此跳过扫描。**

完整扫描和排序仍分别依赖 N 和 `N log N`；teacher-forced 输入会扫描被 mask 的后缀，
不能把每轮成本写成 O(k)。Decoder 桥接、原始 prefix 编码、训练和编译成本也必须计入。

## 可运行任务与验证

### 普通文本预训练：默认开启

原来的预训练命令无需新增 PSR 开关：

```bash
uv run trainer/train_pretrain.py \
  --data_path /Volumes/pan/text/pretrain_t2t_dedup.jsonl \
  --out_dir research_runs/viby41 \
  --pack_sequences --doc_mask \
  --batch_size 16 --accumulation_steps 2 --max_seq_len 1024 \
  --log_interval 1 --seed 1337 --use_swanlab --auto_resume --no_save
```

`--no-psr` 可关闭；`--psr_rounds`、`--psr_dim`、`--psr_slots`、`--psr_topk` 等可覆盖形状。
`--preset tiny` 也保留预训练 PSR 开启状态。默认 4 轮、8×256、每槽 Top-K=16，桥接 gate
初值 0.05，使第一步答案损失就能回传进工作区；行为预测与索引蒸馏权重各 0.1。

`trainer/psr_pretrain.py` 在每行最长有效文档中点建立一个问题前缀，读取与桥接受原文档
掩码约束。边界只依赖文档/padding 布局，标签值不参与选边界。保留全部原始 LM loss，
工作区只作用于选中文档的后半段；不是每 token 或每个 packed 文档各运行一个工作区。

纯文本没有证据 oracle：每轮用**当前 read query 的完整前缀读取分布**作为 stop-gradient
目标，蒸馏到 Indexer 分数，使 hard Top-K 的选址参数获得梯度。它是启动取证学习的代理
目标，不是“下一条必要证据”的真值。同时仅在终态预测前缀后最多 4 个 token，U 对应预测
偏移，跨文档/PAD 目标忽略；不对中间状态施加“每轮更接近答案”的目标。终态专用接口
只执行一次预测头，避免为忽略的中间轮次物化词表 logits。

预训练不启用自动停机或复用跳过；value head 的校准仍需可验证任务。FLOPs 日志按一次
前缀 PSR 的实际固定轮数摊销，包含索引、完整读取教师和桥接的名义计算量；不是实测性能。
SFT/DPO 没有自动套用这个文本前缀策略。

续训时：完全没有 PSR 权重的 checkpoint 只允许新增 PSR 参数缺失，其他主干参数仍严格
检查；升级后初始化新工作区，重置 optimizer/epoch/step。完整 PSR checkpoint 正常恢复；
部分 PSR 参数丢失会报错。已有 `--no_save` 语义保持不变：它也关闭 `auto_resume` 和
SwanLab。因此上面命令实际不会恢复 checkpoint 或上传 SwanLab；需要这些功能时去掉
`--no_save`。

验收覆盖原参数解析、packed 前缀与未来标签隔离、实际 `BaseTrainer` 编译后非零梯度与
更新、checkpoint 升级/损坏/正常恢复、PSR FLOPs 摊销。另以本地小数据通过原
`trainer/train_pretrain.py` 运行了一个两微批的 Muon 梯度累积窗口；日志保留在
`research_runs/psr_pretrain_integration_smoke/console.txt`，没有启动用户的大语料训练。

### 可验证的指针任务

```bash
.venv/bin/python -m trainer.train_psr \
  --out-dir research_runs/psr_first \
  --steps 200 --warmup-steps 50 --budgets 2 4 8 \
  --max-seconds 600 --compile

# 可选：在自身状态上收集有限预算反事实目标并拟合 value head
# 加 --value-updates 8；这不会把 policy_calibrated 改成 True。

.venv/bin/python -m pytest tests/test_psr.py tests/test_v41_config.py \
  tests/test_v41_consistency.py -q
```

runner 使用小主干、8 槽、宽度 64、Top-K=4 作起步任务配置，原模型库默认仍是
8×256 / Top-K=16。随机指针表、起点和深度是全部输入；执行轨迹仅用于监督。
先地址/行为预热，再打开门进行答案联合训练。`--checkpoint DIR` 可继续一个 PSR
checkpoint 的模型权重训练；它是新的实验，不恢复旧 optimizer/schedule。

每次新建输出目录，记录配置、数据身份、git 状态、训练日志、权重、峰值内存、预算×
深度曲线、固定集合消融、只读工作区的受限答案准确率，以及完整同步 prefill 耗时。
U=1 的未训练测试标签保留用于诊断；它含未训练 test embedding，低分不能单独证明状态
没有保存所需信息，高分也不能证明普遍充分性。

实现验收已跑过 FP32/BF16 梯度、固定预算编译前向/反向、零门/关闭一致性、前缀与文档
隔离、padding/无可见证据、单 token、零预算、缓存 decode 一致性、纯计算、真实早停，
以及选择“第一步即时收益为零、第二步有收益”的 Bellman 算例。

`research_runs/psr_implementation_smoke/` 的 3 次更新（1 次预热、2 次联合训练）和
1 次价值更新已完成；仅证明训练入口和产物链可执行，不构成已训练模型效果证据。
`research_runs/psr_multibudget_smoke/` 另跑通 R=2/4/8 的编译训练与预算评估。
PSR、原配置、原因果一致性、原训练和 DSpark engine 相关回归合计 74 项通过；
最终预算元数据修正后再次通过全部 15 项 PSR 专项检查，修改文件 Ruff 与 diff 检查通过。
尚未执行等参数普通计算、等读取量一次性取证、显式 CoT 基线及多种子组合外推研究。
本次不宣称学习到充分预测状态、自动停机已校准或端到端加速。
