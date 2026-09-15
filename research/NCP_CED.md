# 历史 CED-aware NCP（2026-09-14）

> 2026-09-14 后续决定：用户终止 v1/v3；本页保留替换前的机制和默认值记录。
> 当前默认已回到纯 CED，替代实验见 [CED 边界思考态](CED_THINKING.md)。

新建 `VibyConfig()`、tiny 配置与预训练入口默认接入 `ced_state_v3`。
没有 A/B/C/D 架构选择或“预测进入全局 KV”的变体。完整 token encoder、decoder、
原 mHC、MoE 路由和损失继续保留。显式选择旧循环 CED 仍使用其独立执行路径，
不与 NCP 叠加。v1 `ced_shared_kv_v1` 与 v2 `ced_state_v2` 仍可显式选择。

这是用户提出的 CED-aware 结构实现，不是 NCP-ArchPreview 的严格复现，也没有
训练质量或 1.5× token efficiency 结论。论文接口背景见
[NCP-ArchPreview](https://arxiv.org/pdf/2609.10715)，CED 背景见
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。

2026-09-14 结构修订：原 `ced_shared_kv_v1` 共享原始概念 KV、仅边界 PQ 反馈的
实现与机制证据保留在 commit `4f43927` 及 `research_runs/ncp_ced_20260914/`。
那是缓存成本优先的研究选择，不是已经证明质量等价的优化，也不是论文未完整
复现就构成实现缺陷。v2 增加分层记忆与状态通路，保留 v1 的显式 config/sidecar 加载。

2026-09-14 结构修订（v3）：针对 v1/v2 运行暴露的概念层问题（诊断与根因见
[NCP 概念层问题](NCP_CONCEPT_LAYER_PROBLEM_20260914.md)）：PQ softmax 混合把
预测限制在码字凸包内，初始不能表达 copy；预测 MSE 长期输给持久基线。v3 把
预测头改为 copy-residual（预测 = 当前概念 + 零初始化残差 MLP，残差目标为完整
detach 的相邻概念差分），池化从固定均值改为逐通道四槽凸组合（全零 logits
严格等于均值）。PQ codebook 保留但只由 VQ 目标训练，作为量化/塌缩健康探针，
不再参与预测。按用户范围，本轮不做交叉验证、不启动长训练、不改变辅助项权重。

## 数据流与固定契约

1. 在第一个 decoder 层（默认索引 6）入口读取 `hc_pre(h, pre_mix)`。
   在注入前使用原 `apply_hc_pre_norm` 路径保存归一化的 `global_kv_input`。
2. 文档内部每完成四个 token，将该组原 encoder 表示按 FP32 池化后转回工作
   dtype，再 LayerNorm 得到已观察概念。v1/v2 的池化是固定均值；v3 是逐通道
   四槽凸组合 `softmax(pool_slots, axis=slot)`，`pool_slots` 下标 r 从组末
   倒数（r=0 为组末 token），全零时严格退化为均值，训练与增量路径共用同一
   槽序。PAD 间断、相邻文档切换均重置分组；即使
   文档 ID 在后面重复，也不连通先前的概念历史。
3. `memory_project → RMSNorm → RoPE` 为已观察概念生成一份宽度 128 的共享
   K=V。第一层处理后，以独立投影和 RMSNorm 将其输出生成第二份紧凑 K=V，
   第二层读取第一层加工后的同文档历史。默认两层仅更新一次；显式更多概念层
   则每个层间边界各更新一次。RoPE 使用组末原 token 位置；每层包含 pre-norm attention 与
   宽度 `2*dim` 的 SwiGLU。概念 query 按 head 做 RMSNorm，注意力读出在投影前
   按 query 位置反向旋转，保持相对位置语义。
4. v1/v2 的 PQ 预测头输出每个 segment 的 softmax 权重，与可训练 codebook
   加权生成连续预测。v3 改为 copy-residual：预测 = 当前概念 +
   `RMSNorm → Linear → SiLU → Linear` 残差，输出投影零初始化，初始预测严格
   等于 copy，且不受码字凸包约束；codebook 不再进入预测路径。预测经过
   RMSNorm、线性 feedback，两个接入层各有零初始化门控。
   各概念层输出另经独立 RMSNorm 和线性投影，在概念频率计算。因果保持到 token
   位置后，由目标 decoder 的 `hc_pre(h, pre_mix)` 经 RMSNorm、层专属线性路由和
   FP32 softmax 选择概念深度，再经该层独立零初始化状态门控。两条通路相加后提升。
   接入层按 encoder 边界及 decoder 深度的 2/3 处确定、去重；默认 12 层是索引 6、10。
   预测/feedback 投影在接入层间共享，状态投影在接入层间共享，路由和门控各层独立。
5. 组结束位置立即可用该预测：k=4 时，首个预测作用于位置 3 的 logit，
   然后保持到位置 6。它可帮助预测 x4，但 x4 不参与位置 3 的概念输入。
   不完整尾组不构建新概念；最后完整组即使没有下一概念监督，仍提供反馈。
6. 按 `signal / sum(pre_mix)` 等量提升到各 mHC stream，不重算 `pre_mix`。
   `Block → Attention → _compress_dense/decode` 仅让 compressor 消费保存的
   encoder 输入。主 query、indexer query/权重与 SWA KV 消费注入后的 decoder
   状态；index K 来自干净的 compressor latent。chunked、dense 和 decode
   都使用这个分工，既有选择集合和 tie 规则没有改动。

因此，固定输入/权重的概念输出干预不能直接改写 CED 全局 KV。共享 KV **不
stop-gradient**；NTP 仍能经该存储接口训练 encoder。概念能够影响 decoder
残差、注意力 query、局部 SWA 历史与 FFN/MoE。

## 默认参数和训练目标

| 字段 / CLI 参数 | 默认值 |
| --- | --- |
| `ncp_stride` | 4 |
| `ncp_layers` | 2 |
| `ncp_memory_dim` | 128 |
| `ncp_heads` | 4 |
| `ncp_groups` | 8 |
| `ncp_codes` | 256 / PQ segment |
| `ncp_loss_weight` | 1.0 |
| `ncp_vq_weight` | 1.0 |

这些参数使用 `--ncp_stride` 等字段名传入，并通过原 CLI → config → sidecar
链保存。无需新增开关即可开始新架构预训练：

```bash
.venv/bin/python trainer/train_pretrain.py --data_path /path/to/data.jsonl \
  --out_dir /path/to/fresh-output --pack_sequences --doc_mask
```

原 `L_NTP + L_MoE (+ 原有 z/MTP 项)` 加上 `λ L_NCP + β L_VQ`。
v1/v2 的 NCP 是下一完整同文档概念的连续向量 MSE。v3 的 NCP 是残差对完整
detach 差分目标的 MSE：`mean ||r_t − sg(c_{t+1} − c_t)||²`；copy 基座与目标
差分都不产生梯度，该辅助项不会把当前概念拉向下一概念。先对 hidden 维求 mean，
再对有效相邻组求 mean。因此 v3 的 `ncp/loss` 初始值等于 `previous_mse`，
`relative_mse_previous` 初始为 1.0，之后的下降表示残差学到了超出 copy 的变化。
VQ 是已观察概念对各 PQ segment 最近码字的平方距离，同样
使用 per-dimension mean。权重 1.0 是明确记录的起始值，**未经实际更新尺度
校准**，不能当作论文 reduction 或已找到的最佳配方。

- NTP：更新 encoder、decoder、概念历史计算、feedback/状态投影与门控；v3 中
  codebook 不再从 NTP 接收梯度。
- NCP：v1/v2 更新 PQ 预测路径和历史输入，未来概念目标完整 detach；v3 只更新
  残差头，copy 基座与差分目标均 detach。
- VQ：概念和最近码字选择 detach，只拟合 codebook。
- 所有预测和状态门控均为 0 时概念的 NTP 梯度被阻断，门控自身通常有梯度；辅助目标仍训练概念
  分支及 encoder，因此只保证共同权重下初始前向等价，不保护后续轨迹。v3 的
  零初始化残差输出投影在首次更新前阻断残差 MLP 上游梯度；差分辅助目标从
  第一步起给输出投影非零梯度，预测头自我启动。
- codebook 始终进入 embedding 学习率的 AdamW，既不使用 MuonH，也不把
  三维 PQ 表拍平成 Sinkhorn 矩阵。其余原训练配方不变。

同结构关闭辅助项的对照用 `--ncp_loss_weight 0 --ncp_vq_weight 0`，保留预测和
状态计算路径。状态门开启后，NTP 可直接训练概念层、记忆更新、状态投影与深度
路由；该路径不经过预测头。其有用性不能证明 NCP/VQ 监督本身有额外收益。

模型分别返回 `lm_loss / ncp_loss / vq_loss`；训练器将联合目标纳入原累积
窗口和显式参数编译函数。SwanLab 的 `ncp/loss`、`ncp/vq_loss`、`ncp/groups`、
`ncp/pairs`、`ncp/gate` 为当前记录微批的指标。TailSFT 的主 token CE 筛选
保持原逻辑，NCP/VQ 仍按整批合法概念计算；TailSFT 日志优先报告其自身指标。

## 增量状态与兼容性

每条序列增加：逐概念层一份紧凑已观察 K=V、至多 k−1 个未完成组的 encoder
表示、当前连续预测、各概念层的最后一个已投影状态，以及 token 时钟/结构签名。
不保存完整宽度的历史概念状态，不重算历史层输出，不将预测当成已观察历史。

默认两份概念 K=V 共 `2 * floor(T/4) * 128` 个元素，摊到原 token 为 64 个，
v1 为 32 个。另有当前状态 `2*dim`、当前预测 `dim` 和未完成组等开销；这不是
总显存翻倍。`ConceptCache.row_copy/nbytes` 包含新增数组，既有 cache/engine
搬行和 snapshot API 自动运输并统计它们。

- 原生 `prefill/decode_step/generate` 支持非整组 prompt、分块续写和同钟 batch。
- 连续 batch 保留完整 token 批计算，概念分支按各行自己的完成时刻推进。
  搬行、扩容/压实、prefix snapshot/restore、DSpark scratch 与提交/拒绝
  状态都搬运分层概念 KV、未完成组、当前预测和当前投影状态。缓存字节统计包含概念状态。
- packed/PAD 训练和整段无缓存评估受支持；**带缓存的输入仍要求每行一个无
  PAD 文档**。不能将 packed prefill 的末文档状态无声当作普通单文档缓存。
- 原生 `rewind` 要求 fresh prefill；engine 的投机拒绝通过完整状态快照恢复，
  不使用这个不完整的回退接口。结构签名或 token 时钟不匹配会在主干执行前报错。
- 旧 sidecar 缺少 NCP 字段时按旧普通 CED 加载，保证旧权重可评估；这不改变
  新建模型的默认。旧 NCP checkpoint 缺少新执行版本时不冒充新结构。
- 普通 CED 权重迁移到默认新架构，需要显式 checkpoint 与 `--reset_optimizer`
  （使用新输出目录）。只允许补初始化 `model.ncp.*`，共同权重仍严格加载。
  v1 → v2 同样要求显式重置 optimizer，仅允许补初始化新增记忆投影和状态路由等
  指定参数前缀；原概念权重仍严格加载。v1/v2 → v3 也要求显式重置 optimizer，
  允许补初始化 `pool_slots` 与残差头前缀（`residual_norm.`、`residual_in.`、
  `residual_out.`），并丢弃 checkpoint 中已删除的 PQ 预测头键
  （`model.ncp.predict.`、`model.ncp.predict_norm.`）；codebook 与其余概念权重
  严格加载。新架构 checkpoint 缺少概念权重会报错。更换 loss 权重或
  结构也改变 execution identity，不能静默续用旧 optimizer。

## 计算量与验证边界

`trainer/flops.py` 在原 token 层成本上加入按概念频率折算的投影、SwiGLU、
预测头（v3 残差 MLP 或 v1/v2 PQ）/feedback/状态投影 GEMM、每层概念 QK/AV 和
最近码字搜索；深度路由矩阵按
token 频率计数（即使短序列还无完整概念）。训练权重计数包含新增矩阵（v3
残差头为 2·dim²，替代 PQ 的 dim·groups·codes）。
该统计仍是原有名义 FLOPs 近似，省略归一化/池化/softmax、深度加权和提升、optimizer 等；
packed 采用 `floor(T/k)` 静态容量。概念注意力仍随概念长度二次增长，不是
长上下文性能验收。增量反馈投影和按行 dispatch 的实际开销需单独计时。

机制检查入口：

```bash
python3 scripts/check_repo.py
python3 scripts/check_repo.py test ncp
```

定向验证覆盖零门控等价、全局 KV 数值/梯度独立性、query 保留注入、因果对齐、
同文档/PAD/tail、NCP/VQ 梯度边界、FP32/BF16 编译反向、真实训练器参数更新、
原生 cache、连续 batch、prefix、DSpark 拒绝恢复、sidecar 和严格权重加载。
v1 历史结果保存在 `research_runs/ncp_ced_20260914/`；v2 结果保存于
`research_runs/ncp_state_v2_20260914/`，新增加工后历史读取、状态单独 NTP 梯度及
v1 加载/迁移检查；缓存、因果隔离、编译反向和 engine 测试均开启状态门验证。
v3 本轮新增 copy 恒等（初始 `ncp/loss ≈ previous_mse`、
`relative_mse_previous ≈ 1`）、零初始化残差头在全局 trunc-normal init 后保持
为零、池化凸权重与槽序、零初始化梯度边界、v1/v2 → v3 暖启动与 PQ 键丢弃
检查。v3 轮 46 项 NCP 定向测试与 16 项配置测试通过。没有启动长训练，
没有建立验证 CE、token efficiency、吞吐或总显存收益结论。

## 训练状态诊断（2026-09-14 补充）

日志继续保留原五项，新增以下 `ncp/` 指标。所有诊断在图内 detach，不加入
训练目标，不增加可训练参数，也不改变 checkpoint execution identity。
统计为当前记录微批，不能视作完整累积窗口或整个数据集的均值。

| 指标 | 统计口径 |
| --- | --- |
| `concept_rms` | 所有完整有效概念的 RMS，含没有下一概念监督的最后一组 |
| `concept_variance` | 合并有效概念后，逐通道总体方差的均值 |
| `sample_mean_variance` | 每个 batch 行先求有效概念均值，再计算这些均值的逐通道总体方差；有效行等权 |
| `valid_samples` | 含至少一个有效完整概念的 batch 行数；小于 2 时不能用跨样本方差判断多样性 |
| `pool_norm_weight_rms` | 可训练池化 LayerNorm gain 的 RMS |
| `target_energy` / `zero_mse` | 相同有效相邻组上，下一概念的逐维平方均值，即零预测的 MSE |
| `previous_mse` | 相同监督位置上，用当前已观察概念直接预测下一概念的 MSE |
| `relative_mse_zero` | 实际下一概念 MSE / 零预测 MSE |
| `relative_mse_previous` | 实际下一概念 MSE / 当前概念持久预测 MSE |
| `encoder_rms` | 注入前实际 mHC 边界读出的 RMS，排除 PAD |
| `feedback_rms` | 注入后与注入前实际 mHC 读出的差值 RMS，包含提升和工作 dtype 舍入 |
| `feedback_ratio` | 边界层合并预测/状态提升的 `feedback_rms / encoder_rms`（不包含后续接入层）；所有非 PAD token 等权，包括尚无概念的开头位置 |
| `state_gate` | 各接入层状态门控的均值，detach 读数；v1 无此通路，恒为 0 |
| `prediction_gate` | 后续接入层预测门控的均值，detach 读数；v1 恒为 0 |

无有效概念/监督对的统计记 0；结合原 `groups/pairs` 判断是否可用。相对误差
分母下限为 `1e-12`，目标能量接近零时须查看原始分母，不能孤立解释比值。
`relative_mse_zero < 1` 表示胜过零预测；`relative_mse_previous < 1` 表示
胜过复制当前概念（v3 初始严格为 1.0，见上方损失定义）。这仍不证明 token CE 改善。
跨样本均值方差也受到 packed
文档组成和长度影响，不能单独用作完整的概念塌缩判据。

历史 v1 诊断补充只新增日志，当时无需重置 optimizer，也未重启或终止训练。
v2/v3 均为结构修改，迁移必须遵循上面的显式 optimizer-reset 契约。

### 固定验证集上的概念贡献评估

新增 `experiments/eval_ncp_contribution.py`，对同一个 checkpoint 和同一批验证
输入顺序运行 normal/off/swap，分别统计 **NTP CE**，不把 NCP/VQ loss 混入。
每种条件独立无缓存前向：v2/v3 同时干预所有接入层的预测与状态通路，保留干净 CED 全局 KV。
关闭/交换 API 只允许 eval 模式；训练不能误用。

准备固定的、训练未使用的 NPZ：`input_ids` 与 `labels` 均为 `[N,T]` 整数数组，
可选同形状非负 `loss_mask`。每行须为单个无 PAD 文档；`loss_mask` 仅控制计分，
不承担 attention PAD 隔离。常见构造是从每个验证文档截取 T+1 个 token：
`input_ids=tokens[:,:-1]`，`labels=tokens[:,1:]`。N 至少为 2。

```bash
# GPU 空闲时运行；使用不再被训练进程覆盖的 checkpoint 副本及配套 JSON。
.venv/bin/python experiments/eval_ncp_contribution.py \
  --checkpoint /path/to/immutable-checkpoint.safetensors \
  --data /path/to/fixed-validation.npz --batch-size 4 \
  --output research_runs/ncp_validation/paired-ce.json
```

swap 使用 batch 内固定循环置换，完整替换该行对齐后的反馈，绝不保留自己的
信号。最后一行会并入前一批，避免 singleton 自交换；最多使用 batch-size+1 行。
同一批三种条件使用相同 token/label/loss mask；全局 CE 按有效 token 权重汇总，
不是 batch CE 的简单平均。输出保存每批 CE、交换来源行、总体差值、配置和输入/
权重哈希，拒绝覆盖已有结果。batch size 改变交换来源，必须随结果记录。

`off_minus_normal > 0` 表示关闭反馈使这份 checkpoint 的 CE 变差；
`swap_minus_normal > 0` 支持样本相关反馈的重要性，但 swap 属于分布外扰动。
这些结果不能替代独立训练的普通 CED 对照，也不证明 token efficiency。

补充验证：在 GPU 正用于实际训练时，只执行 CPU 定向诊断测试和静态检查。
没有对运行中模型执行贡献评估，也没有将前一轮 GPU 测试成绩当作本轮验证。
