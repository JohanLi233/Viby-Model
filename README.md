# Viby

基于 Apple MLX 的单设备中文大语言模型训练与推理项目。

架构是 **DeepSeek-V4.1 的等比例缩小版**：CED 因果编码器-解码器主干 +
CSA2 跨层复用稀疏注意力 + Single-Pass mHC 残差流 + sqrt(softplus) 路由的
细粒度 MoE + Engram n-gram 条件记忆 + DSpark 块式草稿头。旧版（Kimi Linear
3:1 KDA / iHC / AttnRes / SiTU / MLA / 短卷积）的实现已全部移除，只保留
V4.1 这一条技术路线；唯一的保留项是 **Gated XSA**（默认开，见下）。

架构语义对照官方实现（`deepseek-ai/DeepSeek-V4.1-Flash` 的 `inference/model.py`，
已逐行核对）与技术报告 `DeepSeek_V41_Tech_Report`（下称"报告"）。

## 架构

### CED：因果编码器-解码器（报告 §2.2）

主干前 `n_layers/2` 层是**因果编码器**，后一半是**解码器**。解码段的
全局（压缩）KV 不由解码层各自的 hidden 产生，而是从编码器末层 hidden
投影而来——实现上就是 `compress_ratios` 在 CED 边界从 `r` 切到 `1`，
且边界层是 `kv_source`（见下）。滑动窗口 KV 仍逐层从本层 hidden 计算。

### CSA2：压缩稀疏注意力（报告 §2.3）

每个注意力层静态属于三种模式之一（`config.layer_mode`）：

| 模式 | 主 KV | indexer K | Top-K 索引 | 本层计算 |
| :--- | :--- | :--- | :--- | :--- |
| **Full** | 自产 | 自产 | 自算 | main Q + SWA KV + indexer Q/K |
| **Reindex** | 复用上游 | 复用上游 | 自算（只在候选池里打分） | main Q + SWA KV + indexer Q |
| **Reuse** | 复用上游 | 复用上游 | 复用上游 | main Q + SWA KV |

- 主 KV 与 indexer K 由 `kv_source_layers` 指定的层产生（CED 边界层是
  解码段的全局源），其后同 ratio 的层共享同一份 cache；`Indexer.owns_k`
  只在"既是 kv 源又是 indexer 源"的层为真。
- 每个 query 的可见集合 = 滑动窗口内 `window_size` 个原始 KV ∪ indexer
  挑出的 `index_topk` 个压缩位置。
- 压缩：`compress_ratio` 个连续 token 用可学 softmax 门池化成一个 latent
  （ratio=1 即一次普通投影）；latent 的 RoPE 位置是**组首 token**的位置
  `j*r`，组 j 只对 `t ≥ (j+1)r−1` 的 query 可见。
- **分层稀疏索引**（报告 §2.3.2）：`candidate_source_layer`（解码段第一个
  Full 层）按块最大分为每个 query 选出 `candidate_topk_blocks` 个候选块
  （每块 `candidate_block_size` 个位置），更深的 indexer 只在这个候选池里
  打分，把后续 indexer 的每 query 成本从"随上下文线性"降到常数。

### 注意力细节

- **共享 K=V 的多查询注意力**：`wkv` 只产 1 个 head，同一个张量既当 K 又当 V。
- **部分 RoPE**：只旋转每个 head 尾部 `rope_head_dim` 通道，交错对形式；
  注意力**输出端用同一旋转按 −i 转回去**，使 cache 只保留一种旋转形式。
  压缩层用 `compress_rope_theta` + YaRN，纯滑窗层用 `rope_theta` 且不做 YaRN
  （对应官方实现 `original_seq_len = 0 if compress_ratio == 0`）。
- **逐 head 可学 attention sink**：进 softmax 分母的额外一项（`2.4` 之外的
  gpt-oss 式做法，官方同款）。
- **分组低秩输出投影**：`o_groups` 组 → 组内 block-diagonal 的 `wo_a` →
  `o_lora_rank` → `wo_b` 回 hidden。
- **Q 低秩**：`wq_a → q_norm → wq_b`，主注意力与 indexer 共享同一份 q 低秩表示。
- **Gated XSA**（默认开；非官方 V4.1 组件，本仓库额外挂的后处理）：对注意力输出做逐 head
  后处理，扣掉与自身 V 平行的分量
  `z = y − tanh(α)·(yᵀv/‖v‖²)·v`，α 逐 head 可学、零初始化 ⇒ **起步严格恒等**。
  作用在最深 `xsa_last_n` 层（默认 `max(1, n_layers//3)`），DSpark 草稿层不挂。
  与 CSA2 正交：只读、不改 KV cache / indexer / 候选池，故 prefill、逐 token
  解码、prefix cache、连续 batch 口径天然一致；`v` 只取本层滑窗分支
  （压缩 latent 不属于任何 query 自身，不参与）。
  这是**刻意偏离**官方实现的一处——加它是为了消融，`--no-use_xsa` 可完全关闭。
- 无短卷积、无 RoPE 之外的任何位置编码（报告 §2.4.2 明确省掉 Engram 的短卷积）。

### mHC 与 Single-Pass（报告 §2.4.1）

残差流是 `hc_mult` 条并行副本 `[B,T,hc,dim]`。每个子层的读写系数
`(pre, post, comb)` 由流本身现算：`pre = σ(m·s₀+b₀)+ε`、
`post = 2σ(m·s₁+b₁)`、`comb` 经 Sinkhorn–Knopp 迭代投影到**双随机矩阵**
（信号传播非扩张）。Single-Pass 的"系数错位一格"体现为：注意力用的是
**上一层 FFN 产出**的 pre_mix，FFN 用的是本层注意力产出的 pre_mix。

### MoE（报告 §2.1）

- 亲和度 `sqrt(softplus(x·W)) `（不再是 V3 的 sigmoid），**没有** n_group /
  topk_group 分组约束；top-k 后按未归一化分数归一化并乘 `route_scale`。
- **noaux_tc 偏置**：`e_score_correction_bias` 只参与 top-k 选择、不进梯度
  （`freeze`，但随 checkpoint 保存），训练循环按
  `b += γ·sign(load_frac − 1/E)` 覆写。
- **clamped SwiGLU 专家**：`silu(clamp(gate, max=L)) * clamp(up, ±L)`，
  `L = swiglu_limit`；1 个共享专家每 token 必走。
- 分发两条路径：小 batch 走稠密广播 matmul，训练/大 prefill 走
  `mx.gather_mm(sorted_indices=True)` 的免 padding 分段 GEMM。

### Engram（报告 §2.4.2）

挂在若干层入口的 n-gram 条件记忆：token 先过 normalizer 压到小 id 空间
（" The"/"the"/"THE" 同形），再按 `{2,3,4}`-gram 做多头素数哈希（每
(层, 阶, 头) 独占一段素数桶），查表得 key/value，用**与残差流的归一化点积**
做门控写入（带符号平方根 + sigmoid）。无短卷积。表是 fp32 全表 gather
（官方是 fp8 分片 + RDMA 预取）。

### DSpark / MTP（报告 §2.4.3）

主干后挂草稿层：取主干的 `dspark_target_layer_ids` 层的**注意力输入**（mHC
流均值）拼起来投影成锚点表示；锚点 t 的草稿序列是 `[x_t, 噪声, …, 噪声]`，
一次前向算出 `dspark_block_size` 个草稿位置的 base logits，马尔可夫头按
前一个 token 的嵌入加低秩 logits 偏置，置信度头预测每个位置的接受概率。
训练用 teacher forcing（槽 j 看 token t+j、预测 t+j+1），**DSpark 目标不回传
主干**（`stop_gradient`，与报告"只训 DSpark、主干冻结"一致）。

## 缩放配方

| | V4.1-Flash（官方） | Viby 默认（≈1B） | Viby tiny（自测/冒烟） |
| :--- | :--- | :--- | :--- |
| dim / 层数 | 5120 / 40（20+20） | 1024 / 12（6+6） | 256 / 4（2+2） |
| 注意力头 | 64 × 512 | 16 × 128 | 4 × 64 |
| q_lora / o_groups × o_lora | 1280 / 8 × 1024 | 512 / 4 × 256 | 128 / 2 × 64 |
| rope_head_dim | 64 | 32 | 16 |
| MoE 专家 / 激活 / 共享 | 384 / 6 / 1，inter 2304 | 96 / 6 / 1，inter 256 | 16 / 4 / 1，inter 128 |
| 压缩率（逐层） | [0,0] + [2]×18 + [1]×20 | [0,0] + [2]×4 + [1]×6 | [0,0] + [1]×2 |
| kv 源 / indexer 源 | 2,8,14,20 / +24,28,32,36 | 2,6 / 2,6,10 | 2 / 2 |
| 候选池 | 2048 块 × 8 | 64 块 × 8 | 4 块 × 4 |
| indexer | 32 头 × 128，top-512 | 8 头 × 64，top-64 | 4 头 × 32，top-16 |
| 滑窗 | 128 | 128 | 32 |
| hc_mult | 4 | 4 | 4 |
| Engram | 2 层 × 196B 参数 | 2 层（[1,4]），4 头 × 64 | 2 层，2 头 × 32 |
| 总参 / 激活 | 552B（+196B Engram）/ 8–16B | **1.02B / 0.13B** | 17M |
| DSpark | 3 层草稿，5 位置 | 1 层草稿，4 位置 | 1 层，3 位置 |

`VibyConfig(preset="tiny")` 与 `VibyConfig()`（默认 ≈1B）两套预设，
其余字段都能在构造时覆盖。

## 组件

- `model/`：`config.py`（VibyConfig）、`model.py`（VibyModel / VibyForCausalLM /
  DSpark）、`block.py`、`attention.py`（Compressor / Indexer / Attention）、
  `hc.py`（mHC）、`moe.py`、`engram.py`、`rope.py`、`norms.py`、
  `cache.py`（解码状态）、`init.py`、tokenizer
- `trainer/`：预训练 / SFT / DPO / 训练循环 / 检查点 / MuonH·AdamH 混合优化器
  （compute 自动缩放超参）
- `engine/`：进程内连续 batch 推理引擎（新缓存结构）
- `dataset/`：预训练 / SFT / DPO 数据集（与 MiniMind 格式对齐）
- `tests/`：架构语义回归（CSA2 三种模式、分层索引、CED、mHC 双随机、MoE
  sqrtsoftplus/noaux_tc、Engram 哈希、DSpark 梯度隔离、prefill/decode 一致性）
- `research/`、`research_runs/`、`swanlog/`、`autoresearch-mlx/`：历史资料
  （优化器研究、训练日志、旧架构时期的实验记录），不是当前架构的一部分

## 训练

```bash
# 冒烟（tiny，几分钟）
python trainer/train_pretrain.py --preset tiny --data_path ../dataset/pretrain_hq.jsonl \
  --max_seq_len 128 --batch_size 2 --accumulation_steps 1 --max_steps 50

# ≈1B 配方（与 V4.1 结构比例一致）
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl \
  --hidden_size 1024 --num_hidden_layers 12 --num_attention_heads 16 \
  --n_routed_experts 96 --num_experts_per_tok 6 --moe_intermediate_size 256 \
  --batch_size 12 --accumulation_steps 2 --max_seq_len 1024 \
  --pack_sequences --doc_mask --compile_model --muonh

# 全量 SFT（需要 pretrain 检查点）
python trainer/train_full_sft.py --data_path ../dataset/sft_512.jsonl

# DPO（需要 full_sft 检查点）
python trainer/train_dpo.py --data_path ../dataset/dpo.jsonl

# DSpark 独立阶段（报告 §2.4.3）：预训练默认 --mtp_depth 0（§2.1：骨干预训练省略
# MTP 模块），草稿层在这一阶段才引入并从零初始化；可训练集合变了会自动重置
# 优化器状态。
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl \
  --resume /path/to/pretrain.safetensors --mtp_depth 1 --freeze_backbone \
  --mtp_loss_weight 1.0
```

要点：

- 结构参数只走 `VibyConfig`：`--preset`、`--hidden_size`、`--num_hidden_layers`、
  `--num_attention_heads`、`--head_dim`、`--rope_head_dim`、`--q_lora_rank`、
  `--o_groups`、`--o_lora_rank`、`--window_size`、`--hc_mult`、
  `--compress_ratios`、`--kv_source_layers`、`--index_source_layers`、
  `--index_n_heads`、`--index_head_dim`、`--index_topk`、
  `--candidate_source_layer`、`--candidate_topk_blocks`、`--candidate_block_size`、
  `--n_routed_experts`、`--num_experts_per_tok`、`--moe_intermediate_size`、
  `--score_func`、`--route_scale`、`--swiglu_limit`、`--bias_update_rate`、
  `--engram_*`、`--mtp_depth`、`--dspark_*`、`--z_loss_weight`、
  `--tie_word_embeddings`。
- 检查点：`--out_dir` 下 safetensors + 同名 `.json` sidecar（结构配置）+
  `.optimizer.safetensors`；SFT/DPO 自动从基座 sidecar 继承结构。
- **优化器分组**（对齐 V4.1 报告 §2.5/§4.2.2）：Muon 走核心权重矩阵，其中 Query
  权重 `wq_b` **默认逐头 NS**（注意力 `head_dim`、indexer `index_head_dim`；
  K=V 共享的单头 `wkv` 整矩阵）；token 嵌入 / lm_head / DSpark 马尔可夫头与
  Engram 检索表走 **Sinkhorn 均衡更新**（动量 0.95、γ=0.18、K=11、τ=1e-3、
  ε=1e-20、无 wd；Engram 表 5× lr）；RMSNorm 权重走 AdamW 且**受 wd**；
  偏置/缩放因子（attn_sink、mHC scale/base、Engram q_weight/k_weight）走
  AdamW wd=0。分组变更（切 `--muonh`、`VIBY_SINKHORN=0`、或
  `VIBY_MUONH_PER_HEAD=0` 消融）后旧优化器状态不能直接续，加 `--reset_optimizer`。
- `--pack_sequences` / `--doc_mask`：打包与跨文档屏蔽。压缩组按"整组同
  文档"处理，跨文档的组直接不可见（选择阶段就隔离，不会泄漏）。
- 损失口径对齐报告 §4.2.2：预训练默认**不算 z-loss**（`--z_loss_weight 0`），
  改为权重 `1e-4` 的**序列级均衡损失**（`MoEFeedForward.seq_aux_loss`，与 DeepSeek
  官方 seq-aux 同式，逐序列计算抑制单序列内极端不均衡）；要开 z-loss 显式
  传 `--z_loss_weight 1e-4`（字段保留，作稳定性消融）。

## 数据格式（与 MiniMind 对齐）

数据处理管线与 [minimind](https://github.com/jingyaogong/minimind) 对齐，
可直接使用其发布的 `pretrain_t2t*.jsonl` / `sft_t2t*.jsonl` / `dpo.jsonl`
等数据集，无需转换。

预训练（未开启 `--pack_sequences` 时）：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
```

处理方式与 MiniMind 一致：`add_special_tokens=False` 分词后显式包裹
`[bos] + text + [eos]`，截断长度预留 bos/eos 位置，所有非 PAD 位置都参与
next-token loss。

SFT（多轮对话，可选 `reasoning_content` / `tools` / `tool_calls` / `tool` 角色）：

```jsonl
{"conversations": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]}
```

```jsonl
{"conversations": [
  {"role": "system", "content": "# Tools", "tools": "[{\"name\":\"calc\",\"description\":\"x\",\"parameters\":{}}]"},
  {"role": "user", "content": "帮我算 256*37"},
  {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calc\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
  {"role": "tool", "content": "{\"result\":\"9472\"}"},
  {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
]}
```

处理方式与 MiniMind 一致：无 system 时以 20% 概率补一条随机 system；
渲染后以 80% 概率移除空 `<think>\n\n</think>\n\n` 标签；chat template 负责
展开 `<think>` / `<tool_call>` / `<tool_response>` 片段，loss 只监督 assistant
消息（含其 `reasoning_content` 与 `tool_calls`），不监督 user / system / tool。
超长样本保留头部截断尾部（与 MiniMind 相同），截断后无 assistant 可监督时
会打印一次警告。

DPO：

```json
{
  "chosen": [{"content": "Q", "role": "user"}, {"content": "good answer", "role": "assistant"}],
  "rejected": [{"content": "Q", "role": "user"}, {"content": "bad answer", "role": "assistant"}]
}
```

DPO 同样经过 chat template 渲染与空 `<think>` 清洗，loss mask 只覆盖 assistant 回复。

## 评估与推理

```bash
python eval_model.py --out_dir out          # 交互式 / 自动评估
```

`--model_mode` 支持 `0`（预训练）、`1`（SFT-Chat）、`2`（DPO）；脚本从
`latest_checkpoint.txt` 或 `{mode}_*.safetensors` 发现检查点，并优先从
sidecar JSON 加载结构配置。

直接用模型 API：

```python
from model.config import VibyConfig
from model.model import VibyForCausalLM
model = VibyForCausalLM(VibyConfig())
logits, cache = model.prefill(tokens)          # 整段 prefill
logits, cache = model.decode_step(next_tok, cache)   # 逐 token（cache 自带 start_pos）
out = model.generate(tokens, max_new_tokens=64, temperature=0.7, top_k=50)
```

## 与 V4.1 技术报告的逐条对齐

已按报告 §2 / §4.2 对齐的部分（均有数值或测试支撑）：

| 报告条目 | 本项目实现 |
| :--- | :--- |
| §2.1 主干 40 层 = 20 编码 + 20 解码；前两层纯 SWA，其余每层 SWA + 全局 | n_encoder_layers = n_layers//2；compress_ratios 前两位为 0 |
| §2.2 CED：解码段全局 KV 由第 L/2 层隐状态投影而来 | CED 边界层是 kv_source（Full 模式），解码层以 Reuse/Reindex 共享其主 KV |
| §2.3 CSA2 三模式；主 KV 与 indexer K 跨层共享，Top-K 索引可复用 | config.layer_mode() + kv_source_layers / index_source_layers |
| §2.3 压缩器去掉重叠与绝对位置编码；indexer K 由主 KV 投影得到 | Compressor 每 r 个 token 池化一次（无重叠、无绝对 PE）；Indexer.index_keys = wk(latent) |
| §2.3.2 分层稀疏索引：Full 层建候选池，Reindex 只在池内打分 | candidate_source_layer + select_candidate_blocks |
| §4.2.1 逐层排布（编码器 3x6 组、解码器 5x4 组，组首 Full/Reindex、其余 Reuse） | default_source_layers() 推导出同构排布（缩放后） |
| §2.4.1 Single-Pass mHC：每个子层消费上一个子层产出的混合系数 | Block.forward 的 pre_mix 链（tests/test_v41_mhc_single_pass.py 逐调用点断言） |
| §2.4.2 Engram：分词器压缩 + 多头素数哈希 + 上下文门控 + 无短因果卷积；放在第 1/14 层 | model/engram.py（与官方 torch 参考逐位一致）；缩放后放 [1, ~35% 深度] |
| §2.4.3 DSpark：块式草稿 + Markov 头 + 置信度头；预训练省略 MTP；单独阶段冻结主干；后训练不回传主干 | DSparkStage + --freeze_backbone + stop_gradient；预训练默认 --mtp_depth 0 |
| §2.5 逐头 Muon（Query 权重按头拆）；Engram/嵌入/预测头用动量 + Sinkhorn 均衡；norm 权重受 wd、偏置与缩放因子不受 | trainer/muon.py：_per_head_dim、SinkhornBalanced、adamw_norm / adamw_nowd 分组 |
| §4.2.2 超参（Muon m=0.95/wd=0.1；AdamW beta2=0.95/wd=0.1；Sinkhorn gamma=0.18/K=11/tau=1e-3/eps=1e-20；Engram lr x5） | 同（Sinkhorn 与官方 numpy 对拍 max|dW|=8.9e-9） |
| §4.2.2 无辅助损失负载均衡（bias 更新率 1e-3）+ 权重 1e-4 的序列级均衡损失 | update_expert_bias + MoEFeedForward.seq_aux_loss（与 DeepSeek 官方 seq-aux 同式） |
| §4.2.2 学习率：预热 -> 平台 -> 余弦衰减 | --lr_schedule wsd --wsd_decay_shape cosine（默认） |

## 与官方 V4.1 的刻意偏差

缩小规模之外，以下部分没有实现（或用了等价但更简单的做法），用之前请知悉：

1. **多模态**：官方 V4.1-Flash 带 ViT + Aligner，本项目是纯文本。
2. **量化**：官方 fp8 权重 / fp4 主 KV（QAT）；本项目训练与推理都用 bf16。
3. **稀疏算力**：训练/prefill 走稠密掩码注意力——可见集合与官方**逐位等价**
   （已验证），但没有官方融合稀疏 kernel 的省算力效果；解码路径才是 gather 稀疏。
4. **DSpark**：只挂 1 个草稿 stage（官方 3），5→4 个草稿位置；投机解码的
   置信度调度采样循环未实现（只有草稿头前向 + 训练损失）。
5. **Engram**：表规模按比例缩小；表更新已按报告实现 momentum + Sinkhorn 均衡
   （§2.5 算法 1、§4.2.2 超参 K=11/τ=1e-3/ε=1e-20、Engram 5× lr）；不分片、
   不 fp8。
6. **SWA Bounded Replay**（官方部署期只重放最近 n_win 个 token 的优化）未实现。
7. `norm_eps` 用 1e-6（官方 fp8 口径 1e-20），bf16 训练更稳。
8. **Muon 的更新 RMS 重缩放**（报告 §4.2.2：将每个更新矩阵的 RMS 重缩放至 0.18 以复用 AdamW 学习率）未实现：仓库沿用自校准的 muon_lr = 13/3 x adam_lr + hyperball 范数冻结 + 自研 NS 系数（有独立 probe 校准）。切到报告口径需要重做步长标定，属单独一项工作。
9. **AdamW 的 eps** 仍走 --adam_eps（compute 缩放公式或 1e-8 兜底），没有默认成报告基线的 1e-20；严格对齐可显式 --adam_eps 1e-20。
10. **MoE router** 仍是独立小 lr AdamW 组（报告口径的线性权重走 Muon）：正交化步长恒定偏大，实测会把路由打分持续推向失衡。
11. **DSpark 规模**：只挂 1 个草稿 stage（官方 3）x 4 个草稿位置（官方 5），置信度调度的投机采样循环未实现。
12. **Gated XSA 默认开**：XSA 不是官方 V4.1 组件，是本仓库额外挂的注意力后处理
    （arXiv:2603.09078 / modded-nanogpt record #82）。默认 `use_xsa=True` 且作用于
    最深 `max(1, n_layers//3)` 层。要严格对齐官方口径请加 `--no-use_xsa`。

## 已知问题

- **SFT（带 padding 的批次）偶发非有限梯度/前向**：同一命令重复跑，约 1/3~2/3 的运行会在第 1~3 个微批出现 grad_norm=nan（随后该微批 loss=nan），其余运行全程干净；预训练（无 pad）跑 30+ 步未复现。
  已排查并排除：权重/模块状态/输入**全部有限**（逐张量枚举过 parameters 与私有 buffer）、同一批数据与权重拿到进程外纯模型重复前向+反传 40 次全有限、dump 出的 X/Y/M 逐字节相同、与优化器分组无关（Sinkhorn 新分组与 VIBY_SINKHORN=0 旧分组都会复现，也都能 3/3 干净）、与 --no_muonh / --no_compile 无关。
  签名（随机、进程相关、同一图重复执行结果不同）指向 MLX/Metal 侧的核间竞态或未初始化读，怀疑点是 mx.fast.scaled_dot_product_attention（bool mask + sinks）或 MoE 的 scatter/gather。
  缓解：训练循环的 NaN 守卫会跳过该累积窗口（不污染权重）；若同一进程内前向本身开始出 NaN，重启该次运行。下一步建议逐项关掉 sdpa sinks / MoE 稀疏路径做二分定位（尚未做）。
- `--use_mtp_speculative` / `num_speculative_tokens` 被接受但被忽略：DSpark 的
  投机解码采样循环未实现（只有草稿头前向与训练损失）。

## 验证

```bash
.venv/bin/python -m pytest tests/ -q
```

已测：整段 prefill vs prefix+逐 token 解码（`max|Δlogit| ≈ 1e-6`，含 ratio=2
压缩器）、同位置批量解码 vs 单条、packed 文档隔离、mx.compile 前向、
`nn.value_and_grad` 全参数可微、Engram 哈希与官方 PyTorch 参考逐位一致。
