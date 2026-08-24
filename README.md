# Viby

基于 Apple MLX 的单设备中文大语言模型训练与推理项目。

架构为 decoder-only Transformer：顺序主干 + Kimi Linear 3:1 混合注意力
（每 4 层与最后一层为 global：full-causal NoPE GQA，`n_kv_heads=h/4`；
其余 local 为 KDA——逐通道门控 delta 规则线性注意力，衰减
`g = −5·σ(e^{A_h} z)`，满秩输出门 `y = W_o[σ(W_g x) ⊙ RMSNorm(ō)]`；
全模型无 RoPE / 滑窗；GQA 侧逐 head 无参 RMS qk-norm、XSA、默认开的
注意力输出门 `2·σ`）+ GQA K-path ShortConv 与 KDA q/k/v 短卷积 +
AttnRes（score 用 `w·RMSNorm(v_j)`，混合仍用原始 `v_j`）+ RMSNorm
（GatedNorm 包裹，`2·sigmoid` 门）+ SiTU-GLU（`β1=4, β2=25`）。
FFN 全部为 LatentMoE：路由专家在 `hidden/2` latent 空间计算；
`n_shared_experts` 个独立共享专家各宽 `moe_intermediate_size`、输出相加；
QB 分位数快照偏置做负载均衡。logit z-loss 为 `mean(lse²)`（默认 1e-4），
融合进 CE kernel。MTP（Kimi-K3 / EAGLE-3，预训练默认 depth=1：与主干
同构，输入为低/中/高三层 AttnRes 特征融合 + 下一 token 嵌入；后训练用
`trainer/train_draft.py` 冻结主干做 EAGLE-3 TTT 草稿微调）。
2D 矩阵按 `TruncNormal(0, (0.5/√hidden)²)`、|z|≤2 初始化。
`lm_head` 默认与 embedding 解绑。

## 组件

- `model/`：模型定义（VibyConfig / VibyForCausalLM）、tokenizer
- `dataset/`：预训练 / SFT / DPO 数据集
- `trainer/`：MuonH/AdamH + AdamW 混合优化器（compute 自动缩放超参）、
  训练循环、检查点管理
- `model/kernels/`：手写融合 Metal kernel（flash 注意力反向、KDA
  chunk/scan、conv、SiTU、CE、decode 路径）与训练前预热
- `experiments/`：性能 probe / bench / sweep / verify / A-B 脚本
- `eval_model.py`：交互式 / 自动评估脚本
- `test_consistency.py`：架构正确性回归（因果性、prefill/decode、padding、
  MoE）；`test_align.py`：Marin / K3 公式对齐；`test_kda.py`：KDA 数值
- `research/MLX_PERF.md`：**MLX / Apple Silicon 性能优化手册**——硬件
  roofline 基线、测量口径纪律、mx.compile 与 prewarm 的坑、算子选择
  清单、手写 Metal kernel 范式、失败方案清单。做性能优化前先读它

## 架构开关

```bash
--use_attn_gate / --no-use_attn_gate   # 默认开：2·σ，零初始化门=1，进 Adam
--mtp_depth 1          # MTP 深度（K3/EAGLE-3，默认 1，0 关闭）
--mtp_loss_weight 0.3
# EAGLE-3 草稿微调（冻结主干，只训练 MTP 层，TTT rollout）：
# python trainer/train_draft.py --draft_ttt_steps 4 --data_path ...
--z_loss_weight 1e-4   # logit z-loss：loss += w · mean(lse²)，0 关闭
--tie_word_embeddings  # 绑定输入/输出 embedding（默认 False：独立 lm_head）
--pack_sequences       # 预训练序列打包（消除 padding 浪费）
--doc_mask             # 打包时屏蔽跨文档注意力与边界 loss（需配合 --pack_sequences）
# MoE（LatentMoE + QB 路由，所有层 FFN 均为 MoE）：
--n_routed_experts 32      # 路由专家数（必须 >0）
--num_experts_per_tok 6    # 每 token 激活专家数
--n_shared_experts 2       # 独立共享专家数（每个中间维 = moe_intermediate_size）
--moe_intermediate_size 384  # 单个路由/共享专家中间维
--routed_scaling_factor 2.5  # sigmoid 归一化后的路由权重缩放
--moe_latent_dim           # 路由专家 latent 维度（默认 hidden//2；0 关闭）
--moe_aux_loss_weight 0.001  # 软负载均衡辅助损失（轻正则，0 关闭）
# 学习率：线性 warmup（默认总步数 1%）+ 线性衰减到 --min_lr_ratio（默认 0.05）
# 训练超参（Hyperball 口径 compute 缩放，默认开启 --lr_scale_auto）：
# adam_lr = 0.0876·tokens^-0.3461·hidden^-0.3448·√tpb，muon_lr = 13/3×adam_lr，
# beta2/eps 同样按 tokens/tpb 推导；--token_budget 显式给预算，
# 显式 --learning_rate 覆盖 adam_lr。--muonh（默认开）：Muon 组加
# Frobenius 范数球投影；Q/K/V 按 head 切开 NS；堆叠专家逐专家 NS；
# lm_head 走 AdamH；短卷积（含 KDA q/k/v_conv）与 attn_gate 走 Adam。
```

1080M 常用配方：

```bash
python trainer/train_pretrain.py --hidden_size 768 --num_hidden_layers 8 \
  --num_attention_heads 8 --n_routed_experts 256 --num_experts_per_tok 8 \
  --n_shared_experts 2 --moe_intermediate_size 384 --batch_size 12 \
  --accumulation_steps 2 --max_seq_len 1024 --pack_sequences --doc_mask \
  --compile_model --muonh
```

注：MoE 负载均衡走 QB（Quantile Balancing）：训练期 forward 记录各专家
margin（选择分 − per-token 阈值 alpha），每优化器步取 margin 的
(1−K/E) 上分位数、零均值化后覆写 frozen 的 expert_bias（无梯度、
无 EMA rate 超参）。

MoE 按 (token,choice) 对数 `G = B×T×top_k` 分三条路径（形状只随
`(B,T,E,K)` 变，训练可 `mx.compile`）：

- `G <= 512`（decode / 极小批量）：融合 Metal kernel（router + SiTU-GLU
  前半 + 加权合并）。不可微，仅推理；`model.train()` 或
  `router.collect_stats` 时不走。
- `G <= 4096`（小 prefill）：稠密全专家广播 matmul，按路由权重加权。
- 更大（训练 / 大 prefill）：`mx.gather_mm(sorted_indices=True)` 免 padding
  分组 GEMM，每个 (token,choice) 只算真实行，无桶容量、无 host sync。

训练侧：

- BatchedMuon：同形状权重堆叠批量 Newton-Schulz；Q/K/V 再按 head 切开。
  2D 矩阵与堆叠专家都是每 8 步重算一次 NS（命中复用极因子）。
  `--muon_ns_steps` 默认 5。
- `--cache_limit_gb`（默认 0=不限）：Metal 空闲块缓存上限（GB）。上限内
  释放块常驻复用；大 batch 时峰值+缓存不要超物理内存。
- `--compile_model`：compile 前 `prewarm_all()` 跑完融合 kernel 校验
  （图内不能 `.item()`）。dropout>0 时自动回退 eager。

## 训练

```bash
# 预训练
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl --hidden_size 768 --num_hidden_layers 8

# 全量 SFT（需要 pretrain 检查点）
python trainer/train_full_sft.py --data_path ../dataset/sft_512.jsonl

# DPO（需要 full_sft 检查点）
python trainer/train_dpo.py --data_path ../dataset/dpo.jsonl
```

检查点以 safetensors 保存于 `--out_dir`，并带有同名 `.json` sidecar 与
`.optimizer.safetensors` 优化器状态。

注意：

- SFT / DPO 会自动从基座 checkpoint 的 sidecar JSON 继承模型结构配置，
  CLI 显式传入的结构参数优先；`save_interval` 会自动对齐到
  `accumulation_steps` 的整数倍，避免 resume 丢失梯度。

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
渲染后以 80% 概率移除空 `<think>\n\n</think>\n\n` 标签；chat template
负责展开 `<think>` / `<tool_call>` / `<tool_response>` 片段，loss 只监督
assistant 消息（含其 `reasoning_content` 与 `tool_calls`），不监督
user / system / tool 回复。超长样本保留头部截断尾部（与 MiniMind 相同），
截断后无 assistant 可监督时会打印一次警告。

DPO：

```json
{
  "chosen": [{"content": "Q", "role": "user"}, {"content": "good answer", "role": "assistant"}],
  "rejected": [{"content": "Q", "role": "user"}, {"content": "bad answer", "role": "assistant"}]
}
```

DPO 同样经过 chat template 渲染与空 `<think>` 清洗，loss mask 只覆盖
assistant 回复。

## 评估

```bash
python eval_model.py --out_dir out
```

`--model_mode` 支持 `0`（预训练）和 `1`（SFT-Chat）。脚本会自动从
`latest_checkpoint.txt` 或 `{mode}_*.safetensors` 发现检查点，并优先从
sidecar JSON 加载模型配置。
