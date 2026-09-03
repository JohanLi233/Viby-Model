# Viby

基于 Apple MLX 的单设备中文大语言模型训练与推理项目。

架构为 decoder-only Transformer：顺序主干 + Kimi Linear 3:1 混合注意力
（每 4 层与最后一层为 global：full-causal NoPE GQA，`n_kv_heads=h/4`；
其余 local 为 KDA——逐通道门控 delta 规则线性注意力，衰减
`g = −5·σ(e^{A_h} z)`，满秩输出门 `y = W_o[2·σ(W_g x) ⊙ RMSNorm(ō)]`
（`W_g` 零初始化，初始门 1）；
全模型无 RoPE / 滑窗；GQA 侧逐 head 无参 RMS qk-norm、XSA、默认开的
注意力输出门 `2·σ`）+ GQA K-path ShortConv 与 KDA q/k/v 短卷积 +
AttnRes（score 用 `w·RMSNorm(v_j)`，混合仍用原始 `v_j`）+ RMSNorm
（GatedNorm 包裹，`2·sigmoid` 门）+ SiLU-GLU / SwiGLU（默认
`silu(g)·u`；`hidden_act=situ` 可切回 SiTU-GLU `β1=4, β2=25`）。
FFN 全部为 LatentMoE：路由专家在 `hidden/2` latent 空间计算；
`n_shared_experts` 个独立共享专家各宽 `moe_intermediate_size`、输出相加；
QB 分位数快照偏置做负载均衡。logit z-loss 为 `mean(lse²)`（默认 1e-4），
融合进 CE kernel。MTP（Qwen3.8-Next 口径，预训练默认 depth=1：单层
full-attn MoE，输入为主干末层 hidden + 下一 token 嵌入，`eh_proj
(enorm(e)∥hnorm(h))`，teacher-forced 展开 `--mtp_steps` 次默认 2；后训练
用 `trainer/train_draft.py` 冻结主干做 TTT 草稿微调）。
2D 矩阵按 `TruncNormal(0, (0.5/√fan_in)²)`、|z|≤2 初始化。
`lm_head` 默认与 embedding 解绑。

## 组件

- `model/`：模型定义（VibyConfig / VibyForCausalLM）、tokenizer
- `dataset/`：预训练 / SFT / DPO 数据集
- `trainer/`：MuonH/AdamH + AdamW 混合优化器（compute 自动缩放超参）、
  训练循环、检查点管理
- `model/kernels/`：手写融合 Metal kernel（flash 注意力反向、KDA
  chunk/scan、conv、SiLU-GLU / SiTU-GLU 融合核（`hidden_act` 二选一，
  silu 默认）、CE、decode 路径）与训练前预热
- `experiments/`：性能 probe / bench / sweep / verify / A-B 脚本
- `eval_model.py`：交互式 / 自动评估脚本
- `tests/`：全部测试。`test_consistency.py`：架构正确性回归（因果性、
  prefill/decode、padding、MoE）；`test_align.py`：Marin / K3 公式对齐；
  `test_kda.py`：KDA 数值；`test_pack_dataset.py` / `test_pack_sft.py`：
  pretrain/SFT 打包对拍
- `research/MLX_PERF.md`：**MLX / Apple Silicon 性能优化手册**——硬件
  roofline 基线、测量口径纪律、mx.compile 与 prewarm 的坑、算子选择
  清单、手写 Metal kernel 范式、失败方案清单。做性能优化前先读它

## 架构开关

```bash
--use_attn_gate / --no-use_attn_gate   # 默认开：2·σ，零初始化门=1，进 Adam
--use_xsa / --no-use_xsa   # 默认开：MLAy 的 Gated XSA（可学习版 Exclusive
                          # Self-Attention）。逐 head 学 tanh(α)，α=0 初始化
                          # ⇒ 恒等；z = y − tanh(α)·(yᵀv/‖v‖²)·v。头 1:1，
                          # 无需 GQA 的 V-扩展
--xsa_last_n 0            # 应用的最深 N 层；0=自动取最深 ≈1/3 层
                          # （`max(1, num_hidden_layers // 3)`）；取
                          # num_hidden_layers 则全层。
--attn_res_register / --no-attn_res_register  # 默认关：替换式 AttnRes。开=加法写、AttnRes 只读
--attn_res_read_h / --no-attn_res_read_h  # 默认关。开=寄存器读 h，不再混合 [h]+写入（需 register）
--ihc / --no-ihc  # 默认关。开=iHC（M 条流、H_res=I）；开时接管残差，AttnRes 开关保留但不走 merge
--ihc_streams 4   # iHC 流数（仅 --ihc 时生效）
--ihc_typed / --no-ihc_typed  # 默认关。开=分型流：n-gram 只进一流、进栈前注入，collapse 默认 identity
--ihc_collapse {mean,identity}  # iHC 出口；typed 且未指定时为 identity
--ihc_ngram_stream 1  # 分型时 n-gram 写入的流（identity 塌缩时不能为 0）
--ngram_logit_skip / --no-ngram_logit_skip  # 默认关。开=n-gram 经 lm_head 残差到 logits
--ngram_conf_gate / --no-ngram_conf_gate  # 默认关。开=n-gram 置信度缩放 attn/mlp 写入
--mtp_depth 1          # MTP 开关（>0 挂 1 层 full-attn，0 关闭）
--mtp_steps 2          # 预训练 teacher-forced 展开步数（Qwen3.8-Next）
--mtp_loss_weight 0.3
# 草稿微调（冻结主干，只训练 MTP 层，TTT rollout）：
# python trainer/train_draft.py --draft_ttt_steps 4 --data_path ...
--z_loss_weight 1e-4   # logit z-loss：loss += w · mean(lse²)，0 关闭
--tie_word_embeddings  # 绑定输入/输出 embedding（默认 False：独立 lm_head）
--pack_sequences       # 预训练序列打包（消除 padding 浪费）
--doc_mask             # 打包时屏蔽跨文档注意力与边界 loss（需配合 --pack_sequences）
--no-doc_align         # 关闭文档边界对齐（默认开）：打包时每块首 token 对齐到文档开头
                       # + 限制 max doc length，丢弃跨块尾部位换取边界更干净、减少跨文档
                       # 无效 attention（数据侧通用改进；对应 modded-nanogpt record #26）
--max_doc_len N        # 单篇文档最大 token 数（不含 eos），默认=max_seq_len
# MoE（LatentMoE + QB 路由；默认全层 MoE）：
--first_k_dense_replace 0  # 前 K 层小 dense stem，其后 MoE。0=全层 MoE
--dense_intermediate_size  # 浅层 dense 中间维；未传则与 moe_intermediate_size 同宽
--n_routed_experts 256     # 路由专家数（必须 >0；parser 默认 0 会直接报错）
--num_experts_per_tok 8    # 每 token 激活专家数（1080M 配方；CLI 默认 6）
--n_shared_experts 2       # 独立共享专家数（1080M 配方；CLI 默认 1）
--moe_intermediate_size 384  # 单个路由/共享专家中间维
--routed_scaling_factor 2.5  # sigmoid 归一化后的路由权重缩放
--moe_latent_dim           # 路由专家 latent 维度（默认 hidden//2；0 关闭）
--moe_write_spread / --no-moe_write_spread  # 默认关。专家写出基扩展
--moe_route_scale / --no-moe_route_scale  # 默认关。RMSNorm 后乘未归一化 top-k sigmoid 和
--moe_diversity_loss_weight 0  # router 输入多样性正则（默认关）
# 学习率（Marin Hero #8435）：线性 warmup（默认本轮 horizon 1%）后立刻
# 线性收到 --min_lr_ratio（默认 0.05），无 WSD 平台。短跑 --max_steps
# 会重算衰减斜率，结束时仍落到 0.05×peak；峰值 LR 未传 --token_budget
# 时也按该 horizon 的 token 数缩放。`--lr_schedule wsd` 才有平台。
# 预训练 parser 默认 bs=32 / accum=8 / seq=2048，会把 muon_lr 推到 ~0.03，
# 不要当 1080M 配方。1080M 用下面这条（bs=12, accum=2, seq=1024）。
# 训练超参（Hyperball / Marin 口径 compute 缩放，默认开启 --lr_scale_auto）：
# adam_lr = min(0.05, 0.087571·tokens^-0.3461·hidden^-0.3448·√tpb)，
# muon_lr = min(0.05, 13/3×adam_lr)，beta2/eps 同样按 tokens/tpb 推导；
# --token_budget 显式给原计划预算（缩短 run 时只改斜率），
# 显式 --learning_rate 覆盖 adam_lr。--muonh（默认开）：Muon 组加
# Frobenius 范数球投影；堆叠专家逐专家 NS（每步全量，无降频复用）；
# Q/K/V per-head NS 默认关（VIBY_MUONH_PER_HEAD=1 打开）；lm_head 走 AdamH；
# 短卷积、attn_gate、KDA g_proj、GatedNorm.gate_up 走 Adam 标量组（wd=0）；
# embed / router 仍 wd=0.1。
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
margin（原始 sigmoid 分 − per-token 阈值 alpha；K3 Eq.14，旧 bias 只经
alpha 进入更新），每优化器步取 margin 的
(1−K/E) 上分位数、零均值化后覆写 frozen 的 expert_bias（无梯度、
无 EMA rate 超参）。

MoE 按 (token,choice) 对数 `G = B×T×top_k` 分三条路径（形状只随
`(B,T,E,K)` 变，训练可 `mx.compile`）：

- `G <= 512`（decode / 极小批量）：融合 Metal kernel（router + SiLU-GLU
  前半 + 加权合并）。不可微，仅推理；`model.train()` 或
  `router.collect_stats` 时不走。
- `G <= 4096`（小 prefill）：稠密全专家广播 matmul，按路由权重加权。
- 更大（训练 / 大 prefill）：`mx.gather_mm(sorted_indices=True)` 免 padding
  分组 GEMM，每个 (token,choice) 只算真实行，无桶容量、无 host sync。

训练侧：

- BatchedMuon：同形状权重堆叠批量 Newton-Schulz；正交化每步全量重算
  （NS 降频复用 / Temporal Q 已删，勿重引入）。Q/K/V per-head 切分默认关。
  `--muon_ns_steps` 默认 5。
- `--cache_limit_gb`（默认 0=不限）：Metal 空闲块缓存上限（GB）。上限内
  释放块常驻复用；大 batch 时峰值+缓存不要超物理内存。
- `--compile_model`：compile 前 `prewarm_all()` 跑完融合 kernel 校验
  （图内不能 `.item()`）。dropout>0 时自动回退 eager。

## 训练

```bash
# 预训练：用上面 1080M 配方（须 --n_routed_experts >0；不要依赖 parser 默认 bs/accum/seq）
python trainer/train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl \
  --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 \
  --n_routed_experts 256 --num_experts_per_tok 8 --n_shared_experts 2 \
  --moe_intermediate_size 384 --batch_size 12 --accumulation_steps 2 \
  --max_seq_len 1024 --pack_sequences --doc_mask

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
- 优化器分组变更（MuonH 开关、专家/门控进哪一组）后，旧
  `.optimizer.safetensors` 不能直接续；加 `--reset_optimizer` 或新开 run。
  权重仍可加载。

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

`--model_mode` 支持 `0`（预训练）、`1`（SFT-Chat）和 `2`（DPO）。脚本会自动从
`latest_checkpoint.txt` 或 `{mode}_*.safetensors` 发现检查点，并优先从
sidecar JSON 加载模型配置。
