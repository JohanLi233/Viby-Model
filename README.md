# Viby

基于 Apple MLX 的单设备中文大语言模型训练与推理项目。

架构为 decoder-only Transformer：MLA（Multi-head Latent Attention，
DeepSeek V2/V3 风格）+ RoPE（解耦位置键）+ RMSNorm + SwiGLU + QK-norm，
可选 value residual、注意力输出门（attn gate）与 MTP
（multi-token prediction，默认开启 depth=1，推理侧可用于投机解码加速）。

## 组件

- `model/`：模型定义（VibyConfig / VibyForCausalLM）、tokenizer
- `dataset/`：预训练 / SFT / DPO 数据集
- `trainer/`：Muon + AdamW 混合优化器、训练循环、检查点管理
- `eval_model.py`：交互式 / 自动评估脚本
- `test_consistency.py`：架构正确性回归测试（因果性、prefill/分段/decode
  一致性、padding 等价性、loss mask），`python test_consistency.py` 运行

## 架构开关

```bash
--kv_lora_rank 192     # MLA 的 KV 低秩潜在维度
--qk_rope_head_dim 32  # MLA 解耦 RoPE 键维度（跨 head 共享）
--use_value_res        # value residual：第一层 V 作为跨层值残差混入后续层
--use_attn_gate        # 注意力输出门（逐 head 输入条件 sigmoid 门，零初始化）
--mtp_depth 1          # MTP 模块深度（默认 1，0 关闭）
--mtp_loss_weight 0.3
--pack_sequences       # 预训练序列打包（消除 padding 浪费）
--doc_mask             # 打包时屏蔽跨文档注意力与边界 loss（需配合 --pack_sequences）
```

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

- SFT / DPO 会自动从基座 checkpoint 的 sidecar JSON 继承模型结构配置
  （含 YaRN/rope_scaling），CLI 显式传入的结构参数优先；`save_interval`
  会自动对齐到 `accumulation_steps` 的整数倍，避免 resume 丢失梯度。

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
