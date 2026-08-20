# Viby

基于 Apple MLX 的单设备中文大语言模型训练与推理项目。

架构为 decoder-only Transformer：HRM 双栈循环（L/H 两个 stack 交替迭代，
唯一架构）+ MLA（Multi-head Latent Attention，
DeepSeek V2/V3 风格）+ RoPE（解耦位置键）+ RMSNorm + SwiGLU + QK-norm，
FFN 全部为 MoE（DeepSeekMoE，V3/V4 风格）；可选 value residual、
注意力输出门（attn gate）、cycle 条件路由 / CycleFiLM、engram n-gram 记忆，
以及 MTP（multi-token prediction，默认开启 depth=1，推理侧可用于投机解码加速）。

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
# HRM（唯一架构，必选）：
--hrm_H_cycles 2       # 高层（H 栈）循环次数，必须 >0
--hrm_L_cycles 2       # 每个 H cycle 内低层（L 栈）循环次数，必须 >0
--hrm_cycle_router 1   # cycle 条件化 MoE 路由（可选，0 关闭）
--hrm_cycle_film 1     # cycle 对隐状态做 FiLM 调制（可选，0 关闭）
# DeepSeekMoE（V3/V4 风格，所有层 FFN 均为 MoE）：
--n_routed_experts 32      # 路由专家数（必须 >0）
--num_experts_per_tok 6    # 每 token 激活专家数（V4 为 6）
--n_shared_experts 1       # 共享专家数（中间维 = moe_intermediate_size × 该值）
--moe_intermediate_size 104  # 单个路由专家中间维（默认取 intermediate_size）
--routed_scaling_factor 2.5  # sigmoid 归一化后的路由权重缩放
--moe_bias_update_rate 0.001 # 无辅助损失负载均衡的偏置更新步长（<=0 关闭）
```

注：MoE 的路由专家前向是分组稀疏桶 GEMM（每 token 只算 top-k 个专家），
桶容量取自上一微批实测的滚动容量表（×1.25+64 余量、128 对齐），前向
**无 host sync**——这是硬要求：sync 会迫使前向在 value_and_grad 建图期
（vjp 尚未构建）分段执行，全部中间张量被未完成惰性图钉住（r073 实测
46GB/步、48GB 机器 swap 塌缩；改后峰值 17.6GB、单步 3.3s→1.3s）。
容量表逐微批变化 ⇒ 形状跨步不稳定，训练时 `mx.compile` 仍自动回退
eager。推理（generate/eval）本来就是 eager，无差异。

训练侧优化（默认启用）：
- BatchedMuon：同形状权重堆叠批量跑 Newton-Schulz（kernel 数从
  ~16×张量数 降到 ~16×形状组数），数学上与逐张量 NS 等价（bf16 舍入
  级差异）；`--muon_ns_steps` 可调迭代步数（默认 5，对齐原版）。
- `--cache_limit_gb`（默认 24，0=不限）：Metal 分配器空闲块缓存上限。
  上限内的释放块常驻复用、不归还 OS，避免每步"释放-重分配"抖动
  （bs16x640 实测 10G→24G 提速 4.5%，峰值 14.8G + 缓存 ≈ 39G）。
  大 batch 配置注意峰值+缓存上限不要超物理内存。
- MoE 专家 gate/up 投影合并为单次 GEMM（数学严格等价）；稀疏桶路径按
  8 专家分组（峰值/耗时与全局最大桶解耦，路由塌缩从 swap 悬崖退化为
  热点组变慢的可恢复劣化），argsort 排名 + 全 device 散布/收回，
  输出端 f32 scatter-add 直接累加回 token（省 padded 加权缓冲与
  gather+sumK 往返）；桶容量由滚动容量表给出，前向无 host sync。
- MLA 投影合并：q/kv_down/k_rope 单 GEMM（768→1248）、k_up/v_up 单
  GEMM（192→1536），数学严格等价；Muon 侧按行段分段 NS 保持逐矩阵
  语义（分段 vs 逐矩阵基类单步 diff ~2e-7）。
- engram 注入门/掩码链降回残差 dtype 再乘加：修复残差流从注入层起
  被抬成 f32 的泄漏（曾使整步慢 18%、峰值内存 +3G）。
100M MoE 配置（bs8×512, bf16, M4 Max）实测：fwd+bwd 243→199ms/步
（16.8K→20.6K tok/s）；含 optimizer + 真实数据管线完整步
11.5K→15.3K tok/s（本轮累计 +87%），已反超 dense-73M 的 13.8K
（MoE 激活参数仅其 46%）。

MoE 推理前向按 (token,choice) 对数 G = B×T×top_k 分三条路径：
- `G <= 512`（decode/极小批量）：手写融合 Metal kernel（router 打分+top-k、
  SwiGLU、加权合并共 3 个 kernel，只读 top-k 命中专家权重；simdgroup
  合并访存 + simd_sum 归约）。不可微，仅推理（`model.train()` 或
  `router.collect_stats` 开启时不走本路径）；支持 CycleDeltaRouter
  低秩增量（host 预算 cz=(x·V^T⊙g_c)，kernel 内加 cu·cz；CycleDelta
  形式仍旁路稠密路径）；JIT 编译失败自动回退。
  实测 100M 配置 decode 503→622 tok/s（+24%）。
- `G <= 4096`（小 prefill）：稠密全专家广播 matmul。
- 更大（训练/大 prefill）：分组 (Eg,Cg,D) padded 桶稀疏 batched GEMM
  （无 host sync；桶容量滚动自上一微批实测，见上文训练侧说明）。

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
