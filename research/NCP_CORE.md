# Viby NCP-Core experiment

2026-09-11。仅移植 NCP 核心机制，不叠加 PSR，不移植额外的 IRC/CRC，也不声称完整复现 OLMo。

## 依据与固定口径

来源：[NCP-ArchPreview 论文](https://arxiv.org/abs/2609.10715) §2.2–2.4、§3；
[官方公开模型代码](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage1_Step500000/blob/155259f2a1ccfd56e449e323360686aad5244d1c/modeling_ncp_olmo3.py)，
固定 revision `155259f2a1ccfd56e449e323360686aad5244d1c`。只下载并审阅源码/配置，没有执行远程代码或下载模型权重。

实现 `Token Encoder → 四 token 均值池化 → 因果 Concept Module → PQ codebook 混合预测 → Decoder`。
Viby 的 CED、mHC、MoE、CSA2 和 tokenizer 保留。概念支路在 CED 边界进入 Decoder，作为可微残差，
其后 token KV 会受该支路影响；这是联合架构，不沿用 PSR 的主干轨迹隔离保证。

| 项目 | 默认 |
|---|---|
| Token Encoder / Concept Module / Token Decoder | 6 / 3 / 6 |
| Concept 压缩率 | 4 |
| Concept 宽度 / attention heads / FFN | 1024 / 8 / 2752 |
| PQ codebooks | 8 × 128 entries，entry dim=128 |
| Decoder 融合初始系数 | 0.1 |
| NCP / VQ loss 权重 | 1 / 1 |
| 预测 codeword 混合 | softmax |
| MSE 归约 | 每维均值 |

两处口径明确保留：论文公式使用 softmax，公开 checkpoint 配置和推理代码却使用 `raw_logits`。
本次默认选论文的 softmax，可用 `--ncp_merge raw_logits` 做显式对照。公开训练代码仍未提供，
无法核对真实 MSE 的 reduction；本地默认 `mean`（避免 loss 随宽度线性放大），可用
`--ncp_loss_reduction l2` 对应论文写出的向量平方范数归约。不能把这两个不同口径混称为严格复现。

Concept Block 采用密集因果注意力、QK RMSNorm、post-attention/post-FFN RMSNorm、SwiGLU。
池化后使用 LayerNorm，参考公开实现的 `concept_vq_input_norm`。mHC 融合按实际 pre_mix 权重和
提升到多流，而不是直接将流均值误当成边界状态。没有额外停止策略或 ranking loss。

## 梯度和因果契约

- NTP 更新完整 token/concept 路径及 codebook。
- NCP 以预测 concept 对下一连续 concept 的 MSE 训练；未来目标 stop-gradient，历史 Encoder 表示仍有梯度。
- VQ 以最近 codeword 拟合 detached concept；这个 loss 只更新 codebook，不添加 commitment loss。
- 分组预测使用 `(t+1)//4 - 1` 对齐：第一个 concept 最早影响 logit 3，它预测 token 4。
- 跨 packed 文档或含 padding 的池化组无效；concept attention、反馈与相邻 NCP 目标均受文档 mask 限制。
- 部分尾组不池化；已有预测可以在同文档的尾部继续使用。短于 4 的序列没有 concept 反馈。
- Codebook 作为 embedding 表进入 AdamW，避免 MuonH 固定半径限制质心移动。其余矩阵沿用当前 Viby 优化器分组。

现阶段生成采用全前缀重算参考路径，已验证 prefill/decode 与完整前向的因果一致性。
它可用于正确性检查，不能宣称缓存加速。连续 batch engine 明确拒绝 NCP，避免错误复用普通 token cache。

## 运行

普通预训练现在默认 `ncp_enabled=True, psr_enabled=False`。`--no-ncp` 为 token baseline；
PSR 独立实验需 `--no-ncp --psr`。不能复用 PSR 的 optimizer/checkpoint；NCP 采用独立输出目录。

```bash
uv run trainer/train_pretrain.py \
  --data_path /Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl \
  --out_dir research_runs/viby_ncp_core \
  --pack_sequences --doc_mask \
  --batch_size 16 --accumulation_steps 2 --max_seq_len 1024 \
  --log_interval 1 --seed 1337 --use_swanlab --auto_resume
```

`main_loss` 是完整 NCP 模型的纯 NTP CE；总 loss 另加 NCP/VQ。SwanLab 的 `ncp/*` 和本地
`ncp_metrics.jsonl` 分别记录 NCP/VQ、反馈覆盖率、有效 concept/pair 数、codebook 使用率与表征均方。
`ncp_start_*.json` 保存实际解析配置与 auto-LR/token budget。后台命令、PID、日志路径写入 `launch.json`。

未来对照必须用同一 mini 语料、packing、seed、token budget、LR 日程。不要拿本次 mini run
直接与以前完整语料的 `viby41` 曲线作因果判断；也要报告新增参数、训练时间与资源成本。

## 本次检查

核心和相关回归共 113 项通过；原 pretrain 入口的 BF16/Muon 小配置完成两次累积更新，loss/梯度有限。
检查包括因果 shift、未来替换、跨文档/PAD、VQ 梯度隔离、NCP 历史梯度、两种混合口径、编译更新、
codebook 优化器分组、权重保存重载、全前缀参考生成及旧主干/PSR 回归。这些不是 NCP 效果结论。

切换前的只读审计保存在 `research_runs/ncp_preflight/audit.json`：PSR2 最后 500 个训练微步的
整体修正 CE 没有改善，直接覆盖仅约 3%。同时，PSR2 使用 mini 数据与约 0.001717 的基础 LR，
viby41 使用完整数据与约 0.000890 的基础 LR。因此原曲线不是严格配对对照；该结论不将截图
颜色映射成已确认的 run，也不把训练 telemetry 当成独立验证集结果。
