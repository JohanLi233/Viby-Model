# 实验运行协议（agent 交接用）

本文档是 optimizer/spectral-knee 研究弧线的实验操作手册。**每个已跑 probe 的完整参数
存在 `research_runs/probe_*/pretrain_768.json` 的 `args` 字段里，复现以它为准**；
本文只讲通用协议、命令模板和踩过的坑。

## 环境

- 一切用项目 venv：`.venv/bin/python ...`（不要装新包进 .venv）。
- ruff 在 `/Users/lizhonghan/.local/bin/ruff`。
- matplotlib 不在 venv 里；绘图用临时 env：
  `uv run --no-project --with matplotlib python experiments/paper_figs.py`
- 不 git commit / 不做任何 git 变更。
- **同一时刻只跑一个训练 probe**（单机 MLX，并行会互相污染时序和吞吐）。

## 标准 probe 协议（MoE 768 档）

基线配置（= probe_p30_latesnap/pretrain_768.json，也是论文新代码口径基线）：

```bash
.venv/bin/python trainer/train_pretrain.py \
  --out_dir research_runs/probe_pXX_<name> \
  --epochs 1 --batch_size 12 --accumulation_steps 2 \
  --learning_rate 0.0015450949123618698 \
  --device mlx --dtype bfloat16 --compile_model --no_swanlab \
  --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 \
  --vocab_size 6400 --max_seq_len 1024 \
  --pack_sequences --doc_mask \
  --max_steps 2000 --max_train_minutes 55 --log_interval 500 \
  --save_interval 100000 --min_lr_ratio 0.05 \
  --seed 1337 \
  --use_attn_gate --mtp_depth 1 --mtp_loss_weight 0.3 --z_loss_weight 0.0001 \
  --optimizer muon --muonh \
  --n_routed_experts 256 --num_experts_per_tok 8 --n_shared_experts 2 \
  --moe_intermediate_size 384 --routed_scaling_factor 2.5 \
  --moe_router_logit_norm \
  --data_path /Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl \
  > research_runs/probe_pXX_<name>/console.log 2>&1
```

要点：

- lr / muon_lr / beta2 / eps 由 `--lr_scale_auto`（默认开）公式推导，日志第二行
  `[lr_scale] 公式自动推导:` 会打印；上面显式给的 `--learning_rate` 是之前推出的
  值，保持所有 probe 同口径。
- warmup_iters 自动 = 总步数 1%（日志里 279）。
- **指标读法**：console.log 的 `Epoch:[1/1](N/...)` 中 **N 是微步**
  （accumulation_steps=2，优化器步 = N/2）。论文报的 loss@500 是微步 500
  那一行：`Epoch:[1/1](500/27863) loss:5.264(...)`。
- 500 微步短 probe ≈ 14 分钟（`--max_steps 550`，微批口径；**不要设 500**——
  会在第 500 微步的日志行打印前停下，丢失 loss@500；warmup 279 由 epoch
  总长推出，不受 max_steps 影响）；2000 优化器步全程 ≈ 45–55 分钟
  （--max_steps 2000 + max_train_minutes 55 兜底）。

### 变体开关（env 变量，均定义在 trainer/muon.py）

| env | 默认 | 作用 |
|---|---|---|
| `VIBY_MUONH_NS_COEFF` | `cubic5b05` | NS 迭代系数/膝盖：`classic`/`cubic5`/`cubic5b05`（b05=knee50 0.05）/`cubic5b002`/`cubic5b10` 等 |
| `VIBY_MUONH_PER_HEAD` | 0（关） | 1 = 按 head 切 NS（P21b 500 步 −0.08，P29a 2000 步中性；非默认） |
| `VIBY_MUONH_EDGE` | 0（关） | 1 = EdgeCubic 逐形状膝盖（P29b 2000 步确认有害：+0.18 loss / +0.47 mtp，勿开） |
| `VIBY_MUONH_EXPERTS` | 1 | 0 = 专家不进 MuonH（消融用） |
| `VIBY_MUONH_NO_NS` | 0 | 1 = 跳过谱均衡（消融用） |
| `VIBY_SNAPSHOT_STEPS` | "" | 逗号分隔的**优化器步**列表，dump 动量/梯度快照（见下） |

例：`VIBY_MUONH_NS_COEFF=cubic5b002 .venv/bin/python trainer/train_pretrain.py ...`

稠密对照档见 `research_runs/probe_p26a_dense_b05/pretrain_768.json`（去 MoE/MTP，
loss 行只有 main/z 两项）。**dense 档必须显式 `--moe_latent_dim 0`**——缺省
None 会解析成 hidden/2=384，把「dense」变成带 latent 瓶颈的 96M 模型
（P34/P35 曾因此报废重跑；参数量 132.025M 才是对的）。

## 动量快照 probe（σ* 测量）

```bash
VIBY_SNAPSHOT_STEPS=150,300,500,750,950 \
  .venv/bin/python trainer/train_pretrain.py --out_dir research_runs/probe_pYY_snap ...（同上）
```

**坑：`VIBY_SNAPSHOT_STEPS` 是优化器步口径，不是微步**（曾因此返工一次）。
产物：`mom_*_step{k}.npz` + `grad_mb{j}_step{k}.npz`。

离线分析（MP 拟合、λ̂₊、方向相关坍缩、各向异性）：

```bash
.venv/bin/python experiments/mp_fit.py research_runs/probe_pYY_snap [step ...]
```

理论口径见 `research/SPECTRAL_THEORY.md`。快照 npz 是论文原始证据，**勿删**
（model 权重 *.safetensors 已清，npz 保留）。

## 执行纪律（踩过的坑）

1. **后台 Bash 任务必须显式设 `timeout`**：默认 600s 杀过一个 45 分钟的 run。
   500 微步 probe 给 ≥1800s；全程 run 给 ≥3600s。
2. 同一时刻只跑一个训练任务。
3. probe 命名：`probe_p<编号>_<短名>`，编号递增不重用。
4. 新旧代码口径**不可跨减**：gate_up 零初始化 bug 修复（P19 起）使基线移动 ~0.04，
   P13–P17 是旧口径，P19+ 是新口径。对比只在同口径内进行。
5. 跨 seed 噪声：σ≈0.11（P25 实测），配对差 ±0.05 以内算中性；小于此幅度
   不要下结论。
6. 改优化器代码后跑四个测试：
   `.venv/bin/python -m pytest test_align.py test_consistency.py test_kda.py test_muonh_zeroinit.py`
7. 改完代码核对注释/docstring 是否还描述旧行为（dt_bias 注释曾与实际 init 差 10×）。
8. **发散处理**（P40 教训）：loss 先 spike 后 nan 且不再恢复 = 参数已污染，
   该 run 作废但须如实记录（nan guard 只跳过当窗更新，挡不住已污染的参数）。
   先查同 seed 配对 run 在相同微步是否也 spike——配对数据顺序一致，对照 run
   平稳通过则 spike 是排程特异而非数据批问题（P40 dense b002 @2k：step~600
   spike 8.36、~1000 起 nan 不愈；配对 b05 同批次无任何波动）。

## 关键基线数字（新代码口径，seed 1337，loss@500 微步）

- cubic5b05（现默认，P19）：**5.264**
- cubic5b002（P20）：+0.054（左臂，过保守）
- cubic5b10（P17，旧口径）：右臂 +0.091 → U 型，谷底 ≈ knee50 0.01~0.05
- per-head / edge / per-head+edge（2000 步，P29）：edge 是元凶（+0.18 loss /
  +0.47 mtp），per-head 单独中性
- 稠密 132M（P26/P34/P35）：σ* 比 MoE 低一个量级；500 步平台
  b002 3.943 ≈ b005 3.950 ≈ b01 3.963 < b10 3.974 < b05 3.997。
  **2000 步翻盘（P40–42）**：b002 step~600 spike、~1000 步发散 nan
  （排程特异，对照同数据无波动）；b005 2.977 差于 b05 2.905——
  warmup 档低估膝盖，稳态最优回到 knee50≈0.01（与 MoE 默认同值）。

完整结果表：`research/OPTIMIZER_RESEARCH.md`；论文：`research/PAPER.md`；
论文图重生成：`uv run --no-project --with matplotlib python experiments/paper_figs.py`。
