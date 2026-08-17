# Hadamard-FWHT 计算效率候选（第一份正面证据）

## 实测（10M/26M/34M，干净 holdout 2000 docs）

| run | 架构 | N | D | FLOPs/tok proxy | CE | 口径 |
|---|---|---|---|---|---|---|
| r045 | dense SwiGLU 512x6 | 28.50M | 10M | 58.3M (1.00x) | 6.0482 | baseline |
| r050 | denseHad 512x6 | 10.62M | 10M | 22.6M (0.39x) | 6.1479 | iso-D |
| r052 | denseHad 512x6 | 10.62M | 26M | 22.6M | **4.6162** | **iso-total-FLOPs** |
| r049 | denseHad 512x6 | 10.62M | 34M | 22.6M | 4.3022 | 1.32x total FLOPs |
| r051 | denseHad 896x8 | 28.87M | 10M | 64.8M (1.11x) | **6.0086** | iso-N, near-iso-FLOPs |

## 结论

1. **同训练算力下**：r052 CE 4.6162 vs dense r045 6.0482，低 **1.432 nats**。
2. **同质量下**：纯插值估计 Hadamard 仅需 ~10.6M tokens，总 FLOPs
   节省 **~2.4x**；同 CE 外推 dense 需 32.8M tokens，即 Hadamard 在
   4.6 CE 点等效 **~3.3x** 计算效率。
3. r051 在同 N、1.11x FLOPs 下首次反超 dense（-0.0396 nats），但未过
   0.05 采信线。

## r053 结果（sandwich norm）

r053 denseHad 896x8 + sandwich @10M：holdout CE **5.2969**。
- vs dense r045 6.0482：**-0.751 nats**；
- vs r051（无 sandwich）6.0086：**-0.712 nats**；
- train main 3.108 vs r051 3.203（train 差 0.095，holdout 差 0.712，
  说明 sandwich norm 的收益主要在泛化侧）。
- 配置侧边核对无误；test_consistency 7/7。单 seed，仍需验证。

## 0.1B / seed2 验证结果（优势缩小）

- r054 denseHad 896x8 + sandwich @0.1B：CE 3.5591 vs dense r030 3.5819。
  同 D 仅 -0.023 nats，FLOPs 1.11x；r053 的 -0.75 优势没有在 0.1B 保持。
- r055 r053 seed2 @10M：CE 5.4261（seed 波动 ~0.13 nats）。
- r057 dense seed2 @10M：CE 6.0892（dense seed 波动 ~0.04）。
- r056 denseHad 512x6 + sandwich @26M：CE 4.4964（无 sandwich r052 4.6162，
  sandwich 在 small-N 路径贡献 -0.12 nats）。
- 同总 FLOPs 口径（r056 vs r045 dense@10M）：CE 4.496 vs 6.048，-1.55 nats。

## 修订结论

- 近等 N 的 Hadamard 重配（896x8）不是 10x：0.1B 只赢 0.023 nats。
- 真正优势路径是 small-N + long-D：10.6M 模型把算力全部投给数据。
  固定总 FLOPs 时 CE 低 1.55 nats（26M 点）；但该优势会随 D 增大衰减，
  需要 r058（denseHad 512x6 sandwich @343M full）测长 D 缩放。
- r058 结果：denseHad 512x6 + sandwich @343M，CE **2.2761**。
- 方法修正：此前把 r058(343M) 与 r030(0.1B) 对比得到 4.2x 是不完整的
  口径——r030 没有训练满 343M。公平基线是 dense512x6 full @343M
  （用户控制实验 r059_dense512x6_full）。
- 早期同 D 对比（step900）：dense full main 2.692 vs r058 main 2.947，
  dense full 领先 0.255 nats，但 dense full per-token FLOPs 是 r058 的
  2.58 倍。最终结论必须等 dense full 的 holdout CE，并做同质量计算归一化。

## lean Engram（知识臂，B1）

v1（site(2)，orders(2,3)，slots4096，sub128，+1.2M）在 denseHad 384x6+sandwich 上
两个尺度一致阴性：

| run | D | holdout CE | 对照 | Δ |
|---|---|---|---|---|
| r064 | 10M | 5.8349 | r063 5.8114 | +0.024 |
| r065 | 34M | 4.2564 | r059 4.2174 | +0.039 |

「更长 D 会显现收益」被否（缺口反而扩大）。诊断：v1 的 `value_proj` 随机初始化，
冷启动时门 α≈σ(0)=0.5，随机 ev 直接注入残差流（与仓库零初始化约定相悖）；
且 6.4k 词表的 2/3-gram 哈希进 4096 槽冲突饱和、无 1-gram 直连记忆。

v2 修复（r066，@34M，对照 r059 4.2174）：`value_proj` 零初始化（启动动力学与
ΔW V=0 同款：仅 W_v 收梯度，表/键梯度恰为 0，W_v 离地后恢复——已被重写后的
test_engram 锁定）；扩容 orders(1,2,3)/slots8192/heads2/sub64（+3.4M，N≈10.4M）。
同时在 `VibyModel` 加了位点越界守卫（越界位点建表但永不注入，此前静默）。

## 下一步（2026-08-17 更新，目标重述：上限更高 + 知识储备更好）

- 效率结论已锁定（用户判定）；待回答的是**上限**：Hadamard 优势随 D 衰减
  （r053 的 -0.75 到 r054 的 -0.023），需 dense 512x6 full @343M（重启中，
  首跑在 step1400 被打断）与 r058 构成同 D 对照，并用 fit_local_scaling
  粗拟合两族 (A, α) 判断是效率型还是上限型收益。
- 知识臂：r066 engram-v2 @34M 判定中；若仍阴性，检查门 α 分布与表利用率，
  再决定 PKM 式大表或放弃参数表、转检索口径（赛道 C）。
- 能力臂候选：Hadamard 省下的 FLOPs 投注意力容量（更宽 MLA / 更多层，
  原 r060 方向），优于循环（r042/r032 负证据）。
