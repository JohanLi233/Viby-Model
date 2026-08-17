# Cactus Needle / SAN 评估记录

## 关键澄清（paper vs Needle 2 代码）

SAN 论文（arXiv:2607.18363）的组件消融：
- QK-norm 是唯一不可移除项（移除即发散）；
- post-attention sandwich norm 是唯一正收益变体（-0.009 nats）；
- 标量残差门可有可无；
- depth 在 iso-param 下呈 U 形，20 层为最优点，48 层仍可训练；
- 论文不含 HadamardMLP / Engram / MHC——这三个是 Needle 2 工程实现，
  没有进入该对照实验。

## 来源
- 论文：A Controlled Study of Attention-Only Transformers, arXiv:2607.18363
- 代码：github.com/cactus-compute/needle（Needle 2，45M 端侧工具调用模型）
- 论文核心结论：
  - 同参数下纯注意力 SAN vs FFN transformer：+0.006 nats（几乎打平）
  - 同 FLOPs：FFN 领先 +0.263 nats
  - 同深度：FFN 领先 +0.470 nats
  - 差距集中在低上下文 / 权重知识召回；QK-norm 是深层注意力栈可训练的关键
- Needle 2 的 5-70x 是参数/内存/量化/端侧口径，不是训练 FLOPs 口径。

## 本地 10M 筛查（同一数据切片）
| run | 架构 | N | holdout CE |
|---|---|---|---|
| r045 | dense 512x6 | 28.50M | 6.0482 |
| r046 | SAN pure h384 L44 | 28.45M | 6.1485 (+0.100) |
| r047 | SAN + HadamardMLP h384 L44 | 28.52M | 6.0613（loose 评估） |
| r048 | SAN + Engram h384 L40 sites(2,20) | ~28.58M | 中止：内存 35.3G/峰值47.1G，step200 后 stall（main 3.850 vs SAN-pure 3.847） |

结论更新（00:04）：
- 纯 SAN CE 6.1485，dense 6.0482：SAN 路线关闭。
- SAN+Hadamard CE 6.0613：Hadamard 把 SAN 缺口从 +0.100 收窄到 +0.013 nats，
  且 HadamardMLP 已升级为 O(n log n) FWHT、每层仅 3n 参数。这是目前最有价值的
  计算效率组件，r049/r050 将直接测 dense+Hadamard 的 FLOPs-normalized 表现。
- Engram 训练被内存压力中断（35.3G 常驻/47.1G 峰值，48G 机器），step200
  main 3.850 vs SAN-pure 3.847；该路线在 M4 上暂不可测，不是质量判负。
- compute_compare.py 的 FLOPs 代理（ctx=640）：
  dense512x6=1.00x；SAN=1.63x；denseHad512x6=0.39x；denseHad896x8=1.11x。
  因此 r052 用 26M tokens 做严格 iso-total-FLOPs。
- r049 首果：denseHad 512x6（N=10.62M）@34M tokens，holdout CE 4.3022。
  对比 r045 dense（N=28.5M）@10M CE 6.0482。注意 r049 总 FLOPs 仍为
  1.32x（34*22.6 vs 10*58.3），严格等算力看 r052。
值得保留并独立测试的组件：
1. Engram（n-gram 哈希键值记忆）——攻击 B1 知识容量，r048 判定。
2. HadamardMLP（固定 Walsh-Hadamard + 对角缩放）——r047 判定其能否
   以极低参数恢复 SAN 的逐 token 非线性。
3. ZCN + 标量残差门 + o_proj 深度初始化——训练 recipe，已并入代码。
