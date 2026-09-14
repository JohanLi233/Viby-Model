# Viby 仓库协作指引

适用于在此仓库工作的 Astra / Codex。目标是根据当前源码完成可复核的修改，
保留研究证据，并让下一次工作能从明确的状态继续。

## 开始任务

1. 读取用户目标，查看 `git status --short` 和任务涉及文件的 diff。
   工作区可能有正在进行的研究；未提交及未跟踪文件也是当前实现的一部分。
   对已有文件做局部修改，写入前重新读取；保留其他任务的改动。
2. 从 [README](README.md) 了解主干，从 [研究索引](research/README.md)
   选择本任务所需的契约和证据。按问题读取源码，避免先遍历日志、权重和全部历史文档。
3. 用 `python3 scripts/check_repo.py doctor` 查看环境，用
   `python3 scripts/check_repo.py list` 找到对应检查入口。
4. 先追踪配置到实际调用路径，再修改。方案、源码接通、测试通过、性能或质量收益
   是四种不同证据；报告中分别说明。

## 源码入口

| 问题 | 先读 | 随后追踪 |
| --- | --- | --- |
| 模型字段、层布局、默认值 | [model/config.py](model/config.py) | [model/model.py](model/model.py)、[model/block.py](model/block.py) |
| 训练 CLI 和预设 | [trainer/config.py](trainer/config.py) | [trainer/utils.py](trainer/utils.py) 的 `build_model_kwargs` / sidecar 加载，再到训练入口 |
| loss、累积更新和路由统计 | [trainer/base_trainer.py](trainer/base_trainer.py) | [model/moe.py](model/moe.py)、[trainer/muon.py](trainer/muon.py) |
| CSA2 / CED / 稀疏选择 | [model/attention.py](model/attention.py) | [model/kernels](model/kernels)、[model/cache.py](model/cache.py) |
| 循环 CED | [model/recurrent.py](model/recurrent.py) | 模型前向、训练器、研究契约 |
| 推理、请求排队、缓存 | [engine/engine.py](engine/engine.py) | [engine/memory.py](engine/memory.py)、[engine/prefix.py](engine/prefix.py)、模型 prefill/decode |
| packing、文档隔离 | [dataset/lm_dataset.py](dataset/lm_dataset.py) | 训练 loss mask、attention 的 segment / pad mask |
| 吞吐 / MFU | [experiments/kernel_bench_utils.py](experiments/kernel_bench_utils.py) | 实际 benchmark 的配置与 [trainer/flops.py](trainer/flops.py) |

`VibyConfig()`、预训练 parser、SFT/DPO sidecar 的默认来源不同。先核对当前
`get_pretrain_parser()`、`apply_preset()`、`build_model_kwargs()` 及实际命令；
历史文档里的“默认”不能替代这条链。恢复训练时还要核对结构与优化器状态。

## 修改时保持的契约

- CED 的共享 evidence、压缩组完成时刻、原 token RoPE 位置、同文档/PAD 可见性
  必须沿完整调用路径保持一致。优化 Top-K 时保持集合及 tie 规则。
- mHC 的 `pre_mix` 传递、MoE 选择偏置与路由权重的区别、每次物理调用的统计、
  累积窗口更新边界均属于模型语义。改变它们应作为显式研究变体。
- 循环 CED 的共享参数 / 分轮缓存契约见研究文档。
  前向可用不等于 continuous-batch engine 支持相同状态。
- MLX custom VJP 保持每个 array primal 对应一个 array 梯度叶子；检查 metadata
  占位、shape、dtype 和 cotangent 布局。不要用 `None` 造成槽位错位。
- 编译与 eager、prefill 与 decode、FP32 与 BF16 的路径可能不同。检查目标路径
  的 dispatch 条件和 fallback；测试通过必须能说明实际走了哪个实现。
- 不把近似计算、改变读取预算、损失、精度或优化器配方产生的差异称作等价 kernel 优化。

## 验证和交付

文档与工具操作见 [开发指南](docs/DEVELOPMENT.md)，研究操作见
[实验协议](research/EXPERIMENT_PROTOCOL.md)。纯文档修改先跑默认静态检查；
实现改动选择对应测试文件或 node ID，默认不扩大到全仓测试、长训练或性能实验。
用户指定的测试范围优先；未执行的验证明确列出。

MLX GPU 训练、测试和计时串行执行，运行前检查已有任务。性能比较固定权重、
输入、配置、dtype、编译模式、缓存与优化器状态，先完成同步再计时。
保持 reference 路径，区分单算子、fwd+bwd、完整累积窗口和 decode 的结果。

结束时报告：改了什么、为何修改、实际执行的检查及结果、尚未建立的结论。
研究任务附运行目录、配置与原始样本位置。修正文档中的过期结论时保留历史证据，
用日期和适用范围解释变化；旧交接文档中的任务指令只属于当时任务。

`autoresearch-mlx/` 是独立子项目；进入后先检查其本地约定和依赖。
`research_runs/`、`swanlog/`、`.cache/` 和模型权重按任务定向读取，勿作为整理对象清理。
