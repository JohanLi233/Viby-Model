# 开发与检查入口

从仓库根目录执行命令。日常协作先读 [AGENTS.md](../AGENTS.md)，
架构与研究导航见 [README](../README.md) 和 [研究索引](../research/README.md)。

## 环境

运行模型使用 Apple Silicon / MLX 环境；静态检查只需要 Python 标准库。
Python 约束、运行依赖见 [pyproject.toml](../pyproject.toml)，精确解析版本见
[uv.lock](../uv.lock)。已有 `.venv` 时先检查环境，再决定是否需要同步。

```bash
python3 scripts/check_repo.py doctor

# 新环境，或明确需要恢复锁定依赖时执行；会同步 .venv
uv sync --locked --group dev

# 模型命令显式使用项目解释器
.venv/bin/python trainer/train_pretrain.py --help
```

`doctor` 只读 Python / 已安装包版本与 Git 摘要，不导入 MLX、不初始化 GPU、
不下载模型、不修改依赖。它报告的是已安装版本，不代表运行验证。
检查脚本在根目录存在 `.venv/bin/python` 时使用它；否则使用启动脚本的解释器。
`test --python /path/to/python` 可显式覆盖。

## 一套入口，按任务选择

```bash
# 默认：检查入口文档的本地链接、代码围栏及检查工具语法
python3 scripts/check_repo.py

# 展示分组和每个分组对应的实际测试文件
python3 scripts/check_repo.py list

# 不执行：预览最终解释器、工作目录与 pytest 命令
python3 scripts/check_repo.py test psr --dry-run

# host 使用 NumPy 模拟 MLX，只验证 host dispatch / fallback 契约
python3 scripts/check_repo.py test host

# 检查入口自身的回归；也可用标准库 unittest 独立运行
python3 scripts/check_repo.py test tools

# Astra 修改了某一路径后，显式选择该路径
python3 scripts/check_repo.py test attention
python3 scripts/check_repo.py test recurrent -- --collect-only
python3 scripts/check_repo.py test psr -- -k 'zero' -x

# 任意更窄的 node ID 仍可直接交给 pytest
.venv/bin/python -m pytest tests/test_v41_config.py::test_source_layer_derivation -q

git diff --check
```

`--` 后的参数原样交给 pytest。分组命令返回 pytest 的退出码；缺依赖或文件缺失
会明确失败，不会自动安装依赖或跳过。除 `tools` / `host` 外，分组可能运行 MLX/Metal；
`--collect-only` 也会导入测试模块，不能视为无 GPU 操作。`--dry-run` 才只显示命令。
不要并行运行 GPU 分组；`all` 是显式全量入口，包含训练和较重的测试。

| 改动 | 分组 | 验证重点 |
| --- | --- | --- |
| 配方、CED 层布局 | `config` | 预设、源层推导、非法配置 |
| attention / mask / cache | `attention` | 可见集合、prefill/decode、文档隔离、XSA |
| 自定义 Metal / VJP | `kernels` | 数值和梯度、Indexer、Sinkhorn、稀疏后端 |
| MoE / QB / 路由数据流 | `moe` | 路由、统计、GPU gather/count、QB |
| 受保护 PSR | `psr` | 梯度隔离、训练接口、连续 batch 状态 |
| 循环 CED | `recurrent` | 残差提升、缓存、配置/恢复、真实训练入口 |
| 推理与 DSpark | `engine` | 拒绝修正采样、缓存搬运、PSR 调度 |
| packing / loss mask | `data` | 数据打包、SFT mask |
| TailSFT | `sft` | 序列筛选、梯度、初始损失缓存、恢复契约 |
| 优化器、checkpoint、计时状态 | `training` | 状态恢复、范数、FLOPs 口径、保存 |

文件清单由 [check_repo.py](../scripts/check_repo.py) 的 `TEST_GROUPS` 维护；
新增测试时更新相关分组。全量测试发现仍由 `pytest tests/` 完成。
这些分组是定位入口，并不表示各研究路径已经全部通过。

默认文档检查覆盖根入口、`docs/*.md`、研究索引和当前实验协议。
它检查本地 inline Markdown 链接的目标路径与围栏闭合，不访问外部链接，
不检查 `#anchor`、reference-style 链接或验证代码块中的命令。
历史报告的数字和命令需要按对应快照核实，不由静态检查背书。

## 配置和恢复训练

配置链为：训练 CLI → `setup_training_args` / `apply_preset` →
`build_model_kwargs` → `VibyConfig` → 模型 / 训练器。
SFT/DPO 还会从 checkpoint sidecar 继承结构，不能只看 parser 字段。

模型库的 `VibyConfig()` 与训练入口可有不同默认值。基线和研究变体应显式
写出功能开关，记录最终解析配置。主干、PSR、循环 CED 的使用条件见各自契约。
旧研究记录及 optimizer probe 命令用于溯源；执行前确认参数仍存在。

checkpoint 的结构 sidecar、优化器分组和执行模式是一组恢复条件。改变实验路径
时检查加载逻辑与相应测试，不能从权重 shape 相同推断 optimizer trajectory 相同。

## 性能工作

先读 [当前实验协议](../research/EXPERIMENT_PROTOCOL.md)，再选测量层级：

| 目的 | 入口 | 结果范围 |
| --- | --- | --- |
| 稀疏 VJP 崩溃复现 | [repro_sparse_bwd_crash.py](../experiments/repro_sparse_bwd_crash.py) | 最小前向 / 反向可执行性 |
| A/A 漂移和训练状态 | [bench_kernel_optimizations.py](../experiments/bench_kernel_optimizations.py) | `aa-fb` / `aa-window`；`cfg-dump` 也会构造模型 |
| CSA2 变体 | [bench_csa2_plan.py](../experiments/bench_csa2_plan.py) | `fb` 或含 optimizer/bias 更新的 `window` |
| 推理 | [bench_csa2_inference.py](../experiments/bench_csa2_inference.py) | prefill / 固定前缀 decode，按实际 CLI 选择 |
| 循环 CED | [ced_recurrent_probe.py](../experiments/ced_recurrent_probe.py) | 同模型对照，范围以 probe 的说明为准 |

先读入口的参数和模型构造代码；历史 benchmark 可能固定了当时的实验开关。
`--help` 是查看 CLI，不证明该 benchmark 仍兼容当前模型。完整窗口测量必须
恢复参数、optimizer 的容器与数组、MoE bias 及其他可变状态；参考
[共享计时工具](../experiments/kernel_bench_utils.py) 和
[状态恢复修复记录](../research/MOE_TRAINING_REPAIR.md)。

报告命令、代码状态、配置、原始样本、误差、峰值内存和有效的配对结论。
MFU 同时给出 FLOPs 公式、峰值算力假设及计时范围；逻辑 FLOPs、单 kernel
耗时或机制测试通过，均不能单独建立完整训练收益。
