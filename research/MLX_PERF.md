# MLX / Apple Silicon 性能优化手册

本项目在 M4 Max 单机上把 1080M MoE 模型训到可用吞吐，过程中积累的
优化经验、方法论与**踩过的坑**都记在这里。目标读者是「下一次要继续
优化这个仓库的人」（很可能就是三个月后的自己）。

**怎么用这份文档**：

- 要新做一次优化 → 读 §1（工作流）+ §2（测量口径），照 §8 的 checklist 走。
- 怀疑某个写法慢 → 查 §4（MLX 算子清单）和 §7（失败清单），大概率已经试过。
- 要写/改手写 kernel → 读 §5，特别是 §5.8（prewarm，最容易翻车的地方）。
- 想知道「还有多少优化空间」→ 读 §2.3（roofline）+ §0（硬件常量）。

所有数字都标注了口径（形状、eager/compile、fwd 还是 fwd+bwd）。**换口径
数字就不可比**，这是本文档最重要的纪律。

---

## 0. 硬件基线常量（M4 Max）

优化判断全部基于这几个数，写死在脑子里：

| 常量 | 值 | 备注 |
|------|-----|------|
| 内存带宽（实测可达） | **400 GB/s** | 理论 546，roofline 一律按 400 算 |
| threadgroup 内存硬上限 | **32 KB** | 实测 32KB 通过、48KB 编译/启动失败 |
| simdgroup MMA 峰值（真实形态） | **~12 TFLOPS** | A 在寄存器 + B 从 threadgroup 载入 |
| simdgroup MMA（转置载入） | **~11.4 TFLOPS** | flash 算 QKᵀ 的形态 |
| simdgroup MMA（两操作数都在寄存器） | **1.3 TFLOPS** | 反直觉，见 §5.5 |
| bf16 稠密 GEMM 峰值（经验） | **12.9 ~ 14 TFLOPS** | 大而方的 batched GEMM |
| threadgroup 内存 bank | 32 个 × 4 字节 | bank conflict 规避见 §5.4 |

复现命令（换硬件后**第一件事**就是重跑这几个，刷新上表）：

```bash
.venv/bin/python experiments/probe_simdgroup.py   # 类型支持 / 转置载入 / TG 上限
.venv/bin/python experiments/probe_mma_peak.py    # MMA 三种操作数形态的 TFLOPS
.venv/bin/python experiments/probe_mma_operand.py # reg/TG/device 放置矩阵
```

2026-08 本机实测输出（`probe_mma_peak.py`，每臂 68.7 GFLOP）：

```
两操作数都在寄存器          52.17ms     1.3 TFLOPS
B 每次从 threadgroup 载入    5.69ms    12.1 TFLOPS
B 从 threadgroup 转置载入    6.01ms    11.4 TFLOPS
```

`probe_mma_operand.py` 的放置矩阵（19.3 GFLOP/臂）：A×B 都在 threadgroup
最快（11.0），**两个都从 device 直读只有 7.2** —— 这解释了为什么 flash
反向早期版本卡在 3.2 TFLOPS（dP 那步两个操作数都在 device）。

---

## 1. 优化工作流（SOP）

不要凭直觉改代码。本项目每一次有效优化都走完了这五步，跳步的那几次
全部返工（见 §7）。

```
① 定位   bench_train_step 看整步分段占比，挑最大项
② 归因   prof_* 前缀差分，把大项拆到「段」级别，找出 bwd/fwd 异常的段
③ 判定   算 roofline（带宽下界 / MMA 峰值），确认是 memory-bound 还是
         compute-bound，以及理论上还能快几倍 —— 空间 < 1.5× 就别做了
④ 实现   先在 experiments/ 写 probe 验证假设，再动 model/
⑤ 验收   verify_* 数值对拍 + 检查没有静默回退 + ab_* 同进程 A/B 计时
```

对应的脚本生态（`experiments/` 下 80+ 个脚本，按前缀分工）：

| 前缀 | 职责 | 代表 |
|------|------|------|
| `bench_*` | 端到端 / 组件吞吐基准 | `bench_train_step.py`（整步分段）、`bench_components.py` |
| `probe_*` | 验证一个具体假设、二分定位 | `probe_hotspots.py`、`probe_mma_peak.py` |
| `prof_*` | compile 口径下的分段归因 | `prof_kda_stages.py`、`prof_bwd_attrib.py` |
| `sweep_*` | tile / 几何参数网格搜索 | `sweep_flash_tile.py`、`sweep_kda_scan.py` |
| `verify_*` | 数值对拍 + 静默回退检测 | `verify_prewarm.py`、`verify_flash_bwd.py` |
| `ab_*` | 同进程交替 A/B，抵消热漂移 | `ab_round_opt.py`、`ab_attn_compile.py` |

端到端质量门禁另有一条线：`experiments/run_exp.sh <round> <notes> [args]`
→ 训练 + holdout PPL → 追加一行到 `research/experiments.tsv`。**性能优化
如果可能动到数值，必须跑一轮 run_exp 确认 holdout CE 不退化。**

---

## 2. 测量方法论

### 2.1 MLX 是惰性求值：只有 `mx.eval()` 之后的墙钟才算数

MLX 的 op 只构图不执行。`t1 - t0` 包住一堆 op 却不 eval，测到的是
**构图时间**，通常快得离谱且毫无意义。

全项目统一的计时 helper（照抄即可，`experiments/probe_hotspots.py:19`）：

```python
def timed(fn, iters=10, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)
```

要点：

- 用 `time.perf_counter()`，不用 `time.time()`。
- 本项目**不用 `mx.synchronize()`**，`mx.eval()` 已经隐含等 GPU 算完。
- `warm` 至少 2~3 轮：吃掉 Metal JIT 编译、`mx.compile` 首次 trace、
  融合 kernel 首次校验。
- 分阶段计时就分别 eval：`mx.eval(loss)` 再 `mx.eval(grads)` 能把前向
  和反向切开（也顺带省内存，见 §3.2）。

### 2.2 min / median / 均值：分场景选

| 场景 | 聚合 | 理由 |
|------|------|------|
| 短 kernel、微组件 | **min** | 乐观但方差最小，本项目默认 |
| 跨分钟的 A/B | **median，丢弃前 2 轮** | 抗热漂移 |
| 整步训练步 | **中位 + min + 周期均值三个都报** | 历史上 NS 降频刷新步有尖峰，只看中位会漏掉摊平成本（机制已删，口径保留） |

**热漂移是本机最大的测量陷阱**：跨进程比对可以差 **±30%**，连续跑之后
GPU 降频能让同一段代码慢 **~20%**。所以：

- **禁止**「改代码前跑一次、改完再跑一次」这种跨进程对比。
- A/B 必须**同进程交替轮转**（`ab_moe_fwd.py`：7 轮交替、跳过前 2、取
  median；`ab_attn_compile.py`：9 轮）。
- 必须跨进程时用对称顺序 + 冷却：`ab_round_opt.py` 跑 **ON,OFF,OFF,ON**
  并在中间 sleep 60s。

### 2.3 roofline：先算上界，再决定要不要动手

**memory-bound 下界**（本项目到处在用）：

```python
BW = 400e9  # M4 Max 实测可达带宽
# 逐元素 kernel：数清 fwd/bwd 各要读写多少字节
print(f"fwd 下界 {el * 4 / BW * 1e3:.2f}ms")        # 读 x + 写 y
print(f"bwd 下界 {el * 14 / BW * 1e3:.2f}ms")       # x+dy+dx+dz 往返
```

**compute-bound 上界**：`TFLOPS = FLOPs / time`，再跟 §0 的 12 TFLOPS
（MMA）或 13 TFLOPS（稠密 GEMM）比。

判定表（本项目实际用它砍掉了好几个方向）：

| 对象 | 实测 | 上界 | 倍数 | 结论 |
|------|------|------|------|------|
| KDA scan f+b | 15.01 ms | 带宽下界 3.24 ms | **4.6×** | 发射几何/occupancy 问题，值得做 |
| flash bwd（早期） | 3.2 TFLOPS | MMA 12 | **3.8×** | MMA 操作数形态不对，值得做 |
| MoE 整层 | 7.9 TFLOPS | 12.9 | 1.6× | 差距在索引和碎片 GEMM，收益有限 |
| MoE gather_mm 单独 | 12.7~13.1 TF/s（均匀/倾斜/真实路由三者基本同速） | 同 FLOPs 稠密均匀 batched GEMM 13.4 TF/s | **1.05×** | 手写分段 GEMM 不值得做，见下行注 |
| kda_prep fwd | 2.05 ms | 0.63 ms | 3.3× | 融合后到 1.63 ms，接近到顶 |

**gather_mm 判定注**（2026-08-25，`probe_gather_mm_uniform.py`，受控同进程、
compile 对 compile）：sorted gather_mm 在均匀索引、合成倾斜（段长 std≈19）、
真实路由（std≈32）下 fwd 全部 ≈12.7~13.1 TF/s，与同 FLOPs 稠密均匀 batched
GEMM（13.4）差 ≤5%，且差值里还含段尾 tile 量化的内在损耗（任何分段 kernel
都付）。此前「gather_mm 比等价稠密慢 ~4.2ms/层」的印象来自
`probe_moe_parts.py` 的 eager 稠密对照 vs compile gather_mm 的混合口径 +
跨段热漂移；同口径复测不成立。**结论：手写分段 GEMM 的恢复上限 <1% 整步，
不做。**

**经验阈值**：倍数 < 1.5× 就不要动了，改一天赚不回来。

### 2.4 归因：前缀差分法

想知道一条计算链里每段花多少时间，**不要**合成中间张量单独测——
合成输入的 dtype / 连续性 / 数值范围都跟真实链不一样，测出来会骗人。

正确做法（`prof_kda_stages.py`）：**每个 stage 是真实计算链的前缀**，
各自单独 `mx.compile`，相邻两个前缀的差值就是该段成本。

```
stage1 = 投影                     → t1
stage2 = 投影 + conv              → t2    conv 段 = t2 - t1
stage3 = 投影 + conv + 门控       → t3    门控段 = t3 - t2
```

这个方法当年直接抓出两个反直觉的事实：

- KDA 的 3× causal_conv 段 f+b **16.85 ms/层**，而单个 conv 微基准
  只有 1.57 ms（×3 = 4.7 ms）—— 缺口在 compile 图内的 custom_function
  开销，光看微基准永远发现不了。
- 门控段（rms_unit / softplus / 逐元素）f+b **7.21 ms**，其中 bwd 6.47、
  fwd 0.75，**bwd/fwd = 8.6×**。逐元素链的反向异常昂贵是本项目反复
  出现的模式，见 §4.3。

### 2.5 eager 口径 ≠ compile 口径，不可混比

- `probe_breakdown_v2.py` 是 eager 口径，会**高估逐元素链**的占比。
- `prof_bwd_attrib.py` 是 compile 口径（prewarm → compile loss →
  compile value_and_grad），贴近真实训练。
- **归因结论一律以 compile 口径为准**，eager 口径只用来做算法段拆分。

以 compile 口径得到的整步结构（1080M，bs12×1024）：KDA 占整步反向
约 **51%**；MoE 单层 f+b **57.04 ms** × 9 层 ≈ **513 ms/微批**，是最大
单项。

---

## 3. 惰性求值与内存

### 3.1 eval 时机决定内存峰值

梯度累积窗口内如果不 eval，`accumulation_steps` 个微批的前向+反向图
**全部存活**到 optimizer step，内存按窗口线性膨胀。所以每个微批结束
立刻物化。

### 3.2 拆成两次 eval：省 3× 内存、快 2×

这条是本项目单点收益最大的 Python 层优化（`trainer/base_trainer.py:574`）：

```python
# 一次性 mx.eval(loss + grads) 时，MLX 的图调度会把前向 tape 滞留与
# 反向临时量同时顶到峰值（r073 配置实测 46GB / 4.0s）
mx.eval(loss, mtp_loss, moe_stats, ...)   # 先让前向输出落定 ~14.5GB
mx.eval(grads)                            # 再跑反向，峰值 ~16GB
```

**结果：46GB → ~16GB（省 ~3×），单步快 ~2×，数值完全不变。**

### 3.3 Metal 分配器缓存上限

```python
mx.set_cache_limit(int(cache_gb * 1024**3))   # --cache_limit_gb
```

- 上限内的空闲块常驻复用、不归还 OS，避免每步「释放-重分配」抖动。
- 实测 bs16×640：**10G → 24G 提速 4.5%**；该配置峰值 14.8G，峰值+缓存 ≈39G。
- **约束：峰值 + 缓存上限不能超物理内存**，否则 swap 会把吞吐打穿。
- 历史上曾出现 optimizer 临时 buffer 污染 freelist 拖慢激活分配
  （**243 → 442 ms/步**）；BatchedMuon 批量化后临时块少且形状固定，
  各档上限扫描均未复现。

### 3.4 不要每步 `mx.clear_cache()`

```python
# 反例：这会把 fwd+bwd 可复用的激活缓存块一并清掉，实测反而更慢
mx.clear_cache()
```

缓存治理交给 `set_cache_limit` 统一负责。`mx.clear_cache()` 只用在
**sweep 脚本的候选配置之间**，防止旧 Metal 程序污染下一个候选。

### 3.5 host sync（`.item()`）的纪律

`.item()` 会 stall 整条 pipeline，且在 `mx.compile` 图内直接违法。规则：

- **必须在 eager 侧、且尽量低频**：`mask_has_pad` 每微批一次、梯度范数
  每优化器窗口一次、loss 日志每 `log_interval` 一次。
- **不要在训练循环里 `int(self.state["step"])`** —— 每步每形状组一次
  sync。BatchedMuon 改用 Python 侧 `_local_step` 计数。
- 梯度范数在不裁剪时用 bf16 直接 `square-sum`，避免每个 tensor 物化
  两份 f32：~1B 参数下访存流量 **~20GB → ~7GB / 窗口**。
- compile 图内需要观测的量（MoE 负载、QB margin）**必须作为 loss 的
  返回值**随图物化；写在 Python 属性上的侧信道会被 DCE 剪成无
  primitive 的占位数组，日志时 eval 不出来。

---

## 4. MLX Python 层优化

### 4.1 `mx.compile` 工程规则

**在哪一层 compile**：训练是整步 `mx.value_and_grad(loss)` 一起 compile；
优化器内部另有独立的子图 compile（NS 核心、FusedAdamW）。

**规则 1：参数必须显式传参，不能靠闭包捕获。**

```python
# 反例：nn.value_and_grad + mx.compile —— params 通过闭包被 compile
# 捕获为常量，梯度永远基于初始权重，优化器更新完全无效
# 正解：
def _loss_and_grad_with_params(self, params, X, Y, ...):
    self.model.update(params)      # 参数成为运行时输入
    return self._loss_fn(X, Y, ...)
```

**规则 2：compile 后立刻用真实 params 恢复 module 状态。** compiled fn
内部的 `model.update(params)` 只在 trace 时执行，会把无 primitive 的
占位数组留在 module 上，污染后续 optimizer step 的 eval。

**规则 3：把动态性收敛成有限个图变体。** `mask_has_pad` 在 eager 侧
算成 Python bool 再传入 → compile 最多两个图变体。MoE 从「动态容量桶」
改成 `sorted gather_mm` 后形状只随 `(B,T,E,K)` 静态确定，才敢整步 compile。

**规则 4：dropout > 0 必须回退 eager。** `mx.compile` 按 trace 冻结 RNG，
dropout 掩码会每步恒定不变。本项目检测到 `dropout > 0` 时自动关 compile
并打日志。

**规则 5：compile 之前必须 `prewarm_all()`。** 见 §5.8，这是最贵的坑。

**compile 无效或更慢的场景**（实测）：

| 场景 | 数据 |
|------|------|
| 未 prewarm 就 compile | 整步 f+b **807 → 1189 ms**，峰值内存 **17.4 → 31.0 GB** |
| `causal_conv` 单独 compile | 单 conv f+b **1.57 → 5.7 ms**（custom_function 在图内开销） |
| 整步 compile + MoE 动态桶 | 桶形状每步变 → 反复重编译，收益被抵消 |

### 4.2 算子选择清单

这一节是**查表用**的，每条都有实测。

**① `mx.split` 而不是多次切片 `x[..., a:b]`**

切片的 VJP 各自 scatter 进一份全宽零张量再相加（KDA 六路投影：六写
五加、全宽 74MB/层）；`mx.split` 的 VJP 是单次 concatenate。

```python
# KDA 六路融合投影，f+b 19.20 → 15.46 ms/层，逐位一致
q_in, k_in, v_in, fa, g_in, bl = mx.split(
    x @ w_in.T, [proj, 2*proj, 3*proj, 3*proj+D, 4*proj+D], axis=-1)
```

GQA 的 QKV 同理。**凡是「一次大 GEMM 然后拆开」的地方都用 split。**

**② 广播前先把形状扩到位，避免 VJP 走通用慢路径**

```python
# (H,) 直接对 (B,T,H,D) 广播时，A_log 的 VJP 是「沿 (0,1,3) 归约、保留
# 中间轴」，MLX 走通用慢路径，单独一项就 5.0ms/层。
# 先显式扩到 (H,D)：变成沿 (0,1) 的连续归约 + 一次 768 元素小归约。
# 该段 f+b 5.72 → 1.42 ms/层。
```

**判据：如果一个广播的 VJP 需要「跳过中间轴归约」，就先 broadcast 到位。**

**③ MoE 用 `mx.gather_mm(sorted_indices=True)`，不要容量桶**

三路径按 (token,choice) 对数 `G = B×T×top_k` 静态切换：

| 条件 | 路径 | 说明 |
|------|------|------|
| `G ≤ 512` | 融合 Metal kernel | decode/极小批量，不可微，仅推理 |
| `G ≤ 4096` | 稠密全专家广播 matmul | 小 prefill，可微、可 compile |
| 更大 | `argsort` + `gather_mm(sorted_indices=True)` | 训练主路径，免 padding |

```python
order = mx.argsort(flat)           # 同专家的 pair 连续（sorted_indices 前提）
h = mx.gather_mm(xs[:, None, :], gu_t, rhs_indices=exps_s, sorted_indices=True)
out = mx.zeros((M, D), dtype=mx.float32).at[tok_s].add(yw.astype(mx.float32))
```

实测（M=12288, K=8, E=256, D=384, I=320, mlx 0.32）：fwd **8.8 ms**、
f+b **25.5 ms**，**负载倾斜下耗时不变**，反向峰值内存 ~0.5 GB/层。
被替代的旧容量桶路径在倾斜时 padding **1.7~3×**、反向慢 **3~4×**，
还需要 host sync 读桶容量。

**④ `mx.fast.*` 优先**

- `mx.fast.rms_norm`：bf16 下比手写 f32 square/mean/rsqrt 链快数倍。
  无参 RMS 单位化也走它（传 ones 权重，权重按 `(D, dtype, eps)` 缓存）。
- `mx.fast.scaled_dot_product_attention`：decode 的 GQA **不要
  `mx.repeat` 扩 K/V**，mlx 0.32.1 原生支持 GQA 且与 repeat 逐位一致，
  省掉 O(T×n_rep) 的 K/V 物化。
- **注意**：对拍手写 kernel 时，eager 参考不能随便用 `mx.fast.rms_norm`
  —— 它和核内 f32 归约的舍入路径不同，bf16 下 grad 相对误差能到 0.3，
  会把正确的 kernel 误判成 bug（见 §5.9）。

**⑤ 离散索引后面接 `mx.stop_gradient`**

```python
idx_ext = mx.argpartition(-sel, k1-1, axis=-1)[..., :k1]
idx_ext = mx.stop_gradient(idx_ext)   # 否则 autodiff 向 indices 要 VJP
```

**⑥ 融合 GEMM 与权重堆叠**

- 运行时 `mx.concatenate(weights, axis=0)` 把多个小 GEMM 并成一个大的。
- eval 模式按权重对象 identity 缓存拼接结果（`object.__setattr__` 绕过
  nn.Module 登记）；训练每步必须重拼（权重会变）。
- MoE 专家堆叠成 `(E, out, in)`，gate/up 合并成 `(E, 2I, D)` 单次 GEMM
  再 split。
- 碎片 GEMM 惩罚很大：batch-8 × 36 次只有 **7.52 TFLOPS**，
  batch-288 × 1 次是 **10.95 TFLOPS**。**能并就并。**

**⑦ 切片喂给 kernel 前先物化连续副本**

切片后的非连续视图直接进 conv：**0.89 ms vs 连续输入 0.24 ms**（每 conv）。

### 4.3 自动微分层面

**逐元素链的反向是重灾区。** 反复出现的模式是 bwd/fwd 比例异常
（KDA 门控段 8.6×、conv ≈7×）。原因是每个逐元素 op 的 VJP 都要重新
读写全宽张量。对策：融合成一个 kernel（§5）。

**手写 VJP 替代 autodiff 穿透循环**：KDA 的跨 chunk 状态扫描用
`@mx.custom_function` + 手写反向（对 S 链的 cotangent 逆时间递推），
比让 autodiff 展开 64 步循环快 **~4×**。

**内存换速度**：反向时物化 `S_c` / `vt` 供 VJP 复用，换来 ~1.5× 反向提速。
反之 `vt` 在另一处选择重算而非物化（C ≪ D，重算更便宜）。
**没有普适答案，两个方向都要试。**

**本项目没有用 `mx.checkpoint`**：KDA 走的是「物化中间态 + custom VJP」
路线，比通用梯度检查点更可控。

---

## 5. 手写 Metal kernel

### 5.1 什么时候值得写

看两个信号：

1. **kernel 数量爆炸**：decode 路径一个 KDA 层 eager 要发 **~28 个**小
   kernel、GatedNorm **~7 个**、MoE **~18 个**。融合成 1~3 个是纯赚。
2. **roofline 倍数 > 2×**，且瓶颈是中间量往返 HBM。例：kda_prep 的
   cumsum 链 f+b **4.45 ms** 对 0.67 ms 的 compulsory 下界，融合后
   **1.63 ms**。

反之，如果 MLX 已经在跑一个大而方的 GEMM，别碰。

### 5.2 骨架范式

两阶段：build（编译 Metal 源，lazy）+ call（每次 launch）。

```python
_KERNELS = {}   # 按 (形状常量, dtype, 变体) 缓存

def _build(C, D, dtype):
    src = f"""
        uint gx = thread_position_in_grid.x;
        constexpr uint C = {C};          // 维度编译成常量
        constexpr uint D = {D};
        if (gx >= D) return;
        ...
    """
    return mx.fast.metal_kernel(
        name=f"my_kernel_{C}_{D}_{dtype}",   # name 必须含所有 key
        input_names=["x", "w"],
        output_names=["out"],
        source=src,
        header=_HEADER,       # 需要 simdgroup_matrix 时必须给
    )

def _get(*key):
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = _build(*key)
    return k

(out,) = _get(C, D, dtype)(
    inputs=[x, w],
    output_shapes=[(B, T, D)],
    output_dtypes=[x.dtype],
    grid=(nt, rows, 1),
    threadgroup=(nt, 1, 1),
)
```

`name` 必须包含 cache key 的全部成分——同名不同源码会踩 MLX 内部缓存。

### 5.3 维度编译成常量 vs 运行时传入

**编译成 `constexpr` / `#define`**：驱动循环完全展开、常量折叠、寄存器
数组（`float acc[C]` 只有在 C 是编译期常量时才能待在寄存器里）。本项目
几乎所有 head_dim / chunk / 专家数都这么做。

**留作 scalar input**：会频繁变化的维度。`conv` 把 `T`/`B`/`NC` 作为
input 传入，避免每个序列长度重编译一份 kernel。

**权衡准则**：取值集合有限（≤ 十几种）就编译成常量，否则传参。
`attn_res_fused` 把 N 逐 j 展开、每个 N 一份 kernel，N > 26 直接回退
eager（Metal buffer 数量 `2N+3 ≤ 31` 的限制）。

多编译几个 kernel 是**一次性成本**；漏掉一个变体则整轮训练回退（§5.8）。

### 5.4 threadgroup 内存预算与 bank conflict

**硬上限 32KB**，实践中留余量到 30720。**启动前静态算字节数、超了就
拒绝这个分块**（`attn_fused.py:_tgmem()` 就干这个）。

超限的实际处理：

- KDA scan 的 S 全片 96×96×4 = 36KB 超限 → **Dv 二分**（`dv_split=2`）
  降到 18KB，o/Sall 按半片写、无跨片归约。
- CE 的行 buffer 需要 `V*4 ≤ 28KB` → V=6400 时 25.6KB 刚好放得下，
  更大词表自动回退。
- flash 的 f32 输入 Ks/Vs 占用翻倍（D=96 时 38KB）→ f32 一律回退
  mlx SDPA（训练主路径是 bf16，无收益）。

**bank conflict：行距填充是必须的。**

```python
# threadgroup 内存是 32 个 4 字节 bank。bf16 行距 DK=128 时一行 256 字节
# = 64 bank，回绕后每行都从 bank 0 起——simdgroup_load 转置读一个 8×8
# 片段要跨 8 行，8 路全撞同一 bank，串行化。
# 行距 +8 个 bf16 让相邻行错开 4 个 bank（8 行落在 0/4/…/28，互不相同）。
PAD_B = 8    # bf16 缓冲
PAD_F = 4    # f32 缓冲
```

代价只有每块几百字节。**注意**：`probe_mma_peak` 里单独加行距填充对
纯 MMA 循环**无效**——它只在有转置载入的真实 kernel 里起作用。

### 5.5 MMA 操作数放置：1.3 vs 12 TFLOPS

**这是 Apple GPU 上最反直觉的一条。** 把两个操作数都塞进寄存器循环
MMA 只有 **1.3 TFLOPS**；让 B 每次从 threadgroup 载入反而是
**12.1 TFLOPS**（快 9 倍）。

放置矩阵（`probe_mma_operand.py` 本机实测）：

| A 来源 | B 来源 | TFLOPS |
|--------|--------|--------|
| threadgroup | threadgroup | **11.0** |
| 寄存器 | threadgroup | 10.4 |
| device | threadgroup | 10.2 |
| threadgroup | device | 9.8 |
| device | device | **7.2** |

**行动准则**：写 MMA kernel 时，两个操作数尽量都从 threadgroup 载入；
**绝对避免两个都从 device 直读**。flash 反向早期卡在 3.2 TFLOPS 就是
因为 dP 那步两个操作数都在 device。

### 5.6 grid 布局

**① 避免热路径上的整数除法。** situ 打包核最初用一维 grid + `i / half`
算行号，在 37.7M 元素上把 fwd 抬高了 **5×**。改成二维 grid
`(c, row)` 后消失。

**② 保证足够的并行度。** conv 不做 T 分块时 grid 只有 C×B = 9216 线程，
占 M4 Max 约 20% 并发 → 引入 `_T_BLOCK=128` 分块。

**③ 用 grid.z 的奇偶合成两个独立 kernel。** flash 的 dq 与 dkv 没有
跨 threadgroup 依赖，合成后 GPU 可以在两种线程组间自由调度、互相填充
延迟，还能共享同一块 threadgroup 内存。

**④ 发射几何要 sweep。** KDA scan 旧几何 fwd 只有
`B·H·dv_split = 192` 个 threadgroup、每组 21.5KB 共享内存 → 每核只能
驻留 1 组，occupancy 极差。`sweep_kda_scan.py` 扫
`dv_split × nt_fwd × nt_bwd` 定出默认值 `2 / 256 / 256`。

**实测最优 tile / 几何**（M4 Max，B=12 H=8 T=1024）：

| 模块 | 参数 |
|------|------|
| flash | `NTHREADS=256`, `STR=16`（dq/dkv）, `STR_LSE=32`, split-D `RES=32` |
| conv | `_T_BLOCK=128`, `_TG_W=128`, `_DW_T_CHUNKS=128` |
| kda_scan | `dv_split=2`, `nt_fwd=256`, `nt_bwd=256`, 寄存器分块 `JB=6` |
| attn_res | 256 threads, `_MAX_N=26` |
| ce / gated_norm / swiglu | 256 threads/行 |

### 5.7 寄存器压力：宁可多算一遍

D=128 时 dkv 的两套 64 维累加器会把寄存器撑爆。解法是**两遍扫 query**：
第一遍只累加 dV，第二遍只累加 dK，寄存器里始终只有一套累加器。
**S 算两遍比两套累加器撑爆寄存器更便宜。**

更好的解法是 split-D：把 D 维按 simdgroup 对半分，每个 simdgroup 只
累加自己那一半 dQ，直接消掉了 AccHi 这个 threadgroup 累加器，从而
腾出空间让 NT=256 保持 RES=32 行、threadgroup 内存 ~16KB。
**实测 dq 7.4 → 4.6 ms、dkv 10.1 → 6.1 ms。**

寄存器数组的硬约束：`kda_prep` 的 `acc[C]`/`g[C]` 要求 **C ≤ 32**，
更大会溢出到 thread-local memory（等于回到 HBM）。

### 5.8 prewarm：本项目最贵的坑

**必读。** 这个坑让一次实验白跑了整轮。

机制：每个融合 kernel 首次调用会对照 eager 参考做一次数值校验，而
校验必须 `.item()` 取回标量。如果这次首调用**落在 `mx.compile` 的
trace 内**：

```
.item() 在图内 → 抛异常 → 被 dispatch 的 except 分支吞掉
              → 模块级 _DISABLED 永久置真
              → 整轮训练静默走 eager，且没有任何日志
```

**代价**：KDA 层 fwd+bwd（bs12×1024）kernel 全启用 **49.7 ms**、全回退
**72.0 ms**，每层差 **22.3 ms**。整步层面：**807 → 1189 ms，峰值内存
17.4 → 31.0 GB。**

**解法**：compile 之前在 eager 上下文里把所有**实际会用到的形状**跑一遍。

```python
from model.kernels import prewarm_all
prewarm_all(model, cfg, dtype, seq_len, log=print)   # 必须在前
vg = mx.compile(vg)
```

`prewarm_all` 的设计要点：

- 从**模型结构**（遍历 `named_modules`）推导形状，而不是从配置字段名
  —— 配置改名不会导致漏预热。
- 对每个 conv 通道数同时预热 **seg × silu 四种变体**：变体由运行时是否
  打包序列决定，多编译几个是一次性成本，漏掉一个则整轮回退。
- 返回 `(ok_count, fail_names)` 并打日志。**失败的 kernel 已切到 eager，
  训练仍然正确，只是慢。**

**验收手段**（必做）：

```bash
.venv/bin/python experiments/verify_prewarm.py       # 三种顺序对比
.venv/bin/python experiments/probe_kernel_fallback.py # compile_first True/False
```

这两个脚本用**子进程隔离**模块级 `_DISABLED`，并直接打印
`kda._SCAN_KERNEL_DISABLED` / `conv._DISABLED` / `kda_prep._DISABLED` /
`attn_res_fused._DISABLED`。**提交前确认这些 flag 全是 False。**

同类手法：`verify_muon_hit.py` 验证 Muon 的 NS 缓存命中路径真的走了。
**「优化有没有真的生效」和「优化对不对」是两个独立的验收项。**

### 5.9 正确性校验与容差

标准流程：

```
1. prewarm（compile 前）：value_and_grad 对照 eager，mx.eval + .item()
2. 失败 → _DISABLED / _FAILED = True → 永久 eager 回退
3. 成功 → _VERIFIED.add(key)，此后 compile 图内直接命中缓存不再校验
```

**容差要按 dtype 和链路分别标定**（都是实测标出来的）：

| 对象 | f32 fwd | bf16 fwd | f32 grad rel | bf16 grad rel |
|------|---------|----------|--------------|---------------|
| attn_res / situ（含 softmax+RMSNorm） | 2e-4 | 5e-2 | 5e-3 | **0.5** |
| conv | 1e-5 | 2e-2 | 1e-4 | 5e-2 |
| gated_norm / swiglu / kda_decode | 1e-5 | 2e-2 | — | — |
| kda_prep | 2e-5 (rel) | — | 2e-5 (rel) | — |
| kda_scan（在线） | — | — | **1e-4 (rel)** | — |

三条硬经验：

**① bf16 的 grad 相对误差可以合理地大到 0.5。** 含 softmax+RMSNorm 的
VJP 在 bf16 下 rel ≈ 0.3 是**舍入不是 bug**。早期阈值定太严，把正确的
kernel 误杀了。

**② 状态链要用相对误差，不能用绝对阈值。**

```python
# 状态链的合法 FMA 顺序差会随 NC 累积，绝对阈值在大数值范围下误判；
# rel 1e-4 对 f32 重排足够宽松、对真实 bug（如索引错位 O(1) 偏差）仍然敏感
d1 = ((o - o_ref).abs().max() / (o_ref.abs().max() + 1e-12)).item()
```

**③ eager 参考必须走和 kernel 相同的数值路径。** kernel 内部 f32 归约，
参考就不能用 `mx.fast.rms_norm`（bf16 输入）——两者差到 grad rel 0.3，
刚好卡在阈值上把整条融合路径误杀。decode kernel 更极端：要用显式
`(bfloat16_t)` cast **逐级复现 eager 的 7 级 bf16 舍入链**才能对齐。

**④ 失败要按 key 隔离，不要全局禁用。** `attn_res` 早期一个 N 校验
失败就置全局 `_DISABLED`，株连了其它所有 N。改成 `_FAILED: set` 后
只禁用该 `(N, D, dtype)`。

**⑤ prewarm 的测试形状要挑过**：`B=2, T=4`（不用 2 token，避免 bf16
舍入过阈值）、conv 用 `T=37` 奇数覆盖 tap 边界、kda_scan 用小范数输入
（×0.02~0.05）防状态链数值爆炸。

### 5.10 数值精度策略

| 位置 | 策略 |
|------|------|
| MMA 累加器 | **float**（`simdgroup_matrix<float,8,8>`），操作数可以 bf16 |
| kernel 内部累加 | f32，写回时 cast 回 bf16 |
| SSM / 线性注意力状态 S | **全程 f32** |
| LSE / delta / alpha 等中间量 | f32 |
| 梯度输出 | kernel 输出 f32 → `astype(primal.dtype)` |

特殊数值处理（都是踩过的）：

- **因果 mask 全屏蔽行**：块内 `mcur` 仍是 -inf，`exp(+inf)` 出 inf/nan
  → 必须整体跳过而不是照常更新。
- **padding 行 lse 落 0 而非 -inf**：避免下游 `exp(S-L)` 出 nan
  （这些行梯度贡献本来就该是 0）。
- **MMA 两个操作数 dtype 必须一致**：训练图的 cotangent 常是 f32 而
  K/V 是 bf16，进 kernel 前先 `dout.astype(q.dtype)`。

### 5.11 一些 API 层面的实测

- **不要传 `template`**（至少在 kernel 场景）：host 下发 **0.65 µs vs
  3.11 µs**（~4.8×）。Metal 类型名直接注入源码即可。
- **小输入会被放 constant 地址空间**（随尺寸变化），body 内一律直接
  下标索引，不声明局部 device/constant 指针。
- **JIT 编译是 lazy 的**，首次调用后需要 `mx.eval` 才真正触发。
- **避开 `atomic_outputs`**：conv 的 dw 改成按 T 轴切 8 片写部分和
  `(8,K,C)`，由调用方一次 `sum` 归约（归约张量仅 96KB），绕开原子写和
  零初始化。
- **省掉零初始化**：kda_scan 反向首轮直接载入 cotangent 而不是先清零。

---

## 6. 优化器（Muon / Newton-Schulz）

optimizer 在 M4 Max 上能占整步墙钟 **~48%**，值得单独优化。

**① 批量化 Newton-Schulz。** 同形状权重 reshape 后 `mx.stack` 成
`(N,r,c)` 一次做 NS。100M 模型的 103 个 Muon 张量从 **~1700 kernel/步**
降到 **~16 kernel/形状组**。

**② ~~每 N 步复用 NS 极因子~~（已删除，勿重引入）。** 曾默认每 8 步做
一次完整 Gram-NS5、中间步复用旧极因子，省 ~30% 优化器墙钟。r082 归因
（7 条单变量 probe，同 seed/数据、微步对同步）实测这是 r081→r082 回退
的**最大单项元凶：早期损失 0.4-0.5 nat**——方向仍精确正交但相对当前
动量是 stale 的，warmup 期动量旋转最快，恰好最不能复用。叠加 per-head
NS（单独 −0.14 nat）后总回退 ~0.9 nat @500 步；动量 warmup 经隔离
probe（p9）洗清，单独无害。整套机制（含 Temporal Q 缓存）已删除，
正交化每步全量重算；详见 §7.1 与 BatchedMuon 类 docstring 的负面
结果记录。

**③ Gram-NS 只对 r ≪ c 有利。** `_ns_auto` 实测：

| 形状 | 标准 NS5 | Gram-NS |
|------|----------|---------|
| (59,768,768) 方阵 | 61 ms | **73 ms**（更慢） |
| (2304,384,384) 方阵 | 320 ms | **371 ms**（更慢） |
| (2,6400,768) | 12.9 ms | **5.2 ms**（更快） |
| (2304,768,384) r<c | 527 ms | **414 ms**（更快，~25% FLOPs） |

**④ 大张量不要随便 stack。** 专家栈单份 G/P/V 约 **1.27 GB**，三次
stack + scatter 光搬运就 **30 ms/组/步**。**stack 的收益有 size 上限。**
NS 降频删除后刷新路径也改成了逐张量（`probe_stack_free_opt.py`：
mom/NS/apply 全逐矩阵语义，跨层 stack 是纯搬运；逐张量版与堆叠版
**逐位一致**，两组 −41ms/步，`bench_muonh` 专家 NS 751→712ms）。

**⑤ FusedAdamW**：10 个逐元素算子融合成 1 个
`@partial(mx.compile, shapeless=True)`。MoE 细粒度化后 AdamW 覆盖
621M/655M 参数，未融合 **88 ms**，而 8.7GB 访存的带宽下界只有 **~22 ms**。

**⑥ 范数计算值得写 kernel**：专家栈 (2304,640,384) 上
`mx.linalg.norm(x.astype(f32))` **12 ms → ~2.5 ms**。（bf16 直接 square
有 1.9% 误差，hyperball 投影不能接受。）

**⑦ 含 `_fro_norm` 的投影必须 shapeful compile**（grid 绑形状）；
纯逐元素链才能用 `shapeless=True`。

**⑧ cubic5 低次 NS 系数 + 膝盖扫描（默认 cubic5b05）。** quintic 的
`X←aX+bX³+cX⁵` 换成 NVIDIA relaxed cubic（arXiv 2606.00371，u=1.3、
l₀=7e-3，Chen–Chow 闭式逐迭代系数，复现见
`experiments/probe_cubic5.py`）：每步省掉 Gram 平方项，3→2 个主
GEMM。P13 probe（14 分钟、同 seed 微步对同步）实测：500 步 loss
**5.245 vs classic 5.328（−0.08 nat 不差反略好）**，步速
0.70→0.73 step/s，`apply_gradients` 基准 **−19.5%**
（`bench_muonh.py` exp_bf16_ns5 行）。同批对照证伪了两个方向：
PE5 正交化残差减半却 +0.14 nat、无 NS 只归一化 +0.34 nat——
**训练质量不由 polar 精度单调决定**（与 2606.00371 的核心结论
一致），classic 的 [0.65,1.2] 带宽本身就是好口径。
膝盖位置 l₀ 作为一等超参的扫描（同 probe 协议，loss@500）：
l₀=0.007（cubic5）5.245 → l₀=0.05（b05）**5.223** → l₀=0.10（b10）
5.314。峰值在 l₀=0.05，b05 与 cubic5 之差在 probe 噪声（±0.05）
内、成本相同，按预注册规则翻默认为 **cubic5b05**。切换复现：
`VIBY_MUONH_NS_COEFF=classic|pe|cubic5|cubic5b05|cubic5b10`。

---

## 7. 失败清单（试过、更慢或被回退）

**负面结果和正面结果一样值钱。** 下面每一条都真的实现过。

### 7.1 MLX Python 层

| 方案 | 结果 |
|------|------|
| 一次 `mx.eval(loss + grads)` | 46GB / 4.0s，拆成两次后 ~16GB / 快 2× |
| 每步 `mx.clear_cache()` | 清掉可复用激活块，更慢 |
| 切片代替 `mx.split`（QKV / 六路投影） | 反向 scatter 全宽梯度，f+b +3.7 ms/层 |
| `(H,)` 直接广播到 `(B,T,H,D)` | VJP 慢路径，单项 5.0 ms/层 |
| autodiff 穿透 chunk 循环 | 比 custom VJP 慢 ~4× |
| MoE 容量桶 + padding | 倾斜时 padding 1.7~3×、反向慢 3~4×，且需 host sync |
| `causal_conv` 单独 compile | 1.57 → 5.7 ms |
| 整步 compile + MoE 动态桶 | 反复重编译 |
| dropout > 0 + compile | RNG 被冻结，语义错误（已自动回退） |
| MoE router 走 Muon | 负载失衡：top-1 桶 C 从 ~6K 涨到 13K+，吞吐 -30% |
| router 用 base lr | C → 14K，吞吐 -45%（须 0.05× 或更小） |
| 关掉 `bias_correction` | 前百余窗口有效步长放大 7~15×，beta2=0.9998 下 KDA 衰减参数溢出 NaN |
| Muon NS 降频复用（每 8 步重算正交化、命中步复用旧极因子） | **质量灾难**：方向精确正交但相对当前动量 stale，r082 归因实测早期 −0.4~0.5 nat（r081→r082 总回退 ~0.9 nat 里的最大单项），墙钟只省 ~30%。机制已删除，勿重引入 |
| Muon Temporal Q 缓存（命中步 `D=Q@normalize(U)`） | 正交残差 4~13（完整 NS5 只有 0.02~0.03），残差门几乎每步 fallback，整步 **1585 → 3098 ms**。机制已删除 |
| 命中步 stack 专家栈 G/P/V | 30 ms/组纯搬运（1.27 GB/份） |
| Gram-NS 用在方阵组 | 61→73 ms、320→371 ms |
| Polar Express 5 步系数替换 NS5（同 GEMM 数、正交化残差减半 σ[0.80,1.13]） | 训练反而变差：P12 probe 500 步 **+0.14 nat**（5.465 vs 5.328）。polar 精度高 ≠ 训练好，勿以残差口径选型 |
| Gram 逆平方根 Newton 细化 warm-start（跨步复用 Y≈G^{-1/2} 逐步细化） | **永不收敛**：warm 残差被 κ(G)=κ(X)² 放大（动量每步旋转 ~1.8e-2 × κ² ⇒ 残差 ~15，Newton 盆地要求 ≲1），合成轨迹 100% 回退（`experiments/probe_ns_refine.py`）。逆因子复用与 stale-D 同源，勿重引入 |
| 去掉 NS 只归一化 + hyperball（`VIBY_MUONH_NO_NS` 测量开关） | **+0.34 nat** @500 步（5.669 vs 5.328）：NS 谱均衡本身值 0.34 nat，其墙钟占整步 ~27% |
| 谱指数 NS：幂律谱变换 σ→σ·(σ²+s)^((p−1)/2) 的 Chebyshev-on-Gram 单发实现（frac50/frac25，8/10 GEMM vs cubic5 10） | **家族否定**：bf16 保真合格（max\|Δσ\|≤3.4e-2、单调）但 P14b/P15b probe @500 frac50 **5.523（+0.28）**、frac25 **5.441（+0.20）** vs cubic5 5.245——「幂律主体保大 σ 序信息」在此尺度有害，bulk 必须压平。教训：frac 输出必须把 \|\|D\|\|_F 钉回 √min(r,c)，未钉回（P14）+0.78 nat 主要是范数亏空非形状。勿以换 p/换正则重试同族（设计与验证见 `experiments/probe_frac_design.py`） |

### 7.2 Metal kernel 层

| 方案 | 结果 |
|------|------|
| MMA 两操作数都在寄存器 | **1.3 TFLOPS**，比从 threadgroup 载入慢 9× |
| MMA 两操作数都从 device 直读 | 7.2 TFLOPS（vs 都在 threadgroup 的 11.0） |
| flash 前向在线缩放 O | 每块把 O 卸到 threadgroup 再乘 α 太贵；改成 P=exp(S) 留 f32、末尾除 Z |
| D=128 单遍 dkv（两套累加器） | 寄存器撑爆；改两遍扫 query |
| kda_scan「每线程一输出元素」朴素版 | 与 eager 同速；需寄存器分块 + 发射几何 sweep |
| kda_scan fwd 用 Dv 全片 | 36KB > 32KB 上限；改 `dv_split=2` |
| conv 不做 T 分块 | 9216 线程 ≈ 20% 占用 |
| conv dw 用 atomic 输出 | 改 partial sum + host 归约 |
| situ 一维 grid + `i/half` 除法 | 37.7M 元素上 fwd **慢 5×**；改二维 grid |
| situ 两参数 + `reshape(-1)` | 跨步视图各物化副本；打包核 **8.6 → 1.7 ms** |
| kernel 传 `template` | host 下发 0.65 → 3.11 µs |
| flash 走 f32 训练路径 | threadgroup 38KB 超限且无收益，回退 mlx SDPA |
| eager 参考用 `mx.fast.rms_norm` 对拍 | grad rel ≈ 0.3，误杀整条融合路径 |
| 单个 key 校验失败就全局 `_DISABLED` | 株连其它 N，改按 key 隔离 |
| flash 单独加行距填充（纯 MMA 循环） | 无效（只在有转置载入的真实 kernel 里起作用） |
| 宽 conv 一次算 2304 通道 | 未显著优于 3× conv |

### 7.3 运维层面

| 失败模式 | 后果 |
|----------|------|
| compile 之前没 prewarm | 融合 kernel 静默永久 eager，**无任何日志**，整步 807→1189 ms、内存 17.4→31 GB |
| 漏预热某个 conv 变体（seg × silu 四种） | 该变体首调用落在 compile 内 → 回退 |
| 跨进程「改前/改后」对比计时 | ±30% 漂移，结论不可信 |
| 连续跑 benchmark 不冷却 | GPU 降频，整体慢 ~20% |
| MoE 统计走 Python 属性侧信道 | compile 下被 DCE 剪成占位数组，日志 eval 不出来 |

---

## 8. 落地 checklist

### 8.1 新增一个融合 kernel

- [ ] roofline 算过，倍数 > 2×（§2.3）
- [ ] threadgroup 用量静态算过 ≤ 30720 字节（§5.4）
- [ ] bf16 缓冲行距 +8 / f32 +4 防 bank conflict（§5.4）
- [ ] MMA 操作数不是「两个都从 device」（§5.5）
- [ ] 热路径无整数除法，grid 并行度足够（§5.6）
- [ ] 内部累加 f32，写回 cast（§5.10）
- [ ] `@mx.custom_function` + 手写 VJP，或闭式 Python VJP（§4.3）
- [ ] cache key 完整且写进 `name`（§5.2）
- [ ] 写了 `prewarm()`，并接进 `model/kernels/__init__.py:prewarm_all`（§5.8）
- [ ] 容差按 dtype 标定，eager 参考走同一数值路径（§5.9）
- [ ] 失败按 key 隔离 + 环境变量总开关 + try/except 回退

### 8.2 提交任何性能改动之前

- [ ] `verify_*` 数值对拍过线
- [ ] `verify_prewarm.py` / `probe_kernel_fallback.py`：所有 `_DISABLED` flag 为 False
- [ ] 同进程交替 A/B（`ab_*`），median 和 min 双口径都报
- [ ] 若可能动到数值：跑一轮 `run_exp.sh`，holdout CE 不退化
- [ ] **把负面结果也写进代码注释**——下一个人（或三个月后的自己）
      会感谢你

### 8.3 环境变量总开关（调试 / A/B 用）

| 变量 | 作用 |
|------|------|
| `VIBY_FUSED_KERNELS=0` | 关掉 conv / kda_prep / kda_scan / situ 融合核 |
| `VIBY_SITU_PACKED=0` | situ 回退两参数核，A/B 定位 |
| `VIBY_FLASH_NOPAD=1` | 关掉 flash 行距填充 |
| `VIBY_FLASH_NT` / `VIBY_FLASH_STR` | 覆盖 flash tile |
| `VIBY_FLASH_FWD=0` | 前向走 mlx SDPA，反向重算 LSE |
| `VIBY_MUONH_PER_HEAD` | Q/K/V per-head NS（默认 0；隔离实测单独 −0.14 nat @500 步，NS 降频下放大到 −0.26） |
| `VIBY_MUONH_MOM_WARMUP` | 0.85→0.95 动量 warmup（默认 0；隔离实测单独无害也无益，r081 基线无此机制） |
| `VIBY_SITU=0` | SiTU-GLU 回退无界 SwiGLU（A/B 用；实测早期无差异） |
| `VIBY_KDA_SCAN_ZSC=0` | 关掉 kda_scan 反向的「状态 cotangent 恒零」特化（A/B 用） |

（`VIBY_MUONH_STACK_NS_EVERY` / `VIBY_MUONH_CACHE_Q` / `VIBY_MUONH_CACHE_Q_RES`
已随 NS 降频复用机制一并删除——r082 归因质量灾难，见 §7.1。）
| `VIBY_DEBUG_MEM=1` | 打印 active/cache/peak + 分 gate 负载 + 桶容量 |
| `VIBY_BENCH_B/T/D/E/I/K` | benchmark 脚本的形状口径 |

---

## 9. 附录：关键实测数字总表

口径统一为 **M4 Max、1080M 配方、bs12×1024、bf16**，除注明外均为
fwd+bwd（f+b）。

### 训练整步

整步绝对值随代码演进变化很快，**要用当前基线就现跑
`bench_train_step.py --preset 1080m --compile`**，不要引用下表做绝对
比较；下表只用于看相对关系。

| 项 | 数值 |
|----|------|
| 未 prewarm 的 compile | 807 → 1189 ms；峰值 17.4 → 31.0 GB |
| 一次 eval vs 两次 eval | 46 GB / 4.0 s → ~16 GB / 快 2× |
| cache_limit 10G → 24G（bs16×640） | 提速 4.5%（峰值 14.8G，峰值+缓存 ≈39G） |
| optimizer 占整步 | 历史上 ~48%（含每 8 步 NS 刷新摊平）；cubic5b05 + 每步全量 NS 口径（2026-08-25 实测）：**窗口 2620ms，fwd 590 / bwd 1175 / opt 833（31.8%），9380 tok/s，峰值 22.8GB**；专家组去 stack 逐张量化后 opt 段 833→789ms 中位（同进程 probe 口径 −41ms，逐位一致）。opt 中专家 NS GEMM ~660ms @ ~11.8 TF/s，已贴近 bf16 GEMM 峰值——优化器侧再无 kernel 级余量，只剩质量门控的算法杠杆（如 ns_steps 5→3，bench_muonh 示 −220ms/步） |

### 组件（每层）

| 组件 | 数值 |
|------|------|
| KDA 层（kernel 全开 / 全回退） | 49.7 / 72.0 ms（差 22.3） |
| MoE 层 | 57.04 ms ×9 ≈ 513 ms/微批；7.9 TFLOPS（峰值 12.9） |
| Attention 层 | 47.6 ms（SDPA 26.8、投影 GEMM ~13、胶水 ~8） |
| KDA 融合投影 + 六路切片 | 20.90 ms（→146 ms/步），纯 GEMM 下界 ~12.8 ms |
| KDA 3× causal_conv 段 | 16.85 ms（单 conv 微基准仅 1.57 ms） |
| KDA 门控段 | 7.21 ms（bwd 6.47 / fwd 0.75，比 8.6×） |
| KDA `_chunk_kda` 段 | 18.42 ms → 128.9 ms/步 |
| KDA scan | 15.01 ms，带宽下界 3.24 ms（4.6×），fwd ~1.1 TFLOPS |
| 共享专家 ×2 | 11.61 ms |

### 单点优化收益

| 优化 | 收益 |
|------|------|
| flash split-D（dq / dkv） | 7.4→4.6 / 10.1→6.1 ms |
| flash 手写 vs mlx autodiff | 16.5 vs 18.0 ms（1.10×） |
| situ 打包核 | 8.6 → 1.7 ms（compile 下 5 个逐元素核 ~5.4 ms） |
| kda_prep 融合 | 4.45 → 1.63 ms（下界 0.67）；fwd 2.05 →（下界 0.63） |
| 六路 slice → `mx.split` | 19.20 → 15.46 ms |
| `A_log` 广播先扩到 (H,D) | 5.72 → 1.42 ms |
| 连续输入 vs 切片视图（每 conv） | 0.24 vs 0.89 ms |
| MoE `gather_mm` | fwd 8.8 / f+b 25.5 ms，倾斜下不变，反向 ~0.5 GB/层 |
| `_fro_norm` kernel（专家栈） | 12 → ~2.5 ms |
| BatchedMuon | ~1700 → ~16 kernel/形状组 |
| decode kernel 融合 | KDA ~28→1、GatedNorm ~7→1、MoE ~18→3 |
| kda_scan ZSC（训练时 cot_Sall 恒零特化，逐位等价） | scan bwd 9.66→8.16 ms、`_chunk_kda` f+b 17.26→16.21 ms/层（同进程 compile 口径）；整步 bwd min 1173.0→1163.7 ms、峰值 22.83→22.67 GB（背靠背 bench，min 口径） |

### 硬件上限

见 §0。核心三个数：**带宽 400 GB/s、MMA 12 TFLOPS、threadgroup 32 KB。**

---

## 10. 脚本索引

想干什么 → 跑哪个：

| 目的 | 脚本 |
|------|------|
| 看整步分段占比 | `bench_train_step.py --preset 1080m --compile` |
| compile 口径归因 + 层数外推 | `prof_bwd_attrib.py` |
| KDA 内部分段（前缀差分） | `prof_kda_stages.py` / `prof_kda_parts.py` |
| MoE 内部分段 | `probe_moe_parts.py` |
| 找热点（SDPA 形状敏感性 / MoE padding） | `probe_hotspots.py` |
| 硬件 MMA / simdgroup 上限 | `probe_mma_peak.py` / `probe_mma_operand.py` / `probe_simdgroup.py` |
| flash tile 搜索 | `sweep_flash_tile.py` |
| conv tile 搜索 | `sweep_conv_tile.py` |
| kda_scan 发射几何搜索 | `sweep_kda_scan.py` |
| 检查融合 kernel 有没有静默回退 | `verify_prewarm.py` / `probe_kernel_fallback.py` |
| 检查 Muon NS 缓存路径 | `verify_muon_hit.py` |
| 数值对拍 | `verify_flash_bwd.py` / `verify_kda_prep.py` / `verify_fused_adamw.py` |
| 抗漂移 A/B | `ab_round_opt.py` / `ab_attn_compile.py` / `ab_moe_fwd.py` |
| 端到端质量门禁 | `run_exp.sh <round> <notes> [args]` → `research/experiments.tsv` |
