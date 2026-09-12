# Viby CSA2 / CED Metal Kernel 算法规格（交给 GPT-6 Pro）

> 历史任务规格，文件名保留以兼容既有引用。下文的角色、禁止测试等要求属于
> 2026-09-11 的算法设计任务，不是仓库通用约定。当前 Astra 工作从
> [AGENTS.md](../AGENTS.md)、[研究索引](README.md) 与 [现行实验协议](EXPERIMENT_PROTOCOL.md) 开始；
> 算法描述和瓶颈假设须与当前源码重新核对。

日期：2026-09-11。本文是**自包含算法规格**，不依赖仓库其它文档。收件人只改算法，不测、不写测试、不跑实验、不改模型结构。

---

## 0. 你的任务（硬约束）

你是算法设计者。根据下面已经写全的当前实现，给出**详细、可落地的优化算法**（伪代码或 Metal 风格即可）。

**只做这些：**

- 重写/改进 kernel 算法：分块、归约、跳过无效计算、布局、atomic 归属、tile 调度、compact 表示。
- 写清：输入输出张量、dtype、并行网格、threadgroup 内存、barrier、前向与 VJP 公式。
- 论证为何更快（少读写、少 atomic、少无效 MMA、更好局部性），以及与当前语义何处保持等价。

**禁止这些：**

- 不要写 pytest、benchmark、AB 对照、profile 脚本、复现步骤。
- 不要说「先测再改」「跑了再决定」。按本文给出的瓶颈假设直接给优化算法。
- 不要改模型结构、层数、可见集合语义、Top-K 集合规则、Sinkhorn 迭代次数、MoE 路由概率、noaux_tc。
- 不要 approximate Top-K、token dropping、expert capacity、减少 Sinkhorn iters。
- 不要把 score 输出从 FP32 改成 bf16（`1e-7` tiebreak 会下溢）。
- 不要重写已证明更慢的手写 MoE GEMM 去替代 `mx.gather_mm`。
- 不要要求读仓库其它文件；需要的算法都在本文。

交付格式（按模块分节）：

1. 当前算法的问题（1 段）
2. 优化后的算法（完整伪代码）
3. 与当前语义的等价点 / 允许的舍入差异
4. 预期少掉的计算或访存（定性即可，不给假数字）

优先顺序：稀疏 Attention 反向 → Indexer 无效打分 → Top-K/compact → MoE combine 外围 → mHC 残余。每条给一套完整算法，不要只列 bullet。

---

## 1. 问题设定（架构，已冻结）

目标硬件：Apple GPU（Metal），MLX `mx.fast.metal_kernel` + `simdgroup_matrix<T,8,8>`。统一内存。主训练形状：

```
B=4, T=1024, dim=1024, 12 层
注意力：H=16, D=128（主注意力）；MQA：K=V 共享单头 latent
滑窗 W=128
Indexer：H_idx=8, D_idx=64, index_topk=64
压缩：编码段 ratio=2 ⇒ N≈T/2；解码段 CED ratio=1 ⇒ N=T
MoE：E=96, top-K=6, intermediate=256
mHC：hc_mult=4 条流，Sinkhorn iters=20（或 cfg）
dtype：激活 bf16；kernel MMA 操作数 T∈{bf16,fp16}，累加器 FP32
```

CED：后半层的全局压缩 KV 不由本层 hidden 产生，而由编码器末层（边界层）投影后**跨解码层共享**。每层仍有自己的滑窗 KV。

CSA2 三模式（同 ratio 的层共享一份压缩池）：

- **Full**：本层产压缩 KV + 跑 indexer（选 Top-K）
- **Reindex**：复用上游压缩 KV，本层重算 Top-K
- **Reuse**：压缩 KV 与 Top-K 都复用上游

分层索引：某个 Full 层产出 candidate **块**掩码；更深 indexer 只允许在候选块内打分（语义上是搜索域上界）。当前实现是打完分再 `where(candidates, sc, -inf)`，**点积并没有省掉**——这是 Indexer 优化的核心缺口。

NEG_INF `= -1e30`。不可达 / 非候选位置必须写这个哨兵，不能写 0（ReLU 打分大量真 0）。

---

## 2. 平台约束（算法必须遵守）

- `simdgroup_matrix` 固定 8×8。SIMD group = 32 threads。
- Threadgroup 静态数组硬上限 **32 KiB**。现有稀疏反向在 `D=128, BK=32, NP=2` 时 `tile_grad` 相关合计约 32608 bytes，已贴边。
- `mx.fast.metal_kernel(..., atomic_outputs=True)` 是 **整个 kernel 的全部输出**都变成 atomic，不能同一次调用里一部分 atomic、一部分普通写。
- Custom VJP 必须给每个 array primal 返回一个 array 叶子（metadata 也要占位，不能 `None`，否则 MLX 会错位）。
- Indexer / Top-K 的下标对 MLX **不可微**：score 进选择前 `stop_gradient`。
- 热路径禁止 `.item()` / host 同步来做动态调度。
- 参与 `threadgroup_barrier` 的线程必须走一致控制流。跳过 MMA 必须整 TG 一致。
- 默认 `ensure_row_contiguous=True`；算法按 row-major 连续缓冲设计。
- 训练无 cache：`start_pos=0`，整段 T。Decode：query `T=1`，压缩池随序列增长。

---

## 3. 当前算法 A：Lightning Indexer 打分

文件概念：`indexer_score.py`。默认开启。

### 3.1 数学

输入：

- `q : [B,T,H,D]`  T∈{bf16,fp16}，H∈{4,8}，D∈{32,64}，已 RoPE
- `k : [B,N,D]`  共享 index K（未按 query 复制）
- `w : [B,T,H]`  与 q 同 dtype（weights_proj × softmax_scale × H^{-0.5}）
- `reach : [B,T,N]` bool

输出：`s : [B,T,N]` **FP32**

```
s[b,t,n] =  Σ_h  ReLU( ⟨q[b,t,h], k[b,n]⟩ ) · w[b,t,h]     若 reach[b,t,n]
          =  -1e30                                           否则
```

点积在 kernel 里用 8×8 MMA、**FP32 累加**（H=4 时把 Q 垫到 8 头，多的头 Q=0、w=0）。

VJP（`g = ∂L/∂s`，进入 kernel 前 cast 成 T）：

```
dot[h,n] = ⟨q_h, k_n⟩
gate[h,n] = 1[reach_n] · 1[dot>0] · w[h] · g[n]
dQ[h]     = Σ_n gate[h,n] · k_n
dK[n]     = Σ_h gate[h,n] · q_h          # 先按 head 收成一份再写
dW[h]     = Σ_n 1[reach] · ReLU(dot) · g[n]
```

不可达位置对 dQ/dK/dW 贡献为 0。`reach` 的 cotangent 填零占位。

### 3.2 现行并行

前向：

- 一个 query `(b,t)` 一个 TG。`NT = D`（D=32→32 threads / 1 SIMD group；D=64→64 threads / 2 SIMD group，沿 D 切开再加）。
- Key 按 `BK=16` 切块。不可达 key **仍 load 成 0 并做 MMA**，然后把该位置写成 -1e30。
- Decode（`B·T < 256` 且 N 大）沿 key 维把 TG 切到最多约 256 个：`base = tile*BK; stride = ntiles*BK`。dQ/dW 跨 tile 用 atomic add。

反向：同网格。`dK` 用 `SHARDS=8`，`shard = query % 8`，atomic 到 `[SHARDS,B,N,D]` 再 sum。`dQ`/`dW` 在 TG 内累加后 atomic_add 到全局（因为 decode 多 tile）。Gate 降成 T 再 MMA：`dQ += Gate @ K`，`dK_tile = Gate^T @ Q`。

### 3.3 调用顺序（当前浪费）

```
reach = causal_groups & doc_mask & pad_mask          # [B,T,N]
sc    = indexer_score(q, k, w, reach)                # 对 N 全部做 MMA
if Reindex and candidates is not None:
    sc = where(candidates, sc, -1e30)                # 点积已经算完
keep  = (sc - n*1e-7) >= kth_threshold  AND reach AND (sc > -1e30)
```

Candidate-source（第一个产候选块的 Full 层）必须扫完整可达域。只有 `uses_candidates` 的更深 indexer 才允许缩小搜索域。

Reuse 层不跑 indexer，只消费源层 compact 过的 `(indices, lengths)`。

### 3.4 已知算法缺陷（请针对这些给方案）

1. 候选块只在 MMA **之后**屏蔽 → 大量无效 `q·k`。
2. 空 tile（整块 BK 都 `!reach` 或 `!candidate`）仍做 MMA。
3. 训练时每个 query 单独 TG，同 batch 的 K tile 不跨 query 复用。
4. 写出完整 dense `[B,T,N]` FP32（CED 解码段 N=T 时 B4/T1024 ≈ 16MB/层·次）。
5. ties 可使 keep 个数 **> k**，所以不能把 indices 做成固定 `[B*T, k]` 除非有 overflow 语义。

### 3.5 请给出的优化算法

**P3a（必给）：提前屏蔽 + 空 tile 跳过**

- 仅 `uses_candidates` 层：`score_reach = reach AND candidates`，不覆盖共享的原始 reach。
- 每个 BK tile：整 TG 归约「是否存在任意有效 n」。全无效则直接写该 tile 的 -1e30，**跳过 MMA**；反向该 tile 对 dQ/dK/dW 贡献 0。
- 部分有效仍按原顺序算有效位置；尾块 `n>=N` 不读不写越界。
- 仍输出 dense `[B,T,N]` FP32。不要顺便改 Top-K。

**P3b（建议给完整设计）：按候选块打分，去掉 dense `[B,T,N]`**

- 源层产出按全局位置排序的 block ids + lengths；Reuse 共享。
- Kernel gather 候选块的 K，输出 compact scores + **global key id**（Top-K 偏移必须用全局 id，不是 compact 槽）。
- 处理 ties > k：可变长 count+scan，或显式 overflow fallback。禁止截断可见集合。
- 消费端直接用 compact indices/lengths；`indices` 的 stride/capacity 必须显式，因为现有 load 用 `query * NC`。

**P3c：** 训练侧多 query 共享 K tile（BQ=2/4 的算法，不是实验计划）；decode 保持 key 分片。不要把 s 改成 bf16。

Indexer 的 score 在训练图里通常 `stop_gradient` 后再做选择。VJP 仍要正确（算子契约），但训练热路径以 **fwd** 为主。

---

## 4. 当前算法 B：Top-K 与 compact

### 4.1 `_topk_masks`（默认训练路径）

对每个 query 的向量 `sc[0..N)`：

```
sel[n]    = sc[n] - n * 1e-7          # 同 dtype；sc 是 FP32
selectable[n] = sc[n] > -1e30
thr       = partition(sel).kth_largest(k)     # 可达不足 k 时 thr=-inf ⇒ 全取可达
keep[n]   = (sel[n] >= thr) AND reach[n] AND selectable[n]
```

Decode 另做 `argpartition` 取 k 个下标，**按位置排序**，加 window offset，不可达写 -1。

**精确要求：** 阈值比较保留 **全部 boundary ties**，keep 可以多于 k。不能改成「严格 k 个」。`sc > -1e30` 必须排除候选池外的 -inf，否则 -inf 会靠 tiebreak 混进 keep。

候选块：块得分 = 块内 max(sc)；最新块 `(len-1)//block_size` 钉成 `+inf`；同样 `>=` 阈值 + 下标 tiebreak；再 `repeat(block_size)` 拉回 token 轴。

### 4.2 Compact（已有，Reuse 层在用）

`visible[B,T,N]` → 每 query 扫描，`simd_prefix_exclusive_sum` 把 True 的全局下标写入 `indices[B*T, N]` 前 `lengths[q]` 项。顺序 = 原 key 下标升序。

已有 fused radix `select_topk`（默认关）：对 `sc - j*1e-7` 做 IEEE 序基数选择阈值，然后 `>= prefix AND reach AND sc>-1e30`，同样保留 ties，并 compact。不要再发明一个近似选择器；若优化，只改进这个 exact 算法（例如只在候选域上做 radix，或少扫几遍），语义必须与 `_topk_masks` 相同。

### 4.3 请给出的优化算法

把「partition + keep + compact」收成一次、仍 exact。或：在 compact 域上做 top-k 阈值（输入已是候选块）。零有效、N=0、N<k、k=1、全 0 分、大面积相同正分、全 -inf、不可达高分、decode offset，算法都要定义行为。

---

## 5. 当前算法 C：Indexed MQA Flash Attention（稀疏注意力）

文件概念：`sparse_attention.py`。默认开。仅 H=16、D∈{64,128}、W>0、bf16/fp16、无 cache 训练路径。

### 5.1 数学（K=V 同一份 latent）

每个 query `t` 看见：

- 滑窗：位置 `p ∈ [t-W+1, t]`，且同 document、非 pad
- 压缩：indexer 选出的全局下标 `j`，对应压缩槽 `j`（与 window 下标空间分开）

MQA：所有 16 个 query head 共享同一 `kv ∈ R^D`。

带 learnable sink 的在线 softmax（每 head 独立）：

```
m_0 = sink[h],   l_0 = 1
对可见 key tile：
  s_j = ⟨q_h, k_j⟩ * scale          # 无效槽 -inf
  m' = max(m, max_j s_j)
  p_j = exp(s_j - m')
  l' = l * exp(m-m') + Σ p_j
  o' = o * exp(m-m') + Σ p_j v_j
out_h = o' / l'
lse_h = m' + log(l')
```

反向（保存 out、lse，重算可见 score）：

```
delta_h = ⟨dO_h, O_h⟩
P_j     = exp(s_j - lse)            # 无效 0
dS_j    = P_j * (⟨dO, V_j⟩ - delta) * scale
dQ      = Σ_j dS_j K_j
dKV_j   = dS_j Q + P_j dO           # 因为 K=V，必须两条都加
dSink_h = -exp(sink-lse) * delta
```

**不能只实现 dK 或 dV 一条。**

### 5.2 现行并行

- 一 query 一 TG。`NP = D/64`（D=64→1，D=128→2）。`NT=64*NP`。2 个 head-group（各 8 head）× NP 个 D-partition。
- `BK=16`（32 的代码保留但默认不用）。
- `_LOAD_TILE`：前 W 个槽是滑窗位置 `t-W+1+slot`（越界/pad/跨文档 → key=-1，K 填 0）；其后 `lengths[query]` 个是 `indices[query*NC + i]` 指向压缩池。
- MMA：`S = Q @ K^T`（K 转置 load），softmax 后 `O += P @ V`。
- 反向：同一 load；MMA 出 S 与 ⟨dO,K⟩；构造 `Ds`；`dQ += Ds @ K`；对每 16 行 key：`dKV = Ds^T @ Q + P^T @ dO`，然后 **atomic_add** 到 sharded `dwindow` / `dcompressed`。
- `_SHARDS=4`，`shard = (t/16) % 4`。所有输出因 `atomic_outputs=True` 都是 atomic，包括每元素只写一次的 dQ/dSink（仍 atomic_store，且要先清零）。

Threadgroup 布局（概念）：

```
Ks[BK*(D+8)]          # +8 抗 bank conflict
ids[BK]
scores[NP*H*(BK+4)]
Ps[H*(BK+8)]
Os[H*(D+4)]
反向另加 dprob、Ds、tile_grad[16*(D+4)]
```

### 5.3 已知算法缺陷（请针对这些给方案）

1. dKV 多 query 打同一 key（滑窗重叠 + 压缩 key 复用）→ atomic 竞争。
2. dQ/dSink 本是 query-owned 唯一写，却被绑在 atomic kernel 上，还要清零。
3. 反向一个 kernel 重算 score + 写 dQ + 写 dKV，无法拆「非 atomic dQ」与「atomic dKV」，除非拆成两个 kernel（要权衡重复读 K / 重复算 S）。
4. 没有 key-owned 归约：window key 的可见 query 区间是局部的 `[p, p+W)`；compressed key 需要 key→query 反向邻接（由 selection 构建）。

### 5.4 请给出的优化算法

**C1. 拆反向归属（优先）**

- `backward_q_sink`：query-owned，**非 atomic**，每个 dQ/dSink 元素恰好写一次，无需清零。用保存的 out/lse + 重算可见 S。
- `backward_kv`：head 已归约的 dKV，sharded atomic 或你设计的更好归约。公式必须是 `dKV = dS^T Q + P^T dO`。
- 说明两个 kernel 如何共享/重算 tile，以及何时拆的额外 traffic 会亏。

**C2. Shard / 映射**

给出 S∈{1,2,4,8} 时的缓冲形状 `4*S*B*(T+N)*D` 字节，以及比 `(t/16)%S` 更好的映射（若有），说明对滑窗热点 vs 压缩热点分别意味着什么。不要只说「多测几个 S」。

**C3. KV-owned 归约（完整设计，高风险但请写全）**

- Window：拥有 key `p` 的 TG 遍历 query `t ∈ [p, p+W)`（再滤 segment/pad），寄存器累加后**唯一写** dWindow[p]。
- Compressed：`count → prefix sum → fill` 建 CSR：`key → list of (query, slot_in_selection)`。TG 沿 CSR 归约。邻接表按真实可见数，禁止物化 `[B,T,N,D]`。
- 同一 key 被选多次按多重集合累加，不去重。
- Reuse 层只共享 indices，不共享 P/dS（每层不同）。

**C4. 前向**

保持在线 softmax 的 m/l/rescale 顺序和 FP32 累加。可改 BK 与 NP 的配对，但先列出 threadgroup 字节和同时存活区间，禁止用「编译失败」试探 32 KiB。不要把 BK=32 当成已经证明更快。

---

## 6. 当前算法 D：MoE 外围（不要重写专家 GEMM）

生产路径：MLX `gather_mm` 做 expert GEMM（历史手写 MMA ~5 TFLOPS，原生 ~11–12 TFLOPS，手写更慢，默认关）。

仍可优化的算法：

- 路由仍是现有 argsort，不改概率、不 drop token。
- `route_inverse`：`inverse[order[i]] = i`，token-owned 用 inverse 找该 token 的 K 条 route。
- Combine：`y_token = Σ_{k=1..K} w_k * y_{route(k)}`，加权在寄存器/FP32，**不要 scatter atomic**。
- VJP：`dy_route = dout_token * w`，`dw = ⟨dout_token, y_route⟩`。
- 空专家、热点专家、K=1/6、M 不是 32 倍数都要有定义。

请给出 combine/inverse 的完整并行算法（网格、每线程工作、是否需要清零）。不要设计新的专家 GEMM。

---

## 7. 当前算法 E：mHC / Sinkhorn（残余）

`hc_mult=4`。

**hc_pre + RMSNorm（已融合）** 每 token 一 TG（128 threads）：

```
h[d] = Σ_{c=0..3}  T( T(pre[c]) * x[c,d] )     # 低精度乘加位置必须保持
rstd = rsqrt( mean(h^2) + eps )                 # FP32 归约
y[d] = T( T(h[d] * T(rstd)) * weight[d] )
```

反向保留现有 cast 位置：`gh=T(g*w)`，`dv ∝ -0.5 r^3 ⟨gh,h⟩ / D`，`dh = gh*r + 2 dv * h`，再 `dx[c]=dh*pre[c]`，`dpre[c]=⟨dh,x[c]⟩`。

**hc_post（已融合）** 一线程一个 `(bt,d)`：

```
out[m,d] = post[m]*x[d] + Σ_j comb[m,j]*res[j,d]
dx[d]    = Σ_m g[m,d]*post[m]
dres[j,d]= Σ_m g[m,d]*comb[m,j]
```

`dpost`/`dcomb` 沿 D 的 matmul 在 host 侧小输出上做。

**Sinkhorn 4×4 FP32**：每 token 一线程，16 元矩阵。先按行 `exp(x-max)`（ties 保留 max 的导数），再 `iters` 次交替「行和+eps 归一 / 列和+eps 归一」。反向按保存的 hist/den 逆着除法走。**禁止减少 iters。**

请只在「减少 hc_pre_norm 权梯度 partial 的存/归约」或「相邻逐元素重复读写」里选 **一个**融合边界给出算法。合计占比本来就小，不要设计全层超级 kernel。hc=4、D≤4096、D%32==0 以外必须仍有 fallback 定义。

---

## 8. 语义清单（优化后仍必须成立）

| 项 | 必须 |
|---|---|
| 因果 | 未来 token 对过去零贡献 |
| 滑窗 | 只看 `[t-W+1, t]` |
| 文档/pad | 跨段、pad 位置不可见 |
| CED | 解码层全局 KV 来自边界层，Reuse 不重算 indexer |
| 候选池 | 更深 indexer 的 keep ⊆ 候选块；候选源层自己不受限；最新块钉住 |
| Top-K | `>=` 阈值 + 下标 tiebreak + `sc>-inf` + reach；ties 可 >k |
| Compact 顺序 | 选中下标升序 |
| K=V | dKV 同时含 score 与 value 两条 |
| Sink | 在线 softmax 初值，梯度 `-exp(sink-lse)*δ` |
| Indexer | ReLU 加权和；不可达 -1e30；输出 FP32 |
| MMA | 操作数 T，累加 FP32 |
| VJP 叶子 | 每个 primal 一个 array |
| Sinkhorn | 20 iters（或 cfg）、eps、行列顺序、row-max ties |

允许：FP32 累加顺序变化带来的 bf16 级舍入；atomic 归约顺序造成的同量级差。不允许：可见集合变化、keep 集合变化、丢掉 ties、截断溢出。

---

## 9. 建议你产出的章节标题

直接按这些标题写，每节都是完整算法，不是待办：

1. Indexer：`score_reach` + 空 tile skip（fwd+VJP）
2. Indexer：候选块 gather 打分与可变长 selection
3. 稀疏 Attention：query-owned dQ/dSink + KV atomic/CSR 拆分
4. 稀疏 Attention：window CSR / compressed CSR 的 key-owned dKV
5. exact Top-K+compact 一次扫描
6. MoE token-owned combine（无新 GEMM）
7. （可选）一个 mHC 残余融合

写完即止。不要附录测试计划，不要让读者去跑任何命令。
