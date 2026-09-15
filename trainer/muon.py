"""
Muon 混合优化器（MLX 单设备版）

基于 mlx.optimizers.Muon / mlx.optimizers.MultiOptimizer 实现混合优化器，
分组对齐 DeepSeek-V4.1 技术报告 §2.5（优化）：

- Muon/MuonH：ndim >= 2 且非嵌入/输出头/Engram 检索表/router/零初始化门的
  核心权重（含 muonh 下 3D 堆叠专家的逐专家 NS），weight_decay=0。
  正交化每步全量重算（NS 降频复用已删）。
  **逐头 Muon（head-wise Muon，报告 §2.5 第一条）默认开启**：Query 权重
  （attn.wq_b / indexer.wq_b，形状 [n_heads*head_dim, in]）在施加 NS 前
  reshape 成 (n_heads, head_dim, in) 批量正交化（注意力用 config.head_dim、
  indexer 用 config.index_head_dim）；Key 侧是共享 K=V 的单头 wkv
  （head_dim -> dim），拆不出多头，保持整矩阵 NS，见 BatchedMuon._per_head_dim。
  VIBY_MUONH_PER_HEAD=0 关（消融；旧实现默认关是本仓库历史口径）。
- 嵌入/输出头、router、其余标量参数（ShortConv、1-D gain、KDA A_log/
  dt_bias、attn_gate / KDA g_proj / GatedNorm.gate_up）使用 AdamW；
- SinkhornBalanced（报告 §2.5 第二/三条 + 算法 1）：Engram 检索表、token 嵌入、
  预测头（lm_head / DSpark markov_head 的 2D embed/head）用「Nesterov 动量 +
  Sinkhorn 行/列均衡」更新，有效 lr 缩放 gamma=0.18、**不施加权重衰减**；
  只留一份动量 state（替代 Adam 的一二阶矩）。
- AdamW：归一化层权重（各 RMSNorm 的 weight，**受 wd**）、router（受 wd）、
  其余偏置/缩放因子/3D 专家（**wd=0**）。旧口径「embed 组 wd=0.1 / SFT 0.01、
  标量组 wd=0」随本次分组调整拆开（Sinkhorn 组按报告无 wd）。
- pretrain 下 muon/adam 双基础 lr 与 Adam 组 beta2/eps 由
  trainer.utils.resolve_compute_scaled_hparams 按 Hyperball compute 公式写入 args
"""

import os
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from . import snapshot

_adamw_kernels: dict = {}
_shapeful_adamw_kernels: dict = {}
_stack_mom_kernels: dict = {}
_stack_apply_kernels: dict = {}
_polar_mom_kernels: dict = {}
_sinkhorn_kernels: dict = {}
_ns_core_fns: dict = {}
_fro_norm_kernels: dict = {}
_SINKHORN_FAST_NORM = os.environ.get("VIBY_OPT_SINKHORN_FAST_NORM", "1") == "1"
_ADAM_CONTIG_GRADS = os.environ.get("VIBY_ADAM_CONTIG_GRADS", "0") == "1"
_SINKHORN_FUSED_ROWS = os.environ.get("VIBY_SINKHORN_FUSED_ROWS", "0") == "1"
_SINKHORN_FUSED_PAIRS = os.environ.get("VIBY_SINKHORN_FUSED_PAIRS", "0") == "1"

# shapeful AdamW 缓存上限（按形状+超参），超出回退 shapeless
_SHAPEFUL_CACHE_MAX = 128
# 逐 tensor 路径里多大才走 shapeful 编译（小于它是发射 bound，shapeless 即可）
_SHAPEFUL_MIN_BYTES = 8 << 20

# Newton-Schulz 五次数系数。
# classic: Keller-Jordan 固定系数，5 步迭代共用一组。
_NS_CLASSIC = (3.4445, -4.7750, 2.0315)
# pe: Polar Express 逐迭代 minimax 最优系数（arXiv 2505.16932，
# l=1e-3, degree=5, safety_factor=1.01, cushion=0.02；生成代码逐行移植见
# experiments/probe_polar_express.py）。同 5 步同 GEMM 数下正交化残差约为
# classic 的一半（σ 从 [0.65,1.20] 收到 [0.80,1.13]），但 P12 训练 probe
# 实测 +0.14 nat 变差（500 步 5.465 vs 5.328）——polar 精度不单调决定
# 训练质量，与 arXiv 2606.00371 的核心结论一致。保留作对照，勿作默认。
_PE_COEFFS = (
    (8.237312490495558, -23.15774741455821, 16.68056841144592),
    (4.082441999064829, -2.8930477353325825, 0.525284925697564),
    (3.926347992254658, -2.854746803476532, 0.5318022422894996),
    (3.2982187133085143, -2.424541981026707, 0.48632008358844103),
    (2.297036943455258, -1.636625581259032, 0.4002628455953631),
    (1.8763805351440401, -1.234789657772224, 0.3589188750166845),
)
# cubic5: NVIDIA relaxed cubic（arXiv 2606.00371，u=1.3, l0=7e-3，
# Chen–Chow 闭式推导见 experiments/probe_cubic5.py）。每步 2 GEMM
# （去掉 Gram 平方项，比 quintic 省 1/3），松弛目标带 [0.7,1.3]。
# P13 probe 实测：质量不差反略好（500 步 5.245 vs classic 5.328），
# 优化器墙钟 −19.5%（bench_muonh.py），步速 +4.3%——现为默认。
_CUBIC5_COEFFS = (
    (3.3656576453026874, -3.3420991881255913),
    (2.5744352167003366, -1.4957376498043133),
    (2.536896217781012, -1.4312570057421792),
    (2.441890631764536, -1.276403977018072),
    (2.2230472113940625, -0.9630649572307908),
)
# cubic5b05: 同 cubic5 推导但保护下界 l0=0.05（膝盖上移：σ=1e-3 只抬到
# 0.050 vs cubic5 的 0.119，σ≥0.05 全推满 u=1.3）。自研假设「小 σ 抬升
# 有害、膝盖位置存在内点最优」的检验点。同 5 步 10 GEMM，纯质量变量。
# P16 probe 实测 5.223@500（vs cubic5 5.245，−0.02 在噪声内）——
# 预注册规则（≤基线+0.05 即翻默认）下为默认；cubic5 一键可回。
_CUBIC5B05_COEFFS = (
    (3.292184656117614, -3.1279664191141223),
    (2.4308928122354376, -1.259235477762661),
    (2.2001298193259866, -0.9335862448253945),
    (1.832672820412407, -0.5395905557152266),
    (1.5614350757473685, -0.3337192936368423),
)
# cubic5b10: 膝盖扫描第三个点 l0=0.10（σ=1e-3 抬到 0.034，σ≥0.10 推满
# u=1.3）。P17 实测 5.314@500，比 cubic5/b05 差 +0.07~0.09——平台右缘
# 在 (0.05, 0.10) 之间，膝盖扫描完毕。
_CUBIC5B10_COEFFS = (
    (3.205780252454467, -2.8880903175265464),
    (2.275081454662965, -1.0322867987423627),
    (1.9300875431976465, -0.6302901874507894),
    (1.607262535311003, -0.3639736863869031),
    (1.5059090631553025, -0.29936826730207244),
)
# cubic5b002: U 型左臂检验点 l0=0.002、7 步调度（必须配 --muon_ns_steps 7；
# 5 步对 l0=0.002 不收敛，l_final 仅 0.28，会把中段 σ=0.05–0.3 也压到
# 0.3–0.8，混淆归因）。7 步后响应：σ=1e-3→0.62、2e-3→1.08、≥3e-3→1.3
# 压平——膝盖 0.001–0.002 明显低于 P-1 实测 σ*≈0.003–0.01，且 σ* 以上
# 保持全白化，干净隔离「膝盖低于边界」单变量。预测：若 U 型左臂成立，
# 显著差于 b05；若 ≈b05 或更好，则规则简化为「膝盖 ≤ σ* 即可」。
_CUBIC5B002_COEFFS = (
    (3.3741198987406307, -3.3673716858821225),
    (2.591324399165892, -1.5253688570935597),
    (2.580546792728205, -1.5064153695055407),
    (2.552648593476117, -1.4580842534234484),
    (2.4813270357515167, -1.3392496608733755),
    (2.309499639073599, -1.0798494680744026),
    (1.9813908930620336, -0.6818989318955107),
)
# cubic5b07: 批大小减半证伪用的 l0=0.0707≈0.05·√2 点（5 步收敛，
# l_final=1.30）。bs6/accum2（有效批减半）下若持续边界 σ*∝1/√B 成立，
# 最优膝盖应从 0.05 上移到 0.07：bs6+b07 应优于 bs6+b05。
_CUBIC5B07_COEFFS = (
    (3.25770089587558, -3.030701363732049),
    (2.3666790395365362, -1.1620574945874727),
    (2.076766206523265, -0.7851856382372985),
    (1.7086668198169122, -0.4373022186294428),
    (1.523071381964533, -0.30972074581561343),
)
# cubic5b01 / cubic5b005: 稠密台（P26/P27：σ*≈3e-4–1e-3，比 MoE 低一个
# 量级）的膝盖网格加蜜点，插在 b002（knee50≈5e-4，P26 最优）与 b05
# （≈0.011，P26 最差）之间夹逼稠密最优。b01：l0=0.01、5 步收敛
# （l_final=0.95）、knee50≈0.005；b005：l0=0.005、须 6 步（5 步 l_final
# 仅 0.49 不收敛，配 --muon_ns_steps 6）、knee50≈0.002。
_CUBIC5B01_COEFFS = (
    (3.3605708285105815, -3.326968447193923),
    (2.5643010071185848, -1.478143273289143),
    (2.5109201353338024, -1.3877403646632225),
    (2.378581203715221, -1.1796779733346807),
    (2.098300257941159, -0.809864616184623),
)
_CUBIC5B005_COEFFS = (
    (3.3690449564836613, -3.352200150726263),
    (2.581191794533407, -1.507545227156317),
    (2.5543143012923104, -1.4609404956133534),
    (2.485537132549127, -1.3460781958399577),
    (2.319115009105853, -1.0933932347556181),
    (1.9965212763574187, -0.6976399634714244),
)
# frac: 自研正则化幂律谱变换 t(σ) = σ·(σ²+s)^((p−1)/2) 的 Chebyshev-on-Gram
# 单发实现（非迭代逼近；Muon-p 的不可能定理只排除固定单变量**迭代**，
# 单发 minimax/Chebyshev 逼近不受限）。两区结构：σ≪√s 线性地板（不硬拉
# 极小 σ——实测小 σ 抬升有害：PE 0.86→+0.14 nat、classic 0.47→基准、
# cubic5 0.12→−0.08 nat），σ≫√s 幂律 σ^p（保大 σ 序信息）。
# 系数是 h(t)=(t+s)^q 在 t∈[0,1] 的 T_j(2t−1) 基 Chebyshev 拟合
# （experiments/probe_frac_design.py），系数小且衰减，bf16 安全；
# deg-k 成本 k+1 个主 GEMM（cubic5 为 10，classic 为 15）。
#   frac50: p=0.5, s=1e-2, deg 6, 拟合误差 3.1e-2（8 GEMM）
#   frac25: p=0.25, s=1e-2, deg 8, 拟合误差 4.5e-2（10 GEMM）
# bf16 矩阵级保真（probe_frac_design.realized_check）：max|Δσ| 2.2e-2 /
# 3.4e-2，主体相对误差中位 2.8e-2 / 3.9e-2，单调性保持。
# **否定（P14b/P15b，14 分钟 probe @500）**：frac50 5.523（+0.28）、
# frac25 5.441（+0.20）vs cubic5 5.245。幂律主体不保平和「保大 σ 序
# 信息」在此尺度上是有害而非有益的——bulk 必须压平； frac 末尾必须把
# ||D||_F 钉回 √min(r,c)（P14 未钉回 +0.78 nat 的范数混杂教训）。
# 家族整体否定，勿以「换个 p/正则」重试同族。
_FRAC_PRESETS = {
    "frac50": (1.427143, -0.641426, 0.339964, -0.212826, 0.143354, -0.100648, 0.072585),
    "frac25": (
        1.794163,
        -1.241522,
        0.730512,
        -0.485103,
        0.340377,
        -0.246541,
        0.182323,
        -0.136827,
        0.103814,
    ),
}


def _cubic_schedule(l0, u=1.3, min_steps=4, max_steps=8):
    """closed-form relaxed cubic（arXiv 2606.00371 §2.2，同 probe_cubic5.py）
    + 收敛步数自选：l_final≥0.7 的最少步数（l0 越小需要越多步把下界
    推进松弛带）。返回 (coeffs, steps)。"""
    from math import sqrt

    def gen(steps):
        coeffs = []
        lo, r = l0, 1.0
        for t in range(steps):
            if t > 0:
                r = u
            k2 = (r * r + r * lo + lo * lo) / 3
            alpha = 1.0 / sqrt(k2)
            a, b = 1.5 * u * alpha, -0.5 * u * alpha**3
            coeffs.append((a, b))
            lo = a * lo + b * lo**3
        return coeffs, lo

    for steps in range(min_steps, max_steps + 1):
        coeffs, lfin = gen(steps)
        if lfin >= 0.7:
            return coeffs, steps
    return gen(max_steps)


# EdgeCubic（P-3）：逐形状组膝盖 = P-1 实测持续边界 σ*（F-归一单位，
# probe_p18_snap）×4.5（经验 knee50/l0 比）。规则：σ* 以下≈0 权、
# σ* 以上压平——逐组落地「膝盖=σ*」，替代全局统一 l0=0.05（b05 的
# knee50≈0.011 对 σ* 低的组过度注入非持续方向）。未测量的形状回退
# 0.05（=现默认 b05，保守）。env VIBY_MUONH_EDGE=1 启用（probe 档）。
_EDGE_L0_2D = {
    (768, 768): 0.009,  # attention q/k/v 类，σ*≈0.001–0.003
    (768, 384): 0.014,
    (384, 768): 0.014,  # o_proj 类，σ*≈0.003
    (96, 768): 0.009,
    (768, 96): 0.009,  # KDA g_a/g_b，σ*≈0.001–0.003
    (128, 768): 0.007,  # kv_up 类：视野内无坍缩（σmin=0.024 全持续），压平即可
    (768, 128): 0.02,  # gate_down（P-1 时因钉零 bug 不可测），保守中值
    (768, 1536): 0.0025,
    (1536, 768): 0.0025,  # 共享 FFN，σ*≈3e-4–1e-3
    (768, 2304): 0.0025,
    (2304, 768): 0.0025,  # MTP 大矩阵，同上
}
_EDGE_L0_STACK = {
    (768, 384): 0.027,  # 路由专家 gu，σ*≈0.003–0.01（较软）
    (384, 384): 0.032,  # 路由专家 dw，σ*≈0.005–0.01
    (384, 768): 0.014,  # lat_up/down 类按 o_proj 口径
}
_EDGE_L0_DEFAULT = 0.05


def _fro_norm(x):
    """逐矩阵 Frobenius 范数 (N,r,c)→(N,1,1) f32，Metal 融合 kernel。

    直接读 bf16、f32 寄存器平方累加（thread 内 f32 链 + simd_sum），
    一次读完；替代 `mx.linalg.norm(x.astype(f32))` 的全量 f32 物化
    （(2304,640,384) 专家栈实测 12ms → ~2.5ms）。数值上与 f32 参考
    的差只有归约顺序（~1e-6 相对），而 bf16 直接 square 的 1.9% 误差
    对 hyperball 投影不可接受（范数随机游走漂移）。

    两阶段：kernel 输出 (L, NSLICE) 部分和（L = 前导维展平），再
    mx.sum+sqrt（微小）。输入非连续时 reshape 会先拷贝，仍正确。
    """
    lead, r, c = x.shape[:-2], x.shape[-2], x.shape[-1]
    L = 1
    for d in lead:
        L *= d
    M = r * c
    SLICE = 16384
    NT = 256
    nslice = (M + SLICE - 1) // SLICE
    key = (M, nslice)
    kern = _fro_norm_kernels.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"fro_norm_{M}_{nslice}",
            input_names=["x"],
            output_names=["out"],
            source="""
    constexpr uint NT = 256;
    uint tg = threadgroup_position_in_grid.y;
    uint n = tg / NSLICE;
    uint sl = tg % NSLICE;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float sh[NT / 32];
    uint base = n * M + sl * SLICE;
    uint end = min(base + (uint)SLICE, (n + 1) * (uint)M);
    float acc = 0.0f;
    for (uint i = base + tid; i < end; i += NT) {
        float v = float(x[i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (tid % 32 == 0) sh[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint k = 0; k < NT / 32; k++) s += sh[k];
        out[tg] = s;
    }
""",
        )
        _fro_norm_kernels[key] = kern
    xf = x.reshape(L, M)
    (part,) = kern(
        inputs=[xf],
        template=[("M", M), ("SLICE", SLICE), ("NSLICE", nslice)],
        grid=(NT, L * nslice, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(L * nslice,)],
        output_dtypes=[mx.float32],
    )
    return mx.sqrt(mx.sum(part.reshape(L, nslice), axis=-1)).reshape(*lead, 1, 1)


def _fro_norm_auto(x):
    """bf16 走 Metal 融合 kernel；其余 dtype 回退 mx.linalg.norm。"""
    if x.dtype == mx.bfloat16:
        return _fro_norm(x)
    return mx.linalg.norm(x.astype(mx.float32), axis=(-2, -1), keepdims=True)


def _ns_core_fn(kind, bf16, steps, coeff="classic"):
    """NS 核心的 mx.compile 版：把 5 步迭代里的逐元素链（B=bA+cA²、
    X=aX+B@X、入口 f32 范数+归一化）融进少量 kernel，GEMM 不变。

    数学上与 _ns5/_ns5_gram 的 eager 实现逐项相同（同样的算子顺序、
    同样的弱类型提升），只是 compile 消除中间物化。tr 转置规则、
    eye 初始化、Gram 的 Z/Q/R 递推全部照抄。compiled fn 只能返回数组，
    tr 标志由调用方按形状重新推导（确定性）。

    coeff="pe" 时换用 Polar Express 逐迭代系数（见 _PE_COEFFS 注释），
    归一化带 1.01 安全边际；GEMM 数与 classic 完全相同。
    coeff="cubic5" 时换用 NVIDIA relaxed cubic（见 _CUBIC5_COEFFS
    注释），每步少一个 Gram 平方 GEMM（quintic 的 2/3）。
    coeff 为 _FRAC_PRESETS 键（frac50/frac25）时走自研 Chebyshev-on-Gram
    单发谱变换（见 _FRAC_PRESETS 注释），与 steps 无关。
    """
    key = (kind, bf16, steps, coeff)
    fn = _ns_core_fns.get(key)
    if fn is not None:
        return fn
    if coeff in _FRAC_PRESETS:
        ccoefs = _FRAC_PRESETS[coeff]

        @mx.compile
        def fn(X):
            dt0 = X.dtype
            if bf16 and dt0 != mx.bfloat16:
                X = X.astype(mx.bfloat16)
            nrm = _fro_norm_auto(X)  # 范数是转置不变量
            tr = X.shape[-2] > X.shape[-1]
            if tr:
                X = X.swapaxes(-1, -2)
            X = X / (nrm + 1e-7).astype(X.dtype)
            G = X @ X.swapaxes(-1, -2)
            eye = mx.eye(G.shape[-1], dtype=X.dtype)
            Ah = 2 * G - eye  # Chebyshev 变量映射 t→2t−1
            T_prev, T_cur = eye, Ah
            H = ccoefs[0] * T_prev + ccoefs[1] * T_cur
            for j in range(1, len(ccoefs) - 1):
                T_next = 2 * (Ah @ T_cur) - T_prev
                H = H + ccoefs[j + 1] * T_next
                T_prev, T_cur = T_cur, T_next
            Y = H @ X
            if tr:
                Y = Y.swapaxes(-1, -2)
            # 谱变换只定形状，尺度钉回 NS 输出口径 ||D||_F = √min(r,c)。
            # 必不可少：frac 映射本征范数只有 0.13~0.21×√n，不归一则有效
            # lr 亏空 5~8 倍（P14 教训：+0.78 nat 主要来自范数亏空而非形状）。
            yn = _fro_norm_auto(Y)
            s_target = float(min(Y.shape[-2], Y.shape[-1])) ** 0.5
            Y = Y * (s_target / (yn + 1e-12)).astype(Y.dtype)
            return Y.astype(dt0) if Y.dtype != dt0 else Y

        _ns_core_fns[key] = fn
        return fn
    cubic = coeff in (
        "cubic5",
        "cubic5b05",
        "cubic5b10",
        "cubic5b002",
        "cubic5b07",
        "cubic5b01",
        "cubic5b005",
    )
    edge = coeff.startswith("edge:")
    if coeff == "pe":
        coeffs = list(_PE_COEFFS[:steps])
        while len(coeffs) < steps:
            coeffs.append(coeffs[-1])  # 论文口径：超出后重复最后一组
        norm_scale = 1.01
    elif edge:
        # EdgeCubic 逐形状膝盖：l0 由调用方按组解析（见 _EDGE_L0_*），
        # steps 取该 l0 的收敛步数（_cubic_schedule），c=0 同 cubic。
        ecoefs, esteps = _cubic_schedule(float(coeff[5:]))
        assert esteps == steps, (coeff, esteps, steps)
        coeffs = [(a, b, 0.0) for a, b in ecoefs]
        norm_scale = 1.0
        cubic = True
    elif cubic:
        table = {
            "cubic5": _CUBIC5_COEFFS,
            "cubic5b05": _CUBIC5B05_COEFFS,
            "cubic5b10": _CUBIC5B10_COEFFS,
            "cubic5b002": _CUBIC5B002_COEFFS,
            "cubic5b07": _CUBIC5B07_COEFFS,
            "cubic5b01": _CUBIC5B01_COEFFS,
            "cubic5b005": _CUBIC5B005_COEFFS,
        }[coeff]
        coeffs = [(a, b, 0.0) for a, b in table[:steps]]
        while len(coeffs) < steps:
            coeffs.append(coeffs[-1])
        norm_scale = 1.0
    else:
        coeffs = [_NS_CLASSIC] * steps
        norm_scale = 1.0  # nrm*1.0 在 IEEE 下恒等，classic 路径逐位不变

    if kind == "gram":

        @mx.compile
        def fn(X):
            dt0 = X.dtype
            if bf16 and dt0 != mx.bfloat16:
                X = X.astype(mx.bfloat16)
            nrm = _fro_norm_auto(X)  # 范数是转置不变量，先算保住连续性
            tr = X.shape[-2] > X.shape[-1]
            if tr:
                X = X.swapaxes(-1, -2)
            X = X / (nrm * norm_scale + 1e-7).astype(X.dtype)
            R = X @ X.swapaxes(-1, -2)
            eye = mx.eye(R.shape[-1], dtype=X.dtype)
            Q = None
            for t in range(steps):
                a_, b_, c_ = coeffs[t]
                if cubic:
                    Z = a_ * eye + b_ * R
                else:
                    Z = a_ * eye + b_ * R + c_ * (R @ R)
                Q = Z if Q is None else Q @ Z
                if t < steps - 1:
                    R = Z @ R @ Z
            Y = Q @ X
            if tr:
                Y = Y.swapaxes(-1, -2)
            return Y.astype(dt0) if Y.dtype != dt0 else Y

    else:

        @mx.compile
        def fn(X):
            dt0 = X.dtype
            if bf16 and dt0 != mx.bfloat16:
                X = X.astype(mx.bfloat16)
            n = _fro_norm_auto(X)
            tr = X.shape[-2] > X.shape[-1]
            if tr:
                X = X.swapaxes(-1, -2)
            X = X / (n * norm_scale + 1e-7).astype(X.dtype)
            for t in range(steps):
                a_, b_, c_ = coeffs[t]
                A = X @ X.swapaxes(-1, -2)
                if cubic:
                    X = a_ * X + b_ * (A @ X)
                else:
                    B = b_ * A + c_ * (A @ A)
                    X = a_ * X + B @ X
            if tr:
                X = X.swapaxes(-1, -2)
            return X.astype(dt0) if X.dtype != dt0 else X

    _ns_core_fns[key] = fn
    return fn


def _stack_mom_kernel(m, nest, wd):
    """堆叠组动量/nesterov/wd 融合。超参是 python float，lr 不在这里。"""
    key = (m, nest, wd)
    fn = _stack_mom_kernels.get(key)
    if fn is None:

        @partial(mx.compile, shapeless=True)
        def fn(G, P, V):
            if wd != 0:
                G = G + wd * P
            V = m * V + (1.0 - m) * G
            U = G * (1.0 - m) + V * m if nest else V
            return U, V

        _stack_mom_kernels[key] = fn
    return fn


def _stack_apply_kernel(hyperball, apply_wd=0.0, cautious=False):
    """P - lr*X，可选范数球投影与（cautious）解耦衰减。

    hyperball 版用 shapeful compile：内部 _fro_norm 的 Metal kernel
    grid 依赖具体形状，shapeless 追踪会把首次形状固化给所有组。
    非投影版保持 shapeless（纯逐元素，一个图通吃所有形状）。

    apply_wd != 0：衰减从动量 kernel（耦合式 G += wd*P）移到此处做
    解耦衰减（polar-ema 下动量建在极因子空间，耦合衰减无意义，只能
    移到这里）。cautious=True 时再按 cautious weight decay（CWD，Chen
    et al. 2025，arXiv 2510.12402；亦 speedrun 纪录组件；
    autoresearch-mlx 战役 2026-09-01 双 regime keep：
    d4@467 −0.0090 / d2@1070 −0.0034 bpb）加掩码——只在
    sign(X)==sign(P) 的坐标上衰减：正在沿损失方向移动的坐标不被
    无谓收缩，衰减税只落在被反向推（过冲/噪声主导）的坐标上。
    掩码符号沿用参考实现的约定（作用于被减去的方向 X）。
    """
    key = (hyperball, apply_wd, cautious)
    fn = _stack_apply_kernels.get(key)
    if fn is None:
        if hyperball:

            @mx.compile
            def fn(P, X, lr):
                # 范数球半径取衰减前的 P（冻结口径是「上一步建立的本体
                # 半径」，衰减本身不该先缩它）。
                n0 = _fro_norm_auto(P)
                if apply_wd != 0:
                    if cautious:
                        mask = ((X * P) >= 0).astype(P.dtype)
                        P = P * (1 - lr * apply_wd * mask)
                    else:
                        P = P * (1 - lr * apply_wd)
                NP = P - lr * X
                n1 = _fro_norm_auto(NP)
                # 零初始化矩阵（如 GatedNorm.gate_up）范数球半径为 0，
                # 直接投影会把参数永久钉死在 0（P18 实测：gate 全程恒等、
                # gate_down 梯度恒零）。n0==0 的矩阵当步跳过投影，半径由
                # 首个更新建立，下一步起正常冻结。
                scale = mx.where(n0 > 0, n0 / mx.maximum(n1, 1e-12), mx.ones_like(n1))
                NP = NP * scale.astype(P.dtype)
                return NP

        else:

            @partial(mx.compile, shapeless=True)
            def fn(P, X, lr):
                if apply_wd != 0:
                    if cautious:
                        mask = ((X * P) >= 0).astype(P.dtype)
                        P = P * (1 - lr * apply_wd * mask)
                    else:
                        P = P * (1 - lr * apply_wd)
                return P - lr * X

        _stack_apply_kernels[key] = fn
    return fn


def _polar_mom_kernel(m):
    """polar-ema 动量（autoresearch-mlx 战役 bar 组件）：EMA 建在极因子
    空间而非梯度空间——O = NS(G) 先正交化，V ← m·V + O，Nesterov 读出
    U = (1−m)·O + m·V。NS 每步全量重算（作用于原始梯度而非动量），
    GEMM 数与标准路径相同，只是动量住的空间换了。"""
    fn = _polar_mom_kernels.get(m)
    if fn is None:

        @partial(mx.compile, shapeless=True)
        def fn(O, V):
            V = m * V + O
            U = (1.0 - m) * O + m * V
            return U, V

        _polar_mom_kernels[m] = fn
    return fn


def _sinkhorn_body(
    beta, gamma, K, eps, tau, n, fast_norm=False, fused_rows=False, fused_pairs=False
):
    """算法 1（DeepSeek-V4.1 技术报告 §2.5）的一步更新（未编译）。

    逐行对照（编号 = 报告算法 1 的行号）：
      1  M_t ← β·M_{t-1} + (1−β)·G_t
      2  Ĝ_t ← β·M_t + (1−β)·G_t                ⊳ Nesterov 动量
      3  ρ_i ← ‖Ĝ_{t,i,:}‖₂，ρ̄ ← (1/m)·Σ_i ρ_i
      4  U⁽⁰⁾ ← Ĝ_t
      5  ρ_i ≤ τ·ρ̄ 的行置 0                     ⊳ 掩蔽近零行
      6-16 for k = 1..K：k 奇数 → 逐行 L2 归一化（行 7-10）；
           k 偶数 → 逐列 L2 归一化（行 11-14）；分母一律 +ε
      17 Δ_t ← √n·U⁽ᴷ⁾                          ⊳ 单位行 ℓ₂ 范数 → 单位行 RMS
      18 η̃_t ← γ·η_t                            ⊳ 匹配 Adam 的更新幅度
      19 W_{t+1} ← W_t − η̃_t·Δ_t                ⊳ 无权重衰减

    K 必须为奇数：最后一步落在行归一化上，U⁽ᴷ⁾ 的每一行才是单位 ℓ₂ 范数，
    乘 √n 后才满足式 (7) 的行 RMS≈1（列方向只被交替压到同一尺度）。
    n 是隐维度（= W 的最后一维，嵌入表/预测头口径下即模型 dim 或
    markov rank），作为 python 常量烘进图（每形状一个编译体）。
    范数一律 f32 累加（照 _fro_norm 的数值口径，bf16 参数也不会把
    归一化推向随机方向），除法在权重 dtype 上做。
    全程 mx 算子、无 host sync（可 mx.compile）。
    """
    sqrt_n = float(n) ** 0.5

    def fn(p, g, m, lr):
        # 行 1：动量（一阶矩，state 里唯一的 buffer）
        m = beta * m + (1.0 - beta) * g
        # 行 2：Nesterov 读出
        gh = beta * m + (1.0 - beta) * g
        # 行 3：逐行 ℓ₂ 范数 ρ_i 与均值 ρ̄（f32）
        if fast_norm:
            from .fast_norm import square_sum

            rho = mx.sqrt(square_sum(gh, 1))
        else:
            g32 = gh.astype(mx.float32)
            rho = mx.sqrt(mx.sum(g32 * g32, axis=-1, keepdims=True))
        rho_bar = mx.mean(rho)
        # 行 4 + 5：U⁽⁰⁾ ← Ĝ_t，并把 ρ_i ≤ τ·ρ̄ 的行掩蔽为 0
        U = mx.where(rho <= tau * rho_bar, mx.zeros_like(gh), gh)
        # 行 6-16：K 步交替行/列 L2 归一化（奇数步行、偶数步列）
        for k in range(1, K + 1):
            if fast_norm and fused_pairs and K % 2:
                from .sinkhorn_pairs import first_row_from_norm, column_then_row

                if k == 1:
                    U = first_row_from_norm(gh, rho, rho <= tau * rho_bar, eps)
                elif k % 2 == 0:
                    U = column_then_row(U, eps)
                continue
            if fast_norm and fused_rows and k % 2:
                from .sinkhorn_rows import row_normalize

                U = row_normalize(U, eps)
                continue
            if fast_norm:
                nrm = mx.sqrt(square_sum(U, 1 if k % 2 else 0))
            elif k % 2 == 1:
                U32 = U.astype(mx.float32)
                # 行 7-10：U⁽ᵏ⁾_{i,:} ← U⁽ᵏ⁻¹⁾_{i,:} / (‖U⁽ᵏ⁻¹⁾_{i,:}‖₂ + ε)
                nrm = mx.sqrt(mx.sum(U32 * U32, axis=-1, keepdims=True))
            else:
                U32 = U.astype(mx.float32)
                # 行 11-14：U⁽ᵏ⁾_{:,j} ← U⁽ᵏ⁻¹⁾_{:,j} / (‖U⁽ᵏ⁻¹⁾_{:,j}‖₂ + ε)
                nrm = mx.sqrt(mx.sum(U32 * U32, axis=-2, keepdims=True))
            U = U / (nrm + eps).astype(U.dtype)
        # 行 17：Δ_t ← √n·U⁽ᴷ⁾（单位行 ℓ₂ → 单位行 RMS）
        delta = (U * sqrt_n).astype(p.dtype)
        # 行 18 + 19：W ← W − (γ·η_t)·Δ_t，无权重衰减
        return p - (gamma * lr) * delta, m

    return fn


def _sinkhorn_kernel(beta, gamma, K, eps, tau, n):
    """按 (β, γ, K, ε, τ, n) 缓存的 mx.compile Sinkhorn 一步更新。

    超参是 python float 而非 traced 输入：与 _adamw_kernel 同样的理由
    （MLX 弱类型提升下 float×bf16 保持 bf16，换成 f32 数组会抬 dtype）；
    lr 随 scheduler 每步变，作数组入参。
    """
    key = (
        beta,
        gamma,
        K,
        eps,
        tau,
        n,
        _SINKHORN_FAST_NORM,
        _SINKHORN_FUSED_ROWS,
        _SINKHORN_FUSED_PAIRS,
    )
    fn = _sinkhorn_kernels.get(key)
    if fn is None:
        if _SINKHORN_FUSED_ROWS:
            from .sinkhorn_rows import _kernel

            _kernel()
        if _SINKHORN_FUSED_PAIRS:
            from .sinkhorn_pairs import _kernel

            _kernel()
        fn = partial(mx.compile, shapeless=not _SINKHORN_FAST_NORM)(
            _sinkhorn_body(*key)
        )
        _sinkhorn_kernels[key] = fn
    return fn


def _normuon_tail(X, v, beta2):
    """NorMuon 行 RMS 尾（arXiv 2510.05491）：神经元粒度的自适应。
    X (…, r, c)，v (…, r) 为逐行二阶矩 EMA（None 时以首个样本播种）。
    归一后整体 renorm 回尾前 F 范数——自适应性只重配行间预算，lr 的
    标定语义不变。返回 (X_new, v_new)。"""
    n0 = _fro_norm_auto(X)
    row_sq = mx.mean((X * X).astype(mx.float32), axis=-1)
    v_new = row_sq if v is None else beta2 * v + (1.0 - beta2) * row_sq
    Xn = X / (mx.sqrt(v_new) + 1e-10)[..., None].astype(X.dtype)
    n1 = _fro_norm_auto(Xn)
    Xn = Xn * (n0 / mx.maximum(n1, 1e-12)).astype(X.dtype)
    return Xn, v_new


def _adamw_body(b1, b2, eps, wd, bias_correction, cautious=False):
    """AdamW 更新链（未编译）。shapeless / shapeful 两个编译包装共用同一份
    函数体，保证两条路径逐位一致。

    cautious=True（cautious weight decay，CWD：Chen et al. 2025，
    arXiv 2510.12402；autoresearch-mlx 战役双 regime keep）：衰减加掩码，只落在与 Adam
    步同号的坐标上（该坐标正被损失利用则不收 decay 税）。wd==0 时
    与原路径逐位一致。"""

    def fn(p, g, m, v, lr, step):
        dtype = p.dtype
        p, g, m, v, lr = (x.astype(mx.float32) for x in (p, g, m, v, lr))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * mx.square(g)
        if cautious and wd != 0:
            # 掩码需要 Adam 步方向（不含 lr，符号等价）；bias 修正的
            # 系数为正，不影响符号。
            udir = m / (mx.sqrt(v) + eps)
            mask = ((udir * p) >= 0).astype(p.dtype)
            p = p * (1 - lr * wd * mask)
        else:
            # 算子顺序照抄 optim.AdamW.apply_single → Adam.apply_single：
            # 先解耦 wd 缩放参数，再减去 Adam 步
            p = p * (1 - lr * wd)
        if bias_correction:
            c1 = (lr / (1 - b1**step)).astype(p.dtype)
            c2 = mx.rsqrt(1 - b2**step).astype(p.dtype)
            p = p - (c1 * m) / (mx.sqrt(v) * c2 + eps)
        else:
            p = p - lr * m / (mx.sqrt(v) + eps)
        return p.astype(dtype), m, v

    return fn


def _adamw_kernel(b1, b2, eps, wd, bias_correction, cautious=False):
    """(b1,b2,eps,wd,bias_correction,cautious) 对应的 mx.compile 融合 AdamW 更新。

    超参作为编译缓存键，lr/step 是运行时输入。2026-09-14 起更新算术与 m/v
    显式 FP32，最后仅参数转回原 dtype，不再复现旧 BF16 EMA 的逐位舍入。

    bias_correction 必须开：compute 公式的 beta2/eps 是在带修正的 Adam
    （optax 口径）上拟合的；不修正时有效步长带一个随时间衰减的放大因子
    (1-b1^t)/sqrt(1-b2^t)，beta2=0.9998 时前百余窗口放大 7~15 倍
    （r081_gqa_qb run 在 warmup 末端把 KDA 衰减参数推进 exp(-gc) 溢出区，
    前向永久 NaN）。

    shapeless 版：一个图通吃所有形状（小 tensor / 堆叠组用）。
    """
    key = (b1, b2, eps, wd, bias_correction, cautious)
    fn = _adamw_kernels.get(key)
    if fn is None:
        fn = partial(mx.compile, shapeless=True)(_adamw_body(*key))
        _adamw_kernels[key] = fn
    return fn


def _adamw_kernel_shapeful(
    b1, b2, eps, wd, bias_correction, shape, dtype, cautious=False
):
    """shapeful 版：按具体形状固化 grid 的编译 AdamW。

    shapeless kernel 在 100M 级大 tensor（堆叠专家栈）上只跑到
    ~250GB/s；shapeful 固化形状后 ~410GB/s（同进程对拍逐位一致）。
    只给大 tensor 用：小 tensor 是发射 bound 且形状杂，shapeful 缓存
    会膨胀；超过 _SHAPEFUL_CACHE_MAX 种形状回退 shapeless。
    """
    key = (b1, b2, eps, wd, bias_correction, tuple(shape), dtype, cautious)
    fn = _shapeful_adamw_kernels.get(key)
    if fn is None:
        if len(_shapeful_adamw_kernels) >= _SHAPEFUL_CACHE_MAX:
            return _adamw_kernel(b1, b2, eps, wd, bias_correction, cautious)
        fn = mx.compile(_adamw_body(b1, b2, eps, wd, bias_correction, cautious))
        _shapeful_adamw_kernels[key] = fn
    return fn


def _tree_get(tree, path: str):
    """按 mlx.utils.tree_flatten 的点分路径取叶子。"""
    cur = tree
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, (list, tuple)) else cur[p]
    return cur


class FusedAdamW(optim.AdamW):
    """AdamW 的 mx.compile 融合版：把 10 个逐元素算子收成一个 kernel。

    动机：MoE 细粒度化后堆叠专家权重占了绝大部分参数（E=288/I=104/8 层
    时 621M / 655M），逐算子实现每步要把 p/g/m/v 反复读写十来遍，实测
    88ms —— 而按 8.7GB 的必要访存量算，带宽上限只需 ~22ms。融合后单次
    读写即可。以上是历史 BF16 EMA 性能记录；2026-09-14 精度修复后，m/v 和
    更新算术保持 FP32，不能沿用旧访存量、带宽或逐位等价结论。

    同形状张量再堆成 (N,·) 一次更新：标量组 ~100 个小核变成每个形状
    1 次发射。单组超过 256MB 仍逐张量，避免再物化一份大栈。

    逐张量路径里的大 tensor（>=8MB，堆叠专家栈）改用 shapeful 编译：
    shapeless kernel 的动态形状索引在 100M 级 tensor 上只跑到
    ~250GB/s，shapeful 固化 grid 后 ~410GB/s（同进程对拍逐位一致，
    见 experiments/prof_target_step.py 的 [adamw kernel] 探针）。
    """

    _STACK_BYTES = 256 << 20

    def __init__(self, *args, cautious=False, **kwargs):
        # cautious weight decay（CWD，Chen et al. 2025，arXiv
        # 2510.12402；autoresearch-mlx 战役
        # 2026-09-01 双 regime keep）：掩码式解耦衰减，只衰减与 Adam 步
        # 同号的坐标。wd==0 时逐位不变；create_mixed_optimizer 默认开启
        # （VIBY_ADAM_CAUTIOUS=0 回退）。
        self.cautious = cautious
        super().__init__(*args, **kwargs)

    def init_single(self, parameter: mx.array, state: dict):
        # beta2 can be ~0.99975: BF16 EMA plateaus long before convergence.
        state["m"] = mx.zeros(parameter.shape, mx.float32)
        state["v"] = mx.zeros(parameter.shape, mx.float32)

    def _kernel(self, shape, dtype):
        """按形状选编译变体：大 tensor shapeful（带宽），小 tensor shapeless。"""
        nbytes = int(dtype.size)
        for s in shape:
            nbytes *= int(s)
        if nbytes >= _SHAPEFUL_MIN_BYTES:
            return _adamw_kernel_shapeful(
                self.betas[0],
                self.betas[1],
                self.eps,
                self.weight_decay,
                self.bias_correction,
                shape,
                dtype,
                self.cautious,
            )
        return _adamw_kernel(
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.bias_correction,
            self.cautious,
        )

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(mx.float32)
        p, m, v = self._kernel(gradient.shape, gradient.dtype)(
            parameter, gradient, state["m"], state["v"], lr, self.step
        )
        state["m"] = m
        state["v"] = v
        return p

    def apply_gradients(self, gradients: dict, parameters: dict):
        if not self._initialized:
            self.init(gradients)
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)
        self.state["step"] = self.step + 1

        fn = _adamw_kernel(
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.bias_correction,
            self.cautious,
        )
        flat_g = tree_flatten(gradients)
        if _ADAM_CONTIG_GRADS:
            flat_g = [
                (
                    path,
                    mx.contiguous(g)
                    if g.size * g.dtype.size >= _SHAPEFUL_MIN_BYTES
                    else g,
                )
                for path, g in flat_g
            ]
        flat_p = dict(tree_flatten(parameters))
        gmap = dict(flat_g)
        groups: dict = {}
        for path, g in flat_g:
            groups.setdefault((g.shape, g.dtype), []).append(path)

        new_p = []
        for (shape, dt), paths in groups.items():
            lr = self.learning_rate.astype(mx.float32)
            nbytes = max(int(dt.size), 4) * len(paths)
            for s in shape:
                nbytes *= int(s)
            if len(paths) == 1 or nbytes > self._STACK_BYTES:
                fn_t = self._kernel(shape, dt)
                for path in paths:
                    st = _tree_get(self.state, path)
                    p, m, v = fn_t(
                        flat_p[path], gmap[path], st["m"], st["v"], lr, self.step
                    )
                    st["m"], st["v"] = m, v
                    new_p.append((path, p))
                continue
            P = mx.stack([flat_p[p] for p in paths])
            G = mx.stack([gmap[p] for p in paths])
            M = mx.stack([_tree_get(self.state, p)["m"] for p in paths])
            V = mx.stack([_tree_get(self.state, p)["v"] for p in paths])
            Pn, Mn, Vn = fn(P, G, M, V, lr, self.step)
            for i, path in enumerate(paths):
                st = _tree_get(self.state, path)
                st["m"], st["v"] = Mn[i], Vn[i]
                new_p.append((path, Pn[i]))
        return tree_unflatten(new_p)


class AdamH(FusedAdamW):
    """Adam 方向 + Frobenius 范数球投影（MuonH 体系中的 lm_head 组）。

    readout 的行是逐 token 词表语义，不做 NS 正交化；方向仍由 Adam 一
    二阶矩给出，更新后 rescale 回更新前的 Frobenius 范数（范数走
    _fro_norm 融合 kernel：bf16 读入、f32 寄存器累加，与 MuonH 同口径，
    不再每步物化两份 (V,D) f32），与 MuonH 同一超球约束：方向/范数解耦。
    wd 是径向分量，投影后近似无操作，构造时置 0。lr 用 MuonH 的基础
    学习率而非 adam_lr。
    """

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        n0 = _fro_norm_auto(parameter)
        p = super().apply_single(gradient, parameter, state)
        n1 = _fro_norm_auto(p)
        return p * (n0 / mx.maximum(n1, 1e-12)).astype(p.dtype)

    def apply_gradients(self, gradients: dict, parameters: dict):
        # 超球投影是逐张量范数，不能走父类的同形状堆叠。
        return optim.Optimizer.apply_gradients(self, gradients, parameters)


class SinkhornBalanced(optim.Optimizer):
    """Sinkhorn 均衡更新（报告 §2.5 第二/三条 + 算法 1；超参见 §4.2.2）。

    用于三类 vocab 维大矩阵：Engram 检索表、token 嵌入、预测头（lm_head 与
    DSpark markov_head 的 2D embed/head）。与 Muon 同流程，只把
    Newton–Schulz 正交化换成 Sinkhorn 行/列 L2 均衡：

      Δ_t = √n·U⁽ᴷ⁾ = √n·D_r·Ĝ_t·D_c，  η̃_t = γ·η_t，  W ← W − η̃_t·Δ_t

    状态只有一份动量 m（替代 Adam 的 m/v，报告以此为省显存动机），
    **不施加权重衰减**（报告 §2.5：「Sinkhorn 均衡更新同样使用 Nesterov
    动量，但不施加权重衰减」）。逐行实现见 _sinkhorn_body 的对照注释。

    超参（报告 §4.2.2）：动量 = Muon 的动量系数（0.95）、lr 校正因子
    γ = 0.18、K = 11、τ = 10⁻³、ε = 10⁻²⁰；K 必须为奇数（最后一步落在
    行归一化上，U⁽ᴷ⁾ 才满足式 (7) 的单位行 RMS）。

    lr 语义：η_t 是「本来给这些参数用的 AdamW 学习率」，η̃_t = γ·η_t 由
    本类内部施加（报告 §2.5：「调整有效学习率以匹配 Adam 的更新幅度」）；
    训练循环的 lr 日程照常写 opt.learning_rate。
    """

    def __init__(
        self,
        learning_rate,
        momentum=0.95,
        gamma=0.18,
        sinkhorn_steps=11,
        eps=1e-20,
        tau=1e-3,
    ):
        super().__init__()
        if int(sinkhorn_steps) % 2 != 1:
            raise ValueError(f"K 必须是奇数（收尾在行归一化），收到 {sinkhorn_steps}")
        # 学习率既可给常数也可给 scheduler callable（与 mlx 其它优化器一致）
        self._maybe_schedule("learning_rate", learning_rate)
        # 暴露 momentum 属性：trainer.utils.apply_lr_schedule 会按 Muon 的
        # momentum 日程同步写它（`hasattr(opt, "momentum")`），检查点也随
        # _optimizer_hparams 存取——与 Muon 组同源，符合 §4.2.2「与 Muon
        # 相同的动量系数」。
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.sinkhorn_steps = int(sinkhorn_steps)
        self.eps = float(eps)
        self.tau = float(tau)

    def init_single(self, parameter: mx.array, state: dict):
        """只初始化一份动量（省掉 Adam 的二阶矩）。"""
        state["m"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        assert parameter.ndim == 2, (
            f"SinkhornBalanced 只处理 2D 权重矩阵，收到 {parameter.shape}"
        )
        lr = self.learning_rate.astype(gradient.dtype)
        # n = 隐维度 = W 的最后一维（嵌入/预测头口径）；每形状一个编译体。
        fn = _sinkhorn_kernel(
            self.momentum,
            self.gamma,
            self.sinkhorn_steps,
            self.eps,
            self.tau,
            int(parameter.shape[-1]),
        )
        p, m = fn(parameter, gradient, state["m"], lr)
        state["m"] = m
        return p


class BatchedMuon(optim.Muon):
    """Muon 的批量 Newton-Schulz 版：按形状分组堆叠成 (N, r, c) 一次跑 NS，
    动量更新与 lr 缩放同样在组内批量完成。

    动机：基类逐张量跑 NS5（每矩阵 5 步 × 3 GEMM + norm），100M 模型 103
    个 Muon 张量 ≈ 1700 kernel/步，optimizer 占整步墙钟 ~48%（小 kernel
    海洋，M4 Max 实测）。批量化后 kernel 数 ~16/形状组，数学上与基类逐
    张量 NS 严格等价（batch 维无耦合，Frobenius norm 按矩阵独立计算，
    lr 缩放只依赖形状），单步更新实测逐位一致。

    逐参数语义（转置规则、reshape ndim>2、nesterov、wd、lr scale）全部
    照抄基类 apply_single/_zeropower_via_newtonschulz5。

    hyperball=True（MuonH，Marin 口径）：更新后把每个矩阵 rescale 回
    更新前的 Frobenius 范数（fp32 算范数）——方向由正交化动量决定、
    范数冻结，两者解耦。RMSNorm 架构下矩阵整体尺度大半由 norm 吸收
    （gauge 自由度），冻结它让优化器带宽全花在方向上，lr 语义也更干净。
    与 r073 bias 零均值投影同一哲学。ndim>2 的堆叠专家以 axis0 为
    batch 逐矩阵 NS（无跨专家耦合），norm 投影同样逐矩阵。

    逐头 Muon（报告 §2.5 第一条，默认开）：Query 权重按行 reshape 成
    (n_heads, head_dim, in) 后走同一条 stack 路径——每个头独立 NS，
    hyperball 下每个头的 Frobenius 范数各自冻结（head_dim 由
    _per_head_dim 按路径解析：注意力 head_dim / indexer index_head_dim）。
    这就是「为不同头提供不同预条件器」；单头共享的 wkv 不切。

    【负面结果，勿重引入】曾实现过 NS 降频复用（每 8 步重算正交化、
    命中步复用旧极因子 D；以及 Temporal 变体：缓存左变换 Q、命中步
    D=Q@normalize(U)）省 ~30% 优化器墙钟。r082 归因（7 条单变量
    probe，同 seed/数据、微步对同步）实测 EVERY=8 在早期损失
    0.4-0.5 nat——方向仍精确正交但相对当前动量是 stale 的，而 warmup
    期动量旋转最快，恰好最不能复用。Temporal Q 更差：Q 是对刷新步
    奇异值谱定制的逆缩放，动量一变输出严重非正交（实测残差 4~13，
    完整 NS5 是 0.02~0.03）。整套机制已删除，正交化每步全量重算。
    """

    def __init__(
        self,
        learning_rate,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
        hyperball=False,
        ns_bf16=False,
        stack_ns_steps=None,
        head_dim=0,
        index_head_dim=0,
        polar_ema=False,
        normuon_beta2=0.0,
        cautious_wd=False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        # ---- autoresearch-mlx 战役移植（2026-09-01，双 regime 验证）----
        # polar_ema：动量 EMA 建在极因子空间——NS 每步作用于原始梯度 G，
        # 动量/ Nesterov 读出都在 NS(G) 上做，读出后 renorm 回当前极因子
        # 的 F 范数（尺度语义与标准路径一致）。是我们栈的 bar 组件。
        # 注意：动量 buffer 语义变为极因子 EMA，与旧 checkpoint 的动量
        # state 不兼容；wd 在此模式下从动量 kernel 移到 apply 阶段做
        # 解耦衰减（极因子空间里耦合 wd 无意义）。
        self.polar_ema = polar_ema
        # NorMuon 行 RMS 尾（arXiv 2510.05491）：逐神经元二阶矩 EMA
        # （β2=0.95）归一化正交化更新，再 renorm 回尾前 F 范数。bar 组件。
        # v 存在 side dict（按组/per-tensor），不在 optimizer state 里——
        # 续跑丢失后 ~20 步内热起来，可接受。
        self.normuon_beta2 = float(normuon_beta2)
        self._normuon_v = {}
        # cautious_wd（CWD，Chen et al. 2025，arXiv 2510.12402；战役双
        # regime keep：d4@467
        # −0.0090 / d2@1070 −0.0034 bpb）：apply 阶段的解耦衰减加
        # 逐坐标掩码 sign(X)==sign(P)。只在本组 weight_decay != 0 时有
        # 意义（create_mixed_optimizer 默认 VIBY_MUON_WD=0.1）。
        self.cautious_wd = cautious_wd
        if polar_ema:
            assert nesterov, "polar-ema 的读出公式即 Nesterov 形式"
        self.hyperball = hyperball
        # bf16 NS（Marin 口径：正交化 bf16、范数 fp32）：NS 是迭代求精，
        # bf16 精度足够，GEMM 减半字节流量。仅 muonh 下默认开。
        self.ns_bf16 = ns_bf16
        # 堆叠专家组的 NS 迭代步数（None = 与 ns_steps 相同）。仅迭代
        # 次数，不是降频——正交化每步全量重算（降频复用损害见类 docstring）。
        self.stack_ns_steps = stack_ns_steps
        # 逐头 Muon（报告 §2.5 第一条）：Query 权重沿 head 切开做 NS。
        # head_dim=0 关闭；index_head_dim 是 indexer 的逐头维度（V4.1 的
        # indexer 头数/头维度与主注意力不同：config.index_n_heads/
        # index_head_dim，故两条路径各用各的 head_dim，见 _per_head_dim）。
        self.head_dim = int(head_dim or 0)
        self.index_head_dim = int(index_head_dim or 0)
        # 测量开关（默认关）：VIBY_MUONH_NO_NS=1 时跳过谱均衡，X 改为
        # F-范数匹配 NS 输出的归一化动量——隔离「hyperball 范数冻结」与
        # 「NS 谱均衡」各自的贡献。仅归因测量用，勿用于正式 run。
        self._no_ns = os.environ.get("VIBY_MUONH_NO_NS", "0") == "1"
        # NS 系数：cubic5b05（默认；见下）/ cubic5（NVIDIA relaxed cubic，
        # arXiv 2606.00371，每步 2 GEMM 比 quintic 省 1/3）/ classic
        # （Keller-Jordan 固定五次数）/ pe（Polar Express 逐迭代最优，
        # 同 5 步同 GEMM 数，正交化残差减半但训练实测变差，见
        # _PE_COEFFS 注释）/ cubic5b10、cubic5b002 与 frac50/frac25
        # （否定/检验候选，见各自注释）。
        # P13 probe（14 分钟，微步 500）：cubic5 loss 5.245 vs classic
        # 5.328（−0.08），步速 0.70→0.73 step/s，apply_gradients 基准
        # −19.5%（experiments/bench_muonh.py exp_bf16_ns5 行）。
        # 膝盖扫描（同协议）：l0=0.007→5.245 / 0.05→5.223 / 0.10→5.314，
        # 峰值在 l0=0.05；b05 与 cubic5 之差在 probe 噪声（±0.05）内，
        # 成本相同，按预注册规则翻默认为 b05。
        self._ns_coeff = os.environ.get("VIBY_MUONH_NS_COEFF", "cubic5b05")
        assert self._ns_coeff in (
            "classic",
            "pe",
            "cubic5",
            "cubic5b05",
            "cubic5b10",
            "cubic5b002",
            "cubic5b07",
            "cubic5b01",
            "cubic5b005",
            *_FRAC_PRESETS,
        ), self._ns_coeff
        # EdgeCubic（P-3）：逐形状组膝盖 l0 = 实测 σ*×4.5（_EDGE_L0_*），
        # 收敛步数自选（4–8）。与 _ns_coeff 互斥。probe 档，未验证前勿翻默认。
        self._edge = os.environ.get("VIBY_MUONH_EDGE", "0") == "1"
        self._edge_fns = {}

    def _ns_edge(self, X, kind):
        """EdgeCubic 逐组 NS：按 (kind, r, c) 解析 l0 与收敛步数并缓存。"""
        r, c = X.shape[-2], X.shape[-1]
        key = (kind, r, c)
        fn = self._edge_fns.get(key)
        if fn is None:
            table = _EDGE_L0_STACK if kind == "gram" else _EDGE_L0_2D
            l0 = table.get((r, c), _EDGE_L0_DEFAULT)
            _, steps = _cubic_schedule(l0)
            fn = _ns_core_fn(kind, self.ns_bf16, steps, f"edge:{l0}")
            self._edge_fns[key] = fn
        return fn(X)

    def _orth(self, U):
        """测量模式（_no_ns）下的 X：F-范数匹配 NS 输出的归一化动量。
        NS 输出是半正交矩阵，||X||_F = √min(r,c)，故归一化后同尺度，
        只去掉谱均衡、保留方向——隔离 hyperball 与 NS 的各自贡献。"""
        nrm = _fro_norm_auto(U)
        s = float(min(U.shape[-2], U.shape[-1])) ** 0.5
        return U * (s / (nrm + 1e-12)).astype(U.dtype)

    def _per_head_dim(self, path: str) -> int:
        """该参数按头切开时每头的行数；0 = 不做逐头（整矩阵 NS）。

        报告 §2.5 第一条：**Query 与 Key 权重使用逐头 Muon**。本架构里
        Query 权重是 attn.wq_b（[n_heads*head_dim, q_lora_rank]）与
        indexer.wq_b（[index_n_heads*index_head_dim, q_lora_rank]），按行
        reshape 成 (n_heads, head_dim, in) 恰好是逐头切分（行连续）。
        Key 侧是共享 K=V 的单头 wkv（nn.Linear(dim, head_dim)，形状
        [head_dim, dim]）：它只有**一个** head，拆不出多个头，因此保持整
        矩阵 NS（wkv 的行数就是 head_dim 本身，按行拆开等于把每个通道当
        一个头，语义错误）。旧命名的 q_proj 一并保留以兼容历史 checkpoint；
        k_proj/v_proj/kv_proj 在本架构不存在，也不做逐头。
        """
        parts = path.replace("/", ".").split(".")
        if "wq_b" not in parts and "q_proj" not in parts:
            return 0
        if ".indexer." in path:
            return self.index_head_dim
        return self.head_dim

    def _ns5(self, X, steps=None):
        """批量 Newton-Schulz：X (N, r, c)，batch 维无耦合（照抄基类规则）。
        ns_bf16 时迭代在 bf16 做（归一化范数仍 fp32，避免弱类型提升把
        整条链抬回 f32）。
        """
        steps = steps or self.ns_steps
        return _ns_core_fn("std", self.ns_bf16, steps, self._ns_coeff)(X)

    def _ns5_gram(self, X, steps=None):
        """Gram Newton-Schulz：在 XXᵀ 上迭代，算术上等价标准 NS5。
        迭代 GEMM 全部在短边空间（(N,r,r)³ 替代 (N,r,c) 的 XXᵀ/B@X），
        对 r<c 的专家栈（384<640 / 320<384）每迭代省 ~25% FLOPs。
        """
        steps = steps or self.ns_steps
        return _ns_core_fn("gram", self.ns_bf16, steps, self._ns_coeff)(X)

    def _ns_auto(self, X, steps=None):
        """按矩阵形状选标准 NS5 或 Gram-NS（两者算术等价，见 _ns5_gram）。

        Gram 把每次迭代的 GEMM 搬到短边空间，r≠c 越悬殊越省；但方阵时短边
        等于长边，白搭一次 XXᵀ。实测 (59,768,768) 组 61→73ms、
        (2304,384,384) 专家栈 320→371ms 都是 Gram 更慢，而
        (2,6400,768) 是 12.9→5.2ms、(2304,768,384) 是 527→414ms 更快。
        """
        r, c = X.shape[-2], X.shape[-1]
        if r == c:
            return self._ns5(X, steps)
        return self._ns5_gram(X, steps)

    def apply_gradients(self, gradients: dict, parameters: dict):
        # 基类开头两段：lazy init + scheduler 更新 + step 递增
        if not self._initialized:
            self.init(gradients)
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)
        self.state["step"] = self.step + 1

        flat_g = dict(tree_flatten(gradients))
        flat_p = dict(tree_flatten(parameters))
        flat_s = dict(tree_flatten(self.state))
        state_v = {k[:-2]: v for k, v in flat_s.items() if k.endswith(".v")}

        def get_state(dotted):
            node = self.state
            for part in dotted.split("."):
                node = node[int(part)] if isinstance(node, list) else node[part]
            return node

        # 分组：同 reshape 后 2D 形状的进一组；ndim<2（Muon 组不应出现，
        # 防御）回退基类逐张量路径；ndim>2（堆叠专家，仅 muonh 下会进
        # Muon 组）以 axis0 为 batch 进堆叠组，逐矩阵 NS，无跨专家耦合
        groups: dict = {}
        stack_groups: dict = {}
        singles = []
        for path, g in flat_g.items():
            if g.ndim < 2:
                singles.append(path)
                continue
            orig = g.shape
            if g.ndim > 2:
                b, r = orig[0], orig[1]
                c = g.size // (b * r)
                stack_groups.setdefault((b, r, c), []).append((path, orig))
            elif (hd := self._per_head_dim(path)) > 0 and orig[0] % hd == 0:
                # 逐头 Muon：Query 权重按行拆成 (n_heads, head_dim, in) 后
                # 进 stack 组（batch 维无耦合，等于每头独立 NS + 逐头
                # hyperball 范数冻结）。注意力用 config.head_dim、indexer 用
                # config.index_head_dim——两者头数/头维度不同，key 天然分到
                # 不同 stack 组（key 含 r=head_dim）。
                nh = orig[0] // hd
                stack_groups.setdefault((nh, hd, orig[1]), []).append((path, orig))
            else:
                groups.setdefault(orig, []).append((path, orig))

        m, nest, wd = self.momentum, self.nesterov, self.weight_decay
        lr0 = self.learning_rate
        new_params = {}
        new_v = {}
        # polar-ema / cautious 下衰减从动量 kernel（耦合式 G += wd*P）移到
        # apply 阶段：polar-ema 的动量住在极因子空间，耦合 wd 无意义；
        # cautious 需要按更新方向掩码，动量 kernel 里拿不到 X。
        decay_in_apply = self.polar_ema or self.cautious_wd
        mom_wd = 0.0 if decay_in_apply else wd
        apply_wd = wd if decay_in_apply else 0.0
        # snapshot 默认关闭；开启时整步只 int() 一次。否则每组一次
        # int(self.state["step"]) 就是每步每组一次 host sync（§3.5 纪律）。
        snap_step = int(self.state["step"]) if snapshot.active() else None

        mom_fn = _stack_mom_kernel(m, nest, mom_wd)
        apply_fn = _stack_apply_kernel(self.hyperball, apply_wd, self.cautious_wd)
        polar_mom_fn = _polar_mom_kernel(m) if self.polar_ema else None

        def momentum_stack(items, r, c):
            """逐元素部分（wd/动量/nesterov）整堆叠做，返回 (U, P, V, G.dtype)。"""
            paths = [p for p, _ in items]
            G = mx.stack([flat_g[p].reshape(r, c) for p in paths])
            P = mx.stack([flat_p[p].reshape(r, c) for p in paths])
            V = mx.stack([state_v[p].reshape(r, c) for p in paths])
            U, V = mom_fn(G, P, V)
            return U, P, V, G.dtype

        def scatter_back(items, V, NP):
            for i, (path, orig) in enumerate(items):
                new_v[path] = V[i].reshape(orig)
                new_params[path] = NP[i].reshape(orig).astype(flat_p[path].dtype)

        # 正交化每步全量重算（曾有的降频复用因 r082 归因被删除，见类
        # docstring 的负面结果记录）。
        for (r, c), items in groups.items():
            dt = flat_g[items[0][0]].dtype
            lr = lr0.astype(dt) * (max(1.0, r / c) ** 0.5)
            if self.polar_ema:
                # polar-ema：NS 前置到原始梯度上，动量建在极因子空间。
                paths = [p for p, _ in items]
                G = mx.stack([flat_g[p].reshape(r, c) for p in paths])
                P = mx.stack([flat_p[p].reshape(r, c) for p in paths])
                V = mx.stack([state_v[p].reshape(r, c) for p in paths])
                O = (
                    self._ns_edge(G, "std")
                    if self._edge
                    else self._orth(G)
                    if self._no_ns
                    else self._ns5(G)
                )
                U, V = polar_mom_fn(O, V)
                # Nesterov 读出 renorm 回当前极因子的逐矩阵 F 范数。
                X = U * (
                    _fro_norm_auto(O) / mx.maximum(_fro_norm_auto(U), 1e-12)
                ).astype(U.dtype)
                if self.normuon_beta2 > 0:
                    X, vn = _normuon_tail(
                        X, self._normuon_v.get((r, c)), self.normuon_beta2
                    )
                    self._normuon_v[(r, c)] = vn
                scatter_back(items, V, apply_fn(P, X, lr))
                continue
            # 2D 组不用 Gram（_ns_auto 在非方阵上与 NS5 有谱差，warmup
            # 会把 MTP 打飞）。
            U, P, V, dt = momentum_stack(items, r, c)
            if snap_step is not None:
                snapshot.maybe_dump_momentum(
                    snap_step,
                    "g2d",
                    (r, c),
                    U,
                    [p for p, _ in items],
                )
            X = (
                self._ns_edge(U, "std")
                if self._edge
                else self._orth(U)
                if self._no_ns
                else self._ns5(U)
            )
            if self.normuon_beta2 > 0:
                X, vn = _normuon_tail(
                    X, self._normuon_v.get((r, c)), self.normuon_beta2
                )
                self._normuon_v[(r, c)] = vn
            scatter_back(items, V, apply_fn(P, X, lr))

        # 堆叠组（ndim>2，逐专家语义）：逐张量直接处理，不跨层 stack。
        # NS 的 batch 维无耦合、动量/投影逐矩阵，三份整组 G/P/V stack 是纯
        # 搬运；逐张量路径（experiments/probe_stack_free_opt.py 同进程 A/B）
        # 两组省 ~41ms/步且输出与堆叠路径逐位一致（NS GEMM 的逐矩阵结果
        # 不随 batch 大小变）。
        for (b, r, c), items in stack_groups.items():
            dt = flat_g[items[0][0]].dtype
            lr = lr0.astype(dt) * (max(1.0, r / c) ** 0.5)
            Us, Vs = [], []
            for path, orig in items:
                Gt = flat_g[path].reshape(b, r, c)
                if self.polar_ema:
                    # 与 2D 组同语义：NS 前置到原始梯度，动量在极因子空间。
                    Ot = (
                        self._ns_edge(Gt, "gram")
                        if self._edge
                        else self._orth(Gt)
                        if self._no_ns
                        else self._ns5_gram(Gt, steps=self.stack_ns_steps)
                    )
                    U, Vn = polar_mom_fn(Ot, state_v[path].reshape(b, r, c))
                    U = U * (
                        _fro_norm_auto(Ot) / mx.maximum(_fro_norm_auto(U), 1e-12)
                    ).astype(U.dtype)
                else:
                    U, Vn = mom_fn(
                        Gt,
                        flat_p[path].reshape(b, r, c),
                        state_v[path].reshape(b, r, c),
                    )
                Us.append(U)
                Vs.append(Vn)
            if snap_step is not None:
                # snapshot 的整组 (N·b,r,c) 口径：只在测量步临时拼一份
                # （lazy，未列出的步不会物化）。
                snapshot.maybe_dump_momentum(
                    snap_step,
                    "gst",
                    (b, r, c),
                    mx.stack(Us).reshape(len(items) * b, r, c),
                    [p for p, _ in items],
                    b,
                )
            # 堆叠专家一律 Gram-NS（短边迭代）。_ns_auto 会让方阵
            # 384×384 改走标准 NS5，谱差在 hyperball 下会被放大。
            for (path, orig), U, Vn in zip(items, Us, Vs):
                if self.polar_ema:
                    Xo = U  # NS 已在动量前置段完成
                elif self._edge:
                    Xo = self._ns_edge(U, "gram")
                elif self._no_ns:
                    Xo = self._orth(U)
                else:
                    Xo = self._ns5_gram(U, steps=self.stack_ns_steps)
                if self.normuon_beta2 > 0:
                    Xo, vn = _normuon_tail(
                        Xo, self._normuon_v.get(path), self.normuon_beta2
                    )
                    self._normuon_v[path] = vn
                new_v[path] = Vn.reshape(orig)
                new_params[path] = (
                    apply_fn(flat_p[path].reshape(b, r, c), Xo, lr)
                    .reshape(orig)
                    .astype(flat_p[path].dtype)
                )

        for path in singles:
            state = get_state(path)
            new_params[path] = super().apply_single(flat_g[path], flat_p[path], state)
            # 与批量组统一从 flat_s 重建 state；若忘记回写，重建会用旧快照
            # 覆盖 apply_single 刚更新的动量，导致 1D 参数的动量每步被清零。
            flat_s[path + ".v"] = state["v"]

        if new_v or singles:
            for path, v in new_v.items():
                flat_s[path + ".v"] = v
            self.state = tree_unflatten(list(flat_s.items()))

        return tree_unflatten(list(new_params.items()))


def create_adamw_optimizer(model, args, training_type="pretrain"):
    """全参数 AdamW（用于优化器杠杆对照实验；lr 需单独调优）。"""
    from .utils import Logger

    Logger("正在为优化器进行参数分组 (pure AdamW)")
    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        # token embedding / lm_head / DSpark 头 / Engram 检索表：纯 AdamW 对照
        # 组（--optimizer adamw 的杠杆实验）下不细分，只按 embed 组统一 lr 系数
        return (
            path.endswith(".embed.weight")
            or "lm_head" in path
            or "markov_head" in path
            or path.endswith(".ncp.codebook")
        )

    embed_count = sum(1 for p, a in trainable if _is_embed(p, a))
    other_count = len(trainable) - embed_count
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - 其余参数组: {other_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, other_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, other_lr_mult, adam_wd = 1.0, 1.0, 0.1

    adamw_embed = FusedAdamW(
        learning_rate=args.learning_rate * embed_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_other = FusedAdamW(
        learning_rate=args.learning_rate * other_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_embed.base_lr = args.learning_rate * embed_lr_mult
    adamw_other.base_lr = args.learning_rate * other_lr_mult
    return optim.MultiOptimizer([adamw_embed, adamw_other], filters=[_is_embed])


def create_mixed_optimizer(model, args, training_type="pretrain"):
    """
    创建混合优化器（mlx.optimizers.MultiOptimizer）

    双基础学习率：muon_lr（Muon/AdamH 组）与 adam_lr（AdamW 各组），由
    args.muon_lr / args.learning_rate 给出（train_pretrain 的
    resolve_compute_scaled_hparams 按 Hyperball compute 公式写入；SFT/DPO
    未解析时 adam_lr 回退 args.learning_rate，muon_lr 按 13/3 × adam_lr
    派生，上限 0.05）。
    Adam/AdamH 组的 beta2/eps 同样由 args.adam_beta2 / args.adam_eps 给出
    （兜底旧常数 0.95 / 1e-8），beta1 恒 0.9。

    参数分组（DeepSeek-V4.1 缩放版，对齐报告 §2.5 + §4.2.2）：
    - Muon/MuonH：ndim >= 2 的核心权重矩阵（注意力 wq_a/wq_b/wkv/wo、专家、
      mHC fn、Engram wkv 投影、MTP 投影），wd=0。其中 Query 权重
      （attn.wq_b / indexer.wq_b）**默认逐头 NS**（§2.5 第一条；
      VIBY_MUONH_PER_HEAD=0 关，消融用）；K=V 共享的单头 wkv 保持整矩阵。
    - SinkhornBalanced（§2.5 第二/三条 + 算法 1，超参 §4.2.2）：Engram
      检索表（**5× lr**，§4.2.2「Engram 的学习率被放大 5 倍」）、token 嵌入
      与预测头（lm_head / DSpark markov_head 的 2D embed+head）。
      动量同 Muon（0.95）、γ=0.18、K=11、τ=1e-3、ε=1e-20，**无 wd**；
      只存一份动量。
    - AdamW(norm)：归一化层权重（各 RMSNorm 的 .weight），lr = adam_lr，
      **wd=adam_wd**（§2.5：「归一化层权重同样受权重衰减影响」）
    - AdamW(router)：router.weight，lr = adam_lr * 0.05（MOE_ROUTER_LR_MULT
      可调），wd=adam_wd。router.bias 是 frozen 的 e_score_correction_bias：
      不在 trainable 里，由训练循环按负载覆写（noaux_tc），分组统计看不到它。
    - AdamW(no-wd)：其余全部（偏置/缩放因子：attn_sink 零初始化、mHC
      scale/base、Engram 的 q_weight/k_weight 乘性门、confidence head，
      以及默认不入 Muon 的 3D 专家栈），lr = adam_lr，**wd=0**（§2.5：
      「偏置与缩放因子不受」权重衰减；这些参数要么零初始化、要么有校准过的
      尺度，wd=0.1 会在 1-epoch 内把它们径向缩到 ~10%）
    （SFT 时 embed/scalar lr 分别乘 0.1/0.3；norm 组 wd=0.01，no-wd 组 wd=0。
    VIBY_SINKHORN=0 回退旧分组：lm_head 走 AdamH、embed/Engram 表走 AdamW）

    --muonh（MuonH/AdamH/Adam 体系，Marin 口径，默认关闭，显式开启）：
    - Muon 组更新加 Frobenius 范数球投影（方向/范数解耦，hyperball）；
    - 3D 堆叠专家从 AdamW 移入 MuonH：以 axis0 为 batch 逐专家 NS
      （_ns5 批量维无耦合，语义正确；旧排除是因为 reshape (E,out·in)
      跨专家耦合）。VIBY_MUONH_EXPERTS=0 可只开投影不移专家（消融用，
      也省掉逐专家 NS 的算力开销）；
    - lm_head（非 tied 才存在）走 AdamH：Adam 方向 + 同一范数球投影，
      lr 用 muon 基础学习率。
    """
    from .utils import Logger

    Logger("正在为优化器进行参数分组")

    trainable = tree_flatten(model.trainable_parameters())
    muonh = bool(getattr(args, "muonh", False))
    # 3-D 堆叠专家默认**不进** MuonH：V4.1 架构下走堆叠路径会在训练进程内随机
    # 产生非有限激活（实测 SFT 128 长度 1/3~2/3 的运行里，第 1~3 个微批起
    # loss=nan；同一批数据/权重在进程外纯模型跑 40 次全有限；关掉这条路径
    # ——VIBY_MUONH_EXPERTS=0 或 --no_muonh——后 3/3 运行全程有限）。堆叠
    # kernel 的 (N,r,c) 组合与分配布局相关，属旧实现遗留的隐患；专家改走
    # AdamW 标量组（lr=adam_lr）后 20 步 tiny 预训练 loss 11.77→2.76、
    # router.bias 正常更新。要复现旧行为显式设 VIBY_MUONH_EXPERTS=1。
    experts_in_muon = muonh and os.environ.get("VIBY_MUONH_EXPERTS", "0") == "1"

    # 双基础 lr + Adam 组超参（pretrain 由 resolve_compute_scaled_hparams
    # 写入；未写入时 adam_lr 回退 args.learning_rate，muon_lr 按 Marin
    # 口径 13/3 × adam_lr 派生——与 pretrain「auto + 手动 lr」规则一致。
    # Muon 正交化步长语义本就需要数倍于 Adam 的 lr，1:1 兜底会让 SFT 的
    # 矩阵组步长与 pretrain 校准口径（r086: adam 1.55e-3 / muon 6.7e-3）
    # 脱节；该比例与是否开 hyperball 投影无关（投影约束权重范数，不改步长）。
    adam_lr = float(getattr(args, "adam_lr", None) or args.learning_rate)
    _muon_lr_arg = getattr(args, "muon_lr", None)
    if _muon_lr_arg:
        muon_lr = float(_muon_lr_arg)
    else:
        muon_lr = min(0.05, (13.0 / 3.0) * adam_lr)
    beta2 = float(getattr(args, "adam_beta2", None) or 0.95)
    eps = float(getattr(args, "adam_eps", None) or 1e-8)

    def _is_embed(path, arr):
        # 查表 / readout：token embedding、DSpark 马尔可夫头（embed 与 head 都是
        # vocab 维的稀疏更新目标）、lm_head。Engram 检索表另有 5× 组，先排除；
        # 用 ".embed.weight" 精确匹配，避免误吞别的含 embed 的路径。
        if "engram_layers" in path:
            return False
        return (
            path.endswith(".embed.weight")
            or "lm_head" in path
            or "markov_head" in path
            or path.endswith(".ncp.codebook")
        )

    def _is_ngram_table(path, arr):
        # Engram n-gram 检索表：稀疏 gather 更新，独立 Adam 组（5× lr、wd=0）
        return ".engram_layers." in path and path.endswith(".embed.weight")

    def _is_adamh(path, arr):
        # 旧分组的 lm_head readout：Adam 方向 + 范数球投影（AdamH 类），仅
        # muonh 下启用。默认 lm_head 归 Sinkhorn 组（§2.5：预测头用「动量 +
        # Sinkhorn 均衡」，不再存 Adam 的一二阶矩），只有 VIBY_SINKHORN=0
        # 的消融回退才走这里。
        return muonh and "lm_head" in path and arr.ndim >= 2

    # ---- Sinkhorn 均衡组（报告 §2.5 算法 1；超参 §4.2.2）----
    # 默认开：Engram 检索表（5× lr，§4.2.2）/ token 嵌入 / 预测头改走
    # SinkhornBalanced（Nesterov 动量 + 行/列 L2 均衡，无 wd）。
    # VIBY_SINKHORN=0 回退旧分组（lm_head→AdamH、embed/Engram 表→AdamW），
    # 供既有 checkpoint / 消融使用。
    sinkhorn_on = os.environ.get("VIBY_SINKHORN", "1") == "1"
    _sink_beta = float(
        getattr(args, "muon_momentum", None)
        or os.environ.get("VIBY_SINKHORN_BETA", "0.95")
    )
    _sink_gamma = float(os.environ.get("VIBY_SINKHORN_GAMMA", "0.18"))
    _sink_steps = int(
        getattr(args, "sinkhorn_steps", None) or os.environ.get("VIBY_SINKHORN_K", "11")
    )
    _sink_eps = float(os.environ.get("VIBY_SINKHORN_EPS", "1e-20"))
    _sink_tau = float(os.environ.get("VIBY_SINKHORN_TAU", "1e-3"))
    # Engram 检索表 lr 放大倍数（§4.2.2：「Engram 的学习率被放大 5 倍」，
    # 引 Cheng et al., 2026b）
    _engram_lr_mult = float(os.environ.get("VIBY_ENGRAM_LR_MULT", "5.0"))

    def _is_ngram_sinkhorn(path, arr):
        # Engram 检索表（5× lr）：.engram_layers.*.embed.weight，2-D 才收。
        # 非 2-D（理论不会出现）落 AdamW 无衰减组。
        return sinkhorn_on and arr.ndim == 2 and _is_ngram_table(path, arr)

    def _is_sinkhorn(path, arr):
        # token 嵌入 / lm_head / DSpark markov_head（embed 与 head 都是 vocab
        # 维查表/readout 语义）。2-D 才收：markov 头若退化成别的形状就落
        # AdamW 无衰减组（不做 Sinkhorn 行/列均衡）。
        return sinkhorn_on and arr.ndim == 2 and _is_embed(path, arr)

    def _is_norm_weight(path, arr):
        # 归一化层权重：RMSNorm 的 1-D weight（模块名以 norm 结尾：attn_norm/
        # ffn_norm/q_norm/kv_norm/k_norm/compressor.norm/main_norm/model.norm）。
        # 报告 §2.5：这类参数保留 AdamW，且**受权重衰减影响**——与偏置/
        # 缩放因子（wd=0）分属两组。
        if arr.ndim != 1 or not path.endswith(".weight"):
            return False
        return any(p.endswith("norm") for p in path.split(".")[:-1])

    def _is_muon(path, arr):
        # 3D 堆叠专家：显式启用 muonh 且 VIBY_MUONH_EXPERTS=1 时逐专家 NS；
        # VIBY_MUONH_EXPERTS=0 时排除进 AdamW，避免基类 reshape (E,out·in)
        # 跨专家耦合。MoE router 也不走 Muon：正交化更新步长恒定偏大，会把
        # 路由打分持续推向失衡；单独小 lr AdamW 组。
        # 三类"矩阵形状但不该正交化"的参数落 AdamW 无衰减组：
        # - Engram 的 q_weight/k_weight（ones 初始化的乘性门，MuonH 的超球
        #   半径=初始范数，正交化会改掉门的尺度语义）；
        # - confidence head（输出维 1，正交化步长语义不适用）；
        # - RoPE 表 freq_cos/freq_sin（常量，trainer 已 freeze；这里兜底，
        #   免得绕过 build_model_and_tokenizer 的调用把频率表交给 NS）。
        return (
            arr.ndim >= 2
            and not _is_embed(path, arr)
            and not _is_ngram_table(path, arr)
            and (experts_in_muon or ".experts." not in path)
            and ".router." not in path
            and not (
                ".engram_layers." in path and path.endswith(("q_weight", "k_weight"))
            )
            and "confidence_head" not in path
            and not path.endswith(("freq_cos", "freq_sin"))
        )

    def _is_router(path, arr):
        return ".router." in path

    # 逐头 Muon（报告 §2.5 第一条）：Query 权重默认按头切开做 NS。
    # 旧实现默认关（VIBY_MUONH_PER_HEAD 未设时 hd=0），且旧 _is_qkv_path
    # 匹配的是 q_proj/k_proj/v_proj/kv_proj——V4.1 架构里这些名字都不存在
    # （新名字 wq_b/wkv），逐头实际从未生效；当时的 per-head 负面归因
    # （r082 probe_p8/p7 −0.14 nat @500）还叠加了 NS 降频复用 + stale 极
    # 因子（probe_p2 −0.26 nat），不是干净口径。报告口径是**默认用逐头
    # Muon**，故这里翻默认：VIBY_MUONH_PER_HEAD=0 关闭（保留消融能力）。
    # 两条 Q 路径 head_dim 不同：主注意力 config.head_dim、indexer
    # config.index_head_dim；Key 侧是共享 K=V 的单头 wkv（[head_dim, dim]），
    # 拆不出多头，保持整矩阵 NS（见 BatchedMuon._per_head_dim）。
    _per_head_on = os.environ.get("VIBY_MUONH_PER_HEAD", "1") == "1"
    _cfg = getattr(model, "config", None)
    hd = int(getattr(_cfg, "head_dim", 0) or 0) if _per_head_on else 0
    hd_index = int(getattr(_cfg, "index_head_dim", 0) or 0) if _per_head_on else 0

    def _head_dim_of(path: str) -> int:
        parts = path.replace("/", ".").split(".")
        if "wq_b" not in parts and "q_proj" not in parts:
            return 0
        return hd_index if ".indexer." in path else hd

    if training_type == "sft":
        embed_lr_mult, scalar_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, scalar_lr_mult, adam_wd = 1.0, 1.0, 0.1

    muon_count = sum(1 for p, a in trainable if _is_muon(p, a))
    per_head_count = sum(
        1 for p, a in trainable if _is_muon(p, a) and _head_dim_of(p) > 0
    )
    sinkhorn_engram_count = sum(1 for p, a in trainable if _is_ngram_sinkhorn(p, a))
    sinkhorn_embed_count = sum(1 for p, a in trainable if _is_sinkhorn(p, a))
    sinkhorn_engram_params = sum(
        a.size for p, a in trainable if _is_ngram_sinkhorn(p, a)
    )
    sinkhorn_embed_params = sum(a.size for p, a in trainable if _is_sinkhorn(p, a))
    adamh_count = sum(
        1
        for p, a in trainable
        if not sinkhorn_on and _is_adamh(p, a) and not _is_muon(p, a)
    )
    embed_count = sum(
        1
        for p, a in trainable
        if not sinkhorn_on
        and _is_embed(p, a)
        and not _is_muon(p, a)
        and not _is_adamh(p, a)
    )
    ngram_table_count = sum(
        1 for p, a in trainable if not sinkhorn_on and _is_ngram_table(p, a)
    )
    norm_count = sum(
        1
        for p, a in trainable
        if _is_norm_weight(p, a) and not _is_muon(p, a) and not _is_router(p, a)
    )
    router_count = sum(
        1 for p, a in trainable if _is_router(p, a) and not _is_muon(p, a)
    )
    # 无衰减组：除 Muon / Sinkhorn / norm / router 以外的全部参数
    nowd_count = sum(
        1
        for p, a in trainable
        if not _is_muon(p, a)
        and not _is_ngram_sinkhorn(p, a)
        and not _is_sinkhorn(p, a)
        and not _is_norm_weight(p, a)
        and not _is_router(p, a)
    )
    nowd_params = sum(
        a.size
        for p, a in trainable
        if not _is_muon(p, a)
        and not _is_ngram_sinkhorn(p, a)
        and not _is_sinkhorn(p, a)
        and not _is_norm_weight(p, a)
        and not _is_router(p, a)
    )

    Logger("参数分组完成：")
    Logger(
        f"  - {'MuonH(hyperball)' if muonh else 'Muon'} 参数组 (核心权重): "
        f"{muon_count} 个张量（逐头 Muon 的 Q 权重 {per_head_count} 个：注意力 "
        f"head_dim={hd}、indexer head_dim={hd_index}）"
    )
    if sinkhorn_on:
        Logger(
            f"  - Sinkhorn 均衡组 (token 嵌入/预测头, lr=adam_lr, "
            f"γ={_sink_gamma:g}, K={_sink_steps}, τ={_sink_tau:g}, "
            f"ε={_sink_eps:g}, β={_sink_beta:g}, wd=0): "
            f"{sinkhorn_embed_count} 个张量 / {sinkhorn_embed_params:,} 参数"
        )
        Logger(
            f"  - Sinkhorn 均衡组 (Engram 检索表, lr={_engram_lr_mult:g}×adam_lr, "
            f"wd=0): {sinkhorn_engram_count} 个张量 / "
            f"{sinkhorn_engram_params:,} 参数"
        )
    else:
        Logger("  - Sinkhorn 均衡组: 关闭（VIBY_SINKHORN=0，回退旧分组）")
        if muonh:
            Logger(f"  - AdamH 参数组 (lm_head): {adamh_count} 个张量")
        Logger(f"  - AdamW 嵌入层参数组 (wd={adam_wd:g}): {embed_count} 个张量")
        if ngram_table_count:
            Logger(
                f"  - Engram 检索表参数组 (Adam 5x lr, wd=0): "
                f"{ngram_table_count} 个张量"
            )
    Logger(
        f"  - AdamW 归一化层权重组 (lr=adam_lr, wd={adam_wd:g}): {norm_count} 个张量"
    )
    Logger(
        f"  - MoE router 参数组 (lr=adam_lr×0.05, wd={adam_wd:g}): "
        f"{router_count} 个张量"
    )
    Logger(
        f"  - AdamW 无衰减组 (偏置/缩放因子/3D 专家, wd=0): "
        f"{nowd_count} 个张量 / {nowd_params:,} 参数"
    )
    # router 需要远小于 AdamW 组的 lr：AdamW 每坐标步长≈lr，adam base lr
    # 下 ~10 步就把 (32,768) 的 router 权重打乱到 sigmoid
    # 饱和（实测 C 瞬间冲到 14K、吞吐 -45%）；0.05× 让其慢速移动、
    # bias 均衡项压得住负载。可用 MOE_ROUTER_LR_MULT 覆盖。
    router_lr_mult = float(
        os.environ.get("MOE_ROUTER_LR_MULT", getattr(args, "router_lr_mult", 0.05))
    )

    # autoresearch-mlx 战役（2026-09-01 结算）验证出的优化器组件。
    # 默认开：NorMuon 行 RMS 尾、cautious weight decay（muon 组与 Adam
    # 组）；默认关：polar-ema（用户决策 2026-09-02：先不上生产默认，
    # VIBY_MUON_POLAR_EMA=1 手动开）。均与 muonh/hyperball 兼容（测试
    # 覆盖见 tests/test_campaign_optim.py）：
    # - VIBY_MUON_POLAR_EMA=1：开 polar-ema（动量 EMA 建在极因子空间：
    #   NS 前置到原始梯度，Nesterov 读出 renorm 回逐矩阵 F 范数）。
    # - VIBY_MUON_NORMUON=0：关 NorMuon 行 RMS 尾（arXiv 2510.05491，
    #   beta2=0.95，归一后 renorm 回尾前 F 范数）。
    # - VIBY_MUON_CAUTIOUS=0 / VIBY_MUON_WD：cautious weight decay
    #   （Liang et al. 2024；战役双 regime keep：d4@467 −0.0090 /
    #   d2@1070 −0.0034 bpb），muon 组 wd 默认 0.1。
    # - VIBY_ADAM_CAUTIOUS=0：关 FusedAdamW 各组（embed/router 有 wd）
    #   的 cautious 衰减；wd=0 的组（scalar/ngram/AdamH）本就无操作。
    _polar_ema = os.environ.get("VIBY_MUON_POLAR_EMA", "0") == "1"
    _normuon = os.environ.get("VIBY_MUON_NORMUON", "1") == "1"
    _muon_cautious = os.environ.get("VIBY_MUON_CAUTIOUS", "1") == "1"
    _muon_wd = float(os.environ.get("VIBY_MUON_WD", "0.1"))
    _adam_cautious = os.environ.get("VIBY_ADAM_CAUTIOUS", "1") == "1"
    # KL-SOAP-H（arXiv 2607.20548 Algorithm 2 + arXiv 2509.03378；hyperball
    # 投影）：VIBY_KLSOAP=1 时矩阵组整体改走 KLSoaPH 替代 BatchedMuon
    # （MLA 不切段、无 per-head；lr = muon_lr × VIBY_KLSOAP_LR_MULT；
    # cautious/wd 沿用上面两个 env）。polar-ema/normuon 是 Muon 系组件，
    # 对 KL-SOAP 无意义，不接。栈内未验证，默认关。
    _klsoap = os.environ.get("VIBY_KLSOAP", "0") == "1"
    Logger(
        "战役优化器组件: polar_ema=%s normuon=%s muon_cautious=%s "
        "(muon wd=%g) adam_cautious=%s klsoap=%s（env 可逐项开关）"
        % (_polar_ema, _normuon, _muon_cautious, _muon_wd, _adam_cautious, _klsoap)
    )
    if _klsoap:
        from .klsoap import KLSoaPH

        muon_opt = KLSoaPH(
            learning_rate=muon_lr * float(os.environ.get("VIBY_KLSOAP_LR_MULT", "1.0")),
            weight_decay=_muon_wd,
            cautious=_muon_cautious,
            hyperball=muonh,
            basis_freq=int(os.environ.get("VIBY_KLSOAP_F", "1")),
        )
    else:
        muon_opt = BatchedMuon(
            learning_rate=muon_lr,  # Muon 用 muon 基础学习率（13/3 × adam_lr）
            momentum=0.95,
            weight_decay=_muon_wd,
            polar_ema=_polar_ema,
            normuon_beta2=0.95 if _normuon else 0.0,
            cautious_wd=_muon_cautious,
            ns_steps=int(getattr(args, "muon_ns_steps", 5)),
            hyperball=muonh,
            head_dim=hd,
            index_head_dim=hd_index,
            # muonh 加速旋钮（默认路径 muonh=False 时全部不影响数值）：
            # NS 迭代 bf16（范数仍 fp32）；逐专家 NS 的迭代步数
            ns_bf16=muonh and os.environ.get("VIBY_MUONH_NS_BF16", "1") == "1",
            stack_ns_steps=int(os.environ.get("VIBY_MUONH_STACK_NS_STEPS", "0"))
            or None,
            # 正交化固定每步全量重算。曾有的 NS 降频复用（EVERY=8）与
            # Temporal Q 缓存是 r082 回退的最大单项元凶（早期 −0.4~0.5 nat，
            # probe_p1/p5），已删除——负面结果见 BatchedMuon 类 docstring。
        )
    adamh_head = AdamH(
        learning_rate=muon_lr,  # AdamH 用 MuonH 的基础 lr（muon_lr），非 adam_lr
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=0.0,  # wd 是径向分量，范数球投影后近似无操作
        bias_correction=True,  # 公式拟合口径（optax 带修正）；不修正前期步长放大 7~15×
    )
    adamw_embed = FusedAdamW(
        learning_rate=adam_lr * embed_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=adam_wd,
        bias_correction=True,
        cautious=_adam_cautious,
    )
    # Engram 检索表：论文规格 Adam 5× lr、wd=0（稀疏 gather 更新，wd 会
    # 把大量未被检索的行径向缩没）
    adamw_ngram_table = FusedAdamW(
        learning_rate=adam_lr * embed_lr_mult * 5.0,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=0.0,
        bias_correction=True,
    )
    # Sinkhorn 均衡组（报告 §2.5 算法 1，超参 §4.2.2）：动量同 Muon、
    # γ=0.18、K=11、τ=1e-3、ε=1e-20、无 wd。两个实例只为 lr 口径分开：
    # Engram 检索表 5× lr（§4.2.2 引 Cheng et al., 2026b），token 嵌入与
    # 预测头 1× lr（都是「原来给这些参数的 AdamW lr」，γ 在类内部施加）。
    sinkhorn_engram = SinkhornBalanced(
        learning_rate=adam_lr * embed_lr_mult * _engram_lr_mult,
        momentum=_sink_beta,
        gamma=_sink_gamma,
        sinkhorn_steps=_sink_steps,
        eps=_sink_eps,
        tau=_sink_tau,
    )
    sinkhorn_embed = SinkhornBalanced(
        learning_rate=adam_lr * embed_lr_mult,
        momentum=_sink_beta,
        gamma=_sink_gamma,
        sinkhorn_steps=_sink_steps,
        eps=_sink_eps,
        tau=_sink_tau,
    )
    # 归一化层权重：AdamW 且**受 wd**（§2.5：「归一化层权重同样受权重衰减
    # 影响，而偏置与缩放因子不受」）。lr 沿用旧标量组的 adam_lr 口径。
    adamw_norm = FusedAdamW(
        learning_rate=adam_lr * scalar_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=adam_wd,
        bias_correction=True,
        cautious=_adam_cautious,
    )
    adamw_router = FusedAdamW(
        learning_rate=adam_lr * router_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=adam_wd,
        bias_correction=True,
        cautious=_adam_cautious,
    )
    # 偏置 / 缩放因子 / 默认不入 Muon 的 3D 专家栈：wd=0
    adamw_nowd = FusedAdamW(
        learning_rate=adam_lr * scalar_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=0.0,
        bias_correction=True,
    )

    # 为学习率调度器存储初始学习率
    muon_opt.base_lr = muon_lr
    sinkhorn_engram.base_lr = adam_lr * embed_lr_mult * _engram_lr_mult
    sinkhorn_embed.base_lr = adam_lr * embed_lr_mult
    adamw_norm.base_lr = adam_lr * scalar_lr_mult
    adamw_router.base_lr = adam_lr * router_lr_mult
    adamw_nowd.base_lr = adam_lr * scalar_lr_mult
    adamw_embed.base_lr = adam_lr * embed_lr_mult
    if not sinkhorn_on:  # 消融回退组（VIBY_SINKHORN=0）
        adamh_head.base_lr = muon_lr
        adamw_ngram_table.base_lr = adam_lr * embed_lr_mult * 5.0

    # MultiOptimizer: filters 数量 = len(optimizers) - 1，按顺序首个命中生效，
    # 未命中任何 filter 的参数落到最后一组（AdamW 无衰减兜底，不限 ndim）。
    # 空组必须剔除（如 tied embedding 时无 lm_head/AdamH 组、无 Engram 表）：
    # mlx MultiOptimizer 对空组会在首次 step 的 state init 抛 IndexError。
    # 默认（sinkhorn_on）：Muon → Sinkhorn(Engram 5×) → Sinkhorn(embed/头) →
    #                norm(wd) → router(wd) → no-wd 兜底
    # 回退（VIBY_SINKHORN=0）：Muon → AdamH(lm_head) → AdamW(embed) →
    #                AdamW(Engram 5×) → norm(wd) → router(wd) → no-wd 兜底
    optimizers = [muon_opt]
    counts = [muon_count]
    filters_all = [_is_muon]
    if sinkhorn_on:
        optimizers += [sinkhorn_engram, sinkhorn_embed]
        counts += [sinkhorn_engram_count, sinkhorn_embed_count]
        filters_all += [_is_ngram_sinkhorn, _is_sinkhorn]
        # PQ tables are 3-D embedding-like parameters, never Muon or flattened
        # Sinkhorn matrices. Use the embedding AdamW recipe also in this mode.
        codebook_count = sum(p.endswith(".ncp.codebook") for p, _ in trainable)
        if codebook_count:
            optimizers.append(adamw_embed)
            counts.append(codebook_count)
            filters_all.append(lambda path, arr: path.endswith(".ncp.codebook"))
            Logger(f"  - NCP codebook AdamW embedding group: {codebook_count} tensor")
    else:
        optimizers += [adamh_head, adamw_embed, adamw_ngram_table]
        counts += [adamh_count, embed_count, ngram_table_count]
        filters_all += [_is_adamh, _is_embed, _is_ngram_table]
    optimizers += [adamw_norm, adamw_router, adamw_nowd]
    counts += [norm_count, router_count, nowd_count]
    filters_all += [_is_norm_weight, _is_router, None]
    keep = [
        i
        for i, (c, f) in enumerate(zip(counts, filters_all))
        if c > 0 or f is None  # 无衰减组作为兜底永远保留
    ]
    return optim.MultiOptimizer(
        [optimizers[i] for i in keep],
        filters=[filters_all[i] for i in keep if filters_all[i] is not None],
    )
