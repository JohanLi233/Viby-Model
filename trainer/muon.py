"""
Muon 混合优化器（MLX 单设备版）

基于 mlx.optimizers.Muon / mlx.optimizers.MultiOptimizer 实现混合优化器：
- Muon/MuonH：ndim >= 2 且非嵌入/输出头/ShortConv/router/零初始化门的
  核心权重（含 muonh 下 3D 堆叠专家的逐专家 NS），weight_decay=0。
  正交化每步全量重算（NS 降频复用已删）。Q/K/V per-head NS 默认关。
- 嵌入/输出头、router、其余标量参数（ShortConv、1-D gain、KDA A_log/
  dt_bias、attn_gate / KDA g_proj / GatedNorm.gate_up）使用 AdamW；
  标量组 wd=0，embed/router wd=0.1（SFT 时 embed wd=0.01）
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
_stack_mom_kernels: dict = {}
_stack_apply_kernels: dict = {}
_ns_core_fns: dict = {}
_fro_norm_kernels: dict = {}

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


def _stack_apply_kernel(hyperball):
    """P - lr*X，可选范数球投影。

    hyperball 版用 shapeful compile：内部 _fro_norm 的 Metal kernel
    grid 依赖具体形状，shapeless 追踪会把首次形状固化给所有组。
    非投影版保持 shapeless（纯逐元素，一个图通吃所有形状）。
    """
    fn = _stack_apply_kernels.get(hyperball)
    if fn is None:
        if hyperball:

            @mx.compile
            def fn(P, X, lr):
                NP = P - lr * X
                n0 = _fro_norm_auto(P)
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
                return P - lr * X

        _stack_apply_kernels[hyperball] = fn
    return fn


def _adamw_kernel(b1, b2, eps, wd, bias_correction):
    """(b1,b2,eps,wd,bias_correction) 对应的 mx.compile 融合 AdamW 更新。

    超参必须是 python float 而不是 traced 输入：MLX 的弱类型提升下
    `float * bf16 -> bf16`，换成 f32 数组会把整条链抬到 f32，数值和基类不再
    逐位一致。lr 会随 scheduler 每步变、step 随窗口递增，作为数组入参传进来。

    bias_correction 必须开：compute 公式的 beta2/eps 是在带修正的 Adam
    （optax 口径）上拟合的；不修正时有效步长带一个随时间衰减的放大因子
    (1-b1^t)/sqrt(1-b2^t)，beta2=0.9998 时前百余窗口放大 7~15 倍
    （r081_gqa_qb run 在 warmup 末端把 KDA 衰减参数推进 exp(-gc) 溢出区，
    前向永久 NaN）。
    """
    key = (b1, b2, eps, wd, bias_correction)
    fn = _adamw_kernels.get(key)
    if fn is None:

        @partial(mx.compile, shapeless=True)
        def fn(p, g, m, v, lr, step):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * mx.square(g)
            # 算子顺序照抄 optim.AdamW.apply_single → Adam.apply_single：
            # 先解耦 wd 缩放参数，再减去 Adam 步
            p = p * (1 - lr * wd)
            if bias_correction:
                c1 = (lr / (1 - b1**step)).astype(p.dtype)
                c2 = mx.rsqrt(1 - b2**step).astype(p.dtype)
                p = p - (c1 * m) / (mx.sqrt(v) * c2 + eps)
            else:
                p = p - lr * m / (mx.sqrt(v) + eps)
            return p, m, v

        _adamw_kernels[key] = fn
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
    读写即可，数学与基类逐位一致（算子顺序与弱类型提升都照抄，含
    bias_correction 分支）。

    同形状张量再堆成 (N,·) 一次更新：标量组 ~100 个小核变成每个形状
    1 次发射。单组超过 256MB 仍逐张量，避免再物化一份大栈。
    """

    _STACK_BYTES = 256 << 20

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        p, m, v = _adamw_kernel(
            b1, b2, self.eps, self.weight_decay, self.bias_correction
        )(parameter, gradient, state["m"], state["v"], lr, self.step)
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
        )
        flat_g = tree_flatten(gradients)
        flat_p = dict(tree_flatten(parameters))
        gmap = dict(flat_g)
        groups: dict = {}
        for path, g in flat_g:
            groups.setdefault((g.shape, g.dtype), []).append(path)

        new_p = []
        for (shape, dt), paths in groups.items():
            lr = self.learning_rate.astype(dt)
            nbytes = int(dt.size) * len(paths)
            for s in shape:
                nbytes *= int(s)
            if len(paths) == 1 or nbytes > self._STACK_BYTES:
                for path in paths:
                    st = _tree_get(self.state, path)
                    p, m, v = fn(
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
    二阶矩给出，更新后 rescale 回更新前的 Frobenius 范数（fp32 算范数），
    与 MuonH 同一超球约束：方向/范数解耦。wd 是径向分量，投影后近似
    无操作，构造时置 0。lr 用 MuonH 的基础学习率而非 adam_lr。
    """

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        n0 = mx.linalg.norm(parameter.astype(mx.float32))
        p = super().apply_single(gradient, parameter, state)
        n1 = mx.linalg.norm(p.astype(mx.float32))
        return p * (n0 / mx.maximum(n1, 1e-12)).astype(p.dtype)

    def apply_gradients(self, gradients: dict, parameters: dict):
        # 超球投影是逐张量范数，不能走父类的同形状堆叠。
        return optim.Optimizer.apply_gradients(self, gradients, parameters)


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
        segment_map=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        self.hyperball = hyperball
        # bf16 NS（Marin 口径：正交化 bf16、范数 fp32）：NS 是迭代求精，
        # bf16 精度足够，GEMM 减半字节流量。仅 muonh 下默认开。
        self.ns_bf16 = ns_bf16
        # 堆叠专家组的 NS 迭代步数（None = 与 ns_steps 相同）。仅迭代
        # 次数，不是降频——正交化每步全量重算（降频复用损害见类 docstring）。
        self.stack_ns_steps = stack_ns_steps
        # K3 Per-Head Muon：Q/K/V 沿 head 切开做 NS。0 关闭。
        self.head_dim = int(head_dim or 0)
        # MLA 合并投影的行段分割：dotted path -> 行段尺寸。NS 与 lr
        # 缩放按段独立，与未合并逐矩阵正交化同语义。
        self.segment_map = segment_map or {}
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

    @staticmethod
    def _is_qkv_path(path: str) -> bool:
        return any(
            p in ("q_proj", "k_proj", "v_proj", "kv_proj")
            for p in path.replace("/", ".").split(".")
        )

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
        seg_groups: dict = {}
        singles = []
        for path, g in flat_g.items():
            if g.ndim < 2:
                singles.append(path)
                continue
            orig = g.shape
            if path in self.segment_map and g.ndim == 2:
                r, c = orig[0], orig[1]
                key = (r, c, tuple(self.segment_map[path]))
                seg_groups.setdefault(key, []).append((path, orig))
            elif g.ndim > 2:
                b, r = orig[0], orig[1]
                c = g.size // (b * r)
                stack_groups.setdefault((b, r, c), []).append((path, orig))
            elif (
                self.head_dim > 0
                and self._is_qkv_path(path)
                and orig[0] % self.head_dim == 0
            ):
                nh = orig[0] // self.head_dim
                stack_groups.setdefault((nh, self.head_dim, orig[1]), []).append(
                    (path, orig)
                )
            else:
                groups.setdefault(orig, []).append((path, orig))

        m, nest, wd = self.momentum, self.nesterov, self.weight_decay
        lr0 = self.learning_rate
        new_params = {}
        new_v = {}
        # snapshot 默认关闭；开启时整步只 int() 一次。否则每组一次
        # int(self.state["step"]) 就是每步每组一次 host sync（§3.5 纪律）。
        snap_step = int(self.state["step"]) if snapshot.active() else None

        mom_fn = _stack_mom_kernel(m, nest, wd)
        apply_fn = _stack_apply_kernel(self.hyperball)

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
            scatter_back(items, V, apply_fn(P, X, lr))

        # MLA 合并投影：NS 与 lr 缩放按行段独立（等价未合并的逐矩阵 Muon）
        for (r, c, sizes), items in seg_groups.items():
            dt = flat_g[items[0][0]].dtype
            U, P, V, dt = momentum_stack(items, r, c)
            split_at = []
            acc = 0
            for z in sizes[:-1]:
                acc += z
                split_at.append(acc)
            u_segs = mx.split(U, split_at, axis=1)
            p_segs = mx.split(P, split_at, axis=1)
            np_segs = []
            for rs, us, ps in zip(sizes, u_segs, p_segs):
                if self._edge:
                    xs = self._ns_edge(us, "std")
                elif self._no_ns:
                    xs = self._orth(us)
                else:
                    xs = self._ns5(us)
                lr_s = lr0.astype(dt) * (max(1.0, rs / c) ** 0.5)
                np_segs.append(apply_fn(ps, xs, lr_s))
            scatter_back(items, V, mx.concatenate(np_segs, axis=1))

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
                U, Vn = mom_fn(
                    flat_g[path].reshape(b, r, c),
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
                if self._edge:
                    Xo = self._ns_edge(U, "gram")
                elif self._no_ns:
                    Xo = self._orth(U)
                else:
                    Xo = self._ns5_gram(U, steps=self.stack_ns_steps)
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


def _mla_segment_map(model):
    """MLA 合并投影的 Muon 行段分割表：dotted path -> 行段尺寸。

    qkv_proj 按 [q, kv_down, k_rope] 段、kv_up_proj 按 [k_up, v_up] 段
    分别正交化，保持与未合并逐矩阵 Muon 相同的语义与 lr 缩放。
    """
    cfg = getattr(model, "config", None)
    if cfg is None or bool(getattr(cfg, "use_linear_attn", False)):
        return {}
    qk = cfg.head_dim + cfg.qk_rope_head_dim
    sizes = {
        "self_attn.qkv_proj.weight": [
            cfg.num_attention_heads * qk,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
        ],
        "self_attn.kv_up_proj.weight": [
            cfg.num_attention_heads * cfg.head_dim,
            cfg.num_attention_heads * cfg.head_dim,
        ],
    }
    out = {}
    for path, arr in tree_flatten(model.trainable_parameters()):
        for suffix, seg in sizes.items():
            if path.endswith(suffix) and arr.shape[0] == sum(seg):
                out[path] = seg
    return out


def create_adamw_optimizer(model, args, training_type="pretrain"):
    """全参数 AdamW（用于优化器杠杆对照实验；lr 需单独调优）。"""
    from .utils import Logger

    Logger("正在为优化器进行参数分组 (pure AdamW)")
    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        return (
            "embed_tokens" in path
            or "lm_head" in path
            or (".ngram." in path and arr.ndim >= 2)
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
    未解析时 getattr 兜底回退单一 args.learning_rate 旧行为）。
    Adam/AdamH 组的 beta2/eps 同样由 args.adam_beta2 / args.adam_eps 给出
    （兜底旧常数 0.95 / 1e-8），beta1 恒 0.9。

    参数分组：
    - Muon：ndim >= 2 且非嵌入/输出头/router/ShortConv 的核心权重矩阵，wd=0
    - AdamH(lm_head)：Adam 方向 + 范数球投影，lr = muon_lr（仅 muonh 下）
    - AdamW(embed)：embed_tokens / lm_head / ngram 表，lr = adam_lr，wd=0.1
    - AdamW(router)：base router/expert_bias，lr = adam_lr * 0.05
      （MOE_ROUTER_LR_MULT 可调），wd=0.1
    - AdamW(scalar)：其余全部（含 ShortConv、零初始化门、1-D gain /
      KDA A_log·dt_bias、AttnRes 查询），lr = adam_lr，**wd=0**
      （这些参数要么零初始化、要么有校准过的尺度；wd=0.1 会在 1-epoch
      内把它们径向缩到 ~10%）
    （SFT 时 embed/scalar lr 分别乘 0.1/0.3；embed wd=0.01，scalar wd=0）

    --muonh（MuonH/AdamH/Adam 体系，Marin 口径，默认开启）：
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
    experts_in_muon = muonh and os.environ.get("VIBY_MUONH_EXPERTS", "1") == "1"

    # 双基础 lr + Adam 组超参（pretrain 由 resolve_compute_scaled_hparams
    # 写入；未写入时回退单一 lr 与旧常数）
    adam_lr = float(getattr(args, "adam_lr", None) or args.learning_rate)
    muon_lr = float(getattr(args, "muon_lr", None) or args.learning_rate)
    beta2 = float(getattr(args, "adam_beta2", None) or 0.95)
    eps = float(getattr(args, "adam_eps", None) or 1e-8)

    def _is_embed(path, arr):
        # 精确匹配嵌入/输出头路径段："embed" 子串会误吞 embed_norm 的
        # gate_down/gate_up（2-D 矩阵，按 spec 属 Muon 组）
        return (
            "embed_tokens" in path
            or "lm_head" in path
            or (".ngram." in path and arr.ndim >= 2)
        )

    def _is_adamh(path, arr):
        # lm_head readout：Adam 方向 + 范数球投影（AdamH 类），仅 muonh 下
        return muonh and "lm_head" in path and arr.ndim >= 2

    def _is_shortconv(path, arr):
        # 深度短卷积核 (4, C)：不是矩阵语义，不做 NS 正交化，落到 AdamW
        # 标量组。含 GQA k_conv / o·mlp ShortConv，以及 KDA q/k/v_conv。
        return any(
            s in path
            for s in (
                ".q_conv.",
                ".k_conv.",
                ".v_conv.",
                ".out_conv.",
                ".mlp_out_conv.",
            )
        )

    def _is_muon(path, arr):
        # 3D 堆叠专家：muonh 下（默认）由堆叠组逐专家 NS；`--no_muonh` 或
        # VIBY_MUONH_EXPERTS=0 时排除进 AdamW，避免基类 reshape (E,out·in)
        # 跨专家耦合。MoE router 也不走 Muon：正交化更新步长恒定偏大，会把
        # 路由打分持续推向失衡；单独小 lr AdamW 组。
        # GatedNorm.gate_up / attn_gate / KDA g_proj 零初始化，按 Marin
        # 口径零初始化门进 Adam；hyperball 半径=初始范数=0 会把它永久钉零
        # （见 _stack_apply_kernel）。g_proj 用 ".g_proj." 以免误伤 gate_proj。
        # leaf 精确匹配 gate_up，不误伤专家堆叠 gate_up_w。
        return (
            arr.ndim >= 2
            and not _is_embed(path, arr)
            and not _is_shortconv(path, arr)
            and (experts_in_muon or ".experts." not in path)
            and ".router." not in path
            and "attn_gate" not in path
            and ".g_proj." not in path
            and path.rsplit(".", 1)[-1] != "gate_up"
        )

    def _is_router(path, arr):
        return ".router." in path

    muon_count = sum(1 for p, a in trainable if _is_muon(p, a))
    adamh_count = sum(1 for p, a in trainable if _is_adamh(p, a) and not _is_muon(p, a))
    embed_count = sum(
        1
        for p, a in trainable
        if _is_embed(p, a) and not _is_muon(p, a) and not _is_adamh(p, a)
    )
    router_count = sum(
        1 for p, a in trainable if _is_router(p, a) and not _is_muon(p, a)
    )
    scalar_count = sum(
        1
        for p, a in trainable
        if not _is_muon(p, a)
        and not _is_adamh(p, a)
        and not _is_embed(p, a)
        and not _is_router(p, a)
    )

    Logger("参数分组完成：")
    Logger(
        f"  - {'MuonH(hyperball)' if muonh else 'Muon'} 参数组 (核心权重): "
        f"{muon_count} 个张量"
    )
    if muonh:
        Logger(f"  - AdamH 参数组 (lm_head): {adamh_count} 个张量")
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - MoE router 参数组: {router_count} 个张量")
    Logger(f"  - 标量参数组: {scalar_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, scalar_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, scalar_lr_mult, adam_wd = 1.0, 1.0, 0.1
    # router 需要远小于 AdamW 标量组的 lr：AdamW 每坐标步长≈lr，adam base lr
    # 下 ~10 步就把 (32,768) 的 router 权重打乱到 sigmoid
    # 饱和（实测 C 瞬间冲到 14K、吞吐 -45%）；0.05× 让其慢速移动、
    # bias 均衡项压得住负载。可用 MOE_ROUTER_LR_MULT 覆盖。
    router_lr_mult = float(
        os.environ.get("MOE_ROUTER_LR_MULT", getattr(args, "router_lr_mult", 0.05))
    )

    # per-head NS（Q/K/V 按 head 切分正交化）默认关闭：r082 归因隔离实测
    # 单独就有害（−0.14 nat @500 步，probe_p8 vs p7），且在 NS 降频 regime
    # 下与 stale 极因子叠加放大到 −0.26 nat（probe_p2）。
    # VIBY_MUONH_PER_HEAD=1 可重新打开（消融用）。
    hd = int(getattr(getattr(model, "config", None), "head_dim", 0) or 0)
    if os.environ.get("VIBY_MUONH_PER_HEAD", "0") != "1":
        hd = 0
    muon_opt = BatchedMuon(
        learning_rate=muon_lr,  # Muon 用 muon 基础学习率（13/3 × adam_lr）
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        hyperball=muonh,
        head_dim=hd,
        segment_map=_mla_segment_map(model),
        # muonh 加速旋钮（默认路径 muonh=False 时全部不影响数值）：
        # NS 迭代 bf16（范数仍 fp32）；逐专家 NS 的迭代步数
        ns_bf16=muonh and os.environ.get("VIBY_MUONH_NS_BF16", "1") == "1",
        stack_ns_steps=int(os.environ.get("VIBY_MUONH_STACK_NS_STEPS", "0")) or None,
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
    )
    adamw_router = FusedAdamW(
        learning_rate=adam_lr * router_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=adam_wd,
        bias_correction=True,
    )
    adamw_scalar = FusedAdamW(
        learning_rate=adam_lr * scalar_lr_mult,
        betas=[0.9, beta2],
        eps=eps,
        weight_decay=0.0,
        bias_correction=True,
    )

    # 为学习率调度器存储初始学习率
    muon_opt.base_lr = muon_lr
    adamh_head.base_lr = muon_lr
    adamw_embed.base_lr = adam_lr * embed_lr_mult
    adamw_router.base_lr = adam_lr * router_lr_mult
    adamw_scalar.base_lr = adam_lr * scalar_lr_mult

    # MultiOptimizer: filters 数量 = len(optimizers) - 1，按顺序首个命中生效，
    # 未命中任何 filter 的参数落到最后一组（AdamW 标量兜底，不限 ndim）。
    # 空组必须剔除（如 tied embedding 时无 lm_head/AdamH 组）：mlx
    # MultiOptimizer 对空组会在首次 step 的 state init 抛 IndexError。
    optimizers = [
        muon_opt,
        adamh_head,
        adamw_embed,
        adamw_router,
        adamw_scalar,
    ]
    counts = [
        muon_count,
        adamh_count,
        embed_count,
        router_count,
        scalar_count,
    ]
    filters_all = [
        _is_muon,
        _is_adamh,
        _is_embed,
        _is_router,
        None,
    ]
    keep = [
        i
        for i, (c, f) in enumerate(zip(counts, filters_all))
        if c > 0 or f is None  # scalar 组作为兜底永远保留
    ]
    return optim.MultiOptimizer(
        [optimizers[i] for i in keep],
        filters=[filters_all[i] for i in keep if filters_all[i] is not None],
    )
