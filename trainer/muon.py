"""
Muon 混合优化器（MLX 单设备版）

基于 mlx.optimizers.Muon / mlx.optimizers.MultiOptimizer 实现混合优化器：
- Muon 仅用于 ndim >= 2 且非嵌入/输出头/ShortConv/router 的核心权重矩阵
  （Newton-Schulz 正交化动量），weight_decay=0
- 嵌入/输出头与其余标量参数（含 ShortConv depthwise 卷积核）使用 AdamW
- pretrain 下 muon/adam 双基础 lr 与 Adam 组 beta2/eps 由
  trainer.utils.resolve_compute_scaled_hparams 按 Hyperball compute 公式写入 args
"""

import os
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

_adamw_kernels: dict = {}
_stack_mom_kernels: dict = {}
_stack_momv_kernels: dict = {}
_stack_apply_kernels: dict = {}
_stack_hit_kernels: dict = {}
_ns_core_fns: dict = {}
_cached_q_fns: dict = {}
_fro_norm_kernels: dict = {}


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


_hit_metal_kernels: dict = {}


def _hit_metal(G, P, V, X, lr, m, wd):
    """命中步单 kernel：||P|| / ||P−lrX|| + V 动量 + 超球投影。

    每个前导矩阵一个 threadgroup：先 f32 归约两个范数，再同一趟写 Pn/Vn。
    替代 shapeful compile 里两次 fro_norm + 一串逐元素核。
    """
    lead, r, c = P.shape[:-2], P.shape[-2], P.shape[-1]
    L = 1
    for d in lead:
        L *= d
    M = r * c
    NT = 256
    key = (M, P.dtype, float(m), float(wd))
    kern = _hit_metal_kernels.get(key)
    if kern is None:
        mt = "bfloat" if P.dtype == mx.bfloat16 else "float"
        kern = mx.fast.metal_kernel(
            name=f"muon_hit_{M}_{mt}_{wd != 0}",
            input_names=["G", "P", "V", "X", "lr_g"],
            output_names=["Pn", "Vn"],
            source=f"""
    constexpr uint NT = {NT};
    constexpr uint M = {M};
    constexpr float MOM = {float(m)}f;
    constexpr float WD = {float(wd)}f;
    uint n = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float sh0[NT / 32];
    threadgroup float sh1[NT / 32];
    size_t base = (size_t)n * M;
    float lr = float(lr_g[0]);
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint i = tid; i < M; i += NT) {{
        float p = float(P[base + i]);
        float np = p - lr * float(X[base + i]);
        acc0 += p * p;
        acc1 += np * np;
    }}
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (tid % 32 == 0) {{ sh0[tid / 32] = acc0; sh1[tid / 32] = acc1; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float s0 = 0.0f, s1 = 0.0f;
        for (uint k = 0; k < NT / 32; k++) {{ s0 += sh0[k]; s1 += sh1[k]; }}
        sh0[0] = metal::sqrt(s0);
        sh1[0] = metal::sqrt(s1);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = sh0[0] / metal::max(sh1[0], 1e-12f);
    for (uint i = tid; i < M; i += NT) {{
        float g = float(G[base + i]);
        float p = float(P[base + i]);
        float v = float(V[base + i]);
        float x = float(X[base + i]);
        if (WD != 0.0f) g += WD * p;
        Vn[base + i] = {mt}(MOM * v + (1.0f - MOM) * g);
        Pn[base + i] = {mt}((p - lr * x) * scale);
    }}
""",
        )
        _hit_metal_kernels[key] = kern
    Gf, Pf, Vf, Xf = G.reshape(L, M), P.reshape(L, M), V.reshape(L, M), X.reshape(L, M)
    lr_a = lr.astype(mx.float32).reshape((1,))
    Pn, Vn = kern(
        inputs=[Gf, Pf, Vf, Xf, lr_a],
        grid=(NT, L, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(L, M), (L, M)],
        output_dtypes=[P.dtype, V.dtype],
    )
    return Pn.reshape(P.shape), Vn.reshape(V.shape)


def _ns_core_fn(kind, bf16, steps, return_Q):
    """NS 核心的 mx.compile 版：把 5 步迭代里的逐元素链（B=bA+cA²、
    X=aX+B@X、入口 f32 范数+归一化）融进少量 kernel，GEMM 不变。

    数学上与 _ns5/_ns5_gram 的 eager 实现逐项相同（同样的算子顺序、
    同样的弱类型提升），只是 compile 消除中间物化。tr 转置规则、
    eye 初始化、Gram 的 Z/Q/R 递推全部照抄。compiled fn 只能返回数组，
    tr 标志由调用方按形状重新推导（确定性）。
    """
    key = (kind, bf16, steps, return_Q)
    fn = _ns_core_fns.get(key)
    if fn is not None:
        return fn
    a_, b_, c_ = BatchedMuon._ns_coeffs()

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
            X = X / (nrm + 1e-7).astype(X.dtype)
            R = X @ X.swapaxes(-1, -2)
            eye = mx.eye(R.shape[-1], dtype=X.dtype)
            Q = None
            for t in range(steps):
                Z = a_ * eye + b_ * R + c_ * (R @ R)
                Q = Z if Q is None else Q @ Z
                if t < steps - 1:
                    R = Z @ R @ Z
            Y = Q @ X
            if tr:
                Y = Y.swapaxes(-1, -2)
            Y = Y.astype(dt0) if Y.dtype != dt0 else Y
            if return_Q:
                return Y, Q
            return Y

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
            X = X / (n + 1e-7).astype(X.dtype)
            Q = mx.eye(X.shape[-2], dtype=X.dtype) if return_Q else None
            for _ in range(steps):
                A = X @ X.swapaxes(-1, -2)
                B = b_ * A + c_ * (A @ A)
                X = a_ * X + B @ X
                if Q is not None:
                    Q = a_ * Q + B @ Q
            if tr:
                X = X.swapaxes(-1, -2)
            X = X.astype(dt0) if X.dtype != dt0 else X
            if return_Q:
                return X, Q
            return X

    _ns_core_fns[key] = fn
    return fn


def _cached_q_fn(bf16):
    """命中步 D = Q @ normalize(X) 的 compile 版，归一化改为 GEMM 后
    标量：D = (Q@X) / n。与先归一化再 GEMM 数学相同（仅 f32 舍入顺序
    差异），省掉一次 (N,r,c) 归一化张量的物化（专家栈 1.1GB×2 流量）。
    """
    fn = _cached_q_fns.get(bf16)
    if fn is None:

        @mx.compile
        def fn(Q, X):
            dt0 = X.dtype
            if bf16 and dt0 != mx.bfloat16:
                X = X.astype(mx.bfloat16)
            nrm = _fro_norm_auto(X)  # 转置不变量，先算保住连续性
            tr = X.shape[-2] > X.shape[-1]
            if tr:
                X = X.swapaxes(-1, -2)
            Y = (Q @ X) / (nrm + 1e-7).astype(X.dtype)
            if tr:
                Y = Y.swapaxes(-1, -2)
            return Y.astype(dt0) if Y.dtype != dt0 else Y

        _cached_q_fns[bf16] = fn
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


def _stack_momv_kernel(m, wd):
    """只算新动量 V，不算 nesterov 后的 U。

    堆叠组的命中步（复用旧极因子）不跑 NS，U 是死值；但 compile 的输出
    签名固定，返回 U 就会真的算+写一遍（专家栈单份 1.27GB，约 3ms/组）。
    """
    key = (m, wd)
    fn = _stack_momv_kernels.get(key)
    if fn is None:

        @partial(mx.compile, shapeless=True)
        def fn(G, P, V):
            if wd != 0:
                G = G + wd * P
            return m * V + (1.0 - m) * G

        _stack_momv_kernels[key] = fn
    return fn


def _stack_hit_kernel(hyperball, m, wd):
    """命中步：动量 V + P−lr·X + 可选超球，收成一次 compile。

    刷新步仍走 mom / NS / apply。命中步方向是缓存的极因子，V 与 P
    更新互不依赖，可以同一趟做完。hyperball 版 shapeful：内部
    `_fro_norm` 的 grid 绑形状。
    """
    key = (hyperball, m, wd)
    fn = _stack_hit_kernels.get(key)
    if fn is None:
        if hyperball:

            def fn(G, P, V, X, lr):
                if G.dtype == mx.bfloat16 and G.ndim >= 2:
                    return _hit_metal(G, P, V, X, lr, m, wd)
                if wd != 0:
                    G = G + wd * P
                Vn = m * V + (1.0 - m) * G
                NP = P - lr * X
                n0 = _fro_norm_auto(P)
                n1 = _fro_norm_auto(NP)
                Pn = NP * (n0 / mx.maximum(n1, 1e-12)).astype(P.dtype)
                return Pn, Vn

        else:

            @partial(mx.compile, shapeless=True)
            def fn(G, P, V, X, lr):
                if wd != 0:
                    G = G + wd * P
                Vn = m * V + (1.0 - m) * G
                return P - lr * X, Vn

        _stack_hit_kernels[key] = fn
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
                NP = NP * (n0 / mx.maximum(n1, 1e-12)).astype(P.dtype)
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

    stack_cache_q=True（Temporal MuonH）：堆叠组刷新步用 Gram-NS 并缓存
    左变换 Q，命中步 D=Q@normalize(U)，使中间步仍吸收新动量（CacheMuon
    口径）；需配合 stack_ns_every>1。
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
        stack_ns_every=1,
        stack_cache_q=False,
        stack_cache_q_res=0.0,
        head_dim=0,
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
        # 堆叠专家组的 NS 步数/降频：专家动量 EMA(0.95) 半衰期 ~14 步，
        # NS 输出随动量缓变，每 stack_ns_every 步重算一次、中间步复用上
        # 次的正交化方向即可（近似质量损失很小，NS 算力省 stack_ns_every 倍）。
        self.stack_ns_steps = stack_ns_steps
        self.stack_ns_every = max(1, int(stack_ns_every))
        # Temporal MuonH（CacheMuon）：刷新步跑 NS5 并累积左变换 Q，
        # 命中步只做 Q@X_norm（跟新动量）。默认关闭，见 create_mixed_optimizer
        # 里的说明：Q 是对刷新步动量奇异值谱定制的逆缩放，动量一变输出就
        # 严重非正交（实测残差 4~13，而完整 NS5 是 0.02~0.03）。
        self.stack_cache_q = bool(stack_cache_q)
        # 命中步正交残差阈值：probe 上 ||DD^T-I||_F / n 超限则本步改走完整
        # NS5。0 关闭。避免固定 every 在动量漂移时 silently 漂。
        self.stack_cache_q_res = float(stack_cache_q_res)
        self._stack_ns_cache: dict = {}  # (b,r,c) -> (step, X) 或 (step, Q, tr)
        # 刷新周期用的本地步计数：self.state["step"] 是 mx.array，int() 会
        # 触发 host sync（每优化器步、每形状组一次）。相位在 resume 后从 0
        # 重新起算，只影响刷新时点、不影响数值。
        self._local_step = 0
        # K3 Per-Head Muon：Q/K/V 沿 head 切开做 NS。0 关闭。
        self.head_dim = int(head_dim or 0)

    @staticmethod
    def _is_qkv_path(path: str) -> bool:
        return any(
            p in ("q_proj", "k_proj", "v_proj", "kv_proj")
            for p in path.replace("/", ".").split(".")
        )

    @staticmethod
    def _ns_coeffs():
        return 3.4445, -4.7750, 2.0315

    def _ns5(self, X, steps=None, return_Q=False):
        """批量 Newton-Schulz：X (N, r, c)，batch 维无耦合（照抄基类规则）。
        ns_bf16 时迭代在 bf16 做（归一化范数仍 fp32，避免弱类型提升把
        整条链抬回 f32）。

        return_Q：同步累积左变换 Q（Q_{t+1}=aQ+B@Q，与 X 更新同型），使
        polar(X)≈Q@normalize(X)。X 的更新式与 return_Q=False 相同。
        """
        steps = steps or self.ns_steps
        tr = X.shape[-2] > X.shape[-1]
        r = _ns_core_fn("std", self.ns_bf16, steps, return_Q)(X)
        if return_Q:
            return r[0], r[1], tr
        return r

    def _ns5_gram(self, X, steps=None, return_Q=False):
        """Gram Newton-Schulz：在 XXᵀ 上迭代，算术上等价标准 NS5。
        迭代 GEMM 全部在短边空间（(N,r,r)³ 替代 (N,r,c) 的 XXᵀ/B@X），
        对 r<c 的专家栈（384<640 / 320<384）每迭代省 ~25% FLOPs。

        返回极因子；return_Q 时额外返回左变换 Q（短边空间，已转置约定）
        与 tr 标志，供 Temporal 命中步 D=Q@normalize(X)。
        """
        steps = steps or self.ns_steps
        tr = X.shape[-2] > X.shape[-1]
        r = _ns_core_fn("gram", self.ns_bf16, steps, return_Q)(X)
        if return_Q:
            return r[0], r[1], tr
        return r

    def _ns_auto(self, X, steps=None):
        """按矩阵形状选标准 NS5 或 Gram-NS（两者算术等价，见 _ns5_gram）。

        Gram 把每次迭代的 GEMM 搬到短边空间，r≠c 越悬殊越省；但方阵时短边
        等于长边，白搭一次 XXᵀ。实测 (59,768,768) 组 61→73ms、
        (2304,384,384) 专家栈 320→371ms 都是 Gram 更慢，而
        (2,6400,768) 是 12.9→5.2ms、(2304,768,384) 是 527→414ms 更快。

        return_Q 路径（Temporal 缓存，默认关闭）不走这里：两种实现的 Q
        分别活在原空间/短边空间，混用会让缓存与命中步的约定不一致。
        """
        r, c = X.shape[-2], X.shape[-1]
        if r == c:
            return self._ns5(X, steps)
        return self._ns5_gram(X, steps)

    def _apply_cached_Q(self, Q, X, tr):
        """命中步：D = Q @ normalize(X)（短边约定与缓存 Q 一致）。"""
        return _cached_q_fn(self.ns_bf16)(Q, X)

    @staticmethod
    def _orth_residual(D, probe=32):
        """廉价正交残差：前 probe 张上 ||DDᵀ-I||_F / n（短边在左）。"""
        Y = D[: min(int(D.shape[0]), probe)]
        if Y.shape[-2] > Y.shape[-1]:
            Y = Y.swapaxes(-1, -2)
        n = Y.shape[-2]
        A = Y @ Y.swapaxes(-1, -2)
        eye = mx.eye(n, dtype=A.dtype)
        err = mx.linalg.norm((A - eye).astype(mx.float32), axis=(-2, -1)) / n
        return mx.mean(err)

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
        self._local_step += 1

        mom_fn = _stack_mom_kernel(m, nest, wd)
        _stack_momv_kernel(m, wd)
        apply_fn = _stack_apply_kernel(self.hyperball)
        hit_fn = _stack_hit_kernel(self.hyperball, m, wd)

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

        for (r, c), items in groups.items():
            dt = flat_g[items[0][0]].dtype
            lr = lr0.astype(dt) * (max(1.0, r / c) ** 0.5)
            # 与堆叠专家同一降频：每 stack_ns_every 步重算 NS5，命中复用
            # 上次极因子 D。2D 不用 Gram（_ns_auto 在非方阵上与 NS5 有
            # 谱差，warmup 会把 MTP 打飞）；只跳过重复正交化。
            cache_key = ("2d", r, c, len(items))
            cache = self._stack_ns_cache.get(cache_key)
            need_refresh = (
                cache is None
                or self.stack_ns_every == 1
                or (self._local_step - cache[0]) >= self.stack_ns_every
            )
            if not need_refresh:
                # 2D 组栈只有几十 MB，整组一次 hit kernel 比逐张量
                # momv+apply+两次范数少一个数量级的发射。
                Xs = cache[1]
                G = mx.stack([flat_g[p].reshape(r, c) for p, _ in items])
                P = mx.stack([flat_p[p].reshape(r, c) for p, _ in items])
                V = mx.stack([state_v[p].reshape(r, c) for p, _ in items])
                Pn, Vn = hit_fn(G, P, V, Xs, lr)
                scatter_back(items, Vn, Pn)
                continue
            U, P, V, dt = momentum_stack(items, r, c)
            X = self._ns5(U)
            self._stack_ns_cache[cache_key] = (self._local_step, X)
            scatter_back(items, V, apply_fn(P, X, lr))

        # 堆叠组（ndim>2，逐专家语义）：axis0 为 batch 展平进 _ns5 的
        # (N, r, c) 批量 NS，batch 维无耦合；lr 缩放按矩阵形状 (r, c)。
        for (b, r, c), items in stack_groups.items():
            paths = [p for p, _ in items]
            # Temporal / 降频：每 stack_ns_every 步刷新正交化；
            # stack_cache_q 时缓存左变换 Q（命中 Q@U_norm），否则复用旧极因子 D。
            cache = self._stack_ns_cache.get((b, r, c))
            step_i = self._local_step
            need_refresh = (
                cache is None
                or self.stack_ns_every == 1
                or (step_i - cache[0]) >= self.stack_ns_every
            )
            dt = flat_g[paths[0]].dtype
            lr = lr0.astype(dt) * (max(1.0, r / c) ** 0.5)

            # 复用旧极因子的命中步整组没有 GEMM，只剩逐元素更新，此时把
            # G/P/V 堆成 (N,b,r,c) 再切回去是纯搬运：专家栈单份 1.27GB，
            # 三次 stack + scatter 实测 30ms/组/步。逐张量做同样的逐元素
            # 工作零拷贝（gate_up 组 64→39ms）。X 沿 axis0 切片本身连续。
            if not need_refresh and not self.stack_cache_q:
                # 专家栈单份 ~1GB，不能再 stack G/P/V。逐张量仍走融合
                # hit（momv+apply+超球一次），避免每层 4 个 kernel。
                Xs = cache[1]
                for i, (path, orig) in enumerate(items):
                    Pn, Vn = hit_fn(
                        flat_g[path].reshape(b, r, c),
                        flat_p[path].reshape(b, r, c),
                        state_v[path].reshape(b, r, c),
                        Xs[i],
                        lr,
                    )
                    new_v[path] = Vn.reshape(orig)
                    new_params[path] = Pn.reshape(orig).astype(flat_p[path].dtype)
                continue

            G = mx.stack([flat_g[p].reshape(b, r, c) for p in paths])
            P = mx.stack([flat_p[p].reshape(b, r, c) for p in paths])
            V = mx.stack([state_v[p].reshape(b, r, c) for p in paths])
            U, V = mom_fn(G, P, V)
            N = U.shape[0]
            Uflat = U.reshape(N * b, r, c)
            X = None
            if not need_refresh and self.stack_cache_q:
                _, Q, tr = cache
                if self.stack_cache_q_res > 0:
                    # 残差预检只在 probe 子集上算。整堆 Q@U 有 N·b 个矩阵
                    # （专家栈 2304 个），仅为了测残差就全量算一遍、超限后
                    # 再跑完整 NS5 是纯浪费：命中率低时整步慢 ~2×。
                    probe = min(Uflat.shape[0], 32)
                    Dp = self._apply_cached_Q(Q, Uflat[:probe], tr)
                    if bool((self._orth_residual(Dp) > self.stack_cache_q_res).item()):
                        need_refresh = True
                if not need_refresh:
                    X = self._apply_cached_Q(Q, Uflat, tr).reshape(N, b, r, c)
            if need_refresh:
                if self.stack_cache_q and self.stack_ns_every > 1:
                    # 刷新用 Gram-NS：算术等价标准 NS（test_muonh_cache
                    # cos>0.99），迭代 GEMM 全在短边空间，专家栈（r<c）
                    # 实测快 ~25%，且 Q 累积近免费
                    X, Q, tr = self._ns5_gram(
                        Uflat, steps=self.stack_ns_steps, return_Q=True
                    )
                    X = X.reshape(N, b, r, c)
                    self._stack_ns_cache[(b, r, c)] = (step_i, Q, tr)
                else:
                    # 与 run 81 对齐：堆叠专家一律 Gram-NS（短边迭代）。
                    # _ns_auto 会让方阵 384×384 改走标准 NS5，谱差在
                    # hyperball 下会被放大。
                    X = self._ns5_gram(Uflat, steps=self.stack_ns_steps).reshape(
                        N, b, r, c
                    )
                    self._stack_ns_cache[(b, r, c)] = (step_i, X)
            scatter_back(items, V, apply_fn(P, X, lr))

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
        return "embed_tokens" in path or "lm_head" in path

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
    - AdamW(embed)：embed_tokens / lm_head，lr = adam_lr，wd=0.1
    - AdamW(router)：base router/expert_bias，lr = adam_lr * 0.05
      （MOE_ROUTER_LR_MULT 可调），wd=0.1
    - AdamW(scalar)：其余全部（含 ShortConv 的 (4,C) depthwise 卷积核——
      不是矩阵语义，不做 NS 正交化），lr = adam_lr，wd=0.1
    （SFT 时 embed/scalar lr 分别乘 0.1/0.3，wd=0.01）

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
        return "embed_tokens" in path or "lm_head" in path

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
        # 3D 堆叠专家权重（*.experts.*）默认不走 Muon：基类 Muon 对 ndim>2
        # 会 reshape 成 (E, out*in) 整体正交化，跨专家耦合尺度；分进 AdamW 组。
        # muonh 下改由堆叠组逐专家 NS（见 BatchedMuon.apply_gradients）。
        # MoE router 也不走 Muon：正交化更新步长恒定偏大，会把路由打分持续
        # 推向失衡（实测 top-1 桶容量 C 从 ~6K 漂到 13K+，(E,C,D) 缓冲膨胀
        # 顶爆内存、吞吐掉 ~30%）；单独小 lr AdamW 组，靠 bias 均衡项兜底
        return (
            arr.ndim >= 2
            and not _is_embed(path, arr)
            and not _is_shortconv(path, arr)
            and (experts_in_muon or ".experts." not in path)
            and ".router." not in path
            and "attn_gate" not in path
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

    hd = int(getattr(getattr(model, "config", None), "head_dim", 0) or 0)
    muon_opt = BatchedMuon(
        learning_rate=muon_lr,  # Muon 用 muon 基础学习率（13/3 × adam_lr）
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        hyperball=muonh,
        head_dim=hd,
        # muonh 加速旋钮（默认路径 muonh=False 时全部不影响数值）：
        # NS 迭代 bf16（范数仍 fp32）；逐专家 NS 的步数/降频
        ns_bf16=muonh and os.environ.get("VIBY_MUONH_NS_BF16", "1") == "1",
        stack_ns_steps=int(os.environ.get("VIBY_MUONH_STACK_NS_STEPS", "0")) or None,
        # muonh 默认：每 8 步跑一次完整 Gram-NS5，命中步复用上次的极因子 D。
        # EVERY=1 即每步 NS5。
        stack_ns_every=int(
            os.environ.get("VIBY_MUONH_STACK_NS_EVERY", "8" if muonh else "1")
        ),
        # Temporal（命中步 D=Q@normalize(U)，跟新动量）默认关闭。
        # Q 是对刷新步动量奇异值谱定制的逆缩放（把 σ_min 拉到 1，谱范数
        # 可达 10²~10³），动量方向一变就按错误因子放大新方向：实测真实
        # 训练里命中步残差 ||DDᵀ−I||_F/n = 4~13，而完整 NS5 只有 0.02~0.03。
        # Muon 的更新语义依赖"各奇异方向等步长"，这种谱畸变正是它最不能
        # 接受的误差（hyperball 只修全局范数，不修谱形状）。残差门因此
        # 几乎每步都触发 fallback，等于每步既算 Q@U 又算完整 NS5 ——
        # 实测整步 1585→3098ms。复用旧 D 虽 stale（动量 EMA 0.95 半衰期
        # ~14 步，8 步内漂移可控），但每步都是精确正交的方向。
        stack_cache_q=muonh and os.environ.get("VIBY_MUONH_CACHE_Q", "0") == "1",
        stack_cache_q_res=float(os.environ.get("VIBY_MUONH_CACHE_Q_RES", "0.15"))
        if muonh
        else 0.0,
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
        weight_decay=adam_wd,
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
