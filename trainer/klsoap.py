"""KL-SOAP-H：KL 散度协方差估计的 SOAP + hyperball 范数球投影。

算法依据（两篇独立来源交叉验证）：
- NVIDIA《SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales》
  (arXiv 2607.20548) Algorithm 2 / Table 6：8B dense、30B-A3B、72B-A8B
  MoE、最多 3T token 的大规模预训练里 KL-SOAP 稳定且「consistent,
  albeit slight, edge over Muon」；结论原文「内存不是瓶颈时推荐
  KL-SOAP 优于 Muon」。其参考实现 Emerging-Optimizers 的 REKLS 类
  （Realtime Eigen KL-Soap）：当前梯度先并入 Kronecker 因子再更新
  特征基（消除大 batch 下 stale-preconditioner 的 slingshot 失稳）。
- Lin et al.《Understanding and Improving Shampoo and SOAP via
  Kullback–Leibler Minimization》(arXiv 2509.03378, ICLR 2026)：KL 加权
  的因子累积把条件数开根号（κ_KL = √κ_Shampoo），是 QR/eigh 数值
  稳定性的来源。

每步流程（对权重 W (r,c)，全部状态 f32）：
1. KL 因子累积（用上一步特征基的近似特征值，指数 −1，eps 下截断）：
     L ← β_kron·L + (1−β_kron)/c · G·Q_R·diag(λ_R⁻¹)·Q_Rᵀ·Gᵀ
     R ← β_kron·R + (1−β_kron)/r · Gᵀ·Q_L·diag(λ_L⁻¹)·Q_Lᵀ·G
2. 特征基更新（默认 F=1 每步，REKLS 口径）：Q ← QR(L·Q).Q（一次
   幂迭代 + QR；首个基更新步用 eigh 冷启动），λ 取新基的 Rayleigh 商
   diag(QᵀLQ)，按降序排序并置换 Q 列（逐项对齐 NVIDIA 参考实现
   soap_utils.orthogonal_iteration；MLX 的 qr/eigh 仅 CPU stream，
   统一内存下是 sync 不是拷贝）；动量 m 先转出旧基再转入新基
   （rebase），v 不动（参考实现同——平方不可旋转）。
3. 旋转梯度入基 G′ = Q_Lᵀ·G·Q_R，基内 Adam（β1/β2 + bias correction），
   N = m̂/(√v̂+ε)，转出 ΔW = Q_L·N·Q_Rᵀ。
4. apply 阶段复用 _stack_apply_kernel：cautious 掩码解耦衰减（默认开，
   wd=0.1）+ hyperball 投影（半径=衰减前的逐矩阵 F 范数，零初始化
   矩阵当步跳过投影——与 MuonH 同口径）。

与 BatchedMuon 的关键差异：
- lr 语义：SOAP 更新 RMS≈1（旋转保范数，基内 Adam 归一），不做
  Muon 的 max(1,r/c)^0.5 形状缩放；基准 lr 用 muon_lr（与 AdamH 的
  RMS≈1 更新 + 投影口径一致），VIBY_KLSOAP_LR_MULT 可调。
- MLA 合并投影不切段（整矩阵进 SOAP，与 NVIDIA 对照实验的
  non-split QKV 条件一致）；不做 per-head 切分。
- 状态：每矩阵 L/R/Q_L/Q_R/λ_L/λ_R/m/v，显存约为 2×参数 + 4(r²+c²)
  f32——论文自己也标注「memory 不是瓶颈时才推荐」。
- 优化器状态结构与 Muon 完全不同，从 Muon 系 checkpoint 续训时
  优化器状态需重来（权重不受影响）。

env：VIBY_KLSOAP=1 在 create_mixed_optimizer 里接管 Muon 组（默认关，
栈内未验证前不动默认路径）；VIBY_KLSOAP_LR_MULT（默认 1.0）；
VIBY_KLSOAP_F（特征基更新间隔，默认 1=REKLS 每步）；cautious/wd 沿用
VIBY_MUON_CAUTIOUS / VIBY_MUON_WD。
"""

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .muon import _stack_apply_kernel

_STATE_KEYS = ("L", "R", "QL", "QR", "eigL", "eigR", "m", "v")

# MLX 的 qr/eigh 仅支持 CPU stream（统一内存下无拷贝，只有一次 sync）。
# 为严格对齐参考实现（Emerging-Optimizers soap_utils.orthogonal_iteration /
# get_eigenbasis_eigh），基更新不做任何 GPU 替代改写。


def _orth_iter(Fm, Q, first):
    """正交迭代基更新，逐项对齐 NVIDIA soap_utils：
    first（首步）：eigh 冷启动，特征值降序、列对应排列；
    否则：Q ← QR(Fm @ Q).Q（power_iter_steps=1），λ = diag(QᵀFmQ)
    （Rayleigh 商），按 λ 降序排序并置换 Q 列。
    Fm (N,r,r) 对称半定 f32，Q (N,r,r) 正交。
    返回 (Q_new, eig_new, order)：order 为列置换（v 的坐标跟踪用，
    首步 v 全零故返回 None）。"""
    if first:
        w, Qn = mx.linalg.eigh(Fm, stream=mx.cpu)  # 升序
        rev = mx.arange(Fm.shape[-1] - 1, -1, -1)
        return Qn[..., rev], w[..., rev], None
    M = Fm @ Q
    Qn, _ = mx.linalg.qr(M, stream=mx.cpu)
    eig = mx.sum(Qn * (Fm @ Qn), axis=-2)  # diag(Qnᵀ Fm Qn)
    order = mx.argsort(-eig, axis=-1)  # 降序
    eig = mx.take_along_axis(eig, order, axis=-1)
    lead = Qn.shape[:-2]
    idx = mx.broadcast_to(order[..., None, :], lead + Qn.shape[-2:])
    Qn = mx.take_along_axis(Qn, idx, axis=-1)
    return Qn, mx.maximum(eig, 0.0), order


def _fresh_state(lead, r, c):
    """lazy state：L/R 零、Q 单位阵、eig 全 1（首步因子更新退化为标准
    Shampoo 的 GGᵀ/n——避免 NVIDIA 实现 eig=0 下截到 1/eps 的尺度爆发）；
    m/v 零。lead 为 batch 前导维（2D 组堆叠 N 或专家数 E）。"""
    return {
        "L": mx.zeros(lead + (r, r)),
        "R": mx.zeros(lead + (c, c)),
        "QL": mx.broadcast_to(mx.eye(r), lead + (r, r)),
        "QR": mx.broadcast_to(mx.eye(c), lead + (c, c)),
        "eigL": mx.ones(lead + (r,)),
        "eigR": mx.ones(lead + (c,)),
        "m": mx.zeros(lead + (r, c)),
        "v": mx.zeros(lead + (r, c)),
    }


def _kl_accumulate(st, G, beta_kron, eps):
    """KL 加权 Kronecker 因子累积（就地写回 st）。G (N,r,c) f32。"""
    r, c = G.shape[-2], G.shape[-1]
    wR = mx.maximum(st["eigR"], eps) ** -1.0 / c  # (N,c)
    T = G @ st["QR"]  # G·Q_R, (N,r,c)
    st["L"] = beta_kron * st["L"] + (1 - beta_kron) * (
        (T * wR[:, None, :]) @ T.swapaxes(-1, -2)
    )
    wL = mx.maximum(st["eigL"], eps) ** -1.0 / r  # (N,r)
    U = st["QL"].swapaxes(-1, -2) @ G  # Q_Lᵀ·G, (N,r,c)
    st["R"] = beta_kron * st["R"] + (1 - beta_kron) * (
        (U * wL[:, :, None]).swapaxes(-1, -2) @ U
    )


def _kl_group_step(G, P, st, lr, hp, apply_fn, do_basis, first, c1, c2):
    """一组同形状矩阵（堆叠 (N,r,c)）的一步 KL-SOAP-H，返回 (NP, st)。
    hp = (beta_kron, beta1, beta2, eps)；c1/c2 为 bias 修正系数。"""
    bk, b1, b2, eps = hp
    _kl_accumulate(st, G, bk, eps)
    if do_basis:
        QL_old, QR_old = st["QL"], st["QR"]
        st["QL"], st["eigL"], ordL = _orth_iter(st["L"], QL_old, first)
        st["QR"], st["eigR"], ordR = _orth_iter(st["R"], QR_old, first)
        # 动量 rebase：旧基转出 → 新基转入（旋转自动包含列置换）
        m_orig = QL_old @ st["m"] @ QR_old.swapaxes(-1, -2)
        st["m"] = st["QL"].swapaxes(-1, -2) @ m_orig @ st["QR"]
        if ordL is not None:
            # v 不可旋转（平方），但列排序置换必须跟踪：新坐标 (i,j) ↔
            # 旧坐标 (ordL[i], ordR[j])（参考实现 soap 的 QR 路径对
            # exp_avg_sq 做同样的逐轴置换）。eigh 冷启动时 v 全零，跳过。
            st["v"] = mx.take_along_axis(
                st["v"],
                mx.broadcast_to(ordL[..., None], st["v"].shape),
                axis=-2,
            )
            st["v"] = mx.take_along_axis(
                st["v"],
                mx.broadcast_to(ordR[..., None, :], st["v"].shape),
                axis=-1,
            )
    # 旋转入基 → 基内 Adam（bias 修正）→ 转出
    Gp = st["QL"].swapaxes(-1, -2) @ G @ st["QR"]
    st["m"] = b1 * st["m"] + (1 - b1) * Gp
    st["v"] = b2 * st["v"] + (1 - b2) * (Gp * Gp)
    N = (c1 * st["m"]) / (c2 * mx.sqrt(st["v"]) + eps)
    dW = st["QL"] @ N @ st["QR"].swapaxes(-1, -2)
    # cautious 解耦衰减 + hyperball 投影（_stack_apply_kernel 口径）
    return apply_fn(P, dW, lr), st


class KLSoaPH(optim.Optimizer):
    """KL-SOAP + hyperball（模块 docstring 有完整算法与出处）。

    接口与 BatchedMuon 一致：apply_gradients(gradients, parameters) 接收
    （可能嵌套的）树，返回新参数树；2D 矩阵按形状分组堆叠计算，3D
    堆叠专家以 axis0 为 batch 逐矩阵独立（与 MuonH 同语义）。ndim<2
    不会进本组（create_mixed_optimizer 的分组器保证）。状态全部 f32，
    lazy 初始化。
    """

    def __init__(
        self,
        learning_rate,
        beta_kron=0.95,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        cautious=True,
        hyperball=True,
        basis_freq=1,
        bias_correction=True,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_kron = float(beta_kron)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.cautious = bool(cautious)
        self.hyperball = bool(hyperball)
        # F=1 即 REKLS 每步更新（NVIDIA 的大 batch 稳定性修复）；
        # F>1 退化为 SOAP 的降频基更新（省算力，有 slingshot 风险记录）。
        self.basis_freq = int(basis_freq)
        self.bias_correction = bool(bias_correction)

    def init_single(self, parameter, state):
        """状态在 _get_st 里 lazy 初始化（需要按形状分组后才知道堆叠
        前导维）；基类 init 只建空 dict。"""
        return None

    def _get_st(self, path, shape, lead, flat_s):
        pre = path + "."
        if pre + "L" not in flat_s:
            return _fresh_state(lead, shape[-2], shape[-1])
        return {k: flat_s[pre + k] for k in _STATE_KEYS}

    def apply_gradients(self, gradients: dict, parameters: dict):
        if not self._initialized:
            self.init(gradients)
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)
        self.state["step"] = self.step + 1
        step = int(self.state["step"])

        flat_g = dict(tree_flatten(gradients))
        flat_p = dict(tree_flatten(parameters))
        flat_s = dict(tree_flatten(self.state))
        lr = self.learning_rate.astype(mx.float32)
        apply_fn = _stack_apply_kernel(self.hyperball, self.weight_decay, self.cautious)
        hp = (self.beta_kron, self.beta1, self.beta2, self.eps)

        # bias 修正系数（optax 口径，与 FusedAdamW 一致）
        if self.bias_correction:
            c1 = 1.0 / (1.0 - self.beta1**step)
            c2 = 1.0 / (1.0 - self.beta2**step) ** 0.5
        else:
            c1 = c2 = 1.0

        do_basis = (step % self.basis_freq) == 0
        # 首个基更新步用 eigh 冷启动（参考实现同：state["step"]==0 强制
        # eigh）；此后走幂迭代+QR。F=1 时即 step==1。
        first = step == self.basis_freq

        groups: dict = {}
        stack_paths = []
        for path, g in flat_g.items():
            if g.ndim == 2:
                groups.setdefault(g.shape, []).append(path)
            elif g.ndim == 3:
                stack_paths.append(path)
            # ndim 其它值不进本组（分组器保证），防御性跳过

        new_params = {}
        for (r, c), paths in groups.items():
            G = mx.stack([flat_g[p].astype(mx.float32) for p in paths])
            P = mx.stack([flat_p[p].astype(mx.float32) for p in paths])
            sts = [self._get_st(p, (r, c), (), flat_s) for p in paths]
            st = {k: mx.stack([s[k] for s in sts]) for k in _STATE_KEYS}
            NP, st = _kl_group_step(G, P, st, lr, hp, apply_fn, do_basis, first, c1, c2)
            for i, p in enumerate(paths):
                new_params[p] = NP[i].astype(flat_p[p].dtype)
                for k in _STATE_KEYS:
                    flat_s[p + "." + k] = st[k][i]

        for p in stack_paths:
            b, r, c = flat_g[p].shape
            G = flat_g[p].astype(mx.float32)
            P = flat_p[p].astype(mx.float32)
            st = self._get_st(p, (b, r, c), (b,), flat_s)
            NP, st = _kl_group_step(G, P, st, lr, hp, apply_fn, do_basis, first, c1, c2)
            new_params[p] = NP.astype(flat_p[p].dtype)
            for k in _STATE_KEYS:
                flat_s[p + "." + k] = st[k]

        self.state = tree_unflatten(list(flat_s.items()))
        return tree_unflatten(list(new_params.items()))
