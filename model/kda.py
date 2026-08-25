"""KDA（Kimi Delta Attention）：逐通道门控 delta 规则线性注意力。

Kimi Linear 3:1 混合中承担 "3" 的 local 层（global 层是 full-causal
NoPE GQA，见 attention.py）。全模型无 RoPE。

机制（fla/ops/kda 口径）：
  q/k/v = 线性投影 → 深度因果 ShortConv(kernel 4) + SiLU → 逐 head 无参
  RMS norm（q 折入 scale²、k 折入 scale——k 为单位 L2，满足 delta 规则
  稳定性 β‖k‖²<2；乘积口径与 l2norm+scale 一致）。
  逐通道 log 衰减 g = g_min·σ(e^{A_h} z)，g_min=−5，A_h 初始化为 0（f32）；
  z = f_b(f_a(x)) + dt_bias（低秩 + 逐通道偏置，Kimi Linear 的 z 路径保留）；
  写强度 β = σ(b(x))。delta 规则递推：S ← Diag(e^g)·S + βk(v − Sᵀk)ᵀ，
  o = qᵀS（S 布局 (B, H, K, V)，衰减作用在 K 轴）。
  输出：y = W_o[σ(W_g x) ⊙ RMSNorm(ō)]，W_g 满秩。无 RoPE。

训练前向：chunk 并行（C=16）。chunk 内 cumsum 得 gc，A 矩阵用
  e^{gc_i−gc_j} = e^{gc_i}·e^{−gc_j} 分解为 batched GEMM；e^{−gc} 一侧
  有上溢风险；K3 下界门把每步 g 锁在 (g_min, 0)，16 步累计 > −80，
  f32 安全。(I+L)⁻¹ 用严格下三角幂零的倍倍增
  求逆（4 轮 GEMM 精确）。跨 chunk 状态递推 T/C 步（_kda_scan），全程 f32。
  反向用手写 VJP（mx.custom_function）：状态链的 cotangent 按逆时间
  递推，每 chunk 8 个 GEMM；避免 autodiff 穿透 64 步循环（实测比
  默认 VJP 快 ~4×）。每 chunk 入态 S_c 与伪值 vt 在前向时物化供反向
  复用（内存换 ~1.5× 反向提速）。

解码：逐 token 递推（与 chunk 共用同一数学参考 _recurrent_kda），SSM
state 与 q/k/v conv 尾部存 KVCache.extras；投机解码在 verify 步把
逐步快照追加进 extras["kda_trace"]，rewind 用最近快照精确恢复
state 与 conv 尾部（普通截断无轨迹时回退为清空状态，与 ShortConv
同一近似口径）。

打包序列：conv 按 segment_ids 逐 tap 段掩码（与 ShortConv 同）；循环
状态在文档边界不重置（e^g 衰减使跨文档泄漏快速衰减，f 门可学习在
边界加大衰减；如需严格隔离请逐文档前向）。
"""

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .kernels.conv import causal_conv
from .kernels.kda_decode import kda_decode_step
from .kernels.kda_inner import kda_inner
from .kernels.kda_prep import kda_prep
from .kernels.kda_scan import _supported as _scan_supported
from .kernels.kda_scan import kda_scan_metal
from .norms import _rms_unit

_SCAN_KERNEL_VERIFIED: set = set()
# VIBY_KDA_SCAN=0 或 VIBY_FUSED_KERNELS=0：跨 chunk 扫描走 eager（run 81 口径）
_SCAN_KERNEL_DISABLED = (
    os.environ.get("VIBY_KDA_SCAN", "1") != "1"
    or os.environ.get("VIBY_FUSED_KERNELS", "1") == "0"
)
# VIBY_KDA_SCAN_ZSC=0：关闭 scan 反向的「状态 cotangent 恒零」特化
# （仅训练路径使用；评估/prefill 消费末态 S，cot_Sall 非零，自动走通用版）
_ZSC_ENABLED = os.environ.get("VIBY_KDA_SCAN_ZSC", "1") == "1"


def _scan_dispatch(qe, w, u, Aqk, kd, egl, S, zsc: bool = False):
    """chunk 扫描调度：优先融合 Metal kernel（fwd+bwd 各 1 次发射），
    首调用按形状键对照 eager 参考（_kda_scan 含 VJP）做在线校验，
    失败/不支持形状永久回退 eager。zsc=True 使用「状态 cotangent 恒零」
    的 bwd 变体（仅当调用方保证 Sall/末态的 cotangent 为零时合法）。"""
    global _SCAN_KERNEL_DISABLED
    NC, C, D = qe.shape[2], qe.shape[3], qe.shape[4]
    Dv = u.shape[-1]
    if (
        _SCAN_KERNEL_DISABLED
        or qe.dtype != mx.float32
        or not _scan_supported(NC, C, D, Dv)
    ):
        return _kda_scan(qe, w, u, Aqk, kd, egl, S)
    key = (NC, C, D, Dv, zsc)
    try:
        o, Sall = kda_scan_metal(qe, w, u, Aqk, kd, egl, S, zsc=zsc)
        if key not in _SCAN_KERNEL_VERIFIED:
            o_ref, Sall_ref = _kda_scan(qe, w, u, Aqk, kd, egl, S)
            mx.eval(o, Sall, o_ref, Sall_ref)  # 触发 JIT 编译
            # 相对误差：状态链的合法 FMA 顺序差会随 NC 累积，绝对阈值
            # 在大数值范围下误判；rel 1e-4 对 f32 重排足够宽松、对真实
            # bug（如索引错位 O(1) 偏差）仍然敏感。
            d1 = ((o - o_ref).abs().max() / (o_ref.abs().max() + 1e-12)).item()
            d2 = ((Sall - Sall_ref).abs().max() / (Sall_ref.abs().max() + 1e-12)).item()
            if max(d1, d2) > 1e-4:
                raise RuntimeError(f"kda_scan fused 校验失败 rel={max(d1, d2):.2e}")
            _SCAN_KERNEL_VERIFIED.add(key)
        return o, Sall
    except Exception:
        _SCAN_KERNEL_DISABLED = True
        return _kda_scan(qe, w, u, Aqk, kd, egl, S)


def _scan_prewarm(NC: int, C: int, D: int, Dv: int, B: int = 1, H: int = 1) -> bool:
    """compile 前预编译 + fwd/bwd 校验（梯度对照 eager）。返回 False=已回退。"""
    global _SCAN_KERNEL_DISABLED
    if _SCAN_KERNEL_DISABLED:
        return False
    key = (NC, C, D, Dv, False)
    if key in _SCAN_KERNEL_VERIFIED:
        return True
    try:
        # 稳定量级（egl→1、写侧小范数）：随机大范数输入的状态链本身
        # 数值爆炸，任何 FMA 重排都会指数发散，无法用于实现间对照。
        qe = (mx.random.normal((B, H, NC, C, D)) * 0.05).astype(mx.float32)
        w = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
        u = (mx.random.normal((B, H, NC, C, Dv)) * 0.3).astype(mx.float32)
        Aqk = (mx.random.normal((B, H, NC, C, C)) * 0.05).astype(mx.float32)
        kd = (mx.random.normal((B, H, NC, C, D)) * 0.02).astype(mx.float32)
        egl = mx.random.uniform(0.9, 1.0, (B, H, NC, D)).astype(mx.float32)
        S = (mx.random.normal((B, H, D, Dv)) * 0.1).astype(mx.float32)
        ins = [qe, w, u, Aqk, kd, egl, S]

        def fk(*a):
            o, Sall = kda_scan_metal(*a)
            return (o**2).sum() + (Sall**2).sum()

        def fe(*a):
            o, Sall = _kda_scan(*a)
            return (o**2).sum() + (Sall**2).sum()

        lg, gg = mx.value_and_grad(fk, argnums=list(range(7)))(*ins)
        lr, gr = mx.value_and_grad(fe, argnums=list(range(7)))(*ins)
        mx.eval(lg, lr, *gg, *gr)
        if abs(lg.item() - lr.item()) > 1e-3 * max(1.0, abs(lr.item())):
            raise RuntimeError(
                f"kda_scan prewarm loss 不一致 {lg.item()} vs {lr.item()}"
            )
        for a, b in zip(gg, gr):
            d = (a - b).abs().max().item()
            rel = d / (b.abs().max().item() + 1e-12)
            if rel > 1e-3:
                raise RuntimeError(f"kda_scan prewarm 梯度不一致 rel={rel:.2e}")
        _SCAN_KERNEL_VERIFIED.add(key)
        # ZSC 变体（训练路径实际使用）也必须在 compile 前完成校验，否则首个
        # 训练调用落在 compile trace 内，在线校验被迫放弃 → 静默永久回退
        # eager。loss 只依赖 o，使 cot_Sall 恒零——正是 ZSC 变体的合法条件。
        if _ZSC_ENABLED:
            key_z = (NC, C, D, Dv, True)
            if key_z not in _SCAN_KERNEL_VERIFIED:

                def fk_z(*a):
                    return (kda_scan_metal(*a, zsc=True)[0] ** 2).sum()

                def fe_z(*a):
                    return (_kda_scan(*a)[0] ** 2).sum()

                lz, gz = mx.value_and_grad(fk_z, argnums=list(range(7)))(*ins)
                lz_r, gz_r = mx.value_and_grad(fe_z, argnums=list(range(7)))(*ins)
                mx.eval(lz, lz_r, *gz, *gz_r)
                if abs(lz.item() - lz_r.item()) > 1e-3 * max(1.0, abs(lz_r.item())):
                    raise RuntimeError(
                        f"kda_scan zsc prewarm loss 不一致 {lz.item()} vs {lz_r.item()}"
                    )
                for a, b in zip(gz, gz_r):
                    d = (a - b).abs().max().item()
                    rel = d / (b.abs().max().item() + 1e-12)
                    if rel > 1e-3:
                        raise RuntimeError(
                            f"kda_scan zsc prewarm 梯度不一致 rel={rel:.2e}"
                        )
                _SCAN_KERNEL_VERIFIED.add(key_z)
        return True
    except Exception:
        _SCAN_KERNEL_DISABLED = True
        return False


KDA_CHUNK = 16  # chunk 内分解数值安全的衰减步数上限（见模块 docstring）
KDA_CONV_KERNEL = 4
# K3 下界门：g = g_min · σ(e^{A_h} z) ∈ (g_min, 0)。16-token tile 累计
# log-decay ∈ (−80, 0)，e^{80} 仍在 f32 / bf16 安全范围，不再硬钳 −4。
KDA_G_MIN = -5.0
# 仅给 test_kda 的「无界衰减会溢出」对照用；模块路径不再硬钳。
KDA_MAX_STEP_DECAY = 4.0


def _shift_right_tokens(x: mx.array, offset: int) -> mx.array:
    if offset == 0:
        return x
    pad = mx.zeros_like(x[:, :offset, ...])
    return mx.concatenate([pad, x[:, :-offset, ...]], axis=1)


class _KDAConv(nn.Module):
    """深度因果 1-D 卷积（kernel 4）+ SiLU。KDA 的 q/k/v 短程混合，是机制
    的一部分（随机初始化、正常训练），不同于 block 级 identity-init
    ShortConv。输入 (B, T, C)；解码传入 state=(B, K-1, C) 历史输入尾部，
    返回 (y, 新尾部)。"""

    def __init__(self, dim: int, kernel_size: int = KDA_CONV_KERNEL):
        super().__init__()
        self.dim = dim
        self.kernel_size = int(kernel_size)
        bound = self.kernel_size**-0.5
        self.weight = mx.random.uniform(-bound, bound, (self.kernel_size, dim))

    def __call__(
        self,
        x: mx.array,
        segment_ids: Optional[mx.array] = None,
        state: Optional[mx.array] = None,
    ):
        K = self.kernel_size
        w = self.weight.astype(x.dtype)
        if state is None:
            y = causal_conv(x, w, seg=segment_ids, silu=True)
            tail = None
        else:
            B, T, C = x.shape
            hist = mx.concatenate([state, x], axis=1)  # (B, K-1+T, C)
            y = causal_conv(hist, w, seg=None, silu=True)[:, K - 1 :, :]
            tail = hist[:, T:, :]
        return y, tail

    def tail_of(self, x: mx.array) -> mx.array:
        """训练/prefill 全序列前向后取解码用尾部（左补零到 K-1 宽）。"""
        B, T, C = x.shape
        K = self.kernel_size
        pad = mx.zeros((B, K - 1, C), dtype=x.dtype)
        return mx.concatenate([pad, x], axis=1)[:, -(K - 1) :, :]


def _recurrent_kda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    S0: Optional[mx.array] = None,
    collect: Optional[list] = None,
) -> tuple[mx.array, mx.array]:
    """逐 token 递推参考实现（f32）。q/k/v/log_g (B,H,T,D)，beta (B,H,T)，
    S0 (B,H,D,Dv)。解码（T 小）与数值对照共用。collect 非 None 时每步
    追加该步后的 state，供投机解码 rewind 精确回滚。"""
    B, H, T, D = q.shape
    Dv = v.shape[-1]
    S = (
        mx.zeros((B, H, D, Dv), dtype=mx.float32)
        if S0 is None
        else S0.astype(mx.float32)
    )
    outs = []
    for t in range(T):
        S = S * mx.exp(log_g[:, :, t])[..., None]
        kv = (S * k[:, :, t][..., None]).sum(axis=-2)  # (B,H,Dv)
        delta = beta[:, :, t][..., None] * (v[:, :, t] - kv)
        S = S + k[:, :, t][..., None] * delta[..., None, :]
        outs.append((S * q[:, :, t][..., None]).sum(axis=-2))
        if collect is not None:
            collect.append(S)
    return mx.stack(outs, axis=2), S


@mx.custom_function
def _kda_scan(
    qe: mx.array,  # (B,H,NC,C,D)  q·e^{gc}
    w: mx.array,  # (B,H,NC,C,D)  Afb@ke
    u: mx.array,  # (B,H,NC,C,Dv) Afb@v
    Aqk: mx.array,  # (B,H,NC,C,C) 下三角 qk 亲和
    kd: mx.array,  # (B,H,NC,C,D)  k·e^{gl−gc}（写出侧衰减）
    egl: mx.array,  # (B,H,NC,D)    e^{gl}，chunk 末累计衰减
    S0: mx.array,  # (B,H,D,Dv)
):
    """跨 chunk 状态扫描（f32）。逐 chunk：
        vt = u − w@S;  o = qe@S + Aqk@vt;  S = diag(egl)·S + kdᵀ@vt
    返回 o (B,H,NC,C,Dv) 与 Sall (B,H,NC+1,D,Dv)（含入态 S0，供 VJP 复用）。
    反向见 vjp：对 S 链的 cotangent 逆时间递推，避免 autodiff 穿透
    T/C 步循环（实测比默认 VJP 快数倍）。"""
    NC = qe.shape[2]
    S = S0
    outs = [None] * NC
    states = [S0]
    for c in range(NC):
        vt = u[:, :, c] - w[:, :, c] @ S
        outs[c] = qe[:, :, c] @ S + Aqk[:, :, c] @ vt
        S = S * egl[:, :, c][..., None] + mx.swapaxes(kd[:, :, c], -1, -2) @ vt
        states.append(S)
    return mx.stack(outs, axis=2), mx.stack(states, axis=2)


@_kda_scan.vjp
def _kda_scan_vjp(primals, cotangent, output):
    qe, w, u, Aqk, kd, egl, _S0 = primals
    NC = qe.shape[2]
    cot_o, cot_Sall = cotangent
    Sall = output[1]
    dS = mx.zeros_like(primals[6])
    dqe = [None] * NC
    dw = [None] * NC
    du = [None] * NC
    dAqk = [None] * NC
    dkd = [None] * NC
    degl = [None] * NC
    for c in range(NC - 1, -1, -1):
        dS = dS + cot_Sall[:, :, c + 1]
        Sc = Sall[:, :, c]
        do = cot_o[:, :, c]
        # vt = u − w@Sc；o = qe@Sc + Aqk@vt；S' = diag(egl)⊙Sc + kdᵀ@vt
        vt = u[:, :, c] - w[:, :, c] @ Sc  # 重算（省物化，C≪D）
        dvt = mx.swapaxes(Aqk[:, :, c], -1, -2) @ do + kd[:, :, c] @ dS
        dqe[c] = do @ mx.swapaxes(Sc, -1, -2)
        dAqk[c] = do @ mx.swapaxes(vt, -1, -2)
        dw[c] = -dvt @ mx.swapaxes(Sc, -1, -2)
        du[c] = dvt
        dkd[c] = mx.swapaxes(dS @ mx.swapaxes(vt, -1, -2), -1, -2)
        degl[c] = (Sc * dS).sum(axis=-1)
        dS = (
            mx.swapaxes(qe[:, :, c], -1, -2) @ do
            - mx.swapaxes(w[:, :, c], -1, -2) @ dvt
            + dS * egl[:, :, c][..., None]
        )
    dS0 = dS + cot_Sall[:, :, 0]
    return [
        mx.stack(dqe, axis=2),
        mx.stack(dw, axis=2),
        mx.stack(du, axis=2),
        mx.stack(dAqk, axis=2),
        mx.stack(dkd, axis=2),
        mx.stack(degl, axis=2),
        dS0,
    ]


def _chunk_kda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    S0: Optional[mx.array] = None,
    chunk_size: int = KDA_CHUNK,
    zero_state_cot: bool = False,
) -> tuple[mx.array, mx.array]:
    """chunk 并行训练前向（f32）。形状同 _recurrent_kda；T 非 C 整数倍时
    右侧零填充（k=0 不写状态、log_g=0 不衰减，对输出与最终状态无影响）。
    zero_state_cot=True 向调度器声明：返回的末态 S（即 Sall 全体）的
    cotangent 恒为零，允许 scan 反向走零协梯度特化（仅训练路径满足）。"""
    B, H, T, D = q.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
        log_g = mx.pad(log_g, [(0, 0), (0, 0), (0, pad), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    NC = (T + pad) // C

    def _chunked(x: mx.array) -> mx.array:
        return x.reshape(B, H, NC, C, *x.shape[3:])

    q, k, v, log_g, beta = (
        _chunked(q),
        _chunked(k),
        _chunked(v),
        _chunked(log_g),
        _chunked(beta),
    )
    # 融合 kernel 一次出 qe/ke/ki/kd/egl：chunk 内 inclusive cumsum（≤0）沿
    # 倒数第二维（stride=D，非连续），逐个中间量往返 HBM 时实测 f+b 4.45ms
    # 对 0.67ms 的 compulsory 下界；融合后累加链留在寄存器里，1.63ms。
    # ki 的 exp(−gc) 上溢风险由 C=16 约束（模块 docstring）。
    qe, ke, ki, kd, egl = kda_prep(q, k, log_g)
    # L / (I+L)⁻¹ / Aqk / w / u：合成 GEMM + 闭式 VJP（见 kda_inner）
    w, u, Aqk = kda_inner(qe, ke, ki, v, beta)

    S = (
        mx.zeros((B, H, D, Dv), dtype=mx.float32)
        if S0 is None
        else S0.astype(mx.float32)
    )
    o, Sall = _scan_dispatch(qe, w, u, Aqk, kd, egl, S, zsc=zero_state_cot)
    o = o.reshape(B, H, T + pad, Dv)
    if pad:
        o = o[:, :, :T]
    return o, Sall[:, :, NC]


class KDAAttention(nn.Module):
    """KDA local 层（无 RoPE、无 softmax 注意力；接口与 GQAAttention 对齐）。

    use_cache 时 past_key_value 必为 KVCache（由 VibyBlock 保证创建）：
    conv 尾部与 SSM state 存 extras，offset 同步推进供位置/长度审计。
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        d = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        proj = self.n_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d, proj, bias=False)
        self.k_proj = nn.Linear(d, proj, bias=False)
        self.v_proj = nn.Linear(d, proj, bias=False)
        self.q_conv = _KDAConv(proj)
        self.k_conv = _KDAConv(proj)
        self.v_conv = _KDAConv(proj)
        # 逐通道衰减门（低秩）与写强度门
        self.f_a_proj = nn.Linear(d, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, proj, bias=False)
        self.b_proj = nn.Linear(d, self.n_heads, bias=False)
        # K3 满秩输出门 W_g：y = W_o[σ(W_g x) ⊙ RMSNorm(ō)]
        self.g_proj = nn.Linear(d, proj, bias=False)
        H = self.n_heads
        # A_h 初始化为 0（K3）；仍叫 A_log 以免改 decode kernel 签名
        self.A_log = mx.zeros((H,))
        # 逐通道 z 偏置：直接按目标初始衰减率校准——r ~ U(0.001, 0.1)
        # nats/step（e-folding 记忆 10~1000 token，与 softplus 时代的
        # 初始分布一致），存 logit(r/|g_min|) 使 K3 门在 A_h=0 时
        # σ(z)=r/|g_min| ⇒ g = g_min·σ(z) = −r。z ∈ (−8.5, −3.9)；
        # 慢端 σ′≈r/5 很小（该侧衰减学得慢），快速整体调节靠
        # per-head 的 A_h（e^{A_h} 直接缩放 z）。
        r = mx.random.uniform(0.001, 0.1, (proj,))
        self.dt_bias = mx.log(r / (-KDA_G_MIN - r))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(proj, d, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        x: mx.array,
        position_embeddings=None,
        past_key_value=None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        causal_bias: Optional[mx.array] = None,
        mask_is_full: Optional[bool] = None,
        segment_ids: Optional[mx.array] = None,
    ):
        del position_embeddings, causal_bias
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        cache = past_key_value if use_cache else None
        extras = cache.extras if cache is not None else {}
        S0 = extras.get("kda_state")

        # pad 掩码（0/1）：pad 位清零 q/k/v（不写状态、不进 conv 历史），
        # 衰减置 0（state 原样穿过）。打包 segment_ids 走 conv 段掩码，
        # 与此正交。mask_is_full 由外层预算好（compile 图内不能 .item()）。
        has_pad = False
        if attention_mask is not None:
            # attention_mask 覆盖全历史（含 cache 前缀）；本层只消费当前
            # T 个位置的掩码
            am = attention_mask
            if am.shape[1] != T:
                am = am[:, -T:]
            has_pad = (
                (not mask_is_full)
                if mask_is_full is not None
                else bool(mx.any(am != 1).item())
            )
            pad_mul = am.astype(x.dtype)[..., None] if has_pad else None
        else:
            pad_mul = None

        # 融合单 GEMM：[q|k|v|f_a|g|b] 六投影一次乘完再切分（参数仍各自
        # 独立存储/优化，concat 只发生在前向运行时，autodiff 经 concat 回传
        # 各参数切片）。放大 GEMM 形状、减少 kernel 发射。
        # eval 下缓存拼接结果（权重经 model.update 整体替换，按身份失效；
        # object.__setattr__ 绕过 nn.Module 的参数登记），训练每步重拼。
        proj = H * D
        srcs = (
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.f_a_proj.weight,
            self.g_proj.weight,
            self.b_proj.weight,
        )
        w_in = None
        if not self.training:
            c = self.__dict__.get("_w_in_cache")
            if c is not None and all(s is t for s, t in zip(c[1], srcs)):
                w_in = c[0]
        if w_in is None:
            w_in = mx.concatenate(srcs, axis=0)
            if not self.training:
                object.__setattr__(self, "_w_in_cache", (w_in, srcs))
        # 用 mx.split 而不是六个 `fused[..., a:b]`：切片的 VJP 各自 scatter 进
        # 一份全宽零张量再相加（六写五加，全宽 74MB/层），split 的 VJP 是单次
        # concatenate。实测 f+b 19.20 → 15.46ms/层，逐位一致。
        q_in, k_in, v_in, fa, g_in, bl = mx.split(
            x @ w_in.T,
            [proj, 2 * proj, 3 * proj, 3 * proj + D, 4 * proj + D],
            axis=-1,
        )
        if pad_mul is not None:
            q_in, k_in, v_in = q_in * pad_mul, k_in * pad_mul, v_in * pad_mul
        if cache is not None:
            # state=None 的 prefill 段仍吃 segment_ids（段掩码）；有状态
            # 的 decode/verify 步不使用（解码不打包文档）。
            old_tails = (
                extras.get("q_conv"),
                extras.get("k_conv"),
                extras.get("v_conv"),
            )
            q, q_tail = self.q_conv(q_in, segment_ids=segment_ids, state=old_tails[0])
            k, k_tail = self.k_conv(k_in, segment_ids=segment_ids, state=old_tails[1])
            v, v_tail = self.v_conv(v_in, segment_ids=segment_ids, state=old_tails[2])
            if q_tail is None:  # 首次全序列前向（prefill）：从输入取尾部
                q_tail, k_tail, v_tail = (
                    self.q_conv.tail_of(q_in),
                    self.k_conv.tail_of(k_in),
                    self.v_conv.tail_of(v_in),
                )
            extras["q_conv"], extras["k_conv"], extras["v_conv"] = (
                q_tail,
                k_tail,
                v_tail,
            )
        else:
            q, _ = self.q_conv(q_in, segment_ids=segment_ids)
            k, _ = self.k_conv(k_in, segment_ids=segment_ids)
            v, _ = self.v_conv(v_in, segment_ids=segment_ids)

        # T==1 decode 快速路径：融合 Metal kernel 一次完成 rms 单位化、
        # log_g/β 门、SSM 状态递推与输出（替代下方 ~28 个小 kernel）。
        # pad 位语义（q/k/v 清零、不衰减）核内不表达，有 pad 时回退 eager。
        if cache is not None and T == 1 and pad_mul is None:
            S_in = S0 if S0 is not None else mx.zeros((B, H, D, D), dtype=mx.float32)
            r = kda_decode_step(
                q.reshape(B, H, D),
                k.reshape(B, H, D),
                v.reshape(B, H, D),
                self.f_b_proj(fa).reshape(B, H, D),
                bl.reshape(B, H),
                self.A_log,
                self.dt_bias,
                S_in,
                scale=self.scale,
            )
            if r is not None:
                out, S = r
                trace = extras.get("kda_trace")
                if trace is not None:
                    # 与下方 eager 分支同口径的快照（offset, S, 三个 conv 尾）
                    n_keep = self.q_conv.kernel_size - 1
                    hists = []
                    for proj_in, old in zip((q_in, k_in, v_in), old_tails):
                        if old is None:
                            old = mx.zeros(
                                (B, n_keep, proj_in.shape[-1]), proj_in.dtype
                            )
                        hists.append(mx.concatenate([old, proj_in], axis=1))
                    trace.append(
                        (
                            cache.offset + 1,
                            S,
                            hists[0][:, 1 : 1 + n_keep],
                            hists[1][:, 1 : 1 + n_keep],
                            hists[2][:, 1 : 1 + n_keep],
                        )
                    )
                extras["kda_state"] = S
                cache.offset += 1
                out = out.reshape(B, 1, H, D)
                gate = mx.sigmoid(g_in).reshape(B, 1, H, D)
                out = self.o_norm(out.astype(x.dtype)) * gate
                out = self.o_proj(out.reshape(B, 1, -1).astype(x.dtype))
                return self.resid_dropout(out), cache

        # 逐 head 无参 RMS norm + scale 折叠（k 单位 L2，q 携 scale²）
        q = (self.scale**2) * _rms_unit(q.reshape(B, T, H, D))
        k = self.scale * _rms_unit(k.reshape(B, T, H, D))
        v = v.reshape(B, T, H, D)

        a = self.f_b_proj(fa).reshape(B, T, H, D)
        # K3：g = g_min · σ(e^{A_h} z)，z = 低秩投影 + dt_bias
        z = a.astype(mx.float32) + self.dt_bias.reshape(H, D)
        # e^{A_h} 先显式扩到 (H,D) 再广播：(H,) 直接对 (B,T,H,D) 广播时，
        # A_log 的 VJP 是「沿 (0,1,3) 归约、保留中间轴」，MLX 走通用慢路径
        # （单独一项就 5.0ms/层）；扩成 (H,D) 后变成沿 (0,1) 的连续归约加
        # 一次 768 元素小归约，实测该段 f+b 5.72 → 1.42ms/层。
        e_a = mx.broadcast_to(mx.exp(self.A_log)[:, None], (H, D))
        log_g = KDA_G_MIN * mx.sigmoid(e_a * z)  # (B,T,H,D) f32，∈ (g_min, 0)
        beta = mx.sigmoid(bl.astype(mx.float32))  # (B,T,H)
        if pad_mul is not None:
            # pad 位不衰减（e^0=1）：state 原样穿过；写入已被 k=0 消除
            log_g = log_g * pad_mul.astype(mx.float32)[..., None]

        # (B,H,T,D)
        q = q.astype(mx.float32).transpose(0, 2, 1, 3)
        k = k.astype(mx.float32).transpose(0, 2, 1, 3)
        v = v.astype(mx.float32).transpose(0, 2, 1, 3)
        log_g = log_g.transpose(0, 2, 1, 3)
        beta = beta.transpose(0, 2, 1)  # (B,H,T)

        if cache is not None and (T == 1 or S0 is not None):
            # decode / MTP verify / 带状态的短前向：逐 token 递推
            trace = extras.get("kda_trace")
            steps = [] if trace is not None else None
            out, S = _recurrent_kda(q, k, v, log_g, beta, S0, collect=steps)
            if trace is not None:
                # 逐步快照 (state, q/k/v conv 尾部)：rewind 精确回滚。
                # 步 t 后的 conv 尾部 = 截至该步的最后 K-1 个 conv 输入。
                base = cache.offset
                n_keep = self.q_conv.kernel_size - 1
                hists = []
                for proj_in, old in zip((q_in, k_in, v_in), old_tails):
                    if old is None:
                        old = mx.zeros((B, n_keep, proj_in.shape[-1]), proj_in.dtype)
                    hists.append(mx.concatenate([old, proj_in], axis=1))
                for t in range(T):
                    trace.append(
                        (
                            base + t + 1,
                            steps[t],
                            hists[0][:, t + 1 : t + 1 + n_keep],
                            hists[1][:, t + 1 : t + 1 + n_keep],
                            hists[2][:, t + 1 : t + 1 + n_keep],
                        )
                    )
        else:
            # 训练且不使用 cache 时末态 S 不被下游消费，Sall 全体的
            # cotangent 恒零 → scan 反向走零协梯度特化（逐位等价）。
            out, S = _chunk_kda(
                q,
                k,
                v,
                log_g,
                beta,
                S0,
                zero_state_cot=self.training and cache is None and _ZSC_ENABLED,
            )
        if cache is not None:
            extras["kda_state"] = S
            cache.offset += T

        out = out.transpose(0, 2, 1, 3)  # (B,T,H,D)
        gate = mx.sigmoid(g_in).reshape(B, T, H, D)
        out = self.o_norm(out.astype(x.dtype)) * gate
        out = self.o_proj(out.reshape(B, T, -1).astype(x.dtype))
        return self.resid_dropout(out), cache
