"""EigenGate：KDA 快权重状态的酉等变谱高通。

默认关（VIBY_EIGENGATE=0），与现役 KDA 逐位一致。打开后每隔 K 个
chunk（默认 16 → 256 token）对状态做一次无 SVD 的 cubic 谱收缩：

    S ← (1−λ) S + λ · S g(SᵀS)

g 由 relaxed-cubic（与 Muon cubic5 同一族，l0 默认 0.05）作用在
F-归一后的状态上得到极因子 P≈U τ(σ) Vᵀ，再取 Gram 掩码 PᵀP / u²，
使 f(σ)=σ·τ(σ)²/u²：膝盖以下 →0，以上 →σ（目标 A，保幅）。
目标 B（VIBY_EIGENGATE_TARGET=whiten）改为 λ-混合极因子本身。

实现全部是 batched GEMM（短边 Gram，D=96 时 4–5 步 × 2 次 96³），
不改 delta 递推、不新增状态内存。训练扫描把门控收进同一次 Metal
发射（不再按段切开）。λ 默认常数 1；VIBY_EIGENGATE_LEARN=1
时每头一个可学标量（进 Adam 标量组）。
"""

from __future__ import annotations

import os
from math import sqrt

import mlx.core as mx

U_PEAK = 1.3
_APPLY_FNS: dict = {}


def enabled() -> bool:
    return os.environ.get("VIBY_EIGENGATE", "0") == "1"


def period_chunks() -> int:
    return max(1, int(os.environ.get("VIBY_EIGENGATE_K", "16")))


def period_tokens(chunk_size: int = 16) -> int:
    return period_chunks() * int(chunk_size)


def mix_lambda() -> float:
    return float(os.environ.get("VIBY_EIGENGATE_LAM", "1"))


def learnable() -> bool:
    return os.environ.get("VIBY_EIGENGATE_LEARN", "0") == "1"


def stop_grad_enabled() -> bool:
    return os.environ.get("VIBY_EIGENGATE_STOPGRAD", "0") == "1"


def target_name() -> str:
    t = os.environ.get("VIBY_EIGENGATE_TARGET", "keep").lower()
    return "whiten" if t in ("whiten", "b", "polar") else "keep"


def l0_value() -> float:
    return float(os.environ.get("VIBY_EIGENGATE_L0", "0.05"))


def should_gate(n_done: int, period: int | None = None) -> bool:
    if period is None:
        period = period_tokens()
    return period > 0 and n_done > 0 and (n_done % period) == 0


def lam_for(module) -> float | mx.array | None:
    """模块级 λ：关闭时 None；可学时 clip 到 [0,1] 的 (H,)；否则 env 常数。"""
    if not enabled():
        return None
    raw = getattr(module, "eigengate_lam", None)
    if raw is None:
        return mix_lambda()
    return mx.clip(raw.astype(mx.float32), 0.0, 1.0)


def cubic_coeffs(
    l0: float | None = None,
    u: float = U_PEAK,
    min_steps: int = 4,
    max_steps: int = 8,
    steps: int | None = None,
) -> list[tuple[float, float]]:
    """Chen–Chow relaxed cubic（arXiv 2606.00371 §2.2）。返回 [(a,b), ...]。"""
    if l0 is None:
        l0 = l0_value()
    env_steps = os.environ.get("VIBY_EIGENGATE_STEPS")
    if steps is None and env_steps:
        steps = int(env_steps)

    def gen(n: int):
        coeffs = []
        lo, r = float(l0), 1.0
        for t in range(n):
            if t > 0:
                r = u
            k2 = (r * r + r * lo + lo * lo) / 3.0
            alpha = 1.0 / sqrt(k2)
            a, b = 1.5 * u * alpha, -0.5 * u * alpha**3
            coeffs.append((float(a), float(b)))
            lo = a * lo + b * lo**3
        return coeffs, lo

    if steps is not None:
        return gen(int(steps))[0]
    for n in range(min_steps, max_steps + 1):
        coeffs, lfin = gen(n)
        if lfin >= 0.7:
            return coeffs
    return gen(max_steps)[0]


def scan_segments(NC: int, T: int, C: int, t0: int = 0) -> list[tuple[int, bool]]:
    """把 NC 个 chunk 切成段。每段 (length, gate_after)。

    门控点按**真实 token 计数** t0+min((c+1)C, T)，不是 padding 后的
    chunk 下标——否则 T 非 C 整数倍时会与逐 token 递推分叉。
    """
    if NC <= 0:
        return []
    period = period_tokens(C)
    segs: list[tuple[int, bool]] = []
    start = 0
    for c in range(NC):
        real_done = t0 + min((c + 1) * C, T)
        gate = should_gate(real_done, period)
        if gate or c == NC - 1:
            segs.append((c + 1 - start, gate))
            start = c + 1
    return segs


def chunk_gate_mask(NC: int, T: int, C: int, t0: int = 0) -> list[int]:
    """每个 chunk 写完后是否门控：1/0，长度 NC。按真实 token 计数。"""
    if NC <= 0:
        return []
    period = period_tokens(C)
    return [
        1 if should_gate(t0 + min((c + 1) * C, T), period) else 0 for c in range(NC)
    ]


def needs_split(NC: int, T: int, C: int, t0: int = 0) -> bool:
    """True：本序列至少一次门控。保留给 learnable/whiten 回退切段。"""
    if not enabled() or NC <= 0:
        return False
    return any(chunk_gate_mask(NC, T, C, t0))


def _broadcast_lam(lam, S: mx.array) -> mx.array:
    if isinstance(lam, (int, float)):
        return mx.array(float(lam), dtype=S.dtype)
    lam = lam.astype(S.dtype)
    if lam.ndim == 0:
        return lam
    # (H,) → 对齐 S 的 head 轴： (B,H,D,Dv) 的 axis 1
    shape = [1] * S.ndim
    if lam.ndim == 1 and S.ndim >= 3:
        shape[-3] = lam.shape[0]
        return lam.reshape(shape)
    return lam


def _polar_fn(coeffs, m: int, n: int):
    left = m <= n

    @mx.compile
    def fn(S: mx.array) -> mx.array:
        fro2 = (S * S).sum(axis=(-2, -1), keepdims=True)
        X = S / (mx.sqrt(fro2) + 1e-7)
        for a, b in coeffs:
            if left:
                A = X @ mx.swapaxes(X, -1, -2)
                X = a * X + b * (A @ X)
            else:
                A = mx.swapaxes(X, -1, -2) @ X
                X = a * X + b * (X @ A)
        return X

    return fn


def polar(S: mx.array, coeffs: list[tuple[float, float]] | None = None) -> mx.array:
    """F-归一 relaxed-cubic 极因子（膝盖映射）。"""
    if coeffs is None:
        coeffs = cubic_coeffs()
    m, n = int(S.shape[-2]), int(S.shape[-1])
    key = ("polar", tuple(coeffs), m, n)
    fn = _APPLY_FNS.get(key)
    if fn is None:
        fn = _polar_fn(coeffs, m, n)
        _APPLY_FNS[key] = fn
    return fn(S)


def ste_weight(
    S: mx.array,
    lam=1.0,
    coeffs=None,
    prefer: str = "auto",
) -> mx.array:
    """W = (1−λ)I + λ (PᵀP)/u²，P=stop_grad(polar(S))。

    prefer=left → W 是 (D,D)，S_out = W @ S；right → W 是 (Dv,Dv)，S_out = S @ W。
    """
    if coeffs is None:
        coeffs = cubic_coeffs()
    P = mx.stop_gradient(polar(S, coeffs))
    m, n = int(S.shape[-2]), int(S.shape[-1])
    if prefer == "left":
        left = True
    elif prefer == "right":
        left = False
    else:
        left = m <= n
    u2 = float(U_PEAK * U_PEAK)
    lam_b = _broadcast_lam(lam, S)
    if left:
        M = (P @ mx.swapaxes(P, -1, -2)) / u2
        eye = mx.eye(m, dtype=S.dtype)
        return (1.0 - lam_b) * eye + lam_b * M
    M = (mx.swapaxes(P, -1, -2) @ P) / u2
    eye = mx.eye(n, dtype=S.dtype)
    return (1.0 - lam_b) * eye + lam_b * M


def _highpass_fn(coeffs, target: str, prefer: str, m: int, n: int):
    """compiled S → S_hp（无混合）。prefer=left/right/auto。"""
    left_iter = m <= n
    if prefer == "left":
        left_mask = True
    elif prefer == "right":
        left_mask = False
    else:
        left_mask = left_iter
    polar_fn = _polar_fn(coeffs, m, n)
    u2 = float(U_PEAK * U_PEAK)
    keep = target != "whiten"

    @mx.compile
    def fn(S: mx.array) -> mx.array:
        P = polar_fn(S)
        if not keep:
            fro2 = (S * S).sum(axis=(-2, -1), keepdims=True)
            return P * mx.sqrt(fro2)
        if left_mask:
            M = P @ mx.swapaxes(P, -1, -2)
            return (M / u2) @ S
        M = mx.swapaxes(P, -1, -2) @ P
        return S @ (M / u2)

    return fn


def highpass(
    S: mx.array,
    coeffs: list[tuple[float, float]] | None = None,
    target: str | None = None,
    prefer: str = "auto",
) -> mx.array:
    """目标 A（keep）：S g(SᵀS)；目标 B（whiten）：||S||_F · polar(S)。"""
    if coeffs is None:
        coeffs = cubic_coeffs()
    tgt = target if target is not None else target_name()
    m, n = int(S.shape[-2]), int(S.shape[-1])
    key = (tuple(coeffs), tgt, prefer, m, n)
    fn = _APPLY_FNS.get(key)
    if fn is None:
        fn = _highpass_fn(coeffs, tgt, prefer, m, n)
        _APPLY_FNS[key] = fn
    return fn(S)


def apply(
    S: mx.array,
    lam=None,
    coeffs: list[tuple[float, float]] | None = None,
    target: str | None = None,
    stop_grad: bool | None = None,
    prefer: str = "auto",
) -> mx.array:
    """S ← W S 或 S W。polar 直通（不反传 cubic），反向只过 W。

    λ=0 的 python 标量短路为恒等。target=whiten 仍走极因子混合（少用）。
    VIBY_EIGENGATE_STOPGRAD=1：连 W 也冻结（λ 不可学，仅消融）。
    """
    if lam is None:
        lam = mix_lambda()
    if isinstance(lam, (int, float)) and float(lam) == 0.0:
        return S
    tgt = target if target is not None else target_name()
    if tgt == "whiten":
        S_hp = highpass(S, coeffs, target="whiten", prefer=prefer)
        if stop_grad_enabled() or stop_grad:
            S_hp = mx.stop_gradient(S_hp)
        lam_b = _broadcast_lam(lam, S)
        return S + lam_b * (S_hp - S)
    m, n = int(S.shape[-2]), int(S.shape[-1])
    if prefer == "left":
        left = True
    elif prefer == "right":
        left = False
    else:
        left = m <= n
    side = "left" if left else "right"
    W = ste_weight(S, lam, coeffs, prefer=side)
    if stop_grad_enabled() or stop_grad:
        W = mx.stop_gradient(W)
    return (W @ S) if left else (S @ W)


def maybe_apply(S: mx.array, n_done: int, lam=None, chunk_size: int = 16) -> mx.array:
    """若 n_done 落在门控点上则 apply，否则原样返回。"""
    if lam is None or not should_gate(n_done, period_tokens(chunk_size)):
        return S
    return apply(S, lam)
