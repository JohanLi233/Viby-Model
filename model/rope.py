"""DeepSeek-V4.1 的 RoPE：交错对（adjacent-pair）复数旋转 + YaRN。

V4.1 只在每个 head 的尾部 rope_head_dim 通道上做旋转；压缩分支的位置是
"组首 token"的位置（j * compress_ratio），且压缩分支用 compress_rope_theta
配 YaRN，纯滑窗层用 rope_theta 且不做 YaRN（对应参考实现里
`original_seq_len = 0 if compress_ratio == 0`）。

注意力输出端会把同一个旋转按 -i 再转回去（inverse=True），这样共享的
KV cache 保持"只旋转一次"的单一形式。
"""

import math

import mlx.core as mx


def _corrected_dim(
    dim: float, rotations: float, base: float, original_seq_len: int
) -> float:
    return (
        dim
        * math.log(original_seq_len / (rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
):
    """返回 (cos, sin)，形状 [seqlen, dim//2]，float32。dim 是 rope_head_dim。"""
    if dim % 2:
        raise ValueError("rope dim 必须是偶数")
    inv = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(_corrected_dim(dim, beta_fast, base, original_seq_len)), 0)
        high = min(
            math.ceil(_corrected_dim(dim, beta_slow, base, original_seq_len)), dim - 1
        )
        ramp = mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 1e-3),
            0.0,
            1.0,
        )
        smooth = 1.0 - ramp
        inv = inv / factor * (1.0 - smooth) + inv * smooth
    freqs = mx.outer(mx.arange(seqlen, dtype=mx.float32), inv)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(
    x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """x: [..., S, dim]（dim 为偶数）；cos/sin: [S, dim//2]。

    交错对 (2i, 2i+1) 视作复数做旋转；inverse=True 时用共轭（sin 取反）。
    """
    d = x.shape[-1]
    xr = x.reshape(*x.shape[:-1], d // 2, 2)
    real, imag = xr[..., 0], xr[..., 1]
    if inverse:
        sin = -sin
    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    out = mx.stack([out_real, out_imag], axis=-1)
    if out.size != x.size:
        raise ValueError(
            f"apply_rope: x{tuple(x.shape)} cos{tuple(cos.shape)} real{tuple(real.shape)} out{tuple(out.shape)}"
        )
    return out.reshape(*x.shape[:-1], d)


def rope_partial(
    x: mx.array, cos: mx.array, sin: mx.array, rope_dim: int, inverse: bool = False
):
    """只旋转最后一维的 rope_dim 个通道，其余原样拼回。"""
    from .kernels.rope_decode import enabled_for, rope_partial_decode

    if enabled_for(x, cos, sin, rope_dim):
        return rope_partial_decode(x, cos, sin, rope_dim, inverse)
    if rope_dim == x.shape[-1]:
        return apply_rope(x, cos, sin, inverse)
    head = x[..., : x.shape[-1] - rope_dim]
    tail = x[..., x.shape[-1] - rope_dim :]
    return mx.concatenate([head, apply_rope(tail, cos, sin, inverse)], axis=-1)
