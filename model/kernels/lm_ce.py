# ---------------------------------------------------------------------------
# 融合 lm_head + cross-entropy（chunked / Liger 式）：
# 训练 loss 只需要标量，原路径却要先物化完整 (B,T,V) logits——bs16×4096×
# V6400 的 bf16 logits 就是 ~840MB，且 CE 的 VJP 还要把它留到反向再读一遍。
# 这里把 "hidden @ W.T + CE" 合成一个 custom_function：
#   正向：按行块（默认 4096 行）算 logits → 立即送入 ce.py 的逐行融合
#         kernel，块级 logits 消费完即释放，峰值只有一块；
#   反向：不保存 logits，按块从 hidden/weight 重算，再过 CE 反向 kernel
#         得到块梯度，grad_hidden 逐块写出、grad_weight f32 逐块累加。
# 输出与 ce.cross_entropy(h @ W.T, labels) 完全同语义（行 CE + lse 堆叠，
# 归约走 ce._reduce_ce_rows），z-loss 的 lse 余量照常回传。
# 多付出的代价：反向多一遍 lm_head matmul（重计算换内存）。
# ngram logit skip 不在此融合（logit_scale 可训练，VJP 需额外两项），
# 由调用方回退全量 logits 路径；CE kernel 不可用 / V 超限时同样回退。
# ---------------------------------------------------------------------------

from typing import Optional

import mlx.core as mx

from . import ce as _ce_mod
from .ce import _build_ce_kernels, _reduce_ce_rows, cross_entropy

# 单块行数：4096 行 × V=6400 × bf16 ≈ 52MB 瞬时块
DEFAULT_CHUNK = 4096

_LM_CE_DISABLED = False
_lm_ce_cache: dict = {}


def _chunk_logits(h2: mx.array, weight: mx.array) -> mx.array:
    return h2 @ weight.T


def _fwd_chunks(hidden: mx.array, weight: mx.array, labels: mx.array, chunk: int):
    """分块 logits → 逐行 CE kernel，返回 (2, M) [ce, lse]，f32。"""
    D = hidden.shape[-1]
    V = weight.shape[0]
    fwd, _ = _build_ce_kernels(V, weight.dtype)
    h_flat = hidden.reshape(-1, D)
    lab_flat = labels.reshape(-1)
    M = h_flat.shape[0]
    ces, lses = [], []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        logits = _chunk_logits(h_flat[s:e], weight)
        m = e - s
        ce_c, lse_c = fwd(
            inputs=[logits, lab_flat[s:e]],
            output_shapes=[(m,), (m,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(256, m, 1),
            threadgroup=(256, 1, 1),
        )
        ces.append(ce_c)
        lses.append(lse_c)
    return mx.stack([mx.concatenate(ces), mx.concatenate(lses)])


def _bwd_chunks(
    hidden: mx.array, weight: mx.array, labels: mx.array, cot: mx.array, chunk: int
):
    """按块重算 logits → CE 反向 kernel；grad_weight f32 逐块累加。

    cot 为 (2, M) f32：第 0 行 CE 余量、第 1 行 lse 余量（z-loss）。
    """
    D = hidden.shape[-1]
    V = weight.shape[0]
    _, bwd = _build_ce_kernels(V, weight.dtype)
    cot_ce, cot_lse = cot[0], cot[1]
    h_flat = hidden.reshape(-1, D)
    lab_flat = labels.reshape(-1)
    M = h_flat.shape[0]
    grad_w = mx.zeros((V, D), dtype=mx.float32)
    gh_parts = []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        h2 = h_flat[s:e]
        logits = _chunk_logits(h2, weight)
        g = bwd(
            inputs=[logits, lab_flat[s:e], cot_ce[s:e], cot_lse[s:e]],
            output_shapes=[logits.shape],
            output_dtypes=[logits.dtype],
            grid=(256, e - s, 1),
            threadgroup=(256, 1, 1),
        )[0]
        gh_parts.append(g @ weight)
        grad_w = grad_w + g.T.astype(mx.float32) @ h2.astype(mx.float32)
    grad_h = mx.concatenate(gh_parts).reshape(hidden.shape)
    return grad_h, grad_w.astype(weight.dtype)


def _make_lm_ce(chunk: int):
    """每个 chunk 尺寸一份 custom_function（chunk 是编译期常量）。"""
    fn = _lm_ce_cache.get(chunk)
    if fn is not None:
        return fn

    @mx.custom_function
    def fn(hidden, weight, labels):
        return _fwd_chunks(hidden, weight, labels, chunk)

    def vjp(primals, cotangents, outputs):
        hidden, weight, labels = primals
        # 单输出：cotangents 即 (2, M) 本体而非 list（与 ce._ce_rows 同约定）
        grad_h, grad_w = _bwd_chunks(hidden, weight, labels, cotangents, chunk)
        return grad_h, grad_w, None

    fn.vjp(vjp)
    _lm_ce_cache[chunk] = fn
    return fn


def lm_head_ce_rows(
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    chunk: int = DEFAULT_CHUNK,
) -> mx.array:
    """(B,T,D) hidden + (V,D) weight + (B,T) int32 labels → (2, B*T) f32
    [ce, lse]，语义同 ce._ce_rows(hidden @ weight.T, labels)。"""
    return _make_lm_ce(chunk)(hidden, weight, labels)


def lm_head_cross_entropy(
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    mask: Optional[mx.array] = None,
    return_z: bool = False,
    chunk: int = DEFAULT_CHUNK,
):
    """从 hidden 直接算 CE loss，不物化全量 logits。

    语义与 cross_entropy(hidden @ weight.T, labels, mask, return_z) 一致。
    kernel 不可用（编译失败 / V*4 超过 28KB threadgroup 上限）时回退
    全量 logits 路径。
    """
    global _LM_CE_DISABLED
    V = weight.shape[0]
    if not _LM_CE_DISABLED and not _ce_mod._CE_KERNEL_DISABLED and V * 4 <= 28 * 1024:
        try:
            flat_labels = labels.reshape(-1).astype(mx.int32)
            rows = lm_head_ce_rows(hidden, weight, flat_labels, chunk=chunk)
            return _reduce_ce_rows(rows[0], rows[1], flat_labels, mask, return_z)
        except Exception:
            _LM_CE_DISABLED = True
    return cross_entropy(hidden @ weight.T, labels, mask=mask, return_z=return_z)
