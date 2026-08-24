# ---------------------------------------------------------------------------
# 融合 cross-entropy kernel：逐行 logsumexp + 选中项 + softmax-onehot 反向，
# 各一个 Metal kernel。原版链要物化 f32 (M,V) logits、logsumexp、广播减法、
# take_along 等多趟 105MB 级读写（M=B*T, V=6400）；融合后 fwd 只读一遍
# bf16 logits，bwd 读一遍写一遍梯度。行载入 threadgroup memory（V*4 字节
# ≤ 32KB，V=6400 时 25.6KB），全程 f32 累加。手写 VJP 经 mx.custom_function
# 接入 autodiff。labels=-100 的行 loss/梯度均为 0（与原实现一致）。
# 编译失败或 V 超限自动回退原版链。
# ---------------------------------------------------------------------------

from typing import Optional

import mlx.core as mx

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}

_ce_kernel_cache: dict = {}
_CE_KERNEL_DISABLED = False


def _build_ce_kernels(V: int, dtype):
    key = (V, dtype)
    if key in _ce_kernel_cache:
        return _ce_kernel_cache[key]
    mt = _METAL_TYPE[dtype]
    # 行归约：256 线程先各自扫描，再 threadgroup 树形归约 max / sum
    reduce_src = """
        uint lane = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        threadgroup float row[V_];
        threadgroup float part[256];
        size_t base = (size_t)m * V_;
        for (uint v = lane; v < V_; v += 256) row[v] = float(logits[base + v]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lmax = -INFINITY;
        for (uint v = lane; v < V_; v += 256) lmax = metal::max(lmax, row[v]);
        part[lane] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (lane < s) part[lane] = metal::max(part[lane], part[lane + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mmax = part[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lsum = 0.0f;
        for (uint v = lane; v < V_; v += 256) lsum += metal::exp(row[v] - mmax);
        part[lane] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128; s > 0; s >>= 1) {
            if (lane < s) part[lane] = part[lane] + part[lane + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv_l = 1.0f / part[0];
        int lbl = labels[m];
        bool valid = (lbl >= 0) && (lbl < int(V_));
    """
    fwd_src = (
        reduce_src
        + """
        if (lane == 0) {
            // lse 是 CE 的既有中间量，顺带输出供 logit z-loss 复用
            // （避免二次物化 (M,V) logits 算 logsumexp）
            float l = mmax + metal::log(part[0]);
            lse[m] = l;
            ce[m] = valid ? (l - row[lbl]) : 0.0f;
        }
    """
    )
    bwd_src = (
        reduce_src
        + f"""
        float c = cot_ce[m];
        float cz = cot_lse[m];
        for (uint v = lane; v < V_; v += 256) {{
            float p = metal::exp(row[v] - mmax) * inv_l;
            // d(CE)/dz = p - onehot；d(lse)/dz = p（z-loss 项的梯度）
            float grad = (c + cz) * p;
            if (v == uint(lbl)) grad -= c;
            g[base + v] = {mt}(valid ? grad : 0.0f);
        }}
    """
    )
    fwd_src = fwd_src.replace("V_", str(V))
    bwd_src = bwd_src.replace("V_", str(V))
    fwd = mx.fast.metal_kernel(
        name=f"ce_fwd_{V}_{mt}",
        input_names=["logits", "labels"],
        output_names=["ce", "lse"],
        source=fwd_src,
    )
    bwd = mx.fast.metal_kernel(
        name=f"ce_bwd_{V}_{mt}",
        input_names=["logits", "labels", "cot_ce", "cot_lse"],
        output_names=["g"],
        source=bwd_src,
    )
    _ce_kernel_cache[key] = (fwd, bwd)
    return fwd, bwd


@mx.custom_function
def _ce_rows(logits: mx.array, labels: mx.array) -> mx.array:
    """逐行融合 CE：(M,V) logits + (M,) int32 labels -> (2,M) f32，
    第 0 行为行 CE loss，第 1 行为行 logsumexp（供 z-loss 复用）。"""
    fwd, _ = _build_ce_kernels(logits.shape[-1], logits.dtype)
    M = logits.shape[0]
    ce, lse = fwd(
        inputs=[logits, labels],
        output_shapes=[(M,), (M,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(256, M, 1),
        threadgroup=(256, 1, 1),
    )
    return mx.stack([ce, lse])


def _ce_rows_vjp(primals, cotangents, outputs):
    logits, labels = primals
    _, bwd = _build_ce_kernels(logits.shape[-1], logits.dtype)
    cot_ce, cot_lse = cotangents[0], cotangents[1]
    g = bwd(
        inputs=[logits, labels, cot_ce, cot_lse],
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(256, logits.shape[0], 1),
        threadgroup=(256, 1, 1),
    )[0]
    return g, None


_ce_rows.vjp(_ce_rows_vjp)


def cross_entropy(
    logits: mx.array,
    labels: mx.array,
    mask: Optional[mx.array] = None,
    return_z: bool = False,
):
    """逐行 CE。return_z=True 时额外返回 z_mean：仅 CE 有效位置
    （loss mask 内且 label != -100）的 logsumexp 均值，归一化与 CE 相同，
    供 logit z-loss（loss += z_loss_weight * mean(lse²)，Marin 口径）使用。"""
    global _CE_KERNEL_DISABLED
    V = logits.shape[-1]
    z_mean = None
    # 行 buffer V*4 字节需放进 32KB threadgroup memory
    if not _CE_KERNEL_DISABLED and V * 4 <= 28 * 1024:
        try:
            flat_labels = labels.reshape(-1).astype(mx.int32)
            rows = _ce_rows(logits.reshape(-1, V), flat_labels)
            ce, lse = rows[0], rows[1]
            if return_z:
                valid = (flat_labels != -100).astype(mx.float32)
            if mask is None:
                # 与原链的 mean 语义差异：原链 mean 含 -100 行（其 ce=0），
                # 这里逐行 ce 同样为 0，mean 结果一致
                loss = mx.mean(ce)
                if return_z:
                    z_mean = mx.mean((lse * lse) * valid)
            else:
                mask_flat = mask.reshape(-1).astype(ce.dtype)
                denom = mx.maximum(mx.sum(mask_flat), mx.array(1.0))
                loss = mx.sum(ce * mask_flat) / denom
                if return_z:
                    z_mean = mx.sum((lse * lse) * valid * mask_flat) / denom
            return (loss, z_mean) if return_z else loss
        except Exception:
            _CE_KERNEL_DISABLED = True
    logits_f = logits.astype(mx.float32)
    log_z = mx.logsumexp(logits_f, axis=-1)
    labels_safe = mx.where(labels == -100, 0, labels)
    logp = logits_f - log_z[..., None]
    picked = mx.take_along_axis(logp, labels_safe[..., None], axis=-1).squeeze(-1)
    ce = mx.where(labels == -100, 0.0, -picked)

    ce = ce.reshape(-1)
    lse = log_z.reshape(-1)
    if return_z:
        valid = (labels.reshape(-1) != -100).astype(mx.float32)
    if mask is None:
        loss = mx.mean(ce)
        if return_z:
            z_mean = mx.mean((lse * lse) * valid)
    else:
        mask_flat = mask.reshape(-1).astype(ce.dtype)
        denom = mx.maximum(mx.sum(mask_flat), mx.array(1.0))
        loss = mx.sum(ce * mask_flat) / denom
        if return_z:
            z_mean = mx.sum((lse * lse) * valid * mask_flat) / denom
    return (loss, z_mean) if return_z else loss
