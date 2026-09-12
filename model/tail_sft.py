"""Sequence-offset filtering from TailSFT (arXiv:2608.25756, Algorithm 1).

Only the selection score is length-normalized per sequence. The objective is
token-averaged over survivors. No reference-model gradient or full B*T*V logits.
"""

import mlx.core as mx


def sequence_head_losses(
    hidden, weight, labels, loss_mask=None, z_weight=0.0, chunk=256
):
    """Return per-sequence CE sums, target counts, and z-loss sums in FP32."""
    b, t, d = hidden.shape
    flat_h, flat_y = hidden.reshape(-1, d), labels.reshape(-1)
    losses, zs = [], []
    for start in range(0, b * t, chunk):
        logits = flat_h[start : start + chunk] @ weight.T
        losses.append(
            mx.fast.cross_entropy(logits, flat_y[start : start + chunk]).astype(
                mx.float32
            )
        )
        if z_weight:
            zs.append(mx.square(mx.logsumexp(logits.astype(mx.float32), axis=-1)))
    mask = (
        mx.ones((b, t), mx.float32)
        if loss_mask is None
        else loss_mask.astype(mx.float32)
    )
    ce = mx.concatenate(losses).reshape(b, t)
    sums = mx.sum(mx.where(mask > 0, ce * mask, 0.0), axis=1)
    z_sums = mx.zeros((b,), mx.float32)
    if z_weight:
        z = mx.concatenate(zs).reshape(b, t)
        z_sums = mx.sum(mx.where(mask > 0, z * mask, 0.0), axis=1)
    return sums, mx.sum(mask, axis=1), z_sums


def tail_keep_mask(means, reference, counts, fraction):
    """Drop round(n_valid*f) lowest signed offsets; ties use batch order.

    Round-half-up is explicit; retain at least one valid sequence even for a
    tiny selection batch. Zero-target rows do not enter the ranking or budget.
    Pairwise ranking avoids depending on backend sort stability (b is small).
    """
    score = mx.stop_gradient(means - reference)
    valid = counts > 0
    index = mx.arange(means.shape[0])
    precedes = (score[None, :] < score[:, None]) | (
        (score[None, :] == score[:, None]) & (index[None, :] < index[:, None])
    )
    rank = mx.sum(precedes & valid[None, :], axis=1)
    n = mx.sum(valid)
    drop = mx.minimum(
        mx.floor(n * fraction + 0.5).astype(mx.int32), mx.maximum(n - 1, 0)
    )
    return mx.stop_gradient(valid & (rank >= drop))


def tail_objective(sums, counts, z_sums, reference, fraction):
    means = sums / mx.maximum(counts, 1.0)
    keep = tail_keep_mask(means, reference, counts, fraction)
    retained = mx.sum(mx.where(keep, counts, 0.0))
    denom = mx.maximum(retained, 1.0)
    ce = mx.sum(mx.where(keep, sums, 0.0)) / denom
    z = mx.sum(mx.where(keep, z_sums, 0.0)) / denom
    valid = counts > 0
    n = mx.maximum(mx.sum(valid), 1)
    stats = mx.stop_gradient(
        mx.stack(
            [
                mx.sum(valid).astype(mx.float32),
                mx.sum(keep).astype(mx.float32),
                retained,
                mx.sum(sums) / mx.maximum(mx.sum(counts), 1.0),
                mx.sum(mx.where(valid, means - reference, 0.0)) / n,
            ]
        )
    )
    return ce, z, keep, stats
