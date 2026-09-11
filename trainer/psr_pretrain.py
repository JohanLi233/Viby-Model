"""Content-independent local block plans for protected next-token correction."""

import mlx.core as mx


def anchor_plan(input_ids, attention_mask, segment_ids, horizon, *, count=None, key=0):
    """Uniform sampled blocks via independent stateless hash, or complete plan.

    count=None returns all T candidate slots (invalid slots=-1), suited to a
    small fixed-shape evaluation. compact_anchor_plan trims invalid slots before
    evaluation to avoid wasting workspace compute on short/packed documents.
    """
    b, t = input_ids.shape
    pos = mx.broadcast_to(mx.arange(t)[None], (b, t))
    valid = (
        mx.ones((b, t), mx.bool_)
        if attention_mask is None
        else attention_mask.astype(mx.bool_)
    )
    same = valid[:, 1:] & valid[:, :-1]
    if segment_ids is not None:
        same = same & (segment_ids[:, 1:] == segment_ids[:, :-1])
    starts = mx.concatenate([mx.ones((b, 1), mx.bool_), ~same], 1)
    left = mx.cummax(mx.where(starts, pos, 0), axis=1)
    candidates = valid & ((pos - left) % horizon == 0)
    total = mx.sum(candidates, axis=1)
    if count is None:
        anchors = mx.sort(mx.where(candidates, pos, t), axis=1)
        return mx.where(anchors < t, anchors, -1).astype(mx.int32), mx.ones((b,))
    k = min(int(count), t)
    # Explicit independent PRNG key is a runtime input; never consumes the
    # baseline RNG or data-loader shuffle stream.
    random_key = mx.stack(
        [mx.array(key, dtype=mx.uint32), mx.array(0x505352, mx.uint32)]
    )
    scores = mx.where(candidates, mx.random.uniform(shape=(b, t), key=random_key), -1)
    chosen = mx.argsort(-scores, axis=1)[:, :k].astype(mx.int32)
    anchors = mx.where(mx.take_along_axis(candidates, chosen, axis=1), chosen, -1)
    anchors = mx.sort(anchors, axis=1).astype(mx.int32)
    weights = total.astype(mx.float32) / mx.maximum(mx.minimum(total, k), 1)
    return anchors, weights


def compact_anchor_plan(input_ids, attention_mask, segment_ids, horizon):
    anchors, _ = anchor_plan(input_ids, attention_mask, segment_ids, horizon)
    # Evaluation-only, outside compilation/timing: number depends on layout.
    count = max(1, int(mx.max(mx.sum(anchors >= 0, axis=1))))
    return anchors[:, :count]


def text_psr_inputs(
    config, input_ids, labels, loss_mask, attention_mask, segment_ids=None, key=0
):
    anchors, weights = anchor_plan(
        input_ids,
        attention_mask,
        segment_ids,
        config.psr_horizon,
        count=config.psr_train_anchors,
        key=key,
    )
    return dict(
        psr_mode="recurrent",
        psr_anchors=anchors,
        psr_sample_weights=weights,
        thinking_options={"rounds": config.psr_rounds},
        return_metrics=True,
    )
