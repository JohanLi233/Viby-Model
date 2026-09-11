"""Causal PSR self-supervision for ordinary (optionally packed) text batches."""

import mlx.core as mx


def text_psr_inputs(config, input_ids, labels, loss_mask, attention_mask, segment_ids=None):
    """One workspace per row, anchored halfway through its longest valid document.

    The boundary uses only document/padding layout, never future token values.
    Existing LM supervision is unchanged. Behavior tests ask for the next P
    tokens from the terminal state, without putting those tokens into the state.
    """
    b, t = input_ids.shape
    pos = mx.broadcast_to(mx.arange(t)[None, :], (b, t))
    valid = mx.ones((b, t), dtype=mx.bool_) if attention_mask is None else attention_mask.astype(mx.bool_)
    same = valid[:, 1:] & valid[:, :-1]
    if segment_ids is not None:
        same = same & (segment_ids[:, 1:] == segment_ids[:, :-1])
    starts = mx.concatenate([mx.ones((b, 1), mx.bool_), ~same], axis=1)
    ends = mx.concatenate([~same, mx.ones((b, 1), mx.bool_)], axis=1)
    left = mx.cummax(mx.where(starts, pos, 0), axis=1)
    right = mx.cummin(mx.where(ends, pos, t - 1), axis=1, reverse=True)
    lengths = mx.where(valid, right - left + 1, 0)
    chosen = mx.argmax(lengths, axis=1).astype(mx.int32)
    row = mx.arange(b)
    anchor = left[row, chosen] + (lengths[row, chosen] - 1) // 2
    has_tokens = mx.any(valid, axis=1)
    prefix = mx.where(has_tokens, anchor + 1, 0).astype(mx.int32)
    anchor = mx.maximum(anchor, 0)
    probes = min(config.psr_num_tests, 4)
    offsets = mx.arange(probes)[None, :]
    indices = anchor[:, None] + offsets
    safe = mx.minimum(indices, t - 1)
    # labels[j] is token j+1. Require every preceding prediction to be valid,
    # so a multi-token test cannot jump across a packed document boundary.
    allowed = (indices < t) & has_tokens[:, None]
    if loss_mask is not None:
        allowed = allowed & (loss_mask[row[:, None], safe] > 0)
    if segment_ids is not None:
        allowed = allowed & (segment_ids[row[:, None], safe] == segment_ids[row, anchor][:, None])
    allowed = mx.cumprod(allowed.astype(mx.int32), axis=1).astype(mx.bool_)
    terminal = mx.where(allowed, labels[row[:, None], safe], -1).astype(mx.int32)
    tests = mx.broadcast_to(offsets, (b, probes)).astype(mx.int32)
    return dict(thinking_prefix_lengths=prefix,
                thinking_options={"rounds": config.psr_rounds, "distill_reads": True},
                thinking_targets={"terminal_tests": tests, "terminal_results": terminal}, use_mtp=False)
