"""Parameter-free, document-local residual lifting for recurrent CED.

The padded anchor capacity is floor(T/k), not an asserted executed FLOP count.
All plan operations stay on device and have static shapes for MLX compilation.
"""

from dataclasses import dataclass

import mlx.core as mx

from .cache import SharedAttnState


@dataclass
class AnchorPlan:
    indices: mx.array
    valid: mx.array
    positions: mx.array
    segment_ids: mx.array
    owner: mx.array
    covered: mx.array


def _gather(x, indices):
    idx = indices.reshape(*indices.shape, *((1,) * (x.ndim - 2)))
    return mx.take_along_axis(x, mx.stop_gradient(idx), axis=1)


def build_anchor_plan(positions, segment_ids, pad_mask, stride):
    """Select the end of each completed group in each contiguous valid document.

    A padding gap resets grouping even if document IDs repeat afterwards.
    positions are original token RoPE coordinates; grouping uses local offsets.
    """
    if stride < 1:
        raise ValueError("anchor stride must be positive")
    b, t = positions.shape
    if t == 0:
        raise ValueError("recurrent CED requires a nonempty token sequence")
    if segment_ids is None and pad_mask is None:
        indices = mx.broadcast_to(
            mx.arange(stride - 1, t, stride)[None], (b, t // stride)
        )
        covered = mx.broadcast_to((mx.arange(t) >= stride - 1)[None], (b, t))
        # Explicit maximum avoids MLX's truncation-toward-zero integer division
        # assigning negative pre-anchor indices a surprising group.
        owner = mx.broadcast_to(
            mx.maximum((mx.arange(t) + 1) // stride - 1, 0)[None], (b, t)
        )
        return AnchorPlan(
            indices,
            mx.ones(indices.shape, mx.bool_),
            _gather(positions, indices),
            mx.zeros(indices.shape, mx.int32),
            owner,
            covered,
        )
    docs = mx.zeros((b, t), mx.int32) if segment_ids is None else segment_ids
    valid = mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_)
    index = mx.broadcast_to(mx.arange(t)[None], (b, t))
    reset = mx.concatenate(
        [mx.ones((b, 1), mx.bool_), (docs[:, 1:] != docs[:, :-1]) | ~valid[:, :-1]],
        axis=1,
    )
    start = mx.cummax(mx.where(reset, index, 0), axis=1)
    anchor = valid & ((index - start + 1) % stride == 0)
    capacity = t // stride
    indices = mx.sort(mx.where(anchor, index, t), axis=1)[:, :capacity]
    good = indices < t
    indices = mx.minimum(indices, t - 1)
    latest = mx.cummax(mx.where(anchor, index, -1), axis=1)
    covered = valid & (latest >= start) & (latest >= 0)
    owner = mx.maximum(mx.cumsum(anchor.astype(mx.int32), axis=1) - 1, 0)
    return AnchorPlan(
        mx.stop_gradient(indices),
        good,
        _gather(positions, indices),
        _gather(docs, indices),
        mx.stop_gradient(owner),
        covered,
    )


def sample_anchors(tensor, plan):
    return _gather(tensor, plan.indices)


def hold_anchors(values, plan, fallback=None):
    shape = (*plan.owner.shape, *values.shape[2:])
    if fallback is None:
        fallback = mx.zeros(shape, values.dtype)
    if values.shape[1] == 0:
        return fallback
    held = _gather(values, mx.minimum(plan.owner, values.shape[1] - 1))
    mask = plan.covered.reshape(*plan.covered.shape, *((1,) * (values.ndim - 2)))
    return mx.where(mask, held, fallback)


def residual_lift(hidden, initial, final, plan):
    return hidden + hold_anchors(final - initial, plan)


def materialize_boundary_routes(
    shared,
    batch,
    queries,
    memory_length,
    topk,
    block_size,
    decode=False,
    window_size=0,
    compact_candidates=False,
):
    """Convert existing boundary routing metadata, without rerunning its indexer."""
    if decode and shared.topk_idx is not None:
        idx = shared.topk_idx
        idx = mx.where(idx >= 0, idx - window_size, -1)
    elif shared.sparse_selection is not None:
        _, (ids, lengths) = shared.sparse_selection
        ids = ids.reshape(batch, queries, -1)
        lengths = lengths.reshape(batch, queries)
        k = min(topk, ids.shape[-1])
        idx = mx.where(
            mx.arange(k)[None, None, :] < lengths[..., None], ids[..., :k], -1
        )
    elif shared.keep_mask is not None:
        keep = mx.broadcast_to(shared.keep_mask, (batch, queries, memory_length))
        ids = mx.arange(memory_length)[None, None, :]
        idx = mx.sort(mx.where(keep, ids, memory_length), axis=-1)[
            ..., : min(topk, memory_length)
        ]
        idx = mx.where(idx < memory_length, idx, -1)
    else:
        raise ValueError("recurrent CED requires a boundary sparse selection")
    # A fixed width lets a held selection survive growth of the evidence pool.
    if idx.shape[-1] < topk:
        idx = mx.concatenate(
            [idx, mx.full((batch, queries, topk - idx.shape[-1]), -1, mx.int32)],
            axis=-1,
        )
    shared.topk_idx = mx.stop_gradient(idx)
    if (
        not compact_candidates
        and shared.candidates is None
        and shared.candidate_blocks is not None
    ):
        from .kernels.indexer_select import candidate_mask

        shared.candidates = candidate_mask(
            shared.candidate_blocks, batch, queries, memory_length, block_size
        )
    shared.keep_mask = None
    shared.sparse_selection = None
    shared.reach = None


def anchor_shared(shared, plan, block_size=None):
    """Copy control state; keep projected evidence and index K at token resolution."""
    result = SharedAttnState()
    result.compress_kv = shared.compress_kv
    result.index_k = shared.index_k
    result.topk_idx = sample_anchors(shared.topk_idx, plan)
    if shared.candidates is not None:
        result.candidates = sample_anchors(shared.candidates, plan)
    elif shared.candidate_blocks is not None and block_size is not None:
        # Sample block metadata before expanding a candidate mask: the sparse
        # boundary rows are full length; only A rows are read by recurrent depth.
        ids, lengths = shared.candidate_blocks
        b, t = plan.owner.shape
        ids = sample_anchors(ids.reshape(b, t, -1), plan)
        lengths = sample_anchors(lengths.reshape(b, t), plan)
        result.candidate_blocks = (
            ids.reshape(b * ids.shape[1], ids.shape[-1]),
            lengths.reshape(-1),
        )
        if ids.shape[1]:
            from .kernels.indexer_select import candidate_mask

            result.candidates = candidate_mask(
                result.candidate_blocks,
                b,
                ids.shape[1],
                shared.compress_kv.shape[1],
                block_size,
            )
        else:
            result.candidates = mx.zeros((b, 0, shared.compress_kv.shape[1]), mx.bool_)
    return result


def recurrent_active(config, enabled=True):
    return (
        enabled
        and getattr(config, "ced_recurrent_enabled", False)
        and (config.ced_recurrent_stride, config.ced_recurrent_rounds) != (1, 1)
    )


def execution_signature(config, enabled=True):
    if not recurrent_active(config, enabled):
        return ("baseline",)
    return (
        "residual_lift_v1",
        config.ced_recurrent_stride,
        config.ced_recurrent_rounds,
        config.n_encoder_layers + 1,
        config.n_layers - 1,
    )


def run_middle(
    model,
    hidden,
    pre_mix,
    shared,
    positions,
    segment_ids,
    pad_mask,
    cache=None,
    start_pos=0,
):
    """Execute shared Blocks on anchors; restore full-resolution control state.

    The aux objective averages rounds for each logical layer, retaining its
    original layer coefficient. Loads and QB observations sum all physical calls.
    """
    cfg = model.config
    b, t = hidden.shape[:2]
    first, stop = cfg.n_encoder_layers + 1, cfg.n_layers - 1
    if cache is not None:
        source = cache[cfg.n_encoder_layers]
        shared.compress_kv = source.compress_kv[:, : start_pos + t]
        shared.index_k = source.index_k[:, : start_pos + t]
    n = shared.compress_kv.shape[1]
    memory_positions = mx.broadcast_to(mx.arange(n)[None], (b, n))
    memory_docs = segment_ids if cache is None else None
    memory_pad = pad_mask if cache is None else None
    from .attention import _RECURRENT_OPTIMIZED

    compact_candidates = _RECURRENT_OPTIMIZED and cache is None
    materialize_boundary_routes(
        shared,
        b,
        t,
        n,
        cfg.index_topk,
        cfg.candidate_block_size,
        decode=start_pos > 0,
        window_size=cfg.window_size,
        compact_candidates=compact_candidates,
    )
    boundary_routes = shared.topk_idx
    # Cached generation uses a common, unpadded single-document clock. The
    # training/prefill planner below supports arbitrary document boundaries.
    incremental = cache is not None and start_pos > 0
    if incremental:
        is_anchor = (start_pos + 1) % cfg.ced_recurrent_stride == 0
        valid = mx.full((b, int(is_anchor)), True, mx.bool_)
        plan = AnchorPlan(
            mx.zeros((b, int(is_anchor)), mx.int32),
            valid,
            positions[:, : int(is_anchor)],
            mx.zeros_like(valid, mx.int32),
            mx.zeros((b, t), mx.int32),
            mx.full((b, t), is_anchor, mx.bool_),
        )
    else:
        plan = build_anchor_plan(
            positions, segment_ids, pad_mask, cfg.ced_recurrent_stride
        )
    initial = sample_anchors(hidden, plan)
    z, mix = initial, sample_anchors(pre_mix, plan)
    latent_shared = anchor_shared(
        shared, plan, cfg.candidate_block_size if compact_candidates else None
    )
    calls = 0
    loads, auxes, margins = {}, {}, {}
    if z.shape[1]:
        for round_id in range(cfg.ced_recurrent_rounds):
            for layer_id in range(first, stop):
                layer = model.layers[layer_id]
                stage_cache = (
                    None
                    if cache is None
                    else cache.ced_stages.setdefault((round_id, layer_id), {})
                )
                z, mix = layer.recurrent(
                    z,
                    mix,
                    latent_shared,
                    query_positions=plan.positions,
                    memory_positions=memory_positions,
                    query_segment_ids=plan.segment_ids,
                    memory_segment_ids=memory_docs,
                    query_pad_mask=None
                    if segment_ids is None and pad_mask is None
                    else plan.valid,
                    memory_pad_mask=memory_pad,
                    cache=stage_cache,
                )
                calls += 1
                if model.training:
                    loads.setdefault(layer_id, []).append(layer.ffn.router._last_load)
                    auxes.setdefault(layer_id, []).append(layer.ffn._last_aux)
                    margins.setdefault(layer_id, []).append(
                        layer.ffn.router._last_qb_margins
                    )
    for layer_id in range(first, stop):
        if not model.training:
            continue
        ffn = model.layers[layer_id].ffn
        ffn.router._last_load = (
            mx.stack(loads[layer_id]).sum(axis=0)
            if layer_id in loads
            else mx.zeros((cfg.n_routed_experts,), mx.float32)
        )
        ffn._last_aux = (
            mx.stack(auxes[layer_id]).mean() if layer_id in auxes else mx.array(0.0)
        )
        ffn.router._last_qb_margins = (
            mx.concatenate(margins[layer_id], axis=0)
            if layer_id in margins
            else mx.zeros((0, cfg.n_routed_experts), mx.float32)
        )
    if incremental:
        if z.shape[1]:
            cache.ced_delta = z - initial
            cache.ced_pre_mix = mix
            cache.ced_topk_idx = latent_shared.topk_idx
        if cache.ced_delta is not None:
            hidden = hidden + cache.ced_delta
            pre_mix = cache.ced_pre_mix
            shared.topk_idx = cache.ced_topk_idx
    else:
        hidden = residual_lift(hidden, initial, z, plan)
        pre_mix = hold_anchors(mix, plan, fallback=pre_mix)
        shared.topk_idx = hold_anchors(
            latent_shared.topk_idx, plan, fallback=boundary_routes
        )
        if cache is not None and z.shape[1]:
            cache.ced_delta = (z - initial)[:, -1:]
            cache.ced_pre_mix = mix[:, -1:]
            cache.ced_topk_idx = latent_shared.topk_idx[:, -1:]
    shared.keep_mask = None
    shared.sparse_selection = None
    valid_tokens = mx.array(b * t) if pad_mask is None else mx.sum(pad_mask)
    anchors = mx.sum(plan.valid)
    metrics = mx.stack(
        [
            valid_tokens,
            anchors,
            mx.array(b * z.shape[1]),
            mx.array(calls),
            anchors * cfg.ced_recurrent_rounds * (stop - first),
        ]
    ).astype(mx.float32)
    result = dict(
        plan=plan,
        metrics=metrics,
        initial=initial,
        final=z,
        final_pre_mix=mix,
        final_routes=latent_shared.topk_idx,
        candidate_pool=latent_shared.candidates,
    )
    return hidden, pre_mix, shared, memory_positions, memory_docs, memory_pad, result
