"""Count actual sparse occurrences outside timing; retain the declared 6N estimate."""
import mlx.core as mx

from model.kernels import sparse_attention as sa
from trainer.flops import training_flops_per_token


def audit_training_flops(model, batch):
    cfg = model.config
    if cfg.n_mtp_layers:
        raise ValueError("this occurrence audit currently supports backbone-only training")
    x, y, loss_mask, pad, segment = batch
    b, t = x.shape
    w = min(t, cfg.window_size)
    positions = mx.arange(t)[:, None] - mx.arange(w)[None, :]
    safe = mx.maximum(positions, 0)
    keep = mx.broadcast_to(positions[None] >= 0, (b, t, w))
    if pad is not None:
        keep = keep & pad[:, safe].astype(mx.bool_)
    if segment is not None:
        keep = keep & (segment[:, safe] == segment[:, :, None])
    window_mean = float(mx.mean(mx.sum(keep, axis=-1).astype(mx.float32)))
    compressed = []
    original = sa.indexed_attention

    def record(*args, **kwargs):
        selection = kwargs.get("selection")
        if selection is None:
            visible = mx.broadcast_to(args[3], (b, t, args[2].shape[1]))
            lengths = mx.sum(visible, axis=-1)
        else:
            lengths = selection[1]
        compressed.append(mx.mean(lengths.astype(mx.float32)))
        return original(*args, **kwargs)

    sa.indexed_attention = record
    try:
        out = model(x, labels=y, loss_mask=loss_mask, attention_mask=pad, segment_ids=segment)
        mx.eval(out.loss, compressed)
    finally:
        sa.indexed_attention = original
    expected = sum(r > 0 for r in cfg.compress_ratios[:cfg.n_layers])
    if len(compressed) != expected:
        raise ValueError("occurrence audit requires the indexed sparse training path")
    counts = iter(float(v) for v in compressed)
    lengths = [window_mean + (next(counts) if r else 0.0)
               for r in cfg.compress_ratios[:cfg.n_layers]]
    return dict(
        method="6N GEMM estimate + measured valid attention occurrences; excludes optimizer/recompute",
        attention_mean_occurrences=lengths,
        flops_per_token=training_flops_per_token(model, t, lengths),
    )
