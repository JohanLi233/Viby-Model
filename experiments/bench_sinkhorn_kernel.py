"""Same-process compiled A/B. --model includes resident optimizer state + doc mask."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from model.kernels import sinkhorn_fused as sk


def measure(fn, iters, warmup=2):
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - start)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    mx.set_cache_limit(8 * 1024**3)
    mx.random.seed(1234)
    sk.prewarm_sinkhorn()
    runs = {}
    if args.model:
        from model.config import VibyConfig
        from model.model import VibyForCausalLM
        from trainer.base_trainer import BaseTrainer
        from trainer.config import get_pretrain_parser, setup_training_args
        from trainer.utils import build_model_kwargs, resolve_compute_scaled_hparams

        cli = [
            "--out_dir",
            "research_runs/_bench",
            "--no_save",
            "--batch_size",
            "4",
            "--max_seq_len",
            "1024",
            "--accumulation_steps",
            "2",
            "--cache_limit_gb",
            "8",
        ]
        ta = setup_training_args(get_pretrain_parser().parse_args(cli), "pretrain")
        cfg = VibyConfig(**build_model_kwargs(ta))
        ta = resolve_compute_scaled_hparams(ta, 467617)
        model = VibyForCausalLM(cfg)
        trainer = BaseTrainer(ta, model, None, cfg, "pretrain")
        x = mx.random.randint(0, cfg.vocab_size, (4, 1024))
        y = mx.random.randint(0, cfg.vocab_size, x.shape)
        mask = mx.ones(x.shape)
        attn = mx.ones(x.shape, mx.int32)
        seg = mx.cumsum(
            (mx.random.uniform(shape=x.shape) < 1 / 128).astype(mx.int32), axis=1
        )
        mx.eval(x, y, mask, attn, seg)

        def step():
            return trainer._compute_loss_and_grad(x, y, mask, attn, seg)

        # Allocate optimizer state once; both variants then use identical weights.
        outputs, grads = step()
        mx.eval(outputs, grads)
        trainer.optimizer.update(model, grads)
        mx.eval(model.parameters(), trainer.optimizer.state)
        del outputs, grads
        for name, enabled in [("ref", False), ("fused", True)]:
            sk._ENABLED = enabled
            compiled = trainer._build_loss_and_grad()

            def run(compiled=compiled):
                trainer._loss_and_grad = compiled
                return step()

            mx.eval(run())  # trace now, while the selected flag is in force
            runs[name] = run
    else:
        x = mx.random.normal((4, 1024, 4, 4)) * 3
        g = mx.random.normal(x.shape)
        mx.eval(x, g)
        for name, fn in [("ref", sk.sinkhorn_ref), ("fused", sk.sinkhorn_fused)]:

            def loss(a, fn=fn):
                return mx.sum(fn(a, 20, 1e-6) * g)

            compiled = mx.compile(mx.value_and_grad(loss))
            runs[name] = lambda compiled=compiled: compiled(x)

    for name in ["ref", "fused", "fused", "ref"]:
        mx.reset_peak_memory()
        times = measure(runs[name], args.iters)
        print(
            f"{name}: min={min(times):.6f}s rounds={[round(t, 6) for t in times]} "
            f"peak={mx.get_peak_memory() / 1e9:.2f}GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
