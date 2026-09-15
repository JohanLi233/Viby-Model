"""Read-only, single-microbatch numerical audit of an explicit NCP -> thinking warm start."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.init import apply_trunc_normal_init
from trainer.utils import load_model_weights
from trainer.fast_norm import gradient_square_sum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument(
        "--thinking-arch",
        choices=("ced_pipeline_v1", "ced_iterative_v2", "ced_iterative_tied_v1"),
        default="ced_pipeline_v1",
    )
    p.add_argument("--thinking-steps", type=int, default=2)
    a = p.parse_args()
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError(dest)
    path = Path(a.checkpoint)
    before = path.stat()
    meta = json.loads(path.with_suffix(".json").read_text())
    config = VibyConfig.from_dict(
        dict(
            meta["config"],
            ncp_enabled=False,
            thinking_enabled=a.thinking,
            thinking_dim=128,
            thinking_arch=a.thinking_arch,
            thinking_steps=a.thinking_steps,
            thinking_scale=0.1,
        )
    )
    mx.set_default_device(mx.gpu)
    mx.random.seed(1340)
    mx.set_cache_limit(2 << 30)
    m = VibyForCausalLM(config, skip_init=True)
    if m.model.thinking is not None:
        apply_trunc_normal_init(m.model.thinking, config.dim)
    load_model_weights(
        m,
        str(path),
        allow_fresh_prefixes=("model.thinking.",),
        allow_drop_prefixes=("model.ncp.",),
        strict=True,
    )
    # New tensors must use the same working dtype as the loaded backbone.
    if m.model.thinking is not None:
        m.model.thinking.set_dtype(mx.bfloat16)
    if a.dtype == "float32":
        m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    with np.load(a.data) as d:
        x = mx.array(d["eval"][:1])

    def loss(net):
        return net(x[:, :-1], labels=x[:, 1:], use_mtp=False).loss

    direct = loss(m)
    mx.eval(direct)
    value, g = nn.value_and_grad(m, loss)(m)
    mx.eval(value, g)
    groups = {}
    for name, arr in tree_flatten(g):
        category = "thinking" if name.startswith("model.thinking.") else "backbone"
        groups.setdefault(category, []).append(gradient_square_sum(arr))
    norms = {k: float(mx.sqrt(mx.sum(mx.stack(v)))) for k, v in groups.items()}
    after = path.stat()
    result = {
        "checkpoint": str(path.resolve()),
        "source_execution": meta.get("execution"),
        "config": config.to_dict(),
        "input_shape": list(x[:, :-1].shape),
        "direct_loss": float(direct),
        "value_and_grad_loss": float(value),
        "primal_gap": abs(float(direct) - float(value)),
        "gradient_norms": norms,
        "peak_gib": mx.get_peak_memory() / 2**30,
        "source_unchanged": (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        "updates": 0,
        "dtype": a.dtype,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("VIBY_")},
        "scope": "Loaded full-width checkpoint, short B1/T128 eager GPU; no optimizer or efficacy claim",
    }
    result["passed"] = (
        result["source_unchanged"]
        and result["primal_gap"] <= 1e-5
        and np.isfinite(float(value))
        and all(np.isfinite(v) and v > 0 for v in norms.values())
    )
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Numerical warm-start gate failed")


if __name__ == "__main__":
    main()
