"""解码（自回归推理）基准：prefill + 逐 token decode 吞吐/时延。

用法: .venv/bin/python experiments/bench_decode.py [prefill] [decode] [--spec]
对比项：标准 greedy decode vs MTP 投机解码（--spec）。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.config import VibyConfig
from model.model import VibyForCausalLM

PREFILL = int(sys.argv[1]) if len(sys.argv) > 1 else 512
DECODE = int(sys.argv[2]) if len(sys.argv) > 2 else 128

cfg = VibyConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    vocab_size=6400,
    max_position_embeddings=max(PREFILL + DECODE + 8, 1024),
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=2,
    moe_intermediate_size=320,
    mtp_depth=1,
)
mx.random.seed(0)
model = VibyForCausalLM(cfg)
model.set_dtype(mx.bfloat16)
model.eval()
mx.eval(model.parameters())
ids = mx.random.randint(3, cfg.vocab_size, (1, PREFILL))


def run(spec: bool, label: str):
    # 预热（含 lazy kernel 编译）
    model.generate(
        ids[:, :16],
        max_new_tokens=8,
        do_sample=False,
        eos_token_id=None,
        use_mtp_speculative=spec,
    )
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=DECODE,
        do_sample=False,
        eos_token_id=None,
        use_mtp_speculative=spec,
    )
    mx.eval(out)
    dt = time.perf_counter() - t0
    stats = getattr(model, "_last_spec_stats", None)
    extra = (
        f"  accepted={stats['accepted']}/drafted={stats['drafted']}" if stats else ""
    )
    print(
        f"{label:<22} prefill {PREFILL} + decode {DECODE}: {dt:.2f}s  "
        f"→ {DECODE / dt:7.1f} tok/s{extra}"
    )


print("== decode bench D768 L8 E256 K8 I320 (bf16) ==")
run(False, "标准 decode")
run(True, "MTP 投机 decode")
