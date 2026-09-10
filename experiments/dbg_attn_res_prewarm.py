"""探针：loop anchor 下 AttnRes 融合 kernel 的 prewarm 覆盖与步耗时。

复刻 trainer 流程：convert_model_dtype(bf16) → prewarm_all →
mx.compile(value_and_grad) → 预热 1 步后计时 N 步。
打印 attn_res_fused._VERIFIED / _FAILED 与每步耗时。

用法: uv run python experiments/dbg_attn_res_prewarm.py [n_steps]
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import mlx.core as mx
import mlx.nn as nn

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.kernels import prewarm_all
from model.kernels import attn_res_fused
from trainer.utils import convert_model_dtype


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    meta = json.load(open("research_runs/viby_loop/pretrain_768.json"))
    cfg = VibyConfig.from_dict(meta["config"])
    cfg.ngram_table_size = 4096
    for arg in sys.argv[2:]:
        if arg == "noanchor":
            cfg.loop_anchor = False
        elif arg == "noloop":
            cfg.loop_span = 0
            cfg.loop_anchor = False
    print(f"config: loop_span={cfg.loop_span} loop_anchor={cfg.loop_anchor}")
    model = VibyForCausalLM(cfg, skip_init=True)
    convert_model_dtype(model, "bfloat16")
    model.train()

    prewarm_all(model, cfg, mx.bfloat16, 1024, log=print)
    print("prewarm 后 VERIFIED N:", sorted(k[0] for k in attn_res_fused._VERIFIED))

    def loss_fn(x):
        return model(input_ids=x, labels=x).loss

    fn = mx.compile(nn.value_and_grad(model, loss_fn))
    B, T = 8, 512
    x = mx.random.randint(0, cfg.vocab_size, (B, T))

    loss, grads = fn(x)  # compile + 预热
    mx.eval(loss, grads)
    print(
        "compile 步后 FAILED:",
        sorted(attn_res_fused._FAILED),
    )
    print("VERIFIED N:", sorted(k[0] for k in attn_res_fused._VERIFIED))

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        loss, grads = fn(x)
        mx.eval(loss, grads)
        times.append(time.perf_counter() - t0)
    print(f"steps: {[f'{t * 1000:.0f}ms' for t in times]}")
    print(f"median: {sorted(times)[len(times) // 2] * 1000:.0f}ms/step (B={B},T={T})")


if __name__ == "__main__":
    main()
