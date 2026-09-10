"""一次性诊断：full vs jfb 在真实 viby 配置下的分组梯度范数对比。

用法: uv run python experiments/dbg_jfb_grads.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.model import VibyForCausalLM


def make_config(loop_grad_mode):
    return VibyConfig(
        hidden_size=768,
        num_hidden_layers=9,
        num_attention_heads=8,
        kv_lora_rank=256,
        qk_rope_head_dim=64,
        vocab_size=8192,
        max_position_embeddings=1024,
        mtp_depth=1,
        mtp_loss_weight=0.3,
        mtp_steps=1,
        use_attn_gate=True,
        n_routed_experts=128,
        num_experts_per_tok=8,
        n_shared_experts=2,
        moe_intermediate_size=512,
        moe_latent_dim=384,
        routed_scaling_factor=2.5,
        moe_route_scale=True,
        first_k_dense_replace=1,
        dense_intermediate_size=1024,
        ngram_table_size=4194304,
        ngram_layer=2,
        loop_span=4,
        loop_count=2,
        loop_grad_mode=loop_grad_mode,
        loop_anchor=os.environ.get("LOOP_ANCHOR", "1") == "1",
    )


def group_grads(model, grads):
    """按位置分组汇总梯度 RMS：embed / pre-span(layer0-1) / span(2-5) /
    post-span(6-8) / mtp / lm_head。"""
    groups = {
        "embed": [],
        "pre_span": [],
        "span": [],
        "post_span": [],
        "mtp": [],
        "lm_head": [],
        "other": [],
    }
    for path, g in tree_flatten(grads):
        rms = float(mx.sqrt(mx.mean(g.astype(mx.float32) ** 2)))
        if "embed_tokens" in path:
            groups["embed"].append(rms)
        elif path.startswith("model.stack.layers."):
            idx = int(path.split(".")[3])
            if idx < 2:
                groups["pre_span"].append(rms)
            elif idx < 6:
                groups["span"].append(rms)
            else:
                groups["post_span"].append(rms)
        elif "mtp_modules" in path:
            groups["mtp"].append(rms)
        elif "lm_head" in path:
            groups["lm_head"].append(rms)
        else:
            groups["other"].append(rms)
    return {
        k: (sum(v) / len(v) if v else 0.0, len(v)) for k, v in groups.items()
    }


def main():
    mx.random.seed(1337)
    B, T = 4, 256
    input_ids = mx.random.randint(0, 8192, (B, T))
    labels = mx.concatenate([input_ids[:, 1:], mx.zeros((B, 1), dtype=input_ids.dtype)], axis=1)

    results = {}
    for mode in ("full", "jfb"):
        mx.random.seed(42)  # 两种模式同初始化
        model = VibyForCausalLM(make_config(mode))
        model.train()

        def loss_fn(params):
            model.update(params)
            out = model(input_ids, labels=labels)
            return out.loss

        loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        mx.eval(loss, grads)
        results[mode] = (float(loss), group_grads(model, grads))

    print(f"{'group':<10} {'full_rms':>12} {'jfb_rms':>12} {'jfb/full':>10}")
    lf, gf = results["full"]
    lj, gj = results["jfb"]
    print(f"loss: full={lf:.6f} jfb={lj:.6f}")
    for k in gf:
        f, n = gf[k]
        j, _ = gj[k]
        ratio = j / f if f > 0 else float("nan")
        print(f"{k:<10} {f:>12.4e} {j:>12.4e} {ratio:>10.3f}  (n={n})")


if __name__ == "__main__":
    main()
