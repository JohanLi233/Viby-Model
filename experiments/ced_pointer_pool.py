"""Measure random pointer-chain accessibility in Viby boundary candidate pools.

Uses an untrained MLX model and random labels. This is an actual routing-domain
diagnostic, not a learned task-accuracy benchmark or an attention attribution.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from model.config import VibyConfig
from model.model import VibyForCausalLM


def run(seed=20260912, tasks=64, nodes=512):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    cfg = VibyConfig(
        preset="tiny",
        n_layers=12,
        dim=64,
        n_heads=2,
        o_groups=1,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_activated_experts=2,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=64,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        window_size=128,
        vocab_size=4 * nodes + 16,
        max_seq_len=3 * nodes + 4,
        engram_layer_ids=(),
        n_mtp_layers=0,
        psr_enabled=False,
        ced_recurrent_enabled=True,
    )
    model = VibyForCausalLM(cfg)
    model.eval()
    counts = {phase: {hops: 0 for hops in (1, 2, 4, 8)} for phase in range(4)}
    pool_sizes = []
    for _ in range(tasks):
        permutation = rng.permutation(nodes)
        pointers = np.empty(nodes, dtype=np.int32)
        pointers[permutation] = np.roll(permutation, -1)
        labels = rng.integers(2 * nodes + 1, 4 * nodes + 1, size=nodes)
        order = rng.permutation(nodes)
        rank = np.argsort(order)
        # Each fact is [key, successor, independent random terminal label].
        facts = np.column_stack(
            [order + 1, pointers[order] + 1, labels[order]]
        ).reshape(-1)
        start = int(rng.integers(nodes))
        for phase in range(4):
            tokens = np.concatenate([facts, np.full(phase, 4 * nodes + 1), [start + 1]])
            out = model(mx.array(tokens[None], mx.int32))
            plan = out.ced_result["plan"]
            last = int(mx.sum(plan.valid)) - 1
            pool = np.array(
                out.ced_result["candidate_pool"][0, last].tolist(), dtype=bool
            )
            pool_sizes.append(int(pool.sum()))
            for hops in (1, 2, 4, 8):
                current = start
                accessible = True
                for _ in range(hops):
                    accessible &= bool(pool[3 * rank[current] + 1])
                    current = int(pointers[current])
                accessible &= bool(pool[3 * rank[current] + 2])
                counts[phase][hops] += int(accessible)
    return dict(
        seed=seed,
        tasks_per_phase=tasks,
        nodes=nodes,
        prompt_length_range=[3 * nodes + 1, 3 * nodes + 4],
        config=cfg.to_dict(),
        kind="actual_untrained_viby_candidate_domain_random_pointer_probe",
        mean_candidate_tokens=float(np.mean(pool_sizes)),
        full_memory_chain_accessibility=1.0,
        fixed_pool_chain_accessibility={
            str(p): {str(h): n / tasks for h, n in rows.items()}
            for p, rows in counts.items()
        },
        scope="Real MLX boundary candidate IDs, fixed across recurrent reads; scores are pointer-value/label location coverage, not model prediction accuracy. Key matching, encoder computation, latent SWA and the last token layer can convey additional information, so this is not an upper bound on end-to-end accuracy. Phases 0-2 use the preceding completed anchor; phase 3 sees the query. No training or label memorization.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.tasks < 1 or args.nodes < 12 or args.nodes % 4:
        parser.error("tasks must be positive; nodes must be >=12 and divisible by 4")
    result = run(args.seed, args.tasks, args.nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
