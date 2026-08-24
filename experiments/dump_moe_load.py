"""跑若干真实训练微批，导出各 MoE 调用槽位的逐专家桶计数到 npz。

用于离线评估桶分组策略（分组大小 EG / 是否按负载排序成组）对 padding
倍率的影响——padding 倍率直接决定专家 padded GEMM 白算的比例。

抓取点是 MoEFeedForward.update_capacity_table：_pending_counts 里正是本
微批各调用槽位的真实逐专家 pair 计数。

用法:
    .venv/bin/python experiments/dump_moe_load.py --steps 40 --out /tmp/moe_load.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BASE_ARGV = [
    "--hidden_size",
    "768",
    "--num_hidden_layers",
    "1",
    "--num_attention_heads",
    "8",
    "--learning_rate",
    "0.01",
    "--n_routed_experts",
    "112",
    "--num_experts_per_tok",
    "6",
    "--n_shared_experts",
    "1",
    "--moe_intermediate_size",
    "104",
    "--routed_scaling_factor",
    "2.5",
    "--moe_diversity_loss_weight",
    "0.01",
    "--pack_sequences",
    "--doc_mask",
    "--batch_size",
    "6",
    "--accumulation_steps",
    "2",
    "--max_seq_len",
    "2048",
    "--no_swanlab",
    "--save_interval",
    "10000000",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", type=str, default="/tmp/moe_load.npz")
    p.add_argument("--data_path", type=str, default="/tmp/viby_bench/pretrain.jsonl")
    cli = p.parse_args()

    from model.moe import MoEFeedForward

    rec: list[np.ndarray] = []
    slots: list[int] = []
    orig_upd = MoEFeedForward.update_capacity_table

    def upd(self):
        for k, c in self._pending_counts.items():
            rec.append(np.asarray(c.tolist(), dtype=np.int64))
            slots.append(int(k))
        return orig_upd(self)

    MoEFeedForward.update_capacity_table = upd

    sys.argv = [
        "train_pretrain.py",
        "--data_path",
        cli.data_path,
        "--max_steps",
        str(cli.steps),
        "--log_interval",
        "1000",
        "--out_dir",
        "/tmp/viby_bench/dump",
        *BASE_ARGV,
    ]
    import runpy

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        runpy.run_path(
            os.path.join(root, "trainer", "train_pretrain.py"), run_name="__main__"
        )
    finally:
        MoEFeedForward.update_capacity_table = orig_upd

    arr = np.stack(rec) if rec else np.zeros((0, 112), np.int64)
    np.savez_compressed(cli.out, loads=arr, slots=np.asarray(slots))
    r = arr.max(1) / arr.mean(1)
    print(f"\n导出 {arr.shape[0]} 条逐专家计数 (E={arr.shape[1]}) -> {cli.out}")
    print(f"max/mean 分位 p10/p50/p90: {np.percentile(r, [10, 50, 90]).round(2)}")


if __name__ == "__main__":
    main()
