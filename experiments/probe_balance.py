"""均衡控制器动力学探针：从集中态 checkpoint 出发，冻结权重只做前向，
模拟 per-slot bias 控制器对既有集中的收敛速度。

用法: .venv/bin/python experiments/probe_balance.py <rate> [n_micro]
例:   .venv/bin/python experiments/probe_balance.py 0.01 600
"""

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model import MoEFeedForward, VibyConfig, VibyForCausalLM
from trainer.base_trainer import MLXDataLoader

CKPT_DIR = "research_runs/r073_hrm_moe_cycledelta_bad1"
DATA = "/Volumes/pan/text/pretrain_train_mini.jsonl"
BS, SEQ, ACCUM = 6, 2048, 2


def main():
    rate = float(sys.argv[1])
    n_micro = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    meta = json.load(open(f"{CKPT_DIR}/pretrain_768.json"))
    cfg = VibyConfig(**meta["config"])
    model = VibyForCausalLM(cfg)
    weights = mx.load(f"{CKPT_DIR}/pretrain_768.safetensors")
    expect = dict(tree_flatten(model.parameters()))
    for k, v in list(weights.items()):
        tgt = expect.get(k)
        if (
            k.endswith("expert_bias")
            and tgt is not None
            and v.ndim == 1
            and len(tgt.shape) == 2
            and tgt.shape[1] == v.shape[0]
        ):
            weights[k] = mx.broadcast_to(v[None, :], tuple(tgt.shape))
    model.load_weights(list(weights.items()))
    model.train()  # 走训练路径（collect_stats / 稀疏桶）

    gates = model.moe_gates()
    for g in gates:
        g.collect_stats = True
    moes = [m for m in model.modules() if isinstance(m, MoEFeedForward)]

    tok = AutoTokenizer.from_pretrained("./model/")
    ds = PretrainDataset(DATA, tok, max_length=SEQ, pack_sequences=True, doc_mask=True)
    loader = MLXDataLoader(ds, batch_size=BS, shuffle=True, drop_last=True)
    pad_id = tok.pad_token_id

    t0 = time.time()
    win_stats = None
    overflow = 0
    for step, batch in enumerate(loader):
        if step >= n_micro:
            break
        if len(batch) == 4:
            X, Y, loss_mask, seg_ids = batch
        else:
            X, Y, loss_mask = batch
            seg_ids = None
        attn_mask = (X != pad_id).astype(mx.int32)
        mask_has_pad = bool(mx.any(attn_mask != 1).item())
        res = model(
            input_ids=X,
            labels=Y,
            loss_mask=loss_mask,
            attention_mask=attn_mask,
            mask_has_pad=mask_has_pad,
            segment_ids=seg_ids,
        )
        stats = model.moe_load_stats()
        mx.eval(res.loss, stats)
        for m in moes:
            overflow += m.update_capacity_table()
        win_stats = stats if win_stats is None else mx.add(win_stats, stats)
        if (step + 1) % ACCUM == 0:
            model.update_moe_biases(win_stats, rate)
            win_stats = None
        if step % 50 == 0 or step == n_micro - 1:
            gate_max = [
                float(g.last_load.max()) / 1024
                for g in gates
                if g.last_load is not None
            ]
            bias_ms = [
                f"{float(g.expert_bias.mean()):+.2f}/{float(g.expert_bias.std()):.2f}"
                for g in gates
            ]
            cmax = max((m._c_max_seen for m in moes), default=0)
            for m in moes:
                m._c_max_seen = 0
            el = time.time() - t0
            print(
                f"micro {step:4d} ({el:6.1f}s): loss={float(res.loss):.3f} "
                f"gate_maxK={'/'.join(f'{v:.1f}' for v in gate_max)} "
                f"callC={cmax} overflow_累计={overflow} "
                f"bias(mean/std)={'/'.join(bias_ms)}",
                flush=True,
            )
            overflow = 0


if __name__ == "__main__":
    main()
