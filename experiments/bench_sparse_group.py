"""稀疏桶分组基准：真实集中态 checkpoint + 真实数据，测量
_SPARSE_GROUP（专家分组大小）对训练步时/吞吐/峰值内存的影响。

用法: .venv/bin/python experiments/bench_sparse_group.py <group> [steps]
例:   .venv/bin/python experiments/bench_sparse_group.py 4 24
"""

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model import VibyConfig, VibyForCausalLM, MoEFeedForward
from trainer.base_trainer import MLXDataLoader

CKPT_DIR = "research_runs/r073_hrm_moe_cycledelta_bad1"
DATA = "/Volumes/pan/text/pretrain_train_mini.jsonl"
BS, SEQ = 6, 2048


def main():
    group = int(sys.argv[1])
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    MoEFeedForward._SPARSE_GROUP = group

    meta = json.load(open(f"{CKPT_DIR}/pretrain_768.json"))
    cfg = VibyConfig(**meta["config"])
    model = VibyForCausalLM(cfg)
    if not os.environ.get("FRESH"):
        # 旧格式 expert_bias (E,) → per-cycle (n_cycles, E) 广播（同 from_pretrained）
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
    model.train()

    tok = AutoTokenizer.from_pretrained("./model/")
    if os.environ.get("SYNTH"):
        # 无外接盘时的合成数据：随机 token + 全 1 loss_mask + 零 seg_ids。
        # A/B 两侧看到完全相同的 batch，吞吐对比有效（注意力侧偏轻，
        # 绝对 tok/s 不可与真实数据直接比）。
        vocab = int(cfg.vocab_size)

        def synth_loader():
            while True:
                X = mx.random.randint(0, vocab, (BS, SEQ)).astype(mx.int32)
                Y = mx.concatenate([X[:, 1:], mx.zeros((BS, 1), X.dtype)], axis=1)
                yield (
                    X,
                    Y,
                    mx.ones((BS, SEQ), dtype=mx.float32),
                    mx.zeros((BS, SEQ), dtype=mx.int32),
                )

        loader = synth_loader()
    else:
        ds = PretrainDataset(
            DATA, tok, max_length=SEQ, pack_sequences=True, doc_mask=True
        )
        loader = MLXDataLoader(ds, batch_size=BS, shuffle=True, drop_last=True)

    def loss_fn(X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids):
        res = model(
            input_ids=X,
            labels=Y,
            loss_mask=loss_mask,
            attention_mask=attn_mask,
            mask_has_pad=mask_has_pad,
            segment_ids=seg_ids,
        )
        mtp = res.mtp_loss if res.mtp_loss is not None else mx.array(0.0)
        return res.loss, mtp

    lg = nn.value_and_grad(model, loss_fn)
    pad_id = tok.pad_token_id

    moes = [m for m in model.modules() if isinstance(m, MoEFeedForward)]
    times = []
    for step, batch in enumerate(loader):
        if step >= n_steps + 8:
            break
        if len(batch) == 4:
            X, Y, loss_mask, seg_ids = batch
        else:
            X, Y, loss_mask = batch
            seg_ids = None
        attn_mask = (X != pad_id).astype(mx.int32)
        mask_has_pad = bool(mx.any(attn_mask != 1).item())
        t0 = time.time()
        (loss, mtp), grads = lg(X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids)
        mx.eval(loss, mtp)
        mx.eval(grads)
        for m in moes:
            m.update_capacity_table()
        dt = time.time() - t0
        if step >= 8:
            times.append(dt)
        if step % 8 == 0:
            rows = [
                min(group, 112) * sum(m._cap_table.get(k, []))
                for m in moes
                for k in m._cap_table
            ]
            print(
                f"step {step:3d}: {dt:.2f}s loss={float(loss):.3f} "
                f"peak={mx.get_peak_memory() / 2**30:.1f}G rows={sum(rows)}"
            )
    if times:
        avg = sum(times) / len(times)
        print(
            f"== group={group}: {avg:.2f}s/step, "
            f"{BS * SEQ / avg:.0f} tok/s, peak={mx.get_peak_memory() / 2**30:.1f}G"
        )


if __name__ == "__main__":
    main()
