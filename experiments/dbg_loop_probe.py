"""探针：loop 模型主干 hidden 在位置 j 是否还能线性读出当前 token j。

对比 loop / no-loop 两个 checkpoint：
  - cur_ce: 用 lm_head 从 h_j 直接读 token j 的 CE（self-readout，正常模型很低）
  - next_ce: 正常 LM 任务（h_j 读 token j+1）
若 loop 的 cur_ce 显著高于 no-loop，说明第二遍 visit 把当前 token 信息
从残差流里洗掉了（主 head 平台期、MTP 靠显式 embedding 输入幸免的解释）。

用法: uv run python experiments/dbg_loop_probe.py CKPT_A CKPT_B
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import mlx.core as mx

from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.kernels.ce import cross_entropy


def load(ckpt_dir):
    meta = json.load(open(os.path.join(ckpt_dir, "pretrain_768.json")))
    cfg = VibyConfig.from_dict(meta["config"])
    model = VibyForCausalLM(cfg, skip_init=True)
    model.load_weights(os.path.join(ckpt_dir, "pretrain_768.safetensors"))
    model.eval()
    return model


def probe(model, tokenizer, n_batches=8, B=8, T=512):
    # 真实文本：从训练语料取前若干篇，截断/拼到 T
    import json as _json

    docs = []
    with open("/Volumes/pan/text/pretrain_t2t_dedup.jsonl") as f:
        for line in f:
            docs.append(_json.loads(line)["text"])
            if len(docs) >= n_batches * B:
                break
    cur_ces, next_ces = [], []
    for i in range(n_batches):
        ids = []
        for b in range(B):
            t = tokenizer(docs[i * B + b], add_special_tokens=True)["input_ids"]
            t = t[:T] + [2] * max(0, T - len(t))
            ids.append(t)
        ids = mx.array(ids)
        out = model(ids)
        h = out.hidden_states  # final_norm 后 (B,T,D)
        logits = model._lm_logits(h)
        # 当前 token 自读：位置 j 读 ids[j]
        ce_cur = cross_entropy(logits, ids)
        # 下一 token：位置 j 读 ids[j+1]
        ce_next = cross_entropy(logits[:, :-1], ids[:, 1:])
        cur_ces.append(float(ce_cur))
        next_ces.append(float(ce_next))
    return sum(cur_ces) / len(cur_ces), sum(next_ces) / len(next_ces)


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./model/")
    for d in sys.argv[1:]:
        model = load(d)
        cur, nxt = probe(model, tokenizer)
        print(f"{d}:  cur_token_ce={cur:.3f}  next_token_ce={nxt:.3f}")


if __name__ == "__main__":
    main()
