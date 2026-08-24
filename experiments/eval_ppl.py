"""Held-out 困惑度（PPL）评估：用固定尾部文档集对比各轮 checkpoint。

判据设计：
- 取语料文件尾部 --n_docs 篇文档作为固定评估集（所有轮次同一集合，保证可比）。
- 与训练打包口径一致：tokenizer(add_special_tokens=False) + 每篇末尾追加 eos。
- 仅用主模型 logits 计算 next-token CE（不含 MTP 辅助 loss），因此 mtp0/mtp1
  的 checkpoint 可以公平对比，而训练日志的报告 loss 口径不同不可直接比。
- 主指标为全体 token 加权平均 CE 的指数（overall PPL），辅以逐篇均值 PPL。

用法：
  uv run experiments/eval_ppl.py --ckpt out_exp/round28_packed_seq1024/pretrain_768.safetensors \
      --tag round28_packed_seq1024
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.utils import load_model_weights

warnings.filterwarnings("ignore")

RESULTS_TSV = os.path.join(os.path.dirname(__file__), "ppl_results.tsv")

LAR_TSV_COLUMNS = ("unemb_lar", "unemb_lar_eff_dim")


def load_eval_docs(data_path: str, n_docs: int):
    """流式读取文件尾部 n_docs 篇文档的文本（保持文件原始顺序）。"""
    buf = deque(maxlen=n_docs)
    with open(data_path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                buf.append(json.loads(line.decode("utf-8", errors="ignore"))["text"])
    return list(buf)


def unembedding_lar(
    model: VibyForCausalLM,
    logits: mx.array,
    hidden: mx.array,
    valid_mask: mx.array,
):
    """计算 unembedding 的 log-alignment ratio（LAR）。

    论文定义：LAR = log_n(||WX||_RMS / (||W||_RMS * ||X||_RMS))。
    logits 已是 X @ W.T，因此这里只聚合现有前向输出，不重复做 GEMM。
    返回 LAR 和论文中的等效函数维度 k = n^(2*(1-LAR))。
    """
    lm_head = getattr(model, "lm_head", None)
    weight = lm_head.weight if lm_head is not None else model.model.embed_tokens.weight
    weight_f32 = weight.astype(mx.float32)
    logits_f32 = logits.astype(mx.float32)
    hidden_f32 = hidden.astype(mx.float32)

    valid = valid_mask.astype(mx.float32)
    token_count = mx.maximum(mx.sum(valid), 1.0)
    output_dim, input_dim = weight.shape
    wx_ms = mx.sum(logits_f32 * logits_f32 * valid[:, :, None]) / (
        token_count * output_dim
    )
    w_ms = mx.sum(weight_f32 * weight_f32) / (output_dim * input_dim)
    x_ms = mx.sum(hidden_f32 * hidden_f32 * valid[:, :, None]) / (
        token_count * input_dim
    )
    lar = (
        0.5
        * (mx.log(wx_ms) - mx.log(w_ms) - mx.log(x_ms))
        / mx.log(mx.array(input_dim, dtype=mx.float32))
    )
    eff_dim = mx.exp(2.0 * (1.0 - lar) * mx.log(mx.array(input_dim)))
    return float(lar.item()), float(eff_dim.item())


def append_ppl_result(row: dict):
    """追加结果并兼容旧版 10 列 TSV：缺列时补齐表头和历史行。"""
    base_columns = [
        "round",
        "date",
        "n_docs",
        "tokens",
        "mean_ce",
        "overall_ppl",
        "doc_ppl",
        "eval_s",
        "ckpt",
        "notes",
    ]
    rewrite_needed = False
    lines = []
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
        if lines:
            old_columns = lines[0].split("\t")
            rewrite_needed = any(col not in old_columns for col in LAR_TSV_COLUMNS)
            if rewrite_needed:
                columns = old_columns + [
                    col for col in LAR_TSV_COLUMNS if col not in old_columns
                ]
                lines[1:] = [
                    line + "\t" * (len(columns) - len(line.split("\t")))
                    for line in lines[1:]
                ]
                lines[0] = "\t".join(columns)
            else:
                columns = old_columns
        else:
            columns = base_columns + list(LAR_TSV_COLUMNS)
            rewrite_needed = True
    else:
        columns = base_columns + list(LAR_TSV_COLUMNS)

    missing = [col for col in columns if col not in row]
    if missing:
        raise ValueError(f"缺少结果字段: {missing}")
    line = "\t".join(str(row[col]) for col in columns)
    if lines:
        lines.append(line)
        content = "\n".join(lines) + "\n"
    else:
        content = "\t".join(columns) + "\n" + line + "\n"

    if rewrite_needed or not os.path.exists(RESULTS_TSV):
        temp_path = RESULTS_TSV + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, RESULTS_TSV)
    else:
        with open(RESULTS_TSV, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Viby held-out PPL evaluation")
    parser.add_argument(
        "--ckpt", required=True, type=str, help="safetensors 检查点路径"
    )
    parser.add_argument(
        "--data_path",
        # HQ 语料已于 2026-08-16 删除，主力数据换为 t2t_mini；评估集=该文件
        # 尾部 n_docs 篇。ppl_results.tsv 中 HQ 时代记录（round11~33）与
        # t2t_mini 时代记录口径不同，不可直接对比。
        default="/Volumes/pan/text/pretrain_t2t_mini.jsonl",
        type=str,
    )
    parser.add_argument("--n_docs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--tag", type=str, default=None, help="写入 tsv 的轮次名")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--no_save", action="store_true", help="只打印，不追加 ppl_results.tsv"
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="宽松加载：忽略 checkpoint 中多余/缺失/形状不一致的非必需参数"
        "（用于 checkpoint 保存后模型代码发生非破坏性升级的评估）",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    sidecar = os.path.splitext(args.ckpt)[0] + ".json"
    if not os.path.exists(sidecar):
        # step 快照没有独立 sidecar，回退到 canonical pretrain_{hidden}.json
        import re

        canonical = re.sub(r"_step\d+$", "", os.path.splitext(args.ckpt)[0]) + ".json"
        if os.path.exists(canonical):
            sidecar = canonical
    with open(sidecar, "r", encoding="utf-8") as f:
        meta = json.load(f)
    config = VibyConfig.from_dict(meta["config"])
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    model = VibyForCausalLM(config)
    if not load_model_weights(
        model,
        args.ckpt,
        strict=not args.loose,
        label="checkpoint",
    ):
        raise SystemExit(f"无法加载检查点 {args.ckpt}")
    model.eval()

    t0 = time.time()
    texts = load_eval_docs(args.data_path, args.n_docs)
    eos_id = tokenizer.eos_token_id
    # 输入长度须 ≤ 模型上下文（640 的老 checkpoint 截断到 640）
    seq_cap = min(args.max_seq_len, config.max_position_embeddings)

    # 与训练打包口径一致：不加特殊 token，末尾显式 eos
    doc_ids = []
    for enc in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        ids = (enc + [eos_id])[:seq_cap]
        if len(ids) >= 2:
            doc_ids.append(ids)
    print(
        f"[ppl] {os.path.basename(os.path.dirname(args.ckpt))}: "
        f"{len(doc_ids)} docs, mtp_depth={config.mtp_depth}, "
        f"加载+tokenize {time.time() - t0:.1f}s"
    )

    total_ce = 0.0
    total_tok = 0
    doc_ce_sum = 0.0  # 逐篇平均 CE 的和（每篇等权）
    lar_sum = 0.0
    lar_weight_sum = 0
    t1 = time.time()
    for i in range(0, len(doc_ids), args.batch_size):
        batch = doc_ids[i : i + args.batch_size]
        max_len = max(len(x) for x in batch)
        ids = np.full((len(batch), max_len), eos_id, dtype=np.int32)
        mask = np.zeros((len(batch), max_len), dtype=np.float32)
        for r, x in enumerate(batch):
            ids[r, : len(x)] = x
            mask[r, : len(x)] = 1.0

        out = model(mx.array(ids), attention_mask=mx.array(mask))
        logits = out.logits.astype(mx.float32)[:, :-1, :]  # (B, T-1, V)
        hidden = out.hidden_states[:, :-1, :]
        batch_lar, _ = unembedding_lar(
            model,
            logits,
            hidden,
            mx.array(mask[:, :-1]),
        )
        batch_valid = int(mask[:, :-1].sum().item())
        lar_sum += batch_lar * batch_valid
        lar_weight_sum += batch_valid
        tgt = mx.array(ids[:, 1:])  # (B, T-1)
        valid = mx.array(mask[:, 1:])
        log_z = mx.logsumexp(logits, axis=-1)
        gathered = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
        ce = (log_z - gathered) * valid
        tok_per_doc = valid.sum(axis=1)
        ce_per_doc = ce.sum(axis=1) / mx.maximum(tok_per_doc, 1.0)
        # 同步取回，释放显存
        batch_ce = float(ce.sum().item())
        batch_tok = int(valid.sum().item())
        tok_np = np.array(tok_per_doc)
        doc_ce = np.array(ce_per_doc)
        total_ce += batch_ce
        total_tok += batch_tok
        doc_ce_sum += float(doc_ce[tok_np > 0].sum())

    mean_ce = total_ce / max(total_tok, 1)
    ppl = math.exp(mean_ce)
    doc_ppl = math.exp(doc_ce_sum / max(len(doc_ids), 1))
    mean_lar = lar_sum / max(lar_weight_sum, 1)
    # 论文等效维度公式以矩阵输入维 n 为底；tied unembedding 的 n=hidden_size。
    unembedding_input_dim = (
        model.lm_head.weight.shape[1]
        if model.lm_head is not None
        else model.model.embed_tokens.weight.shape[1]
    )
    mean_lar_eff_dim = unembedding_input_dim ** (2.0 * (1.0 - mean_lar))
    elapsed = time.time() - t1

    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))
    line = (
        f"[ppl] {tag}: overall_ppl={ppl:.4f} doc_ppl={doc_ppl:.4f} "
        f"mean_ce={mean_ce:.4f} tokens={total_tok} docs={len(doc_ids)} "
        f"unemb_lar={mean_lar:.6f} "
        f"unemb_lar_eff_dim={mean_lar_eff_dim:.3f} "
        f"eval_time={elapsed:.1f}s"
    )
    print(line)

    if not args.no_save:
        append_ppl_result(
            {
                "round": tag,
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "n_docs": len(doc_ids),
                "tokens": total_tok,
                "mean_ce": f"{mean_ce:.4f}",
                "overall_ppl": f"{ppl:.4f}",
                "doc_ppl": f"{doc_ppl:.4f}",
                "eval_s": f"{elapsed:.1f}",
                "ckpt": args.ckpt,
                "notes": args.notes,
                "unemb_lar": f"{mean_lar:.6f}",
                "unemb_lar_eff_dim": f"{mean_lar_eff_dim:.4f}",
            }
        )


if __name__ == "__main__":
    main()
