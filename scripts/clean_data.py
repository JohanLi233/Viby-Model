"""用 LFM2.5-Encoder-350M-corpus-cleaner 清洗预训练语料。

逐 token KEEP/DELETE 分类，按 offset 重组仅保留 KEEP 的文本。
用法:
    python scripts/clean_data.py --input '/Volumes/pan/text/pretrain_t2t_mini.jsonl' \
        --output '/Volumes/pan/text/pretrain_t2t_mini_cleaned.jsonl' --max_docs 150000
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", default="yuyijiong/LFM2.5-Encoder-350M-corpus-cleaner")
    ap.add_argument("--max_docs", type=int, default=150000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--min_keep_chars", type=int, default=100,
                    help="清洗后少于此字符数的文档整条丢弃")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.to(device).eval()

    # 读取文档
    docs = []
    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if i >= args.max_docs:
                break
            docs.append(json.loads(line.decode("utf-8", errors="ignore"))["text"])
    print(f"读取 {len(docs)} 篇文档")

    n_written, n_dropped = 0, 0
    total_in_chars, total_out_chars = 0, 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(docs), args.batch_size):
            batch = docs[start : start + args.batch_size]
            enc = tokenizer(
                batch,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").tolist()
            inputs = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
            labels = logits.argmax(dim=-1).tolist()

            for text, offs, labs in zip(batch, offsets, labels):
                ranges = [
                    (s, e)
                    for (s, e), lab in zip(offs, labs)
                    if lab == 0 and e > s
                ]
                ranges.sort()
                merged = []
                for s, e in ranges:
                    if merged and s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))
                cleaned = "".join(text[s:e] for s, e in merged).strip()
                total_in_chars += len(text)
                total_out_chars += len(cleaned)
                if len(cleaned) < args.min_keep_chars:
                    n_dropped += 1
                    continue
                out.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
                n_written += 1

            done = min(start + args.batch_size, len(docs))
            rate = total_in_chars / max(time.time() - t0, 1e-9)
            print(
                f"[{done}/{len(docs)}] {rate / 1000:.0f}k chars/s "
                f"keep率 {total_out_chars / max(total_in_chars, 1):.2%} "
                f"整篇丢弃 {n_dropped}",
                flush=True,
            )

    print(
        f"完成: {n_written} 篇写入 {args.output}, {n_dropped} 篇整篇丢弃, "
        f"字符保留率 {total_out_chars / max(total_in_chars, 1):.2%}, "
        f"耗时 {(time.time() - t0) / 60:.1f} 分钟"
    )


if __name__ == "__main__":
    main()
