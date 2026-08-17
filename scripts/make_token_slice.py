"""按 token 预算切取 train 文件前缀，供短预算实验独立训练（cosine 终点对齐）。
用法: uv run scripts/make_token_slice.py <src> <dst> <max_tokens>
token 计数口径 = len(enc)+2（<bos>+text+<eos>，与训练一致）。
"""
import json, os, sys, time
from transformers import AutoTokenizer

src, dst, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained('./model/')
if os.path.exists(dst):
    print('exists, skip:', dst)
    sys.exit(0)
t0=time.time(); count=0; wrote=0; docs=0
with open(src,'rb') as f, open(dst,'w',encoding='utf-8') as out:
    for line in f:
        if not line.strip(): continue
        text=line.decode('utf-8','ignore')
        try:
            t=len(tok(json.loads(text)['text'], add_special_tokens=False)['input_ids'])+2
        except Exception:
            t=0
        if count + t > max_tokens and wrote > 0:
            break
        out.write(text)
        count += t; wrote += 1; docs += 1
        if docs % 100000 == 0:
            print(f'{docs} docs, {count/1e6:.1f}M tokens, {time.time()-t0:.0f}s', flush=True)
print(f'done: {docs} docs, {count/1e6:.2f}M tokens -> {dst} ({time.time()-t0:.0f}s)')
