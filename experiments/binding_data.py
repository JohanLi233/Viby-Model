"""Deterministic Chinese random relation worlds; no latent/address supervision."""
import hashlib
import json
from pathlib import Path

import numpy as np


def make_world(tokenizer, rng, symbols, *, confirmation=False, world_id=0):
    ids = [tokenizer.bos_token_id]
    answers, questions = [], []
    def text(s):
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    def entity(n):
        ids.append(symbols[n][1])
    n = len(symbols)
    maps = rng.integers(n, size=(2,n))
    text("以下关系以最新记录为准。\n")
    records = [(h,i) for h in range(2) for i in range(n)]
    rng.shuffle(records)
    def record(h, i):
        text("甲：" if h == 0 else "乙："); entity(i);text("→");entity(int(maps[h,i]));text("。\n")
    for h,i in records: record(h,i)
    for phase in range(2):
        if phase:
            h,i = int(rng.integers(2)),int(rng.integers(n))
            old=int(maps[h,i]);maps[h,i]=(old+int(rng.integers(1,n)))%n
            text("更正记录：");record(h,i)
        for _ in range(6):
            operations = "gf" if confirmation else str(rng.choice(["f","g","ff","fg","gg"]))
            start=int(rng.integers(n));result=start;intermediate=start
            question_start=len(ids)
            text("问：从");entity(start)
            for j,op in enumerate(operations):
                text("先沿" if j==0 else "再沿");text("甲" if op=="f" else "乙")
                result=int(maps[0 if op=="f" else 1,result])
                if j==0:intermediate=result
            text("，到哪里？答：")
            position=len(ids)-1  # input position predicting the answer
            answers.append(position)
            questions.append(dict(position=position, question_start=question_start, operations=operations,
                                  start=start, intermediate=intermediate, target=result,
                                  second_map=maps[0 if operations[-1]=="f" else 1].tolist(), phase=phase))
            entity(result);text("。\n")
    ids.append(tokenizer.eos_token_id)
    if len(ids)-1 > 1024:
        raise ValueError(f"world context {len(ids)-1} exceeds 1024")
    return dict(world_id=world_id, ids=ids, answers=answers, questions=questions)


def prepare(root, tokenizer_path, seed=20260914):
    from transformers import AutoTokenizer
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True)
    symbols=[]
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        encoded=tokenizer.encode(letter,add_special_tokens=False)
        if len(encoded)==1 and tokenizer.decode(encoded)==letter:
            symbols.append((letter,encoded[0]))
        if len(symbols)==8:break
    if len(symbols)!=8:raise ValueError("need eight existing single-token symbols")
    streams={}
    for split,offset,count in [("development",1,16),("confirmation",2,128)]:
        rng=np.random.default_rng(seed+offset)
        streams[split]=[make_world(tokenizer,rng,symbols,confirmation=split=="confirmation",world_id=i) for i in range(count)]
    rng=np.random.default_rng(seed)
    worlds=[];plan=[];total=0;world_index=0;world=None;cursor=0
    for step in range(256):
        window=[];remaining=2048
        while remaining:
            if world is None:
                world=make_world(tokenizer,rng,symbols,world_id=world_index)
                worlds.append(world);world_index+=1;cursor=0
            n=len(world["ids"])-1
            take=min(n-cursor,remaining)
            window.append(dict(world=world["world_id"],begin=cursor,end=cursor+take))
            cursor+=take;remaining-=take;total+=take
            if cursor==n:world=None
        plan.append(window)
    assert total==524288
    streams['train']=worlds
    for split,worlds in streams.items():
        with (root/f'{split}.jsonl').open('w') as f:
            for world in worlds:f.write(json.dumps(world,ensure_ascii=False)+'\n')
    (root/'plan.json').write_text(json.dumps(plan))
    manifest=dict(seed=seed,symbols=symbols,train_tokens=total,steps=256,global_batch=2048,
                  worlds={k:len(v) for k,v in streams.items()},max_context=max(len(w['ids'])-1 for ws in streams.values() for w in ws),
                  tokenizer_sha256=hashlib.sha256((Path(tokenizer_path)/'tokenizer.json').read_bytes()).hexdigest(),
                  policy='All ordinary next-token labels scored once. Worlds crossing update boundaries are recomputed with disjoint loss masks; repeated physical context is counted separately. No latent targets. Confirmation has gf on new worlds; train has f,g,ff,fg,gg.')
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def batch(world, begin=0, end=None, pad_id=0):
    ids=np.array(world['ids'],np.int32);length=len(ids)-1
    width=((length+31)//32)*32
    x=np.full((1,width),pad_id,np.int32);y=x.copy();valid=np.zeros((1,width),np.float32)
    x[0,:length]=ids[:-1];y[0,:length]=ids[1:]
    valid[0,begin:length if end is None else end]=1
    return x,y,valid,(x!=pad_id)
