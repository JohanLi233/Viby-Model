"""Same-model serial compiled A/B for key tiling, top-k and native MoE combine."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from model.config import VibyConfig
from model.model import VibyForCausalLM
from model.kernels import sparse_attention as sa, moe_dispatch as moe
from trainer.base_trainer import BaseTrainer
from trainer.config import get_pretrain_parser, setup_training_args
from trainer.utils import build_model_kwargs, convert_model_dtype, resolve_compute_scaled_hparams


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--seg', type=float, default=200)
    args=ap.parse_args()
    cli=['--out_dir','research_runs/_bench','--no_save','--batch_size',str(args.batch),
         '--max_seq_len',str(args.seq),'--accumulation_steps','2','--cache_limit_gb','8',
         '--dtype','bfloat16']
    ta=setup_training_args(get_pretrain_parser().parse_args(cli),'pretrain')
    cfg=VibyConfig(**build_model_kwargs(ta))
    ta=resolve_compute_scaled_hparams(ta,467617)
    mx.random.seed(1234)
    model=VibyForCausalLM(cfg)
    convert_model_dtype(model, getattr(ta, 'dtype', ''))
    for tile in (16,32):
        sa.prewarm_sparse_attention(cfg.head_dim,cfg.window_size,cfg.head_dim**-0.5,mx.bfloat16,key_tile=tile)
    trainer=BaseTrainer(ta,model,None,cfg,'pretrain')
    x=mx.random.randint(0,cfg.vocab_size,(args.batch,args.seq))
    y=mx.random.randint(0,cfg.vocab_size,x.shape)
    mask=mx.ones(x.shape,mx.float32)
    attn=mx.ones(x.shape,mx.int32)
    seg=mx.cumsum((mx.random.uniform(shape=x.shape)<1/args.seg).astype(mx.int32),axis=1)
    mx.eval(x,y,mask,attn,seg)

    def step():
        out,g=trainer._compute_loss_and_grad(x,y,mask,attn,seg)
        mx.eval(*[o for o in out if o is not None])
        mx.eval(g)
        return out,g

    out,g=step()
    trainer.optimizer.update(model,g)
    mx.eval(model.parameters(),trainer.optimizer.state)
    del out,g
    print(json.dumps({'params':model.num_parameters(),'engram_vocab_size':cfg.engram_vocab_size}),flush=True)
    variants=[
        ('base16',16,False,False),
        ('base16_topk',16,True,False),
        ('base16_combine',16,False,True),
        ('tile32',32,False,False),
        ('tile32_topk',32,True,False),
        ('tile32_topk_combine',32,True,True),
    ]
    compiled={}
    for name,tile,topk,combine in variants:
        sa._KEY_TILE=tile; sa._TOPK_ENABLED=topk; moe._COMBINE_ENABLED=combine
        trainer._loss_and_grad=trainer._build_loss_and_grad()
        compiled[name]=trainer._loss_and_grad
        for _ in range(args.warmup):
            step()
        mx.reset_peak_memory()
        times=[]
        for _ in range(args.iters):
            start=time.perf_counter(); step(); times.append(time.perf_counter()-start)
        print(json.dumps({'variant':name,'rounds_s':times,'min_s':min(times),
                          'tokens_s':x.size/min(times),'peak_gb':mx.get_peak_memory()/1e9}),flush=True)
    # Close the drift bracket with the same baseline graph and same weights.
    trainer._loss_and_grad=compiled['base16']
    for _ in range(args.warmup): step()
    times=[]
    for _ in range(args.iters):
        start=time.perf_counter(); step(); times.append(time.perf_counter()-start)
    print(json.dumps({'variant':'base16_repeat','rounds_s':times,'min_s':min(times),
                      'tokens_s':x.size/min(times)}),flush=True)


if __name__=='__main__':
    main()
