#!/usr/bin/env python3
"""Fixed three-arm binding-workspace experiment; serial GPU subprocesses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.binding_data import batch


def write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(path)


def status(root, phase, **kw):
    value=dict(pid=os.getpid(),phase=phase,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    write(Path(root)/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def load_data(root, split):
    return [json.loads(x) for x in (Path(root)/'data'/f'{split}.jsonl').read_text().splitlines()]


class BaseView(dict):
    """Only the original parameter names reach the restored backbone optimizer."""
    def __init__(self, model):
        self.owner,self.config=model,model.config
        super().__init__(self.trainable_parameters())
    def trainable_parameters(self):
        return {k:v for k,v in self.owner.trainable_parameters().items() if k!='binding'}
    def parameters(self):
        return {k:v for k,v in self.owner.parameters().items() if k!='binding'}
    def update(self,p):
        self.owner.update(p)


def load_system(args, with_binding):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from model.config import VibyConfig
    from model.model import VibyForCausalLM
    from model.binding_workspace import BindingConfig,BindingWorkspace
    from trainer.muon import create_mixed_optimizer
    from trainer.utils import convert_model_dtype,load_checkpoint,get_optimizer_steps
    mx.set_cache_limit(1<<30)
    source=json.loads(Path(args.checkpoint).with_suffix('.json').read_text())
    old_args=SimpleNamespace(**source['args'])
    old_args.reset_optimizer=False
    model=convert_model_dtype(VibyForCausalLM(VibyConfig.from_dict(source['config']),skip_init=True),source['args']['dtype'])
    optimizer=create_mixed_optimizer(model,old_args)
    load_checkpoint(args.checkpoint,model,optimizer,old_args)
    actual=[[] for _ in optimizer.filters]
    for key,value in tree_flatten(model.trainable_parameters()):
        for i,fn in enumerate(optimizer.filters):
            if fn(key,value):actual[i].append(key);break
    if actual!=source['optimizer_parameter_groups']:
        raise ValueError('backbone optimizer parameter grouping differs from saved checkpoint')
    provenance=dict(checkpoint=args.checkpoint,source_step=source['step'],source_optimizer_steps=get_optimizer_steps(optimizer),
                    groups_match=True,learning_rates=[float(o.learning_rate) for o in optimizer.optimizers],
                    policy='Clone saved original optimizer state, radii, clocks and current group LRs. New synthetic sample cursor starts at zero. DPR/PSR/recurrent/MTP disabled at runtime; original dormant DPR parameters retained in original optimizer grouping.')
    side=None
    if with_binding:
        rng=list(mx.random.state)[0].tolist()
        model.binding=BindingWorkspace(BindingConfig(dim=model.config.dim,seed=args.seed),'fixed' if args.arm=='fixed' else 'full')
        mx.random.seed((int(rng[0])<<32)|int(rng[1]))
        side=optim.AdamW(learning_rate=.001,betas=(.9,.999),weight_decay=0)
        side.init(model.binding.trainable_parameters())
    model.train();mx.eval(model.parameters(),optimizer.state)
    if side is not None:mx.eval(side.state)
    return model,optimizer,side,provenance


def make_gradient(model, use_binding, compiled=True):
    import mlx.core as mx
    def loss(params,bias,x,y,mask,pad):
        model.update(params);model.apply_moe_biases(bias)
        out=model(x,labels=y,loss_mask=mask,attention_mask=pad,use_dpr=False,use_ced_recurrent=False,use_mtp=False,psr_mode='off',use_binding=use_binding)
        return out.loss, (out.lm_loss,out.moe_loads,out.moe_qb_margins)
    fn=mx.value_and_grad(loss)
    return mx.compile(fn) if compiled else fn


def update_window(model,optimizer,side,fn,worlds,window,*,update=True):
    import mlx.core as mx
    from mlx.utils import tree_flatten,tree_map
    from trainer.fast_norm import gradient_square_sum
    total=sum(w['end']-w['begin'] for w in window)
    grads_sum=None;loads=None;qb=[];ce=0.;objective=0.;physical=0
    for item in window:
        arrays=[mx.array(v) for v in batch(worlds[item['world']],item['begin'],item['end'],model.config.pad_token_id)]
        x,y,mask,pad=arrays;physical+=x.size
        # One physical call per document fragment. QB samples stay bounded by
        # these tiny contexts; original cumulative-window update happens once.
        params=model.trainable_parameters();bias=model.moe_bias_stack()
        (loss,(lm,count,margins)),grads=fn(params,bias,x,y,mask,pad)
        mx.eval(loss,lm,count,margins,grads)
        model.update(params);model.apply_moe_biases(bias)
        weight=(item['end']-item['begin'])/total
        scaled=tree_map(lambda v:v*weight,grads)
        grads_sum=scaled if grads_sum is None else tree_map(lambda a,b:a+b,grads_sum,scaled)
        loads=count if loads is None else loads+count
        qb.append(margins);ce+=float(lm)*weight;objective+=float(loss)*weight
        mx.eval(grads_sum,loads)
        del grads,scaled
    norm=mx.sqrt(sum(gradient_square_sum(v) for _,v in tree_flatten(grads_sum)))
    mx.eval(norm)
    if not np.isfinite(float(norm)) or not np.isfinite(objective):
        raise FloatingPointError('nonfinite objective/gradient; no update applied')
    if update:
        base_grads={k:v for k,v in grads_sum.items() if k!='binding'}
        optimizer.update(BaseView(model),base_grads)
        if side is not None and 'binding' in grads_sum:
            side.update(model.binding,grads_sum['binding'])
        model.update_moe_biases(loads,qb_margins=mx.concatenate(qb,axis=1))
        mx.eval(model.parameters(),optimizer.state)
        if side is not None:mx.eval(side.state)
    return dict(ce=ce,total_loss=objective,grad_norm=float(norm),valid_tokens=total,physical_tokens=physical)


def benchmark(args):
    import mlx.core as mx
    from experiments.kernel_bench_utils import snapshot_train_state,restore_train_state,abba_blocks
    model,opt,side,provenance=load_system(args,True)
    opt.psr_optimizer=side  # reuse tested snapshot/restore for independent state
    world=load_data(args.run_dir,'train')
    window=json.loads((Path(args.run_dir)/'data/plan.json').read_text())[0]
    base_fn=make_gradient(model,False,not args.eager)
    full_fn=make_gradient(model,True,not args.eager)
    snap=snapshot_train_state(model,opt)
    def restore():restore_train_state(model,opt,snap)
    def a():return update_window(model,opt,None,base_fn,world,window)
    def b():return update_window(model,opt,side,full_fn,world,window)
    result=abba_blocks(a,b,warmup=3,block_iters=5,n_blocks=3,before_a=restore,before_b=restore,label_a='CED',label_b='binding')
    result['provenance']=provenance
    result['pass']=not result['inconclusive_aa_drift'] and result['paired_block_median_B_over_A']<=1.2
    result['peak_gib']=max(s['memory']['peak_gib'] for block in result['blocks'] for s in block['slots'])
    result['pass']=result['pass'] and result['peak_gib']<40
    write(Path(args.run_dir)/'benchmark.json',result)
    status(args.run_dir,'benchmark_complete',passed=result['pass'],ratio=result['paired_block_median_B_over_A'],peak_gib=result['peak_gib'],aa_drift=result['aa_end_over_start_abs_rel'])


def evaluate(model,worlds,*,intervention=False,entity_ids=None):
    import mlx.core as mx
    import mlx.nn as nn
    rows=[];model.eval()
    for world in worlds:
        x,y,mask,pad=[mx.array(v) for v in batch(world,pad_id=model.config.pad_token_id)]
        out=model(x,attention_mask=pad,use_dpr=False,use_ced_recurrent=False,use_mtp=False,psr_mode='off')
        positions=mx.array(world['answers'])
        logits=out.logits[0,positions].astype(mx.float32)
        target=y[0,positions]
        nll=nn.losses.cross_entropy(logits,target,reduction='none');correct=mx.argmax(logits,-1)==target
        mx.eval(nll,correct)
        row=dict(world_id=world['world_id'],answer_nll=float(nll.sum()),correct=int(correct.sum()),answers=len(world['answers']))
        if intervention:
            # Only earlier questions in the same unchanged fact phase can be
            # donors: no future donor state enters this causal intervention.
            q1=out.binding_trace[0]
            override=q1
            targets=[];selected=[]
            for i,q in enumerate(world['questions']):
                donors=[p for p in world['questions'][:i] if p['phase']==q['phase'] and p['operations'][-1]==q['operations'][-1] and p['intermediate']!=q['intermediate']]
                if not donors:continue
                donor=donors[-1]
                override=override.at[0,q['position']].add(q1[0,donor['position']]-q1[0,q['position']])
                selected.append(q['position']);targets.append(q['second_map'][donor['intermediate']])
            if selected:
                changed=model(x,attention_mask=pad,use_dpr=False,use_ced_recurrent=False,use_mtp=False,psr_mode='off',binding_options=dict(address_override=override))
                if entity_ids is None:raise ValueError('intervention requires manifest entity ids')
                token_targets=mx.array([entity_ids[i] for i in targets])
                altered=changed.logits[0,mx.array(selected)].astype(mx.float32)
                original=out.logits[0,mx.array(selected)].astype(mx.float32)
                gain=nn.losses.cross_entropy(original,token_targets,reduction='none')-nn.losses.cross_entropy(altered,token_targets,reduction='none')
                mx.eval(gain)
                row.update(donor_answers=len(selected),donor_successor_logprob_gain=float(gain.sum()))
        rows.append(row)
    model.train()
    n=sum(r['answers'] for r in rows)
    return dict(answer_nll=sum(r['answer_nll'] for r in rows)/n,accuracy=sum(r['correct'] for r in rows)/n,answers=n,worlds=rows)


def save_model(model,directory,label):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    start=time.perf_counter()
    mx.save_safetensors(str(directory/f'{label}.safetensors'),dict(tree_flatten(model.parameters())))
    return time.perf_counter()-start


def flops_per_world(model,world,with_binding):
    from trainer.flops import gemm_active_params,attn_fwdbwd_flops_per_token
    t=batch(world)[0].shape[1]
    baseline=(6*gemm_active_params(BaseView(model))+attn_fwdbwd_flops_per_token(model.config,t))*t
    if not with_binding:return baseline
    c=model.binding.config;d,r,h=c.dim,c.rank,c.banks
    def combines(n):
        if n==1:return 0
        if n==2:return 1
        m=(n+1)//2
        return m+combines(m)+m-1
    # Dense GEMM accounting for explicit forward + analytic reverse scans.
    # Excludes scalar kernels, index search, optimizer, allocator and I/O.
    projections=t*(6*(3*d*r+3*h*d+2*h*r)+8*d*r)
    scans=8*h*r**3*combines(t)+2*h*r**3*(t+1)
    matvecs=t*32*h*r*r
    return baseline+projections+scans+matvecs


def train(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from experiments.kernel_bench_utils import clone_tree
    root=Path(args.run_dir);directory=root/args.arm;directory.mkdir(exist_ok=False)
    model,opt,side,provenance=load_system(args,args.arm!='baseline')
    write(directory/'initialization.json',provenance)
    if side is not None:model.binding.save_weights(str(directory/'initial_binding.safetensors'))
    fn=make_gradient(model,args.arm!='baseline',not args.eager)
    worlds=load_data(root,'train');dev=load_data(root,'development');confirm=load_data(root,'confirmation')
    plan=json.loads((root/'data/plan.json').read_text())
    write(directory/'eval_000.json',evaluate(model,dev))
    baseline=None if args.arm=='baseline' else json.loads((root/'baseline/training.json').read_text())
    costs=[sum(flops_per_world(model,worlds[x['world']],args.arm!='baseline') for x in w) for w in plan]
    prefix_f=None
    if baseline:
        candidates=np.flatnonzero(np.cumsum(costs)<=baseline['nominal_flops'])
        prefix_f=int(candidates[-1]+1) if len(candidates) else 0
    elapsed=0.;flops=0;tokens=0;physical=0;save_seconds=0.;time_prefix_saved=False;previous=None
    mx.reset_peak_memory()
    with (directory/'metrics.jsonl').open('w',buffering=1) as log:
        for step,window in enumerate(plan[:args.steps],1):
            # Keep at most one previous model tree near the time threshold.
            # Arrays stay at their stored dtype; no FP32 parameter snapshots.
            previous=clone_tree(model.parameters()) if baseline and not time_prefix_saved and elapsed>=baseline['train_seconds']*.85 else None
            before=time.perf_counter();metrics=update_window(model,opt,side,fn,worlds,window)
            seconds=time.perf_counter()-before
            old_elapsed=elapsed;elapsed+=seconds;flops+=costs[step-1];tokens+=metrics['valid_tokens'];physical+=metrics['physical_tokens']
            row=dict(step=step,**metrics,consumed_tokens=tokens,seconds=seconds,train_seconds=elapsed,nominal_flops=flops,peak_gib=mx.get_peak_memory()/2**30)
            log.write(json.dumps(row)+'\n')
            if step<=2 or step%8==0:status(root,'training',arm=args.arm,**row)
            if row['peak_gib']>=40:raise RuntimeError('40 GiB memory kill line')
            if baseline and elapsed>1.2*baseline['train_seconds'] and step==256:
                raise RuntimeError('20% measured complete-training overhead kill line')
            if elapsed>2400:raise RuntimeError('40 minute per-arm safety budget')
            if baseline and not time_prefix_saved and elapsed>baseline['train_seconds']:
                if previous is None:raise RuntimeError('no safe previous time-prefix checkpoint')
                mx.save_safetensors(str(directory/'matched_time.safetensors'),dict(tree_flatten(previous)))
                write(directory/'matched_time.json',dict(step=step-1,tokens=tokens-2048,train_seconds=old_elapsed,budget=baseline['train_seconds']))
                time_prefix_saved=True
            previous=None
            if step==prefix_f:
                save_seconds+=save_model(model,directory,'matched_flops')
                write(directory/'matched_flops.json',dict(step=step,tokens=tokens,nominal_flops=flops,budget=baseline['nominal_flops'],scope='nominal GEMM/attention work, not measured total FLOPs'))
            if step in (128,256) or step==args.steps:
                save_seconds+=save_model(model,directory,f'step_{step:03d}')
                if step in (128,256):write(directory/f'eval_{step:03d}.json',evaluate(model,dev))
        if baseline and not time_prefix_saved:
            write(directory/'matched_time.json',dict(step=args.steps,tokens=tokens,train_seconds=elapsed,budget=baseline['train_seconds'],checkpoint=f'step_{args.steps:03d}.safetensors'))
    mx.save_safetensors(str(directory/'optimizer.safetensors'),dict(tree_flatten(opt.state)))
    if side is not None:mx.save_safetensors(str(directory/'binding_optimizer.safetensors'),dict(tree_flatten(side.state)))
    write(directory/'training.json',dict(completed=True,steps=args.steps,valid_tokens=tokens,physical_tokens=physical,train_seconds=elapsed,checkpoint_seconds=save_seconds,nominal_flops=flops,peak_gib=mx.get_peak_memory()/2**30,base_config=model.config.to_dict(),binding_config=None if side is None else model.binding.config.to_dict(),provenance=provenance))
    if args.steps==256:
        entity_ids=[p[1] for p in json.loads((root/'data/manifest.json').read_text())['symbols']]
        write(directory/'confirmation.json',evaluate(model,confirm,intervention=args.arm=='full',entity_ids=entity_ids))
        for label in ('matched_flops','matched_time'):
            path=directory/f'{label}.safetensors'
            if path.exists():
                model.load_weights(str(path),strict=True)
                write(directory/f'{label}_confirmation.json',evaluate(model,confirm))
    status(root,'arm_complete',arm=args.arm,steps=args.steps,valid_tokens=tokens,train_seconds=elapsed)


def summarize(root):
    root=Path(root)
    data={a:json.loads((root/a/'confirmation.json').read_text()) for a in ('baseline','full','fixed')}
    result=dict(scope='single-seed synthetic relation composition; not natural-language efficiency',metrics={a:{k:v for k,v in d.items() if k!='worlds'} for a,d in data.items()},comparisons={})
    rng=np.random.default_rng(20260914)
    full=data['full']['worlds']
    for arm in ('baseline','fixed'):
        other=data[arm]['worlds'];pairs=np.array([[a['answer_nll']-b['answer_nll'],a['correct']-b['correct'],a['answers']] for a,b in zip(full,other)])
        means=pairs.sum(0);samples=[]
        for _ in range(2000):
            s=pairs[rng.integers(0,len(pairs),len(pairs))].sum(0);samples.append(s[:2]/s[2])
        result['comparisons'][arm]=dict(nll_delta=float(means[0]/means[2]),accuracy_delta=float(means[1]/means[2]),paired_world_95ci=np.quantile(samples,[.025,.975],axis=0).T.tolist())
    result['mechanism_gate_pass']=all(v['nll_delta']<=-.05 and v['accuracy_delta']>=.05 for v in result['comparisons'].values())
    result['decision']='mechanism_pass_only_no_natural_language_claim' if result['mechanism_gate_pass'] else 'stop_this_variant_no_automatic_extension'
    write(root/'results.json',result);status(root,'completed',**result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--phase',choices=['all','benchmark','train','summary'],default='all')
    p.add_argument('--arm',choices=['baseline','full','fixed'],default='baseline')
    p.add_argument('--seed',type=int,default=20260914);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--eager',action='store_true');args=p.parse_args();root=Path(args.run_dir)
    root.mkdir(exist_ok=True,parents=True)
    try:
        if args.phase=='benchmark':benchmark(args)
        elif args.phase=='train':train(args)
        elif args.phase=='summary':summarize(root)
        else:
            command=[sys.executable,'-u',__file__,'--run-dir',str(root),'--checkpoint',args.checkpoint,'--seed',str(args.seed),'--steps',str(args.steps)]
            if args.eager:command+=['--eager']
            status(root,'benchmark_start')
            if not (root/'benchmark.json').exists():subprocess.run(command+['--phase','benchmark'],check=True)
            if not json.loads((root/'benchmark.json').read_text())['pass']:
                status(root,'stopped_resource_gate',reason='benchmark overhead, drift or memory failed');return
            for arm in ('baseline','full','fixed'):
                status(root,'arm_start',arm=arm)
                subprocess.run(command+['--phase','train','--arm',arm],check=True)
            if args.steps==256:summarize(root)
            else:status(root,'smoke_completed',steps=args.steps)
    except BaseException as error:
        status(root,'failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
