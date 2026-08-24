"""MuonH 刷新频率质量 A/B：同一初始化、同一微批序列。

  stale_d_e8    当前默认：每 8 步刷新 NS5，中间复用旧极因子
  stale_d_e16   候选：每 16 步刷新 NS5，中间复用旧极因子

语料：一阶马尔可夫链（可学结构，不是纯随机 token）。
指标：train CE + 固定 held-out eval CE + 参数范数漂移 + 正交残差 + 专家负载。

用法: .venv/bin/python experiments/ab_muonh_quality.py [steps]
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer

VOCAB = 256
SEQ = 128
BS = 8


def snapshot(params):
    flat = tree_flatten(params)
    return tree_unflatten([(k, mx.array(v)) for k, v in flat])


def make_cfg():
    return VibyConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=64,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        mtp_depth=0,
        n_routed_experts=32,
        num_experts_per_tok=4,
        n_shared_experts=1,
        moe_intermediate_size=64,
        moe_latent_dim=128,
        tie_word_embeddings=False,
        moe_router_noise=0.0,
        dropout=0.0,
    )


def markov_table(vocab, seed=0):
    rng = np.random.default_rng(seed)
    # 每个 token：0.55 到 next、0.20 自环、其余摊到 8 个近邻
    trans = np.full((vocab, vocab), 1e-6, dtype=np.float64)
    for i in range(vocab):
        trans[i, (i + 1) % vocab] += 0.55
        trans[i, i] += 0.20
        nbr = rng.choice(vocab, size=8, replace=False)
        trans[i, nbr] += 0.25 / 8
        trans[i] /= trans[i].sum()
    return trans


def sample_batch(trans, rng, bs=BS, sl=SEQ):
    v = trans.shape[0]
    x = np.empty((bs, sl), dtype=np.int32)
    x[:, 0] = rng.integers(0, v, size=bs)
    for t in range(1, sl):
        prev = x[:, t - 1]
        # 逐行抽样（V=256、bs=8 可忽略）
        x[:, t] = [rng.choice(v, p=trans[p]) for p in prev]
    return mx.array(x)


def eval_ce(model, batches):
    model.eval()
    tot = 0.0
    for ids in batches:
        loss = model(ids, labels=ids).loss
        mx.eval(loss)
        tot += float(loss.item())
    model.train()
    return tot / len(batches)


def run_arm(init_params, train_batches, eval_batches, muonh, env, lr=0.01):
    for k, v in list(os.environ.items()):
        if k.startswith("VIBY_MUONH_"):
            del os.environ[k]
    for k, v in env.items():
        os.environ[k] = str(v)

    cfg = make_cfg()
    model = VibyForCausalLM(cfg)
    model.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            model.parameters(),
        )
    )
    model.update(init_params)
    model.train()
    for gate in model.moe_gates():
        gate.collect_stats = True
    mx.eval(model.parameters())
    mx.random.seed(42)
    args = types.SimpleNamespace(
        learning_rate=lr,
        muon_ns_steps=5,
        muonh=muonh,
        router_lr_mult=0.05,
    )
    opt = create_mixed_optimizer(model, args)

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    train_losses = []
    evals = []
    for step, ids in enumerate(train_batches):
        val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        mx.eval(val, grads)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        train_losses.append(float(val.item()))
        if (step + 1) % 16 == 0 or step == 0 or step + 1 == len(train_batches):
            evals.append((step + 1, eval_ce(model, eval_batches)))

    flat0 = dict(tree_flatten(init_params))
    flat1 = dict(tree_flatten(model.parameters()))
    fro_drift = 0.0
    for path, p0 in flat0.items():
        if ".experts." not in path or path not in flat1 or p0.ndim < 3:
            continue
        p1 = flat1[path]
        n0 = mx.linalg.norm(p0.astype(mx.float32), axis=(-2, -1))
        n1 = mx.linalg.norm(p1.astype(mx.float32), axis=(-2, -1))
        drift = mx.max(mx.abs(n1 - n0) / mx.maximum(n0, 1e-12))
        fro_drift = max(fro_drift, float(drift))

    muon = next((o for o in opt.optimizers if hasattr(o, "_stack_ns_cache")), None)
    orth_residual = 0.0
    if muon is not None:
        for cache in muon._stack_ns_cache.values():
            if len(cache) != 2:
                continue
            direction = cache[1].reshape(-1, *cache[1].shape[-2:])
            orth_residual = max(
                orth_residual, float(muon._orth_residual(direction, probe=32))
            )
    loads = model.moe_load_stats()
    if loads is None or loads.size == 0:
        load_ratio = 0.0
    else:
        mx.eval(loads)
        load_ratio = float(mx.max(loads)) / max(float(mx.mean(loads)), 1.0)
    return {
        "train": train_losses,
        "eval": evals,
        "fro_drift": fro_drift,
        "orth_residual": orth_residual,
        "load_ratio": load_ratio,
    }


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    mx.random.seed(0)
    cfg = make_cfg()
    model0 = VibyForCausalLM(cfg)
    model0.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16)
            if mx.issubdtype(a.dtype, mx.floating)
            else a,
            model0.parameters(),
        )
    )
    mx.eval(model0.parameters())
    init = snapshot(model0.parameters())

    trans = markov_table(VOCAB, seed=0)
    train_batches = [
        sample_batch(trans, np.random.default_rng(1000 + i)) for i in range(steps)
    ]
    eval_batches = [
        sample_batch(trans, np.random.default_rng(9000 + i)) for i in range(8)
    ]
    mx.eval(*train_batches, *eval_batches)

    arms = [
        (
            "stale_d_e8",
            True,
            {
                "VIBY_MUONH_STACK_NS_EVERY": "8",
                "VIBY_MUONH_CACHE_Q": "0",
                "VIBY_MUONH_EXPERTS": "1",
            },
        ),
        (
            "stale_d_e16",
            True,
            {
                "VIBY_MUONH_STACK_NS_EVERY": "16",
                "VIBY_MUONH_CACHE_Q": "0",
                "VIBY_MUONH_EXPERTS": "1",
            },
        ),
    ]

    results = {}
    for name, muonh, env in arms:
        print(f"running {name} ...", flush=True)
        results[name] = run_arm(snapshot(init), train_batches, eval_batches, muonh, env)

    def mean(xs):
        return sum(xs) / len(xs)

    print(f"\n{'step':>6}", end="")
    for name, *_ in arms:
        print(f"{name:>12}", end="")
    print(f"{'e16-e8':>10}")
    ref_tr = results["stale_d_e8"]["train"]
    for i in range(steps):
        if i + 1 in (1, 8, 16, 32, 64, 96, steps) or (i + 1) % 32 == 0:
            print(f"{i + 1:6d}", end="")
            for name, *_ in arms:
                print(f"{results[name]['train'][i]:12.4f}", end="")
            print(f"{results['stale_d_e16']['train'][i] - ref_tr[i]:+10.4f}")

    print("\nheld-out eval CE")
    print(f"{'step':>6}", end="")
    for name, *_ in arms:
        print(f"{name:>12}", end="")
    print(f"{'e16-e8':>10}")
    eval_map = {name: {s: v for s, v in results[name]["eval"]} for name, *_ in arms}
    steps_e = [s for s, _ in results["stale_d_e8"]["eval"]]
    for s in steps_e:
        print(f"{s:6d}", end="")
        for name, *_ in arms:
            print(f"{eval_map[name][s]:12.4f}", end="")
        print(f"{eval_map['stale_d_e16'][s] - eval_map['stale_d_e8'][s]:+10.4f}")

    print("\n末 16 步 train / 最终 eval")
    r = mean(ref_tr[-16:])
    re = eval_map["stale_d_e8"][steps_e[-1]]
    for name, *_ in arms:
        m = mean(results[name]["train"][-16:])
        e = eval_map[name][steps_e[-1]]
        print(
            f"  {name:12s} train {m:.4f} Δ={m - r:+.4f} ({(m - r) / r * 100:+.2f}%)"
            f"  eval {e:.4f} Δ={e - re:+.4f} ({(e - re) / re * 100:+.2f}%)"
        )

    # 下降段：e8 train CE 仍 >0.05，相对百分比才有意义。
    idx = [i for i, v in enumerate(ref_tr) if v > 0.05]
    print(f"\n下降段 (e8 train>0.05, {len(idx)} steps) 相对 e8")
    for name, *_ in arms:
        if name == "stale_d_e8":
            continue
        ds = [results[name]["train"][i] - ref_tr[i] for i in idx]
        rs = [(results[name]["train"][i] - ref_tr[i]) / ref_tr[i] * 100 for i in idx]
        print(
            f"  {name:12s} meanΔ={mean(ds):+.4f}  "
            f"mean%={mean(rs):+.2f}%  max|Δ|={max(abs(x) for x in ds):.4f}"
        )

    print("\n质量门禁")
    for name, *_ in arms:
        r = results[name]
        print(
            f"  {name:12s} fro漂移={r['fro_drift']:.3e} "
            f"orth={r['orth_residual']:.3e} load max/mean={r['load_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
