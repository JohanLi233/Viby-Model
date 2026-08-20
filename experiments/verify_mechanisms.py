"""加载完成的 checkpoint，验证 HRM / CycleDelta / CycleFiLM / Engram /
ValueResidual / AttnGate / MoE 是否真正接线并产生可观测贡献。

用法：
  .venv/bin/python experiments/verify_mechanisms.py \
      --checkpoint research_runs/r073_hrm_moe_cycledelta/pretrain_768.safetensors

检查项：
  1. HRM 循环的 KV cache 槽位数与 prefill==chunk-decode 一致性
  2. 机制消融：逐项关闭机制，看 logits / argmax 变化
  3. CycleDelta 路由审计：每 slot 的 delta/base 贡献、跨 cycle top6 重叠
  4. 各机制参数是否收到梯度（小批量）
  5. 关键参数统计（per-cycle V_c、FiLM 幅度等）
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.model import (  # noqa: E402
    MoEGate,
    VibyConfig,
    VibyForCausalLM,
    migrate_cycle_delta_weights,
)


def load_model(checkpoint):
    meta_path = Path(checkpoint).with_suffix(".json")
    meta = json.load(open(meta_path))
    cfg = VibyConfig.from_dict(meta["config"])
    model = VibyForCausalLM(cfg)
    shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
    weights = migrate_cycle_delta_weights(dict(mx.load(checkpoint).items()), shapes)
    weights = {k: v for k, v in weights.items() if k in shapes and v.shape == shapes[k]}
    model.load_weights(list(weights.items()))
    return model, cfg, meta


def _logits(model, X, attn):
    out = model(X, attention_mask=attn, mask_has_pad=False)
    mx.eval(out.logits)
    return out.logits


def _report_diff(name, base, new):
    base_np = np.asarray(base.astype(mx.float32))
    new_np = np.asarray(new.astype(mx.float32))
    d = np.abs(new_np - base_np)
    denom = max(float(np.sqrt((base_np * base_np).mean())), 1e-9)
    arg_change = float(
        (np.argmax(new_np, axis=-1) != np.argmax(base_np, axis=-1)).mean()
    )
    print(
        f"  {name:28s} maxdiff={float(d.max()):10.4g} "
        f"rel_rms={float(np.sqrt((d * d).mean()) / denom):8.4g} "
        f"argmax_change={arg_change:7.2%}"
    )


def check_cache(model, cfg, seq_len=64, cut=None):
    print("\n[1] HRM 循环 / KV cache")
    model.eval()  # 关键：train 模式下 router noise 会让两次前向不可比
    seq_len = min(seq_len, int(cfg.max_position_embeddings))
    cut = cut or seq_len // 2
    B = 1
    mx.random.seed(3)
    X = mx.random.randint(0, cfg.vocab_size, (B, seq_len)).astype(mx.int32)
    attn = mx.ones_like(X)
    full = model(X, attention_mask=attn, use_cache=True, mask_has_pad=False).logits
    first = model(
        X[:, :cut], attention_mask=attn[:, :cut], use_cache=True, mask_has_pad=False
    )
    second = model(
        X[:, cut:],
        attention_mask=attn,
        past_key_values=first.past_key_values,
        use_cache=True,
        mask_has_pad=False,
    )
    chunk = mx.concatenate([first.logits, second.logits], axis=1)
    mx.eval(full, chunk)
    diff = float(mx.max(mx.abs(full.astype(mx.float32) - chunk.astype(mx.float32))))
    expected_slots = cfg.hrm_H_cycles * (cfg.hrm_L_cycles + 1) * cfg.num_hidden_layers
    print(f"  cache slots: {len(first.past_key_values)} (expected {expected_slots})")
    print(f"  prefill vs chunk-decode maxdiff: {diff:.6g} (bf16 模型通常 <1)")


def check_ablation(model, cfg, seq_len=128):
    print("\n[2] 机制消融（关闭该机制后输出应显著变化）")
    B, T = 1, min(seq_len, int(cfg.max_position_embeddings))
    mx.random.seed(7)
    X = mx.random.randint(0, cfg.vocab_size, (B, T)).astype(mx.int32)
    attn = mx.ones_like(X)
    model.eval()
    base = _logits(model, X, attn)

    saved = [(g, g.cycle_v) for g in model.moe_gates()]
    for g, _ in saved:
        g.cycle_v = mx.zeros_like(g.cycle_v)
    _report_diff("CycleDelta (V=0)", base, _logits(model, X, attn))
    for g, v in saved:
        g.cycle_v = v

    if model.model.hrm_film_scale is not None:
        sf, sh = model.model.hrm_film_scale, model.model.hrm_film_shift
        model.model.hrm_film_scale = mx.zeros_like(sf)
        model.model.hrm_film_shift = mx.zeros_like(sh)
        _report_diff("CycleFiLM (off)", base, _logits(model, X, attn))
        model.model.hrm_film_scale, model.model.hrm_film_shift = sf, sh

    if model.model.engrams:
        vp = model.model.engrams[0].value_proj.weight
        model.model.engrams[0].value_proj.weight = mx.zeros_like(vp)
        _report_diff("Engram (value=0)", base, _logits(model, X, attn))
        model.model.engrams[0].value_proj.weight = vp

    gates = []
    for m in model.modules():
        gate = getattr(m, "attn_gate", None)
        if gate is not None:
            gates.append((gate, gate.weight, gate.bias))
            gate.weight = mx.zeros_like(gate.weight)
            gate.bias = mx.zeros_like(gate.bias)
    _report_diff("AttnGate (gate=0.5)", base, _logits(model, X, attn))
    for gate, w, b in gates:
        gate.weight, gate.bias = w, b

    vres = []
    for m in model.modules():
        lam = getattr(m, "v_res_lambda", None)
        if lam is not None:
            vres.append((m, lam))
            m.v_res_lambda = mx.array(-20.0)
    _report_diff("ValueResidual (lambda~0)", base, _logits(model, X, attn))
    for m, lam in vres:
        m.v_res_lambda = lam


def check_cycle_routing(model, cfg, seq_len=512):
    print("\n[3] CycleDelta 路由审计")
    B, T = 1, min(seq_len, int(cfg.max_position_embeddings))
    mx.random.seed(123)
    X = mx.random.randint(0, cfg.vocab_size, (B, T)).astype(mx.int32)
    Y = mx.random.randint(0, cfg.vocab_size, (B, T)).astype(mx.int32)
    model.eval()
    gates = model.moe_gates()
    gate_by_id = {id(g): i for i, g in enumerate(gates)}
    records = []
    orig = MoEGate.__call__

    def patched(self, x, step_idx=None, collect_aux=False):
        idx, w = orig(self, x, step_idx, collect_aux)
        records.append(
            (
                id(self),
                step_idx,
                x.reshape(-1, x.shape[-1]),
                idx.reshape(-1, idx.shape[-1]),
            )
        )
        return idx, w

    MoEGate.__call__ = patched
    out = model(
        X,
        labels=Y,
        loss_mask=mx.ones_like(Y),
        attention_mask=mx.ones_like(X),
        mask_has_pad=False,
    )
    mx.eval(out.loss)
    MoEGate.__call__ = orig

    print(f"  gate 调用序列: {[r[1] for r in records]}  (L=0,1,2,4,5,6; H=3,7; MTP=7)")
    calls = defaultdict(list)
    for gid, slot, x, idx in records:
        calls[gate_by_id[gid]].append((slot, x, idx))

    for gi in sorted(calls):
        print(f"  --- gate {gi} ---")
        slots = []
        for slot, x, idx in calls[gi]:
            mx.eval(x, idx)
            a = np.asarray(idx)
            counts = np.bincount(a.reshape(-1), minlength=cfg.n_routed_experts)
            p = counts / counts.sum()
            ent = -float((p * np.log(p + 1e-12)).sum()) / math.log(cfg.n_routed_experts)
            x32 = np.asarray(x.astype(mx.float32))
            common = np.linalg.norm(x32.mean(axis=0))
            residual = float(np.sqrt((np.square(x32 - x32.mean(axis=0))).mean()))
            slots.append((slot, a, counts))
            print(
                f"     slot {slot}: max/mean={counts.max() / counts.mean():5.2f} "
                f"entropy_norm={ent:.3f} router_x(common/residual)="
                f"{common:.2f}/{residual:.3f} (res/common={residual / max(common, 1e-6):.4f})"
            )
        for (s1, a1, _), (s2, a2, _) in zip(slots, slots[1:]):
            jac = np.mean(
                [
                    len(set(a1[i]) & set(a2[i])) / cfg.num_experts_per_tok
                    for i in range(a1.shape[0])
                ]
            )
            top1 = set(
                np.argsort(
                    -np.bincount(a1.reshape(-1), minlength=cfg.n_routed_experts)
                )[:6]
            )
            top2 = set(
                np.argsort(
                    -np.bincount(a2.reshape(-1), minlength=cfg.n_routed_experts)
                )[:6]
            )
            print(
                f"     slot{s1}->slot{s2}: token_top6_mean_overlap={jac:.3f} "
                f"aggregate_top6_jaccard={len(top1 & top2) / 6:.2f}"
            )

    print("  delta vs base 贡献：")
    for gi in sorted(calls):
        g = gates[gi]
        for slot, x, idx in calls[gi]:
            mx.eval(x)
            base = (x @ g.weight.T.astype(x.dtype)).astype(mx.float32)
            z = x @ g.cycle_v[slot].T.astype(x.dtype)
            delta = (z @ g.cycle_u.T.astype(x.dtype)).astype(mx.float32)
            mx.eval(base, delta)
            bias = g.expert_bias[slot] if g.expert_bias.ndim == 2 else g.expert_bias
            sel_base = mx.sigmoid(base) + bias.astype(mx.float32)
            base_idx = np.asarray(
                mx.argpartition(-sel_base, g.top_k - 1, axis=-1)[..., : g.top_k]
            )
            full_idx = np.asarray(idx)
            overlap = np.mean(
                [
                    len(set(base_idx[i]) & set(full_idx[i])) / g.top_k
                    for i in range(full_idx.shape[0])
                ]
            )
            print(
                f"     gate{gi} slot{slot}: delta_std/base_std="
                f"{float(delta.std() / max(base.std(), 1e-9)):.2f} "
                f"(delta={float(delta.std()):.3f}, base={float(base.std()):.3f}) "
                f"top6_overlap_full_vs_base={overlap:.3f}"
            )


def check_gradients(model, cfg, seq_len=16):
    print("\n[4] 梯度可达性（小批量 value_and_grad）")
    B, T = 1, seq_len
    mx.random.seed(0)
    X = mx.random.randint(0, cfg.vocab_size, (B, T)).astype(mx.int32)
    Y = mx.random.randint(0, cfg.vocab_size, (B, T)).astype(mx.int32)
    mask = mx.ones_like(Y)
    attn = mx.ones_like(X)
    model.train()
    params = model.trainable_parameters()

    def loss_fn(p):
        model.update(p)
        out = model(
            X, labels=Y, loss_mask=mask, attention_mask=attn, mask_has_pad=False
        )
        return out.loss

    val, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(val, grads)
    gflat = dict(tree_flatten(grads))
    print(f"  loss={float(val):.4f}")
    for key, v in gflat.items():
        if any(
            s in key
            for s in (
                "cycle_v",
                "cycle_u",
                "hrm_film",
                "engrams.0.table",
                "engrams.0.value_proj",
                "attn_gate",
                "v_res_lambda",
            )
        ):
            a = v.astype(mx.float32)
            mx.eval(a)
            print(
                f"     {key:55s} max={float(mx.abs(a).max()):.3g} "
                f"rms={float(mx.sqrt(mx.mean(a * a))):.3g}"
            )


def check_weights(model):
    print("\n[5] 关键参数统计")
    w = dict(tree_flatten(model.parameters()))
    for key in (
        "model.l_module.layers.0.mlp.router.cycle_v",
        "model.h_module.layers.0.mlp.router.cycle_v",
        "mtp_modules.0.block.mlp.router.cycle_v",
        "model.hrm_film_scale",
        "model.hrm_film_shift",
    ):
        if key not in w:
            continue
        a = w[key].astype(mx.float32)
        mx.eval(a)
        if a.ndim == 3 and "cycle_v" in key:
            row_norms = mx.sqrt((a * a).sum(-1)).mean(-1)
            mx.eval(row_norms)
            print(
                f"  {key}: per-cycle row_norm mean="
                f"{[round(float(x), 3) for x in row_norms]}"
            )
        else:
            print(
                f"  {key}: mean={float(a.mean()):.4g} std={float(a.std()):.4g} "
                f"absmax={float(mx.abs(a).max()):.4g}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="research_runs/r073_hrm_moe_cycledelta/pretrain_768.safetensors",
    )
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--skip_grad", action="store_true")
    args = ap.parse_args()

    model, cfg, meta = load_model(args.checkpoint)
    print(f"checkpoint step={meta.get('step')} config={Path(args.checkpoint).parent}")
    check_cache(model, cfg)
    check_ablation(model, cfg, seq_len=args.seq_len)
    check_cycle_routing(model, cfg, seq_len=args.seq_len)
    if not args.skip_grad:
        check_gradients(model, cfg, seq_len=16)
    check_weights(model)


if __name__ == "__main__":
    main()
