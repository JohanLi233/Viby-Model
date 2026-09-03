"""E2：复用谱测量——真实模型 KDA 层里「读而不写」的检索质量上界。

整条「记忆的价格」弧线的基石测量：touch/replay 类机制（以及等边际分配律
的 R*）的收益上界 = 查询能量落在「曾被写入、之后只被读未被复写」的方向
上的比例。本脚本在真实 checkpoint 的长上下文前向中直接测它。

方法：monkey-patch model.kda._chunk_kda，捕获每个 KDA 层进入 scan 的
q/k（post-conv、RMS 归一后，(B,H,T,D)，k 已单位范数）。逐头算：

  1. 复用曲线：对每个查询 t，在过去 key（lag≥16）中按 lag 分桶取
     max cos(q_t, k_s)，减去**跨序列零模型**（同样候选数、不同文档的
     key）——超出零模型的部分才是真实复用。
  2. 只读复用质量：top-match 对齐超过零模型 +0.05 的查询中，目标 key
     在 (s*, t) 内**未被复写**（无 cos(k_s', k_s*)>τ 的后续 key，
     τ=0.7，附 0.5/0.9 敏感性）的比例。
  3. k 侧自复写率：每个已写 key 之后被相似 key 复现的比例
     （数据流自己提供 LRU 的那部分——rewrite 模式的天然发生率）。

用法:
  .venv/bin/python experiments/reuse_spectrum.py \
      --run_dir research_runs/r084_gqa_qb_e256k8w384 --t 4096 --n_seq 4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

import model.kda as kda_mod
from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.utils import load_model_weights

LAG_BINS = [(16, 64), (64, 256), (256, 1024), (1024, 10**9)]
REWRITE_TAUS = (0.5, 0.7, 0.9)

_captures = []


def _make_capture_wrapper():
    orig = kda_mod._chunk_kda

    def wrapper(q, k, v, log_g, beta, *a, **kw):
        _captures.append((np.array(q[0]), np.array(k[0])))  # (H,T,D) f32
        return orig(q, k, v, log_g, beta, *a, **kw)

    return wrapper


def load_checkpoint_model(run_dir):
    ckp = os.path.join(run_dir, "pretrain_768.safetensors")
    sidecar = os.path.splitext(ckp)[0] + ".json"
    with open(sidecar, encoding="utf-8") as f:
        meta = json.load(f)
    config = VibyConfig.from_dict(meta["config"])
    model = VibyForCausalLM(config)
    assert load_model_weights(model, ckp, strict=True, label="checkpoint")
    model.eval()
    print(f"[checkpoint] {ckp} (step={meta.get('step')}) 已加载")
    return model


def iter_sequences(data_path, tokenizer, T, n_seq, mode, seed):
    """packed：多文档拼接（训练口径）；single：最长单文档截断。"""
    import random

    docs = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            try:
                docs.append(json.loads(line)["text"])
            except (KeyError, json.JSONDecodeError):
                continue
    random.Random(seed).shuffle(docs)
    seqs = []
    if mode == "single":
        docs.sort(key=len, reverse=True)
        for d in docs[: n_seq * 4]:
            ids = tokenizer(d, add_special_tokens=False)["input_ids"]
            if len(ids) >= T:
                seqs.append(ids[:T])
            if len(seqs) >= n_seq:
                break
    else:
        buf = []
        for d in docs:
            ids = tokenizer(d, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= T:
                seqs.append(buf[:T])
                buf = buf[T:]
                if len(seqs) >= n_seq:
                    break
            if len(seqs) >= n_seq:
                break
    print(f"[data] mode={mode} 得到 {len(seqs)} 条 {T}-token 序列")
    return seqs


def analyze(q, k, k_null):
    """单层单序列分析。q/k/k_null: (H,T,D)。返回 per-head 指标 dict 列表。"""
    H, T, D = q.shape
    qn = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    kn = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)
    kn_null = k_null / (np.linalg.norm(k_null, axis=-1, keepdims=True) + 1e-12)
    C_all = qn @ kn.transpose(0, 2, 1)  # (H,T,T) cos(q_t, k_s)
    Cn_all = qn @ kn_null.transpose(0, 2, 1)
    KK_all = kn @ kn.transpose(0, 2, 1)  # (H,T,T) cos(k_s, k_s')
    ar = np.arange(T)
    age = ar[:, None] - ar[None, :]  # age[t,s] = t - s
    out = []
    for h in range(H):
        C, Cn, KK = C_all[h], Cn_all[h], KK_all[h]
        rec = {"lag_reuse": [], "readonly_mass": {}}
        for lo, hi in LAG_BINS:
            m = (age >= lo) & (age < hi)
            a = np.where(m, C, -np.inf).max(axis=1)
            an = np.where(m, Cn, -np.inf).max(axis=1)
            valid = np.isfinite(a) & np.isfinite(an)
            rec["lag_reuse"].append(
                (lo, hi, float((a[valid] - an[valid]).mean()), int(valid.sum()))
            )
        # 只读复用质量与复写率（lag≥16 的 top-match）
        m16 = age >= 16
        m_c = np.where(m16, C, -np.inf)
        a = m_c.max(axis=1)
        an = np.where(m16, Cn, -np.inf).max(axis=1)
        s_star = m_c.argmax(axis=1)
        hit = (a - an) > 0.05
        idx = np.where(hit)[0]
        for tau in REWRITE_TAUS:
            rw = np.zeros(len(idx), dtype=bool)
            for j, t in enumerate(idx):
                s0 = s_star[t]
                if s0 + 1 < t:
                    rw[j] = bool((KK[s0, s0 + 1 : t] > tau).any())
            rec["readonly_mass"][tau] = (
                int(hit.sum()),
                float((~rw).mean()) if len(idx) else 0.0,
            )
        # k 侧自复现：key s 之后（s' − s ≥ 16）最相似 key 的 cos
        KKt = KK.copy()
        KKt[np.tril_indices(T, 15)] = -np.inf
        later = KKt.max(axis=1)
        rec["self_rewrite"] = np.where(np.isfinite(later), later, 0.0)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="research_runs/r084_gqa_qb_e256k8w384")
    ap.add_argument(
        "--data_path", default="/Volumes/pan/text/pretrain_t2t_mini_dedup.jsonl"
    )
    ap.add_argument("--t", type=int, default=4096)
    ap.add_argument("--n_seq", type=int, default=4)
    ap.add_argument("--mode", default="packed", choices=["packed", "single"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./model/")
    model = load_checkpoint_model(args.run_dir)
    # 测量需要超长前向：全模型无 RoPE/滑窗（KDA 递归 + NoPE GQA），
    # max_position_embeddings 只是护栏，安全放宽到 T。
    for cfg_owner in (model, getattr(model, "model", None)):
        if cfg_owner is not None and hasattr(cfg_owner, "config"):
            cfg_owner.config.max_position_embeddings = max(
                cfg_owner.config.max_position_embeddings, args.t
            )
    kda_mod._chunk_kda = _make_capture_wrapper()

    seqs = iter_sequences(
        args.data_path, tokenizer, args.t, args.n_seq, args.mode, args.seed
    )
    # 前 N-1 条作测量，序列 si+1 的 key 作序列 si 的跨序列零模型
    all_caps, layers = [], None
    for ids in seqs:
        _captures.clear()
        x = mx.array(ids)[None, :]
        out = model(x)
        mx.eval(out.logits if hasattr(out, "logits") else out[0])
        if layers is None:
            layers = len(_captures)
            print(
                f"[capture] {layers} 个 KDA 层，每层 q/k 形状 {_captures[0][0].shape}"
            )
        all_caps.append(list(_captures))
    _captures.clear()
    H = all_caps[0][0][0].shape[0]

    print(f"\n=== E2 复用谱（T={args.t}, n_seq={len(all_caps)}, mode={args.mode}）===")
    n_meas = len(all_caps) - 1
    for li in range(layers):
        agg_lag = np.zeros((len(LAG_BINS),))
        agg_n = np.zeros((len(LAG_BINS),))
        ro = {tau: [] for tau in REWRITE_TAUS}
        self_rw = []
        for si in range(n_meas):
            q, k = all_caps[si][li]
            _, k_null = all_caps[si + 1][li]
            recs = analyze(q, k, k_null)
            for rec in recs:
                for bi, (lo, hi, exc, nv) in enumerate(rec["lag_reuse"]):
                    agg_lag[bi] += exc * nv
                    agg_n[bi] += nv
                for tau in REWRITE_TAUS:
                    ro[tau].append(rec["readonly_mass"][tau])
                self_rw.append(rec["self_rewrite"])
        print(f"\n[KDA 层 {li}]（heads  pooled）")
        print("  复用曲线（excess max-cos vs lag，已减零模型）:")
        for bi, (lo, hi) in enumerate(LAG_BINS):
            exc = agg_lag[bi] / max(agg_n[bi], 1)
            print(f"    lag {lo:>5}-{min(hi, 10**5):>6}: excess = {exc:+.4f}")
        for tau in REWRITE_TAUS:
            hits = sum(r[0] for r in ro[tau])
            frac = np.mean([r[1] for r in ro[tau]]) if hits else 0.0
            print(
                f"  只读复用（τ={tau}）: 超零模型查询占比 "
                f"{hits / (n_meas * H * args.t):.3f}，其中目标未被复写比例 {frac:.3f}"
            )
        sr = np.concatenate(self_rw)
        print(
            f"  k 侧自复现率（max later cos 分位）: "
            f"p50={np.percentile(sr, 50):.3f} p90={np.percentile(sr, 90):.3f} "
            f"p99={np.percentile(sr, 99):.3f}"
        )


if __name__ == "__main__":
    main()
