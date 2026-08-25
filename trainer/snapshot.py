"""P-0/P-1 动量与微批梯度快照（论文弧线测量基建，见 OPTIMIZER_RESEARCH §0.5）。

env 门控（默认全关，零开销）：
    VIBY_SNAPSHOT_DIR    输出目录（设置即启用）
    VIBY_SNAPSHOT_STEPS  逗号分隔的优化器步列表，如 "150,300,500"
                         （BatchedMuon.state["step"] 口径，即第 k 次更新后）

每个列出的优化器步 k 产出：
  mom_step{k}.npz        该步送进 NS 的动量 U（nesterov 后，F-归一前的原始
                         尺度），2D 组全量、堆叠组按 8 张量槽 × 6 专家
                         结构化子采样；
  grad_step{k}_mb{j}.npz 构成第 k 次与第 k+1 次更新的 2×accum 个独立微批
                         梯度（乘回 accumulation_steps，恢复真实微批尺度），
                         与动量同一路径/同一专家子采样——供跨 microbatch
                         方向相关（信号持续性）估计。

分析脚本：experiments/mp_fit.py。
"""

import json
import os

import numpy as np

_DIR = os.environ.get("VIBY_SNAPSHOT_DIR", "")
_STEPS = frozenset(
    int(s) for s in os.environ.get("VIBY_SNAPSHOT_STEPS", "").split(",") if s.strip()
)
# 堆叠组子采样：张量槽数 × 专家数
_N_SLOTS = 8
_N_EXPERTS = 6


def active() -> bool:
    return bool(_DIR) and bool(_STEPS)


def stack_subsample(n_tensors: int, b: int):
    """堆叠组 (N, b, r, c) 的结构化子采样下标（跨全部层/投影均匀取）。"""
    ns = np.linspace(0, n_tensors - 1, min(_N_SLOTS, n_tensors)).astype(int)
    js = np.linspace(0, b - 1, min(_N_EXPERTS, b)).astype(int)
    return sorted(set(ns.tolist())), sorted(set(js.tolist()))


def _save(step: int, tag: str, arrays: dict, meta: dict):
    os.makedirs(_DIR, exist_ok=True)
    payload = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    payload["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(os.path.join(_DIR, f"{tag}_step{step}.npz"), **payload)


def maybe_dump_momentum(step: int, kind: str, key, U, paths, b: int = 0):
    """BatchedMuon.apply_gradients 内调用。U: (N,r,c) 或 (N·b,r,c) 原始动量。

    kind: "g2d"（2D 组，key=(r,c)）或 "gst"（堆叠组，key=(b,r,c)，
    U 已按 N·b 展平，b 为每组张量的堆叠数）。
    """
    if not active() or step not in _STEPS:
        return
    import mlx.core as mx

    mx.eval(U)
    if kind == "g2d":
        r, c = key
        arr = np.array(U.astype(mx.float32))
        _save(
            step,
            f"mom_{kind}_{r}x{c}",
            {"U": arr},
            {
                "kind": kind,
                "r": r,
                "c": c,
                "paths": list(paths),
            },
        )
    else:
        bb, r, c = key
        n_t = U.shape[0] // bb
        ns, js = stack_subsample(n_t, bb)
        U4 = np.array(U.astype(mx.float32)).reshape(n_t, bb, r, c)
        sub = U4[np.ix_(ns, js)]  # (len(ns), len(js), r, c)
        _save(
            step,
            f"mom_{kind}_{bb}x{r}x{c}",
            {"U": sub},
            {
                "kind": kind,
                "r": r,
                "c": c,
                "b": bb,
                "paths": [paths[i] for i in ns],
                "slot_idx": ns,
                "expert_idx": js,
            },
        )


def maybe_dump_grads(opt_step: int, mb_idx: int, flat_g: dict, accum_steps: int):
    """base_trainer 微批循环内调用：第 opt_step 次更新的第 mb_idx 个微批。

    flat_g 是 tree_flatten 后的本微批梯度（已被 loss 侧除以
    accumulation_steps，这里乘回）。与动量同一子采样规则。
    """
    if not active():
        return
    src = None
    for k in _STEPS:
        if opt_step in (k, k + 1):
            src = k
            break
    if src is None:
        return
    import mlx.core as mx

    # 同一时点 src 捕获 opt_step=src 与 src+1 两个更新窗：文件内 mb 序号
    # 全局化，避免后一窗口覆盖前一窗口
    file_mb = (opt_step - src) * accum_steps + mb_idx

    mx.eval(dict(flat_g))
    arrays, meta = {}, {"paths": {}, "expert_idx": {}}
    for path, g in flat_g.items():
        if g.ndim < 2:
            continue  # 只关心 Muon 组（ndim>=2）
        if "embed_tokens" in path or "lm_head" in path:
            continue  # AdamW/AdamH 组，与谱分析无关且占体积
        a = np.array(g.astype(mx.float32)) * float(accum_steps)
        if g.ndim > 2:
            b = g.shape[0]
            _, js = stack_subsample(1, b)
            a = a[js]
            meta["expert_idx"][path] = js
        arrays[path] = a
        meta["paths"][path] = list(g.shape)
    _save(src, f"grad_mb{file_mb}", arrays, meta)
