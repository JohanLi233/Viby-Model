"""AttnRes bwd 归约趟合并的逐位 A/B 对拍。

改动内容：bwd kernel 的"rstd 复算"与"da_j=cot·v_j"两趟 v 全读合并为单趟
（每 lane 的元素序列与累加顺序不变 → 预期逐位一致）。本脚本在改动前
--save 参考梯度到 /tmp/attnres_bwd_ref.npz，改动后 --check 逐位对拍。

用法:
  .venv/bin/python experiments/ab_attnres_bwd.py --save
  .venv/bin/python experiments/ab_attnres_bwd.py --check
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_res_fused

REF = "/tmp/attnres_bwd_ref.npz"
D = 768
DT = mx.bfloat16


def _case(n, b, t, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((D,)) * 0.25).astype(DT)
    vs = [(mx.random.normal((b, t, D)) * 0.5).astype(DT) for _ in range(n)]
    mx.eval(w, *vs)
    attn_res_fused.prewarm(D, DT, [n])
    vg = mx.value_and_grad(
        lambda w_, *v_: (
            attn_res_fused.merge(w_, list(v_)).astype(mx.float32) ** 2
        ).sum()
        / 2,
        argnums=range(n + 1),
    )
    loss, grads = vg(w, *vs)
    out = attn_res_fused.merge(w, vs)
    mx.eval(loss, out, *grads)
    return out, loss, grads


def main():
    check = "--check" in sys.argv
    cases = [
        (2, 2, 32, 0),
        (3, 2, 32, 1),
        (8, 2, 64, 2),
        (17, 2, 64, 3),
        (25, 12, 1024, 4),
    ]
    blobs = {}
    for n, b, t, seed in cases:
        out, loss, grads = _case(n, b, t, seed)
        blobs[f"out_{n}"] = out.astype(mx.float32)
        blobs[f"loss_{n}"] = mx.array([loss.item()], dtype=mx.float32)
        for i, g in enumerate(grads):
            blobs[f"g{n}_{i}"] = g.astype(mx.float32)
    mx.eval(*blobs.values())
    if not check:
        import numpy as np

        np.savez(REF, **{k: v for k, v in blobs.items()})
        print(f"saved {len(blobs)} arrays -> {REF}")
        return
    import numpy as np

    ref = np.load(REF)
    ok = True
    for k in blobs:
        a = np.asarray(blobs[k])
        r = ref[k]
        same = a.shape == r.shape and bool((a == r).all())
        if not same:
            ok = False
            d = abs(a - r).max() if a.shape == r.shape else float("nan")
            print(f"  {k}: 逐位同=False max|Δ|={d:.3e}")
    print("BITWISE-OK" if ok else "BITWISE-FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
