"""dkv mask TG 暂存改造的逐位 A/B：patch 前 `--save` 存基线，patch 后
`--check` 对同输入重算并要求 dq/dk/dv 逐位相同，顺带报 bwd 耗时。

用法:
  .venv/bin/python experiments/ab_dkv_maskstage.py --save
  .venv/bin/python experiments/ab_dkv_maskstage.py --check
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels.attn_fused import flash_backward, flash_forward

B, H, T, DK, DV = 12, 12, 1024, 112, 64
REF = "/tmp/dkv_maskstage_ref.npz"


def run():
    mx.random.seed(7)
    q = (mx.random.normal((B, H, T, DK)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, DK)) * 0.3).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, DV)) * 0.3).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, DV)) * 0.5).astype(mx.bfloat16)
    # 真实风格 doc 偏置（平均 doc ~176）
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 176).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    mask = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(q, k, v, do, mask)
    scale = DK**-0.5
    o, lse = flash_forward(q, k, v, scale, mask)
    mx.eval(o, lse)
    for _ in range(3):
        dq, dk, dv = flash_backward(q, k, v, o, do, scale, mask, lse=lse)
        mx.eval(dq, dk, dv)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        dq, dk, dv = flash_backward(q, k, v, o, do, scale, mask, lse=lse)
        mx.eval(dq, dk, dv)
        ts.append(time.perf_counter() - t0)
    return dq, dk, dv, min(ts) * 1e3


def main():
    check = "--check" in sys.argv
    dq, dk, dv, ms = run()
    print(f"flash_backward(doc_mask) 单层: {ms:.2f}ms")
    if check:
        ref = mx.load(REF)
        ok = True
        for name, arr in (("dq", dq), ("dk", dk), ("dv", dv)):
            same = bool(mx.array_equal(arr, ref[name]).item())
            ok &= same
            print(f"  {name} 逐位同={same}")
        print("BITWISE-OK" if ok else "BITWISE-DIFF")
        sys.exit(0 if ok else 1)
    else:
        mx.savez(REF, dq=dq, dk=dk, dv=dv)
        print(f"基线已存 {REF}")


if __name__ == "__main__":
    main()
