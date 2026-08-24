"""attn_v2 变体 kernel 的数值对拍：vs af(v1) 输出 + vs f32 参考梯度。

用法: uv run experiments/verify_v2.py [cfg_name]
cfg_name 可选：default / nofast / nomax / nodelta / nohoist 等（见 CFGS）。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from experiments import attn_v2 as v2
from model.kernels import attn_fused as af

CFGS = {
    "default": {},
    "nofast": {"fast_exp": False},
    "nomax": {"lse_nomax": False},
    "nodelta": {"lse_delta": False},
    "nohoist": {"dq_hoist": False},
    "qcache": {"dq_qcache": True},
    "nokhoist": {"dkv_khoist": False},
    "nopad": {"dkv_pad": False},
}


def maxrel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a - b)
    fin = mx.isfinite(a) & mx.isfinite(b)
    d = mx.where(fin, d, 0.0)
    scale = max(mx.abs(mx.where(fin, a, 0.0)).max().item(), 1e-6)
    return d.max().item() / scale


def make_inputs(B, H, T, Dk, Dv, doc_mask, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((B, H, T, Dk)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, Dk)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, Dv)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, T, Dv)) * 0.4).astype(mx.bfloat16)
    mask = None
    if doc_mask:
        seg = mx.cumsum(
            (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
        )
        same = seg[:, :, None] == seg[:, None, :]
        mask = mx.where(same[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(q, k, v, do, mask)
    return q, k, v, do, mask


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    cfg = CFGS[cfg_name]
    print(f"=== attn_v2 对拍（cfg={cfg_name}）===")
    ok = True
    for B, H, T, Dk, Dv, dm in (
        (1, 1, 64, 128, 96, False),
        (2, 4, 256, 128, 96, False),
        (2, 4, 256, 128, 96, True),
        (2, 4, 256, 64, 64, False),
        (2, 8, 1024, 128, 96, True),
    ):
        q, k, v, do, mask = make_inputs(B, H, T, Dk, Dv, dm)
        scale = 1.0 / math.sqrt(Dk)
        o = af.reference_attention(q, k, v, scale, mask).astype(q.dtype)

        got = v2.flash_backward(q, k, v, o, do, scale, mask, cfg=cfg)
        ref = af.flash_backward(q, k, v, o, do, scale, mask)
        mx.eval(got, ref)

        # 参考梯度（f32 朴素 autodiff）
        def loss(q_, k_, v_):
            return (af.reference_attention(q_, k_, v_, scale, mask) * do).sum()

        _, g32 = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(g32)

        row, bad = [], False
        for nm, g, r, g3 in zip("qkv", got, ref, g32):
            e_v1, e_v2 = maxrel(g, r), maxrel(g, g3)
            bad |= e_v2 > max(4 * e_v1, 2e-2)
            row.append(f"d{nm} v2/v1 {e_v2:.1e}/{e_v1:.1e}")
        ok &= not bad
        print(
            f"  [{'FAIL' if bad else 'ok'}] B={B} H={H} T={T} Dk={Dk} Dv={Dv} "
            f"{'doc_mask' if dm else 'causal  '}  " + "  ".join(row)
        )
    print("全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
