"""手写 flash 反向 kernel 的数值对拍。

逐个 kernel 与 mlx 算子参考实现比对，跑通一个再写下一个——这类 kernel 一旦
三个一起上，错在哪几乎无法定位。

用法: uv run experiments/verify_flash_bwd.py [lse|dq|dkv|all]
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kernels import attn_fused as af


def maxrel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    d = mx.abs(a - b)
    fin = mx.isfinite(a) & mx.isfinite(b)
    d = mx.where(fin, d, 0.0)
    scale = max(mx.abs(mx.where(fin, a, 0.0)).max().item(), 1e-6)
    return d.max().item() / scale


def make_inputs(B, H, Tq, Tk, Dk, Dv, doc_mask, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((B, H, Tq, Dk)) * 0.4).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, Tk, Dk)) * 0.4).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, Tk, Dv)) * 0.4).astype(mx.bfloat16)
    do = (mx.random.normal((B, H, Tq, Dv)) * 0.4).astype(mx.bfloat16)
    mask = None
    if doc_mask:
        # 打包序列的文档边界：平均每 340 token 一篇
        seg = mx.cumsum(
            (mx.random.uniform(shape=(B, Tq)) < 1 / 340).astype(mx.int32), axis=1
        )
        same = seg[:, :, None] == seg[:, None, :]
        mask = mx.where(same[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(q, k, v, do, mask)
    return q, k, v, do, mask


def run_lse(q, k, scale, mask, nt=None, strb=None):
    nt = nt or af.NTHREADS
    strb = strb or af.STR
    B, H, Tq, Dk = q.shape
    key = (Dk, 96, Tq, k.shape[2], H, scale, mask is not None, "bfloat16_t", nt, strb)
    return af._get("lse", *key)(
        inputs=[q, k] + ([mask] if mask is not None else []),
        output_shapes=[(B, H, Tq)],
        output_dtypes=[mx.float32],
        grid=(nt, Tq // af._res(nt), B * H),
        threadgroup=(nt, 1, 1),
    )[0]


def check_lse():
    print("=== kernel 1: logsumexp ===")
    ok = True
    for B, H, Tq, Tk, Dk, dm in (
        (1, 1, 64, 64, 128, False),
        (1, 2, 128, 128, 128, False),
        (2, 4, 256, 256, 128, False),
        (2, 4, 256, 256, 128, True),
        (2, 8, 1024, 1024, 128, True),
    ):
        q, k, _, _, mask = make_inputs(B, H, Tq, Tk, Dk, 96, dm)
        scale = 1.0 / math.sqrt(Dk)
        got = run_lse(q, k, scale, mask)
        ref = af.reference_lse(q, k, scale, mask)
        mx.eval(got, ref)
        e = maxrel(got, ref)
        bad = e > 3e-3
        ok &= not bad
        tag = "FAIL" if bad else "ok"
        print(
            f"  [{tag}] B={B} H={H} T={Tq} Dk={Dk} "
            f"{'doc_mask' if dm else 'causal  '}  最大相对误差 {e:.2e}"
        )
    return ok


def check_bwd():
    """dq/dk/dv 三路梯度。

    kernel 内部 P/dS 以 bf16 落 threadgroup 再喂 MMA，与 f32 参考必然有
    bf16 量级的偏差。所以同时量出 mlx autodiff（同样 bf16 输入）相对 f32
    参考的偏差作为标尺——手写 kernel 只要不明显差于它就是对的。
    """
    print("=== kernel 2+3: dq / dk / dv ===")
    ok = True
    for B, H, Tq, Tk, Dk, Dv, dm in (
        (1, 1, 64, 64, 128, 96, False),
        (1, 2, 128, 128, 128, 96, False),
        (2, 4, 256, 256, 128, 96, False),
        (2, 4, 256, 256, 128, 96, True),
        (2, 4, 256, 256, 64, 64, False),
        (2, 8, 1024, 1024, 128, 96, True),
    ):
        q, k, v, do, mask = make_inputs(B, H, Tq, Tk, Dk, Dv, dm)
        scale = 1.0 / math.sqrt(Dk)

        def loss(q_, k_, v_):
            return (af.reference_attention(q_, k_, v_, scale, mask) * do).sum()

        _, ref = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)

        def loss_bf(q_, k_, v_):
            s = (q_ @ mx.swapaxes(k_, -1, -2)).astype(mx.float32) * scale
            if mask is not None:
                s = s + mask.astype(mx.float32)
            s = s + mx.triu(mx.full((Tq, Tk), -mx.inf), k=1)
            o_ = mx.softmax(s, axis=-1).astype(q_.dtype) @ v_
            return (o_.astype(mx.float32) * do).sum()

        _, base = mx.value_and_grad(loss_bf, argnums=(0, 1, 2))(q, k, v)

        o = af.reference_attention(q, k, v, scale, mask).astype(q.dtype)
        got = af.flash_backward(q, k, v, o, do, scale, mask)
        mx.eval(got, ref, base)

        row = []
        bad = False
        for nm, g, r, b in zip("qkv", got, ref, base):
            eg, eb = maxrel(g, r), maxrel(b, r)
            bad |= eg > max(4 * eb, 2e-2)
            row.append(f"d{nm} {eg:.1e}/{eb:.1e}")
        ok &= not bad
        print(
            f"  [{'FAIL' if bad else 'ok'}] B={B} H={H} T={Tq} Dk={Dk} Dv={Dv} "
            f"{'doc_mask' if dm else 'causal  '}  " + "  ".join(row)
        )
    print("  （每格是 手写kernel误差/mlx-autodiff误差，同以 f32 为基准）")
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    allok = True
    if which in ("lse", "all"):
        allok &= check_lse()
    if which in ("bwd", "dq", "dkv", "all"):
        allok &= check_bwd()
    print("\n全部通过" if allok else "\n存在失败项")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
