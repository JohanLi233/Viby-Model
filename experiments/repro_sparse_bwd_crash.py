"""最小复现：model/kernels/sparse_attention.py 的反向 SIGSEGV。

现状（2026-09-10 21:0x 实测，MLX 0.32.2 / M4 Max）：

    indexed_attention 前向            → OK
    mx.vjp(..., indexed_attention)    → Segmentation fault: 11

因为 `VibyForCausalLM.__init__` 里 `prewarm_sparse_attention(...)` 会做一次
vjp 预热（model/model.py），所以**默认配置下模型连构造都会 segfault**，
训练/基准/测试全部跑不起来。临时绕过：`VIBY_SPARSE_ATTN_KERNEL=0`。

用法：
    .venv/bin/python experiments/repro_sparse_bwd_crash.py            # 默认：前向（应当 OK）
    .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd      # 反向（当前 segfault）
    .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd --b 1 --t 1 --n 1
"""

import argparse
import sys

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from model.kernels.sparse_attention import indexed_attention, prewarm_sparse_attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--bwd", action="store_true", help="跑 mx.vjp（当前会 segfault）")
    ap.add_argument("--prewarm", action="store_true", help="直接调 prewarm_sparse_attention")
    args = ap.parse_args()

    if args.prewarm:
        print("prewarm_sparse_attention(...) ...", flush=True)
        prewarm_sparse_attention(args.head_dim, args.window, args.head_dim ** -0.5, mx.bfloat16)
        print("prewarm OK")
        return

    q = mx.zeros((args.b, args.t, 16, args.head_dim), mx.bfloat16)
    kv = mx.zeros((args.b, args.n, args.head_dim), mx.bfloat16)
    mask = mx.ones((args.b, args.t, args.n), mx.bool_)
    sinks = mx.zeros((16,), mx.float32)
    scale = args.head_dim ** -0.5

    if not args.bwd:
        out = indexed_attention(q, kv, kv, mask, None, None, sinks, args.window, scale)
        mx.eval(out)
        print("forward OK", out.shape)
        return

    def fn(a, b, c, s):
        return indexed_attention(a, b, c, mask, None, None, s, args.window, scale)

    cot = mx.ones((args.b, 16, args.t, args.head_dim), mx.bfloat16)
    print("mx.vjp(...) B=%d T=%d N=%d ..." % (args.b, args.t, args.n), flush=True)
    out, grads = mx.vjp(fn, [q, kv, kv, sinks], [cot])
    mx.eval(out, grads)
    print("forward+backward OK")


if __name__ == "__main__":
    main()
