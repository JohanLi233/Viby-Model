"""稀疏注意力 GPU 路径的独立-KV 冒烟：前向、VJP、prewarm。

window 长度必须等于 query 的 T，compressed 才是 N。二者不能共用同一块
kv，否则 T≠N 时会构造错误的窗口。

历史：2026-09-10 的 VJP SIGSEGV 已在源码侧修过（metadata 零叶子）。本脚本
不再假设会崩溃；退出码和日志才是当前证据。

用法：
    .venv/bin/python experiments/repro_sparse_bwd_crash.py
    .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd
    .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd --b 1 --t 1 --n 1
    .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd --t 8 --n 4
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
    ap.add_argument("--bwd", action="store_true", help="跑 mx.vjp")
    ap.add_argument("--prewarm", action="store_true", help="直接调 prewarm_sparse_attention")
    args = ap.parse_args()

    if args.prewarm:
        print("prewarm_sparse_attention(...) ...", flush=True)
        prewarm_sparse_attention(args.head_dim, args.window, args.head_dim ** -0.5, mx.bfloat16)
        print("prewarm OK")
        return

    q = mx.zeros((args.b, args.t, 16, args.head_dim), mx.bfloat16)
    window = mx.zeros((args.b, args.t, args.head_dim), mx.bfloat16)
    compressed = mx.zeros((args.b, args.n, args.head_dim), mx.bfloat16)
    mask = mx.ones((args.b, args.t, args.n), mx.bool_)
    sinks = mx.zeros((16,), mx.float32)
    scale = args.head_dim ** -0.5

    if not args.bwd:
        out = indexed_attention(q, window, compressed, mask, None, None, sinks, args.window, scale)
        mx.eval(out)
        print("forward OK", tuple(out.shape), "window T=%d compressed N=%d" % (args.t, args.n))
        return

    def fn(a, w, c, s):
        return indexed_attention(a, w, c, mask, None, None, s, args.window, scale)

    cot = mx.ones((args.b, 16, args.t, args.head_dim), mx.bfloat16)
    print("mx.vjp(...) B=%d T=%d N=%d ..." % (args.b, args.t, args.n), flush=True)
    out, grads = mx.vjp(fn, [q, window, compressed, sinks], [cot])
    mx.eval(out, grads)
    print("forward+backward OK")


if __name__ == "__main__":
    main()
