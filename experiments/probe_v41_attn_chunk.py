"""滑动窗口分块 probe：稠密 [T, T+N] sdpa vs 把 query 切成 W 大小的块、每块只看 [2W+N]。

训练路径现状：滑窗分支把整个 block 的 T 个 key 都算进去，靠掩码只留 W 个可见。
分块后每块的 query 只对 [前一块 W + 本块 W + 压缩 N] 做注意力，语义等价（窗口
W 内的 key 必落在这两块里），但 S 从 T+N 降到 2W+N。
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlx.core as mx

def bench(fn, n=5, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[0]*1000, ts[len(ts)//2]*1000

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--window", type=int, default=128)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--head-dim", type=int, default=128)
args = ap.parse_args()
B,T,W,H,D = args.batch, args.seq, args.window, args.heads, args.head_dim
C = T // W
mx.random.seed(0)
q = mx.random.normal((B,T,H,D)).astype(mx.bfloat16)
modes = (("sliding", 0), ("ratio2", T//2), ("ratio1", T))

def dense(N):
    S = T + N
    k = mx.random.normal((B,S,D)).astype(mx.bfloat16)
    mask = (mx.random.uniform(0,1,(B,1,T,S)) > 0.7)
    mx.eval(k, mask)
    return lambda: mx.eval(mx.fast.scaled_dot_product_attention(
        q.transpose(0,2,1,3), k[:,None], k[:,None], scale=D**-0.5, mask=mask))

def chunked(N):
    S = 2*W + N
    comp = mx.random.normal((B,N,D)).astype(mx.bfloat16)
    kv = mx.random.normal((B,T,D)).astype(mx.bfloat16)
    win = (mx.random.uniform(0,1,(W,2*W)) > 0.5)
    cmask = mx.random.uniform(0,1,(B,C,W,N)) > 0.7
    mx.eval(comp, kv, win, cmask)
    def run():
        kc = kv.reshape(B,C,W,D)
        prev = mx.concatenate([mx.zeros((B,1,W,D), dtype=kv.dtype), kc[:,:-1]], axis=1)
        kb = mx.concatenate([prev, kc], axis=2)                       # [B,C,2W,D]
        kb = mx.concatenate([kb[:,:,None], mx.broadcast_to(comp[:,None,None], (B,C,1,N,D))], axis=3)
        kb = kb.reshape(B*C, 1, S, D)
        qc = q.reshape(B,C,W,H,D).reshape(B*C, W, H, D).transpose(0,2,1,3)
        m = mx.concatenate([mx.broadcast_to(win[None,None,None], (B,C,1,W,2*W)), cmask[:,:,None]], axis=-1)
        o = mx.fast.scaled_dot_product_attention(qc, kb, kb, scale=D**-0.5, mask=m.reshape(B*C,1,W,2*W+N))
        mx.eval(o)
    return run

print("B=%d T=%d W=%d H=%d D=%d" % (B,T,W,H,D))
for name, N in modes:
    a = bench(dense(N)); b = bench(chunked(N))
    print("  %-8s N=%4d  稠密 S=%4d %6.2f/%6.2f ms  分块 S=%4d %6.2f/%6.2f ms"
          % (name, N, T+N, a[0], a[1], 2*W+N, b[0], b[1]), flush=True)
