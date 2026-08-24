"""原型：flash 前向 + 手写分块 causal VJP，对比 MLX autodiff 反向。

现状（bs12×1024×8head、qk=v=128）：flash 前向 3.7ms，但反向 26ms —— MLX
没有 SDPA 的反向 kernel，退化成对朴素式子求导，会物化整块 (B,H,T,T) 的
scores/softmax（f32 下 402MB），且不利用 causal 只算一半。按 FLOPs 估反向
只需 ~6ms，所以这里有 4× 的空间。

手写 VJP 的关键：softmax 反向的行修正项不需要额外 matmul ——
  rowsum(dA⊙A) = Σ_j (dO_i·V_j)·A_ij = dO_i·O_i
所以 dS = A ⊙ (dO@Vᵀ − (dO·O))，只需 5 个 matmul，且按 query 分块后每块只
扫 causal 允许的 key 范围（约一半），中间张量缩到 (B,H,Bq,Kj) 能进缓存。

用法: uv run experiments/proto_attn_vjp.py [chunk]
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx


def sdpa_flash(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def make_chunked_sdpa(scale, mask, chunk, T):
    """flash 前向 + 手写分块 VJP。mask 为 "causal" 或 (B,1,T,T) 加性数组。"""
    is_str = isinstance(mask, str)

    @mx.custom_function
    def attn(q, k, v):
        return sdpa_flash(q, k, v, scale, mask)

    def _vjp(primals, cotangents, outputs):
        q, k, v = primals
        do = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        o = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        # 行修正项：rowsum(dA⊙A) = dO·O，逐 query 一个标量
        row = (do.astype(mx.float32) * o.astype(mx.float32)).sum(axis=-1, keepdims=True)
        dq_parts = []
        dk = mx.zeros_like(k)
        dv = mx.zeros_like(v)
        for qs in range(0, T, chunk):
            qe = min(qs + chunk, T)
            ke = qe  # causal：query 块只看到 key 的前 qe 个
            qc = q[:, :, qs:qe]
            kc = k[:, :, :ke]
            vc = v[:, :, :ke]
            s = (qc @ kc.swapaxes(-1, -2)).astype(mx.float32) * scale
            if is_str:
                tri = mx.triu(
                    mx.full((qe - qs, ke), -mx.inf, dtype=mx.float32), k=qs + 1
                )
                s = s + tri
            else:
                s = s + mask[:, :, qs:qe, :ke].astype(mx.float32)
            a = mx.softmax(s, axis=-1)
            doc = do[:, :, qs:qe]
            da = doc.astype(mx.float32) @ vc.swapaxes(-1, -2).astype(mx.float32)
            ds = (a * (da - row[:, :, qs:qe])).astype(q.dtype)
            a_t = a.astype(q.dtype)
            dq_parts.append((ds @ kc) * scale)
            dk = dk.at[:, :, :ke].add(ds.swapaxes(-1, -2) @ qc * scale)
            dv = dv.at[:, :, :ke].add(a_t.swapaxes(-1, -2) @ doc)
        return mx.concatenate(dq_parts, axis=2), dk, dv

    attn.vjp(_vjp)
    return attn


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    B = int(os.environ.get("VIBY_BENCH_B", 12))
    T = int(os.environ.get("VIBY_BENCH_T", 1024))
    H, d = 8, 128
    scale = d**-0.5
    print(f"B={B} T={T} H={H} d={d} chunk={chunk}")

    q = (mx.random.normal((B, H, T, d)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, d)) * 0.3).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, d)) * 0.3).astype(mx.bfloat16)
    C = mx.random.normal((B, H, T, d)) * 0.5
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    bias = mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)
    mx.eval(q, k, v, C, bias)

    for mname, mask in (("causal", "causal"), ("doc_mask 数组", bias)):
        chunked = make_chunked_sdpa(scale, mask, chunk, T)

        def loss_ref(q_, k_, v_):
            return (sdpa_flash(q_, k_, v_, scale, mask).astype(mx.float32) * C).sum()

        def loss_new(q_, k_, v_):
            return (chunked(q_, k_, v_).astype(mx.float32) * C).sum()

        vg_ref = mx.value_and_grad(loss_ref, argnums=(0, 1, 2))
        vg_new = mx.value_and_grad(loss_new, argnums=(0, 1, 2))

        vr, gr = vg_ref(q, k, v)
        vn, gn = vg_new(q, k, v)
        mx.eval(vr, gr, vn, gn)
        rel = [
            float(
                (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
                / max(b.astype(mx.float32).abs().max().item(), 1e-9)
            )
            for a, b in zip(gn, gr)
        ]
        print(
            f"\n[{mname}] 梯度相对差 dq/dk/dv = " + ", ".join(f"{r:.2e}" for r in rel)
        )

        arms = {"autodiff 反向": vg_ref, "手写分块 VJP": vg_new}
        samples = {name: [] for name in arms}
        for rnd in range(9):
            for name, fn in arms.items():
                t0 = time.perf_counter()
                mx.eval(fn(q, k, v))
                if rnd >= 2:
                    samples[name].append(time.perf_counter() - t0)
        med = {n: statistics.median(s) for n, s in samples.items()}
        for n, t in med.items():
            print(f"  {n:<16}{t * 1e3:7.2f}ms")
        a, b = med["autodiff 反向"], med["手写分块 VJP"]
        print(f"  提速 {a / b:.2f}x")


if __name__ == "__main__":
    main()
