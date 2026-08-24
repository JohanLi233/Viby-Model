"""Metal flash-attention 反向：相对 MLX sdpa 的前向 / 梯度回归。

训练路径用自定义 fwd（同时写出 logsumexp）+ 手写 Metal VJP，替代 MLX
对 flash 前向的 autodiff（会物化 (B,H,T,T)）。本测试锁定：

  1. 前向贴近 mx.fast.scaled_dot_product_attention（bf16 舍入级）
  2. 梯度贴近对「朴素 f32 SDPA」的 autodiff（flash 反向的正确性锚）
  3. causal 与加性 doc_mask 两条训练会走到的 mask 路径
  4. T 不是 tile 倍数的尾巴（余数 tile）

用法: uv run experiments/test_attn_fused.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["VIBY_ATTN_FUSED"] = "1"

import mlx.core as mx

from model.kernels.attn_fused import flash_sdpa

FAIL = 0


def relmax(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    den = max(float(mx.abs(b).max().item()), 1e-9)
    return float(mx.abs(a - b).max().item()) / den


def naive_sdpa(q, k, v, scale, mask):
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    if mask == "causal":
        T = q.shape[2]
        s = s + mx.triu(mx.full((T, T), -1e9, dtype=mx.float32), k=1)
    elif mask is not None:
        s = s + mask.astype(mx.float32)
    p = mx.softmax(s, axis=-1)
    return p @ v.astype(mx.float32)


def make_doc_mask(B, T, seed=0):
    mx.random.seed(seed)
    seg = mx.cumsum(
        (mx.random.uniform(shape=(B, T)) < 1 / 340).astype(mx.int32), axis=1
    )
    same = seg[:, :, None] == seg[:, None, :]
    tril = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    return mx.where((same & tril[None])[:, None], 0.0, -1e9).astype(mx.bfloat16)


def check(name, ok, detail=""):
    global FAIL
    if ok:
        print(f"[PASS] {name}{detail}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}{detail}")


def run_case(name, B, H, T, D, mask, fwd_tol, grad_tol):
    mx.random.seed(1)
    q = (mx.random.normal((B, H, T, D)) * 0.3).astype(mx.bfloat16)
    k = (mx.random.normal((B, H, T, D)) * 0.3).astype(mx.bfloat16)
    v = (mx.random.normal((B, H, T, D)) * 0.3).astype(mx.bfloat16)
    C = mx.random.normal((B, H, T, D)) * 0.5
    scale = D**-0.5
    mx.eval(q, k, v, C)
    if mask is not None and not isinstance(mask, str):
        mx.eval(mask)

    ref_o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    new_o = flash_sdpa(q, k, v, scale=scale, mask=mask)
    mx.eval(ref_o, new_o)
    r_fwd = relmax(new_o, ref_o)
    check(f"{name} 前向 vs flash", r_fwd < fwd_tol, f"  relmax={r_fwd:.2e}")

    def loss_naive(q_, k_, v_):
        o = naive_sdpa(q_, k_, v_, scale, mask)
        return (o * C).sum()

    def loss_new(q_, k_, v_):
        o = flash_sdpa(q_, k_, v_, scale=scale, mask=mask)
        return (o.astype(mx.float32) * C).sum()

    vg_n = mx.value_and_grad(loss_naive, argnums=(0, 1, 2))
    vg_f = mx.value_and_grad(loss_new, argnums=(0, 1, 2))
    ln, gn = vg_n(q, k, v)
    lf, gf = vg_f(q, k, v)
    mx.eval(ln, gn, lf, gf)
    tags = ("dq", "dk", "dv")
    rels = [relmax(a, b) for a, b in zip(gf, gn)]
    ok = all(r < grad_tol for r in rels)
    detail = "  " + ", ".join(f"{t}={r:.2e}" for t, r in zip(tags, rels))
    check(f"{name} 梯度 vs f32 naive", ok, detail)


def main():
    D = 128
    run_case("单 tile causal T=16", 1, 1, 16, D, "causal", 3e-2, 5e-2)
    run_case("两 tile causal T=32", 1, 2, 32, D, "causal", 3e-2, 5e-2)
    run_case("余数 causal T=40", 1, 1, 40, D, "causal", 3e-2, 5e-2)
    bias = make_doc_mask(2, 48)
    run_case("doc_mask T=48", 2, 2, 48, D, bias, 3e-2, 5e-2)
    run_case("训练形状 causal", 2, 8, 128, D, "causal", 3e-2, 8e-2)
    bias = make_doc_mask(2, 128)
    run_case("训练形状 doc_mask", 2, 8, 128, D, bias, 3e-2, 8e-2)

    print()
    if FAIL:
        print(f"{FAIL} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
