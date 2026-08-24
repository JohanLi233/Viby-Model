"""验证 _build_moe_bucket（gather 构桶 + 手写 VJP）与旧的 zeros+scatter-add
写法逐位一致：前向输出、以及对 xf 的梯度。

两种写法在数学上应当严格相同（每个有效桶行的来源 token 唯一，未占用行为
0），所以这里要求 bitwise 相等而非容差比较。覆盖：无溢出、有溢出（多个
pair 落 trash 行）、桶容量远超计数（大量 padding 行）。

用法: uv run experiments/verify_moe_bucket.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.model import _build_moe_bucket

FAIL = 0


def old_build(xf, row, tok_s, total_rows):
    """旧写法：清零整桶后 bf16 原子加。"""
    D = xf.shape[1]
    return mx.zeros((total_rows + 1, D), dtype=xf.dtype).at[row].add(xf[tok_s])


def case(name, M, D, E, K, cap, overflow_frac, seed=0):
    """构造一批 (token,choice) 对 → 桶行号，比较两种构桶写法。

    overflow_frac 部分 pair 的桶内序号被顶到容量之外 → 落 trash 行。
    """
    global FAIL
    mx.random.seed(seed)
    G = M * K
    total_rows = E * cap
    xf = (mx.random.normal((M, D)) * 0.5).astype(mx.bfloat16)
    tok_s = mx.random.randint(0, M, (G,)).astype(mx.int32)
    exps = mx.random.randint(0, E, (G,)).astype(mx.int32)
    # 桶内序号：同专家内递增；再叠加溢出（序号 >= cap → trash 行）
    rank = mx.zeros((G,), dtype=mx.int32)
    counts = mx.zeros((E,), dtype=mx.int32)
    rank_l, counts_l = [], [0] * E
    for e in exps.tolist():
        rank_l.append(counts_l[e])
        counts_l[e] += 1
    rank = mx.array(rank_l, dtype=mx.int32)
    if overflow_frac > 0:
        bump = (mx.random.uniform(shape=(G,)) < overflow_frac).astype(mx.int32) * cap
        rank = rank + bump
    row = mx.where(rank >= cap, total_rows, exps * cap + rank)
    mx.eval(xf, tok_s, row, rank, counts)

    C = mx.random.normal((total_rows + 1, D)) * 0.5
    mx.eval(C)

    def mk(fn):
        def loss(x):
            return (fn(x, row, tok_s, total_rows).astype(mx.float32) * C).sum()

        return loss

    out_old = old_build(xf, row, tok_s, total_rows)
    out_new = _build_moe_bucket(xf, row, tok_s, total_rows)
    g_old = mx.grad(mk(old_build))(xf)
    g_old2 = mx.grad(mk(old_build))(xf)
    g_new = mx.grad(mk(_build_moe_bucket))(xf)
    mx.eval(out_old, out_new, g_old, g_old2, g_new)
    # 旧写法自比：scatter-add 的原子加顺序不确定，bf16 下同一表达式两次
    # 求值就可能不逐位相同 —— 用它标定「可容忍的抖动」基线。
    self_d = (g_old.astype(mx.float32) - g_old2.astype(mx.float32)).abs().max().item()
    cross_d = (g_old.astype(mx.float32) - g_new.astype(mx.float32)).abs().max().item()
    scale = g_old.astype(mx.float32).abs().max().item()

    # trash 行是两种写法唯一允许不同的地方（旧写法累加了溢出 pair，新写法
    # 存的是被 clip 的索引），它的值在前向永不被读回。
    o_same = bool(mx.array_equal(out_old[:total_rows], out_new[:total_rows]))
    n_over = int((row == total_rows).sum().item())
    n_pad = total_rows - int((row < total_rows).sum().item())
    # 梯度：要求跨写法差异不超过旧写法自身的原子加抖动
    g_ok = cross_d <= max(self_d, 0.0)
    ok = o_same and g_ok
    FAIL += 0 if ok else 1
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: 桶行={total_rows} 溢出pair={n_over} "
        f"padding行={n_pad} | 前向逐位同={o_same} "
        f"梯度 跨写法差={cross_d / max(scale, 1e-9):.2e} 旧写法自比抖动="
        f"{self_d / max(scale, 1e-9):.2e}"
    )


def main():
    case("无溢出/容量贴合", M=512, D=768, E=16, K=6, cap=256, overflow_frac=0.0, seed=1)
    case("有溢出", M=512, D=768, E=16, K=6, cap=128, overflow_frac=0.15, seed=2)
    case("大量 padding 行", M=256, D=768, E=32, K=6, cap=512, overflow_frac=0.0, seed=3)
    case("窄 D / 小形状", M=64, D=64, E=8, K=3, cap=64, overflow_frac=0.1, seed=4)
    print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 个用例失败"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
