"""FusedAdamW（mx.compile 融合）vs optim.AdamW：多步逐位一致性 + 提速。

融合改写了算子的执行方式而非语义，所以要求参数与 m/v 状态在连续多步后仍
逐位相同（bf16 下任何提升次序的偏差都会立刻显形）。同时对真实的堆叠专家
权重形状测吞吐。

用法: uv run experiments/verify_fused_adamw.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.optimizers as optim

from trainer.muon import FusedAdamW

FAIL = 0


def run(opt, params, grads, steps):
    for _ in range(steps):
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state)
    return params


def case(name, shapes, dtype, lr, wd, betas, eps, steps=5, seed=0):
    global FAIL
    mx.random.seed(seed)
    params = {
        f"w{i}": (mx.random.normal(s) * 0.02).astype(dtype)
        for i, s in enumerate(shapes)
    }
    grads = {
        f"w{i}": (mx.random.normal(s) * 0.01).astype(dtype)
        for i, s in enumerate(shapes)
    }
    mx.eval(params, grads)

    ref = optim.AdamW(learning_rate=lr, betas=betas, eps=eps, weight_decay=wd)
    new = FusedAdamW(learning_rate=lr, betas=betas, eps=eps, weight_decay=wd)
    p_ref = run(ref, dict(params), grads, steps)
    p_new = run(new, dict(params), grads, steps)

    same_p = all(bool(mx.array_equal(p_ref[k], p_new[k])) for k in params)
    sr, sn = ref.state, new.state
    same_s = all(
        bool(mx.array_equal(sr[k][s], sn[k][s])) for k in params for s in ("m", "v")
    )
    ok = same_p and same_s
    FAIL += 0 if ok else 1
    if not ok:
        d = max(
            (p_ref[k].astype(mx.float32) - p_new[k].astype(mx.float32))
            .abs()
            .max()
            .item()
            for k in params
        )
        extra = f" maxdiff={d:.3e}"
    else:
        extra = ""
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: {steps} 步后 参数逐位同={same_p} "
        f"m/v 逐位同={same_s}{extra}"
    )


def bench():
    E, I, D = 288, 104, 768  # noqa: E741
    shapes = [(E, 2 * I, D)] * 9 + [(E, D, I)] * 9  # 8 层 + 1 个 MTP block
    n = sum(int(mx.zeros(s).size) for s in shapes)
    print(f"\n吞吐（堆叠专家权重真实形状，{n / 1e6:.0f}M 参数）")
    for label, cls in (("optim.AdamW", optim.AdamW), ("FusedAdamW", FusedAdamW)):
        params = {
            f"w{i}": (mx.random.normal(s) * 0.02).astype(mx.bfloat16)
            for i, s in enumerate(shapes)
        }
        grads = {k: mx.ones_like(v) * 0.01 for k, v in params.items()}
        opt = cls(learning_rate=mx.array(0.01), weight_decay=0.1)
        mx.eval(params, grads)
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state)
        ts = []
        for _ in range(8):
            t0 = time.perf_counter()
            params = opt.apply_gradients(grads, params)
            mx.eval(params, opt.state)
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        # 必要访存量：读 p/g/m/v + 写 p/m/v = 7 × n × 2B
        bw = 7 * n * 2 / t / 2**30
        print(f"  {label:<14}{t * 1e3:7.1f}ms   等效带宽 {bw:6.1f} GB/s")


def main():
    case(
        "堆叠专家权重形状/bf16",
        [(64, 208, 768), (64, 768, 104)],
        mx.bfloat16,
        lr=0.01,
        wd=0.1,
        betas=[0.9, 0.95],
        eps=1e-8,
    )
    case(
        "1D/2D 混合/bf16",
        [(768,), (768, 768)],
        mx.bfloat16,
        0.003,
        0.1,
        [0.9, 0.95],
        1e-8,
    )
    case("f32", [(512, 512)], mx.float32, 0.01, 0.01, [0.9, 0.999], 1e-8)
    case("wd=0", [(256, 256)], mx.bfloat16, 0.01, 0.0, [0.9, 0.95], 1e-8)
    case(
        "同形状堆叠/bf16",
        [(768,)] * 24 + [(96,)] * 12,
        mx.bfloat16,
        0.003,
        0.1,
        [0.9, 0.95],
        1e-8,
    )
    bench()
    print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 个用例失败"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
