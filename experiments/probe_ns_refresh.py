"""专家栈刷新步的子步骤计时（真实优化器上下文 + 内存压力对照）。

probe_ns_gram 在干净进程里测 _ns5_gram（2304,640,384）459ms +
（2304,384,320）251ms = 710ms，接近 11.6 TFLOPS 峰值；但 prof_muon 在
真实优化器里同样两个调用是 664 + 780 = 1444ms。差额要么在被 mx.eval
一并求值的上游（3 次 mx.stack + mom_fn），要么是常驻内存导致的分配抖动。

本脚本用真实模型的梯度/参数/动量形状构造刷新步，逐子步加屏障计时，
并用 ballast 张量单独验证内存压力效应。

用法: uv run python experiments/probe_ns_refresh.py [ballast_GB]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import BatchedMuon, _stack_apply_kernel, _stack_mom_kernel

BALLAST_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

# 1080M 真实专家栈：9 个 MoE 层（8 主干 + 1 MTP），E=256，I=384，
# latent 维 384（moe_latent_dim 默认值），实测自 VibyForCausalLM
E, I, DE = 256, 384, 384  # noqa: E741
NL = 9
GROUPS = [
    ("gate_up (E,2I,DE)", (E, 2 * I, DE)),
    ("down    (E,DE,I)", (E, DE, I)),
]

opt = BatchedMuon(learning_rate=1e-3, hyperball=True, ns_bf16=True, stack_ns_every=8)
mom_fn = _stack_mom_kernel(opt.momentum, opt.nesterov, opt.weight_decay)
apply_fn = _stack_apply_kernel(opt.hyperball)
lr = mx.array(1e-3, dtype=mx.bfloat16)


def timed(fn, it=4, w=2):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


ballast = None
if BALLAST_GB > 0:
    n = int(BALLAST_GB * 2**30 / 2)
    ballast = mx.zeros((n,), dtype=mx.bfloat16)
    mx.eval(ballast)
    print(f"ballast {BALLAST_GB:.1f} GB 常驻")

print(f"\n{'组':<20}{'子步骤':<18}{'ms':>9}")
total = 0.0
for label, (b, r, c) in GROUPS:
    mx.random.seed(0)
    gs = [(mx.random.normal((b, r, c)) * 0.01).astype(mx.bfloat16) for _ in range(NL)]
    ps = [(mx.random.normal((b, r, c)) * 0.05).astype(mx.bfloat16) for _ in range(NL)]
    vs = [(mx.random.normal((b, r, c)) * 0.01).astype(mx.bfloat16) for _ in range(NL)]
    mx.eval(gs, ps, vs)
    one = gs[0].nbytes * NL / 2**30
    print(f"{label:<20}{'单份栈 GB':<18}{one:>9.2f}")

    t_stack = timed(lambda: [mx.stack(gs), mx.stack(ps), mx.stack(vs)])  # noqa: F821
    G, P, V = mx.stack(gs), mx.stack(ps), mx.stack(vs)
    mx.eval(G, P, V)
    t_mom = timed(lambda: mom_fn(G, P, V))  # noqa: F821
    U, Vn = mom_fn(G, P, V)
    mx.eval(U, Vn)
    Uflat = U.reshape(NL * b, r, c)
    t_ns = timed(lambda: opt._ns5_gram(Uflat))  # noqa: F821
    X = opt._ns5_gram(Uflat).reshape(NL, b, r, c)
    mx.eval(X)
    t_apply = timed(lambda: apply_fn(P, X, lr))  # noqa: F821

    # 端到端刷新（与 apply_gradients 里的顺序一致，单次 eval）
    def whole():  # noqa: F821
        G_, P_, V_ = mx.stack(gs), mx.stack(ps), mx.stack(vs)  # noqa: F821
        U_, V2 = mom_fn(G_, P_, V_)
        X_ = opt._ns5_gram(U_.reshape(NL * b, r, c)).reshape(NL, b, r, c)
        return apply_fn(P_, X_, lr), V2

    t_whole = timed(whole)
    for name, t in (
        ("3× mx.stack", t_stack),
        ("mom_fn", t_mom),
        ("_ns5_gram", t_ns),
        ("apply_fn", t_apply),
        ("—— 端到端刷新", t_whole),
    ):
        print(f"{'':<20}{name:<18}{t:>9.1f}")
    total += t_whole
    del gs, ps, vs, G, P, V, U, Vn, Uflat, X
    mx.clear_cache()

print(f"\n两组刷新合计 {total:.1f}ms  峰值内存 {mx.get_peak_memory() / 2**30:.2f} GB")
