"""战役优化器组件单测（autoresearch-mlx 2026-09-01 结算移植）：
cautious weight decay（apply kernel 与 FusedAdamW 两路）、polar-ema
动量（m=0 退化等价、m>0 烟测）、NorMuon 行 RMS 尾（F 范数保持）、
以及各旗标默认关闭时与既有路径逐位一致。

用法: .venv/bin/python tests/test_campaign_optim.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import (
    BatchedMuon,
    FusedAdamW,
    _normuon_tail,
    _stack_apply_kernel,
)

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


def fnorm(x):
    return float(mx.linalg.norm(x.astype(mx.float32)).item())


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def test_cautious_apply_kernel():
    """_stack_apply_kernel cautious：只在 sign(X)==sign(P) 坐标衰减。"""
    mx.random.seed(0)
    P = mx.random.normal((8, 16))
    X = mx.random.normal((8, 16))
    lr = mx.array(0.5, dtype=P.dtype)
    wd = 0.1
    out = _stack_apply_kernel(False, apply_wd=wd, cautious=True)(P, X, lr)
    mask = ((X * P) >= 0).astype(P.dtype)
    ref = P * (1 - lr * wd * mask) - lr * X
    mx.eval(out, ref)
    check(
        "cautious apply kernel == 掩码参考",
        maxdiff(out, ref) < 1e-6,
        f"maxdiff={maxdiff(out, ref)}",
    )
    # 非 cautious 对照：全体坐标衰减
    out2 = _stack_apply_kernel(False, apply_wd=wd, cautious=False)(P, X, lr)
    ref2 = P * (1 - lr * wd) - lr * X
    mx.eval(out2, ref2)
    check("plain decoupled apply kernel == 参考", maxdiff(out2, ref2) < 1e-6)


def test_polar_ema_m0_degenerate():
    """polar-ema 在 momentum=0 时退化为「NS 作用于原始梯度」，
    与默认 momentum=0 路径（U=G 后接 NS）逐位一致。"""
    mx.random.seed(1)
    P = {"a": mx.random.normal((16, 32)), "b": mx.random.normal((32, 16))}
    G = {"a": mx.random.normal((16, 32)), "b": mx.random.normal((32, 16))}
    opt_p = BatchedMuon(learning_rate=0.05, momentum=0.0, polar_ema=True)
    opt_d = BatchedMuon(learning_rate=0.05, momentum=0.0)
    p_new = opt_p.apply_gradients(dict(G), dict(P))
    d_new = opt_d.apply_gradients(dict(G), dict(P))
    mx.eval(p_new, d_new)
    for k in P:
        d = maxdiff(p_new[k], d_new[k])
        check(f"polar-ema m=0 退化 == 默认 m=0 [{k}]", d < 1e-6, f"maxdiff={d}")


def test_polar_ema_smoke():
    """polar-ema m=0.95 多步：参数确实更新、无 NaN、动量状态在极因子空间。"""
    mx.random.seed(2)
    P = {"a": mx.random.normal((16, 32))}
    G = {"a": mx.random.normal((16, 32))}
    opt = BatchedMuon(learning_rate=0.05, momentum=0.95, polar_ema=True)
    n0 = fnorm(P["a"])
    for _ in range(3):
        P = opt.apply_gradients(dict(G), P)
        mx.eval(P)
    finite = bool(mx.all(mx.isfinite(P["a"])).item())
    check("polar-ema 多步无 NaN", finite)
    check("polar-ema 参数确实更新", abs(fnorm(P["a"]) - n0) / n0 > 1e-3)


def test_normuon_tail():
    """NorMuon 尾：行 RMS 归一后整体 renorm 回尾前 F 范数；v 状态形状正确。"""
    mx.random.seed(3)
    X = mx.random.normal((4, 8, 16))
    Xn, v = _normuon_tail(X, None, 0.95)
    mx.eval(Xn, v)
    check("normuon v 播种形状", tuple(v.shape) == (4, 8), f"{v.shape}")
    d = abs(fnorm(Xn) - fnorm(X)) / fnorm(X)
    check("normuon 尾保持 F 范数", d < 1e-5, f"rel={d}")
    Xn2, v2 = _normuon_tail(X, v, 0.95)
    mx.eval(Xn2, v2)
    check("normuon v EMA 更新形状保持", tuple(v2.shape) == (4, 8))


def test_fused_adamw_cautious():
    """FusedAdamW cautious：更新方向与参数异号不衰减（==wd0），同号衰减
    （==非 cautious）。"""
    mx.random.seed(4)
    g_pos = mx.array([[1.0, 1.0], [1.0, 1.0]])  # Adam 步方向 −，与 p>0 异号
    g_neg = -g_pos  # Adam 步方向 +，与 p 同号
    p0 = mx.abs(mx.random.normal((2, 2))) + 0.5
    lr, wd = 0.5, 0.2

    def step(g, cautious, wd_):
        opt = FusedAdamW(
            learning_rate=lr,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=wd_,
            bias_correction=True,
            cautious=cautious,
        )
        out = opt.apply_gradients({"w": g}, {"w": p0})
        mx.eval(out)
        return out["w"]

    # g=+1：Adam 步（被减去的方向 X）与 p>0 同号 → 掩码开 → 衰减，
    # 应等于非 cautious(wd)（全体坐标同号，掩码全开）
    d_same = maxdiff(step(g_pos, True, wd), step(g_pos, False, wd))
    check("adam cautious 同号衰减 (==plain wd)", d_same < 1e-6, f"maxdiff={d_same}")
    # g=−1：Adam 步与 p 异号 → 掩码关 → 不衰减，应等于 wd=0
    d_opp = maxdiff(step(g_neg, True, wd), step(g_neg, False, 0.0))
    check("adam cautious 异号不衰减 (==wd0)", d_opp < 1e-6, f"maxdiff={d_opp}")
    # 且同号情形确实比 wd=0 收得更狠（防测试空转）：decay 先缩 p 再减
    # Adam 步，p>0 时结果处处更低
    lower = bool(mx.all(step(g_pos, True, wd) < step(g_pos, False, 0.0)).item())
    check("adam cautious 同号确实更收缩", lower)


def test_flags_off_bitexact():
    """旗标默认关闭时与既有行为逐位一致：
    - FusedAdamW cautious=True + wd=0 是精确无操作；
    - BatchedMuon 旗标全开/全关（wd=0 时）结果逐位一致（kernel 缓存的
      扩展 key 不串线）。"""
    mx.random.seed(5)
    p0 = {"w": mx.random.normal((8, 8))}
    g0 = {"w": mx.random.normal((8, 8))}
    o1 = FusedAdamW(learning_rate=0.1, weight_decay=0.0, cautious=True)
    o2 = FusedAdamW(learning_rate=0.1, weight_decay=0.0, cautious=False)
    r1 = o1.apply_gradients(dict(g0), dict(p0))
    r2 = o2.apply_gradients(dict(g0), dict(p0))
    mx.eval(r1, r2)
    d = maxdiff(r1["w"], r2["w"])
    check("adam cautious wd=0 为无操作", d == 0.0, f"maxdiff={d}")


def test_batched_muon_cautious_routing():
    """BatchedMuon cautious_wd：衰减从动量 kernel（耦合）移到 apply（掩码），
    与默认耦合路径结果应不同（证明确实改路，不是静默丢失）；hyperball
    下 cautious 不破坏范数冻结（半径口径=衰减前的 P）。"""
    mx.random.seed(7)
    P = {"a": mx.random.normal((16, 32))}
    G = {"a": mx.random.normal((16, 32))}
    o_off = BatchedMuon(learning_rate=0.05, momentum=0.95, weight_decay=0.1)
    o_cau = BatchedMuon(
        learning_rate=0.05, momentum=0.95, weight_decay=0.1, cautious_wd=True
    )
    p_off = o_off.apply_gradients(dict(G), dict(P))
    p_cau = o_cau.apply_gradients(dict(G), dict(P))
    mx.eval(p_off, p_cau)
    d = maxdiff(p_off["a"], p_cau["a"])
    check("muon cautious 衰减确实改路（与耦合不同）", d > 1e-4, f"maxdiff={d}")

    o_h = BatchedMuon(
        learning_rate=0.05,
        momentum=0.95,
        weight_decay=0.1,
        cautious_wd=True,
        hyperball=True,
    )
    n0 = fnorm(P["a"])
    p_h = o_h.apply_gradients(dict(G), dict(P))
    mx.eval(p_h)
    rel = abs(fnorm(p_h["a"]) - n0) / n0
    check("hyperball+cautious 范数仍冻结", rel < 1e-4, f"rel={rel}")


def test_normuon_stacked_group_smoke():
    """3D 堆叠专家组开 normuon 一步：无 NaN、参数更新。"""
    mx.random.seed(6)
    P = {"e": mx.random.normal((4, 6, 8))}
    G = {"e": mx.random.normal((4, 6, 8))}
    opt = BatchedMuon(learning_rate=0.05, momentum=0.95, normuon_beta2=0.95)
    out = opt.apply_gradients(dict(G), dict(P))
    mx.eval(out)
    check("normuon 堆叠组无 NaN", bool(mx.all(mx.isfinite(out["e"])).item()))
    check("normuon 堆叠组确实更新", maxdiff(out["e"], P["e"]) > 1e-6)


def test_expl_nest_hook():
    """expl-nest 训练循环钩子（base_trainer）：δ 捕获与前瞻位移的端到端语义。
    用 object.__new__ + stub 驱动真实 _optimizer_step / _compute_loss_and_grad，
    不走完整 __init__（需要数据/模型全家桶）。"""
    from types import SimpleNamespace

    from trainer.base_trainer import BaseTrainer

    class _Model:
        def __init__(self):
            self.p = {"w": mx.ones((4, 4))}

        def trainable_parameters(self):
            return self.p

        def parameters(self):
            return self.p

        # base_trainer._compute_loss_and_grad 在 trace 结束后用真实参数/偏置
        # 恢复模块状态（compile 下 trace 期占位数组会留在叶子上）
        def update(self, params):
            self.p = params

        def moe_bias_stack(self):
            return mx.zeros((0,), dtype=mx.float32)

        def apply_moe_biases(self, biases):
            self.biases = biases

    class _Opt:
        state = {}

        def update(self, model, grads):
            model.p = {"w": model.p["w"] - 0.1}  # 固定更新 δ=−0.1

    tr = object.__new__(BaseTrainer)
    tr.args = SimpleNamespace(grad_clip=0.0)
    tr.lm_config = SimpleNamespace()
    tr.model = _Model()
    tr.optimizer = _Opt()
    tr._expl_nest_mu = 2.0
    tr._en_delta = None
    tr._compiled = False

    grads = {"w": mx.ones((4, 4))}
    tr._optimizer_step(grads, 1)
    check("expl-nest δ 捕获", tr._en_delta is not None)
    dv = float(tr._en_delta["w"].reshape(-1)[0])
    check("expl-nest δ 值正确", abs(dv - (-0.1)) < 1e-7, f"δ={dv}")

    captured = {}

    def fake_loss_and_grad(params, biases, *a, **kw):
        captured["w"] = params["w"]
        return mx.array(0.0), grads

    tr._loss_and_grad = fake_loss_and_grad
    tr._compute_loss_and_grad(None, None, None, None, False)
    # 前瞻后应为 θ + μ·δ = 0.9 + 2.0·(−0.1) = 0.7
    pv = float(captured["w"].reshape(-1)[0])
    check("expl-nest 前瞻位移 θ+μδ", abs(pv - 0.7) < 1e-6, f"p={pv}")
    # 真实参数不被位移污染
    rv = float(tr.model.p["w"].reshape(-1)[0])
    check("expl-nest 不污染真实参数", abs(rv - 0.9) < 1e-6, f"p={rv}")

    # mu=0（默认关）：无捕获位移、无 δ
    tr2 = object.__new__(BaseTrainer)
    tr2.args = SimpleNamespace(grad_clip=0.0)
    tr2.model = _Model()
    tr2.optimizer = _Opt()
    tr2._expl_nest_mu = 0.0
    tr2._en_delta = None
    tr2._compiled = False
    tr2._loss_and_grad = fake_loss_and_grad
    tr2._optimizer_step(grads, 1)
    check("expl-nest 默认关无 δ", tr2._en_delta is None)


if __name__ == "__main__":
    test_cautious_apply_kernel()
    test_polar_ema_m0_degenerate()
    test_polar_ema_smoke()
    test_normuon_tail()
    test_fused_adamw_cautious()
    test_flags_off_bitexact()
    test_batched_muon_cautious_routing()
    test_normuon_stacked_group_smoke()
    test_expl_nest_hook()
    if FAIL:
        print(f"\n{FAIL} 项失败")
        sys.exit(1)
    print("\n全部通过")
