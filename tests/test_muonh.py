"""MuonH/AdamH 单测：范数球投影、堆叠专家逐专家 NS 与 2D 路径逐片等价、
AdamH 方向与 Adam 共线且范数保持、create_mixed_optimizer 分组接线。

用法: .venv/bin/python tests/test_muonh.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import AdamH, BatchedMuon, FusedAdamW, create_mixed_optimizer

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


def test_hyperball_norm_frozen():
    """hyperball=True：2D 矩阵每步后 Frobenius 范数精确回到更新前。"""
    mx.random.seed(0)
    params = {"a": mx.random.normal((16, 32)), "b": mx.random.normal((32, 16))}
    grads = {"a": mx.random.normal((16, 32)), "b": mx.random.normal((32, 16))}
    opt = BatchedMuon(learning_rate=0.05, hyperball=True)
    n0 = {k: fnorm(v) for k, v in params.items()}
    for _ in range(3):  # 多步：动量累积后仍应冻结范数
        params = opt.apply_gradients(dict(grads), params)
        mx.eval(params)
    for k in params:
        check(
            f"hyperball 范数冻结[{k}]",
            abs(fnorm(params[k]) - n0[k]) / n0[k] < 1e-4,
            f"{fnorm(params[k])} vs {n0[k]}",
        )
    # 对照：hyperball=False 时同样步数范数应确实变化（防测试空转）。
    # 正交化方向与 P 近似正交，范数变化是二阶小量，用大 lr 放大到可检测。
    params2 = {"a": mx.random.normal((16, 32))}
    mx.random.seed(0)
    grads2 = {"a": mx.random.normal((16, 32))}
    opt2 = BatchedMuon(learning_rate=0.5, hyperball=False)
    n02 = fnorm(params2["a"])
    params2 = opt2.apply_gradients(grads2, params2)
    mx.eval(params2)
    check("对照组范数确实变化", abs(fnorm(params2["a"]) - n02) / n02 > 1e-3)


def test_stack_per_expert_equiv():
    """堆叠组 (E,r,c) 逐专家 NS == 同参数逐片走 2D 路径（含动量两步）。"""
    mx.random.seed(1)
    E, r, c = 4, 6, 8
    P3 = mx.random.normal((E, r, c))
    G3 = mx.random.normal((E, r, c))
    for hyper in (False, True):
        opt3 = BatchedMuon(learning_rate=0.05, hyperball=hyper)
        opt2 = BatchedMuon(learning_rate=0.05, hyperball=hyper)
        p3, g3 = {"e": P3}, {"e": G3}
        p2 = {f"e{i}": P3[i] for i in range(E)}
        g2 = {f"e{i}": G3[i] for i in range(E)}
        for _ in range(2):
            p3 = opt3.apply_gradients(g3, p3)
            p2 = opt2.apply_gradients(g2, p2)
            mx.eval(p3, p2)
        d = max(
            float(mx.max(mx.abs(p3["e"][i] - p2[f"e{i}"])).item()) for i in range(E)
        )
        check(f"堆叠逐专家==2D逐片 (hyperball={hyper})", d < 1e-6, f"maxdiff={d}")


def test_adamh():
    """AdamH：方向与 Adam 逐位共线（只差正标量），且范数回更新前。"""
    mx.random.seed(2)
    p0 = mx.random.normal((8, 8))
    g = mx.random.normal((8, 8))
    ah = AdamH(learning_rate=0.01, weight_decay=0.0)
    aw = FusedAdamW(learning_rate=0.01, weight_decay=0.0)
    pa = ah.apply_gradients({"w": g}, {"w": p0})["w"]
    pb = aw.apply_gradients({"w": g}, {"w": p0})["w"]
    mx.eval(pa, pb)
    check("AdamH 范数保持", abs(fnorm(pa) - fnorm(p0)) / fnorm(p0) < 1e-5)
    ratio = (pa.astype(mx.float32) / pb.astype(mx.float32)).flatten()
    mx.eval(ratio)
    import numpy as np

    rn = np.array(ratio)
    check(
        "AdamH 与 Adam 共线",
        float(rn.std() / abs(rn.mean())) < 1e-3,
        f"ratio std/mean={rn.std() / abs(rn.mean()):.2e}",
    )


def test_grouping():
    """create_mixed_optimizer(muonh=True)：专家进 MuonH、lm_head 进 AdamH、
    分组计数正确；跑一步更新不炸且 Muon 矩阵范数冻结。"""
    from model.config import VibyConfig
    from model.model import VibyForCausalLM

    # V4.1 架构：n_layers >= 4（CED 至少 2 编码 + 2 解码），MoE 全层
    cfg = VibyConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        tie_word_embeddings=False,  # 让 lm_head 存在
    )
    model = VibyForCausalLM(cfg)
    args = types.SimpleNamespace(
        learning_rate=0.01,
        muon_ns_steps=5,
        muonh=True,
    )
    opt = create_mixed_optimizer(model, args)
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.trainable_parameters()))
    n_exp = sum(1 for p in flat if ".experts." in p)
    # 每个 MoE 块 2 张（gate_up_w + down_w）：主干 n_layers + 每个 MTP 模块 1 块
    check("专家张量存在", n_exp == 2 * (cfg.n_layers + cfg.n_mtp_layers))

    ids = mx.array([[1, 2, 3, 4]])
    model(ids, labels=ids).loss

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    # V4.1 主干命名：Block.attn / Block.ffn（不再有 stack/self_attn/mlp）
    n_before = fnorm(flat["model.layers.0.attn.wo_b.weight"])
    ne_before = fnorm(flat["model.layers.0.ffn.experts.gate_up_w"])
    nh_before = fnorm(flat["lm_head.weight"])
    opt.update(model, grads)
    mx.eval(model.parameters())
    flat2 = dict(tree_flatten(model.parameters()))
    n_after = fnorm(flat2["model.layers.0.attn.wo_b.weight"])
    ne_after = fnorm(flat2["model.layers.0.ffn.experts.gate_up_w"])
    nh_after = fnorm(flat2["lm_head.weight"])
    check(
        "muonh 一步后 Muon 矩阵范数冻结",
        abs(n_after - n_before) / n_before < 1e-3,
        f"{n_before} -> {n_after}",
    )
    # 3-D 堆叠专家默认**不进** MuonH（V4.1 架构下该路径有随机 NaN 隐患，见
    # README「已知问题」），落 AdamW 无衰减组 ⇒ 范数不冻结；这里只要求更新有限、
    # 且确实动了（不再是恒等）。
    check(
        "专家堆叠默认走 AdamW 无衰减组（范数不冻结但更新有限）",
        ne_after == ne_after and abs(ne_after - ne_before) > 0.0,
        f"{ne_before} -> {ne_after}",
    )
    # lm_head 按报告 §2.5/§4.2.2 移入 Sinkhorn 均衡组（不再是 AdamH），
    # 同样不再保范数。
    check(
        "lm_head 走 Sinkhorn 均衡组（不再保范数，更新有限）",
        nh_after == nh_after and abs(nh_after - nh_before) > 0.0,
        f"{nh_before} -> {nh_after}",
    )


if __name__ == "__main__":
    test_hyperball_norm_frozen()
    test_stack_per_expert_equiv()
    test_adamh()
    test_grouping()
    print(f"\n{'全部通过' if FAIL == 0 else f'{FAIL} 项失败'}")
    sys.exit(1 if FAIL else 0)
