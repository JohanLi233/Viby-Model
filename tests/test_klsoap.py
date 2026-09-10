"""KL-SOAP-H 单测：KL 因子累积对参考、特征基正交性、hyperball 范数冻结、
零初始化防钉死、3D 堆叠专家烟测、create_mixed_optimizer 接线（VIBY_KLSOAP）。

用法: .venv/bin/python tests/test_klsoap.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten

from trainer.klsoap import KLSoaPH

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


def get_state(opt, path):
    return {k[len(path) + 1:]: v for k, v in tree_flatten(opt.state) if k.startswith(path + ".")}


def test_factor_accumulation_reference():
    """首步（Q=I、eig=1）：L = (1−β_kron)/c·GGᵀ，R = (1−β_kron)/r·GᵀG。"""
    mx.random.seed(0)
    P = {"a": mx.random.normal((8, 16))}
    G = {"a": mx.random.normal((8, 16))}
    opt = KLSoaPH(learning_rate=0.05, hyperball=False, weight_decay=0.0)
    opt.apply_gradients(dict(G), dict(P))
    st = get_state(opt, "a")
    bk = 0.95
    ref_L = (1 - bk) / 16 * (G["a"].astype(mx.float32) @ G["a"].astype(mx.float32).T)
    ref_R = (1 - bk) / 8 * (G["a"].astype(mx.float32).T @ G["a"].astype(mx.float32))
    mx.eval(st, ref_L, ref_R)
    d = maxdiff(st["L"], ref_L)
    check("KL 因子 L 首步 == 参考", d < 1e-5, f"maxdiff={d}")
    d = maxdiff(st["R"], ref_R)
    check("KL 因子 R 首步 == 参考", d < 1e-5, f"maxdiff={d}")


def test_basis_orthonormal():
    """多步 QR 基更新后特征基保持正交：QᵀQ ≈ I。"""
    mx.random.seed(1)
    P = {"a": mx.random.normal((16, 32))}
    opt = KLSoaPH(learning_rate=0.05, hyperball=False, weight_decay=0.0)
    for _ in range(5):
        G = {"a": mx.random.normal((16, 32))}
        P = opt.apply_gradients(G, P)
        mx.eval(P)
    st = get_state(opt, "a")
    for key, n in (("QL", 16), ("QR", 32)):
        QtQ = st[key].T @ st[key]
        I = mx.eye(n)
        d = maxdiff(QtQ, I)
        check(f"特征基正交[{key}]", d < 1e-4, f"maxdiff={d}")
    check("eigL 非负", bool(mx.all(st["eigL"] >= 0).item()))


def test_hyperball_norm_frozen():
    """hyperball=True：多步后逐矩阵 F 范数回初值（含 cautious wd）。"""
    mx.random.seed(2)
    P = {"a": mx.random.normal((16, 32)), "b": mx.random.normal((32, 16))}
    n0 = {k: fnorm(v) for k, v in P.items()}
    opt = KLSoaPH(learning_rate=0.05, hyperball=True, weight_decay=0.1, cautious=True)
    for _ in range(3):
        G = {k: mx.random.normal(v.shape) for k, v in P.items()}
        P = opt.apply_gradients(G, P)
        mx.eval(P)
    for k in P:
        rel = abs(fnorm(P[k]) - n0[k]) / n0[k]
        check(f"klsoap-h 范数冻结[{k}]", rel < 1e-3, f"rel={rel}")


def test_zero_init_not_pinned():
    """零初始化矩阵：n0==0 当步跳过投影，范数由首个更新建立（不钉零）。"""
    P = {"a": mx.zeros((8, 16))}
    mx.random.seed(3)
    G = {"a": mx.random.normal((8, 16))}
    opt = KLSoaPH(learning_rate=0.05, hyperball=True)
    P = opt.apply_gradients(G, P)
    mx.eval(P)
    n = fnorm(P["a"])
    check("零初始化不被钉死", n > 1e-6, f"norm={n}")


def test_stacked_experts_smoke():
    """3D 堆叠专家 (E,r,c)：多步无 NaN、参数更新、逐专家范数冻结。"""
    mx.random.seed(4)
    P = {"e": mx.random.normal((4, 6, 8))}
    n0 = [fnorm(P["e"][i]) for i in range(4)]
    opt = KLSoaPH(learning_rate=0.05, hyperball=True, weight_decay=0.1)
    for _ in range(3):
        G = {"e": mx.random.normal((4, 6, 8))}
        P = opt.apply_gradients(G, P)
        mx.eval(P)
    check("堆叠专家无 NaN", bool(mx.all(mx.isfinite(P["e"])).item()))
    for i in range(4):
        rel = abs(fnorm(P["e"][i]) - n0[i]) / n0[i]
        check(f"专家{i}范数冻结", rel < 1e-3, f"rel={rel}")


def test_wiring():
    """create_mixed_optimizer：VIBY_KLSOAP=1 时 muon 组换成 KLSoaPH，
    默认（未设 env）仍是 BatchedMuon；klsoap 下真实模型跑一步不炸。"""
    from model.config import VibyConfig
    from model.model import VibyForCausalLM
    from trainer.muon import BatchedMuon, create_mixed_optimizer

    # V4.1 架构：n_layers >= 4（CED 至少 2 编码 + 2 解码）
    cfg = VibyConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        tie_word_embeddings=False,
    )
    args = types.SimpleNamespace(learning_rate=0.01, muon_ns_steps=5, muonh=True)

    model = VibyForCausalLM(cfg)
    os.environ.pop("VIBY_KLSOAP", None)
    opt0 = create_mixed_optimizer(model, args)
    check("默认 muon 组仍是 BatchedMuon", isinstance(opt0.optimizers[0], BatchedMuon))

    model2 = VibyForCausalLM(cfg)
    os.environ["VIBY_KLSOAP"] = "1"
    try:
        opt1 = create_mixed_optimizer(model2, args)
    finally:
        os.environ.pop("VIBY_KLSOAP", None)
    check("VIBY_KLSOAP=1 接管 muon 组", isinstance(opt1.optimizers[0], KLSoaPH))

    ids = mx.array([[1, 2, 3, 4]])

    def loss_fn(p):
        model2.update(p)
        return model2(ids, labels=ids).loss

    val, grads = mx.value_and_grad(loss_fn)(model2.trainable_parameters())
    mx.eval(val, grads)
    opt1.update(model2, grads)
    mx.eval(model2.parameters())
    finite = all(
        bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(model2.parameters())
    )
    check("klsoap 真实模型一步后参数有限", finite)


if __name__ == "__main__":
    test_factor_accumulation_reference()
    test_basis_orthonormal()
    test_hyperball_norm_frozen()
    test_zero_init_not_pinned()
    test_stacked_experts_smoke()
    test_wiring()
    print(f"\n{'全部通过' if FAIL == 0 else f'{FAIL} 项失败'}")
    sys.exit(1 if FAIL else 0)
