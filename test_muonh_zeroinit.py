"""MuonH hyperball 对零初始化参数的回归测试。

Bug（r081/r082 及全部历史 run 均携带）：hyperball 范数球投影把更新后矩阵
rescale 回初始 Frobenius 范数；GatedNorm.gate_up 零初始化 → 半径 0 →
参数被永久钉死在 0，门全程恒等 g≡1，gate_down 梯度恒零（P18 快照实测）。

修复：gate_up 按 Marin 口径移入 AdamW 标量组（零初始化门进 Adam）；
_stack_apply_kernel 对 n0==0 的矩阵跳过当步投影（防御未来零初始化 2D
参数再被钉零）。
"""

import types

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.muon import create_mixed_optimizer


def _tiny_model():
    cfg = VibyConfig(
        768,
        2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=384,
        vocab_size=2048,
    )
    model = VibyForCausalLM(cfg)
    model.train()
    return model, cfg


def _loss_fn(model, ids):
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return mx.mean(
        nn.losses.cross_entropy(
            logits[:, :-1].astype(mx.float32), ids[:, 1:], reduction="none"
        )
    )


def test_zero_init_gate_up_escapes_hyperball():
    args = types.SimpleNamespace(
        muonh=True,
        learning_rate=3e-3,
        muon_lr=3e-3,
        adam_lr=3e-3,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        ns_steps=5,
    )
    model, cfg = _tiny_model()
    opt = create_mixed_optimizer(model, args, "pretrain")

    mx.random.seed(7)
    ids = mx.random.randint(0, cfg.vocab_size, (4, 32))
    lg = nn.value_and_grad(model, lambda m: _loss_fn(m, ids))

    for _ in range(3):
        _, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    flat = dict(tree_flatten(model.parameters()))
    # 只看主干路径：toy loss 不经 MTP block，其参数无梯度、保持零属预期
    gu = [
        mx.linalg.norm(v).item()
        for k, v in flat.items()
        if k.endswith(".gate_up") and not k.startswith("mtp_modules")
    ]
    assert gu, "未找到 GatedNorm.gate_up 参数"
    assert all(n > 0 for n in gu), f"gate_up 仍被钉在零: {gu}"


def test_gate_down_receives_gradient_once_gate_up_nonzero():
    model, cfg = _tiny_model()
    mx.random.seed(7)
    ids = mx.random.randint(0, cfg.vocab_size, (4, 32))
    lg = nn.value_and_grad(model, lambda m: _loss_fn(m, ids))

    # gate_up ≡ 0 时 gate_down 梯度结构性地为 0（数学必然，非 bug）
    _, grads = lg(model)
    mx.eval(grads)
    flat = dict(tree_flatten(grads))
    gd0 = [mx.linalg.norm(v).item() for k, v in flat.items() if "gate_down" in k]
    assert gd0 and all(n == 0.0 for n in gd0)

    # gate_up 非零后（修复前这在训练中永远不会发生）梯度必须出现
    for _, mod in model.named_modules():
        if hasattr(mod, "gate_up"):
            mod.gate_up = mod.gate_up + 0.01 * mx.random.normal(mod.gate_up.shape)
    mx.eval(model.parameters())
    _, grads = lg(model)
    mx.eval(grads)
    flat = dict(tree_flatten(grads))
    gd1 = [
        mx.linalg.norm(v).item()
        for k, v in flat.items()
        if "gate_down" in k and not k.startswith("mtp_modules")
    ]
    assert gd1 and all(n > 0 for n in gd1), f"gate_down 梯度仍为零: {gd1}"


if __name__ == "__main__":
    test_zero_init_gate_up_escapes_hyperball()
    test_gate_down_receives_gradient_once_gate_up_nonzero()
    print("test_muonh_zeroinit: all passed")
