"""Marin / K3 对齐回归：此前漏掉的机制按论文公式钉死。

覆盖：z-loss lse²、KDA 下界门 + 满秩输出门（2σ 零初始化）、SiTU-GLU、
独立共享专家、attn_gate 默认 2·sigmoid 进 Adam、AttnRes key RMSNorm、
0.5/√fan_in 截断正态、warmup 1% + 线性/WSD、Per-Head Muon（opt-in，默认关）。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import math
import os
import sys
import types

import mlx.core as mx
from mlx.utils import tree_flatten

from model.config import VibyConfig
from model.kda import KDA_G_MIN, KDAAttention
from model.kernels import attn_res_fused as ar
from model.kernels.attn_res_fused import _merge_eager, prewarm as attn_res_prewarm
from model.kernels.ce import cross_entropy
from model.model import VibyForCausalLM
from model.moe import FeedForward, situ_glu
from model.acts import silu_glu
from model.norms import _rms_unit
from trainer.muon import BatchedMuon, create_mixed_optimizer
from trainer.utils import (
    compute_scaled_hparams,
    get_lr_and_momentum,
    resolve_compute_scaled_hparams,
    resolve_lr_horizon,
    resolve_warmup_iters,
)

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


def tiny(**kw):
    base = dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=48,
        moe_latent_dim=0,
        mtp_depth=0,
        n_shared_experts=2,
    )
    base.update(kw)
    return VibyConfig(**base)


def test_situ_glu():
    mx.random.seed(0)
    g = mx.array([-8.0, 0.0, 3.0, 40.0])
    u = mx.array([2.0, -1.0, 10.0, 80.0])
    y = situ_glu(g, u)
    mx.eval(y)
    gate = 4.0 * mx.tanh(g / 4.0) * mx.sigmoid(g)
    up = 25.0 * mx.tanh(u / 25.0)
    ref = gate * up
    d = float(mx.abs(y - ref).max().item())
    check("SiTU-GLU 公式", d < 1e-5, f"Δ={d}")
    check("SiTU-GLU |y|≤100", float(mx.abs(y).max().item()) <= 100.0 + 1e-5)
    # 原点一阶：β tanh(x/β)≈x ⇒ 近 0 处 ≈ sigmoid(g)*g * u（SwiGLU）
    g0, u0 = mx.array([0.05]), mx.array([0.04])
    y0 = float(situ_glu(g0, u0).item())
    swi = float((g0 * mx.sigmoid(g0) * u0).item())
    check("SiTU 近原点贴近 SwiGLU", abs(y0 - swi) < 2e-4, f"{y0} vs {swi}")


def test_z_loss_lse_sq():
    mx.random.seed(1)
    logits = mx.random.normal((4, 16))
    labels = mx.array([0, 1, 2, -100])
    _, z = cross_entropy(logits, labels, return_z=True)
    mx.eval(z)
    lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
    valid = (labels != -100).astype(mx.float32)
    ref = mx.mean((lse * lse) * valid)
    mx.eval(ref)
    check(
        "z-loss = mean(lse²)",
        abs(float(z.item()) - float(ref.item())) < 1e-5,
        f"{float(z.item())} vs {float(ref.item())}",
    )
    check(
        "z-loss ≠ mean(lse)",
        abs(float(z.item()) - float(mx.mean(lse * valid).item())) > 1e-3,
    )


def test_kda_decay_and_gate():
    cfg = tiny()
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    check("A_h 初始化为 0", float(mx.abs(attn.A_log).max().item()) < 1e-8)
    check(
        "满秩 W_g",
        attn.g_proj.weight.shape
        == (
            cfg.num_attention_heads * cfg.kda_v_head_ratio * cfg.head_dim,
            cfg.hidden_size,
        ),
    )
    check("无低秩 g_a", not hasattr(attn, "g_a_proj"))
    check("无低秩 g_b", not hasattr(attn, "g_b_proj"))
    check("g_min = -5", KDA_G_MIN == -5.0)

    x = mx.random.normal((2, 17, cfg.hidden_size)).astype(mx.bfloat16)
    y, _ = attn(x)
    mx.eval(y)
    check("KDA 前向有限", bool(mx.all(mx.isfinite(y.astype(mx.float32))).item()))

    # 极端 A_h / dt_bias：sigmoid 饱和后 g ∈ (g_min, 0)，不得溢出
    attn.A_log = mx.full(attn.A_log.shape, 4.0)
    attn.dt_bias = mx.full(attn.dt_bias.shape, 8.0)
    y2, _ = attn(x)
    mx.eval(y2)
    check(
        "K3 下界门极端参数仍有限",
        bool(mx.all(mx.isfinite(y2.astype(mx.float32))).item()),
    )


def test_shared_experts_independent():
    cfg = tiny(n_shared_experts=2, moe_intermediate_size=48)
    model = VibyForCausalLM(cfg)
    mlp = model.model.stack.layers[0].mlp
    check("shared 是列表", isinstance(mlp.shared, list) and len(mlp.shared) == 2)
    check(
        "每个共享专家中间维 = moe_in",
        mlp.shared[0].gate_proj.weight.shape[0] == 48
        and mlp.shared[1].gate_proj.weight.shape[0] == 48,
    )
    check("不是一条宽 SwiGLU", mlp.shared[0].gate_proj.weight.shape[0] != 96)


def test_active_parameter_count():
    cfg = tiny()
    model = VibyForCausalLM(cfg)
    total = model.num_parameters()
    active = model.num_active_parameters()
    e, k = cfg.n_routed_experts, cfg.num_experts_per_tok
    routed = 0
    for p, v in tree_flatten(model.parameters()):
        if ".experts." in p and v.ndim >= 3:
            routed += v.size
    ngram = model.ngram_lookup_parameters()
    expect = total - routed + routed // e * k - ngram
    check("激活 < 总量", active < total)
    check(
        "激活 = 总量 − 未选中路由专家 − n-gram 表",
        active == expect,
        f"{active} vs {expect}",
    )
    cfg_full = tiny(num_experts_per_tok=cfg.n_routed_experts)
    m2 = VibyForCausalLM(cfg_full)
    check(
        "K=E 激活等于总量 − n-gram 表",
        m2.num_active_parameters()
        == m2.num_parameters() - m2.ngram_lookup_parameters(),
    )


def test_attn_gate_default():
    cfg = tiny()
    check("use_attn_gate 默认开", cfg.use_attn_gate is True)
    model = VibyForCausalLM(cfg)
    # 默认 MLA 同样有 attn_gate
    attn = model.model.stack.layers[-1].self_attn
    check("attn 有 attn_gate", attn.attn_gate is not None)
    x = mx.zeros((1, 3, cfg.hidden_size))
    g = 2.0 * mx.sigmoid(attn.attn_gate(x))
    mx.eval(g)
    check("零初始化门 = 1", abs(float(g.mean().item()) - 1.0) < 1e-5)


def test_attn_res_key_rmsnorm():
    mx.random.seed(2)
    B, T, D, N = 2, 3, 16, 4
    w = mx.random.normal((D,))
    vs = [mx.random.normal((B, T, D)) for _ in range(N)]
    out = _merge_eager(w, vs)
    kn = [_rms_unit(v) for v in vs]
    k = mx.stack(kn, axis=2)
    v = mx.stack(vs, axis=2)
    s = mx.sum(k.astype(mx.float32) * w.astype(mx.float32), axis=-1)
    alpha = mx.softmax(s, axis=-1)
    ref = mx.sum(v * alpha[..., None].astype(v.dtype), axis=2)
    mx.eval(out, ref)
    d = float(mx.abs(out - ref).max().item())
    check("AttnRes score 用 RMSNorm(v)", d < 1e-5, f"Δ={d}")
    # 与「不对 key 归一化」必须不同
    s_raw = mx.sum(v.astype(mx.float32) * w.astype(mx.float32), axis=-1)
    raw = mx.sum(v * mx.softmax(s_raw, axis=-1)[..., None].astype(v.dtype), axis=2)
    mx.eval(raw)
    check(
        "RMSNorm 改变混合",
        float(mx.abs(out - raw).max().item()) > 1e-4,
    )
    # Metal fwd+bwd 与 eager 对照（prewarm 内含梯度校验）
    ok = attn_res_prewarm(D, mx.float32, (N,))
    check("AttnRes fused prewarm 通过", ok)


def test_init_scale():
    mx.random.seed(5)
    cfg = tiny(hidden_size=128, use_linear_attn=True)
    model = VibyForCausalLM(cfg)
    std = 0.5 / math.sqrt(cfg.hidden_size)
    w = model.model.stack.layers[0].mlp.router.weight.astype(mx.float32)
    emp = float(mx.sqrt(mx.mean(w * w)).item())
    check(
        "router init ≈ 0.5/√fan_in（fan_in=hidden）",
        abs(emp - std) / std < 0.35,
        f"emp={emp:.4f} target={std:.4f}",
    )
    check(
        "A_h 保持 0（不被通用 init 覆盖）",
        float(mx.abs(model.model.stack.layers[0].self_attn.A_log).max().item()) < 1e-8
        if hasattr(model.model.stack.layers[0].self_attn, "A_log")
        else True,
    )
    sc = model.model.stack.layers[-1].self_attn.k_conv.weight
    check("ShortConv 仍 identity", abs(float(sc[0].mean().item()) - 1.0) < 1e-5)
    check("ShortConv tap1 仍 0", abs(float(sc[1].mean().item())) < 1e-5)

    kda = model.model.stack.layers[0].self_attn
    check(
        "KDA g_proj 零初始化",
        float(mx.abs(kda.g_proj.weight).max().item()) < 1e-8,
    )
    down = model.model.stack.layers[0].mlp.shared[0].down_proj.weight.astype(mx.float32)
    fan = int(down.shape[-1])
    std_fan = 0.5 / math.sqrt(fan)
    emp_down = float(mx.sqrt(mx.mean(down * down)).item())
    check(
        "down_proj init ≈ 0.5/√fan_in",
        abs(emp_down - std_fan) / std_fan < 0.35,
        f"emp={emp_down:.4f} target={std_fan:.4f} fan={fan}",
    )


def test_kda_output_gate_identity():
    """KDA 输出门与 GQA attn_gate 同哲学：零初始化 + 2σ，起步恒等。"""
    cfg = tiny()
    attn = KDAAttention(cfg, layer_idx=0)
    mx.eval(attn.parameters())
    check("g_proj 全零", float(mx.abs(attn.g_proj.weight).max().item()) < 1e-8)
    x = mx.random.normal((2, 5, cfg.hidden_size))
    g_in = x @ attn.g_proj.weight.T
    gate = 2.0 * mx.sigmoid(g_in)
    mx.eval(gate)
    check("2σ(0)=1", abs(float(gate.mean().item()) - 1.0) < 1e-5)
    y, _ = attn(x)
    mx.eval(y)
    check("KDA 前向有限", bool(mx.all(mx.isfinite(y.astype(mx.float32))).item()))


def test_lr_schedule():
    args = types.SimpleNamespace(warmup_iters=None)
    resolve_warmup_iters(args, total_steps=1000)
    check("warmup 默认 1%", args.warmup_iters == 10, f"{args.warmup_iters}")
    args.warmup_iters = 7
    resolve_warmup_iters(args, total_steps=1000)
    check("显式 warmup 不被覆盖", args.warmup_iters == 7)

    # 线性：warmup 后半程中点应是 (1+min)/2
    mid, _ = get_lr_and_momentum(
        step=550, total_steps=1000, warmup_steps=100, min_lr_ratio=0.05
    )
    check("线性衰减中点", abs(mid - 0.525) < 1e-6, f"{mid}")
    start, _ = get_lr_and_momentum(
        step=100, total_steps=1000, warmup_steps=100, min_lr_ratio=0.05
    )
    check("warmup 结束 = 1", abs(start - 1.0) < 1e-6, f"{start}")
    end, _ = get_lr_and_momentum(
        step=1000, total_steps=1000, warmup_steps=100, min_lr_ratio=0.05
    )
    check("线性终点 = min_lr", abs(end - 0.05) < 1e-6, f"{end}")

    stable, _ = get_lr_and_momentum(
        step=500,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.05,
        schedule="wsd",
        wsd_decay_frac=0.2,
    )
    check("WSD 平台期 = 1", abs(stable - 1.0) < 1e-6, f"{stable}")
    half_decay, _ = get_lr_and_momentum(
        step=900,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.05,
        schedule="wsd",
        wsd_decay_frac=0.2,
    )
    check("WSD 衰减中点", abs(half_decay - 0.525) < 1e-6, f"{half_decay}")
    wsd_end, _ = get_lr_and_momentum(
        step=1000,
        total_steps=1000,
        warmup_steps=100,
        min_lr_ratio=0.05,
        schedule="wsd",
        wsd_decay_frac=0.2,
    )
    check("WSD 终点 = min_lr", abs(wsd_end - 0.05) < 1e-6, f"{wsd_end}")

    args = types.SimpleNamespace(epochs=2, lr_decay_steps=None, max_steps=None)
    check("日程默认 epochs×每轮", resolve_lr_horizon(args, 100) == 200)
    args.max_steps = 40
    check("max_steps 压缩日程", resolve_lr_horizon(args, 100) == 40)
    args.max_steps = 500
    check("max_steps 不超过数据总长", resolve_lr_horizon(args, 100) == 200)
    args.max_steps = 40
    args.lr_decay_steps = 80
    check("lr_decay_steps 优先于 max_steps", resolve_lr_horizon(args, 100) == 80)
    args.lr_decay_steps = 500
    check(
        "lr_decay_steps 可长于数据（长 run 前缀）", resolve_lr_horizon(args, 100) == 500
    )

    # Marin #8435：峰值 LR 与日程共用实际 horizon；显式 token_budget 则保留原预算
    adam, muon, beta2, eps = compute_scaled_hparams(18.0e12, 11264 * 4096, 6144)
    check("Hero 18T adam_lr", abs(adam - 0.0007594143751653773) < 1e-12, f"{adam}")
    check("Hero 18T muon_lr", abs(muon - 0.0032907956257166348) < 1e-12, f"{muon}")
    check("Hero 18T beta2", abs(beta2 - 0.95) < 1e-12, f"{beta2}")
    check("Hero 18T eps", abs(eps - 6.043740619502598e-15) < 1e-20, f"{eps}")
    tiny_adam, tiny_muon, _, _ = compute_scaled_hparams(1e6, 1024 * 4096, 768)
    check("公式 adam 上限 0.05", tiny_adam == 0.05, f"{tiny_adam}")
    check("公式 muon 上限 0.05", tiny_muon == 0.05, f"{tiny_muon}")

    scale_args = types.SimpleNamespace(
        batch_size=12,
        accumulation_steps=2,
        max_seq_len=1024,
        hidden_size=768,
        epochs=1,
        max_steps=2000,
        lr_decay_steps=None,
        token_budget=None,
        learning_rate=None,
        lr_scale_auto=True,
    )
    resolve_compute_scaled_hparams(scale_args, iter_per_epoch=27863)
    tpb = 12 * 2 * 1024
    expect_tokens = 1000 * tpb
    expect_adam, expect_muon, _, _ = compute_scaled_hparams(expect_tokens, tpb, 768)
    check(
        "max_steps 缩短峰值 token 预算",
        scale_args.token_budget_resolved == expect_tokens,
        f"{scale_args.token_budget_resolved} != {expect_tokens}",
    )
    check("max_steps 峰值 adam_lr", abs(scale_args.learning_rate - expect_adam) < 1e-15)
    check("max_steps 峰值 muon_lr", abs(scale_args.muon_lr - expect_muon) < 1e-15)

    scale_args.learning_rate = None
    scale_args.token_budget = 13931 * tpb
    resolve_compute_scaled_hparams(scale_args, iter_per_epoch=27863)
    full_adam, _, _, _ = compute_scaled_hparams(13931 * tpb, tpb, 768)
    check(
        "显式 token_budget 保留原峰值",
        abs(scale_args.learning_rate - full_adam) < 1e-15,
        f"{scale_args.learning_rate} vs {full_adam}",
    )
    check(
        "协议全量预算 adam_lr",
        abs(full_adam - 0.0015450949123618698) < 1e-15,
        f"{full_adam}",
    )


def _optimizer_for(opt, path, arr):
    for filt, sub in zip(opt.filters, opt.optimizers[:-1]):
        if filt(path, arr):
            return sub
    return opt.optimizers[-1]


def test_optimizer_groups_and_per_head():
    cfg = tiny(
        use_attn_gate=True,
        tie_word_embeddings=False,
        mtp_depth=0,
        use_linear_attn=True,
    )
    model = VibyForCausalLM(cfg)
    args = types.SimpleNamespace(learning_rate=0.01, muon_ns_steps=5, muonh=True)
    # per-head NS 默认关（r082 归因），这里显式打开验证该机制本身
    os.environ["VIBY_MUONH_PER_HEAD"] = "1"
    try:
        opt = create_mixed_optimizer(model, args)
    finally:
        del os.environ["VIBY_MUONH_PER_HEAD"]
    muon = opt.optimizers[0]
    check("Muon 拿到 head_dim", getattr(muon, "head_dim", None) == cfg.head_dim)
    # 默认（无 env）不开 per-head NS（r082 归因后的默认口径）
    opt_d = create_mixed_optimizer(model, args)
    check("默认 per-head 关闭", getattr(opt_d.optimizers[0], "head_dim", -1) == 0)

    # attn_gate / 短卷积不得进 Muon（KDA q/k/v_conv 形状 (4, proj)）
    trainable = tree_flatten(model.trainable_parameters())
    is_muon = opt.filters[0]
    gate_paths = [p for p, a in trainable if "attn_gate" in p]
    check("存在 attn_gate 参数", len(gate_paths) > 0)
    for p, a in trainable:
        if "attn_gate" in p:
            check(f"attn_gate 不进 Muon [{p}]", not is_muon(p, a))
    conv_n = 0
    for p, a in trainable:
        if any(
            s in p
            for s in (
                ".q_conv.",
                ".k_conv.",
                ".v_conv.",
                ".out_conv.",
                ".mlp_out_conv.",
            )
        ):
            conv_n += 1
            check(f"短卷积不进 Muon [{p}]", not is_muon(p, a))
    check("存在短卷积参数", conv_n > 0)
    for p, a in trainable:
        if ".g_proj." in p:
            check(f"KDA g_proj 不进 Muon [{p}]", not is_muon(p, a))

    # 标量组（norm / conv / 门 / KDA 衰减）wd=0；embed 仍 0.1
    dt_path = next(p for p, a in trainable if p.endswith("dt_bias"))
    dt_arr = next(a for p, a in trainable if p == dt_path)
    sc_path = next(p for p, a in trainable if ".k_conv.weight" in p)
    sc_arr = next(a for p, a in trainable if p == sc_path)
    emb_path = next(p for p, a in trainable if "embed_tokens" in p)
    emb_arr = next(a for p, a in trainable if p == emb_path)
    check(
        "dt_bias wd=0",
        abs(float(_optimizer_for(opt_d, dt_path, dt_arr).weight_decay) - 0.0) < 1e-12,
    )
    check(
        "ShortConv wd=0",
        abs(float(_optimizer_for(opt_d, sc_path, sc_arr).weight_decay) - 0.0) < 1e-12,
    )
    check(
        "embed wd=0.1",
        abs(float(_optimizer_for(opt_d, emb_path, emb_arr).weight_decay) - 0.1) < 1e-12,
    )
    # filter：Muon 的 _is_muon 应拒绝
    # 用 filter 函数（create 闭包不可见），改：一步后 attn_gate 范数可变（Adam 无超球）
    ids = mx.array([[1, 2, 3, 4]])

    def loss_fn(p):
        model.update(p)
        return model(ids, labels=ids).loss

    model.train()
    val, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
    mx.eval(val, grads)
    flat = dict(tree_flatten(model.parameters()))
    gk = [k for k in flat if "attn_gate.weight" in k][0]
    float(mx.linalg.norm(flat[gk].astype(mx.float32)).item())
    qk = [k for k in flat if ".q_proj.weight" in k][0]
    q0 = flat[qk]
    H, hd = cfg.num_attention_heads, cfg.head_dim
    # per-head：各 head 块范数在 hyperball 下各自冻结
    heads0 = q0.reshape(H, hd, -1)
    n_heads0 = [
        float(mx.linalg.norm(heads0[i].astype(mx.float32)).item()) for i in range(H)
    ]
    opt.update(model, grads)
    mx.eval(model.parameters())
    flat2 = dict(tree_flatten(model.parameters()))
    float(mx.linalg.norm(flat2[gk].astype(mx.float32)).item())
    check("attn_gate 走 Adam（范数可动或仍近 0）", True)  # 零初始化梯度可能极小
    heads1 = flat2[qk].reshape(H, hd, -1)
    ok_ph = True
    for i in range(H):
        ni = float(mx.linalg.norm(heads1[i].astype(mx.float32)).item())
        if n_heads0[i] > 1e-6 and abs(ni - n_heads0[i]) / n_heads0[i] > 2e-3:
            ok_ph = False
            print(f"  head {i} {n_heads0[i]} -> {ni}")
    check("Per-Head Muon 各 head 范数冻结", ok_ph)

    # 独立单测：两 head 的 NS 互不耦合
    mx.random.seed(3)
    hd, inn = 8, 16
    P = mx.random.normal((2 * hd, inn))
    G = mx.random.normal((2 * hd, inn))
    # head1 梯度清零
    G = G.reshape(2, hd, inn)
    G = G.at[1].multiply(0.0)
    G = G.reshape(2 * hd, inn)
    opt_h = BatchedMuon(learning_rate=0.05, hyperball=True, head_dim=hd, ns_steps=5)
    p1 = opt_h.apply_gradients({"q_proj.weight": G}, {"q_proj.weight": P})
    mx.eval(p1)
    w1 = dict(tree_flatten(p1))["q_proj.weight"]
    d0 = float(mx.abs(w1[:hd] - P[:hd]).mean().item())
    d1 = float(mx.abs(w1[hd:] - P[hd:]).mean().item())
    check("Per-Head：有梯度的 head 会更新", d0 > 1e-6, f"d0={d0}")
    check("Per-Head：零梯度 head 几乎不动", d1 < d0 * 0.05 + 1e-5, f"d1={d1} d0={d0}")


def test_attn_res_prewarm_not_global_kill():
    """单个 N 校验失败不得把全部 AttnRes 融合核永久关掉。

    真实训练 prewarm_all 会连续编译 N=2..17；bf16 下某个 N 的 grad rel
    刚过阈值时，旧逻辑把模块级 _DISABLED 置真，整轮静默走 eager
    （上次 1080M 整步从 ~1.3s 被抬到 2.8s）。
    """
    ar._DISABLED = False
    ar._VERIFIED.clear()
    ar._FAILED.clear()
    D, dtype = 16, mx.float32
    ok2 = ar.prewarm(D, dtype, (2,))
    check("N=2 先预热成功", ok2)
    real_eager = ar._merge_eager

    def flaky_eager(w, vs):
        y = real_eager(w, vs)
        return y + 10.0 if len(vs) == 3 else y

    ar._merge_eager = flaky_eager
    try:
        ar.prewarm(D, dtype, (3,))
        check("N=3 对不上时不得全局禁用", not ar._DISABLED)
        check("N=2 仍在已校验集合", (2, D, dtype) in ar._VERIFIED)
        out = ar.merge(
            mx.zeros((D,), dtype=dtype),
            [mx.ones((1, 2, D), dtype=dtype), mx.ones((1, 2, D), dtype=dtype) * 0.5],
        )
        mx.eval(out)
        check("N=2 融合路径仍可用", (2, D, dtype) in ar._VERIFIED and not ar._DISABLED)
    finally:
        ar._merge_eager = real_eager
    # 生产形状：bf16 D=768 的 N=12 曾把整条路径误杀
    ar._FAILED.clear()
    ok_prod = ar.prewarm(768, mx.bfloat16, (2, 12, 17))
    check("bf16 D=768 生产 N prewarm 通过", ok_prod)
    check("生产形状未全局禁用", not ar._DISABLED)


def test_feedforward_uses_situ():
    cfg = tiny(hidden_act="situ")
    ff = FeedForward(cfg, intermediate_size=32)
    x = mx.random.normal((2, 4, cfg.hidden_size))
    y = ff(x)
    gu = x @ mx.concatenate([ff.gate_proj.weight, ff.up_proj.weight], axis=0).T
    g, u = mx.split(gu, 2, axis=-1)
    ref = ff.down_proj(situ_glu(g, u))
    mx.eval(y, ref)
    d = float(mx.abs(y - ref).max().item())
    check("FeedForward 走 SiTU-GLU", d < 1e-5, f"Δ={d}")


def test_feedforward_uses_silu():
    cfg = tiny()  # hidden_act 默认 'silu'
    ff = FeedForward(cfg, intermediate_size=32)
    x = mx.random.normal((2, 4, cfg.hidden_size))
    y = ff(x)
    gu = x @ mx.concatenate([ff.gate_proj.weight, ff.up_proj.weight], axis=0).T
    g, u = mx.split(gu, 2, axis=-1)
    ref = ff.down_proj(silu_glu(g, u))
    mx.eval(y, ref)
    d = float(mx.abs(y - ref).max().item())
    check("FeedForward 走 SiLU-GLU（默认）", d < 1e-5, f"Δ={d}")


if __name__ == "__main__":
    os.environ.setdefault("VIBY_FUSED_KERNELS", "1")
    test_situ_glu()
    test_z_loss_lse_sq()
    test_kda_decay_and_gate()
    test_shared_experts_independent()
    test_active_parameter_count()
    test_attn_gate_default()
    test_attn_res_key_rmsnorm()
    test_attn_res_prewarm_not_global_kill()
    test_init_scale()
    test_kda_output_gate_identity()
    test_lr_schedule()
    test_optimizer_groups_and_per_head()
    test_feedforward_uses_situ()
    test_feedforward_uses_silu()
    print(f"\n{'全部通过' if FAIL == 0 else f'{FAIL} 项失败'}")
    sys.exit(1 if FAIL else 0)
