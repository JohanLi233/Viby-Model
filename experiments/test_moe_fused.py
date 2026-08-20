"""fused MoE 训练 kernel 数值 A/B：model/moe_fused.py vs 旧逐组 padded GEMM
稀疏路径（MoEFeedForward._sparse_forward 的 env 回退分支）。

覆盖用例（真实形状 E=112/K=6/I=104/D=768 与小形状 E=16/K=3/I=26/D=64）：
  1. 均衡路由（随机 top-k）
  2. 集中路由（~50% pair 压向单一专家）
  3. 溢出（容量表故意调小 → trash 行路径）
  4. 默认容量（不预热容量表，走首微批 1.5× 均值分支）
  5. 多实例：同一次前向里两个不同容量表的 MoE 调用，验证每微批
     custom_function 实例的 vjp 闭包不串线
比较前向输出与 x/gate_up_w/down_w 的梯度（bf16 噪声容忍）。

用法：.venv/bin/python experiments/test_moe_fused.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.model import MoEFeedForward, VibyConfig

FAIL = 0


def build_moe(E, K, I, D, group=8):
    cfg = VibyConfig(
        hidden_size=D,
        num_hidden_layers=1,
        num_attention_heads=1,
        n_routed_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        n_shared_experts=0,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        hrm_H_cycles=2,
        hrm_L_cycles=3,
        hrm_cycle_router=0,
    )
    m = MoEFeedForward(cfg)
    m._SPARSE_GROUP = group
    return m


def make_idx_w(B, T, K, E, concentrate=None, seed=0):
    mx.random.seed(seed)
    M = B * T
    scores = mx.random.normal((M, E))
    idx = mx.argsort(scores, axis=-1)[:, -K:].astype(mx.int32)  # 每 token K 个不同专家
    if concentrate is not None:  # 前 concentrate 列全部压向专家 3
        idx[:, :concentrate] = 3
    w = mx.random.uniform(0.2, 1.0, (M, K)).astype(mx.bfloat16)
    return idx.reshape(B, T, K), w.reshape(B, T, K)


def prime_caps(m, idx, scale_num=5, scale_den=4, extra=64, flat=None):
    E, _K = m.n_routed, m.top_k
    G = idx.size
    counts = mx.zeros((E,), dtype=mx.int32).at[idx.reshape(-1)].add(1).tolist()
    EG = min(m._SPARSE_GROUP, E)
    n_groups = (E + EG - 1) // EG
    caps = []
    for gi in range(n_groups):
        e0, e1 = gi * EG, min(gi * EG + EG, E)
        cm = max(counts[e0:e1])
        base = cm if flat is None else flat
        c = ((base * scale_num // scale_den + extra) + 127) // 128 * 128
        caps.append(max(c, 128))
    m._cap_table[-1] = caps
    m._cap_G[-1] = G
    return caps


def run_case(name, m, x0, gu0, dw0, idx, w, check_dense_ref=None):
    global FAIL
    C = mx.random.normal((x0.shape[0], x0.shape[1], x0.shape[2])) * 0.5

    def loss_fn(x, gu, dw):
        m.experts.gate_up_w = gu
        m.experts.down_w = dw
        out = m._sparse_forward(x, idx, w, step_idx=None)
        return (out.astype(mx.float32) * C).sum()

    vg = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))
    MoEFeedForward._FUSED_DISABLED = True
    val_ref, (gx_r, ggu_r, gdw_r) = vg(x0, gu0, dw0)
    out_ref = m._sparse_forward(x0, idx, w, step_idx=None)
    mx.eval(val_ref, gx_r, ggu_r, gdw_r, out_ref)
    m._pending_counts.clear()
    MoEFeedForward._FUSED_DISABLED = False
    val_new, (gx_n, ggu_n, gdw_n) = vg(x0, gu0, dw0)
    out_new = m._sparse_forward(x0, idx, w, step_idx=None)
    mx.eval(val_new, gx_n, ggu_n, gdw_n, out_new)
    m._pending_counts.clear()

    def rel(a, b):
        d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
        n = b.astype(mx.float32).abs().max().item()
        return d / max(n, 1e-6)

    # 主度量用 out 张量 maxdiff/max|out|：标量 val 在 |val|≈0 时 rel 虚高
    # （bf16 噪声经 78 万元素随机正负抵消后，|val| 可任意小）
    rows = [
        ("out", rel(out_new, out_ref)),
        ("grad_x", rel(gx_n, gx_r)),
        ("grad_gate_up_w", rel(ggu_n, ggu_r)),
        ("grad_down_w", rel(gdw_n, gdw_r)),
    ]
    ok = all(v < 5e-2 for _, v in rows) and rows[0][1] < 2e-2
    FAIL += 0 if ok else 1
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: "
        + ", ".join(f"{k}={v:.2e}" for k, v in rows)
    )
    if check_dense_ref is not None:
        # f32 稠密参考：两条 bf16 路径都应落在它的噪声范围内
        MoEFeedForward._FUSED_DISABLED = False
        xf = x0.astype(mx.float32)
        out32 = m._dense_forward(xf, idx, w.astype(mx.float32))
        MoEFeedForward._FUSED_DISABLED = True
        m.experts.gate_up_w = gu0
        m.experts.down_w = dw0
        out_ref = m._sparse_forward(x0, idx, w, step_idx=None)
        MoEFeedForward._FUSED_DISABLED = False
        m.experts.gate_up_w = gu0
        m.experts.down_w = dw0
        out_new = m._sparse_forward(x0, idx, w, step_idx=None)
        mx.eval(out32, out_ref, out_new)
        d_ref = (out_ref.astype(mx.float32) - out32).abs().max().item()
        d_new = (out_new.astype(mx.float32) - out32).abs().max().item()
        scale = out32.abs().max().item()
        print(
            f"       vs f32 dense: old={d_ref / scale:.2e} fused={d_new / scale:.2e} (相对 max|out|)"
        )
        m._pending_counts.clear()


def run_multi_instance(m, x0, gu0, dw0, idxA, wA, idxB, wB, capsA, capsB):
    """同一前向内两个不同容量表的 MoE 调用：验证 vjp 闭包各自捕获本批
    caps/starts，不串线（梯度与旧路径逐位可比）。"""
    global FAIL
    G = idxA.size
    C1 = mx.random.normal(x0.shape) * 0.5
    C2 = mx.random.normal(x0.shape) * 0.3

    def loss2(x):
        m.experts.gate_up_w = gu0
        m.experts.down_w = dw0
        m._cap_table[-1] = capsA
        m._cap_G[-1] = G
        o1 = m._sparse_forward(x, idxA, wA, step_idx=None)
        m._cap_table[-1] = capsB
        o2 = m._sparse_forward(x * 1.7, idxB, wB, step_idx=None)
        return (o1.astype(mx.float32) * C1).sum() + (o2.astype(mx.float32) * C2).sum()

    vg = mx.value_and_grad(loss2)
    MoEFeedForward._FUSED_DISABLED = True
    v_r, g_r = vg(x0)
    mx.eval(v_r, g_r)
    m._pending_counts.clear()
    MoEFeedForward._FUSED_DISABLED = False
    v_n, g_n = vg(x0)
    mx.eval(v_n, g_n)
    m._pending_counts.clear()

    def fwd_outs(dis):
        MoEFeedForward._FUSED_DISABLED = dis
        m.experts.gate_up_w = gu0
        m.experts.down_w = dw0
        m._cap_table[-1] = capsA
        m._cap_G[-1] = G
        o1 = m._sparse_forward(x0, idxA, wA, step_idx=None)
        m._cap_table[-1] = capsB
        o2 = m._sparse_forward(x0 * 1.7, idxB, wB, step_idx=None)
        mx.eval(o1, o2)
        m._pending_counts.clear()
        return o1, o2

    o1r, o2r = fwd_outs(True)
    o1n, o2n = fwd_outs(False)

    def rel(a, b):
        d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
        n = b.astype(mx.float32).abs().max().item()
        return d / max(n, 1e-6)

    relg = rel(g_n, g_r)
    relo = max(rel(o1n, o1r), rel(o2n, o2r))
    ok = relg < 5e-2 and relo < 2e-2
    FAIL += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] 多实例闭包: out={relo:.2e}, grad_x={relg:.2e}")


def main():
    # ---------------- 真实形状 ----------------
    B, T, D, I, E, K = 2, 512, 768, 104, 112, 6
    m = build_moe(E, K, I, D)
    mx.random.seed(42)
    x0 = (mx.random.normal((B, T, D)) * 0.5).astype(mx.bfloat16)
    gu0 = m.experts.gate_up_w.astype(mx.bfloat16)
    dw0 = m.experts.down_w.astype(mx.bfloat16)

    idx, w = make_idx_w(B, T, K, E, seed=1)
    prime_caps(m, idx)
    run_case("真实形状/均衡", m, x0, gu0, dw0, idx, w, check_dense_ref=True)

    idx_c, w_c = make_idx_w(B, T, K, E, concentrate=3, seed=2)
    prime_caps(m, idx_c)
    run_case("真实形状/集中(50%→专家3)", m, x0, gu0, dw0, idx_c, w_c)

    prime_caps(m, idx_c, flat=110)  # 容量远小于计数 → 大量溢出走 trash 行
    run_case("真实形状/溢出trash", m, x0, gu0, dw0, idx_c, w_c)

    m._cap_table.clear()
    m._cap_G.clear()
    run_case("真实形状/默认容量(首微批)", m, x0, gu0, dw0, idx, w)

    # 多实例：A 用均衡容量、B 用大容量
    capsA = prime_caps(m, idx)
    capsB = [c * 2 for c in capsA]
    idxB, wB = make_idx_w(B, T, K, E, seed=3)
    run_multi_instance(m, x0, gu0, dw0, idx, w, idxB, wB, capsA, capsB)

    # ---------------- 小形状（非 256 整除，走 guard 分支） ----------------
    B2, T2, D2, I2, E2, K2 = 2, 64, 64, 26, 16, 3
    m2 = build_moe(E2, K2, I2, D2)
    x2 = (mx.random.normal((B2, T2, D2)) * 0.5).astype(mx.bfloat16)
    gu2 = m2.experts.gate_up_w.astype(mx.bfloat16)
    dw2 = m2.experts.down_w.astype(mx.bfloat16)
    idx2, w2 = make_idx_w(B2, T2, K2, E2, seed=4)
    prime_caps(m2, idx2)
    run_case("小形状/均衡", m2, x2, gu2, dw2, idx2, w2, check_dense_ref=True)
    idx2c, w2c = make_idx_w(B2, T2, K2, E2, concentrate=2, seed=5)
    prime_caps(m2, idx2c)
    run_case("小形状/集中", m2, x2, gu2, dw2, idx2c, w2c, check_dense_ref=True)

    print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 个用例失败"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
