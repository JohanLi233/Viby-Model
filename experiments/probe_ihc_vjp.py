"""iHC 读写 VJP 归因（B12 T1024 D768 M=4 bf16）。

假设：write 把 (B,T,D) 扩到 (B,T,M,D) 会跳过流轴，走 MLX 通用广播
VJP（KDA A_log 同类，单项曾 5ms/层）。对照：

  A 现状：delta[..., None, :] * h_post[..., None]
  B 末轴广播再转置（不跳轴）
  C 流布局改成 (B,T,D,M)，读写都沿最后一轴
  D 16 次 sublayer 链（8 层 × 2）拼出整栈残差成本
  E 1 层真实 VibyBlock f+b：register vs iHC

用法: .venv/bin/python experiments/probe_ihc_vjp.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from model.block import VibyBlock
from model.config import VibyConfig
from model.ihc import IHCGate, ihc_expand

B, T, D, M = 12, 1024, 768, 4
DT = mx.bfloat16
N_SUB = 16
BYTES_R = B * T * M * D * 2
BYTES_X = B * T * D * 2


def _time(fn, reps=9, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0] * 1e3, ts[len(ts) // 2] * 1e3


def _cfg(**kw):
    base = dict(
        hidden_size=D,
        num_hidden_layers=1,
        num_attention_heads=8,
        head_dim=96,
        vocab_size=256,
        max_position_embeddings=T,
        kv_lora_rank=192,
        qk_rope_head_dim=32,
        mtp_depth=0,
        n_routed_experts=32,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        moe_latent_dim=0,
        ngram_table_size=0,
        use_linear_attn=False,
        attn_res_window=4,
    )
    base.update(kw)
    return VibyConfig(**base)


def write_a(r, delta, h_post):
    return r + h_post[..., None] * delta[..., None, :]


def write_b(r, delta, h_post):
    b, t, d = delta.shape
    m = int(h_post.shape[-1])
    delta_s = mx.broadcast_to(delta[..., None], (b, t, d, m)).transpose(0, 1, 3, 2)
    return r + h_post[..., None] * delta_s


def write_c(r_dm, delta, h_post):
    """r: (B,T,D,M)。"""
    b, t, d = delta.shape
    m = int(h_post.shape[-1])
    h = mx.broadcast_to(h_post[..., None], (b, t, m, d)).transpose(0, 1, 3, 2)
    return r_dm + h * delta[..., None]


def read_a(r, h_pre):
    return (h_pre[..., None] * r).sum(axis=2)


def read_c(r_dm, h_pre):
    return (r_dm * h_pre[:, :, None, :]).sum(axis=-1)


def collapse_a(r):
    return mx.mean(r, axis=-2)


def collapse_b(r):
    return mx.mean(r.transpose(0, 1, 3, 2), axis=-1)


def expand_a(h, m):
    return ihc_expand(h, m)


def expand_b(h, m):
    b, t, d = h.shape
    return mx.broadcast_to(h[..., None], (b, t, d, m)).transpose(0, 1, 3, 2)


def _one(label, loss_fn, *xs):
    vg = mx.value_and_grad(loss_fn, argnums=tuple(range(len(xs))))

    def fwd():
        return loss_fn(*xs)

    def both():
        return vg(*xs)

    fmin, fmed = _time(fwd)
    tmin, tmed = _time(both)
    print(
        f"  {label:<28} fwd {fmed:6.2f}ms  f+b {tmed:6.2f}ms  "
        f"bwd {max(tmed - fmed, 0):6.2f}ms"
    )
    return tmed


def probe_ops():
    print(f"== 单次读写（B{B} T{T} D{D} M{M} bf16，R={BYTES_R / 1e6:.1f}MB）==")
    mx.random.seed(0)
    r = (mx.random.normal((B, T, M, D)) * 0.5).astype(DT)
    r_dm = r.transpose(0, 1, 3, 2)
    delta = (mx.random.normal((B, T, D)) * 0.5).astype(DT)
    h_pre = mx.sigmoid(mx.random.normal((B, T, M))).astype(DT)
    h_post = (2.0 * mx.sigmoid(mx.random.normal((B, T, M)))).astype(DT)
    h = (mx.random.normal((B, T, D)) * 0.5).astype(DT)
    mx.eval(r, r_dm, delta, h_pre, h_post, h)

    def la(r_, d_, hp):
        return write_a(r_, d_, hp).astype(mx.float32).square().sum()

    def lb(r_, d_, hp):
        return write_b(r_, d_, hp).astype(mx.float32).square().sum()

    def lc(rdm, d_, hp):
        return write_c(rdm, d_, hp).astype(mx.float32).square().sum()

    print("-- write --")
    _one("A 跳轴广播 (现状)", la, r, delta, h_post)
    _one("B 末轴广播+转置", lb, r, delta, h_post)
    _one("C 布局 (B,T,D,M)", lc, r_dm, delta, h_post)

    def ra(r_, hp):
        return read_a(r_, hp).astype(mx.float32).square().sum()

    def rc(rdm, hp):
        return read_c(rdm, hp).astype(mx.float32).square().sum()

    print("-- read --")
    _one("A sum 中轴 (现状)", ra, r, h_pre)
    _one("C sum 末轴", rc, r_dm, h_pre)

    def ca(r_):
        return collapse_a(r_).astype(mx.float32).square().sum()

    def cb(r_):
        return collapse_b(r_).astype(mx.float32).square().sum()

    print("-- collapse --")
    _one("A mean 中轴 (现状)", ca, r)
    _one("B 转置+mean 末轴", cb, r)

    def ea(h_):
        return expand_a(h_, M).astype(mx.float32).square().sum()

    def eb(h_):
        return expand_b(h_, M).astype(mx.float32).square().sum()

    print("-- expand --")
    _one("A broadcast 中轴 (现状)", ea, h)
    _one("B 末轴广播+转置", eb, h)


def _gate_step_a(gate, r, scale):
    h_pre, h_post = gate.gates(r)
    x = gate.read(r, h_pre)
    delta = x * scale
    return gate.write(r, delta, h_post)


def _gate_step_b(gate, r, scale):
    h_pre, h_post = gate.gates(r)
    x = gate.read(r, h_pre)
    delta = x * scale
    return write_b(r, delta, h_post)


def probe_chain():
    print(f"\n== {N_SUB} 次 gate+read+write 链（假 Δ=scale·x̃）==")
    mx.random.seed(1)
    gate = IHCGate(D, M)
    gate.eval()
    r0 = (mx.random.normal((B, T, M, D)) * 0.5).astype(DT)
    scale = mx.array(0.1, dtype=DT)
    mx.eval(r0, scale, *dict(tree_flatten(gate.parameters())).values())

    def chain_a(r):
        for _ in range(N_SUB):
            r = _gate_step_a(gate, r, scale)
        return r.astype(mx.float32).square().sum()

    def chain_b(r):
        for _ in range(N_SUB):
            r = _gate_step_b(gate, r, scale)
        return r.astype(mx.float32).square().sum()

    _one(f"A 现状 ×{N_SUB}", chain_a, r0)
    _one(f"B write 末轴 ×{N_SUB}", chain_b, r0)

    vg_a = mx.compile(mx.value_and_grad(chain_a))
    vg_b = mx.compile(mx.value_and_grad(chain_b))
    mx.eval(vg_a(r0))
    mx.eval(vg_b(r0))

    def ca():
        return vg_a(r0)

    def cb():
        return vg_b(r0)

    _, ta = _time(ca, reps=7, warmup=2)
    _, tb = _time(cb, reps=7, warmup=2)
    print(f"  compile A                      f+b {ta:6.2f}ms")
    print(f"  compile B                      f+b {tb:6.2f}ms")


def probe_block():
    print("\n== 1 层真实 VibyBlock f+b（B12 T256，含 MLA+MoE）==")
    bt, tt = 12, 256
    mx.random.seed(2)
    x = (mx.random.normal((bt, tt, D)) * 0.5).astype(DT)
    mx.eval(x)

    def _run(label, **kw):
        block = VibyBlock(_cfg(**kw), layer_idx=0)
        block.update(
            tree_map(
                lambda a: a.astype(DT) if mx.issubdtype(a.dtype, mx.floating) else a,
                block.parameters(),
            )
        )
        block.train()
        mx.eval(block.parameters())
        residuals = [] if kw.get("attn_res_register") or kw.get("ihc") else [x]

        def loss(h):
            y, _ = block(h, residuals=list(residuals), mask_is_full=True)
            return y.astype(mx.float32).square().sum()

        vg = mx.value_and_grad(loss)
        mx.eval(vg(x))

        def both():
            return vg(x)

        _, tmed = _time(both, reps=5, warmup=2)
        print(f"  {label:<28} f+b {tmed:6.2f}ms")

    _run("register", attn_res_register=True)
    _run("ihc", ihc=True, ihc_streams=4)


def main():
    os.environ.setdefault("MLX_METAL_DEBUG", "0")
    probe_ops()
    probe_chain()
    probe_block()


if __name__ == "__main__":
    main()
