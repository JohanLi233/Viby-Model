"""MoE 分发的内部分段计时（bf16，真实形状 B=4 T=1024 D=1024 E=96 K=6 I=256）。

MoE 现在占整步 44.7%（桩掉它 → 10,839 tok/s），按 FLOPs 折算专家 GEMM 只跑出
~5 TFLOPS（稠密 bf16 上限 12~14），所以要定位到底是 GEMM、元数据、还是散射慢。
"""

import sys
import time

sys.path.insert(0, "/Users/lizhonghan/Desktop/Viby-Model")
import mlx.core as mx
from model.config import VibyConfig
from model.moe import MoEFeedForward, expert_act
from model.kernels.moe_dispatch import (
    enabled_for,
    gather_gate_up,
    down_project_routes,
    route_metadata,
)
from trainer.utils import convert_model_dtype


def bench(fn, n=4, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0] * 1000, ts[len(ts) // 2] * 1000


cfg = VibyConfig(n_mtp_layers=0)
moe = MoEFeedForward(cfg, 6)
moe.train()
model = mx.zeros((1,), mx.float32)  # 只为 dtype 转换占位
B, T, D = 4, 1024, cfg.dim
E, K, I = cfg.n_routed_experts, cfg.n_activated_experts, cfg.moe_inter_dim
x = mx.random.normal((B, T, D)).astype(mx.bfloat16)
convert_model_dtype(moe, "bfloat16")
mx.eval(x, moe.parameters())
M, G = B * T, B * T * K
gu = moe.experts.gate_up_w.astype(mx.bfloat16)
dw = moe.experts.down_w.astype(mx.bfloat16)
idx = (mx.arange(M)[:, None] + mx.arange(K)[None, :] * (E // K)) % E
idx = idx.astype(mx.int32)
w, _, _ = moe.router(x.reshape(M, D))
flat = idx.reshape(G)
order = mx.argsort(mx.stop_gradient(flat))
exps_s = flat[order].astype(mx.int32)
meta = route_metadata(order, exps_s, E)
h = gather_gate_up(x.reshape(M, D), gu, order, exps_s, meta, K)
gate, up = mx.split(h, 2, axis=-1)
act = expert_act(gate, up, cfg.swiglu_limit)
mx.eval(w, order, exps_s, meta, h, act)

fwd_flops = (2 * G * D * 2 * I + 2 * G * I * D) / 1e9
print(
    "MoE E=%d K=%d I=%d D=%d G=%d  bf16  enabled=%s  专家 GEMM fwd %.1f GFLOP"
    % (E, K, I, D, G, enabled_for(x.reshape(M, D), E, 2 * I), fwd_flops)
)
print(
    "  %-22s fwd %7.2f/%7.2f ms   (%5.1f TFLOPS fwd)"
    % (
        "整层 __call__",
        *bench(lambda: mx.eval(moe(x))),
        fwd_flops / bench(lambda: mx.eval(moe(x)))[0],
    )
)
print(
    "  %-22s      %7.2f/%7.2f ms"
    % ("  路由+topk", *bench(lambda: mx.eval(moe.router(x.reshape(M, D)))))
)
print(
    "  %-22s      %7.2f/%7.2f ms"
    % (
        "  argsort+index",
        *bench(lambda: mx.eval(mx.argsort(mx.stop_gradient(idx.reshape(G))))),
    )
)
print(
    "  %-22s      %7.2f/%7.2f ms"
    % ("  route_metadata", *bench(lambda: mx.eval(route_metadata(order, exps_s, E))))
)
print(
    "  %-22s      %7.2f/%7.2f ms   (%5.1f TFLOPS)"
    % (
        "  gather_gate_up",
        *bench(
            lambda: mx.eval(gather_gate_up(x.reshape(M, D), gu, order, exps_s, meta, K))
        ),
        fwd_flops
        * 2
        / 3
        / bench(
            lambda: mx.eval(gather_gate_up(x.reshape(M, D), gu, order, exps_s, meta, K))
        )[0],
    )
)
print(
    "  %-22s      %7.2f/%7.2f ms"
    % ("  expert_act", *bench(lambda: mx.eval(expert_act(gate, up, cfg.swiglu_limit))))
)
print(
    "  %-22s      %7.2f/%7.2f ms   (%5.1f TFLOPS)"
    % (
        "  down_project_routes",
        *bench(
            lambda: mx.eval(down_project_routes(act, dw, w.reshape(G), order, meta, K))
        ),
        fwd_flops
        / 3
        / bench(
            lambda: mx.eval(down_project_routes(act, dw, w.reshape(G), order, meta, K))
        )[0],
    )
)
