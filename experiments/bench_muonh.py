"""MuonH 优化器单步开销基准：1080M 口径（8 层 + MTP，E=256，d=384，I=384，
bf16）下 apply_gradients 各变体耗时，定位 MuonH 变慢的来源并量化加速
旋钮（bf16 NS / 专家 NS 降步 / 专家留 AdamW）。

变体：
  base            现状默认：2D 矩阵 Muon（无投影、无专家）
  hb              + 范数球投影（只 2D）
  exp_f32         + 专家逐专家 NS（f32，ns5）
  exp_bf16        + 专家 NS bf16
  exp_bf16_ns3    + 专家 NS 3 步
（NS 降频复用 / Temporal Q 变体已随机制删除——r082 归因：EVERY=8 早期
损失 0.4-0.5 nat，负面结果见 BatchedMuon 类 docstring）

用法: .venv/bin/python experiments/bench_muonh.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from trainer.muon import BatchedMuon

# 1080M 每层 Muon 组 2D 形状（attention + shared FFN + latent 投影）
LAYER_2D = [
    (1232, 768),  # qkv_proj（MLA 合并；这里不按 segment_map 分段，量级不变）
    (1536, 192),  # kv_up_proj
    (768, 768),  # o_proj
    (384, 768),  # shared gate/up（I=384）
    (384, 768),
    (768, 384),  # shared down
    (384, 768),  # lat_down
    (768, 384),  # lat_up
]
N_CALLS = 9  # 8 层 + 1 MTP
EXPERT_GU = (256, 768, 384)  # (E, 2I, d)
EXPERT_DW = (256, 384, 768)  # (E, d, 2I)


def build_params(with_experts):
    mx.random.seed(0)
    params, grads = {}, {}
    for i in range(N_CALLS):
        for j, (r, c) in enumerate(LAYER_2D):
            params[f"l{i}.w{j}"] = mx.random.normal((r, c)).astype(mx.bfloat16)
            grads[f"l{i}.w{j}"] = mx.random.normal((r, c)).astype(mx.bfloat16)
        if with_experts:
            for name, shape in (("gu", EXPERT_GU), ("dw", EXPERT_DW)):
                params[f"l{i}.{name}"] = mx.random.normal(shape).astype(mx.bfloat16)
                grads[f"l{i}.{name}"] = mx.random.normal(shape).astype(mx.bfloat16)
    mx.eval(params, grads)
    return params, grads


def timed(opt, params, grads, iters):
    # 预热（编译 + state init）
    out = opt.apply_gradients(dict(grads), dict(params))
    mx.eval(out)
    ts = []
    for _ in range(iters):
        t0 = time.time()
        out = opt.apply_gradients(dict(grads), dict(params))
        mx.eval(out)
        ts.append(time.time() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    variants = [
        ("base(2D only)", False, dict(hyperball=False)),
        ("hb(2D+投影)", False, dict(hyperball=True)),
        ("exp_f32_ns5", True, dict(hyperball=True, ns_bf16=False)),
        ("exp_bf16_ns5", True, dict(hyperball=True, ns_bf16=True)),
        ("exp_bf16_ns3", True, dict(hyperball=True, ns_bf16=True, stack_ns_steps=3)),
    ]
    params_x, grads_x = build_params(with_experts=True)
    params_2d = {k: v for k, v in params_x.items() if ".gu" not in k and ".dw" not in k}
    grads_2d = {k: v for k, v in grads_x.items() if ".gu" not in k and ".dw" not in k}
    print(
        f"Muon 组参数：2D {len(params_2d)} 个 + 专家栈 {len(params_x) - len(params_2d)} 个"
    )
    for name, with_exp, kw in variants:
        p, g = (params_x, grads_x) if with_exp else (params_2d, grads_2d)
        opt = BatchedMuon(learning_rate=0.01, **kw)
        mn, avg = timed(opt, p, g, iters)
        print(f"{name:18s} {mn * 1e3:7.1f}/{avg * 1e3:7.1f} ms (min/mean)")


if __name__ == "__main__":
    main()
