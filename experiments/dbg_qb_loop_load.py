"""探针：loop 模型里各 MoE gate 的负载/margin 统计到底覆盖几遍执行。

用 viby_loop 的真实配置跑一次小 batch 训练 forward，打印：
  - moe_gates() 顺序与所属层（含 loop 执行位置）
  - 每 gate last_load 总和（= 被统计的 token×K 次数，除以 B·T·K 即遍数）
  - 每 gate last_margins 行数（= 被统计的 token 数，除以 B·T 即遍数）
  - qb_margin_stats() 截断后每 gate 剩余行数
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import mlx.core as mx

from model.config import VibyConfig
from model.model import VibyForCausalLM


def main():
    meta = json.load(open("research_runs/viby_loop/pretrain_768.json"))
    cfg = VibyConfig.from_dict(meta["config"])
    cfg.ngram_table_size = 4096  # 避免 6.4GB 表分配
    model = VibyForCausalLM(cfg, skip_init=True)
    model.train()

    trunk = model.model.stack  # VibyStack
    print("exec_order:", trunk.layers and getattr(trunk, "exec_order", None))

    # gate -> 所属模块路径
    owner = {}
    for name, mod in model.named_modules() if hasattr(model, "named_modules") else []:
        pass
    # 手工建立：遍历 trunk.layers 与 mtp_modules，按对象 identity 匹配
    gates = model.moe_gates()
    layer_of = {}
    for i, blk in enumerate(trunk.layers):
        for m in blk.modules():
            if m.__class__.__name__ == "MoEGate":
                layer_of[id(m)] = f"layer{i}"
    for i, mtpm in enumerate(model.mtp_modules):
        for m in mtpm.modules():
            if m.__class__.__name__ == "MoEGate":
                layer_of[id(m)] = f"mtp{i}"

    for g in gates:
        g.collect_stats = True

    B, T = 2, 64
    x = mx.random.randint(0, cfg.vocab_size, (B, T))
    out = model(input_ids=x, labels=x)
    mx.eval(out.loss)

    n_min_rows = None
    rows = []
    for gi, g in enumerate(gates):
        load_sum = float(g.last_load.sum()) if g.last_load is not None else -1
        m_rows = g.last_margins.shape[0] if g.last_margins is not None else -1
        rows.append(m_rows)
        passes_load = load_sum / (B * T * g.top_k)
        passes_marg = m_rows / (B * T)
        print(
            f"gate{gi} owner={layer_of.get(id(g), '?')} "
            f"load_sum={load_sum:.0f} (~{passes_load:.1f} pass) "
            f"margin_rows={m_rows} (~{passes_marg:.2f} pass)"
        )
    stats = model.qb_margin_stats()
    print("qb_margin_stats cropped rows per gate:", stats.shape)
    # 验证：loop gate 的截断样本里应包含 visit-2 的行（修复后）
    for gi, g in enumerate(gates):
        if g.last_margins is not None and g.last_margins.shape[0] == 2 * B * T:
            pass2 = g.last_margins[B * T :]
            cropped = stats[gi]
            # cropped 的每一行与 pass2 块的最小距离；有 0 即说明 visit-2 进入了统计
            d = mx.min(mx.abs(cropped[:, None, :] - pass2[None, :, :]), axis=(1, 2))
            n_pass2 = int((d == 0).sum())
            print(f"gate{gi}: cropped 中来自 visit-2 的行数 = {n_pass2}/{cropped.shape[0]}")


if __name__ == "__main__":
    main()
