"""手写 Metal kernel 的训练前预热。

每个融合 kernel 首次调用时会对照 eager 参考做一次校验，而校验必须
`.item()` 取回标量。如果这次首调用落在 `mx.compile` 的 trace 内，host
sync 会抛异常、被 dispatch 的 except 分支吞掉，并把模块级 `_DISABLED`
永久置真——训练全程静默走 eager 回退，且没有任何日志。

实测代价（KDA 层 fwd+bwd，bs12×1024）：kernel 全启用 49.7ms，全回退
72.0ms，每层 22.3ms。所以在 compile 之前必须在 eager 上下文里把所有
实际会用到的形状跑一遍。

`prewarm_all` 从模型结构（而非配置字段名）推导 conv 通道数，对每个通道
同时预热 seg×silu 四种变体：变体由运行时是否打包序列决定，多编译几个
kernel 是一次性成本，漏掉一个则整轮训练回退。
"""

import os

import mlx.core as mx  # noqa: F401

# 总开关：0 时不预热、也不启用这轮新接入训练图的 conv / kda_scan / kda_prep。
# attn_res 在 run 81 里已经 prewarm，保持原样。
_FUSED = os.environ.get("VIBY_FUSED_KERNELS", "1") != "0"


def _conv_specs(model):
    """扫出所有深度卷积层的 (通道数, 是否 SiLU)。

    按类名而非 isinstance 判断，避免 model.kernels 与 model.kda /
    model.attention 之间的循环导入。
    """
    specs = set()
    try:
        mods = model.named_modules()
    except Exception:
        return specs
    for _, mod in mods:
        name = type(mod).__name__
        if name not in ("_KDAConv", "ShortConv"):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.ndim != 2:
            continue
        specs.add((int(w.shape[1]), name == "_KDAConv"))
    return specs


def _situ_halves(model):
    """扫出所有 SiTU-GLU 打包核需要的半宽 I（每个 I 一份 kernel）。

    打包核把 I 编译成常量，所以要覆盖实际用到的每个宽度：稠密/共享专家
    的 gate_proj 输出宽，以及堆叠路由专家 gate_up_w 的 2I 的一半。
    """
    halves = set()
    try:
        mods = model.named_modules()
    except Exception:
        return halves
    for _, mod in mods:
        gp = getattr(mod, "gate_proj", None)
        w = getattr(gp, "weight", None) if gp is not None else None
        if w is not None and w.ndim == 2:
            halves.add(int(w.shape[0]))
        gu = getattr(mod, "gate_up_w", None)
        if gu is not None and getattr(gu, "ndim", 0) == 3:
            halves.add(int(gu.shape[1]) // 2)
    return halves


def prewarm_all(model, config, dtype, seq_len, log=None):
    """在 mx.compile 之前预编译 + 校验全部融合 kernel。

    返回 (ok_count, fail_names)。失败的 kernel 已切到 eager 回退，训练
    仍然正确，只是慢。
    """

    def note(msg):
        if log is not None:
            log(msg)

    fails = []
    ok = 0

    # ---- AttnRes 合并：N 覆盖每层两个残差入口 ----
    try:
        from . import attn_res_fused

        L = int(getattr(config, "num_hidden_layers", 0) or 0)
        D = int(config.hidden_size)
        if L > 0:
            if attn_res_fused.prewarm(D, dtype, range(2, 2 * L + 2)):
                ok += 1
            else:
                fails.append("attn_res")
    except Exception as e:
        fails.append(f"attn_res({e})")

    # ---- conv / kda_prep / kda_scan：run 81 未 prewarm，compile 下静默 eager ----
    if not _FUSED:
        note("VIBY_FUSED_KERNELS=0：conv / kda_prep / kda_scan 走 eager（对齐 run 81）")
    else:
        try:
            from . import conv

            specs = _conv_specs(model)
            if not specs:
                note("未发现深度卷积层，跳过 causal_conv 预热")
            for C, silu in sorted(specs):
                for has_seg in (True, False):
                    if conv.prewarm(C, dtype, has_seg, silu):
                        ok += 1
                    else:
                        fails.append(f"conv(C={C},seg={has_seg},silu={silu})")
                        break
        except Exception as e:
            fails.append(f"conv({e})")

        try:
            from model.kda import KDA_CHUNK, _scan_prewarm

            from . import kda_prep

            heads = int(getattr(config, "num_attention_heads", 0) or 0)
            hd = int(getattr(config, "head_dim", 0) or 0) or (
                int(config.hidden_size) // heads if heads else 0
            )
            if hd > 0 and seq_len > 0:
                C = KDA_CHUNK
                NC = (int(seq_len) + C - 1) // C
                if kda_prep.prewarm(C, hd, NC):
                    ok += 1
                else:
                    fails.append("kda_prep")
                from . import kda_inner as kda_inner_mod

                if kda_inner_mod.prewarm(C, hd):
                    ok += 1
                else:
                    fails.append("kda_inner")
                if _scan_prewarm(NC, C, hd, hd):
                    ok += 1
                else:
                    fails.append("kda_scan")
        except Exception as e:
            fails.append(f"kda({e})")

        try:
            from . import situ

            if situ.prewarm(dtype):
                ok += 1
            else:
                fails.append("situ")
            halves = _situ_halves(model)
            if not halves:
                note("未发现 gate/up 投影，跳过 situ 打包核预热")
            elif situ.prewarm_packed(dtype, halves):
                ok += 1
            else:
                fails.append(f"situ_packed(I={sorted(halves)})")
        except Exception as e:
            fails.append(f"situ({e})")

    if fails:
        note(f"融合 kernel 预热失败（已回退 eager）: {', '.join(fails)}")
    else:
        note(f"融合 kernel 预热完成（{ok} 组形状）")
    return ok, fails


__all__ = ["prewarm_all"]
