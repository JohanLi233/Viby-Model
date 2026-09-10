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


def _glu_halves(model):
    """扫出所有 GLU（SiLU/SiTU）打包核需要的半宽 I（每个 I 一份 kernel）。

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
            max_n = 2 * L + 1
            win = int(getattr(config, "attn_res_window", 0) or 0)
            if win > 0:
                max_n = min(max_n, win)
                if getattr(config, "attn_res_register", False):
                    # 读侧混合 = 寄存器 + 近 W 个写入，N 最大 W+1
                    max_n = min(2 * L + 1, win + 1)
                elif (
                    bool(getattr(config, "loop_anchor", True))
                    and not bool(getattr(config, "ihc", False))
                    and config.loop_layer_range() is not None
                ):
                    # loop 锚点读出（replace 模式）：vs = 常驻寄存器
                    # （x_entry + ≤r−1 个 visit 摘要）+ pin 份 pre-span
                    # 写入（embedding + 每浅层 2 份）+ 最近 win 份写入，
                    # 且 merge 发生在本 sublayer 写入 append 之后。
                    # N 最大 pin + win + r；窗口填满前的 ramp 会经过
                    # 其间全部值，range 连续覆盖。漏预热会让这些 N 在
                    # compile trace 内首次校验（host sync 异常被吞）而
                    # 整轮训练回退 eager。
                    start, _ = config.loop_layer_range()
                    pin = 1 + 2 * start
                    regs = int(getattr(config, "loop_count", 2) or 2)
                    max_n = max(max_n, pin + win + regs)
            if getattr(config, "attn_res_read_h", False):
                # 子层读 h，不走 AttnRes merge
                note("attn_res_read_h：跳过 AttnRes 合并预热")
            elif getattr(config, "ihc", False):
                note("ihc：跳过 AttnRes 合并预热")
            elif attn_res_fused.prewarm(D, dtype, range(2, max_n + 1)):
                ok += 1
            else:
                fails.append("attn_res")
    except Exception as e:
        fails.append(f"attn_res({e})")

    try:
        from . import ihc_fused

        if not bool(getattr(config, "ihc", False)):
            pass
        elif not _FUSED:
            note("VIBY_FUSED_KERNELS=0：跳过 iHC kernel 预热")
        elif ihc_fused.prewarm_from_model(model, dtype):
            ok += 1
        else:
            fails.append("ihc")
    except Exception as e:
        fails.append(f"ihc({e})")

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
            from model import eigengate as _eg

            from . import kda_prep

            if not bool(getattr(config, "use_linear_attn", False)):
                note("use_linear_attn=0：跳过 KDA kernel 预热")
            else:
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
                    ncs = {NC}
                    scan_ok = True
                    if _eg.enabled():
                        if _eg.target_name() == "keep" and not _eg.learnable():
                            mask = tuple(_eg.chunk_gate_mask(NC, int(seq_len), C, 0))
                            if any(mask):
                                from model.kda import _scan_prewarm_gated

                                if not _scan_prewarm_gated(NC, C, hd, hd, mask):
                                    scan_ok = False
                        else:
                            for length, _ in _eg.scan_segments(NC, int(seq_len), C, 0):
                                ncs.add(int(length))
                    for nc in sorted(ncs):
                        if not _scan_prewarm(nc, C, hd, hd):
                            scan_ok = False
                            break
                    if scan_ok:
                        ok += 1
                    else:
                        fails.append("kda_scan")
        except Exception as e:
            fails.append(f"kda({e})")

        # GLU 融合核按 hidden_act 择一预编译/校验：'silu'（默认）用
        # kernels/silu.py，'situ' 用 kernels/situ.py（二者结构同构）。
        if getattr(config, "hidden_act", "situ") == "situ":
            try:
                from . import situ

                if situ.prewarm(dtype):
                    ok += 1
                else:
                    fails.append("situ")
                halves = _glu_halves(model)
                if not halves:
                    note("未发现 gate/up 投影，跳过 situ 打包核预热")
                elif situ.prewarm_packed(dtype, halves):
                    ok += 1
                else:
                    fails.append(f"situ_packed(I={sorted(halves)})")
            except Exception as e:
                fails.append(f"situ({e})")
        elif getattr(config, "hidden_act", "silu") == "silu":
            # 默认 hidden_act='silu'：SiLU-GLU 融合核（fwd+bwd + 打包入口）。
            try:
                from . import silu

                if silu.prewarm(dtype):
                    ok += 1
                else:
                    fails.append("silu")
                halves = _glu_halves(model)
                if not halves:
                    note("未发现 gate/up 投影，跳过 silu 打包核预热")
                elif silu.prewarm_packed(dtype, halves):
                    ok += 1
                else:
                    fails.append(f"silu_packed(I={sorted(halves)})")
            except Exception as e:
                fails.append(f"silu({e})")

    try:
        from . import layer_decode

        if getattr(config, "use_linear_attn", False):
            note("use_linear_attn=1：跳过 layer_decode 预热")
        elif layer_decode.prewarm_from_model(model, dtype):
            ok += 1
        else:
            fails.append("layer_decode")
    except Exception as e:
        fails.append(f"layer_decode({e})")

    try:
        from . import moe_write_spread

        if moe_write_spread.prewarm_from_model(model, dtype):
            ok += 1
        else:
            fails.append("moe_write_spread")
    except Exception as e:
        fails.append(f"moe_write_spread({e})")

    if fails:
        note(f"融合 kernel 预热失败（已回退 eager）: {', '.join(fails)}")
    else:
        note(f"融合 kernel 预热完成（{ok} 组形状）")
    return ok, fails


__all__ = ["prewarm_all"]
