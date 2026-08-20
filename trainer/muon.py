"""
Muon 混合优化器（MLX 单设备版）

基于 mlx.optimizers.Muon / mlx.optimizers.MultiOptimizer 实现混合优化器：
- Muon 仅用于 ndim >= 2 且非嵌入/输出头的核心权重矩阵（Newton-Schulz
  正交化动量），weight_decay=0
- 嵌入/输出头与其余标量参数使用 AdamW
"""

import os

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


class BatchedMuon(optim.Muon):
    """Muon 的批量 Newton-Schulz 版：按形状分组堆叠成 (N, r, c) 一次跑 NS，
    动量更新与 lr 缩放同样在组内批量完成。

    动机：基类逐张量跑 NS5（每矩阵 5 步 × 3 GEMM + norm），100M 模型 103
    个 Muon 张量 ≈ 1700 kernel/步，optimizer 占整步墙钟 ~48%（小 kernel
    海洋，M4 Max 实测）。批量化后 kernel 数 ~16/形状组，数学上与基类逐
    张量 NS 严格等价（batch 维无耦合，Frobenius norm 按矩阵独立计算，
    lr 缩放只依赖形状），单步更新实测逐位一致。

    逐参数语义（转置规则、reshape ndim>2、nesterov、wd、lr scale）全部
    照抄基类 apply_single/_zeropower_via_newtonschulz5。
    """

    def __init__(
        self,
        learning_rate,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
        segment_map=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        # 合并投影（如 MLA qkv_proj = [q, kv_down, k_rope] 行拼接）的逐段
        # 语义：dotted path -> 行段尺寸列表。NS 与 lr 缩放按段独立进行，
        # 与未合并时逐矩阵正交化严格等价；动量/wd 是逐元素运算，整堆叠做。
        self.segment_map = segment_map or {}

    def _ns5(self, X):
        """批量 Newton-Schulz：X (N, r, c)，batch 维无耦合（照抄基类规则）。"""
        a_, b_, c_ = 3.4445, -4.7750, 2.0315
        tr = X.shape[-2] > X.shape[-1]
        if tr:
            X = X.swapaxes(-1, -2)
        X = X / (mx.linalg.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)
        for _ in range(self.ns_steps):
            A = X @ X.swapaxes(-1, -2)
            B = b_ * A + c_ * (A @ A)
            X = a_ * X + B @ X
        if tr:
            X = X.swapaxes(-1, -2)
        return X

    def apply_gradients(self, gradients: dict, parameters: dict):
        # 基类开头两段：lazy init + scheduler 更新 + step 递增
        if not self._initialized:
            self.init(gradients)
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)
        self.state["step"] = self.step + 1

        flat_g = dict(tree_flatten(gradients))
        flat_p = dict(tree_flatten(parameters))
        flat_s = dict(tree_flatten(self.state))
        state_v = {k[:-2]: v for k, v in flat_s.items() if k.endswith(".v")}

        def get_state(dotted):
            node = self.state
            for part in dotted.split("."):
                node = node[int(part)] if isinstance(node, list) else node[part]
            return node

        # 分组：同 reshape 后 2D 形状的进一组；ndim<2（Muon 组不应出现，
        # 防御）回退基类逐张量路径；segment_map 命中的合并投影进分段组
        groups: dict = {}
        seg_groups: dict = {}
        singles = []
        for path, g in flat_g.items():
            if g.ndim < 2:
                singles.append(path)
                continue
            orig = g.shape
            r = orig[0]
            c = g.size // r
            if path in self.segment_map:
                key = (r, c, tuple(self.segment_map[path]))
                seg_groups.setdefault(key, []).append((path, orig))
            else:
                groups.setdefault((r, c), []).append((path, orig))

        m, nest, wd = self.momentum, self.nesterov, self.weight_decay
        lr0 = self.learning_rate
        new_params = {}
        new_v = {}

        def momentum_stack(items, r, c):
            """逐元素部分（wd/动量/nesterov）整堆叠做，返回 (U, P, V, G.dtype)。"""
            paths = [p for p, _ in items]
            G = mx.stack([flat_g[p].reshape(r, c) for p in paths])
            P = mx.stack([flat_p[p].reshape(r, c) for p in paths])
            V = mx.stack([state_v[p].reshape(r, c) for p in paths])
            if wd != 0:
                G = G + wd * P
            V = m * V + (1.0 - m) * G
            U = G * (1.0 - m) + V * m if nest else V
            return U, P, V, G.dtype

        def scatter_back(items, V, NP):
            vs = mx.split(V, V.shape[0], axis=0)
            nps = mx.split(NP, NP.shape[0], axis=0)
            for i, (path, orig) in enumerate(items):
                new_v[path] = vs[i].reshape(orig)
                new_params[path] = nps[i].reshape(orig).astype(flat_p[path].dtype)

        for (r, c), items in groups.items():
            U, P, V, dt = momentum_stack(items, r, c)
            X = self._ns5(U)
            lr = lr0.astype(dt) * (max(1.0, r / c) ** 0.5)
            scatter_back(items, V, P - lr * X)

        # 分段组：NS 与 lr 缩放按行段独立（等价未合并的逐矩阵 Muon）
        for (r, c, sizes), items in seg_groups.items():
            U, P, V, dt = momentum_stack(items, r, c)
            split_at = []
            acc = 0
            for z in sizes[:-1]:
                acc += z
                split_at.append(acc)
            u_segs = mx.split(U, split_at, axis=1)
            p_segs = mx.split(P, split_at, axis=1)
            np_segs = []
            for rs, us, ps in zip(sizes, u_segs, p_segs):
                xs = self._ns5(us)
                lr_s = lr0.astype(dt) * (max(1.0, rs / c) ** 0.5)
                np_segs.append(ps - lr_s * xs)
            scatter_back(items, V, mx.concatenate(np_segs, axis=1))

        for path in singles:
            state = get_state(path)
            new_params[path] = super().apply_single(flat_g[path], flat_p[path], state)
            # 与批量组统一从 flat_s 重建 state；若忘记回写，重建会用旧快照
            # 覆盖 apply_single 刚更新的动量，导致 1D 参数的动量每步被清零。
            flat_s[path + ".v"] = state["v"]

        if new_v or singles:
            for path, v in new_v.items():
                flat_s[path + ".v"] = v
            self.state = tree_unflatten(list(flat_s.items()))

        return tree_unflatten(list(new_params.items()))


def create_adamw_optimizer(model, args, training_type="pretrain"):
    """全参数 AdamW（用于优化器杠杆对照实验；lr 需单独调优）。"""
    from .utils import Logger

    Logger("正在为优化器进行参数分组 (pure AdamW)")
    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        return "embed" in path or "lm_head" in path

    embed_count = sum(1 for p, a in trainable if _is_embed(p, a))
    other_count = len(trainable) - embed_count
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - 其余参数组: {other_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, other_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, other_lr_mult, adam_wd = 1.0, 1.0, 0.1

    adamw_embed = optim.AdamW(
        learning_rate=args.learning_rate * embed_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_other = optim.AdamW(
        learning_rate=args.learning_rate * other_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_embed.base_lr = args.learning_rate * embed_lr_mult
    adamw_other.base_lr = args.learning_rate * other_lr_mult
    return optim.MultiOptimizer([adamw_embed, adamw_other], filters=[_is_embed])


def _mla_segment_map(model):
    """MLA 合并投影的 Muon 行段分割表：dotted path -> 行段尺寸。

    qkv_proj 按 [q, kv_down, k_rope] 段、kv_up_proj 按 [k_up, v_up] 段
    分别正交化，保持与未合并逐矩阵 Muon 相同的语义与 lr 缩放。
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    qk = cfg.head_dim + cfg.qk_rope_head_dim
    sizes = {
        "self_attn.qkv_proj.weight": [
            cfg.num_attention_heads * qk,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
        ],
        "self_attn.kv_up_proj.weight": [
            cfg.num_attention_heads * cfg.head_dim,
            cfg.num_attention_heads * cfg.head_dim,
        ],
    }
    out = {}
    for path, arr in tree_flatten(model.trainable_parameters()):
        for suffix, seg in sizes.items():
            if path.endswith(suffix) and arr.shape[0] == sum(seg):
                out[path] = seg
    return out


def create_mixed_optimizer(model, args, training_type="pretrain"):
    """
    创建混合优化器（mlx.optimizers.MultiOptimizer）

    参数分组：
    - Muon：ndim >= 2 且非嵌入/输出头/router 的核心权重矩阵，wd=0
    - AdamW(embed)：lr = args.learning_rate，wd=0.1
    - AdamW(cycle router)：CycleDeltaRouter 的 U/V_c，lr = args.learning_rate
      * cycle_router_lr_mult（默认 0.1，MOE_CYCLE_ROUTER_LR_MULT 可调）
    - AdamW(router)：base router/expert_bias，lr = args.learning_rate * 0.05
      （MOE_ROUTER_LR_MULT 可调），wd=0.1
    - AdamW(engram)：n-gram table/key/value/taps，lr = args.learning_rate
      * engram_lr_mult（默认 1.0），wd=0.1
    - AdamW(scalar)：lr = args.learning_rate，wd=0.1
    （SFT 时 embed/scalar lr 分别乘 0.1/0.3，wd=0.01）
    """
    from .utils import Logger

    Logger("正在为优化器进行参数分组")

    trainable = tree_flatten(model.trainable_parameters())

    def _is_embed(path, arr):
        return "embed" in path or "lm_head" in path

    def _is_muon(path, arr):
        # 3D 堆叠专家权重（*.experts.*）不走 Muon：Muon 对 ndim>2 会
        # reshape 成 (E, out*in) 整体正交化，跨专家耦合尺度；分进 AdamW 组
        # MoE router 也不走 Muon：正交化更新步长恒定偏大，会把路由打分持续
        # 推向失衡（实测 top-1 桶容量 C 从 ~6K 漂到 13K+，(E,C,D) 缓冲膨胀
        # 顶爆内存、吞吐掉 ~30%）；单独小 lr AdamW 组，靠 bias 均衡项兜底
        # hrm_film（per-cycle FiLM 向量堆叠）同理不是矩阵语义，归 AdamW 标量组
        # engram table 是 3D 哈希记忆，也不是 Muon 的矩阵语义，走独立 AdamW
        return (
            arr.ndim >= 2
            and not _is_embed(path, arr)
            and ".experts." not in path
            and ".router." not in path
            and ".engrams." not in path
            and "hrm_film" not in path
        )

    def _is_router(path, arr):
        return ".router." in path

    def _is_engram(path, arr):
        return ".engrams." in path

    def _is_cycle_router(path, arr):
        # CycleDeltaRouter 的 U/V_c 需要比 base router 更高的 lr。
        # base router 必须慢速移动，否则 E=112 细粒度路由会失衡；
        # cycle delta 是零初始化低秩项，参数少且初始不影响选择，
        # 可以用 10× router lr 真正学会特化。
        return ".router." in path and ("cycle_u" in path or "cycle_v" in path)

    muon_count = sum(1 for p, a in trainable if _is_muon(p, a))
    embed_count = sum(1 for p, a in trainable if _is_embed(p, a) and not _is_muon(p, a))
    cycle_router_count = sum(
        1 for p, a in trainable if _is_cycle_router(p, a) and not _is_muon(p, a)
    )
    router_count = sum(
        1
        for p, a in trainable
        if _is_router(p, a) and not _is_cycle_router(p, a) and not _is_muon(p, a)
    )
    engram_count = sum(
        1 for p, a in trainable if _is_engram(p, a) and not _is_muon(p, a)
    )
    scalar_count = sum(
        1
        for p, a in trainable
        if not _is_muon(p, a)
        and not _is_embed(p, a)
        and not _is_router(p, a)
        and not _is_engram(p, a)
    )

    Logger("参数分组完成：")
    Logger(f"  - Muon 参数组 (核心权重): {muon_count} 个张量")
    Logger(f"  - 嵌入层参数组: {embed_count} 个张量")
    Logger(f"  - CycleRouter 参数组: {cycle_router_count} 个张量")
    Logger(f"  - MoE router 参数组: {router_count} 个张量")
    Logger(f"  - Engram 参数组: {engram_count} 个张量")
    Logger(f"  - 标量参数组: {scalar_count} 个张量")

    if training_type == "sft":
        embed_lr_mult, scalar_lr_mult, adam_wd = 0.1, 0.3, 0.01
    else:
        embed_lr_mult, scalar_lr_mult, adam_wd = 1.0, 1.0, 0.1
    # router 需要远小于 AdamW 标量组的 lr：AdamW 每坐标步长≈lr，base lr
    # (Muon 尺度 0.01) 下 ~10 步就把 (32,768) 的 router 权重打乱到 sigmoid
    # 饱和（实测 C 瞬间冲到 14K、吞吐 -45%）；0.05× ≈ 5e-4 让其慢速移动、
    # bias 均衡项压得住负载。可用 MOE_ROUTER_LR_MULT 覆盖。
    router_lr_mult = float(
        os.environ.get("MOE_ROUTER_LR_MULT", getattr(args, "router_lr_mult", 0.05))
    )
    cycle_router_lr_mult = float(
        os.environ.get(
            "MOE_CYCLE_ROUTER_LR_MULT",
            getattr(args, "cycle_router_lr_mult", 0.1),
        )
    )
    engram_lr_mult = float(
        os.environ.get("ENGRAM_LR_MULT", getattr(args, "engram_lr_mult", 1.0))
    )

    muon_opt = BatchedMuon(
        learning_rate=args.learning_rate,  # Muon 使用基础学习率 (e.g., 0.01)
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
        segment_map=_mla_segment_map(model),
    )
    adamw_embed = optim.AdamW(
        learning_rate=args.learning_rate * embed_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_cycle_router = optim.AdamW(
        learning_rate=args.learning_rate * cycle_router_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_router = optim.AdamW(
        learning_rate=args.learning_rate * router_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_engram = optim.AdamW(
        learning_rate=args.learning_rate * engram_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )
    adamw_scalar = optim.AdamW(
        learning_rate=args.learning_rate * scalar_lr_mult,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=adam_wd,
    )

    # 为学习率调度器存储初始学习率
    muon_opt.base_lr = args.learning_rate
    adamw_embed.base_lr = args.learning_rate * embed_lr_mult
    adamw_cycle_router.base_lr = args.learning_rate * cycle_router_lr_mult
    adamw_router.base_lr = args.learning_rate * router_lr_mult
    adamw_engram.base_lr = args.learning_rate * engram_lr_mult
    adamw_scalar.base_lr = args.learning_rate * scalar_lr_mult

    # MultiOptimizer: filters 数量 = len(optimizers) - 1，按顺序首个命中生效，
    # 未命中任何 filter 的参数落到最后一组。
    # 空组必须剔除（如 CycleRouter 关闭时无 cycle 参数）：mlx MultiOptimizer
    # 对空组会在首次 step 的 state init 抛 IndexError。
    optimizers = [
        muon_opt,
        adamw_embed,
        adamw_cycle_router,
        adamw_router,
        adamw_engram,
        adamw_scalar,
    ]
    counts = [
        muon_count,
        embed_count,
        cycle_router_count,
        router_count,
        engram_count,
        scalar_count,
    ]
    filters_all = [
        _is_muon,
        _is_embed,
        _is_cycle_router,
        _is_router,
        _is_engram,
        None,
    ]
    keep = [
        i
        for i, (c, f) in enumerate(zip(counts, filters_all))
        if c > 0 or f is None  # scalar 组作为兜底永远保留
    ]
    return optim.MultiOptimizer(
        [optimizers[i] for i in keep],
        filters=[filters_all[i] for i in keep if filters_all[i] is not None],
    )
