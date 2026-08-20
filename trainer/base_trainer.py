"""
基础训练器类，提供通用的训练逻辑（MLX 单设备版）
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map
from .muon import create_mixed_optimizer
from .utils import (
    Logger,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    apply_lr_schedule,
    log_training_progress,
    set_ddp_flag,
)

_SENTINEL = object()


def _collate_numpy(samples):
    """样本堆叠为 numpy batch（纯 CPU 工作，在后台线程执行）"""
    first = samples[0]
    if isinstance(first, dict):
        return {
            key: np.stack([np.asarray(sample[key]) for sample in samples])
            for key in first
        }
    return tuple(
        np.stack([np.asarray(sample[i]) for sample in samples])
        for i in range(len(first))
    )


def _to_mx(batch):
    """numpy batch -> mx.array（在主线程执行，开销极小）"""
    if isinstance(batch, dict):
        return {key: mx.array(value) for key, value in batch.items()}
    return tuple(mx.array(value) for value in batch)


class _PrefetchIterator:
    """后台线程预取：样本读取 + tokenize + numpy 堆叠全部在后台完成，
    主线程取到的是就绪的 numpy batch，与 GPU 计算重叠。"""

    def __init__(self, loader):
        self.loader = loader
        self._queue = queue.Queue(maxsize=loader.prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self):
        loader = self.loader
        try:
            n = len(loader.dataset)
            indices = np.random.permutation(n) if loader.shuffle else np.arange(n)
            with ThreadPoolExecutor(max_workers=loader.num_workers) as pool:
                for start in range(0, n, loader.batch_size):
                    if self._stop.is_set():
                        return
                    batch_indices = indices[start : start + loader.batch_size]
                    if loader.drop_last and len(batch_indices) < loader.batch_size:
                        break
                    samples = list(
                        pool.map(lambda i: loader.dataset[int(i)], batch_indices)
                    )
                    self._put(_collate_numpy(samples))
                    if self._stop.is_set():
                        return
        except Exception as e:  # 把后台异常传递给主线程
            self._put(e)
            return
        self._put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is _SENTINEL:
            self._stop.set()
            raise StopIteration
        if isinstance(item, Exception):
            self._stop.set()
            raise item
        return _to_mx(item)

    def __del__(self):
        self._stop.set()


class MLXDataLoader:
    """简单的单设备数据加载器（异步预取版）。

    每个 epoch 迭代时重新 shuffle，按 batch_size 切分，drop_last。
    数据准备（读文件 + tokenize + 堆叠）在后台线程池完成，
    主线程只做 numpy -> mx.array 转换，避免 GPU 等待数据。
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_batches=4,
        num_workers=8,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return _PrefetchIterator(self)


class BaseTrainer:
    """基础训练器类"""

    def __init__(self, args, model, tokenizer, lm_config, training_type="pretrain"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lm_config = lm_config
        self.training_type = training_type

        # 单设备 MLX 训练，无分布式概念（保留属性以兼容调用方）
        self.ddp = False
        self.ddp_local_rank = 0
        self.device = args.device

        set_ddp_flag(False)

        # 初始化训练组件
        self._init_training_components()

        # 训练时长预算（max_train_minutes）的起始时间，train() 开始时设置
        self._train_start_time = None
        # Ctrl-C 时保存检查点使用的当前 step 位置（train_epoch 内实时更新）
        self._last_epoch = 0
        self._last_step = 0

    def _time_limit_exceeded(self):
        """是否已达最长训练时长（分钟）。未设置或训练未开始时返回 False。"""
        limit = getattr(self.args, "max_train_minutes", None)
        if not limit or self._train_start_time is None:
            return False
        return (time.time() - self._train_start_time) >= limit * 60

    def _init_training_components(self):
        """初始化训练组件"""
        # 创建优化器
        if getattr(self.args, "optimizer", "muon") == "adamw":
            from .muon import create_adamw_optimizer

            self.optimizer = create_adamw_optimizer(
                self.model, self.args, self.training_type
            )
        else:
            self.optimizer = create_mixed_optimizer(
                self.model, self.args, self.training_type
            )

        # MoE 无辅助损失负载均衡：开启 router 负载统计（图节点随 loss 输出），
        # 供 _optimizer_step 按 V3 规则更新 expert_bias
        self._moe_bias_rate = float(
            getattr(self.lm_config, "moe_bias_update_rate", 0.0) or 0.0
        )
        self._moe_gates = (
            self.model.moe_gates() if hasattr(self.model, "moe_gates") else []
        )
        if self._moe_bias_rate > 0:
            for gate in self._moe_gates:
                gate.collect_stats = True

        # loss + 梯度函数（mx.compile 默认启用；MoE 容量表逐微批滚动、
        # 形状跨步不稳定会反复 retrace，见 _build_loss_and_grad 的自动回退）
        self._loss_and_grad = self._build_loss_and_grad()

        # Metal 分配器缓存上限（--cache_limit_gb，默认 24GB，0=不限）：
        # 上限内的空闲块常驻复用、不归还 OS，避免每步"释放-重分配"抖动
        # （bs16x640 实测 10G→24G 提速 4.5%；该配置峰值 14.8G，峰值+缓存
        # ≈39G）。历史上设限是为防 optimizer 临时 buffer 污染 freelist
        # 拖慢激活分配（243→442ms/步）；BatchedMuon 批量化后临时块少且
        # 形状固定，各档上限扫描均未复现污染。大 batch 配置注意
        # 峰值+缓存上限不要超过物理内存。
        cache_gb = float(getattr(self.args, "cache_limit_gb", 24.0))
        if cache_gb > 0:
            mx.set_cache_limit(int(cache_gb * 1024**3))

        # 处理检查点恢复
        self.start_epoch, self.start_step = self._handle_checkpoint_resume()

    def _loss_fn(self, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids=None):
        """训练 loss 函数（model 走闭包引用）

        返回 (加权和 loss / accumulation_steps, mtp 分量 loss, moe 负载统计)。
        mtp 分量仅作日志展示、不缩放；无 MTP 时返回 0 常量。
        总 loss 中已包含 moe_aux_loss_weight 加权的软负载均衡项。
        moe 负载统计为各 router 每专家 token 计数拼接向量，供偏置均衡
        更新；无 MoE 时返回零长占位，保持 compile 图结构稳定。
        """
        res = self.model(
            input_ids=X,
            labels=Y,
            loss_mask=loss_mask,
            attention_mask=attn_mask,
            mask_has_pad=mask_has_pad,
            segment_ids=seg_ids,
        )
        mtp_loss = res.mtp_loss if res.mtp_loss is not None else mx.array(0.0)
        lm_loss = res.lm_loss if res.lm_loss is not None else res.loss
        aux_loss = res.aux_loss if res.aux_loss is not None else mx.array(0.0)
        div_loss = (
            res.diversity_loss if res.diversity_loss is not None else mx.array(0.0)
        )
        moe_stats = self.model.moe_load_stats() if self._moe_gates else None
        if moe_stats is None:
            moe_stats = mx.zeros((0,), dtype=mx.float32)
        return (
            res.loss / self.args.accumulation_steps,
            mtp_loss,
            moe_stats,
            lm_loss / self.args.accumulation_steps,
            aux_loss,
            div_loss,
        )

    def _loss_and_grad_with_params(
        self, params, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids=None
    ):
        """参数显式作为入参的 loss 函数（value_and_grad 的目标）。

        不能用 nn.value_and_grad + mx.compile：那样 params 通过闭包
        （model.trainable_parameters()）被 compile 捕获为常量，梯度永远
        基于初始权重、优化器更新完全无效。显式传参后参数成为运行时输入。
        mask_has_pad 是 python bool，作为编译期常量（最多两个图变体）。
        """
        self.model.update(params)
        return self._loss_fn(X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids)

    def _build_loss_and_grad(self):
        fn = mx.value_and_grad(self._loss_and_grad_with_params)
        use_compile = getattr(self.args, "compile_model", False)
        if use_compile and self._moe_gates:
            Logger(
                "MoE 稀疏桶容量表逐微批滚动、形状跨步不稳定，mx.compile 会"
                "反复 retrace，自动回退 eager；如要 compile 请关闭 MoE"
            )
            use_compile = False
        if use_compile:
            Logger("使用 mx.compile 编译 loss 函数")
            fn = mx.compile(fn)
        self._compiled = use_compile
        return fn

    def _handle_checkpoint_resume(self):
        """处理检查点恢复"""
        start_epoch = 0
        start_step = 0

        if self.args.resume:
            if os.path.exists(self.args.resume):
                start_epoch, start_step = load_checkpoint(
                    self.args.resume, self.model, self.optimizer, self.args
                )
            else:
                Logger(
                    f"Warning: Checkpoint file {self.args.resume} not found, starting from scratch"
                )
        elif getattr(self.args, "auto_resume", False):
            latest_checkpoint = find_latest_checkpoint(self.args.save_dir)
            if latest_checkpoint:
                start_epoch, start_step = load_checkpoint(
                    latest_checkpoint,
                    self.model,
                    self.optimizer,
                    self.args,
                )
            else:
                Logger("No checkpoint found for auto resume, starting from scratch")

        return start_epoch, start_step

    def create_data_loader(self, dataset):
        """创建数据加载器"""
        return MLXDataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _optimizer_step(self, accum_grads, accum_count, moe_stats=None):
        """累积窗口结束：梯度裁剪 + 优化器更新 + MoE 路由偏置更新"""
        if accum_count <= 0 or accum_grads is None:
            return 0.0

        # 每个微批的 loss 已除以 accumulation_steps，
        # 累加的梯度即窗口平均梯度，无需再除
        grads, grad_norm = optim.clip_grad_norm(accum_grads, self.args.grad_clip)

        self.optimizer.update(self.model, grads)

        # 无辅助损失负载均衡：比例-截断 + 零均值投影（见 update_moe_biases）。
        # 负载统计为累积窗口内各微批之和（V3 每步更新的窗口近似）
        if self._moe_bias_rate > 0 and moe_stats is not None and moe_stats.size > 0:
            self.model.update_moe_biases(moe_stats, self._moe_bias_rate)

        mx.eval(self.model.parameters(), self.optimizer.state)
        # 注：optimizer step 的临时 buffer 治理由 __init__ 里的
        # mx.set_cache_limit（--cache_limit_gb）统一负责；不要在这里每步
        # mx.clear_cache()——那会把 fwd+bwd 可复用的激活缓存块一并清掉，
        # 实测反而更慢。
        return float(grad_norm)

    def _save_if_needed(self, epoch, step):
        if (step + 1) % self.args.save_interval != 0:
            return
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            step,
            self.args,
            self.lm_config,
            self.training_type,
        )

    def train_epoch(
        self,
        epoch,
        train_loader,
        iter_per_epoch,
        total_training_steps,
        swanlab=None,
        skip_steps: int = 0,
    ):
        """训练一个epoch"""
        start_time = time.time()
        base_step_offset_for_speed = skip_steps
        accum_grads = None
        accum_count = 0
        last_grad_norm = 0.0  # Store last calculated gradient norm
        last_moe_stats = None  # 窗口内累加的 MoE 负载统计（偏置更新用）
        win_overflow = 0  # 上次日志点以来的桶容量溢出 pair 累计

        self.model.train()

        for step, batch in enumerate(train_loader):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue
            self._last_epoch = epoch
            self._last_step = step
            # doc_mask 打包模式下 dataset 多返回一项 segment_ids
            if len(batch) == 4:
                X, Y, loss_mask, seg_ids = batch
            else:
                X, Y, loss_mask = batch
                seg_ids = None

            # 应用学习率调度（每微批 step 应用）
            global_step = epoch * iter_per_epoch + step
            apply_lr_schedule(
                self.optimizer,
                global_step,
                total_training_steps,
                self.args.warmup_iters,
                min_lr_ratio=getattr(self.args, "min_lr_ratio", 0.1),
            )

            # 构造 attention_mask，屏蔽 PAD 位置
            attn_mask = (X != self.tokenizer.pad_token_id).astype(mx.int32)
            # mx.compile 图内不允许 .item() host sync，在 eager 侧算好传入；
            # eager 模式下模型内部也会做同样的判断，成本相同
            mask_has_pad = bool(mx.any(attn_mask != 1).item())

            # 前向 + 反向（loss 内部已除以 accumulation_steps）
            # 参数显式传入：compile 下保证梯度基于当前权重而非初始快照
            # mtp_loss 是辅助输出，仅用于日志展示，不参与梯度
            params = self.model.trainable_parameters()
            (
                (
                    loss,
                    mtp_loss,
                    moe_stats,
                    lm_loss,
                    aux_loss,
                    div_loss,
                ),
                grads,
            ) = self._loss_and_grad(
                params, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids
            )
            if self._compiled:
                # compiled fn 内部的 model.update(params) 只在 trace 时执行，
                # 会把无 primitive 的占位数组留在 module 上；立即用真实参数
                # 恢复，避免污染后续 optimizer step 的 eval
                self.model.update(params)

            # 立即物化本微批的 loss/grads 并释放反向图。MLX 是惰性求值，
            # 若不 eval，accumulation_steps 个微批的前向+反向图会全部存活到
            # optimizer step，显存按窗口大小成倍增长。
            # 拆成两次 eval：前向输出与梯度分开物化。一次性 eval(loss+grads)
            # 时 MLX 的图调度会把前向 tape 滞留与反向临时量同时顶到峰值
            # （r073 配置实测 46GB/4.0s）；拆开后前向先落定（~14.5GB）再跑
            # 反向（峰值 ~16GB），显存省 ~3×、单步快 ~2×，数值不变。
            mx.eval(loss, mtp_loss, moe_stats, lm_loss, aux_loss, div_loss)
            mx.eval(grads)
            # 负载统计跨微批累加：偏置均衡看到整个累积窗口的负载，
            # 比只用最后一个微批噪声更小（V3 每步更新的窗口近似）
            last_moe_stats = (
                moe_stats
                if last_moe_stats is None
                else mx.add(last_moe_stats, moe_stats)
            )

            # MoE 稀疏桶容量表滚动更新：本微批实测桶计数 → 下微批容量。
            # counts 已随上面的 eval 物化，tolist 不再钉住前向图；
            # 这是稀疏前向无 host sync 的前提（见 _sparse_forward）。
            # 溢出 pair 累计到日志点统一报告（逐微批打印在路由漂移期
            # 会刷屏；单微批小溢出会被容量抬升自愈）。
            for _m in self.model.modules():
                _f = getattr(_m, "update_capacity_table", None)
                if _f is not None:
                    win_overflow += _f()

            # 梯度累加
            accum_grads = (
                grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
            )
            accum_count += 1

            # 梯度累积窗口结束，执行更新
            if (step + 1) % self.args.accumulation_steps == 0:
                last_grad_norm = self._optimizer_step(
                    accum_grads, accum_count, moe_stats=last_moe_stats
                )
                accum_grads = None
                accum_count = 0
                last_moe_stats = None

                # 时长/步数预算只在窗口边界检查：中途停止会丢掉已累加但未更新的梯度
                if self._time_limit_exceeded():
                    Logger(
                        f"已达最长训练时长 {self.args.max_train_minutes} 分钟，"
                        f"在梯度累积窗口边界（epoch {epoch + 1}, step {step}）停止训练"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        step,
                        self.args,
                        self.lm_config,
                        self.training_type,
                    )
                    return True

                max_steps = getattr(self.args, "max_steps", None)
                if max_steps and (epoch * iter_per_epoch + step + 1) >= max_steps:
                    Logger(
                        f"已达最大步数 {max_steps}（微批口径），"
                        f"在梯度累积窗口边界（epoch {epoch + 1}, step {step}）停止训练"
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        step,
                        self.args,
                        self.lm_config,
                        self.training_type,
                    )
                    return True

            # 日志记录
            if step % self.args.log_interval == 0:
                # 无论何时记录，都计算当前微批次的原始损失值
                # loss 是已经被 accumulation_steps 缩放过的损失
                # 将其乘回去，就得到了当前单个微批次的原始损失，确保日志值量级一致
                mx.eval(loss)
                current_loss = float(loss.item()) * self.args.accumulation_steps
                # 各 loss 分量（未加权），用于日志与 swanlab
                has_mtp = getattr(self.lm_config, "mtp_depth", 0) > 0
                current_mtp_loss = float(mtp_loss.item()) if has_mtp else None
                current_main_loss = float(lm_loss.item()) * self.args.accumulation_steps
                current_aux_loss = float(aux_loss.item()) * float(
                    getattr(self.lm_config, "moe_aux_loss_weight", 0.0) or 0.0
                )
                current_div_loss = float(div_loss.item()) * float(
                    getattr(self.lm_config, "moe_diversity_loss_weight", 0.0) or 0.0
                )

                # 使用上次计算的梯度范数
                grad_norm_to_log = last_grad_norm

                # 详细指标（swanlab 上报 + VIBY_DEBUG_MEM stderr 打印共用一份
                # 计算）：MoE 分 gate 负载最大值、单次桶容量峰值、内存三件套
                debug_mem = bool(os.environ.get("VIBY_DEBUG_MEM"))
                extra = None
                if swanlab is not None or debug_mem:
                    act = mx.get_active_memory() / 2**30
                    cache = mx.get_cache_memory() / 2**30
                    peak = mx.get_peak_memory() / 2**30
                    gate_max = [
                        float(g.last_load.max())
                        for g in self._moe_gates
                        if g.last_load is not None
                    ]
                    # 各 MoE 模块本窗口见过的最大单次桶容量（决定 (E,C,D) 峰值）
                    cmax = 0
                    for m in self.model.modules():
                        v = getattr(m, "_c_max_seen", None)
                        if v:
                            cmax = max(cmax, v)
                            m._c_max_seen = 0
                    extra = {
                        "mem/active_gb": round(act, 3),
                        "mem/cache_gb": round(cache, 3),
                        "mem/peak_gb": round(peak, 3),
                        "moe/call_c_max": cmax,
                    }
                    for gi, v in enumerate(gate_max):
                        extra[f"moe/gate{gi}_max_load_k"] = round(v / 2**10, 3)
                    extra["moe/overflow_pairs"] = win_overflow
                    if debug_mem:
                        Logger(
                            f"[mem] active={act:.2f}G cache={cache:.2f}G "
                            f"peak={peak:.2f}G "
                            f"gate_maxK={'/'.join(f'{v / 2**10:.1f}' for v in gate_max)} "
                            f"callC={cmax}"
                        )

                log_training_progress(
                    epoch,
                    step,
                    iter_per_epoch,
                    current_loss,
                    self.optimizer,
                    start_time,
                    self.args,
                    swanlab,
                    grad_norm_to_log,
                    base_step_offset=base_step_offset_for_speed,
                    mtp_loss=current_mtp_loss,
                    main_loss=current_main_loss,
                    aux_loss=current_aux_loss,
                    diversity_loss=current_div_loss,
                    extra=extra,
                )
                if win_overflow > 0:
                    Logger(
                        f"MoE 桶容量溢出：最近 {self.args.log_interval} 微批累计 "
                        f"{win_overflow} 对（对应 pair 输出被置零，容量已抬升；"
                        f"持续增长请排查路由均衡）"
                    )
                win_overflow = 0

            # 模型保存
            self._save_if_needed(epoch, step)

        return False

    def train(self, train_loader, swanlab=None):
        """主训练循环"""
        iter_per_epoch = len(train_loader)
        total_training_steps = self.args.epochs * iter_per_epoch
        # 短时训练（如 --max_train_minutes 限时跑）时，用 --lr_decay_steps 把
        # cosine 衰减的终点对齐到实际会跑的步数，否则 lr 几乎不衰减
        lr_decay_steps = getattr(self.args, "lr_decay_steps", None)
        if lr_decay_steps:
            total_training_steps = lr_decay_steps
        Logger(f"训练总步数: {total_training_steps}, 每轮步数: {iter_per_epoch}")
        if getattr(self.args, "max_train_minutes", None):
            Logger(f"最长训练时长: {self.args.max_train_minutes} 分钟")

        self._train_start_time = time.time()
        time_limit_hit = False

        for epoch in range(self.start_epoch, self.args.epochs):
            # 计算需要跳过的步骤
            skip_steps = self.start_step if epoch == self.start_epoch else 0

            try:
                time_limit_hit = self.train_epoch(
                    epoch,
                    train_loader,
                    iter_per_epoch,
                    total_training_steps,
                    swanlab,
                    skip_steps,
                )
            except KeyboardInterrupt:
                Logger(
                    "检测到 Ctrl-C：在最后完成的微批位置保存检查点后退出"
                    f"（epoch {self._last_epoch + 1}, step {self._last_step}）"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self._last_epoch,
                    self._last_step,
                    self.args,
                    self.lm_config,
                    self.training_type,
                )
                time_limit_hit = True

            # 重置start_step
            if epoch == self.start_epoch:
                self.start_step = 0

            if time_limit_hit:
                break

        # 训练结束保存最后一个检查点（若最后一个步恰好按 save_interval 已保存则跳过；
        # 因时长限制停止时已保存过，不再重复保存）
        if (
            not time_limit_hit
            and self.args.epochs > 0
            and iter_per_epoch > 0
            and iter_per_epoch % self.args.save_interval != 0
        ):
            last_epoch = max(self.start_epoch, self.args.epochs - 1)
            save_checkpoint(
                self.model,
                self.optimizer,
                last_epoch,
                iter_per_epoch - 1,
                self.args,
                self.lm_config,
                self.training_type,
            )
