"""
基础训练器类，提供通用的训练逻辑（MLX 单设备版）
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
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
            indices = (
                np.random.permutation(n) if loader.shuffle else np.arange(n)
            )
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

        # loss + 梯度函数（mx.compile 默认启用）
        self._loss_and_grad = self._build_loss_and_grad()

        # 处理检查点恢复
        self.start_epoch, self.start_step = self._handle_checkpoint_resume()

    def _loss_fn(self, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids=None):
        """训练 loss 函数（model 走闭包引用）

        返回 (加权和 loss / accumulation_steps, mtp 分量 loss)。
        mtp 分量仅作日志展示、不缩放；无 MTP 时返回 0 常量，
        保持 compile 图结构稳定。
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
        return res.loss / self.args.accumulation_steps, mtp_loss

    def _loss_and_grad_with_params(self, params, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids=None):
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

    def _optimizer_step(self, accum_grads, accum_count):
        """累积窗口结束：梯度裁剪 + 优化器更新"""
        if accum_count <= 0 or accum_grads is None:
            return 0.0

        # 每个微批的 loss 已除以 accumulation_steps，
        # 累加的梯度即窗口平均梯度，无需再除
        grads, grad_norm = optim.clip_grad_norm(accum_grads, self.args.grad_clip)

        self.optimizer.update(self.model, grads)

        mx.eval(self.model.parameters(), self.optimizer.state)
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
        wandb=None,
        skip_steps: int = 0,
    ):
        """训练一个epoch"""
        start_time = time.time()
        base_step_offset_for_speed = skip_steps
        accum_grads = None
        accum_count = 0
        last_grad_norm = 0.0  # Store last calculated gradient norm

        self.model.train()

        for step, batch in enumerate(train_loader):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue
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
            (loss, mtp_loss), grads = self._loss_and_grad(
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
            mx.eval(loss, mtp_loss, grads)

            # 梯度累加
            accum_grads = (
                grads
                if accum_grads is None
                else tree_map(mx.add, accum_grads, grads)
            )
            accum_count += 1

            # 梯度累积窗口结束，执行更新
            if (step + 1) % self.args.accumulation_steps == 0:
                last_grad_norm = self._optimizer_step(accum_grads, accum_count)
                accum_grads = None
                accum_count = 0

                # 时长预算只在窗口边界检查：中途停止会丢掉已累加但未更新的梯度
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

            # 日志记录
            if step % self.args.log_interval == 0:
                # 无论何时记录，都计算当前微批次的原始损失值
                # loss 是已经被 accumulation_steps 缩放过的损失
                # 将其乘回去，就得到了当前单个微批次的原始损失，确保日志值量级一致
                mx.eval(loss)
                current_loss = float(loss.item()) * self.args.accumulation_steps
                # MTP 分量（未加权）仅在开启 MTP 时展示
                has_mtp = getattr(self.lm_config, "mtp_depth", 0) > 0
                current_mtp_loss = float(mtp_loss.item()) if has_mtp else None

                # 使用上次计算的梯度范数
                grad_norm_to_log = last_grad_norm

                log_training_progress(
                    epoch,
                    step,
                    iter_per_epoch,
                    current_loss,
                    self.optimizer,
                    start_time,
                    self.args,
                    wandb,
                    grad_norm_to_log,
                    base_step_offset=base_step_offset_for_speed,
                    mtp_loss=current_mtp_loss,
                )

            # 模型保存
            self._save_if_needed(epoch, step)

        return False

    def train(self, train_loader, wandb=None):
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

            time_limit_hit = self.train_epoch(
                epoch,
                train_loader,
                iter_per_epoch,
                total_training_steps,
                wandb,
                skip_steps,
            )

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
