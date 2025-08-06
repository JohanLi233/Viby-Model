"""
基础训练器类，提供通用的训练逻辑
"""

import os
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from typing import Optional
from .utils import (
    Logger,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    create_mixed_optimizer,
    apply_lr_schedule,
    log_training_progress,
    set_ddp_flag,
)


class BaseTrainer:
    """基础训练器类"""

    def __init__(self, args, model, tokenizer, lm_config, training_type="pretrain"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.lm_config = lm_config
        self.training_type = training_type

        # 初始化分布式训练
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.ddp_local_rank = 0
        self.device = args.device

        if self.ddp:
            self._init_distributed_mode()

        set_ddp_flag(self.ddp)

        # 初始化训练组件
        self._init_training_components()

    def _init_distributed_mode(self):
        """初始化分布式训练模式"""
        dist.init_process_group(backend="nccl")
        self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        self.args.device = torch.device(self.device)

        # 设置随机种子
        base_seed = 1337
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    def _init_training_components(self):
        """初始化训练组件"""
        # 混合精度训练
        self.scaler = torch.amp.GradScaler(  # type: ignore
            enabled=(self.args.dtype in ["float16", "bfloat16"])
        )

        # 自动类型转换上下文
        device_type = (
            "cuda"
            if "cuda" in str(self.device)
            else "mps" if "mps" in str(self.device) else "cpu"
        )
        if device_type == "cpu":
            self.ctx = nullcontext()
        elif device_type == "cuda":
            dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            self.ctx = torch.autocast(device_type="cuda", dtype=dtype)
        elif device_type == "mps":
            dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            self.ctx = torch.autocast(device_type="mps", dtype=dtype)

        # 创建优化器
        self.optimizer = create_mixed_optimizer(
            self.model, self.args, self.ddp, self.training_type
        )

        # 处理检查点恢复
        self.start_epoch, self.start_step = self._handle_checkpoint_resume()

        # 包装DDP
        if self.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}  # type: ignore
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.ddp_local_rank]
            )

    def _handle_checkpoint_resume(self):
        """处理检查点恢复"""
        start_epoch = 0
        start_step = 0

        if self.args.resume:
            if os.path.exists(self.args.resume):
                start_epoch, start_step = load_checkpoint(
                    self.args.resume, self.model, self.optimizer, self.scaler, self.args
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
                    self.scaler,
                    self.args,
                )
            else:
                Logger("No checkpoint found for auto resume, starting from scratch")

        return start_epoch, start_step

    def create_data_loader(self, dataset):
        """创建数据加载器"""
        train_sampler = DistributedSampler(dataset) if self.ddp else None
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            pin_memory=getattr(self.args, "pin_memory", True),
            drop_last=True,
            shuffle=(train_sampler is None),
            num_workers=getattr(self.args, "num_workers", 1),
            sampler=train_sampler,
            prefetch_factor=(
                getattr(self.args, "prefetch_factor", 2)
                if getattr(self.args, "num_workers", 1) > 0
                else None
            ),
            persistent_workers=getattr(self.args, "persistent_workers", True)
            and getattr(self.args, "num_workers", 1) > 0,
        )

    def train_epoch(
        self,
        epoch,
        train_loader,
        iter_per_epoch,
        total_training_steps,
        wandb=None,
        profiler: Optional[torch.profiler.profile] = None,
        skip_steps: int = 0,
    ):
        """训练一个epoch"""
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        start_time = time.time()
        valid_loss_steps = 0

        self.model.train()

        for step, (X, Y, loss_mask) in enumerate(train_loader):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue

            # 非阻塞数据传输
            X = X.to(self.args.device, non_blocking=True)
            Y = Y.to(self.args.device, non_blocking=True)
            loss_mask = loss_mask.to(self.args.device, non_blocking=True)

            # 应用学习率调度
            global_step = epoch * iter_per_epoch + step
            apply_lr_schedule(
                self.optimizer,
                global_step,
                total_training_steps,
                self.args.warmup_iters,
            )

            # 前向传播
            with self.ctx:
                res = self.model(X)

                if (
                    step == 0 and epoch == self.start_epoch
                ):  # 只在训练开始时打印一次即可
                    Logger(f"--- Verification ---")
                    Logger(f"Tensor dtype inside autocast context: {res.logits.dtype}")
                    Logger(f"Expected dtype: {self.args.dtype}")
                    Logger(f"Device: {self.device}")
                    Logger(f"Autocast dtype: {torch.get_autocast_dtype('mps')}")
                    Logger(f"--------------------")

                logits_flat = res.logits.view(-1, res.logits.size(-1))
                targets_flat = Y.view(-1)
                loss = loss_fct(logits_flat, targets_flat).view(Y.size())

                # 计算有效损失
                loss_mask_sum = loss_mask.sum()
                if loss_mask_sum > 0:
                    loss = (loss * loss_mask).sum() / loss_mask_sum
                else:
                    loss = torch.tensor(
                        0.0, device=self.args.device, requires_grad=True
                    )

                # 添加辅助损失（如果存在）
                if hasattr(res, "aux_loss") and res.aux_loss is not None:
                    loss += res.aux_loss

                loss = loss / self.args.accumulation_steps

            # 记录有效步骤
            current_loss_mask_sum = loss_mask_sum.item()
            if current_loss_mask_sum > 0:
                valid_loss_steps += 1

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积和更新
            if (step + 1) % self.args.accumulation_steps == 0:
                grad_norm = 0.0
                if valid_loss_steps > 0:
                    if hasattr(self.optimizer, "unscale_gradients"):
                        self.optimizer.unscale_gradients(self.scaler)
                    else:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )

                    if hasattr(self.optimizer, "optimizers"):
                        for opt in self.optimizer.optimizers:
                            self.scaler.step(opt)
                    else:
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
                valid_loss_steps = 0

            # 性能分析
            if profiler is not None:
                profiler.step()

            # 日志记录
            if step % self.args.log_interval == 0:
                # 无论何时记录，都计算当前微批次的原始损失值
                # loss.item() 是已经被 accumulation_steps 缩放过的损失
                # 将其乘回去，就得到了当前单个微批次的原始损失，确保日志值量级一致
                current_loss = loss.item() * self.args.accumulation_steps

                # 梯度范数只在梯度更新步骤才会被计算
                grad_norm_to_log = 0.0
                if (step + 1) % self.args.accumulation_steps == 0:
                    if "grad_norm" in locals():
                        grad_norm_to_log = float(grad_norm)  # type: ignore

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
                )

            # 模型保存
            if (step + 1) % self.args.save_interval == 0 and (
                not self.ddp or dist.get_rank() == 0
            ):
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    epoch,
                    step,
                    self.args,
                    self.lm_config,
                    self.training_type,
                )

    def train(self, train_loader, wandb=None):
        """主训练循环"""
        iter_per_epoch = len(train_loader)
        total_training_steps = self.args.epochs * iter_per_epoch
        Logger(f"训练总步数: {total_training_steps}, 每轮步数: {iter_per_epoch}")

        # 设置性能分析器
        profiler = None
        if getattr(self.args, "profile", False) and (
            not self.ddp or dist.get_rank() == 0
        ):
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./log/profiler"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()

        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # 设置分布式采样器的epoch
                if (
                    self.ddp
                    and hasattr(train_loader, "sampler")
                    and train_loader.sampler is not None
                ):
                    train_loader.sampler.set_epoch(epoch)

                # 计算需要跳过的步骤
                skip_steps = self.start_step if epoch == self.start_epoch else 0

                self.train_epoch(
                    epoch,
                    train_loader,
                    iter_per_epoch,
                    total_training_steps,
                    wandb,
                    profiler,
                    skip_steps,
                )

                # 重置start_step
                if epoch == self.start_epoch:
                    self.start_step = 0

                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            if profiler is not None:
                profiler.stop()
