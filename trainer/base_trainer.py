"""
基础训练器类，提供通用的训练逻辑
"""

import os
import time
import torch
import torch.distributed as dist
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
        # 根据可用硬件选择合适的后端
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.ddp_local_rank)
            self.args.device = torch.device(self.device)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS 暂不支持 NCCL，使用 Gloo，并保持单设备 'mps'
            self.device = "mps"
            self.args.device = torch.device("mps")
        else:
            self.device = "cpu"
            self.args.device = torch.device("cpu")

        # 设置随机种子
        base_seed = 1337
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed + rank)

    def _init_training_components(self):
        """初始化训练组件"""
        # 自动类型转换上下文
        device_type = (
            "cuda"
            if "cuda" in str(self.device)
            else "mps"
            if "mps" in str(self.device)
            else "cpu"
        )

        # 混合精度训练（MPS 也启用 GradScaler）
        use_grad_scaler = device_type in ("cuda", "mps") and (
            self.args.dtype in ["float16", "bfloat16"]
        )
        self.scaler = torch.amp.GradScaler(enabled=use_grad_scaler)  # type: ignore
        if device_type == "cpu":
            self.ctx = nullcontext()
        elif device_type == "cuda":
            dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            self.ctx = torch.autocast(device_type="cuda", dtype=dtype)
        elif device_type == "mps":
            dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            self.ctx = torch.autocast(device_type="mps", dtype=dtype)

        # 创建优化器
        use_muon_clip = getattr(self.args, 'use_muon_clip', False)
        self.optimizer = create_mixed_optimizer(
            self.model, self.args, self.ddp, self.training_type, use_muon_clip
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
        start_time = time.time()
        base_step_offset_for_speed = skip_steps
        valid_loss_steps = 0
        last_grad_norm = 0.0  # Store last calculated gradient norm

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
                # 构造 attention_mask，屏蔽 PAD 位置，确保注意力与 Canon 不受填充影响
                attn_mask = (X != self.tokenizer.pad_token_id).long()
                # 直接将 labels、loss_mask、attention_mask 传递给模型
                res = self.model(
                    input_ids=X,
                    labels=Y,
                    loss_mask=loss_mask,
                    attention_mask=attn_mask,
                )
                loss = res.loss  # 使用模型返回的loss
                
                # 获取真实的注意力 max logit（用于 QK-Clip）
                attention_max_logit = 0.0
                
                # 更新注意力统计信息（如果启用了 MuonClip）
                if hasattr(self.optimizer, 'apply_qk_clip'):
                    # 获取原始模型（去掉包装）
                    model_for_stats = self.model
                    if hasattr(self.model, 'module'):
                        model_for_stats = self.model.module
                    if hasattr(model_for_stats, '_orig_mod'):
                        model_for_stats = model_for_stats._orig_mod
                    
                    if hasattr(model_for_stats, 'update_attention_stats_from_forward'):
                        # 现在这会更新并聚合 model_for_stats.attention_max_logits
                        model_for_stats.update_attention_stats_from_forward()
                        
                        # [更改] 从聚合的张量中获取最大 logit 用于日志记录
                        if hasattr(model_for_stats, 'attention_max_logits'):
                             # 我们取当前缓冲区状态的最大值
                             attention_max_logit = model_for_stats.attention_max_logits.max().item()

                if (
                    step == 0 and epoch == self.start_epoch
                ):  # 只在训练开始时打印一次即可
                    Logger("--- Verification ---")
                    Logger(f"Tensor dtype inside autocast context: {res.logits.dtype}")
                    Logger(f"Expected dtype: {self.args.dtype}")
                    Logger(f"Device: {self.device}")
                    # 安全打印 autocast dtype
                    dev_str = str(self.device)
                    dev_type = (
                        "cuda"
                        if "cuda" in dev_str
                        else "mps"
                        if "mps" in dev_str
                        else "cpu"
                    )
                    try:
                        if dev_type == "cuda" and hasattr(
                            torch, "get_autocast_gpu_dtype"
                        ):
                            Logger(f"Autocast dtype: {torch.get_autocast_gpu_dtype()}")
                        elif dev_type == "cpu" and hasattr(
                            torch, "get_autocast_cpu_dtype"
                        ):
                            Logger(f"Autocast dtype: {torch.get_autocast_cpu_dtype()}")
                        else:
                            Logger("Autocast dtype: n/a")
                    except Exception:
                        Logger("Autocast dtype: n/a")
                    Logger("--------------------")

                loss = loss / self.args.accumulation_steps

            # 记录有效步骤
            current_loss_mask_sum = loss_mask.sum().item()
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
                    last_grad_norm = float(
                        grad_norm
                    )  # Update last calculated grad norm

                    # 对于 MuonClip 优化器，先设置模型引用
                    if hasattr(self.optimizer, 'apply_qk_clip'):
                        # 获取原始模型（去掉 DDP 和 compile 包装）
                        model_to_pass = self.model
                        if hasattr(self.model, 'module'):
                            model_to_pass = self.model.module
                        if hasattr(model_to_pass, '_orig_mod'):
                            model_to_pass = model_to_pass._orig_mod
                        
                        # 临时设置模型引用
                        self.optimizer._temp_model_ref = model_to_pass
                    
                    if hasattr(self.optimizer, "optimizers"):
                        for opt in self.optimizer.optimizers:
                            self.scaler.step(opt)
                    else:
                        self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # [添加] 在更新之后重置 QK-Clip 统计缓冲区
                    if hasattr(self.optimizer, 'apply_qk_clip'):
                        # 获取原始模型（去掉 DDP 和 compile 包装）
                        model_to_pass = self.model
                        if hasattr(self.model, 'module'):
                            model_to_pass = self.model.module
                        if hasattr(model_to_pass, '_orig_mod'):
                            model_to_pass = model_to_pass._orig_mod
                        
                        if model_to_pass is not None and hasattr(model_to_pass, 'attention_max_logits'):
                            # 将缓冲区重置为零，为下一个累积周期做准备
                            model_to_pass.attention_max_logits.data.zero_()
                    
                    # 清理临时引用
                    if hasattr(self.optimizer, '_temp_model_ref'):
                         self.optimizer._temp_model_ref = None
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

                # 使用上次计算的梯度范数
                grad_norm_to_log = last_grad_norm

                # 获取 QK-Clip 统计信息（如果使用 MuonClip 优化器）
                qk_clip_stats = None
                if hasattr(self.optimizer, 'qk_clip_stats'):
                    qk_clip_stats = self.optimizer.qk_clip_stats.copy()

                # 使用注意力 max logit 用于日志记录
                logit_to_log = attention_max_logit
                
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
                    logit_to_log,
                    base_step_offset=base_step_offset_for_speed,
                    qk_clip_stats=qk_clip_stats,
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
        # torch.autograd.set_detect_anomaly(True)
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
