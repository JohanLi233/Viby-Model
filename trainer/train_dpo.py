import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import torch
import torch.nn.functional as F
from model.model import VibyConfig, VibyForCausalLM
from dataset.lm_dataset import DPODataset
from .base_trainer import BaseTrainer
from .config import get_dpo_parser, setup_training_args
from .utils import (
    build_model_and_tokenizer,
    init_wandb,
    load_model_weights,
    log_training_progress,
    move_model_to_device,
)

warnings.filterwarnings("ignore")


def logits_to_log_probs(logits, labels):
    """从logits中提取与labels对应的token对数概率。

    参数:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)

    返回:
        log_probs: (batch_size, seq_len) 的对数概率
    """
    log_probs = F.log_softmax(logits, dim=2)
    log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs


def dpo_loss(ref_log_probs, log_probs, mask, beta):
    """DPO 损失（基于序列平均log概率）。

    参数:
        ref_log_probs: (B, T) 参考模型对标签的log概率
        log_probs: (B, T) 当前模型对标签的log概率
        mask: (B, T) 有效位置为1，padding为0
        beta: DPO温度系数

    返回:
        标量损失
    """
    # 序列级平均log概率，避免除零
    lengths = mask.sum(dim=1).clamp_min(1)
    ref_seq_logp = (ref_log_probs * mask).sum(dim=1) / lengths
    seq_logp = (log_probs * mask).sum(dim=1) / lengths

    # 前半为chosen，后半为rejected
    batch_size = ref_seq_logp.shape[0]
    half = batch_size // 2
    chosen_ref = ref_seq_logp[:half]
    rejected_ref = ref_seq_logp[half:]
    chosen = seq_logp[:half]
    rejected = seq_logp[half:]

    # 计算logit并应用DPO目标
    pi_logratios = chosen - rejected
    ref_logratios = chosen_ref - rejected_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载SFT权重"""
    sft_checkpoint_name = getattr(
        args,
        "sft_checkpoint",
        f"full_sft_{lm_config.hidden_size}.pth",
    )
    model, tokenizer = build_model_and_tokenizer(
        lm_config,
        args,
        checkpoint_name=sft_checkpoint_name,
        checkpoint_label="SFT checkpoint",
    )

    # 初始化参考模型
    ref_model = move_model_to_device(VibyForCausalLM(lm_config), args)
    load_model_weights(
        ref_model,
        os.path.join(args.save_dir, sft_checkpoint_name),
        args.device,
        strict=False,
        label="SFT checkpoint",
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    return model, ref_model, tokenizer


class DPOTrainer(BaseTrainer):
    """DPO训练器，继承自BaseTrainer"""

    def __init__(self, args, model, ref_model, tokenizer, lm_config):
        # 先初始化父类
        super().__init__(args, model, tokenizer, lm_config, "dpo")
        self.ref_model = ref_model
        self.beta = getattr(args, "dpo_beta", 0.1)

    def train_epoch(
        self,
        epoch,
        train_loader,
        iter_per_epoch,
        total_training_steps,
        wandb=None,
        profiler=None,
        skip_steps=0,
    ):
        """DPO训练一个epoch"""
        import time
        from .utils import apply_lr_schedule

        start_time = time.time()
        base_step_offset_for_speed = skip_steps
        valid_loss_steps = 0
        last_grad_norm = 0.0

        self.model.train()

        for step, batch in enumerate(train_loader):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue

            # 非阻塞数据传输
            x_chosen = batch["x_chosen"].to(self.args.device, non_blocking=True)
            x_rejected = batch["x_rejected"].to(self.args.device, non_blocking=True)
            y_chosen = batch["y_chosen"].to(self.args.device, non_blocking=True)
            y_rejected = batch["y_rejected"].to(self.args.device, non_blocking=True)
            mask_chosen = batch["mask_chosen"].to(self.args.device, non_blocking=True)
            mask_rejected = batch["mask_rejected"].to(
                self.args.device, non_blocking=True
            )

            # 合并数据
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

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
                # 参考模型前向传播（不计算梯度）
                with torch.no_grad():
                    attn_mask = (x != self.tokenizer.pad_token_id).long()
                    ref_res = self.ref_model(
                        input_ids=x,
                        attention_mask=attn_mask,
                    )
                    ref_logits = ref_res.logits

                ref_log_probs = logits_to_log_probs(ref_logits, y)
                ref_log_probs = ref_log_probs * mask

                # 当前模型前向传播
                attn_mask = (x != self.tokenizer.pad_token_id).long()
                res = self.model(
                    input_ids=x,
                    attention_mask=attn_mask,
                )
                logits = res.logits
                log_probs = logits_to_log_probs(logits, y)
                log_probs = log_probs * mask

                # 计算DPO损失
                loss = dpo_loss(ref_log_probs, log_probs, mask, self.beta)
                loss = loss / self.args.accumulation_steps

            valid_loss_steps += 1

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积和更新
            if (step + 1) % self.args.accumulation_steps == 0:
                last_grad_norm = self._optimizer_step(valid_loss_steps)
                valid_loss_steps = 0

            # 性能分析
            if profiler is not None:
                profiler.step()

            # 日志记录
            if step % self.args.log_interval == 0:
                current_loss = loss.item() * self.args.accumulation_steps
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
                    0.0,  # attention_max_logit placeholder
                    base_step_offset=base_step_offset_for_speed,
                )

            # 模型保存
            self._save_if_needed(epoch, step)


if __name__ == "__main__":
    # 解析参数
    parser = get_dpo_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "dpo")

    # 创建模型配置
    lm_config = VibyConfig(
        max_position_embeddings=args.max_seq_len,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
        router_scoring_func=args.router_scoring_func,
        routed_scaling_factor=args.routed_scaling_factor,
        swiglu_limit=args.swiglu_limit,
        use_deepseek_v4_attention=args.use_deepseek_v4_attention,
        attention_sink=args.attention_sink,
        use_mhc=args.use_mhc,
        o_groups=args.o_groups,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        **({"q_lora_rank": args.q_lora_rank} if args.q_lora_rank is not None else {}),
        **({"o_lora_rank": args.o_lora_rank} if args.o_lora_rank is not None else {}),
        **(
            {"moe_intermediate_size": args.moe_intermediate_size}
            if args.moe_intermediate_size is not None
            else {}
        ),
    )

    # 初始化模型
    model, ref_model, tokenizer = init_model(lm_config, args)

    # 创建DPO训练器
    trainer = DPOTrainer(args, model, ref_model, tokenizer, lm_config)

    # 创建数据集和数据加载器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    wandb = init_wandb(args, trainer)

    # 开始训练
    trainer.train(train_loader, wandb)

# 执行命令示例:
#
# 标准DPO训练:
# python train_dpo.py
#
# 自定义配置:
# python train_dpo.py --data_path ../dataset/dpo.jsonl --max_seq_len 1024 --batch_size 4 --accumulation_steps 1 --learning_rate 1e-8
#
# 分布式训练:
# torchrun --nproc_per_node 2 train_dpo.py
