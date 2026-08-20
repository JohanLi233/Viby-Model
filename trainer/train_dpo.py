import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import mlx.core as mx
import mlx.nn as nn
from model.model import VibyConfig, VibyForCausalLM
from dataset.lm_dataset import DPODataset
from .base_trainer import BaseTrainer
from .config import get_dpo_parser, setup_training_args
from .utils import (
    Logger,
    build_config_from_sidecar,
    build_model_and_tokenizer,
    convert_model_dtype,
    init_swanlab,
    load_model_weights,
    log_training_progress,
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
    logits_f = logits.astype(mx.float32)
    log_probs = logits_f - mx.logsumexp(logits_f, axis=-1, keepdims=True)
    log_probs = mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
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
    mask_f = mask.astype(mx.float32)
    lengths = mx.maximum(mask_f.sum(axis=1), 1.0)
    ref_seq_logp = (ref_log_probs * mask_f).sum(axis=1) / lengths
    seq_logp = (log_probs * mask_f).sum(axis=1) / lengths

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
    loss = -nn.log_sigmoid(beta * logits).mean()
    return loss


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载SFT权重"""
    sft_checkpoint_name = (
        args.sft_checkpoint
        if getattr(args, "sft_checkpoint", None)
        else f"full_sft_{lm_config.hidden_size}.safetensors"
    )
    model, tokenizer = build_model_and_tokenizer(
        lm_config,
        args,
        checkpoint_name=sft_checkpoint_name,
        checkpoint_label="SFT checkpoint",
    )

    # 初始化参考模型
    ref_model = VibyForCausalLM(lm_config)
    convert_model_dtype(ref_model, getattr(args, "dtype", ""))
    # 与 policy 模型共用同一 config，严格加载：配置不一致时应直接报错，
    # 而不是静默跳过缺失参数
    load_model_weights(
        ref_model,
        os.path.join(args.save_dir, sft_checkpoint_name),
        strict=True,
        label="SFT checkpoint",
    )
    ref_model.eval()

    return model, ref_model, tokenizer


class DPOTrainer(BaseTrainer):
    """DPO训练器，继承自BaseTrainer"""

    def __init__(self, args, model, ref_model, tokenizer, lm_config):
        # beta 需在 super().__init__（内部构建 loss 函数）之前设置
        self.beta = getattr(args, "dpo_beta", 0.1)
        # 先初始化父类
        super().__init__(args, model, tokenizer, lm_config, "dpo")
        self.ref_model = ref_model

    def _dpo_loss_fn(self, params, x, y, mask, attn_mask, mask_has_pad, ref_log_probs):
        """DPO loss 函数（value_and_grad 的目标）。

        params 必须显式作为第一个入参：mx.compile 会捕获 Python 闭包里的
        常量，若走 nn.value_and_grad(model, ...) + mx.compile，梯度会永远
        基于 trace 时的初始权重，优化器更新完全无效。
        mask_has_pad 在 eager 侧计算后以 python bool 传入（compile 图内
        不允许 .item() host sync）。
        """
        self.model.update(params)
        res = self.model(
            input_ids=x,
            attention_mask=attn_mask,
            mask_has_pad=mask_has_pad,
        )
        log_probs = logits_to_log_probs(res.logits, y)
        log_probs = log_probs * mask
        loss = dpo_loss(ref_log_probs, log_probs, mask, self.beta)
        return loss / self.args.accumulation_steps

    def _build_loss_and_grad(self):
        fn = mx.value_and_grad(self._dpo_loss_fn)
        use_compile = getattr(self.args, "compile_model", False)
        if use_compile and getattr(self, "_moe_gates", []):
            Logger(
                "MoE 稀疏桶前向含数据依赖形状（host sync），mx.compile 不可用，"
                "自动回退 eager；如要 compile 请关闭 MoE"
            )
            use_compile = False
        self._compiled = bool(use_compile)
        if use_compile:
            Logger("使用 mx.compile 编译 loss 函数")
            fn = mx.compile(fn)
        return fn

    def train_epoch(
        self,
        epoch,
        train_loader,
        iter_per_epoch,
        total_training_steps,
        swanlab=None,
        skip_steps=0,
    ):
        """DPO训练一个epoch"""
        import time

        from mlx.utils import tree_map

        from .utils import apply_lr_schedule

        start_time = time.time()
        base_step_offset_for_speed = skip_steps
        accum_grads = None
        accum_count = 0
        last_grad_norm = 0.0

        self.model.train()
        self.ref_model.eval()

        for step, batch in enumerate(train_loader):
            # 跳过步骤（恢复训练时）
            if step < skip_steps:
                continue

            # 合并数据
            x = mx.concatenate([batch["x_chosen"], batch["x_rejected"]], axis=0)
            y = mx.concatenate([batch["y_chosen"], batch["y_rejected"]], axis=0)
            mask = mx.concatenate(
                [batch["mask_chosen"], batch["mask_rejected"]], axis=0
            )

            # 应用学习率调度（每微批 step 应用）
            global_step = epoch * iter_per_epoch + step
            apply_lr_schedule(
                self.optimizer,
                global_step,
                total_training_steps,
                self.args.warmup_iters,
            )

            # 参考模型前向传播（不走 value_and_grad，不会计算梯度）
            attn_mask = (x != self.tokenizer.pad_token_id).astype(mx.int32)
            # mx.compile 图内不允许 .item() host sync，在 eager 侧预计算
            mask_has_pad = bool(mx.any(attn_mask != 1).item())
            ref_res = self.ref_model(
                input_ids=x,
                attention_mask=attn_mask,
                mask_has_pad=mask_has_pad,
            )
            ref_log_probs = logits_to_log_probs(ref_res.logits, y)
            ref_log_probs = ref_log_probs * mask
            mx.eval(ref_log_probs)

            # 当前模型前向 + 反向（loss 内部已除以 accumulation_steps）。
            # 参数显式传入：compile 下保证梯度基于当前权重而非初始快照。
            params = self.model.trainable_parameters()
            loss, grads = self._loss_and_grad(
                params, x, y, mask, attn_mask, mask_has_pad, ref_log_probs
            )
            if self._compiled:
                # compiled fn 内部的 model.update(params) 只在 trace 时执行，
                # 会把占位数组留在 module 上；立即恢复真实参数
                self.model.update(params)
            # 立即物化本微批的 loss/grads 并释放反向图（与 BaseTrainer 同理，
            # 避免 accumulation_steps 个惰性图同时存活导致显存倍增）。
            # 前向 loss 与梯度分两次 eval：合并 eval 会把 tape 滞留与反向
            # 临时量同时顶到峰值，拆开显存省 ~3×、单步更快（数值不变）。
            mx.eval(loss)
            mx.eval(grads)

            # MoE 稀疏桶容量表滚动更新（与 BaseTrainer 同理：counts 已随
            # eval 物化，此处 tolist 不钉图；稀疏前向无 host sync 的前提）
            for _m in self.model.modules():
                _f = getattr(_m, "update_capacity_table", None)
                if _f is not None:
                    _f()

            # 梯度累加
            accum_grads = (
                grads if accum_grads is None else tree_map(mx.add, accum_grads, grads)
            )
            accum_count += 1

            # 梯度累积窗口结束，执行更新
            if (step + 1) % self.args.accumulation_steps == 0:
                last_grad_norm = self._optimizer_step(accum_grads, accum_count)
                accum_grads = None
                accum_count = 0

            # 日志记录
            if step % self.args.log_interval == 0:
                mx.eval(loss)
                current_loss = float(loss.item()) * self.args.accumulation_steps
                grad_norm_to_log = last_grad_norm

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
                )

            # 模型保存
            self._save_if_needed(epoch, step)


if __name__ == "__main__":
    # 解析参数
    parser = get_dpo_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "dpo")

    # 优先从 SFT checkpoint 的 sidecar config 继承模型结构（含 YaRN/
    # rope_scaling），CLI 显式传入的参数优先；无 sidecar 时回退库默认值。
    checkpoint_name = (
        args.sft_checkpoint
        if getattr(args, "sft_checkpoint", None)
        else f"full_sft_{args.hidden_size}.safetensors"
    )
    cfg, has_sidecar = build_config_from_sidecar(args, checkpoint_name)
    # DPO 上下文不低于 SFT 的实际上下文，保证继承 YaRN/rope 配置时行为一致
    cfg["max_position_embeddings"] = max(
        args.max_seq_len, cfg.get("max_position_embeddings", 0)
    )
    lm_config = VibyConfig.from_dict(cfg) if has_sidecar else VibyConfig(**cfg)

    # 初始化模型
    model, ref_model, tokenizer = init_model(lm_config, args)

    # 创建DPO训练器
    trainer = DPOTrainer(args, model, ref_model, tokenizer, lm_config)

    # 创建数据集和数据加载器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    swanlab = init_swanlab(args, trainer)

    # 开始训练
    trainer.train(train_loader, swanlab)
    if swanlab is not None:
        swanlab.finish()

# 执行命令示例:
#
# 标准DPO训练:
# python train_dpo.py
#
# 自定义配置:
# python train_dpo.py --data_path ../dataset/dpo.jsonl --max_seq_len 1024 --batch_size 4 --accumulation_steps 1 --learning_rate 1e-8
