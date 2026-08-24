import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from model.config import VibyConfig
from dataset.lm_dataset import PretrainDataset
from .base_trainer import BaseTrainer
from .config import get_pretrain_parser, setup_training_args
from .utils import (
    build_model_and_tokenizer,
    finish_training,
    init_swanlab,
    resolve_compute_scaled_hparams,
)

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer"""
    return build_model_and_tokenizer(
        lm_config,
        args,
    )


if __name__ == "__main__":
    # 解析参数
    parser = get_pretrain_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "pretrain")
    if getattr(args, "doc_mask", False) and not getattr(args, "pack_sequences", False):
        raise ValueError("--doc_mask 必须与 --pack_sequences 同时使用")

    # 创建模型配置
    lm_config = VibyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        use_attn_gate=args.use_attn_gate,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        routed_scaling_factor=args.routed_scaling_factor,
        moe_router_logit_norm=args.moe_router_logit_norm,
        moe_router_logit_temp=args.moe_router_logit_temp,
        moe_diversity_loss_weight=args.moe_diversity_loss_weight,
        z_loss_weight=args.z_loss_weight,
        moe_latent_dim=args.moe_latent_dim,
        tie_word_embeddings=args.tie_word_embeddings,
        **({"head_dim": args.head_dim} if args.head_dim is not None else {}),
        **(
            {"intermediate_size": args.intermediate_size}
            if args.intermediate_size is not None
            else {}
        ),
    )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 数据集先于 trainer 创建：lr_scale_auto 在未传 --token_budget 时按
    # epochs × 每轮步数推导总 token 预算，需要数据集长度
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=getattr(args, "pack_sequences", False),
        doc_mask=getattr(args, "doc_mask", False),
    )
    iter_per_epoch = len(train_ds) // args.batch_size  # drop_last 口径
    args = resolve_compute_scaled_hparams(args, iter_per_epoch)
    # setup_training_args 早于解析运行，run name 里的 lr 此时刷新为解析值
    args.swanlab_run_name = args.swanlab_run_name.replace(
        "-LRauto", f"-LR{args.learning_rate:.4g}"
    )

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "pretrain")

    train_loader = trainer.create_data_loader(train_ds)

    swanlab = init_swanlab(args, trainer)

    try:
        trainer.train(train_loader, swanlab)
    except KeyboardInterrupt:
        trainer.interrupted = True
    finally:
        finish_training(swanlab, interrupted=trainer.interrupted)

# 执行命令示例:
#
# 标准训练 (短序列, 大批量):
# python train_pretrain.py --batch_size 32 --learning_rate 0.01 --accumulation_steps 8 --max_seq_len 640
#
# 长序列训练:
# python train_pretrain.py --batch_size 16 --learning_rate 0.005 --accumulation_steps 4 --max_seq_len 1024 --log_interval 1
