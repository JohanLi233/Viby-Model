import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from model.model import VibyConfig
from dataset.lm_dataset import PretrainDataset
from .base_trainer import BaseTrainer
from .config import get_pretrain_parser, setup_training_args
from .utils import build_model_and_tokenizer, init_wandb

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer"""
    compile_mode = getattr(args, "compile_mode", None)
    if compile_mode is None and str(args.device).startswith("cuda"):
        compile_mode = "max-autotune"
    return build_model_and_tokenizer(
        lm_config,
        args,
        compile_mode=compile_mode,
    )


if __name__ == "__main__":
    # 解析参数
    parser = get_pretrain_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "pretrain")

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
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "pretrain")

    # 创建数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    wandb = init_wandb(args, trainer)

    # 开始训练
    trainer.train(train_loader, wandb)

# 执行命令示例:
#
# 标准训练 (短序列, 大批量):
# python train_pretrain.py --batch_size 32 --learning_rate 0.01 --accumulation_steps 8 --max_seq_len 640
#
# 长序列训练:
# python train_pretrain.py --batch_size 16 --learning_rate 0.005 --accumulation_steps 4 --max_seq_len 1024 --log_interval 1
#
# 分布式训练:
# torchrun --nproc_per_node 2 train_pretrain.py
