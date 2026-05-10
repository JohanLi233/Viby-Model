import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from model.model import VibyConfig
from dataset.lm_dataset import SFTDataset
from .base_trainer import BaseTrainer
from .config import build_sft_rope_scaling, get_sft_parser, setup_training_args
from .utils import build_model_and_tokenizer, init_wandb

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载预训练权重"""
    checkpoint_name = getattr(
        args,
        "pretrain_checkpoint",
        f"pretrain_{lm_config.hidden_size}.pth",
    )
    return build_model_and_tokenizer(
        lm_config,
        args,
        checkpoint_name=checkpoint_name,
        checkpoint_label="Pretrain checkpoint",
    )


if __name__ == "__main__":
    # 解析参数
    parser = get_sft_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "sft")

    lm_config = VibyConfig(
        max_position_embeddings=args.max_seq_len,
        original_max_position_embeddings=args.original_max_seq_len,
        rope_scaling=build_sft_rope_scaling(args),
    )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "sft")

    # 创建数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    wandb = init_wandb(args, trainer)

    # 开始训练
    trainer.train(train_loader, wandb)

# 执行命令示例:
#
# 标准SFT训练:
# python train_full_sft.py
#
# 自定义配置:
# python train_full_sft.py --data_path /Volumes/pan/sft_512.jsonl --max_seq_len 1024 --batch_size 8 --accumulation_steps 4
# python train_full_sft.py --data_path /Volumes/pan/sft_1024.jsonl --max_seq_len 1024 --batch_size 8 --accumulation_steps 4
# python train_full_sft.py --data_path /Volumes/pan/sft_2048.jsonl --max_seq_len 2048 --batch_size 4 --accumulation_steps 4
#
# 使用YaRN进行2048长度训练 (会自动启用):
# python train_full_sft.py --data_path /Volumes/pan/sft_2048.jsonl --max_seq_len 2048 --batch_size 4 --accumulation_steps 4
#
# 分布式训练:
# torchrun --nproc_per_node 2 train_full_sft.py
