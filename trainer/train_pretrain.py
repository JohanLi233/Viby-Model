import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import torch
from transformers import AutoTokenizer
from model.model import VibyConfig, VibyForCausalLM
from dataset.lm_dataset import PretrainDataset
from .base_trainer import BaseTrainer
from .config import get_pretrain_parser, setup_training_args
from .utils import Logger

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    # 初始化模型
    model = VibyForCausalLM(lm_config)

    # 设置设备与dtype
    target_dtype = None
    if args.dtype == "bfloat16":
        target_dtype = torch.bfloat16
    elif args.dtype == "float16":
        target_dtype = torch.float16
    if target_dtype is not None:
        model = model.to(device=args.device, dtype=target_dtype)  # type: ignore
    else:
        model = model.to(device=args.device)  # type: ignore

    # 编译模型以提升性能
    model = torch.compile(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Logger(
        f"总参数量：{total_params / 1e6:.3f}M, 可训练参数量：{trainable_params / 1e6:.3f}M"
    )

    return model, tokenizer


if __name__ == "__main__":
    # 解析参数
    parser = get_pretrain_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "pretrain")

    # 创建模型配置
    lm_config = VibyConfig()

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "pretrain")

    # 创建数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    # 初始化wandb
    wandb = None
    if args.use_wandb and (not trainer.ddp or trainer.ddp_local_rank == 0):
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        except ImportError:
            Logger("Warning: wandb not installed, logging disabled")
            wandb = None

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
