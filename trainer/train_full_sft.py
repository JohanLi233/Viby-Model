import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import torch
from transformers import AutoTokenizer
from model.model import VibyConfig, VibyForCausalLM
from dataset.lm_dataset import SFTDataset
from .base_trainer import BaseTrainer
from .config import get_sft_parser, setup_training_args
from .utils import Logger

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载预训练权重"""
    model_path = getattr(args, 'model_path', '../model/')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    # 加载预训练模型
    pretrain_checkpoint_name = getattr(args, 'pretrain_checkpoint', f'pretrain_{lm_config.hidden_size}.pth')
    ckp = f"{args.save_dir}/{pretrain_checkpoint_name}"

    if os.path.exists(ckp):
        # 如果是完整的检查点文件
        checkpoint = torch.load(ckp, map_location=args.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            # 旧格式，直接是状态字典
            model.load_state_dict(checkpoint, strict=False)
    else:
        Logger(f"Warning: Pretrain checkpoint {ckp} not found, starting from scratch")

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
    parser = get_sft_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "sft")

    # 创建模型配置
    # 自动启用YaRN当max_seq_len >= 2048时
    rope_scaling = None
    original_max_seq_len = args.max_seq_len

    if args.max_seq_len >= 2048 or args.enable_yarn:
        # 当序列长度为2048或更长时，自动启用YaRN
        if not hasattr(args, "yarn_scaling_factor") or args.yarn_scaling_factor == 2.0:
            # 根据序列长度自动计算缩放因子
            args.yarn_scaling_factor = args.max_seq_len / args.original_max_seq_len

        rope_scaling = {
            "type": "yarn",
            "factor": args.yarn_scaling_factor,
            "beta_fast": getattr(args, "yarn_beta_fast", 32.0),
            "beta_slow": getattr(args, "yarn_beta_slow", 1.0),
        }
        Logger(
            f"[YaRN] 启用上下文扩展: {args.original_max_seq_len} → {args.max_seq_len} (scaling factor: {args.yarn_scaling_factor})"
        )

    lm_config = VibyConfig(
        max_position_embeddings=args.max_seq_len,
        original_max_position_embeddings=args.original_max_seq_len,
        rope_scaling=rope_scaling,
    )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "sft")

    # 创建数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
