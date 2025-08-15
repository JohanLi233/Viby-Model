"""
训练配置和参数解析模块
"""

import argparse
import torch


def add_common_args(parser):
    """添加通用参数"""
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "mps"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="When resuming, do not load optimizer/scaler states and restart from step 0",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    
    # MuonClip 优化器相关参数
    parser.add_argument("--use_muon_clip", action="store_true", 
                       help="使用 MuonClip 优化器而不是标准 Muon")
    parser.add_argument("--qk_clip_tau", type=float, default=25.0, 
                       help="QK-Clip 阈值，默认 25.0")


def get_pretrain_parser():
    """获取预训练参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Pretraining")
    add_common_args(parser)

    # 预训练特定参数
    parser.set_defaults(
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        accumulation_steps=8,
        max_seq_len=1024,
    )

    parser.add_argument("--wandb_project", type=str, default="Viby-Pretrain")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")

    return parser


def get_sft_parser():
    """获取SFT参数解析器"""
    parser = argparse.ArgumentParser(description="Viby Full SFT")
    add_common_args(parser)

    # SFT特定参数
    parser.set_defaults(
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        accumulation_steps=1,
        max_seq_len=2048,
    )

    parser.add_argument("--wandb_project", type=str, default="Viby-Full-SFT")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl")

    # YaRN scaling parameters
    parser.add_argument(
        "--enable_yarn", action="store_true", help="Enable YaRN scaling"
    )
    parser.add_argument(
        "--yarn_scaling_factor", default=2.0, type=float, help="YaRN scaling factor"
    )
    parser.add_argument(
        "--original_max_seq_len",
        default=512,
        type=int,
        help="Original context length before scaling",
    )
    parser.add_argument(
        "--yarn_beta_fast", default=32.0, type=float, help="YaRN beta_fast parameter"
    )
    parser.add_argument(
        "--yarn_beta_slow", default=1.0, type=float, help="YaRN beta_slow parameter"
    )

    return parser


def setup_training_args(args, training_type="pretrain"):
    """设置训练参数"""
    import os

    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置wandb运行名称
    if training_type == "pretrain":
        args.wandb_run_name = f"Viby-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    else:
        args.wandb_run_name = f"Viby-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置随机种子
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    return args
