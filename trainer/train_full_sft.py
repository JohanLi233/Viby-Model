import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from model.model import VibyConfig
from dataset.lm_dataset import SFTDataset
from .base_trainer import BaseTrainer
from .config import build_sft_rope_scaling, get_sft_parser, setup_training_args
from .utils import build_config_from_sidecar, build_model_and_tokenizer, init_swanlab

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载预训练权重"""
    checkpoint_name = (
        args.pretrain_checkpoint
        if getattr(args, "pretrain_checkpoint", None)
        else f"pretrain_{lm_config.hidden_size}.safetensors"
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

    # 优先从 pretrain checkpoint 的 sidecar config 继承模型结构配置，
    # CLI 显式传入的参数优先；无 sidecar 时回退 VibyConfig 库默认值，
    # 避免 pretrain/SFT 结构参数不一致导致 strict 加载失败。
    checkpoint_name = (
        args.pretrain_checkpoint
        if getattr(args, "pretrain_checkpoint", None)
        else f"pretrain_{args.hidden_size}.safetensors"
    )
    cfg, has_sidecar = build_config_from_sidecar(args, checkpoint_name)
    # SFT 的上下文长度由 max_seq_len 决定；YaRN 的"原始长度"取 pretrain
    # 的实际训练上下文（sidecar），而不是硬编码 1024
    pretrain_ctx = cfg.get("max_position_embeddings")
    cfg["max_position_embeddings"] = args.max_seq_len
    if args.original_max_seq_len is None:
        args.original_max_seq_len = pretrain_ctx or 1024
    cfg["original_max_position_embeddings"] = args.original_max_seq_len
    cfg["rope_scaling"] = build_sft_rope_scaling(args)
    lm_config = VibyConfig.from_dict(cfg) if has_sidecar else VibyConfig(**cfg)

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "sft")

    # 创建数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = trainer.create_data_loader(train_ds)

    swanlab = init_swanlab(args, trainer)

    # 开始训练
    trainer.train(train_loader, swanlab)
    if swanlab is not None:
        swanlab.finish()

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
