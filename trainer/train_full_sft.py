import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from model.config import VibyConfig
from dataset.lm_dataset import SFTDataset
from .base_trainer import BaseTrainer
from .config import get_sft_parser, setup_training_args
from .utils import (
    base_checkpoint_name,
    build_config_from_sidecar,
    build_model_and_tokenizer,
    finish_training,
    init_swanlab,
    sidecar_checkpoint_hint,
)

warnings.filterwarnings("ignore")


def init_model(lm_config, args):
    """初始化模型和tokenizer，加载预训练权重"""
    checkpoint_name = base_checkpoint_name(
        args, lm_config, "pretrain_checkpoint", "pretrain"
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
    # 先按 --pretrain_checkpoint / --hidden_size（缺省即 VibyConfig 默认 dim）
    # 猜出基座文件名，再读它的 sidecar config 继承结构；CLI 显式参数优先
    checkpoint_name = sidecar_checkpoint_hint(args, "pretrain_checkpoint", "pretrain")
    cfg, has_sidecar = build_config_from_sidecar(args, checkpoint_name)
    # SFT 的上下文长度由 max_seq_len 决定
    cfg["max_position_embeddings"] = args.max_seq_len
    lm_config = VibyConfig.from_dict(cfg) if has_sidecar else VibyConfig(**cfg)

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "sft")

    # 创建数据集和数据加载器（打包模式与 pretrain 同口径：定长块、无 pad、不截断）
    if getattr(args, "doc_mask", False) and not getattr(args, "pack_sequences", False):
        raise ValueError("--doc_mask 必须与 --pack_sequences 同时使用")
    train_ds = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=getattr(args, "pack_sequences", False),
        doc_mask=getattr(args, "doc_mask", False),
        empty_think_ratio=getattr(args, "empty_think_ratio", 0.0),
    )
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
# 标准SFT训练:
# python train_full_sft.py
#
# 自定义配置:
# python train_full_sft.py --data_path /Volumes/pan/sft_512.jsonl --max_seq_len 1024 --batch_size 8 --accumulation_steps 4
# python train_full_sft.py --data_path /Volumes/pan/sft_1024.jsonl --max_seq_len 1024 --batch_size 8 --accumulation_steps 4
# python train_full_sft.py --data_path /Volumes/pan/sft_2048.jsonl --max_seq_len 2048 --batch_size 4 --accumulation_steps 4
#
# python train_full_sft.py --data_path /Volumes/pan/sft_2048.jsonl --max_seq_len 2048 --batch_size 4 --accumulation_steps 4
