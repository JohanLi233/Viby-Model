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
    build_model_kwargs,
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

    # 创建模型配置：CLI 结构参数（含 --preset 回填）→ VibyConfig kwargs，
    # 未显式传入的字段由 VibyConfig 的 ≈1B 默认配方补齐
    lm_config = VibyConfig(**build_model_kwargs(args))
    if lm_config.dpr_enabled:
        from .utils import Logger

        Logger(
            f"DPR-JEPA ({lm_config.dpr_variant}): full token CED, boundary={lm_config.n_encoder_layers}, "
            f"M={lm_config.dpr_particles}, r={lm_config.dpr_dim}, k={lm_config.dpr_horizon}, "
            f"w={lm_config.dpr_width}, objective={lm_config.dpr_objective}, "
            f"lambda={lm_config.dpr_loss_weight}; targets used only in auxiliary loss"
        )
    if lm_config.ced_recurrent_enabled:
        from .utils import Logger

        Logger(
            f"Residual-lifted recurrent CED: k={lm_config.ced_recurrent_stride}, "
            f"q={lm_config.ced_recurrent_rounds}, full evidence KV, shared middle blocks; "
            "NTP + original MoE balancing, MTP/protected PSR off. "
            "Logical FLOPs estimates do not establish timing or quality gains."
        )
    if lm_config.psr_enabled:
        from .utils import Logger

        Logger(
            f"Protected PSR: output-only zero head, dense R={lm_config.psr_rounds}, "
            f"H={lm_config.psr_horizon}, anchors/row={lm_config.psr_train_anchors}; "
            "direct NTP, detached backbone, independent optimizer"
        )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 数据集先于 trainer 创建：lr_scale_auto 在未传 --token_budget 时按
    # 与 LR 日程相同的 horizon（max_steps / 全量 epoch）推导 token 预算
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=getattr(args, "pack_sequences", False),
        doc_mask=getattr(args, "doc_mask", False),
        align_docs=getattr(args, "doc_align", True),
        max_doc_len=getattr(args, "max_doc_len", None),
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

    if lm_config.dpr_enabled:
        import json

        with open(os.path.join(args.save_dir, "run_config.json"), "w") as file:
            json.dump(
                {"args": vars(args), "model": lm_config.to_dict()}, file, indent=2
            )
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
