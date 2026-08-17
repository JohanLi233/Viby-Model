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


def _parse_hrm_bp_cycles(raw):
    if raw is None or not str(raw).strip():
        return None
    return [max(1, int(x)) for x in str(raw).split(",")]


def _parse_int_tuple(raw):
    if raw is None or not str(raw).strip():
        return ()
    return tuple(int(x) for x in str(raw).split(",") if str(x).strip())


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
    hrm_bp_cycles = _parse_hrm_bp_cycles(getattr(args, "hrm_bp_cycles", None))
    engram_layers = _parse_int_tuple(getattr(args, "engram_layers", ""))
    engram_orders = _parse_int_tuple(getattr(args, "engram_orders", "2,3"))

    # 创建模型配置
    lm_config = VibyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
        use_value_res=args.use_value_res,
        use_attn_gate=args.use_attn_gate,
        loop_k=args.loop_k,
        dw_rank=args.dw_rank,
        ws_loop=args.ws_loop,
        hrm_H_cycles=args.hrm_H_cycles,
        hrm_L_cycles=args.hrm_L_cycles,
        hrm_bp_cycles=hrm_bp_cycles,
        hrm_emb_scale=args.hrm_emb_scale,
        ffn_type=args.ffn_type,
        zero_centered_norm=args.zero_centered_norm,
        use_res_gate=args.use_res_gate,
        sandwich_norm=args.sandwich_norm,
        san_res_init=args.san_res_init,
        emb_scale=args.emb_scale,
        engram_layers=engram_layers,
        engram_orders=engram_orders,
        engram_heads=args.engram_heads,
        engram_slots=args.engram_slots,
        engram_sub_dim=args.engram_sub_dim,
        **({"head_dim": args.head_dim} if args.head_dim is not None else {}),
        **(
            {"intermediate_size": args.intermediate_size}
            if args.intermediate_size is not None
            else {}
        ),
    )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args)

    # 创建训练器
    trainer = BaseTrainer(args, model, tokenizer, lm_config, "pretrain")

    # 创建数据集和数据加载器
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=getattr(args, "pack_sequences", False),
        doc_mask=getattr(args, "doc_mask", False),
    )
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
