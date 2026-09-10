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
        mtp_steps=args.mtp_steps,
        use_attn_gate=args.use_attn_gate,
        use_xsa=args.use_xsa,
        xsa_last_n=getattr(args, "xsa_last_n", 0),
        hidden_act=getattr(args, "hidden_act", "silu"),
        attn_res_window=getattr(args, "attn_res_window", 4),
        attn_res_register=getattr(args, "attn_res_register", False),
        attn_res_read_h=getattr(args, "attn_res_read_h", False),
        ihc=getattr(args, "ihc", False),
        ihc_streams=getattr(args, "ihc_streams", 4),
        ihc_typed=getattr(args, "ihc_typed", False),
        ihc_collapse=getattr(args, "ihc_collapse", None),
        ihc_ngram_stream=getattr(args, "ihc_ngram_stream", 1),
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
        moe_write_spread=getattr(args, "moe_write_spread", False),
        moe_route_scale=getattr(args, "moe_route_scale", False),
        first_k_dense_replace=getattr(args, "first_k_dense_replace", 0),
        **(
            {"dense_intermediate_size": args.dense_intermediate_size}
            if getattr(args, "dense_intermediate_size", None) is not None
            else {}
        ),
        kda_v_head_ratio=args.kda_v_head_ratio,
        ngram_table_size=args.ngram_table_size,
        ngram_layer=args.ngram_layer,
        ngram_heads=getattr(args, "ngram_heads", 8),
        ngram_logit_skip=getattr(args, "ngram_logit_skip", False),
        ngram_conf_gate=getattr(args, "ngram_conf_gate", False),
        tie_word_embeddings=args.tie_word_embeddings,
        use_linear_attn=args.use_linear_attn,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        loop_span=getattr(args, "loop_span", 0),
        loop_count=getattr(args, "loop_count", 2),
        loop_res_scale=getattr(args, "loop_res_scale", "rsqrt"),
        loop_grad_mode=getattr(args, "loop_grad_mode", "full"),
        loop_extrap=getattr(args, "loop_extrap", 0.0),
        loop_anchor=getattr(args, "loop_anchor", True),
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
