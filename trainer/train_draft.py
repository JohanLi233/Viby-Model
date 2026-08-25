"""EAGLE-3 草稿微调（Kimi-K3 后训练口径）：冻结目标模型，只训练 MTP 草稿层。

预训练已带 1 个 MTP 层（mtp_depth=1，与主干同构）；本阶段把该层微调为
EAGLE-3 风格的草稿模型：输入 = 目标模型低/中/高三层特征（fc_l 融合）+
下一 token 嵌入，用 training-time test (TTT) 多步 rollout 训练：
step 1 消费目标特征 + 真实下一 token 嵌入；step≥2 消费草稿自身上一步
block 输出 + 上一步 argmax 预测 token 的嵌入（离散选择天然 stop-grad，
梯度经草稿 hidden 回传）；各步 CE 对真实未来 token 计算后取平均。
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import mlx.core as mx
from model.config import VibyConfig
from model.kernels.ce import cross_entropy
from dataset.lm_dataset import PretrainDataset
from .base_trainer import BaseTrainer
from .config import get_draft_parser, setup_training_args
from .utils import (
    build_config_from_sidecar,
    build_model_and_tokenizer,
    finish_training,
    init_swanlab,
)

warnings.filterwarnings("ignore")


def draft_ttt_loss(
    model, X, Y, loss_mask, attn_mask, mask_has_pad, ttt_steps=4, segment_ids=None
):
    """EAGLE-3 TTT 多步 rollout 草稿 loss（目标模型冻结，只反传草稿参数）。

    X: (B, T) 输入 token；Y: next-token 移位的 labels；ttt_steps: rollout
    步数。step s 在流位置 j 预测 Y[:, s:][j]（即 token j+s+1）：
    - step 1：输入目标模型多层特征 feats[:, j] 与 Emb(X[j+1])；
    - step s≥2：输入草稿上一步 block 输出 h[j]（梯度回传）与上一步
      argmax 预测 token 的嵌入（约等于 Emb(X[j+s]) 的自预测版）。
    返回各步 CE 的平均（掩码口径与 _mtp_loss 一致；segment_ids 非 None
    时按源位置的文档边界掩码自注意力，与 _mtp_loss 的 doc_mask 口径
    一致）。目标前向只跑一次，特征全程复用；主干参数已冻结、不在梯度
    树内，等价于 stop-gradient。
    """
    core = model.model  # VibyModel
    mtp = model.mtp_modules[0]
    _, T = X.shape
    _, _, feats = core(
        input_ids=X,
        attention_mask=attn_mask,
        mask_has_pad=mask_has_pad,
        output_features=True,
    )
    emb_all = core.embed_tokens(X)
    if mask_has_pad is None:
        mask_has_pad = attn_mask is not None and bool(mx.any(attn_mask != 1).item())

    ces = []
    h_prev = None
    tok_prev = None
    for s in range(1, ttt_steps + 1):
        sub = T - s
        if sub <= 0:
            break
        am = attn_mask[:, :sub] if attn_mask is not None else None
        mask_is_full = am is None or not mask_has_pad
        causal_bias = None
        if am is not None and not mask_is_full:
            pad_bias = mx.where(am.astype(mx.bool_), 0.0, -1e9)
            causal = mx.triu(mx.full((sub, sub), -1e9), k=1)
            causal_bias = (
                causal[None, None, :, :] + pad_bias[:, None, None, :]
            ).astype(emb_all.dtype)
        seg = None
        if segment_ids is not None:
            # 与 _mtp_loss 的 doc_mask 口径一致：按源位置 j 的文档边界
            # 掩码，避免打包序列内 MTP 自注意力跨文档泄漏
            seg = segment_ids[:, :sub]
            same_doc = seg[:, :, None] == seg[:, None, :]
            causal_tril = mx.tril(mx.ones((sub, sub), dtype=mx.bool_))
            allowed = same_doc & causal_tril[None, :, :]
            seg_bias = mx.where(allowed[:, None, :, :], 0.0, -1e9).astype(emb_all.dtype)
            causal_bias = seg_bias if causal_bias is None else causal_bias + seg_bias
            mask_is_full = False
        if s == 1:
            h_in = [f[:, :sub, :] for f in feats]
            e_in = emb_all[:, 1 : sub + 1, :]
        else:
            h_in = h_prev[:, :sub, :]
            # 上一步 argmax 预测 token（离散 id 选择本身无梯度，天然 stop-grad）
            e_in = core.embed_tokens(tok_prev[:, :sub])
        h, _ = mtp(
            h_in,
            e_in,
            attention_mask=am,
            causal_bias=causal_bias,
            mask_is_full=mask_is_full,
            segment_ids=seg,
        )
        logits = model._lm_logits(h)
        ce = cross_entropy(
            logits,
            Y[:, s:],
            mask=loss_mask[:, s:] if loss_mask is not None else None,
        )
        ces.append(ce)
        h_prev = h
        # 上一步 argmax 预测 token：stop_gradient 明确截断（gather 对
        # indices 无 VJP；离散选择本不应携带梯度，梯度只经 h_prev 回传）
        tok_prev = mx.stop_gradient(mx.argmax(logits, axis=-1))
    return sum(ces) / len(ces)


class DraftTrainer(BaseTrainer):
    """EAGLE-3 草稿微调训练器：目标模型冻结，只更新 mtp_modules。

    loss 为 TTT 多步 rollout CE；主 LM loss / z-loss 不参与
    （主干冻结、QB 偏置不更新），日志位以草稿 CE 填充。
    """

    def _loss_fn(self, X, Y, loss_mask, attn_mask, mask_has_pad, seg_ids=None):
        ce = draft_ttt_loss(
            self.model,
            X,
            Y,
            loss_mask,
            attn_mask,
            mask_has_pad,
            ttt_steps=self.args.draft_ttt_steps,
            segment_ids=seg_ids,
        )
        zero = mx.array(0.0)
        # 返回结构与 BaseTrainer._loss_fn 一致：
        # (加权 loss, mtp 分量, moe margin 统计, lm 分量, aux, div, lar, z)
        return (
            ce / self.args.accumulation_steps,
            ce,
            mx.zeros((0,), dtype=mx.float32),
            ce / self.args.accumulation_steps,
            zero,
            zero,
            zero,
            zero,
        )


def init_model(lm_config, args):
    """初始化模型和 tokenizer，加载基座权重"""
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
    parser = get_draft_parser()
    args = parser.parse_args()
    args = setup_training_args(args, "draft")
    if args.draft_ttt_steps <= 0:
        raise ValueError("--draft_ttt_steps 必须大于 0")

    # 从基座 checkpoint 的 sidecar config 继承模型结构配置（CLI 显式参数优先）
    checkpoint_name = (
        args.pretrain_checkpoint
        if getattr(args, "pretrain_checkpoint", None)
        else f"pretrain_{args.hidden_size}.safetensors"
    )
    cfg, has_sidecar = build_config_from_sidecar(args, checkpoint_name)
    lm_config = VibyConfig.from_dict(cfg) if has_sidecar else VibyConfig(**cfg)
    if lm_config.mtp_depth < 1:
        raise ValueError(
            "基座 checkpoint 不含 MTP 层（mtp_depth=0），无法做草稿微调；"
            "请使用 mtp_depth>=1 预训练的基座"
        )

    # 初始化模型并加载基座权重
    model, tokenizer = init_model(lm_config, args)

    # 冻结目标模型，只留 MTP 草稿模块可训练（EAGLE-3：target frozen）；
    # create_mixed_optimizer 只对 trainable_parameters 分组，优化器随之
    # 只含草稿参数（Muon 矩阵组 / router 组 / 标量组仍按路径正常工作）。
    model.freeze()
    for m in model.mtp_modules:
        m.unfreeze()

    # 创建训练器
    trainer = DraftTrainer(args, model, tokenizer, lm_config, "draft")

    # 草稿训练用普通预训练语料（LM 目标，无需 SFT 格式）
    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        pack_sequences=getattr(args, "pack_sequences", False),
        doc_mask=getattr(args, "doc_mask", False),
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
# 标准草稿微调（基座需 mtp_depth>=1 的预训练 checkpoint）:
# python train_draft.py --data_path /Volumes/pan/pretrain_hq.jsonl --max_seq_len 1024 --batch_size 8
#
# 调整 TTT rollout 步数:
# python train_draft.py --draft_ttt_steps 6
