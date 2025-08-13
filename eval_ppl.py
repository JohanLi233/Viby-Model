import argparse
import math
import os
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from model.model import VibyConfig, VibyForCausalLM
from dataset.lm_dataset import PretrainDataset, SFTDataset


def build_model_and_tokenizer(
    hidden_size: int,
    num_hidden_layers: Optional[int],
    max_seq_len: int,
    device: str,
    ckpt_path: str,
    model_dir: str,
) -> tuple[VibyForCausalLM, AutoTokenizer]:
    """Load tokenizer and model using checkpoint's saved config when available."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Prefer the exact config used during training to avoid shape mismatches
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        try:
            ckpt_cfg = checkpoint["config"]
            config = VibyConfig(**ckpt_cfg)
            print(
                f"[Info] Loaded config from checkpoint: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, vocab_size={config.vocab_size}"
            )
        except Exception as e:
            print(f"[Warn] Failed to load config from checkpoint ({e}); falling back to args.")
            config = VibyConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers if num_hidden_layers else VibyConfig().num_hidden_layers,
                max_position_embeddings=max_seq_len,
                original_max_position_embeddings=max_seq_len,
            )
    else:
        config = VibyConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers if num_hidden_layers else VibyConfig().num_hidden_layers,
            max_position_embeddings=max_seq_len,
            original_max_position_embeddings=max_seq_len,
        )

    # Build model with the resolved config
    model = VibyForCausalLM(config).to(device=device)

    # Restore weights (strip compile prefix if present)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_ppl(
    model: VibyForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    mode: str,
    max_seq_len: int,
    batch_size: int,
    device: str,
    limit_batches: Optional[int] = None,
    sample_ratio: float = 0.0,
    sample_count: int = 0,
    seed: int = 1337,
    num_workers: int = 0,
) -> tuple[float, float, int]:
    """Compute dataset perplexity as exp(mean token NLL), masked appropriately.

    mode: "pretrain" or "full_sft" to decide dataset class and loss_mask semantics.
    """
    if mode == "pretrain":
        ds = PretrainDataset(dataset_path, tokenizer, max_length=max_seq_len)
    elif mode == "full_sft":
        ds = SFTDataset(dataset_path, tokenizer, max_length=max_seq_len)
    else:
        raise ValueError("mode must be 'pretrain' or 'full_sft'")

    # Optional random subset sampling
    dataset_for_loader = ds
    if sample_ratio > 0.0 or sample_count > 0:
        n = len(ds)
        if sample_ratio > 0.0:
            k = max(1, int(n * sample_ratio))
        else:
            k = min(sample_count, n)
        rnd = random.Random(seed)
        indices = rnd.sample(range(n), k)
        dataset_for_loader = Subset(ds, indices)

    def collate(batch):
        xs, ys, masks = zip(*batch)
        return (
            torch.stack(xs, dim=0),
            torch.stack(ys, dim=0),
            torch.stack(masks, dim=0),
        )

    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    total_nll = 0.0
    total_tokens = 0

    for step, (x, y, loss_mask) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)

        attention_mask = (x != tokenizer.pad_token_id).long()
        outputs = model(input_ids=x, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits.float()

        # Compute token-level NLL without z-loss
        nll = F.cross_entropy(
            logits.flatten(end_dim=-2),
            y.flatten(end_dim=-1),
            reduction="none",
        )
        nll = nll * loss_mask.flatten()
        total_nll += nll.sum().item()
        total_tokens += int(loss_mask.sum().item())

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    return ppl, mean_nll, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL for Viby model")
    parser.add_argument("--model_mode", type=int, default=0, help="0: pretrain, 1: full_sft")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--hidden_size", type=int, default=640)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=640)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit_batches", type=int, default=0, help="0 means use full dataset")
    parser.add_argument("--sample_ratio", type=float, default=0.0, help="Randomly sample this fraction of the dataset (0-1], exclusive of sample_count if >0)")
    parser.add_argument("--sample_count", type=int, default=100, help="Randomly sample this many samples; has priority over sample_ratio when >0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--ckpt_path", type=str, default="./out/pretrain_640.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    modes = {0: "pretrain", 1: "full_sft"}
    mode_str = modes.get(args.model_mode, "pretrain")

    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(args.out_dir, f"{mode_str}_{args.hidden_size}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, tokenizer = build_model_and_tokenizer(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        device=args.device,
        ckpt_path=ckpt_path,
        model_dir=args.model_dir,
    )

    ppl, mean_nll, total_tokens = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.data_path,
        mode=mode_str,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        limit_batches=(None if args.limit_batches == 0 else args.limit_batches),
        sample_ratio=args.sample_ratio,
        sample_count=args.sample_count,
        seed=args.seed,
        num_workers=0,
    )

    print(
        f"Perplexity ({mode_str}) on {args.data_path}: {ppl:.4f} (mean_nll={mean_nll:.4f}, tokens={total_tokens})"
    )


if __name__ == "__main__":
    main()


