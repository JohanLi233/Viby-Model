import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.generation.streamers import TextStreamer
from model.model import VibyConfig, VibyForCausalLM

warnings.filterwarnings("ignore")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    modes = {0: "pretrain", 1: "full_sft"}
    # ckp = f"./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}.pth"
    ckp = f"./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}_moe.pth"

    # Configure YaRN if enabled
    rope_scaling = None
    original_max_seq_len = args.max_seq_len  # 保存原始值

    if hasattr(args, "enable_yarn") and args.enable_yarn:
        # 自动将 max_seq_len 乘以 scaling factor
        args.max_seq_len = int(args.max_seq_len * args.yarn_scaling_factor)

        rope_scaling = {
            "type": "yarn",
            "factor": args.yarn_scaling_factor,
            "beta_fast": getattr(args, "yarn_beta_fast", 32.0),
            "beta_slow": getattr(args, "yarn_beta_slow", 1.0),
        }
        print(
            f"[YaRN] 启用上下文扩展: {original_max_seq_len} → {args.max_seq_len} (scaling factor: {args.yarn_scaling_factor})"
        )

    model = VibyForCausalLM(
        VibyConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_position_embeddings=args.max_seq_len,
            original_max_position_embeddings=(
                original_max_seq_len
                if hasattr(args, "enable_yarn") and args.enable_yarn
                else getattr(args, "original_max_seq_len", 1024)
            ),
            rope_scaling=rope_scaling,
            use_moe=args.use_moe,
        )
    )

    checkpoint = torch.load(ckp, map_location=args.device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New checkpoint format from training script
        state_dict = checkpoint["model_state_dict"]
    else:
        # Old format - direct state dict
        state_dict = checkpoint

    # Handle torch.compile prefix if present
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"总参数量：{total_params / 1e6:.3f}M, 可训练参数量：{trainable_params / 1e6:.3f}M"
    )

    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            "马克思主义基本原理",
            "人类大脑的主要功能",
            "万有引力原理是",
            "世界上最高的山峰是",
            "二氧化碳在空气中",
            "地球上最大的动物有",
            "杭州市的美食有",
        ]
    else:
        # 通用对话问题
        prompt_datas = [
            "请介绍一下自己。",
            "你更擅长哪一个学科？",
            "鲁迅的《狂人日记》是如何批判封建礼教的？",
            "我咳嗽已经持续了两周，需要去医院检查吗？",
            "详细的介绍光速的物理概念。",
            "推荐一些杭州的特色美食吧。",
            "请为我讲解“大语言模型”这个概念。",
            "如何理解ChatGPT？",
            "Introduce the history of the United States, please.",
        ]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with Viby")
    parser.add_argument("--lora_name", default="None", type=str)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument(
        "--repetition_penalty",
        default=1.2,
        type=float,
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--history_cnt", default=0, type=int)
    parser.add_argument(
        "--model_mode",
        default=1,
        type=int,
        help="0: 预训练模型，1: SFT-Chat模型",
    )
    # YaRN parameters
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
    # MoE parameters
    parser.add_argument(
        "--use_moe", action="store_true", help="Enable Mixture of Experts"
    )
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(
        prompts if test_mode == 0 else iter(lambda: input("👶: "), "")
    ):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0:
            print(f"👶: {prompt}")

        messages = messages[-args.history_cnt :] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if args.model_mode != 0
            else (tokenizer.bos_token + prompt)
        )

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
            args.device
        )
        prompt_len = inputs["input_ids"].shape[1]
        # 计算还能生成多少新的token
        max_new = args.max_seq_len - prompt_len

        print("🤖️: ", end="")
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

        response = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
