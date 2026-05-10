import argparse
import os
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.generation.streamers import TextStreamer
from model.model import VibyConfig, VibyForCausalLM

warnings.filterwarnings("ignore")


def transfer_cache_to_device(cache, target_device):
    """Transfer cache state from one device to another."""
    if cache is None:
        return None

    return [
        None
        if layer_cache is None
        else tuple(state.to(target_device) for state in layer_cache)
        for layer_cache in cache
    ]


def init_model(args):
    # Validate model_mode
    modes = {0: "pretrain", 1: "full_sft"}
    if args.model_mode not in modes:
        print(
            f"错误：不支持的模型模式 {args.model_mode}，支持的模式：{list(modes.keys())}"
        )
        exit(1)

    # Check if tokenizer directory exists
    tokenizer_path = "./model/"
    if not os.path.exists(tokenizer_path):
        print(f"错误：找不到tokenizer目录: {tokenizer_path}")
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ckp = f"./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}.pth"

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

    config = VibyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        original_max_position_embeddings=(
            original_max_seq_len
            if hasattr(args, "enable_yarn") and args.enable_yarn
            else getattr(args, "original_max_seq_len", 1024)
        ),
        rope_scaling=rope_scaling,
    )

    # Check if checkpoint file exists
    if not os.path.exists(ckp):
        print(f"错误：找不到模型检查点文件: {ckp}")
        print("请检查文件路径是否正确，或者确保模型已经训练完成。")
        exit(1)

    # Load checkpoint first
    checkpoint = torch.load(ckp, map_location="cpu")

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

    # Check if hybrid mode is enabled
    if hasattr(args, "hybrid_mps_cpu") and args.hybrid_mps_cpu:
        # Create two models for hybrid inference
        prefill_device = getattr(args, "prefill_device", "mps")
        generation_device = getattr(args, "generation_device", "cpu")

        # Create prefill model (MPS)
        prefill_model = VibyForCausalLM(config)
        prefill_model.load_state_dict(state_dict, strict=True)
        prefill_model = prefill_model.eval().to(prefill_device)

        # Create generation model (CPU)
        generation_model = VibyForCausalLM(config)
        generation_model.load_state_dict(state_dict, strict=True)
        generation_model = generation_model.eval().to(generation_device)

        model_for_counting = prefill_model

        return_value = {
            "prefill_model": prefill_model,
            "generation_model": generation_model,
            "prefill_device": prefill_device,
            "generation_device": generation_device,
            "hybrid_mode": True,
        }
    else:
        # Original single-device mode
        model = VibyForCausalLM(config)
        model.load_state_dict(state_dict, strict=True)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model = model.eval().to(args.device)

        model_for_counting = model
        return_value = model

    # 计算参数量（统一处理）
    total_params = sum(p.numel() for p in model_for_counting.parameters())
    trainable_params = sum(
        p.numel() for p in model_for_counting.parameters() if p.requires_grad
    )

    print(
        f"总参数量：{total_params / 1e6:.3f}M, 可训练参数量：{trainable_params / 1e6:.3f}M"
    )

    return return_value, tokenizer


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
    parser.add_argument("--hidden_size", default=640, type=int)
    parser.add_argument("--num_hidden_layers", default=18, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument(
        "--history_cnt",
        default=0,
        type=int,
        help="保留的对话历史轮次数量（0代表无历史）",
    )
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
        default=1024,
        type=int,
        help="Original context length before scaling",
    )
    parser.add_argument(
        "--yarn_beta_fast", default=32.0, type=float, help="YaRN beta_fast parameter"
    )
    parser.add_argument(
        "--yarn_beta_slow", default=1.0, type=float, help="YaRN beta_slow parameter"
    )

    # Hybrid inference parameters
    parser.add_argument(
        "--hybrid_mps_cpu",
        action="store_true",
        help="Enable hybrid inference: MPS for prefill, CPU for generation",
    )
    parser.add_argument(
        "--prefill_device",
        default="mps",
        type=str,
        help="Device for prefill phase (default: mps)",
    )
    parser.add_argument(
        "--generation_device",
        default="cpu",
        type=str,
        help="Device for generation phase (default: cpu)",
    )

    args = parser.parse_args()

    # Basic input validation
    if args.temperature <= 0:
        print("错误：temperature 必须大于 0")
        exit(1)
    if args.top_p <= 0 or args.top_p > 1:
        print("错误：top_p 必须在 (0, 1] 范围内")
        exit(1)
    if args.repetition_penalty <= 0:
        print("错误：repetition_penalty 必须大于 0")
        exit(1)
    if args.max_seq_len <= 0:
        print("错误：max_seq_len 必须大于 0")
        exit(1)
    if args.history_cnt < 0:
        print("错误：history_cnt 不能为负数")
        exit(1)

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for prompt in prompts if test_mode == 0 else iter(lambda: input("👶: "), ""):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0:
            print(f"👶: {prompt}")

        # 先添加当前的用户输入
        messages.append({"role": "user", "content": prompt})

        # 然后根据 history_cnt 对整个消息历史进行截断（按对话轮次）
        if args.history_cnt > 0:
            # 一个完整的对话轮次是2条消息 (user, assistant)
            # 我们要保留 history_cnt 轮对话，再加上当前刚输入的用户消息
            # 总共需要保留的消息数是 2 * history_cnt + 1
            num_to_keep = 2 * args.history_cnt + 1
            messages = messages[-num_to_keep:]
        else:
            # 如果 history_cnt 为 0，则清空历史，只保留当前输入
            messages = messages[-1:]

        new_prompt = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if args.model_mode != 0
            else (tokenizer.bos_token + prompt)
        )

        slice_start = 0
        if isinstance(model, dict) and model.get("hybrid_mode", False):
            prefill_model = model["prefill_model"]
            generation_model = model["generation_model"]
            prefill_device = model["prefill_device"]
            generation_device = model["generation_device"]

            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
                prefill_device
            )
            prompt_len = inputs["input_ids"].shape[1]
            max_new = max(1, args.max_seq_len - prompt_len)

            print("🤖️: ", end="")

            with torch.no_grad():
                outputs = prefill_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

            past_key_values = transfer_cache_to_device(
                past_key_values, generation_device
            )

            # 只取 prompt 的最后一个 token 作为生成模型的初始输入
            last_token_id = inputs["input_ids"][:, -1:].to(generation_device)
            # 直接传递原始attention mask，让generate()内部处理扩展
            attention_mask = inputs["attention_mask"].to(generation_device)

            generated_ids = generation_model.generate(
                input_ids=last_token_id,  # 传入最后一个 token
                max_new_tokens=max_new,
                num_return_sequences=1,
                do_sample=True,
                attention_mask=attention_mask,  # 传入更新后的 attention_mask
                past_key_values=past_key_values,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                top_p=args.top_p,
                temperature=args.temperature,
                use_cache=True,
                repetition_penalty=args.repetition_penalty,
            )

            # generate()的输入是last_token_id(长度为1)，所以从索引1开始解码
            slice_start = 1
        else:
            # Original single-device mode
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(
                args.device
            )
            # generate的输入是完整的prompt，所以从prompt长度之后开始解码
            slice_start = inputs["input_ids"].shape[1]
            prompt_len = inputs["input_ids"].shape[1]
            max_new = max(1, args.max_seq_len - prompt_len)

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
                use_cache=True,
                repetition_penalty=args.repetition_penalty,
            )

        # 统一处理响应解码（移出 if/else 块以避免重复）
        response = tokenizer.decode(
            generated_ids[0][slice_start:], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print("\n\n")


if __name__ == "__main__":
    main()
