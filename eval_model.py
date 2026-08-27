import argparse
import glob
import json
import os
import random
import time
import warnings

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from model.config import VibyConfig
from model.model import VibyForCausalLM
from trainer.utils import load_model_weights

warnings.filterwarnings("ignore")


class TextStreamer:
    """简单的流式输出器，替代 transformers TextStreamer。

    累积生成的 token，每次 put() 时对全量已生成序列重新 decode，
    只打印新解出的文本。byte-level BPE 中一个汉字常由 2~3 个 byte
    token 拼成：若 decode 结果以替换字符  结尾，说明末尾还有半个
    多字节字符，先不打印，等后续 token 补全后再一起输出。
    """

    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self._prompt_skipped = False
        self._token_ids = []
        self._print_len = 0

    def put(self, token_ids):
        arr = np.array(token_ids)
        if arr.ndim > 1:
            arr = arr[0]  # eval 只跑 batch=1
        if self.skip_prompt and not self._prompt_skipped:
            # 第一次 put 收到的是完整 prompt，直接跳过
            self._prompt_skipped = True
            return
        self._token_ids.extend(arr.tolist())
        text = self.tokenizer.decode(
            self._token_ids, skip_special_tokens=self.skip_special_tokens
        )
        # 末尾的  是尚未拼完的多字节字符，截掉等补全（不能打印，
        # 否则 _print_len 会越过它，补全后的汉字反而再也打不出来）
        if text.endswith(""):
            text = text[:-1]
        new_text = text[self._print_len :]
        if new_text:
            print(new_text, end="", flush=True)
            self._print_len = len(text)

    def end(self):
        if self._token_ids:
            text = self.tokenizer.decode(
                self._token_ids, skip_special_tokens=self.skip_special_tokens
            )
            new_text = text[self._print_len :]
            if new_text:
                print(new_text, end="", flush=True)
                self._print_len = len(text)
        print()
        # 每次 generate 结束，重置状态以跳过下一次的 prompt
        self._prompt_skipped = False
        self._token_ids = []
        self._print_len = 0


def _find_checkpoint(out_dir, mode_name, hidden_size):
    """查找检查点：latest_checkpoint.txt -> 精确命名 -> 同前缀最新文件。"""
    latest_file = os.path.join(out_dir, "latest_checkpoint.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r", encoding="utf-8") as f:
            latest = f.read().strip()
        if latest and os.path.basename(latest).startswith(f"{mode_name}_"):
            if os.path.exists(latest):
                return latest

    exact = os.path.join(out_dir, f"{mode_name}_{hidden_size}.safetensors")
    if os.path.exists(exact):
        return exact

    candidates = [
        p
        for p in glob.glob(os.path.join(out_dir, f"{mode_name}_*.safetensors"))
        if not p.endswith(".optimizer.safetensors")
    ]
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


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

    ckp = _find_checkpoint(args.out_dir, modes[args.model_mode], args.hidden_size)
    if ckp is None:
        print(
            f"错误：在 {args.out_dir} 中找不到 {modes[args.model_mode]}_*.safetensors "
            "检查点（已尝试 latest_checkpoint.txt 与 hidden_size 精确匹配）"
        )
        print(
            "注意：旧的 .pth 检查点不再兼容，请使用训练脚本新保存的 safetensors 格式。"
        )
        exit(1)

    sidecar = os.path.splitext(ckp)[0] + ".json"
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            meta = json.load(f)
        config = VibyConfig.from_dict(meta["config"])
        print(
            f"[checkpoint] {os.path.basename(ckp)} 从 {sidecar} 加载配置 "
            f"(epoch={meta.get('epoch')}, step={meta.get('step')})"
        )
    else:
        # Fallback: build config from args
        config = VibyConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=getattr(args, "num_attention_heads", 8),
            vocab_size=getattr(args, "vocab_size", 6400),
            max_position_embeddings=args.max_seq_len,
            use_attn_gate=getattr(args, "use_attn_gate", False),
            mtp_depth=getattr(args, "mtp_depth", 0),
            mtp_loss_weight=getattr(args, "mtp_loss_weight", 0.3),
            mtp_steps=getattr(args, "mtp_steps", 2),
            **(
                {"head_dim": args.head_dim}
                if getattr(args, "head_dim", None) is not None
                else {}
            ),
        )

    model = VibyForCausalLM(config)
    if not load_model_weights(model, ckp, strict=True, label="checkpoint"):
        print(f"错误：无法加载检查点 {ckp}")
        exit(1)
    model.eval()

    total_params = model.num_parameters()
    active_params = model.num_active_parameters()
    print(
        f"总参数量：{total_params / 1e6:.3f}M, "
        f"每次激活：{active_params / 1e6:.3f}M"
    )

    return model, tokenizer


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
    mx.random.seed(seed)


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
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--vocab_size", default=6400, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--mtp_depth", type=int, default=1)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument(
        "--use_attn_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--history_cnt",
        default=0,
        type=int,
        help="保留的对话历史轮次数量（0代表无历史）",
    )
    parser.add_argument(
        "--use_mtp_speculative",
        action="store_true",
        help="使用 MTP 投机解码加速（需模型带 MTP 模块，batch=1）",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        default=3,
        type=int,
        help="MTP 投机解码每轮草稿的 token 数（默认 3，对齐 DeepSeek V4 等主流配置；"
        "需配合 --use_mtp_speculative，模型只有 1 个 MTP 模块时会循环复用）",
    )
    parser.add_argument(
        "--model_mode",
        default=0,
        type=int,
        help="0: 预训练模型，1: SFT-Chat模型",
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
    if args.num_speculative_tokens <= 0:
        print("错误：num_speculative_tokens 必须大于 0")
        exit(1)
    if args.max_seq_len <= 0:
        print("错误：max_seq_len 必须大于 0")
        exit(1)
    if args.history_cnt < 0:
        print("错误：history_cnt 不能为负数")
        exit(1)

    # 显式锁定 GPU（Metal）为默认计算设备，确保推理跑在 GPU 上
    mx.set_default_device(mx.gpu)
    print(f"[device] 默认计算设备: {mx.default_device()}")

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 自动测试模式下累积每题耗时，最后输出汇总
    perf_stats = [] if test_mode == 0 else None

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

        input_ids = mx.array(
            [
                tokenizer(
                    new_prompt,
                    truncation=True,
                    max_length=model.config.max_position_embeddings,
                ).input_ids
            ]
        )
        # generate的输入是完整的prompt，所以从prompt长度之后开始解码
        slice_start = input_ids.shape[1]
        prompt_len = input_ids.shape[1]
        max_new = max(0, model.config.max_position_embeddings - prompt_len)

        print("🤖️: ", end="")
        t0 = time.perf_counter()
        generated_ids = model.generate(
            input_ids,
            attention_mask=None,
            max_new_tokens=max_new,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            use_mtp_speculative=args.use_mtp_speculative,
            num_speculative_tokens=args.num_speculative_tokens,
        )
        elapsed = time.perf_counter() - t0
        if args.use_mtp_speculative and hasattr(model, "_last_spec_stats"):
            stats = model._last_spec_stats
            acc = stats["accepted"] / max(stats["drafted"], 1)
            print(
                f"\n[MTP speculative] 草稿接受率: {acc:.1%} "
                f"({stats['accepted']}/{stats['drafted']}，"
                f"每轮草稿 {args.num_speculative_tokens})"
            )

        gen_len = generated_ids.shape[1] - slice_start
        tps = gen_len / elapsed if elapsed > 0 else 0.0
        print(f"[TPS] 生成 {gen_len} tokens，耗时 {elapsed:.2f}s，{tps:.2f} tokens/s")
        if perf_stats is not None:
            perf_stats.append((gen_len, elapsed, tps))

        response = tokenizer.decode(
            np.array(generated_ids)[0][slice_start:], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
        print("\n\n")

    if perf_stats:
        total_tokens = sum(s[0] for s in perf_stats)
        total_time = sum(s[1] for s in perf_stats)
        tps_list = [s[2] for s in perf_stats]
        overall_tps = total_tokens / total_time if total_time > 0 else 0.0
        print(
            f"[TPS 汇总] 共 {len(perf_stats)} 题，生成 {total_tokens} tokens，"
            f"总耗时 {total_time:.2f}s，整体 {overall_tps:.2f} tokens/s"
            f"（单题 {min(tps_list):.2f} ~ {max(tps_list):.2f} tokens/s）"
        )


if __name__ == "__main__":
    main()
