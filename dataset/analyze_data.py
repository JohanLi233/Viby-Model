#!/usr/bin/env python3
"""
数据分析脚本 - 分析token长度分布和其他数据统计信息
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class DataStats:
    """数据统计信息的结构化存储"""
    token_lengths: List[int]
    text_lengths: List[int]
    conversation_lengths: List[int]
    user_msg_lengths: List[int]
    assistant_msg_lengths: List[int]
    empty_samples: int
    total_samples: int

    def __post_init__(self):
        """计算派生统计信息"""
        self.valid_samples = len(self.token_lengths)
        self.empty_ratio = self.empty_samples / self.total_samples if self.total_samples > 0 else 0

def load_jsonl_streaming(file_path: str, sample_size: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
    """流式加载JSONL数据"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if sample_size and count >= sample_size:
                break
            try:
                yield json.loads(line.strip())
                count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} has invalid JSON: {e}")

def print_distribution(values: List[int], name: str, bins: int = 10) -> None:
    """打印数值分布的直方图（文本版）"""
    if len(values) == 0:
        return
    
    min_val, max_val = min(values), max(values)
    
    # 处理所有值相同的情况
    if min_val == max_val:
        print(f"\n{name} 分布:")
        print(f"所有值都是: {min_val}")
        print(f"  [{min_val:6.0f} - {min_val:6.0f}]: {len(values):4d} (100.0%) {'█' * 50}")
        return
    
    print(f"\n{name} 分布:")
    print(f"范围: {min_val} - {max_val}")
    
    # 使用numpy的histogram进行高效计算
    counts, bin_edges = np.histogram(values, bins=bins, range=(min_val, max_val))
    max_count = max(counts) if len(counts) > 0 else 1
    
    for i, count in enumerate(counts):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # 最后一个bin包含右边界
        bracket = ']' if i == bins - 1 else ')'
        
        percentage = count / len(values) * 100
        bar_length = int(count / max_count * 50)
        bar = '█' * bar_length
        print(f"  [{bin_start:6.0f} - {bin_end:6.0f}{bracket}: {count:4d} ({percentage:5.1f}%) {bar}")

def print_percentiles(values: List[int], name: str) -> None:
    """打印百分位数信息"""
    if len(values) == 0:
        return
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n{name} 百分位数:")
    
    # 使用numpy一次性计算所有百分位数
    percentile_values = np.percentile(values, percentiles)
    for p, val in zip(percentiles, percentile_values):
        print(f"  P{p:2d}: {val:8.1f}")

def analyze_data(data_generator: Generator[Dict[str, Any], None, None], 
                tokenizer, total_samples: Optional[int] = None) -> DataStats:
    """分析数据"""
    text_lengths = []
    token_lengths = []
    empty_samples = 0
    conversation_lengths = []
    user_msg_lengths = []
    assistant_msg_lengths = []
    
    processed_count = 0
    
    for i, sample in enumerate(data_generator):
        processed_count += 1
        
        # 检测数据类型并处理
        if "text" in sample:
            # Pretrain格式
            text = str(sample.get("text", ""))
            if not text.strip():
                empty_samples += 1
                continue
            text_lengths.append(len(text))
            
            try:
                tokens = tokenizer(text, truncation=False)["input_ids"]
                token_lengths.append(len(tokens))
            except Exception as e:
                print(f"Warning: Tokenization failed for sample {i}: {e}")
                empty_samples += 1
                continue
                
        elif "conversations" in sample:
            # SFT格式
            conversations = sample.get("conversations", [])
            if not conversations:
                empty_samples += 1
                continue
                
            conversation_lengths.append(len(conversations))
            
            # 构建完整对话
            messages = []
            user_tokens_total = 0
            assistant_tokens_total = 0
            
            for j, turn in enumerate(conversations):
                # 优先使用数据中的role字段，否则按顺序推断
                if "role" in turn:
                    role = turn["role"]
                elif "from" in turn:
                    role = "user" if turn["from"] == "human" else "assistant"
                else:
                    role = "user" if j % 2 == 0 else "assistant"
                
                content = turn.get("content", turn.get("value", ""))
                messages.append({"role": role, "content": content})
                
                try:
                    turn_tokens = tokenizer(content, truncation=False)["input_ids"]
                    if role == "user":
                        user_tokens_total += len(turn_tokens)
                    else:
                        assistant_tokens_total += len(turn_tokens)
                except Exception as e:
                    print(f"Warning: Tokenization failed for conversation turn {j} in sample {i}: {e}")
                    continue
            
            if user_tokens_total > 0 or assistant_tokens_total > 0:
                user_msg_lengths.append(user_tokens_total)
                assistant_msg_lengths.append(assistant_tokens_total)
            
            # 计算完整对话的token数
            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer(full_prompt, truncation=False)["input_ids"]
                token_lengths.append(len(tokens))
            except Exception as e:
                print(f"Warning: Chat template failed for sample {i}: {e}")
                empty_samples += 1
                continue
        else:
            print(f"Warning: Unknown data format at sample {i}")
            empty_samples += 1
            continue
        
        # 进度显示
        if processed_count % 1000 == 0:
            if total_samples:
                print(f"处理进度: {processed_count}/{total_samples}")
            else:
                print(f"已处理: {processed_count} 个样本")
    
    return DataStats(
        token_lengths=token_lengths,
        text_lengths=text_lengths,
        conversation_lengths=conversation_lengths,
        user_msg_lengths=user_msg_lengths,
        assistant_msg_lengths=assistant_msg_lengths,
        empty_samples=empty_samples,
        total_samples=processed_count
    )

def print_results(stats: DataStats) -> None:
    """显示分析结果"""
    print(f"\n{'='*50}")
    print(f"数据分析")
    print(f"{'='*50}")
    print(f"数据样本数: {stats.total_samples}")
    
    if not stats.token_lengths:
        print("没有有效的数据！")
        return
    
    # 统计信息
    print(f"\n数据质量:")
    print(f"  有效样本数: {stats.valid_samples}")
    print(f"  无效/空样本: {stats.empty_samples} ({stats.empty_ratio*100:.2f}%)")
    
    # Token长度统计
    token_array = np.array(stats.token_lengths)
    print(f"\nToken长度统计:")
    print(f"  count: {len(stats.token_lengths)}")
    print(f"  mean: {np.mean(token_array):.2f}")
    print(f"  median: {np.median(token_array):.2f}")
    print(f"  std: {np.std(token_array):.2f}")
    print(f"  min: {np.min(token_array):.0f}")
    print(f"  max: {np.max(token_array):.0f}")
    
    # 如果是pretrain数据，还显示文本长度
    if stats.text_lengths:
        text_array = np.array(stats.text_lengths)
        print(f"\n文本字符长度统计:")
        print(f"  mean: {np.mean(text_array):.2f}")
        print(f"  median: {np.median(text_array):.2f}")
        print(f"  std: {np.std(text_array):.2f}")
        print(f"  min: {np.min(text_array):.0f}")
        print(f"  max: {np.max(text_array):.0f}")
    
    # 如果是SFT数据，显示对话信息
    if stats.conversation_lengths:
        conv_array = np.array(stats.conversation_lengths)
        print(f"\n对话轮数统计:")
        print(f"  mean: {np.mean(conv_array):.2f}")
        print(f"  median: {np.median(conv_array):.2f}")
        print(f"  std: {np.std(conv_array):.2f}")
        print(f"  min: {np.min(conv_array):.0f}")
        print(f"  max: {np.max(conv_array):.0f}")
            
        if stats.user_msg_lengths and stats.assistant_msg_lengths:
            user_array = np.array(stats.user_msg_lengths)
            assistant_array = np.array(stats.assistant_msg_lengths)
            
            print(f"\n用户消息Token统计:")
            print(f"  平均: {np.mean(user_array):.2f}")
            print(f"  中位数: {np.median(user_array):.2f}")
            print(f"  最大: {np.max(user_array):.0f}")
            
            print(f"\n助手消息Token统计:")
            print(f"  平均: {np.mean(assistant_array):.2f}")
            print(f"  中位数: {np.median(assistant_array):.2f}")
            print(f"  最大: {np.max(assistant_array):.0f}")
            
            # 分析用户vs助手token比例
            total_tokens = user_array + assistant_array
            ratios = np.where(total_tokens > 0, assistant_array / total_tokens, 0)
            
            print(f"\n助手Token占比统计:")
            print(f"  平均占比: {np.mean(ratios)*100:.1f}%")
            print(f"  中位数占比: {np.median(ratios)*100:.1f}%")
    
    # 打印分布和百分位数
    print_distribution(stats.token_lengths, "Token长度", bins=20)
    print_percentiles(stats.token_lengths, "Token长度")
    
    if stats.conversation_lengths:
        print_distribution(stats.conversation_lengths, "对话轮数", bins=10)
    
    # 长度分段统计
    print_length_segments(stats.token_lengths)

def print_length_segments(token_lengths: List[int]) -> None:
    """按常见长度区间统计分布"""
    if not token_lengths:
        return
    
    segments = [
        (0, 128, "0-128"),
        (129, 256, "129-256"), 
        (257, 512, "257-512"),
        (513, 1024, "513-1024"),
        (1025, 2048, "1025-2048"),
        (2049, 4096, "2049-4096"),
        (4097, 8192, "4097-8192"),
        (8193, float('inf'), "8193+")
    ]
    
    print(f"\n长度区间分布:")
    print(f"{'区间':<12} {'数量':<8} {'占比':<8} {'累计占比':<8}")
    print("-" * 40)
    
    token_array = np.array(token_lengths)
    total = len(token_lengths)
    cumulative_count = 0
    
    for min_len, max_len, label in segments:
        if max_len == float('inf'):
            mask = token_array >= min_len
        else:
            mask = (token_array >= min_len) & (token_array <= max_len)
        
        count = np.sum(mask)
        
        if count > 0:
            cumulative_count += count
            percentage = count / total * 100
            cumulative_percentage = cumulative_count / total * 100
            print(f"{label:<12} {count:<8} {percentage:<7.1f}% {cumulative_percentage:<7.1f}%")

def main() -> None:
    parser = argparse.ArgumentParser(description='分析数据集的token长度分布')
    parser.add_argument('--data_path', required=True, help='数据文件路径 (.jsonl)')
    parser.add_argument('--sample_size', type=int, help='采样分析的样本数量')
    
    args = parser.parse_args()
    
    # 加载tokenizer
    print("加载本地tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained("../model/")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # 分析数据
    print(f"加载数据: {args.data_path}")
    
    try:
        data_generator = load_jsonl_streaming(args.data_path, args.sample_size)
        if args.sample_size:
            print(f"采样分析最多 {args.sample_size} 个样本")
        
        stats = analyze_data(data_generator, tokenizer, args.sample_size)
        print_results(stats)
        
    except FileNotFoundError:
        print(f"错误: 文件 {args.data_path} 不存在")
        return
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()