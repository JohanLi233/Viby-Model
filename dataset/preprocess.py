import json
import os
import argparse
import sys
import time
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Tuple, Optional, Iterator
from transformers import AutoTokenizer


# Global tokenizer for worker processes (avoids pickling issues)
WORKER_TOKENIZER = None


def init_worker_tokenizer(model_path="model/"):
    """Initialize tokenizer in worker process."""
    global WORKER_TOKENIZER
    try:
        WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        return True
    except Exception as e:
        print(f"Worker tokenizer initialization failed: {e}")
        return False


def worker_process_chunk(
    chunk_data: List[Tuple[str, int]], 
    max_length: int,
    min_length: int,
    is_jsonl: bool,
    text_key: str,
    model_path: str = "model/"
) -> Tuple[List[Dict[str, str]], int, int, int]:
    """
    Worker function to process a chunk of data.
    
    Args:
        chunk_data: List of (line_content, line_number) tuples
        max_length: Maximum token length allowed
        min_length: Minimum token length required
        is_jsonl: Whether input is JSONL format
        text_key: Key to extract from JSONL
        
    Returns:
        (valid_records, filtered_count, error_count, skipped_count)
    """
    global WORKER_TOKENIZER
    
    if WORKER_TOKENIZER is None:
        if not init_worker_tokenizer(model_path):
            return [], 0, len(chunk_data), 0
    
    def estimate_token_length(text: str) -> int:
        """Fast approximation of token length."""
        return len(text) // 3
    
    def parse_input_line(line: str, line_num: int) -> Tuple[str, bool]:
        """Parse input line and extract text content."""
        line = line.strip()
        if not line:
            return "", False
            
        try:
            if is_jsonl:
                data = json.loads(line)
                text_content = data.get(text_key)
                if text_content is None or not isinstance(text_content, str):
                    return "", False
                return text_content, True
            else:
                return line, True
        except (json.JSONDecodeError, Exception):
            return "", False
    
    def batch_tokenize(texts: List[str]) -> List[int]:
        """Batch tokenize multiple texts."""
        try:
            tokens_batch = WORKER_TOKENIZER(texts, truncation=False, padding=False)["input_ids"] # type: ignore
            return [len(tokens) for tokens in tokens_batch]
        except Exception:
            # Fallback to individual tokenization
            lengths = []
            for text in texts:
                try:
                    tokens = WORKER_TOKENIZER(text, truncation=False)["input_ids"] # type: ignore
                    lengths.append(len(tokens))
                except:
                    lengths.append(float("inf"))
            return lengths
    
    # Process chunk
    valid_records = []
    skipped_count = 0
    early_filtered_count = 0
    pre_filtered_batch = []
    
    # Parse and early filter
    for line_content, line_num in chunk_data:
        text_content, is_valid = parse_input_line(line_content, line_num)
        if not is_valid:
            skipped_count += 1
            continue
            
        formatted_text = f"<|im_start|>{text_content}<|im_end|>"
        estimated_length = estimate_token_length(formatted_text)
        
        if estimated_length > max_length * 1.5 or estimated_length < min_length * 0.7:
            early_filtered_count += 1
            continue
            
        pre_filtered_batch.append((formatted_text, line_num))
    
    if not pre_filtered_batch:
        return [], early_filtered_count, 0, skipped_count
    
    # Batch tokenization
    texts_to_tokenize = [item[0] for item in pre_filtered_batch]
    token_lengths = batch_tokenize(texts_to_tokenize)
    
    # Filter by actual token length
    length_filtered_count = 0
    error_count = 0
    
    for (formatted_text, line_num), token_length in zip(pre_filtered_batch, token_lengths):
        if token_length == float("inf"):
            error_count += 1
            continue
            
        if token_length > max_length or token_length < min_length:
            length_filtered_count += 1
            continue
            
        valid_records.append({"text": formatted_text})
    
    total_filtered = early_filtered_count + length_filtered_count
    return valid_records, total_filtered, error_count, skipped_count


class OptimizedTextProcessor:
    """
    Production-ready optimized text processor with batch processing, buffered I/O,
    memory efficiency, and progress reporting.
    """

    def __init__(self, batch_size: int = 1000, buffer_size: int = 10000, num_workers: Optional[int] = None, model_path: str = "model/"):
        """
        Initialize the processor with configurable batch and buffer sizes.

        Args:
            batch_size: Number of texts to process in each batch for tokenization
            buffer_size: Number of output records to buffer before writing to disk
            num_workers: Number of worker processes (default: CPU count)
            model_path: Path to the tokenizer model
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = batch_size * 20  # Larger chunks for workers
        self.model_path = model_path
        self.tokenizer = None

    def _load_tokenizer(self) -> bool:
        """Load tokenizer with error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("INFO: 成功加载本地tokenizer")
            return True
        except Exception as e:
            print(f"错误: 无法加载tokenizer: {e}")
            return False

    def _estimate_token_length(self, text: str) -> int:
        """
        Fast approximation of token length before expensive tokenization.
        Rough estimate: 1 token per 3-4 characters for Chinese/English mixed text.
        """
        return len(text) // 3

    def _parse_input_line(
        self, line: str, line_num: int, is_jsonl: bool, text_key: str
    ) -> Tuple[str, bool]:
        """
        Parse a single input line and extract text content.

        Returns:
            (text_content, is_valid) where is_valid indicates successful parsing
        """
        line = line.strip()
        if not line:
            return "", False

        try:
            if is_jsonl:
                data = json.loads(line)
                text_content = data.get(text_key)
                if text_content is None:
                    if line_num % 1000 == 0:  # Reduce warning frequency
                        print(f"警告: 第 {line_num} 行找不到键 '{text_key}'")
                    return "", False
                if not isinstance(text_content, str):
                    if line_num % 1000 == 0:
                        print(f"警告: 第 {line_num} 行中键 '{text_key}' 的值不是字符串")
                    return "", False
                return text_content, True
            else:
                return line, True
        except json.JSONDecodeError:
            if line_num % 1000 == 0:
                print(f"警告: 第 {line_num} 行无法解析为JSON")
            return "", False
        except Exception as e:
            if line_num % 1000 == 0:
                print(f"错误: 处理第 {line_num} 行时发生错误: {e}")
            return "", False

    def _batch_tokenize(self, texts: List[str]) -> List[int]:
        """
        Batch tokenize multiple texts for efficiency.

        Returns:
            List of token lengths for each text
        """
        try:
            # Batch tokenization is much more efficient
            tokens_batch = self.tokenizer(texts, truncation=False, padding=False)[ # type: ignore
                "input_ids"
            ]
            return [len(tokens) for tokens in tokens_batch]
        except Exception as e:
            print(f"警告: 批量tokenization失败: {e}")
            # Fallback to individual tokenization
            lengths = []
            for text in texts:
                try:
                    tokens = self.tokenizer(text, truncation=False)["input_ids"] # type: ignore
                    lengths.append(len(tokens))
                except:
                    lengths.append(float("inf"))  # Mark as invalid
            return lengths

    def _process_batch(
        self, batch_data: List[Tuple[str, int]], max_length: int, min_length: int
    ) -> Tuple[List[Dict[str, str]], int, int]:
        """
        Process a batch of texts with early filtering and batch tokenization.

        Args:
            batch_data: List of (text, line_number) tuples
            max_length: Maximum token length allowed
            min_length: Minimum token length required

        Returns:
            (valid_records, filtered_count, error_count)
        """
        if not batch_data:
            return [], 0, 0

        # Early filtering by approximate length
        pre_filtered_batch = []
        early_filtered_count = 0

        for text, line_num in batch_data:
            formatted_text = f"<|im_start|>{text}<|im_end|>"
            # Quick length estimation to avoid expensive tokenization
            estimated_length = self._estimate_token_length(formatted_text)

            # If estimated length is way over limit or under minimum, skip expensive tokenization
            if (
                estimated_length > max_length * 1.5 or estimated_length < min_length * 0.7
            ):  # Add some buffer for estimation error
                early_filtered_count += 1
                continue

            pre_filtered_batch.append((formatted_text, line_num))

        if not pre_filtered_batch:
            return [], early_filtered_count, 0

        # Batch tokenization for remaining texts
        texts_to_tokenize = [item[0] for item in pre_filtered_batch]
        token_lengths = self._batch_tokenize(texts_to_tokenize)

        # Filter by actual token length and build output records
        valid_records = []
        length_filtered_count = 0
        error_count = 0

        for (formatted_text, line_num), token_length in zip(
            pre_filtered_batch, token_lengths
        ):
            if token_length == float("inf"):  # Tokenization error
                error_count += 1
                continue

            if token_length > max_length or token_length < min_length:
                length_filtered_count += 1
                continue

            valid_records.append({"text": formatted_text})

        total_filtered = early_filtered_count + length_filtered_count
        return valid_records, total_filtered, error_count

    def _write_buffer(self, buffer: List[Dict[str, str]], outfile) -> None:
        """Write buffered records to file efficiently."""
        if not buffer:
            return

        # Batch JSON serialization and write in one operation
        json_lines = [json.dumps(record, ensure_ascii=False) for record in buffer]
        outfile.write("\n".join(json_lines) + "\n")

    def _get_file_size(self, filepath: str) -> int:
        """Get approximate file size for progress reporting."""
        try:
            return os.path.getsize(filepath)
        except:
            return 0

    def _show_progress(
        self,
        processed: int,
        total_size: int,
        start_time: float,
        skipped: int,
        filtered: int,
    ) -> None:
        """Show processing progress with statistics."""
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0

        if total_size > 0:
            # Rough progress estimation based on processed lines
            progress = min(
                100, (processed + skipped + filtered) * 100 / (total_size / 100)
            )
            print(
                f"\rProgress: {progress:.1f}% | "
                f"Processed: {processed:,} | "
                f"Skipped: {skipped:,} | "
                f"Filtered: {filtered:,} | "
                f"Rate: {rate:.0f} lines/sec",
                end="",
                flush=True,
            )
        else:
            print(
                f"\rProcessed: {processed:,} | "
                f"Skipped: {skipped:,} | "
                f"Filtered: {filtered:,} | "
                f"Rate: {rate:.0f} lines/sec",
                end="",
                flush=True,
            )


def process_file(
    input_path: str,
    output_path: str,
    text_key: str,
    max_length: int = 1024,
    min_length: int = 256,
    batch_size: int = 1000,
    buffer_size: int = 10000,
    num_workers: Optional[int] = None,
    model_path: str = "model/",
):
    """
    Optimized file processor with batch processing, buffered I/O, and progress reporting.

    Args:
        input_path (str): 输入文件的路径
        output_path (str): 输出的 .jsonl 文件的路径
        text_key (str): 当输入是 .jsonl 文件时，需要提取文本的键名
        max_length (int): 最大token长度，超过此长度的文本将被跳过
        min_length (int): 最小token长度，小于此长度的文本将被跳过
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"INFO: 开始处理文件: '{input_path}'")
    print(f"INFO: 输出将保存至: '{output_path}'")
    print(f"INFO: 最大token长度限制: {max_length}")
    print(f"INFO: 最小token长度限制: {min_length}")

    # Initialize processor
    processor = OptimizedTextProcessor(
        batch_size=batch_size, 
        buffer_size=buffer_size, 
        num_workers=num_workers,
        model_path=model_path
    )
    if not processor._load_tokenizer():
        sys.exit(1)
        
    print(f"INFO: 使用 {processor.num_workers} 个worker进程进行并行处理")
    print(f"INFO: Worker chunk大小: {processor.chunk_size}")

    # Get file size for progress reporting
    file_size = processor._get_file_size(input_path)
    is_jsonl = input_path.lower().endswith(".jsonl")

    if is_jsonl:
        print(f"INFO: 检测到输入为 JSONL 文件。将提取键 '{text_key}' 的内容。")
    else:
        print("INFO: 检测到输入为纯文本文件。将处理每一行。")

    # Statistics
    processed_count = 0
    skipped_count = 0
    length_filtered_count = 0
    start_time = time.time()

    def read_file_chunks(filepath: str) -> Iterator[List[Tuple[str, int]]]:
        """Generator that yields chunks of lines from the input file."""
        chunk_data = []
        line_num = 0
        
        with open(filepath, "r", encoding="utf-8", buffering=8192) as infile:
            for line in infile:
                line_num += 1
                chunk_data.append((line, line_num))
                
                if len(chunk_data) >= processor.chunk_size:
                    yield chunk_data
                    chunk_data = []
            
            # Yield remaining data
            if chunk_data:
                yield chunk_data
    
    try:
        with open(output_path, "w", encoding="utf-8", buffering=8192) as outfile:
            output_buffer = []
            
            # Create worker function with fixed parameters
            worker_func = partial(
                worker_process_chunk,
                max_length=max_length,
                min_length=min_length,
                is_jsonl=is_jsonl,
                text_key=text_key,
                model_path=model_path
            )
            
            # Create process pool
            with mp.Pool(
                processes=processor.num_workers,
                initializer=init_worker_tokenizer
            ) as pool:
                
                # Process chunks in parallel using imap_unordered for overlapping IO/computation
                chunk_generator = read_file_chunks(input_path)
                results_iter = pool.imap_unordered(worker_func, chunk_generator, chunksize=1)
                
                chunks_processed = 0
                for valid_records, filtered_count, error_count, chunk_skipped_count in results_iter:
                    chunks_processed += 1
                    
                    # Update statistics
                    processed_count += len(valid_records)
                    length_filtered_count += filtered_count
                    skipped_count += error_count + chunk_skipped_count
                    
                    # Add to output buffer
                    output_buffer.extend(valid_records)
                    
                    # Write buffer if it's full
                    if len(output_buffer) >= processor.buffer_size:
                        processor._write_buffer(output_buffer, outfile)
                        output_buffer.clear()
                    
                    # Show progress periodically
                    if chunks_processed % 5 == 0:
                        processor._show_progress(
                            processed_count,
                            file_size,
                            start_time,
                            skipped_count,
                            length_filtered_count,
                        )
            
            # Write remaining buffer
            if output_buffer:
                processor._write_buffer(output_buffer, outfile)

    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_path}' 不存在。请检查路径是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"\n严重错误: 处理过程中发生错误: {e}")
        sys.exit(1)

    # Final statistics
    elapsed_time = time.time() - start_time
    total_lines = processed_count + skipped_count + length_filtered_count

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  - 成功处理行数: {processed_count:,}")
    print(f"  - 跳过行数: {skipped_count:,}")
    print(f"  - 因长度超限过滤: {length_filtered_count:,}")
    print(f"  - 总处理行数: {total_lines:,}")
    print(f"  - 处理时间: {elapsed_time:.2f} 秒")
    print(
        f"  - 处理速度: {total_lines/elapsed_time:.0f} 行/秒"
        if elapsed_time > 0
        else "  - 处理速度: N/A"
    )
    print(f"  - 结果已保存至: '{output_path}'")
    print("=" * 60)


def main():
    """主函数，用于解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="一个通用的数据格式化工具，将txt或jsonl文件转换为模型微调所需的格式。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="必需：输入文件的路径（支持 .txt 或 .jsonl 格式）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="必需：输出的 .jsonl 文件的路径。",
    )
    parser.add_argument(
        "--text_key",
        "-k",
        type=str,
        default="text",
        help="可选：当输入文件为 .jsonl 时，指定包含文本的键名。\n默认为 'text'。",
    )
    parser.add_argument(
        "--max_length",
        "-m",
        type=int,
        default=1024,
        help="可选：最大token长度限制，超过此长度的文本将被过滤。\n默认为 1024。",
    )
    parser.add_argument(
        "--min_length",
        "-n",
        type=int,
        default=256,
        help="可选：最小token长度限制，小于此长度的文本将被过滤。\n默认为 256。",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1000,
        help="可选：批处理大小，用于批量tokenization。\n默认为 1000。",
    )
    parser.add_argument(
        "--buffer_size",
        "-s",
        type=int,
        default=10000,
        help="可选：输出缓冲区大小，用于批量写入文件。\n默认为 10000。",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=None,
        help="可选：并行处理的worker进程数量。\n默认为CPU核心数。",
    )
    parser.add_argument(
        "--model_path",
        "-p",
        type=str,
        default="model/",
        help="可选：tokenizer模型路径。\n默认为 'model/'。",
    )

    args = parser.parse_args()
    process_file(
        args.input,
        args.output,
        args.text_key,
        args.max_length,
        args.min_length,
        args.batch_size,
        args.buffer_size,
        args.num_workers,
        args.model_path,
    )


if __name__ == "__main__":
    main()
