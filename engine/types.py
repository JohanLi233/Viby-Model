"""引擎的公共数据类型（对外接口名保持不变）。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    """生成超参。字段名沿用旧引擎（serving API 风格），新增 stop 相关字段。"""

    max_new_tokens: int = 128
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: Optional[int] = 2
    n: int = 1
    logprobs: bool = False
    # DSpark 投机解码尚未在新引擎里实现（TODO）：保留字段以兼容老调用方。
    use_mtp_speculative: bool = False
    num_speculative_tokens: Optional[int] = None
    # ---- 新增：停止条件 ----
    stop: Optional[list] = None  # 停止字符串（需要 tokenizer），命中即截断
    stop_token_ids: Optional[list] = None  # 命中即停的 token id


@dataclass
class CompletionOutput:
    index: int
    token_ids: list
    text: str = ""
    logprobs: Optional[list] = None
    finish_reason: Optional[str] = None


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list
    outputs: list = field(default_factory=list)
    finished: bool = False
