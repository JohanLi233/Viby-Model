from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    """生成超参。字段名按常见 serving API 对齐，便于以后挂 HTTP。"""

    max_new_tokens: int = 128
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    eos_token_id: Optional[int] = 2
    n: int = 1
    logprobs: bool = False
    use_mtp_speculative: bool = False
    num_speculative_tokens: Optional[int] = None


@dataclass
class CompletionOutput:
    index: int
    token_ids: list[int]
    text: str = ""
    logprobs: Optional[list[float]] = None
    finish_reason: Optional[str] = None


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False
