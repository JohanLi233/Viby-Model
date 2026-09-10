"""Viby 推理引擎：进程内 generate / score，接口形状预留给后续 HTTP。

新架构（DeepSeek-V4.1 缩放版）版本的实现：
- engine/engine.py   调度 + 连续 batch + 采样
- engine/memory.py   三类 KV 池（window / compress_kv / index_k）的行级状态池
- engine/prefix.py   token 前缀 → 状态快照的复用表
- engine/sampling.py temperature / top_k / top_p / repetition_penalty
"""

from .engine import Sequence, VibyEngine
from .memory import PrefixState, StatePool, capture_state, restore_state
from .prefix import RadixPrefixCache
from .types import CompletionOutput, RequestOutput, SamplingParams

__all__ = [
    "VibyEngine",
    "Sequence",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "StatePool",
    "RadixPrefixCache",
    "PrefixState",
    "capture_state",
    "restore_state",
]
