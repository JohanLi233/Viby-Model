"""采样与停止条件：temperature / top_k / top_p / repetition_penalty。

语义沿用旧引擎（transformers 风格），处理顺序固定为：
    重复惩罚 → temperature → top_k → top_p → 采样（do_sample=False 或
    temperature<=0 时直接 argmax）。
greedy 路径（temperature<=0 且 penalty=1）与 `model.sample_tokens` 的
argmax 完全一致，保证 engine 与 `VibyForCausalLM.generate` 逐 token 对齐。
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .types import SamplingParams

NEG_INF = -1e30


def log_softmax(logits: mx.array) -> mx.array:
    """数值稳定的 log_softmax（fp32）。"""
    x = logits.astype(mx.float32)
    m = mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x - m)
    return x - m - mx.log(mx.sum(e, axis=-1, keepdims=True))


def _repetition_penalty(row: mx.array, seen_ids, penalty: float) -> mx.array:
    """对已出现 token 的 logit 按符号缩放（transformers 的写法）。"""
    if penalty is None or penalty == 1.0 or not seen_ids:
        return row
    ids = mx.array(sorted({int(t) for t in seen_ids}), dtype=mx.int32)
    vals = row[ids]
    scaled = mx.where(vals < 0, vals * penalty, vals / penalty)
    return row.at[ids].add(scaled - row[ids])


def _top_k(row: mx.array, top_k: int) -> mx.array:
    if not top_k or top_k <= 0 or top_k >= row.shape[-1]:
        return row
    k = int(top_k)
    thr = mx.partition(row, kth=row.shape[-1] - k)[..., row.shape[-1] - k]
    return mx.where(row >= thr, row, NEG_INF)


def _top_p(row: mx.array, top_p: float) -> mx.array:
    if top_p is None or top_p >= 1.0:
        return row
    order = mx.argsort(-row)
    sorted_l = row[order]
    probs = mx.softmax(sorted_l)
    cum = mx.cumsum(probs)
    # 保留"累积概率首次超过 top_p"的那个 token（经典 nucleus 实现）
    keep = mx.concatenate([mx.array([True]), (cum - probs)[1:] < top_p])
    filtered = mx.where(keep, sorted_l, NEG_INF)
    return filtered[mx.argsort(order)]


def transform_logits(row: mx.array, seen_ids, params: SamplingParams) -> mx.array:
    """对单行 logits [V] 施加采样超参，返回处理后的 logits [V]。"""
    row = row.astype(mx.float32)
    row = _repetition_penalty(row, seen_ids, params.repetition_penalty)
    if not params.do_sample or params.temperature <= 0:
        return row
    row = row / float(params.temperature)
    row = _top_k(row, params.top_k)
    row = _top_p(row, params.top_p)
    return row


def sample_one(row: mx.array, params: SamplingParams) -> int:
    """从处理后的单行 logits 里取一个 token（greedy 或 categorical）。"""
    if not params.do_sample or params.temperature <= 0:
        return int(mx.argmax(row).item())
    return int(mx.random.categorical(row[None])[0].item())


def speculative_correction(target, proposal, token, params, *, uniform=None):
    """Accept a proposal with min(1,p/q), otherwise sample normalized (p-q)+.

    target is the target's already transformed logit row; proposal is the
    normalized distribution that actually produced token (same filters/history).
    Greedy uses exact target argmax matching and never probabilistic acceptance.
    """
    if not params.do_sample or params.temperature <= 0:
        expected = int(mx.argmax(target).item())
        return expected, expected == token
    p = mx.softmax(target.astype(mx.float32))
    pt, qt = float(p[token].item()), float(proposal[token].item())
    alpha = min(1.0, pt / qt) if qt > 0 else 0.0
    u = float(mx.random.uniform().item()) if uniform is None else float(uniform)
    if u < alpha:
        return token, True
    residual = mx.maximum(p - proposal, 0)
    mass = mx.sum(residual)
    # A rounding-degenerate residual must still produce a valid target draw.
    residual = mx.where(mass > 0, residual / mx.maximum(mass, 1e-30), p)
    corrected = int(mx.random.categorical(mx.log(residual)[None])[0].item())
    return corrected, False


def find_stop_length(tokenizer, gen_ids: list, stops: Optional[list]) -> Optional[int]:
    """停止字符串命中时返回应保留的 token 数（其余截掉），未命中返回 None。

    逐个回退尾 token 再 decode：token 与字符不是一一对应，只能这样精确定位。
    """
    if not stops or tokenizer is None or not gen_ids:
        return None
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if not any(s in text for s in stops if s):
        return None
    keep = len(gen_ids)
    while keep > 0:
        keep -= 1
        t = tokenizer.decode(gen_ids[:keep], skip_special_tokens=True)
        if not any(s in t for s in stops if s):
            return keep
    return 0
