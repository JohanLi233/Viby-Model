"""进程内推理引擎：continuous batch + gather 分页 KV + radix 前缀复用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from model.cache import KVCache, _eval_kv_caches
from model.model import (
    VibyForCausalLM,
    _probs_mx,
    _sample_from_logits_mx,
    _transform_logits_mx,
)

from .memory import BatchSlots, PagePool, PagedKVCache
from .prefix import RadixPrefixCache
from .types import CompletionOutput, RequestOutput, SamplingParams

_NEG = -1e9


def _log_softmax(logits: mx.array) -> mx.array:
    x = logits.astype(mx.float32)
    m = mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x - m)
    return x - m - mx.log(mx.sum(e, axis=-1, keepdims=True))


def _attn_bias(starts: list[int], q_lens: list[int], k_len: int, mask: mx.array, dtype):
    """query i 可见 key j iff j <= start+i 且 j < end 且 query 本身有效。"""
    len(starts)
    q_len = mask.shape[1]
    start = mx.array(starts, dtype=mx.int32)
    end = mx.array([s + q for s, q in zip(starts, q_lens)], dtype=mx.int32)
    q_pos = start[:, None] + mx.arange(q_len, dtype=mx.int32)[None, :]
    k_idx = mx.arange(k_len, dtype=mx.int32)[None, None, :]
    causal = k_idx <= q_pos[:, :, None]
    k_ok = k_idx < end[:, None, None]
    q_ok = mask.astype(mx.bool_)[:, :, None]
    allowed = causal & k_ok & q_ok
    return mx.where(allowed, mx.array(0.0, dtype=dtype), mx.array(_NEG, dtype=dtype))[
        :, None, :, :
    ]


@dataclass
class Sequence:
    request_id: str
    output_index: int
    prompt_ids: list[int]
    token_ids: list[int]
    max_new: int
    page_ids: list[int] = field(default_factory=list)
    seqlen: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None
    logprobs: list[float] = field(default_factory=list)
    mtp_page_ids: list[int] = field(default_factory=list)
    mtp_seqlen: int = 0
    page_seqlen: int = 0
    mtp_page_seqlen: int = 0
    kv_caches: Optional[list] = None
    mtp_cache: Optional[KVCache] = None
    feat_last: Optional[mx.array] = None
    logits_last: Optional[mx.array] = None
    held_bonus: Optional[int] = None

    @property
    def n_generated(self) -> int:
        return max(0, len(self.token_ids) - len(self.prompt_ids))


class VibyEngine:
    """MLA 专用生成引擎。``use_linear_attn`` 模型请继续走 ``model.generate``。"""

    def __init__(
        self,
        model: VibyForCausalLM,
        tokenizer=None,
        *,
        page_size: int = 16,
        max_num_seqs: int = 8,
        max_num_pages: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ):
        cfg = model.config
        if bool(getattr(cfg, "use_linear_attn", False)):
            raise ValueError(
                "VibyEngine 只支持 MLA（use_linear_attn=False）；"
                "线性注意力请用 model.generate"
            )
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.page_size = int(page_size)
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len or cfg.max_position_embeddings)
        self.max_blocks = (self.max_model_len + self.page_size - 1) // self.page_size
        n_pages = max_num_pages or (
            self.max_num_seqs * self.max_blocks + self.max_blocks
        )
        dtype = model.model.embed_tokens.weight.dtype
        self.pool = PagePool(
            num_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            page_size=self.page_size,
            k_dim=cfg.head_dim + cfg.qk_rope_head_dim,
            v_dim=cfg.head_dim,
            num_pages=int(n_pages),
            dtype=dtype,
        )
        self.prefix = RadixPrefixCache(self.page_size, self.pool)
        self.mtp_pool: Optional[PagePool] = None
        self.mtp_prefix: Optional[RadixPrefixCache] = None
        if len(model.mtp_modules) > 0:
            mtp_pages = max_num_pages or (
                self.max_num_seqs * self.max_blocks + self.max_blocks
            )
            self.mtp_pool = PagePool(
                num_layers=1,
                n_heads=cfg.num_attention_heads,
                page_size=self.page_size,
                k_dim=cfg.head_dim + cfg.qk_rope_head_dim,
                v_dim=cfg.head_dim,
                num_pages=int(mtp_pages),
                dtype=dtype,
            )
            self.mtp_prefix = RadixPrefixCache(self.page_size, self.mtp_pool)
        self.stats = {
            "prefix_tokens": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "mtp_accepted": 0,
            "mtp_drafted": 0,
        }

    def generate(
        self,
        prompts: list[Union[str, list[int]]],
        params: Optional[SamplingParams] = None,
        streamer=None,
    ) -> list[RequestOutput]:
        params = params or SamplingParams()
        if params.n < 1:
            raise ValueError("n 必须 >= 1")
        if params.max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负")
        self.stats = {
            "prefix_tokens": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "mtp_accepted": 0,
            "mtp_drafted": 0,
        }
        spec = self._spec_enabled(params)

        reqs: list[RequestOutput] = []
        waiting: list[Sequence] = []
        running: list[Sequence] = []
        for i, prompt in enumerate(prompts):
            ids = self._encode(prompt)
            if not ids:
                raise ValueError("prompt 不能为空")
            if len(ids) >= self.max_model_len:
                raise ValueError(
                    f"prompt 长度 {len(ids)} 超过 max_model_len {self.max_model_len}"
                )
            rid = str(i)
            reqs.append(RequestOutput(request_id=rid, prompt_token_ids=list(ids)))
            max_new = min(params.max_new_tokens, self.max_model_len - len(ids))
            for k in range(params.n):
                waiting.append(
                    Sequence(
                        request_id=rid,
                        output_index=k,
                        prompt_ids=list(ids),
                        token_ids=list(ids),
                        max_new=max_new,
                    )
                )

        all_seqs = list(waiting)
        stream_seq = (
            all_seqs[0]
            if streamer is not None and len(prompts) == 1 and params.n == 1
            else None
        )
        if stream_seq is not None:
            streamer.put([stream_seq.prompt_ids])

        while waiting or running:
            still_wait = []
            for seq in waiting:
                if seq.seqlen == 0:
                    self._try_prefix(seq, spec=spec)
                if seq.seqlen < len(seq.prompt_ids):
                    still_wait.append(seq)
                elif not seq.finished:
                    running.append(seq)
            waiting = still_wait

            if waiting:
                batch, waiting = self._schedule_prefill(waiting)
                prev_n = stream_seq.n_generated if stream_seq is not None else 0
                self._prefill(batch, params, spec=spec)
                self._maybe_stream(streamer, stream_seq, prev_n)
                for seq in batch:
                    self._commit_prompt_prefix(seq)
                    if seq.finished:
                        self._release(seq)
                    else:
                        running.append(seq)
                continue

            if not running:
                break
            batch = running[: self.max_num_seqs]
            running = running[self.max_num_seqs :]
            prev_n = stream_seq.n_generated if stream_seq is not None else 0
            if self._spec_enabled(params):
                self._spec_decode(batch, params)
            else:
                self._forward(batch, params, decode=True)
            self._maybe_stream(streamer, stream_seq, prev_n)
            still = []
            for seq in batch:
                if seq.finished:
                    self._release(seq)
                else:
                    still.append(seq)
            running = still + running
            if stream_seq is not None and stream_seq.finished:
                break

        if streamer is not None and stream_seq is not None:
            streamer.end()

        by_req = {r.request_id: r for r in reqs}
        for seq in all_seqs:
            gen_ids = seq.token_ids[len(seq.prompt_ids) :]
            text = ""
            if self.tokenizer is not None and gen_ids:
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            by_req[seq.request_id].outputs.append(
                CompletionOutput(
                    index=seq.output_index,
                    token_ids=gen_ids,
                    text=text,
                    logprobs=list(seq.logprobs) if params.logprobs else None,
                    finish_reason=seq.finish_reason or "length",
                )
            )
        for r in reqs:
            r.outputs.sort(key=lambda o: o.index)
            r.finished = True
        return reqs

    def score(
        self,
        sequences: list[list[int]],
        prompt_lens: list[int],
    ) -> mx.array:
        """Teacher-forced 完成段 logprob，形状 (B, max_gen_len)，短序列 pad 0。"""
        if len(sequences) != len(prompt_lens):
            raise ValueError("sequences 与 prompt_lens 长度不一致")
        if not sequences:
            return mx.zeros((0, 0))
        max_t = max(len(s) for s in sequences)
        bsz = len(sequences)
        ids = np.zeros((bsz, max_t), dtype=np.int32)
        mask = np.zeros((bsz, max_t), dtype=np.int32)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1
        out = self.model(
            mx.array(ids),
            attention_mask=mx.array(mask),
            mask_has_pad=True,
            use_cache=False,
        )
        logp = _log_softmax(out.logits)
        gen = max(len(sequences[i]) - prompt_lens[i] for i in range(bsz))
        if gen <= 0:
            return mx.zeros((bsz, 0))
        rows = []
        for i, seq in enumerate(sequences):
            pl = prompt_lens[i]
            n = len(seq) - pl
            if n <= 0:
                rows.append(mx.zeros((gen,)))
                continue
            tok = mx.array(seq[pl : pl + n], dtype=mx.int32)
            lp = logp[i, pl - 1 : pl - 1 + n]
            taken = mx.take_along_axis(lp, tok[:, None], axis=-1).reshape((n,))
            if n < gen:
                taken = mx.concatenate([taken, mx.zeros((gen - n,), dtype=taken.dtype)])
            rows.append(taken)
        return mx.stack(rows)

    def _encode(self, prompt: Union[str, list[int]]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("字符串 prompt 需要 tokenizer")
            return list(
                self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_model_len,
                ).input_ids
            )
        return [int(t) for t in prompt]

    def _schedule_prefill(
        self, waiting: list[Sequence]
    ) -> tuple[list[Sequence], list[Sequence]]:
        """同一 prompt 只进一条 prefill，其余等插入前缀树后再 match。"""
        batch: list[Sequence] = []
        delayed: list[Sequence] = []
        leftover: list[Sequence] = []
        seen: set[tuple[int, ...]] = set()
        for seq in waiting:
            if len(batch) >= self.max_num_seqs:
                leftover.append(seq)
                continue
            key = tuple(seq.prompt_ids)
            if key in seen:
                delayed.append(seq)
            else:
                seen.add(key)
                batch.append(seq)
        return batch, delayed + leftover

    def _spec_enabled(self, params: SamplingParams) -> bool:
        return bool(params.use_mtp_speculative) and self.mtp_pool is not None

    def _try_prefix(self, seq: Sequence, spec: bool = False) -> None:
        n_raw, pages = self.prefix.match(seq.token_ids)
        L = len(seq.prompt_ids)
        ps = self.page_size
        if spec and self.mtp_prefix is not None:
            n_mtp, mtp_pages = (0, [])
            if L > 1:
                n_mtp, mtp_pages = self.mtp_prefix.match(seq.prompt_ids[:-1])
            usable = n_raw >= max(L - 1, 0) and (L <= 1 or n_mtp >= L - 1)
            if not usable:
                return
            self.stats["prefix_tokens"] += n_raw
            n = min(n_raw, max(L - 1, 0))
            n_pages = (n + ps - 1) // ps if n else 0
            pages = pages[:n_pages]
            if pages:
                self.pool.retain(pages)
            seq.page_ids = list(pages)
            seq.seqlen = n
            seq.page_seqlen = n
            n_mtp_pages = (n_mtp + ps - 1) // ps if n_mtp else 0
            mtp_pages = mtp_pages[:n_mtp_pages]
            if mtp_pages:
                self.mtp_pool.retain(mtp_pages)
            seq.mtp_page_ids = list(mtp_pages)
            seq.mtp_seqlen = n_mtp
            seq.mtp_page_seqlen = n_mtp
            return
        self.stats["prefix_tokens"] += n_raw
        n = min(n_raw, max(L - 1, 0))
        n_pages = (n + ps - 1) // ps if n else 0
        pages = pages[:n_pages]
        if pages:
            self.pool.retain(pages)
        seq.page_ids = list(pages)
        seq.seqlen = n
        seq.page_seqlen = n

    def _commit_prompt_prefix(self, seq: Sequence) -> None:
        n = len(seq.prompt_ids)
        if seq.kv_caches is not None:
            self._spill_kv_to_pages(seq, n)
        n_pages = (n + self.page_size - 1) // self.page_size
        self.prefix.insert(seq.prompt_ids, seq.page_ids[:n_pages])
        if self.mtp_prefix is not None and n > 1 and seq.mtp_seqlen >= n - 1:
            if seq.mtp_cache is not None:
                self._spill_kv_to_pages(seq, n - 1, mtp=True)
            mtp_tok = seq.prompt_ids[:-1]
            n_mtp = (len(mtp_tok) + self.page_size - 1) // self.page_size
            self.mtp_prefix.insert(mtp_tok, seq.mtp_page_ids[:n_mtp])

    def _alloc(self, n: int, *, mtp: bool = False) -> list[int]:
        pool = self.mtp_pool if mtp else self.pool
        tree = self.mtp_prefix if mtp else self.prefix
        try:
            return pool.alloc(n)
        except MemoryError:
            if tree is not None:
                tree.reset()
            return pool.alloc(n)

    def _prepare_pages(self, seq: Sequence, n_new: int, *, mtp: bool = False) -> None:
        if n_new <= 0:
            return
        ps = self.page_size
        pages = seq.mtp_page_ids if mtp else seq.page_ids
        seqlen = seq.mtp_seqlen if mtp else seq.seqlen
        pool = self.mtp_pool if mtp else self.pool
        if pages and seqlen % ps != 0:
            last = pages[-1]
            if pool.refcount[last] > 1:
                pages[-1] = pool.cow(last)
        need = (seqlen + n_new + ps - 1) // ps
        extra = need - len(pages)
        if extra > 0:
            pages.extend(self._alloc(extra, mtp=mtp))
        if mtp:
            seq.mtp_page_ids = pages
        else:
            seq.page_ids = pages

    def _rewind(self, seq: Sequence, new_len: int, *, mtp: bool = False) -> None:
        new_len = max(0, int(new_len))
        if mtp and seq.mtp_cache is not None:
            seq.mtp_cache.rewind(new_len)
        elif not mtp and seq.kv_caches is not None:
            for cache in seq.kv_caches:
                cache.rewind(new_len)
        ps = self.page_size
        pages = seq.mtp_page_ids if mtp else seq.page_ids
        pool = self.mtp_pool if mtp else self.pool
        need = (new_len + ps - 1) // ps if new_len else 0
        extra = pages[need:]
        keep = pages[:need]
        if extra:
            pool.free_pages(extra)
        if mtp:
            seq.mtp_page_ids = keep
            seq.mtp_seqlen = new_len
            if seq.mtp_page_seqlen > new_len:
                seq.mtp_page_seqlen = new_len
        else:
            seq.page_ids = keep
            seq.seqlen = new_len
            if seq.page_seqlen > new_len:
                seq.page_seqlen = new_len

    def _release(self, seq: Sequence) -> None:
        self.pool.free_pages(seq.page_ids)
        seq.page_ids = []
        seq.page_seqlen = 0
        seq.kv_caches = None
        seq.held_bonus = None
        if self.mtp_pool is not None:
            self.mtp_pool.free_pages(seq.mtp_page_ids)
            seq.mtp_page_ids = []
            seq.mtp_page_seqlen = 0
            seq.mtp_cache = None

    def _ngram_tails(self, seqs: list[Sequence]) -> Optional[mx.array]:
        ngram = self.model.model.ngram
        if ngram is None:
            return None
        ctx = ngram.context_len
        rows = []
        for seq in seqs:
            hist = seq.token_ids[: seq.seqlen]
            if len(hist) >= ctx:
                row = hist[-ctx:]
            else:
                row = [0] * (ctx - len(hist)) + hist
            rows.append(row)
        return mx.array(rows, dtype=mx.int32)

    def _paged_inputs(
        self,
        seqs: list[Sequence],
        q_ids: list[list[int]],
        q_lens: list[int],
        *,
        mtp: bool = False,
    ):
        q_max = max(q_lens)
        bsz = len(seqs)
        ids_np = np.zeros((bsz, q_max), dtype=np.int32)
        mask_np = np.zeros((bsz, q_max), dtype=np.bool_)
        for i, q in enumerate(q_ids):
            ids_np[i, : len(q)] = q
            mask_np[i, : len(q)] = True
            self._prepare_pages(seqs[i], len(q), mtp=mtp)
        token_mask = mx.array(mask_np)
        input_ids = mx.array(ids_np)
        starts = [(s.mtp_seqlen if mtp else s.seqlen) for s in seqs]
        slots = BatchSlots(
            seqlens=list(starts), q_len=q_max, token_mask=token_mask, q_lens=q_lens
        )
        table = np.zeros((bsz, self.max_blocks), dtype=np.int32)
        for i, seq in enumerate(seqs):
            pids = seq.mtp_page_ids if mtp else seq.page_ids
            table[i, : len(pids)] = pids
        block_table = mx.array(table)
        extras: dict = {}
        if not mtp:
            tails = self._ngram_tails(seqs)
            if tails is not None:
                extras["ngram_tail"] = tails
        pool = self.mtp_pool if mtp else self.pool
        caches = [
            PagedKVCache(pool, layer, block_table, slots, extras)
            for layer in range(pool.num_layers)
        ]
        pos = (
            mx.array(starts, dtype=mx.int32)[:, None]
            + mx.arange(q_max, dtype=mx.int32)[None, :]
        )
        pos = mx.where(token_mask, pos, mx.zeros_like(pos))
        end_lens = [s + q for s, q in zip(starts, q_lens)]
        k_len = max(end_lens) if end_lens else 0
        dtype = self.model.model.embed_tokens.weight.dtype
        attn_bias = _attn_bias(starts, q_lens, k_len, token_mask, dtype)
        return input_ids, caches, pos, attn_bias, end_lens

    def _offset_rope_ok(
        self, seqs: list[Sequence], q_lens: list[int], *, mtp: bool
    ) -> bool:
        """同起点且无 query padding 时走 mx.fast.rope，与 model.generate 一致。"""
        starts = [(s.mtp_seqlen if mtp else s.seqlen) for s in seqs]
        q_max = max(q_lens) if q_lens else 0
        return len(set(starts)) == 1 and all(q == q_max for q in q_lens)

    def _block_table(self, seqs: list[Sequence], *, mtp: bool = False) -> mx.array:
        table = np.zeros((len(seqs), self.max_blocks), dtype=np.int32)
        for i, seq in enumerate(seqs):
            pids = seq.mtp_page_ids if mtp else seq.page_ids
            table[i, : len(pids)] = pids
        return mx.array(table)

    def _load_kv_from_pages(self, seq: Sequence, *, mtp: bool = False):
        """把已物化的 page 收成 KVCache，供单序列稠密 decode。"""
        page_len = seq.mtp_page_seqlen if mtp else seq.page_seqlen
        pool = self.mtp_pool if mtp else self.pool
        n_layers = pool.num_layers
        if page_len <= 0:
            empty = [KVCache() for _ in range(n_layers)]
            return empty[0] if mtp else empty
        slots = BatchSlots(seqlens=[page_len], q_len=0, q_lens=[0])
        block_table = self._block_table([seq], mtp=mtp)
        out: list[KVCache] = []
        for layer in range(n_layers):
            k, v = PagedKVCache(pool, layer, block_table, slots).gather()
            cache = KVCache()
            cache.update(k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
            out.append(cache)
        if not mtp:
            ngram = self.model.model.ngram
            if ngram is not None:
                ctx = ngram.context_len
                hist = seq.token_ids[: seq.seqlen]
                if len(hist) >= ctx:
                    row = hist[-ctx:]
                else:
                    row = [0] * (ctx - len(hist)) + hist
                out[0].extras["ngram_tail"] = mx.array([row], dtype=mx.int32)
        _eval_kv_caches(out)
        return out[0] if mtp else out

    def _spill_kv_to_pages(
        self, seq: Sequence, target: int, *, mtp: bool = False
    ) -> None:
        """把稠密 KV 的 [page_seqlen, target) 写进 page，供前缀树分享。"""
        caches = [seq.mtp_cache] if mtp else seq.kv_caches
        if not caches or caches[0] is None:
            return
        page_len = seq.mtp_page_seqlen if mtp else seq.page_seqlen
        if target <= page_len:
            return
        live = seq.mtp_seqlen if mtp else seq.seqlen
        if mtp:
            seq.mtp_seqlen = page_len
        else:
            seq.seqlen = page_len
        try:
            self._prepare_pages(seq, target - page_len, mtp=mtp)
            q_len = target - page_len
            token_mask = mx.ones((1, q_len), dtype=mx.bool_)
            slots = BatchSlots(
                seqlens=[page_len], q_len=q_len, token_mask=token_mask, q_lens=[q_len]
            )
            block_table = self._block_table([seq], mtp=mtp)
            pool = self.mtp_pool if mtp else self.pool
            for layer, cache in enumerate(caches):
                k = cache.keys[:, :, page_len:target].transpose(0, 2, 1, 3)
                v = cache.values[:, :, page_len:target].transpose(0, 2, 1, 3)
                PagedKVCache(pool, layer, block_table, slots).update(k, v)
            mx.eval(*pool.keys, *pool.values)
            if mtp:
                seq.mtp_page_seqlen = target
            else:
                seq.page_seqlen = target
        finally:
            if mtp:
                seq.mtp_seqlen = live
            else:
                seq.seqlen = live

    def _run_trunk_dense(
        self,
        seq: Sequence,
        q_ids: list[int],
        *,
        output_features: bool = False,
        decode: bool = False,
    ):
        if seq.kv_caches is None:
            seq.kv_caches = (
                self._load_kv_from_pages(seq)
                if seq.seqlen > 0
                else [KVCache() for _ in range(self.model.config.num_hidden_layers)]
            )
        out = self.model(
            mx.array([q_ids], dtype=mx.int32),
            past_key_values=seq.kv_caches,
            use_cache=True,
            mask_has_pad=False,
            output_features=output_features,
        )
        bufs = [out.logits]
        if output_features and out.features:
            bufs.append(out.features[-1])
        mx.eval(*bufs)
        _eval_kv_caches(out.past_key_values)
        seq.kv_caches = list(out.past_key_values)
        seq.seqlen = seq.kv_caches[0].offset
        if decode:
            self.stats["decode_tokens"] += len(q_ids)
        else:
            self.stats["prefill_tokens"] += len(q_ids)
        return out

    def _run_mtp_dense(
        self, seq: Sequence, h_in: mx.array, q_ids: list[int]
    ) -> mx.array:
        if seq.mtp_cache is None:
            seq.mtp_cache = (
                self._load_kv_from_pages(seq, mtp=True)
                if seq.mtp_seqlen > 0
                else KVCache()
            )
        input_ids = mx.array([q_ids], dtype=mx.int32)
        emb = self.model.model.embed_tokens(input_ids)
        pos_emb = self.model.model.position_embeddings(
            seq.mtp_seqlen, input_ids.shape[1], emb.dtype
        )
        h_out, present = self.model.mtp_modules[0](
            h_in,
            emb,
            past_key_value=seq.mtp_cache,
            use_cache=True,
            mask_is_full=True,
            position_embeddings=pos_emb,
        )
        mx.eval(h_out)
        _eval_kv_caches([present])
        seq.mtp_cache = present
        seq.mtp_seqlen = present.offset
        return h_out

    def _run_trunk(
        self,
        seqs: list[Sequence],
        q_ids: list[list[int]],
        q_lens: list[int],
        *,
        output_features: bool = False,
        decode: bool = False,
    ):
        if len(seqs) == 1:
            return self._run_trunk_dense(
                seqs[0],
                q_ids[0][: q_lens[0]],
                output_features=output_features,
                decode=decode,
            )
        input_ids, caches, pos, attn_bias, end_lens = self._paged_inputs(
            seqs, q_ids, q_lens, mtp=False
        )
        extra = {}
        if not self._offset_rope_ok(seqs, q_lens, mtp=False):
            extra["position_ids"] = pos
            extra["attn_bias"] = attn_bias
        out = self.model(
            input_ids,
            past_key_values=caches,
            use_cache=True,
            mask_has_pad=False,
            output_features=output_features,
            **extra,
        )
        mx.eval(out.logits, *self.pool.keys, *self.pool.values)
        for i, seq in enumerate(seqs):
            seq.seqlen = end_lens[i]
            seq.page_seqlen = end_lens[i]
            seq.kv_caches = None
            if decode:
                self.stats["decode_tokens"] += q_lens[i]
            else:
                self.stats["prefill_tokens"] += q_lens[i]
        return out

    def _run_mtp(
        self,
        seqs: list[Sequence],
        h_in: mx.array,
        q_ids: list[list[int]],
        q_lens: list[int],
    ) -> mx.array:
        if len(seqs) == 1:
            q = q_ids[0][: q_lens[0]]
            return self._run_mtp_dense(seqs[0], h_in[0, : q_lens[0]][None], q)
        input_ids, caches, pos, attn_bias, end_lens = self._paged_inputs(
            seqs, q_ids, q_lens, mtp=True
        )
        emb = self.model.model.embed_tokens(input_ids)
        if self._offset_rope_ok(seqs, q_lens, mtp=True):
            pos_emb = self.model.model.position_embeddings(
                seqs[0].mtp_seqlen, input_ids.shape[1], emb.dtype
            )
            mask_is_full, causal_bias = True, None
        else:
            pos_emb = self.model.model.position_embeddings(
                0, input_ids.shape[1], emb.dtype, position_ids=pos
            )
            mask_is_full, causal_bias = False, attn_bias
        h_out, _ = self.model.mtp_modules[0](
            h_in,
            emb,
            past_key_value=caches[0],
            use_cache=True,
            mask_is_full=mask_is_full,
            causal_bias=causal_bias,
            position_embeddings=pos_emb,
        )
        mx.eval(h_out, *self.mtp_pool.keys, *self.mtp_pool.values)
        for i, seq in enumerate(seqs):
            seq.mtp_seqlen = end_lens[i]
            seq.mtp_page_seqlen = end_lens[i]
            seq.mtp_cache = None
        return h_out

    def _pad_h(self, rows: list[mx.array], t_max: int) -> mx.array:
        out = []
        d = rows[0].shape[-1]
        dtype = rows[0].dtype
        for h in rows:
            t = h.shape[0]
            if t == t_max:
                out.append(h)
            else:
                out.append(
                    mx.concatenate([h, mx.zeros((t_max - t, d), dtype=dtype)], axis=0)
                )
        return mx.stack(out)

    def _prefill(
        self, seqs: list[Sequence], params: SamplingParams, spec: bool
    ) -> None:
        q_ids: list[list[int]] = []
        q_lens: list[int] = []
        for seq in seqs:
            q = seq.token_ids[seq.seqlen : len(seq.prompt_ids)]
            if not q:
                raise RuntimeError("空 chunk")
            q_ids.append(q)
            q_lens.append(len(q))
        out = self._run_trunk(seqs, q_ids, q_lens, output_features=spec, decode=False)
        if spec:
            self._stash_last(seqs, out, q_lens)
            self._mtp_prefill(seqs, out, q_ids, q_lens)
            for seq in seqs:
                if seq.max_new <= 0:
                    seq.finished = True
                    seq.finish_reason = "length"
            return
        bsz = len(seqs)
        step_logits = mx.stack([out.logits[i, q_lens[i] - 1] for i in range(bsz)])
        self._sample_append(seqs, step_logits, params)

    def _stash_last(self, seqs: list[Sequence], out, q_lens: list[int]) -> None:
        feats = out.features[-1] if out.features else None
        for i, seq in enumerate(seqs):
            seq.logits_last = out.logits[i, q_lens[i] - 1]
            if feats is not None:
                seq.feat_last = feats[i, q_lens[i] - 1]
        mx.eval(*[s.logits_last for s in seqs if s.logits_last is not None])
        if feats is not None:
            mx.eval(*[s.feat_last for s in seqs if s.feat_last is not None])

    def _mtp_prefill(
        self,
        seqs: list[Sequence],
        out,
        q_ids: list[list[int]],
        q_lens: list[int],
    ) -> None:
        if self.mtp_pool is None or not out.features:
            return
        feats = out.features[-1]
        need_i = []
        h_rows = []
        tok_rows = []
        ql = []
        for i, seq in enumerate(seqs):
            L = len(seq.prompt_ids)
            if q_lens[i] <= 1 or seq.mtp_seqlen >= max(L - 1, 0):
                continue
            need_i.append(i)
            h_rows.append(feats[i, : q_lens[i] - 1])
            tok_rows.append(q_ids[i][1:])
            ql.append(q_lens[i] - 1)
        if not need_i:
            return
        t_max = max(ql)
        h_in = self._pad_h(h_rows, t_max)
        tok_pad = [t + [0] * (t_max - len(t)) for t in tok_rows]
        self._run_mtp([seqs[i] for i in need_i], h_in, tok_pad, ql)

    def _forward(
        self, seqs: list[Sequence], params: SamplingParams, decode: bool
    ) -> None:
        q_ids: list[list[int]] = []
        q_lens: list[int] = []
        for seq in seqs:
            if decode:
                q = [seq.token_ids[seq.seqlen]]
            else:
                q = seq.token_ids[seq.seqlen : len(seq.prompt_ids)]
            if not q:
                raise RuntimeError("空 chunk")
            q_ids.append(q)
            q_lens.append(len(q))
        out = self._run_trunk(seqs, q_ids, q_lens, decode=decode)
        step_logits = mx.stack([out.logits[i, q_lens[i] - 1] for i in range(len(seqs))])
        self._sample_append(seqs, step_logits, params)

    def _sample_one(
        self, logits: mx.array, seen_ids: list[int], params: SamplingParams
    ) -> int:
        transformed = _transform_logits_mx(
            logits,
            mx.array(seen_ids, dtype=mx.int32),
            params.temperature,
            params.top_p,
            params.top_k,
            params.do_sample,
            params.repetition_penalty,
        )
        return int(_sample_from_logits_mx(transformed, params.do_sample).item())

    def _spec_decode(self, seqs: list[Sequence], params: SamplingParams) -> None:
        draft_cap = params.num_speculative_tokens
        if draft_cap is None:
            draft_cap = max(1, int(self.model.config.mtp_steps))
        bonuses: list[int] = []
        iter_lens: list[int] = []
        mtp_before = [s.mtp_seqlen for s in seqs]
        old_lens = [s.seqlen for s in seqs]
        for seq in seqs:
            remaining = seq.max_new - seq.n_generated
            room = self.max_model_len - seq.seqlen
            n_draft = min(int(draft_cap), max(0, remaining - 1), max(0, room - 1))
            iter_lens.append(n_draft)
            if seq.held_bonus is not None:
                bonuses.append(seq.held_bonus)
                seq.held_bonus = None
            else:
                bonuses.append(self._sample_one(seq.logits_last, seq.token_ids, params))

        drafts: list[list[int]] = [[] for _ in seqs]
        draft_probs: list[list[mx.array]] = [[] for _ in seqs]
        h_cur = list(seq.feat_last for seq in seqs)
        tok_cur = list(bonuses)
        max_d = max(iter_lens) if iter_lens else 0
        for step in range(max_d):
            active = [i for i, n in enumerate(iter_lens) if step < n]
            if not active:
                break
            sub = [seqs[i] for i in active]
            h_in = mx.stack([h_cur[i] for i in active])[:, None, :]
            toks = [[tok_cur[i]] for i in active]
            h_out = self._run_mtp(sub, h_in, toks, [1] * len(active))
            logits = self.model._lm_logits(h_out)
            mx.eval(logits)
            for j, i in enumerate(active):
                seen = seqs[i].token_ids + [bonuses[i]] + drafts[i]
                transformed = _transform_logits_mx(
                    logits[j, 0],
                    mx.array(seen, dtype=mx.int32),
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    params.do_sample,
                    params.repetition_penalty,
                )
                probs = _probs_mx(transformed)
                tok = int(_sample_from_logits_mx(transformed, params.do_sample).item())
                drafts[i].append(tok)
                draft_probs[i].append(probs)
                h_cur[i] = h_out[j, 0]
                tok_cur[i] = tok
            self.stats["mtp_drafted"] += len(active)

        q_ids = [[bonuses[i]] + drafts[i] for i in range(len(seqs))]
        q_lens = [len(q) for q in q_ids]
        vout = self._run_trunk(seqs, q_ids, q_lens, output_features=True, decode=True)
        feats = vout.features[-1] if vout.features else None
        for i, seq in enumerate(seqs):
            self._spec_commit(
                seq,
                params,
                bonuses[i],
                drafts[i],
                draft_probs[i],
                iter_lens[i],
                vout.logits[i],
                None if feats is None else feats[i],
                old_lens[i],
                mtp_before[i],
            )

    def _spec_commit(
        self,
        seq: Sequence,
        params: SamplingParams,
        bonus: int,
        drafts: list[int],
        draft_probs: list[mx.array],
        iter_draft_len: int,
        vlogits: mx.array,
        vfeats: Optional[mx.array],
        old_len: int,
        mtp_before: int,
    ) -> None:
        n_acc = 0
        seen = list(seq.token_ids)
        for i, d in enumerate(drafts):
            seen_i = seen + [bonus] + drafts[:i]
            p_logits = _transform_logits_mx(
                vlogits[i],
                mx.array(seen_i, dtype=mx.int32),
                params.temperature,
                params.top_p,
                params.top_k,
                params.do_sample,
                params.repetition_penalty,
            )
            if params.do_sample:
                p = _probs_mx(p_logits)
                q = draft_probs[i]
                ratio = float(p[d].item()) / max(float(q[d].item()), 1e-12)
                if float(mx.random.uniform().item()) < min(1.0, ratio):
                    n_acc += 1
                else:
                    break
            else:
                if int(mx.argmax(p_logits).item()) == d:
                    n_acc += 1
                else:
                    break
        self.stats["mtp_accepted"] += n_acc
        accepted = [bonus] + drafts[:n_acc]
        new_tokens = accepted
        stop = False
        eos = params.eos_token_id
        if eos is not None and eos in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(eos) + 1]
            stop = True
        room = seq.max_new - seq.n_generated
        if len(new_tokens) > room:
            new_tokens = new_tokens[:room]
            stop = True
            seq.finish_reason = "length"
        if params.logprobs and new_tokens:
            lp0 = float(_log_softmax(seq.logits_last)[new_tokens[0]].item())
            seq.logprobs.append(lp0)
            for k, tok in enumerate(new_tokens[1:]):
                seq.logprobs.append(float(_log_softmax(vlogits[k])[tok].item()))
        seq.token_ids.extend(new_tokens)
        keep = old_len + len(new_tokens)
        if stop:
            self._rewind(seq, keep)
            self._rewind(seq, mtp_before, mtp=True)
            seq.finished = True
            if seq.finish_reason is None:
                seq.finish_reason = "stop"
            return
        self._rewind(seq, keep)
        self._rewind(seq, mtp_before, mtp=True)
        n_e = len(new_tokens)
        h_rows = [seq.feat_last[None, :]]
        if n_e > 1:
            h_rows.append(vfeats[: n_e - 1])
        feats_in = mx.concatenate(h_rows, axis=0)[None, :, :]
        self._run_mtp([seq], feats_in, [new_tokens], [n_e])
        last = n_e - 1
        seq.logits_last = vlogits[last]
        if vfeats is not None:
            seq.feat_last = vfeats[last]
        mx.eval(seq.logits_last)
        if seq.feat_last is not None:
            mx.eval(seq.feat_last)
        if (
            params.do_sample
            and drafts
            and n_acc < iter_draft_len
            and seq.n_generated < seq.max_new
        ):
            p_logits = _transform_logits_mx(
                vlogits[n_acc],
                mx.array(seen + new_tokens, dtype=mx.int32),
                params.temperature,
                params.top_p,
                params.top_k,
                True,
                params.repetition_penalty,
            )
            p = _probs_mx(p_logits)
            q = draft_probs[n_acc]
            resid = mx.clip(p - q, 0.0, None)
            if float(resid.sum().item()) > 0:
                seq.held_bonus = int(
                    mx.random.categorical(mx.log(resid + 1e-12)).item()
                )
            else:
                seq.held_bonus = int(mx.argmax(p_logits).item())
        if seq.n_generated >= seq.max_new:
            seq.finished = True
            seq.finish_reason = "length"

    def _sample_append(
        self, seqs: list[Sequence], logits: mx.array, params: SamplingParams
    ) -> None:
        rows = []
        for i, seq in enumerate(seqs):
            seen = mx.array(seq.token_ids, dtype=mx.int32)
            rows.append(
                _transform_logits_mx(
                    logits[i],
                    seen,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    params.do_sample,
                    params.repetition_penalty,
                )
            )
        transformed = mx.stack(rows)
        nxt = _sample_from_logits_mx(transformed, params.do_sample)
        mx.eval(nxt)
        tokens = [int(t) for t in nxt.tolist()]
        lp_row = None
        if params.logprobs:
            lp_row = _log_softmax(logits)
            mx.eval(lp_row)
        for i, seq in enumerate(seqs):
            if seq.seqlen < len(seq.prompt_ids):
                continue
            if seq.max_new <= 0:
                seq.finished = True
                seq.finish_reason = "length"
                continue
            tok = tokens[i]
            seq.token_ids.append(tok)
            if lp_row is not None:
                seq.logprobs.append(float(lp_row[i, tok].item()))
            if params.eos_token_id is not None and tok == params.eos_token_id:
                seq.finished = True
                seq.finish_reason = "stop"
            elif seq.n_generated >= seq.max_new:
                seq.finished = True
                seq.finish_reason = "length"

    def _maybe_stream(self, streamer, seq: Optional[Sequence], prev_n: int) -> None:
        if streamer is None or seq is None:
            return
        new = seq.n_generated - prev_n
        if new <= 0:
            return
        streamer.put([seq.token_ids[-new:]])
