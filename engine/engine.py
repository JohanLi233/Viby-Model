"""Viby 推理引擎（DeepSeek-V4.1 缩放版）：连续 batch 解码 + 三类 KV 池 + 前缀复用。

与模型层的分工（见 /tmp/dsv41_ref/API.md 与 model/attention.py）：
- `Attention.decode` 已支持逐序列的 [B] start_pos，所以真正的连续 batch 只需要
  把 batch 内各请求"当前步"的 token 拼成 [B,1] 输入、start_pos 传 [B] int32，
  直接调 `VibyModel.__call__(..., decode=True)`（不走只吃标量的 decode_step）；
- 每条请求的状态按"行"存在 StatePool 里（window 环 / compress_kv / index_k /
  压缩器 kv_state / engram_prev），行号 = 它在当前 batch 里的下标；
- prefill 逐条跑：前缀命中时先把快照 restore 到 scratch cache，再用稠密 prefill
  续跑剩余 prompt（数值上等价于整段 prefill，max|Δlogit| ≈ 1e-6）；
- 采样 / 停止条件在 engine 侧实现（旧 model 的 _transform_logits_mx 已删）。

对外接口与旧引擎保持同名同义：`VibyEngine(model, tokenizer, ...)`、
`generate` / `add_request` / `step` / `score` / `stats`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from model.cache import VibyCache

from .memory import StatePool, capture_state, copy_state_row, restore_state
from .prefix import RadixPrefixCache
from .sampling import find_stop_length, log_softmax, sample_one, transform_logits
from .types import CompletionOutput, RequestOutput, SamplingParams


@dataclass
class Sequence:
    """一条正在生成的序列（n>1 时一个请求对应多条）。"""

    request_id: str
    output_index: int
    prompt_ids: list
    params: SamplingParams
    token_ids: list = field(default_factory=list)
    max_new: int = 0
    row: int = -1  # StatePool 里的行号
    finished: bool = False
    finish_reason: Optional[str] = None
    logprobs: list = field(default_factory=list)
    streamer: object = None

    @property
    def n_generated(self) -> int:
        return max(0, len(self.token_ids) - len(self.prompt_ids))


class VibyEngine:
    """连续 batch 生成引擎；`model.generate` 是它的 batch=1 特例。"""

    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        max_num_seqs: int = 8,
        max_model_len: Optional[int] = None,
        max_prefix_states: int = 32,
        prefix_stride: int = 64,
        enable_prefix_cache: bool = True,
        seed: Optional[int] = None,
        page_size: Optional[int] = None,  # 旧参数（分页 KV 已删除），保留以便老调用方传入
        max_num_pages: Optional[int] = None,
        **kwargs,
    ):
        cfg = model.config
        model.eval()
        self.model = model
        self.config = cfg
        self.tokenizer = tokenizer
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_model_len = int(max_model_len or cfg.max_seq_len)
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.pool = StatePool(cfg, 0)
        # 前缀快照的粒度：prefill 按 prefix_stride 分块，块尾插一个可复用点
        self.prefix_stride = max(1, int(prefix_stride))
        self.prefix = (
            RadixPrefixCache(int(max_prefix_states))
            if enable_prefix_cache and int(max_prefix_states) > 0
            else None
        )
        # Engram 的最近 token 历史宽度（模型侧 prev_tokens 的长度）
        self._engram = getattr(model.model, "engram_hash", None) is not None
        self._w = max(int(cfg.engram_max_ngram_size) - 1, 0) if self._engram else 0
        # 稠密路径的 indexer 只在"本块凑满至少一个压缩组"时才打分（见 _dense_min_chunk）
        self._index_ratios = tuple(sorted({
            int(cfg.compress_ratios[i])
            for i in cfg.index_source_layers
            if i in cfg.kv_source_layers and int(cfg.compress_ratios[i]) > 1
        }))
        if seed is not None:
            mx.random.seed(int(seed))
        self.stats = self._fresh_stats()
        self._waiting: list = []
        self._running: list = []
        self._requests: dict = {}
        self._expected: dict = {}
        self._counter = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: list,
        params: Optional[SamplingParams] = None,
        streamer=None,
    ) -> list:
        """同步生成：内部 `add_request` + `step` 循环，返回与输入等长的 RequestOutput。"""
        params = params or SamplingParams()
        if params.n < 1:
            raise ValueError("n 必须 >= 1")
        if params.max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负")
        self.reset(keep_prefix=True)
        use_streamer = streamer if (len(prompts) == 1 and params.n == 1) else None
        reqs = [
            self.add_request(p, params, request_id=str(i), streamer=use_streamer)
            for i, p in enumerate(prompts)
        ]
        while self.has_unfinished():
            self.step()
        if use_streamer is not None:
            use_streamer.end()
        self._finalize(reqs)
        return reqs

    def add_request(
        self,
        prompt: Union[str, list],
        params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        streamer=None,
    ) -> RequestOutput:
        """登记一条请求（n>1 时展开成 n 条序列），返回待填充的 RequestOutput。

        streamer 只在 n==1 时生效（与旧引擎一致）。
        """
        params = params or SamplingParams()
        if params.stop and self.tokenizer is None:
            raise ValueError("stop 字符串需要 tokenizer")
        ids = self._encode(prompt)
        if not ids:
            raise ValueError("prompt 不能为空")
        if len(ids) >= self.max_model_len:
            raise ValueError(
                f"prompt 长度 {len(ids)} 超过 max_model_len {self.max_model_len}"
            )
        if request_id is None:
            request_id = f"req-{self._counter}"
            self._counter += 1
        out = RequestOutput(request_id=request_id, prompt_token_ids=list(ids))
        self._requests[request_id] = out
        max_new = max(0, min(int(params.max_new_tokens), self.max_model_len - len(ids)))
        n = max(1, int(params.n))
        self._expected[request_id] = n
        for k in range(n):
            seq = Sequence(
                request_id=request_id,
                output_index=k,
                prompt_ids=list(ids),
                params=params,
                token_ids=list(ids),
                max_new=max_new,
                streamer=streamer if (streamer is not None and n == 1) else None,
            )
            self._waiting.append(seq)
        return out

    def step(self) -> list:
        """跑一轮调度：先尽量补满 batch（prefill 新请求），再对 running 做一步解码。"""
        touched: dict = {}

        def mark(seq):
            touched.setdefault(seq.request_id, self._requests[seq.request_id])

        while self._waiting and len(self._running) < self.max_num_seqs:
            seq = self._waiting.pop(0)
            self._admit(seq)
            mark(seq)

        if self._running:
            logits = self._decode(self._running)
            self._append_step(self._running, logits)
            for seq in self._running:
                mark(seq)
            self._compact()

        for req in touched.values():
            if self._all_done(req):
                req.finished = True
        return list(touched.values())

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)

    def score(self, sequences: list, prompt_lens: list) -> mx.array:
        """Teacher-forced 完成段 logprob，形状 (B, max_gen_len)，短序列 pad 0。"""
        if len(sequences) != len(prompt_lens):
            raise ValueError("sequences 与 prompt_lens 长度不一致")
        if not sequences:
            return mx.zeros((0, 0))
        gen = max(len(s) - int(pl) for s, pl in zip(sequences, prompt_lens))
        if gen <= 0:
            return mx.zeros((len(sequences), 0))
        rows = []
        for seq, pl in zip(sequences, prompt_lens):
            pl = int(pl)
            logits, _ = self.model.prefill(mx.array([list(seq)], dtype=mx.int32))
            lp = log_softmax(logits)[0]
            n = len(seq) - pl
            vals = [float(lp[pl - 1 + j, int(seq[pl + j])].item()) for j in range(n)]
            rows.append(vals + [0.0] * (gen - n))
        return mx.array(np.array(rows, dtype=np.float32))

    def reset(self, keep_prefix: bool = False) -> None:
        """清空调度状态与统计；keep_prefix=False 时同时清掉前缀缓存。"""
        self._waiting, self._running = [], []
        self._requests, self._expected = {}, {}
        self.pool.ensure(0)
        if not keep_prefix and self.prefix is not None:
            self.prefix.reset()
        self.stats = self._fresh_stats()

    def memory_stats(self) -> dict:
        """池子 / 前缀缓存常驻字节数 + MLX 显存计数。"""
        out = {
            "pool_bytes": self.pool.memory_bytes(),
            "prefix_bytes": self.prefix.memory_bytes() if self.prefix is not None else 0,
            "active_memory": int(mx.get_active_memory()),
            "peak_memory": int(mx.get_peak_memory()),
        }
        return out

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------
    def _admit(self, seq: Sequence) -> None:
        """prefill 一条新序列（含前缀复用），把状态搬进池子并采样第一个 token。"""
        prompt = seq.prompt_ids
        if seq.max_new <= 0:
            self._finish(seq, "length")
            return
        # 1) 前缀匹配：拿最长可复用的状态快照
        n_hit, state = (0, None)
        if self.prefix is not None:
            n_hit, state = self.prefix.match(prompt)
            if state is not None and state.hidden is None and n_hit >= len(prompt):
                # 快照缺末位 hidden 时无法产出首个 logits，退回整段 prefill
                n_hit, state = 0, None
        scratch = VibyCache(self.config, 1)
        start = restore_state(scratch, 0, state) if state is not None else 0
        end = len(prompt)
        hidden = None
        if start >= end:
            # 整段命中：用快照里的末位 hidden 重算 logits（省一次前向）
            hidden = state.hidden[None]
        else:
            # 稠密续跑剩余 prompt；开前缀缓存时按 stride 分块，块尾插复用点
            stride = self.prefix_stride if self.prefix is not None else end
            pos = start
            while pos < end:
                nxt = min(end, (pos // stride + 1) * stride)
                if nxt <= pos:
                    nxt = end
                need = self._dense_min_chunk(pos)
                if nxt - pos < need and nxt < end:
                    nxt = min(end, pos + need)  # 往后扩到能凑满一个压缩组
                if nxt - pos < need:
                    # 末尾这段凑不满一个压缩组：逐 token 走 decode 路径
                    # （decode 的 indexer 直接读池子，不受"本块是否成组"限制）
                    hidden = None
                    for p in range(pos, end):
                        one = mx.array([[prompt[p]]], dtype=mx.int32)
                        hidden = self._trunk(one, start_pos=p, cache=scratch, decode=True)[:, -1]
                        self._update_engram_decode(scratch, one)
                        mx.eval(hidden)
                    pos = end
                else:
                    ids = mx.array([prompt[pos:nxt]], dtype=mx.int32)
                    hidden = self._trunk(ids, start_pos=pos, cache=scratch, decode=False)[:, -1]
                    self._update_engram_prefill(scratch, ids)
                    pos = nxt
                    mx.eval(hidden)
                # 快照必须在后续前向改动 scratch 之前取（capture 用值语义拷贝）；
                # 缓存满时只保留"请求末尾"这个复用点，避免长 prompt 一路搬快照
                if self.prefix is not None and (self.prefix.has_room() or pos >= end):
                    self.prefix.insert(
                        prompt[:pos],
                        capture_state(scratch, 0, pos, hidden=mx.array(hidden[0])),
                    )
        logits = self.model.logits(hidden)[0]
        mx.eval(logits)
        # 2) 状态入池（行号 = 当前 batch 的位置）
        row = len(self._running)
        self.pool.ensure(row + 1, [len(s.token_ids) for s in self._running])
        copy_state_row(self.pool.cache, row, scratch, 0, end)
        seq.row = row
        # 3) 统计 + 采样首 token
        self.stats["prefill_tokens"] += len(prompt) - start
        if n_hit:
            self.stats["prefix_hits"] += 1
            self.stats["prefix_tokens"] += n_hit
        self._running.append(seq)
        if seq.streamer is not None:
            seq.streamer.put([list(prompt)])
        self._append_step([seq], logits[None])

    def _dense_min_chunk(self, pos: int) -> int:
        """续跑时稠密块至少要覆盖的 token 数。

        `Attention.__call__` 的稠密路径只有当本块产出了新的压缩组（ratio>1 的
        indexer 源层 latent 非空）时才会跑 indexer 并把 keep 掩码传给后续层；
        块太短时它会退化成"看全部可达压缩位置"，与整段 prefill 不等价。
        所以 engine 侧保证每个续跑块都凑满至少一组，凑不满的尾巴改走 decode。
        """
        if pos <= 0:
            return 1
        need = 1
        for r in self._index_ratios:
            filled = pos % r
            need = max(need, r - filled if filled else r)
        return need

    def _decode(self, seqs: list) -> mx.array:
        """连续 batch 解码一步：各请求位置不同，start_pos 传 [B] int32。"""
        toks = mx.array([[s.token_ids[-1]] for s in seqs], dtype=mx.int32)
        # start_pos = 已处理的 token 数：token_ids 末位尚未过模型，位置是 len-1
        lens = [len(s.token_ids) for s in seqs]
        pos = mx.array([n - 1 for n in lens], dtype=mx.int32)
        cache = self.pool.cache
        cache.decode_max_pos = max(lens)
        hidden = self._trunk(toks, start_pos=pos, cache=cache, decode=True)
        logits = self.model.logits(hidden)[:, 0]
        mx.eval(logits)
        self._update_engram_decode(cache, toks)
        self.stats["decode_tokens"] += len(seqs)
        self.stats["batch_steps"] += 1
        self.stats["max_batch"] = max(self.stats["max_batch"], len(seqs))
        return logits

    def _append_step(self, seqs: list, logits: mx.array) -> None:
        """逐条按各自的 SamplingParams 采样、追加 token、判定停止 + 流式输出。"""
        for i, seq in enumerate(seqs):
            if seq.finished:
                continue
            params = seq.params
            row = logits[i]
            tok = sample_one(transform_logits(row, seq.token_ids, params), params)
            if params.logprobs:
                seq.logprobs.append(float(log_softmax(row)[tok].item()))
            seq.token_ids.append(tok)
            if seq.streamer is not None:
                seq.streamer.put([[tok]])
            self.stats["generated_tokens"] += 1
            self._check_finish(seq)

    def _compact(self) -> None:
        """把结束的序列移出 running，并把状态行压实到 0..B-1。"""
        keep = []
        for seq in self._running:
            if seq.finished:
                continue
            dst = len(keep)
            if seq.row != dst:
                self.pool.move(dst, seq.row)
                seq.row = dst
            keep.append(seq)
        changed = len(keep) != len(self._running)
        self._running = keep
        if changed:
            self.pool.ensure(len(keep), [len(s.token_ids) for s in keep])

    # ------------------------------------------------------------------
    # 前向 / Engram 记账
    # ------------------------------------------------------------------
    def _trunk(self, ids: mx.array, start_pos, cache: VibyCache, decode: bool) -> mx.array:
        """主干前向，返回末层 hidden [B,T,D]（等价于 VibyForCausalLM 的 hidden）。

        直接走 `VibyModel.__call__`：它同时支持标量 / [B] 的 start_pos，并且会
        把 per-row 的位置一路传到 `Attention.decode`。
        """
        hidden, _, _ = self.model.model(
            ids,
            start_pos=start_pos,
            cache=cache,
            decode=decode,
            prev_tokens=cache.engram_prev,
            collect_main=False,
        )
        return hidden

    def _update_engram_prefill(self, cache: VibyCache, ids: mx.array) -> None:
        """稠密 prefill 后的 engram 历史推进（与 model.prefill 的取值口径一致）。"""
        if self._w <= 0:
            return
        prev = cache.engram_prev
        if prev is None or prev.shape[1] != self._w:
            # 序列开头：用 DEAD(-1) 左补齐到固定宽度（engram 里负值 = 无历史）
            prev = mx.full((ids.shape[0], self._w), -1, dtype=mx.int32)
        cache.engram_prev = mx.concatenate([prev, ids], axis=1)[:, -self._w:]

    def _update_engram_decode(self, cache: VibyCache, toks: mx.array) -> None:
        """解码一步后的历史推进（逐字对应 model.decode_step 的写法）。"""
        if self._w <= 0:
            return
        prev = cache.engram_prev
        if prev is None:
            cache.engram_prev = mx.broadcast_to(toks, (toks.shape[0], self._w))
        else:
            cache.engram_prev = mx.concatenate([prev, toks], axis=1)[:, -self._w:]

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _check_finish(self, seq: Sequence) -> None:
        """每次只追加一个 token，所以 stop id 只看末位（O(1)）；stop 字符串要看全文。"""
        params = seq.params
        base = len(seq.prompt_ids)
        gen = seq.token_ids[base:]
        stops = set()
        if params.eos_token_id is not None:
            stops.add(int(params.eos_token_id))
        if params.stop_token_ids:
            stops.update(int(t) for t in params.stop_token_ids)
        if stops and seq.token_ids[-1] in stops:
            self._finish(seq, "stop")
            return
        if params.stop and self.tokenizer is not None:
            keep = find_stop_length(self.tokenizer, gen, params.stop)
            if keep is not None:
                del seq.token_ids[base + keep :]
                self._finish(seq, "stop")
                return
        if len(gen) >= seq.max_new:
            self._finish(seq, "length")

    def _finish(self, seq: Sequence, reason: str) -> None:
        if seq.finished:
            return
        seq.finished = True
        seq.finish_reason = reason
        gen = seq.token_ids[len(seq.prompt_ids) :]
        text = ""
        if self.tokenizer is not None and gen:
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
        self._requests[seq.request_id].outputs.append(
            CompletionOutput(
                index=seq.output_index,
                token_ids=list(gen),
                text=text,
                logprobs=list(seq.logprobs) if seq.params.logprobs else None,
                finish_reason=reason,
            )
        )

    def _all_done(self, req: RequestOutput) -> bool:
        return len(req.outputs) >= self._expected.get(req.request_id, 1)

    def _finalize(self, reqs: list) -> None:
        for req in reqs:
            req.outputs.sort(key=lambda o: o.index)
            req.finished = True

    def _encode(self, prompt: Union[str, list]) -> list:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("字符串 prompt 需要 tokenizer")
            return list(
                self.tokenizer(prompt, truncation=True, max_length=self.max_model_len).input_ids
            )
        return [int(t) for t in prompt]

    @staticmethod
    def _fresh_stats() -> dict:
        return {
            "prefix_tokens": 0,  # 被前缀缓存省掉的 prefill token 数（兼容旧字段名）
            "prefix_hits": 0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "generated_tokens": 0,
            "batch_steps": 0,
            "max_batch": 0,
            "mtp_accepted": 0,  # DSpark 投机解码 TODO，占位保持旧字段
            "mtp_drafted": 0,
        }

    @property
    def prefix_tokens(self) -> int:
        return self.stats["prefix_tokens"]
