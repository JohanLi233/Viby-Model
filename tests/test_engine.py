"""VibyEngine：MLA + n-gram 的分页 KV / 连续 batch / 前缀复用。"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import mlx.core as mx
import numpy as np

from model.config import VibyConfig
from model.model import VibyForCausalLM


def _cfg(**kw):
    base = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        mtp_depth=0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_latent_dim=0,
        ngram_table_size=256,
        ngram_layer=2,
        ngram_orders=(2, 3),
        use_linear_attn=False,
        dropout=0.0,
    )
    base.update(kw)
    return VibyConfig(**base)


def _tiny_model(**kw):
    mx.random.seed(0)
    model = VibyForCausalLM(_cfg(**kw))
    model.eval()
    return model


def test_page_pool_alloc_free():
    from engine.memory import PagePool

    pool = PagePool(
        num_layers=2,
        n_heads=4,
        page_size=4,
        k_dim=24,
        v_dim=16,
        num_pages=8,
        dtype=mx.float32,
    )
    assert pool.num_free == 8
    a = pool.alloc(3)
    b = pool.alloc(2)
    assert len(a) == 3 and len(b) == 2
    assert 0 not in a and 0 not in b
    assert pool.num_free == 3
    pool.free_pages(a)
    assert pool.num_free == 6
    print("page pool alloc/free: OK")


def test_cow_overwrites_dirty_page():
    """回收页上仍有旧 KV 时，COW 必须是拷贝而不是叠加上去。"""
    from engine.memory import PagePool

    pool = PagePool(
        num_layers=1,
        n_heads=2,
        page_size=2,
        k_dim=4,
        v_dim=4,
        num_pages=3,
        dtype=mx.float32,
    )
    src_id = pool.alloc(1)[0]
    pool.retain([src_id])
    ones_k = mx.ones((pool.n_heads, pool.page_size, pool.k_dim), dtype=mx.float32)
    ones_v = mx.ones((pool.n_heads, pool.page_size, pool.v_dim), dtype=mx.float32)
    pool.keys[0] = pool.keys[0].at[src_id].add(ones_k)
    pool.values[0] = pool.values[0].at[src_id].add(ones_v)
    mx.eval(pool.keys[0], pool.values[0])

    dirty = pool.alloc(2)
    five_k = mx.full((pool.n_heads, pool.page_size, pool.k_dim), 5.0)
    five_v = mx.full((pool.n_heads, pool.page_size, pool.v_dim), 5.0)
    for pid in dirty:
        pool.keys[0] = pool.keys[0].at[pid].add(five_k)
        pool.values[0] = pool.values[0].at[pid].add(five_v)
    mx.eval(pool.keys[0], pool.values[0])
    pool.free_pages(dirty)

    new_id = pool.cow(src_id)
    mx.eval(pool.keys[0], pool.values[0])
    got_k = np.array(pool.keys[0][new_id])
    got_v = np.array(pool.values[0][new_id])
    np.testing.assert_allclose(got_k, np.ones_like(got_k), atol=1e-5)
    np.testing.assert_allclose(got_v, np.ones_like(got_v), atol=1e-5)
    print("cow overwrites dirty page: OK")


def test_prefix_reuse_after_recycle_matches():
    """上一次 generate 释放的脏页被 COW 复用时，greedy 仍应与首次一致。"""
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompt = [3, 4, 5, 6, 7, 8, 9]
    params = SamplingParams(
        max_new_tokens=8,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    engine = VibyEngine(model, page_size=4, max_num_seqs=2, max_num_pages=6)
    mx.random.seed(11)
    first = engine.generate([prompt], params)[0].outputs[0].token_ids
    pool = engine.pool
    n_free = pool.num_free
    if n_free:
        poison_ids = pool.alloc(n_free)
        pk = mx.full((pool.n_heads, pool.page_size, pool.k_dim), 7.0)
        pv = mx.full((pool.n_heads, pool.page_size, pool.v_dim), 7.0)
        for pid in poison_ids:
            for layer in range(pool.num_layers):
                pool.keys[layer] = pool.keys[layer].at[pid].add(pk)
                pool.values[layer] = pool.values[layer].at[pid].add(pv)
        mx.eval(*pool.keys, *pool.values)
        pool.free_pages(poison_ids)
    mx.random.seed(11)
    second = engine.generate([prompt], params)[0].outputs[0].token_ids
    assert first == second, f"recycle greedy {first} vs {second}"
    print("prefix reuse after recycle: OK")


class _RecStreamer:
    def __init__(self):
        self.chunks = []
        self.ended = False

    def put(self, token_ids):
        arr = np.array(token_ids)
        self.chunks.append(arr.tolist())

    def end(self):
        self.ended = True


def test_generate_streams_tokens():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompt = [3, 7, 11, 19]
    streamer = _RecStreamer()
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    params = SamplingParams(
        max_new_tokens=6,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    out = engine.generate([prompt], params, streamer=streamer)
    gen = out[0].outputs[0].token_ids
    assert streamer.ended
    assert streamer.chunks, "streamer.put 从未被调用"
    assert streamer.chunks[0] == [prompt] or streamer.chunks[0] == prompt
    streamed = []
    for ch in streamer.chunks[1:]:
        row = ch[0] if ch and isinstance(ch[0], list) else ch
        streamed.extend(row)
    assert streamed == gen, f"streamed {streamed} vs gen {gen}"
    assert len(streamer.chunks) > 2, "应逐步 put，而不是结束时一次性倒出"
    print(f"generate streams tokens: {len(streamer.chunks)} puts OK")


def test_mtp_spec_streams_tokens():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model(mtp_depth=1, mtp_steps=2)
    prompt = [3, 7, 11, 19, 23]
    streamer = _RecStreamer()
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    out = engine.generate(
        [prompt],
        SamplingParams(
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            use_mtp_speculative=True,
            num_speculative_tokens=2,
        ),
        streamer=streamer,
    )
    gen = out[0].outputs[0].token_ids
    assert streamer.ended and streamer.chunks
    streamed = []
    for ch in streamer.chunks[1:]:
        row = ch[0] if ch and isinstance(ch[0], list) else ch
        streamed.extend(row)
    assert streamed == gen, f"MTP streamed {streamed} vs gen {gen}"
    print("MTP spec streams tokens: OK")


def test_paged_prefill_decode_matches_dense():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompt = [3, 7, 11, 19, 23]
    mx.random.seed(1)
    dense = model.generate(
        mx.array([prompt]),
        max_new_tokens=8,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    mx.eval(dense)
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    mx.random.seed(1)
    out = engine.generate(
        [prompt],
        SamplingParams(
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
        ),
    )
    got = np.array(out[0].outputs[0].token_ids)
    exp = np.array(dense)[0, len(prompt) :]
    assert got.tolist() == exp.tolist(), f"paged {got.tolist()} vs dense {exp.tolist()}"
    print("paged greedy matches generate: OK")


def test_batch_two_prompts_match_sequential():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompts = [[3, 5, 7, 9], [11, 13, 17]]
    engine = VibyEngine(model, page_size=4, max_num_seqs=4)
    params = SamplingParams(
        max_new_tokens=6,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    mx.random.seed(2)
    batched = engine.generate(prompts, params)
    sequential = []
    for p in prompts:
        mx.random.seed(2)
        one = engine.generate([p], params)
        sequential.append(one[0].outputs[0].token_ids)
    for i, p in enumerate(prompts):
        assert batched[i].outputs[0].token_ids == sequential[i], (
            f"prompt {i} batch {batched[i].outputs[0].token_ids} vs seq {sequential[i]}"
        )
    print("batched vs sequential: OK")


def test_prefix_reuse_same_prompt():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompt = [3, 4, 5, 6, 7, 8, 9, 10]
    engine = VibyEngine(model, page_size=4, max_num_seqs=4)
    params = SamplingParams(
        max_new_tokens=4,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        n=2,
    )
    mx.random.seed(3)
    out = engine.generate([prompt], params)
    assert len(out[0].outputs) == 2
    a, b = out[0].outputs[0].token_ids, out[0].outputs[1].token_ids
    assert a == b, f"greedy n=2 应相同: {a} vs {b}"
    assert engine.stats["prefix_tokens"] >= len(prompt), engine.stats
    print(f"prefix reuse n=2: hit {engine.stats['prefix_tokens']} tokens OK")


def test_logprobs_and_score():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model()
    prompt = [3, 8, 12]
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    params = SamplingParams(
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=None,
        logprobs=True,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    out = engine.generate([prompt], params)
    lp = out[0].outputs[0].logprobs
    ids = out[0].outputs[0].token_ids
    assert lp is not None and len(lp) == len(ids)
    assert all(x <= 0.0 for x in lp), lp
    tokens = prompt + ids
    scored = engine.score([tokens], prompt_lens=[len(prompt)])
    arr = np.array(scored)
    assert arr.shape == (1, len(ids))
    np.testing.assert_allclose(arr[0], np.array(lp), atol=3e-2)
    print("logprobs vs score: OK")


def test_linear_attn_rejected():
    from engine import VibyEngine

    model = _tiny_model(use_linear_attn=True)
    try:
        VibyEngine(model)
    except ValueError as e:
        assert "MLA" in str(e) or "linear" in str(e).lower()
        print("linear attn rejected: OK")
        return
    raise AssertionError("use_linear_attn 模型应被拒绝")


def test_mtp_spec_greedy_matches_generate():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model(mtp_depth=1, mtp_steps=2)
    prompt = [3, 7, 11, 19, 23, 29]
    params = dict(
        max_new_tokens=8,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    mx.random.seed(4)
    dense = model.generate(
        mx.array([prompt]),
        use_mtp_speculative=True,
        num_speculative_tokens=2,
        **params,
    )
    mx.eval(dense)
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    mx.random.seed(4)
    out = engine.generate(
        [prompt],
        SamplingParams(
            use_mtp_speculative=True,
            num_speculative_tokens=2,
            **params,
        ),
    )
    got = out[0].outputs[0].token_ids
    exp = np.array(dense)[0, len(prompt) :].tolist()
    assert got == exp, f"engine MTP {got} vs generate spec {exp}"
    assert engine.stats["mtp_drafted"] > 0, engine.stats
    print(
        f"engine MTP greedy matches generate: drafted={engine.stats['mtp_drafted']} "
        f"accepted={engine.stats['mtp_accepted']} OK"
    )


def test_mtp_spec_greedy_matches_plain():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model(mtp_depth=1, mtp_steps=2)
    prompt = [4, 8, 15, 16, 23]
    engine = VibyEngine(model, page_size=4, max_num_seqs=2)
    params = SamplingParams(
        max_new_tokens=6,
        do_sample=False,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    mx.random.seed(5)
    plain = engine.generate([prompt], params)
    mx.random.seed(5)
    spec = engine.generate(
        [prompt],
        SamplingParams(
            use_mtp_speculative=True,
            num_speculative_tokens=2,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
        ),
    )
    a, b = plain[0].outputs[0].token_ids, spec[0].outputs[0].token_ids
    assert a == b, f"plain {a} vs MTP spec {b}"
    print("engine MTP greedy matches plain decode: OK")


def test_mtp_spec_prefix_n2():
    from engine import SamplingParams, VibyEngine

    model = _tiny_model(mtp_depth=1)
    prompt = [3, 4, 5, 6, 7, 8, 9, 10]
    engine = VibyEngine(model, page_size=4, max_num_seqs=4)
    mx.random.seed(6)
    out = engine.generate(
        [prompt],
        SamplingParams(
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            n=2,
            use_mtp_speculative=True,
            num_speculative_tokens=2,
        ),
    )
    a, b = out[0].outputs[0].token_ids, out[0].outputs[1].token_ids
    assert a == b, f"greedy n=2 MTP {a} vs {b}"
    assert engine.stats["prefix_tokens"] >= len(prompt) - 1, engine.stats
    print(f"engine MTP prefix n=2: hit {engine.stats['prefix_tokens']} OK")


if __name__ == "__main__":
    test_page_pool_alloc_free()
    test_cow_overwrites_dirty_page()
    test_prefix_reuse_after_recycle_matches()
    test_generate_streams_tokens()
    test_mtp_spec_streams_tokens()
    test_paged_prefill_decode_matches_dense()
    test_batch_two_prompts_match_sequential()
    test_prefix_reuse_same_prompt()
    test_logprobs_and_score()
    test_linear_attn_rejected()
    test_mtp_spec_greedy_matches_generate()
    test_mtp_spec_greedy_matches_plain()
    test_mtp_spec_prefix_n2()
    print("all engine tests passed")
