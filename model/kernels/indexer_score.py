"""Lightning Indexer 打分：操作数一律 bf16/fp16 MMA。

CED 解码段 ratio=1 ⇒ N=T。公式仍是

    score[n] = Σ_h ReLU(q_h·k_n)·w_h    （reach=false 写 -inf）

Q、K、Gate、w 都走 ``T``（bf16/fp16）8×8 ``simdgroup_matrix``；累加器
仍是 fp32（和稀疏注意力同一套 tensor-core 口径）。只有写出的 [B,T,N]
打分留 fp32——``_topk_masks`` 的 1e-7 tiebreak 在 bf16 会下溢。

反向：同一套 MMA 重算点积；Gate 降成 T 之后 ``dQ = Gate @ K``、
``dK = Gateᵀ @ Q`` 也走 MMA；dK 按 head 归约后再 atomic。

decode（T=1、B 小、N 大）把 key 维切到多个 threadgroup。

开关：`VIBY_INDEXER_KERNEL=0` 回退 einsum（同样是 bf16 操作数）。
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_INDEXER_KERNEL", "1") != "0"
_SHARDS = 8
_BK = 16
NEG_INF = -1e30
_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"

_FWD = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint query = thread_position_in_grid.y;
    uint tile = thread_position_in_grid.z;
    uint Tlen = dims[0], N = dims[1], ntiles = dims[3];
    uint b = query / Tlen;
    constexpr uint NP = ID / 32;
    constexpr uint NT = 32 * NP;
    constexpr uint DD = 32;
    constexpr uint H8 = 8;
    constexpr uint BK = 16;
    constexpr uint SS = BK + 4;
    constexpr uint SP = ID + 8;
    uint sg = tid / 32;
    uint col = sg * DD;

    threadgroup T Qs[H8 * SP];
    threadgroup T Ks[BK * SP];
    threadgroup float scores[NP * H8 * SS];
    threadgroup T Ws[H8];

    for (uint i = tid; i < H8 * ID; i += NT) {
        uint h = i / ID, d = i % ID;
        Qs[h * SP + d] = (h < IH) ? q[(size_t)query * IH * ID + h * ID + d] : T(0);
    }
    if (tid < IH) Ws[tid] = w[(size_t)query * IH + tid];
    if (tid >= IH && tid < H8) Ws[tid] = T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<T, 8, 8> Qf[DD / 8];
    for (uint d = 0; d < DD / 8; ++d)
        simdgroup_load(Qf[d], Qs + col + d * 8, SP);

    const device T* kb = k + (size_t)b * N * ID;
    const device bool* rb = reach + (size_t)query * N;
    device float* ob = out + (size_t)query * N;

    for (uint base = tile * BK; base < N; base += ntiles * BK) {
        // 整块都不可达：省掉 K 读取与 MMA，只协作写一遍哨兵。判定用的是与
        // 逐元素完全相同的谓词 `reach[q,n]`，有效集合不变。
        // Scheduling splits do not change the physical per-key-tile stride.
        bool tile_ok = tile_valid[(size_t)query * ((N+BK-1)/BK) + base/BK];
        if (!tile_ok) {
            for (uint i = tid; i < BK; i += NT)
                if (base + i < N) ob[base + i] = -1e30f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        for (uint i = tid; i < BK * ID; i += NT) {
            uint row = i / ID, d = i % ID;
            uint n = base + row;
            T val = T(0);
            if (n < N && rb[n]) val = kb[(size_t)n * ID + d];
            Ks[row * SP + d] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> Sf[BK / 8];
        for (uint j = 0; j < BK / 8; ++j)
            Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < DD / 8; ++d)
            for (uint j = 0; j < BK / 8; ++j) {
                simdgroup_matrix<T, 8, 8> Kf;
                simdgroup_load(Kf, Ks + j * 8 * SP + col + d * 8, SP, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sf[j], Qf[d], Kf, Sf[j]);
            }
        for (uint j = 0; j < BK / 8; ++j)
            simdgroup_store(Sf[j], scores + (sg * H8) * SS + j * 8, SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < BK) {
            uint n = base + tid;
            bool ok = n < N && rb[n];
            float acc = 0.f;
            if (ok) {
                for (uint h = 0; h < IH; ++h) {
                    float dot = scores[h * SS + tid];
                    for (uint p = 1; p < NP; ++p)
                        dot += scores[(p * H8 + h) * SS + tid];
                    acc += max(dot, 0.f) * float(Ws[h]);
                }
            }
            if (n < N) ob[n] = ok ? acc : -1e30f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_BWD = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint query = thread_position_in_grid.y;
    uint tile = thread_position_in_grid.z;
    uint Tlen = dims[0], N = dims[1], B = dims[2], ntiles = dims[3];
    uint b = query / Tlen;
    constexpr uint NP = ID / 32;
    constexpr uint NT = 32 * NP;
    constexpr uint DD = 32;
    constexpr uint H8 = 8;
    constexpr uint BK = 16;
    constexpr uint SS = BK + 4;
    constexpr uint SP = ID + 8;
    uint sg = tid / 32;
    uint col = sg * DD;
    uint shard = query % SHARDS;

    threadgroup T Qs[H8 * SP];
    threadgroup T Ks[BK * SP];
    threadgroup T Gs[H8 * (BK + 8)];
    threadgroup float scores[NP * H8 * SS];
    threadgroup float tileK[BK * SP];
    threadgroup T Ws[H8];
    threadgroup float dWacc[H8];

    for (uint i = tid; i < H8 * ID; i += NT) {
        uint h = i / ID, d = i % ID;
        Qs[h * SP + d] = (h < IH) ? q[(size_t)query * IH * ID + h * ID + d] : T(0);
    }
    if (tid < IH) {
        Ws[tid] = w[(size_t)query * IH + tid];
        dWacc[tid] = 0.f;
    }
    if (tid >= IH && tid < H8) {
        Ws[tid] = T(0);
        dWacc[tid] = 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<T, 8, 8> Qf[DD / 8];
    simdgroup_matrix<float, 8, 8> DQ[DD / 8];
    for (uint d = 0; d < DD / 8; ++d) {
        simdgroup_load(Qf[d], Qs + col + d * 8, SP);
        DQ[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const device T* kb = k + (size_t)b * N * ID;
    const device T* gb = g + (size_t)query * N;
    const device bool* rb = reach + (size_t)query * N;

    for (uint base = tile * BK; base < N; base += ntiles * BK) {
        // 整块不可达：反向同样跳过 K 读取 / MMA / dK atomic。
        if (!tile_valid[(size_t)query * ((N+BK-1)/BK) + base/BK]) continue;
        for (uint i = tid; i < BK * ID; i += NT) {
            uint row = i / ID, d = i % ID;
            uint n = base + row;
            T val = T(0);
            if (n < N && rb[n]) val = kb[(size_t)n * ID + d];
            Ks[row * SP + d] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> Sf[BK / 8];
        for (uint j = 0; j < BK / 8; ++j)
            Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < DD / 8; ++d)
            for (uint j = 0; j < BK / 8; ++j) {
                simdgroup_matrix<T, 8, 8> Kf;
                simdgroup_load(Kf, Ks + j * 8 * SP + col + d * 8, SP, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sf[j], Qf[d], Kf, Sf[j]);
            }
        for (uint j = 0; j < BK / 8; ++j)
            simdgroup_store(Sf[j], scores + (sg * H8) * SS + j * 8, SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < H8) {
            float s = 0.f;
            for (uint j = 0; j < BK; ++j) {
                uint n = base + j;
                bool ok = n < N && rb[n];
                float gn = ok ? float(gb[n]) : 0.f;
                float dot = scores[tid * SS + j];
                for (uint p = 1; p < NP; ++p)
                    dot += scores[(p * H8 + tid) * SS + j];
                float gate = 0.f;
                if (tid < IH && ok) {
                    s += max(dot, 0.f) * gn;
                    if (dot > 0.f) gate = float(Ws[tid]) * gn;
                }
                Gs[tid * (BK + 8) + j] = T(gate);
            }
            if (tid < IH) dWacc[tid] += s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = 0; d < DD / 8; ++d)
            for (uint j = 0; j < BK / 8; ++j) {
                simdgroup_matrix<T, 8, 8> Gmat, Kf;
                simdgroup_load(Gmat, Gs + j * 8, BK + 8);
                simdgroup_load(Kf, Ks + j * 8 * SP + col + d * 8, SP);
                simdgroup_multiply_accumulate(DQ[d], Gmat, Kf, DQ[d]);
            }
        for (uint j = 0; j < BK / 8; ++j) {
            for (uint d = 0; d < DD / 8; ++d) {
                simdgroup_matrix<float, 8, 8> DK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_matrix<T, 8, 8> Gmat;
                simdgroup_load(Gmat, Gs + j * 8, BK + 8, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(DK, Gmat, Qf[d], DK);
                simdgroup_store(DK, tileK + j * 8 * SP + col + d * 8, SP);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK * ID; i += NT) {
            uint row = i / ID, d = i % ID;
            uint n = base + row;
            if (n >= N || !rb[n]) continue;
            atomic_fetch_add_explicit(
                dk + ((((size_t)shard * B + b) * N + n) * ID + d),
                tileK[row * SP + d], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0; d < DD / 8; ++d)
        simdgroup_store(DQ[d], tileK + col + d * 8, SP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < IH * ID; i += NT) {
        uint h = i / ID, d = i % ID;
        atomic_fetch_add_explicit(
            dq + ((size_t)query * IH + h) * ID + d,
            tileK[h * SP + d], memory_order_relaxed);
    }
    if (tid < IH)
        atomic_fetch_add_explicit(dw + (size_t)query * IH + tid,
                                 dWacc[tid], memory_order_relaxed);
"""


@lru_cache(None)
def _kernels():
    fwd = mx.fast.metal_kernel(
        name="indexer_score_fwd_mma",
        input_names=["q", "k", "w", "reach", "tile_valid", "dims"],
        output_names=["out"],
        source=_FWD,
        header=_HEADER,
    )
    bwd = mx.fast.metal_kernel(
        name="indexer_score_bwd_mma",
        input_names=["q", "k", "w", "reach", "tile_valid", "g", "dims"],
        output_names=["dq", "dk", "dw"],
        source=_BWD,
        header=_HEADER,
        atomic_outputs=True,
    )
    return fwd, bwd


def eager_indexer_score(q, k, w, reach):
    """参考实现：Q/K/w 保持激活 dtype（bf16/fp16），只在 8 头归约时升 fp32。"""
    w = w.astype(q.dtype)
    sc = mx.einsum("bthd,bnd->bthn", q, k)
    sc = mx.maximum(sc, 0.0)
    sc = mx.einsum("bthn,bth->btn", sc.astype(mx.float32), w.astype(mx.float32))
    return mx.where(reach, sc, mx.array(NEG_INF, dtype=mx.float32))


def _supported(q, k):
    return (
        _ENABLED
        and mx.default_device() == mx.gpu
        and q.dtype in (mx.bfloat16, mx.float16)
        and k.dtype == q.dtype
        and q.ndim == 4
        and k.ndim == 3
        and q.shape[-1] == k.shape[-1]
        and q.shape[-2] in (4, 8)
        and q.shape[-1] in (32, 64)
    )


def _n_tiles(batch, seq, keys):
    """训练 B·T 足够大时每 query 一个 TG；decode 把 key 维切开填满占用。"""
    queries = batch * seq
    if keys == 0 or queries >= 256:
        return 1
    return min((keys + _BK - 1) // _BK, max(1, (256 + queries - 1) // max(queries, 1)))


def _dims(batch, seq, keys, tiles):
    return mx.array([seq, keys, batch, tiles], mx.uint32)


def _tile_valid(reach, tiles, block):
    """One flag per actual BK key tile, independent of dispatch split count."""
    b, t, n = reach.shape
    nt = -(-n // block)
    pad = nt * block - n
    if pad:
        reach = mx.concatenate([reach, mx.zeros((b, t, pad), mx.bool_)], axis=-1)
    return mx.any(reach.reshape(b * t, nt, block), axis=-1)


@lru_cache(None)
def _operation():
    forward, backward = _kernels()

    @mx.custom_function
    def op(q, k, w, reach):
        b, t, h, d = q.shape
        n = k.shape[1]
        from . import indexer_select as fused

        if fused._BQ_ENABLED and t > 1 and fused.supported(q, k):
            return fused.score_bq(q, k, w, reach)
        tiles = _n_tiles(b, t, n)
        nt = 32 * (d // 32)
        tv = _tile_valid(reach, tiles, _BK)
        (out,) = forward(
            inputs=[q, k, w, reach, tv, _dims(b, t, n, tiles)],
            template=[("T", q.dtype), ("IH", h), ("ID", d)],
            grid=(nt, b * t, tiles),
            threadgroup=(nt, 1, 1),
            output_shapes=[(b, t, n)],
            output_dtypes=[mx.float32],
        )
        return out

    @op.vjp
    def vjp(primals, cotangent, _output):
        q, k, w, reach = primals
        g = cotangent
        b, t, h, d = q.shape
        n = k.shape[1]
        tiles = _n_tiles(b, t, n)
        nt = 32 * (d // 32)
        tv = _tile_valid(reach, tiles, _BK)
        dq, dk, dw = backward(
            inputs=[q, k, w, reach, tv, g.astype(q.dtype), _dims(b, t, n, tiles)],
            template=[("T", q.dtype), ("IH", h), ("ID", d), ("SHARDS", _SHARDS)],
            grid=(nt, b * t, tiles),
            threadgroup=(nt, 1, 1),
            output_shapes=[q.shape, (_SHARDS,) + k.shape, w.shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            init_value=0,
        )
        return (
            dq.astype(q.dtype),
            mx.sum(dk, axis=0).astype(k.dtype),
            dw.astype(w.dtype),
            mx.zeros_like(reach),
        )

    return op


def indexer_score(q, k, w, reach):
    """q [B,T,H,D] × k [B,N,D]，w [B,T,H] → [B,T,N] fp32（不可达 = -inf）。"""
    n = k.shape[1]
    if n == 0:
        return mx.zeros((q.shape[0], q.shape[1], 0), dtype=mx.float32)
    b, t, _, _ = q.shape
    if reach.shape != (b, t, n):
        reach = mx.broadcast_to(reach, (b, t, n))
    w = w.astype(q.dtype)
    if not _supported(q, k):
        return eager_indexer_score(q, k, w, reach)
    try:
        return _operation()(q, k, w, reach)
    except Exception as exc:  # noqa: BLE001
        global _ENABLED
        _ENABLED = False
        print(
            f"[indexer_score] kernel 不可用，回退 einsum：{type(exc).__name__}: {exc}"
        )
        return eager_indexer_score(q, k, w, reach)


def prewarm_indexer_score(
    n_heads: int, head_dim: int, dtype=mx.bfloat16, seq: int = 32
):
    """Eager callable/JIT creation for training and key-split decode grids."""
    from .indexer_select import _kernel, _selection_mask_kernel

    _kernel()  # create callable before mx.compile traces a fused selection
    _selection_mask_kernel()
    if not _ENABLED or mx.default_device() != mx.gpu:
        return False
    if n_heads not in (4, 8) or head_dim not in (32, 64):
        return False
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    try:
        for t, n in ((seq, seq), (1, max(seq, 64))):
            q = mx.zeros((1, t, n_heads, head_dim), dtype)
            k = mx.zeros((1, n, head_dim), dtype)
            w = mx.zeros((1, t, n_heads), dtype)
            reach = mx.ones((1, t, n), mx.bool_)
            out, grads = mx.vjp(
                lambda a, b, c, r: indexer_score(a, b, c, r),
                [q, k, w, reach],
                [mx.ones((1, t, n), mx.float32)],
            )
            mx.eval(out, grads)
        return True
    except Exception:  # noqa: BLE001
        return False
