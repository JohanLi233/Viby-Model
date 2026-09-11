"""Indexed MQA flash attention for the 16-head training recipe.

KV stays in its original window/compressed pools. One threadgroup owns one query;
its 16 heads share each indirectly loaded key tile (16 keys by default, 32 with
VIBY_SPARSE_ATTN_KEY_TILE=32) and use SIMD-group MMA. Online softmax includes the
learned sink. No [B,T,visible,D] gathered KV tensor or [B,H,T,N] score tensor is
materialized.

The existing Boolean mask is compacted verbatim, including ties that can select
more than index_topk positions. Full/Reindex/Reuse selection and document/padding
rules are not approximated. VIBY_SPARSE_ATTN_KERNEL=0 restores native SDPA.

2026-09-11 反向拆分（`VIBY_SPARSE_ATTN_BWD_SPLIT`，默认开）：

- `_BWD_Q_SINK`：一个 TG 一个 query，写 `Delta=⟨dO,O⟩` 与 `dSinkRow`，
  同一份可见槽顺序重算 S/U，唯一写非 atomic 的 `dQ`，
  Q/G staging 区域在最后一次 barrier 后复用为 dQ store scratch；
- `_BWD_KV`：读全局 `Delta`（不重算 `⟨dO,O⟩`），在片上先把两个 head group 的
  score/value 两条路径合并成每个 key 的 `[D]` 向量，再对每个 (key,d) 只发
  一次 atomic；
- `_FINALIZE_POOL`：shard 求和后唯一写 `dWindow`/`dCompressed`；
  `_FINALIZE_SINK` 跨 (b,t) 归约 `dSinkRow`。

所以 `dQ`/`dWindow`/`dCompressed`/`dSink` 都不再需要 atomic 输出接口，代价是
两个反向 kernel 各自重算一遍 S/U。`=0` 回退到原来的单 kernel sharded atomic
路径；两条路径共享同一套 metadata、`LSE` 与可见集合。

threadgroup 预算：拆分的 `_BWD_KV` 在 D=128/BK=32 时会超出 Metal 的
32 KiB 静态上限，因此反向固定 BK=16（前向仍可用 `VIBY_SPARSE_ATTN_KEY_TILE=32`）。

§2 key-owned dKV（`VIBY_SPARSE_ATTN_KEY_BWD=1`，默认关）：
window 用解析邻接，compressed 直接从 selection 构建 occurrence CSR
（count / GPU 递归 exclusive scan / fill），不去重。`_ONE_KEY` 的四个 SIMD
group 分担 heads，唯一写梯度；compressed 固定 dispatch 四段，按 GPU degree
决定有效段数，再归约 FP32 partial。整条 dKV 路径没有浮点 atomic。
它替换 `_BWD_KV`+`_FINALIZE_POOL`，依赖 `VIBY_SPARSE_ATTN_BWD_SPLIT=1`。
定向数值验收见 tests/test_sparse_attention_key_owned.py：与原后端保留相同的
P/Ds 降精度位置，但 SIMD/MMA 点积顺序可能使系数跨过舍入边界，不承诺逐位一致。
主训练形状的直接 ABBA 比 sharded 后端慢约 1.4%，因此维持 opt-in；
详见 research/CSA2_KERNEL_ACCEPTANCE.md 和 research/KEY_OWNED_DKV.md。

§5 exact threshold select：`select_topk` 与 `select_topk_packed` 共用 `_SELECT`
的 4-bit 片上 radix；通用入口保持完整 rank 域，阈值只由现有 FP32
`sc - global_id*1e-7` 决定，等于阈值的
boundary ties 全部保留，不按名义 k 截断。

2026-09-11 验收后默认：前向 BK=16，exact radix 开启；融合 Indexer 与
token-owned MoE combine 的组合通过同协议整步 ABBA（见 research/CSA2_KERNEL_ACCEPTANCE.md）。
"""
import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("VIBY_SPARSE_ATTN_KERNEL", "1") != "0"
_HEADER = "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
_TOPK_ENABLED = os.environ.get("VIBY_SPARSE_TOPK_KERNEL", "1") != "0"
# 2026-09-11：反向拆分（query-owned dQ/dSink + 片上归约的 dKV）是主训练路径，
# `VIBY_SPARSE_ATTN_BWD_SPLIT=0` 回退到改动前的单 kernel atomic 路径。
# 两条路径共享同一套 metadata / LSE / 可见集合；对拍见
# tests/test_sparse_attention_kernel.py::test_indexed_attention_split_backward_matches_eager。
_SPLIT_BWD = os.environ.get("VIBY_SPARSE_ATTN_BWD_SPLIT", "1") != "0"
# §2：`VIBY_SPARSE_ATTN_KEY_BWD=1` 把 dKV 换成 key-owned 归约后端（window 隐式
# 邻接 + compressed occurrence CSR），无浮点 atomic。它替换的是
# `_BWD_KV` + `_FINALIZE_POOL` 这一对可互换的归约后端，不叠加。
#
# 数值定向验收已通过；SIMD/MMA 的点积与归约顺序不同，不要求逐位一致。
# 同协议直接 ABBA 未胜过 sharded 后端，默认关。
_KEY_OWNED_BWD = os.environ.get("VIBY_SPARSE_ATTN_KEY_BWD", "0") != "0"
_SHARDS = 4
_KEY_TILE = int(os.environ.get("VIBY_SPARSE_ATTN_KEY_TILE", "16"))
if _KEY_TILE not in (16, 32):
    raise ValueError("VIBY_SPARSE_ATTN_KEY_TILE must be 16 or 32")

_COMPACT = r"""
    uint lane = thread_position_in_grid.x;
    uint query = thread_position_in_grid.y;
    uint N = dims[0];
    uint count = 0;
    for (uint base=0;base<N;base+=32) {
        uint j=base+lane;
        uint yes=(j<N && visible[(size_t)query*N+j]) ? 1u : 0u;
        uint rank=simd_prefix_exclusive_sum(yes);
        if (yes) indices[(size_t)query*N+count+rank]=j;
        count+=simd_sum(yes);
    }
    if (lane==0) lengths[query]=count;
"""

# Exact radix selection over the IEEE-754 ordered representation. Only the
# threshold is selected: the final >= comparison retains ALL boundary ties,
# exactly as _topk_masks does. This fuses selection, mask and index compaction.
_SELECT = r"""
    #pragma clang fp contract(off)
    uint tid=thread_position_in_grid.x, sg=tid/32, lane=tid%32;
    uint query=thread_position_in_grid.y, N=dims[0], kth=dims[1];
    uint keys[ITEMS];
    bool eligible[ITEMS];
    for (uint i=0;i<ITEMS;++i) {
        uint j=i*128+tid;
        float raw=j<N ? scores[(size_t)query*N+j] : -INFINITY;
        eligible[i]=j<N && reach[(size_t)query*N+j] && raw>-1e30f;
        float offset=float(j)*1e-7f;
        float value=raw-offset;
        if (value==0.0f) value=0.0f;
        uint bits=as_type<uint>(value);
        keys[i]=bits ^ ((bits&0x80000000u) ? 0xffffffffu : 0x80000000u);
    }
    threadgroup uint hist[4*16];
    threadgroup uint prefix, prefix_mask, rank, selected_count;
    if (tid==0) { prefix=0; prefix_mask=0; rank=kth; selected_count=N; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kth<N) for (int shift=28;shift>=0;shift-=4) {
        // Once only one candidate remains, its full bits give the threshold.
        if (selected_count==1) {
            uint winner=0;
            for (uint i=0;i<ITEMS;++i)
                if (i*128+tid<N && (keys[i]&prefix_mask)==prefix) winner=max(winner,keys[i]);
            winner=simd_max(winner);
            if (lane==0) hist[sg]=winner;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid==0) prefix=max(max(hist[0],hist[1]),max(hist[2],hist[3]));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }
        uint counts[16];
        for (uint bucket=0;bucket<16;++bucket) counts[bucket]=0;
        for (uint i=0;i<ITEMS;++i)
            if (i*128+tid<N && (keys[i]&prefix_mask)==prefix) ++counts[(keys[i]>>shift)&15u];
        for (uint bucket=0;bucket<16;++bucket) {
            uint count=simd_sum(counts[bucket]);
            if (lane==0) hist[sg*16+bucket]=count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid==0) {
            for (int bucket=15;bucket>=0;--bucket) {
                uint count=hist[bucket]+hist[16+bucket]+hist[32+bucket]+hist[48+bucket];
                if (rank>count) rank-=count;
                else {
                    prefix|=uint(bucket)<<shift;
                    prefix_mask|=15u<<shift;
                    selected_count=count;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reuse hist for a four-SIMD-group exclusive prefix, preserving key order.
    uint total=0;
    for (uint i=0;i<ITEMS;++i) {
        uint j=i*128+tid;
        bool yes=kth>0 && eligible[i] && keys[i]>=prefix;
        if (j<N) keep[(size_t)query*N+j]=yes;
        uint local=simd_prefix_exclusive_sum(uint(yes));
        uint count=simd_sum(uint(yes));
        if (lane==0) hist[sg]=count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset=0;
        for (uint g=0;g<sg;++g) offset+=hist[g];
        if (yes) indices[(size_t)query*N+total+offset+local]=j;
        total+=hist[0]+hist[1]+hist[2]+hist[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid==0) lengths[query]=total;
"""


@lru_cache(None)
def _selection_kernel():
    return mx.fast.metal_kernel(
        name="mqa_topk_compact", input_names=["scores", "reach", "dims"],
        output_names=["keep", "indices", "lengths"], source=_SELECT,
    )


def _selection_items(n):
    return 1 << max(0, ((n+127)//128-1).bit_length())


def topk_enabled():
    return _TOPK_ENABLED


def select_topk(scores, reach, k):
    """Drop-in threshold selection plus ready-to-use indexed-attention metadata.

    §5.1 的通用 dense 兼容入口：阈值仍在 `sc - global_id*1e-7` 的原始完整 rank
    域里取，`reach` 只控制最终输出。所以“不可达的高分”依旧参与定阈值——
    `sc=[100,1], reach=[False,True], k=1` 的结果就是空集，和旧 `_topk_masks`
    逐位一致。只在 eligibility 域选阈值的融合 Indexer 入口另见
    indexer_select.fused_select；这里始终保留通用 dense rank 域。
    """
    b, t, n=scores.shape
    if n==0:
        return mx.zeros(scores.shape, mx.bool_), (mx.zeros((b*t, 0), mx.int32), mx.zeros((b*t,), mx.int32))
    keep, indices, lengths = _selection_kernel()(
        inputs=[mx.stop_gradient(scores), mx.broadcast_to(reach, scores.shape),
                mx.array([n, min(k, n)], mx.uint32)],
        template=[("ITEMS", _selection_items(n))],
        grid=(128, b*t, 1), threadgroup=(128, 1, 1),
        output_shapes=[scores.shape, (b*t, n), (b*t,)],
        output_dtypes=[mx.bool_, mx.int32, mx.int32],
    )
    return keep, (indices, lengths)


@lru_cache(None)
def _pack_selection_kernel():
    return mx.fast.metal_kernel(
        name="mqa_pack_selection", input_names=["indices", "lengths", "offsets", "dims"],
        output_names=["flat"], source=r"""
        uint tid=thread_position_in_grid.x, row=thread_position_in_grid.y;
        for (uint slot=tid;slot<uint(lengths[row]);slot+=128)
            flat[offsets[row]+slot]=indices[(size_t)row*dims[0]+slot];
        """,
    )


def select_topk_packed(scores, reach, k):
    """GPU offsets plus a static-capacity arena; only flat[:offsets[-1]] is valid.

    The Attention consumer continues to use row-major metadata. This separate
    packed entry never passes its flat arena to the row-major _LOAD_TILE.
    """
    keep, (indices, lengths) = select_topk(scores, reach, k)
    rows, width = indices.shape
    row_offsets = _csr_exclusive_scan(lengths)
    if width == 0 or rows == 0:
        return row_offsets, keep, (mx.zeros((0,), mx.int32), lengths)
    (flat,) = _pack_selection_kernel()(
        inputs=[indices, lengths, row_offsets, mx.array([width], mx.uint32)],
        grid=(128, rows, 1), threadgroup=(128, 1, 1),
        output_shapes=[(indices.size,)], output_dtypes=[mx.int32],
    )
    return row_offsets, keep, (flat, lengths)


@lru_cache(None)
def prewarm_topk(max_keys):
    if not _TOPK_ENABLED or mx.default_device()!=mx.gpu:
        return
    items=1
    while items<=_selection_items(max_keys):
        n=items*128
        values=mx.zeros((1, 1, n), mx.float32)
        mx.eval(select_topk(values, mx.ones(values.shape, mx.bool_), min(64, n)))
        items*=2


# Every forward/backward key tile has exactly the same visibility and addresses.
# Invalid window slots load zeros and are assigned -inf before softmax.
_LOAD_TILE = r"""
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<BK) {
            uint slot=base+tid;
            int key=-1;
            if (slot<W) {
                int p=int(t)-int(W)+1+int(slot);
                if (p>=0 && pad[b*TQ+uint(p)] && segment[b*TQ+uint(p)]==segment[query])
                    key=p;
            } else if (slot<W+uint(lengths[query])) {
                key=int(TQ)+indices[(size_t)query*NC+slot-W];
            }
            ids[tid]=key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid==0) {
            uint yes=0; for (uint j=0;j<BK;++j) yes|=uint(ids[j]>=0);
            any_key=yes;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!any_key) continue;
        for (uint i=tid;i<BK*D;i+=NT) {
            uint row=i/D,d=i%D;
            int key=ids[row];
            T value=T(0);
            if (key>=0) {
                value=uint(key)<TQ ? window[((size_t)b*TQ+uint(key))*D+d]
                    : compressed[((size_t)b*NC+uint(key)-TQ)*D+d];
            }
            Ks[row*(D+8)+d]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_FWD = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    constexpr uint H=16;
    constexpr uint SS=BK+4, SP=BK+8;
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup uint any_key;
    threadgroup float scores[NP*H*SS];
    threadgroup T Ps[H*SP];
    threadgroup float Os[H*(D+4)];
    threadgroup float ms[H], ls[H], rescale[H];
    if (tid<H) { ms[tid]=sinks[tid]; ls[tid]=1.0f; }
    simdgroup_matrix<T,8,8> Qf[DD/8];
    simdgroup_matrix<float,8,8> Of[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],q+((size_t)query*H+hg*8)*D+col+d*8,D);
        Of[d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    }
    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8];
        for (uint j=0;j<BK/8;++j) Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
        }
        for (uint j=0;j<BK/8;++j) simdgroup_store(Sf[j],scores+(dp*H+hg*8)*SS+j*8,SS);
        for (uint d=0;d<DD/8;++d) simdgroup_store(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) {
            float old=ms[tid], next=old;
            for (uint j=0;j<BK;++j) {
                float dot=scores[tid*SS+j];
                for (uint p=1;p<NP;++p) dot+=scores[(p*H+tid)*SS+j];
                float s=ids[j]>=0 ? dot*scale[0] : -INFINITY;
                scores[tid*SS+j]=s;
                next=max(next,s);
            }
            float alpha=exp(old-next), sum=0.0f;
            for (uint j=0;j<BK;++j) {
                float p=exp(scores[tid*SS+j]-next);
                Ps[tid*SP+j]=T(p);
                sum+=p;
            }
            ms[tid]=next; ls[tid]=ls[tid]*alpha+sum; rescale[tid]=alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<H*D;i+=NT) Os[(i/D)*(D+4)+i%D]*=rescale[i/D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=0;d<DD/8;++d) {
            simdgroup_load(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
            for (uint j=0;j<BK/8;++j) {
                simdgroup_matrix<T,8,8> Pf,Vf;
                simdgroup_load(Pf,Ps+hg*8*SP+j*8,SP);
                simdgroup_load(Vf,Ks+j*8*(D+8)+col+d*8,D+8);
                simdgroup_multiply_accumulate(Of[d],Pf,Vf,Of[d]);
            }
        }
    }
    for (uint d=0;d<DD/8;++d) simdgroup_store(Of[d],Os+hg*8*(D+4)+col+d*8,D+4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<H*D;i+=NT) out[(size_t)query*H*D+i]=Os[(i/D)*(D+4)+i%D]/ls[i/D];
    if (tid<H) lse[query*H+tid]=ms[tid]+log(ls[tid]);
"""

# ---------------------------------------------------------------------------
# 2026-09-11：反向按规格 §1 拆成 query-owned 阶段与 KV 梯度阶段。
#
# 阶段一 `_BWD_Q_SINK`：一个 TG 一个 query，唯一写 dQ，并把 ⟨dO,O⟩ 落成 fp32
# Delta 供后续复用（不再重算）。
# 阶段二 `_BWD_KV`：一个 TG 一个 query，读 Delta 重算 P/Ds，先把两个 head
# group 的 dKV 在片上归约，再对每个 (key,d) 只发一次 atomic。
# 阶段三 `_FINALIZE_POOL`/`_FINALIZE_SINK`：shard 求和与 dSink 跨 (b,t) 归约，
# 全部唯一写。
#
# 旧单 kernel 路径保留为 `_BWD`，VIBY_SPARSE_ATTN_BWD_SPLIT=0 原样回退。

# ---------------------------------------------------------------------------
# §2 key-owned dKV：真正无浮点 atomic 的归约后端。
#
# 一个 TG 拥有一个 key，遍历所有指向它的 query。Window key 的邻居是解析区间
# [p, min(T,p+W))，不建 CSR；compressed key 走显式反向邻接（count / scan /
# fill），每条真实 occurrence 保留，不去重、不建 [B,T,N] 补齐邻接。
#
# Selection already encodes compressed visibility, exactly as in _LOAD_TILE.
# Only slots below lengths[query] are real occurrences; do not reapply the
# window document/pad predicate or deduplicate compressed ids.
_CSR_COUNT = r"""
    uint tid=thread_position_in_grid.x;
    uint query=thread_position_in_grid.y;
    uint TQ=dims[1], NC=dims[2], b=query/TQ;
    for (uint slot=tid;slot<uint(lengths[query]);slot+=128) {
        uint j=uint(indices[(size_t)query*NC+slot]);
        atomic_fetch_add_explicit(counts+b*NC+j,1,memory_order_relaxed);
    }
"""

# Hierarchical GPU exclusive scan. Each block scans 256 counts and emits its
# total. Recursively scan those totals, then add the preceding block prefix.
# The final entry is the logical edge count; it never travels through Python.
_CSR_SCAN_BLOCK = r"""
    uint tid=thread_position_in_threadgroup.x, lane=tid%32, sg=tid/32;
    uint block=threadgroup_position_in_grid.x, i=block*256+tid;
    uint n=dims[0];
    threadgroup int totals[8];
    int value=i<n ? counts[i] : 0;
    int prefix=simd_prefix_exclusive_sum(value);
    int sum=simd_sum(value);
    if (lane==31) totals[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) {
        int v=lane<8 ? totals[lane] : 0;
        int p=simd_prefix_exclusive_sum(v);
        if (lane<8) totals[lane]=p;
        int total=simd_sum(v);
        if (lane==0) block_total[block]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i<n) {
        local_prefix[i]=prefix+totals[sg];
        if (i==n-1) local_prefix[n]=prefix+totals[sg]+value;
    }
"""

_CSR_SCAN_ADD = r"""
    uint i=thread_position_in_grid.x, n=dims[0];
    if (i<=n) row_ptr[i]=local_prefix[i]+block_prefix[min(i,n-1)/256];
"""

# §2.3 阶段三 fill。写入位置唯一，但接口要求同一调用的输出统一按 atomic 处理。
_CSR_FILL = r"""
    uint tid=thread_position_in_grid.x;
    uint query=thread_position_in_grid.y;
    uint TQ=dims[1], NC=dims[2], b=query/TQ;
    for (uint slot=tid;slot<uint(lengths[query]);slot+=128) {
        uint key=b*NC+uint(indices[(size_t)query*NC+slot]);
        uint local=atomic_fetch_add_explicit(cursor+key,1u,memory_order_relaxed);
        uint dst=row_ptr[key]+local;
        atomic_store_explicit(edge_q+dst,query,memory_order_relaxed);
        atomic_store_explicit(edge_slot+dst,slot,memory_order_relaxed);
    }
"""

# §2.4 key-owned 归约。约定：
#  * 每个 SIMD group 独占 4 个 head（group g → h = g, g+4, ...）；
#  * 每 lane 拥有 d = lane + 32*r，用 SIMD 点积而不是“只有一列有效的 8x8 MMA”；
#  * 各组的 occurrence/head 循环长度可以不同，barrier 只放在所有独立循环之后；
#  * 每个 key 唯一写，没有浮点 atomic。
_ONE_KEY = r"""
    uint tid=thread_position_in_grid.x, lane=tid%32, group=tid/32;
    uint key=thread_position_in_grid.y;
    uint part=thread_position_in_grid.z;
    uint TQ=dims[1], NC=dims[2], B=dims[0];
    uint pool_size=COMPRESSED ? NC : TQ;
    uint b=key/pool_size, pool=key%pool_size;
    threadgroup T Ksh[D];
    threadgroup float Partial[4*(D+4)];

    uint start=0, degree=COMPRESSED ? 0u : min(uint(WIN), TQ-pool), parts=1;
    if (COMPRESSED) {
        start=uint(row_ptr[key]);
        degree=uint(row_ptr[key+1])-start;
        parts=min(4u,max(1u,(degree+31)/32));
    }
    for (uint dd=tid;dd<D;dd+=128)
        Ksh[dd]=COMPRESSED ? compressed[(size_t)key*D+dd]
                           : window[(size_t)key*D+dd];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[D/32];
    for (uint r=0;r<D/32;++r) acc[r]=0.0f;
    // Direct iteration removes any degree-dependent threadgroup allocation.
    // Inactive parts still reach the barrier and uniquely overwrite zeros.
    for (uint occ=part;part<parts && occ<degree;occ+=parts) {
        uint q=COMPRESSED ? uint(edge_q[start+occ]) : b*TQ+pool+occ;
        if (!COMPRESSED && (!pad[key] || segment[q]!=segment[key])) continue;
        for (uint h=group;h<H;h+=4) {
            float qv[D/32], gv[D/32];
            float dot_q=0.0f, dot_g=0.0f;
            for (uint r=0;r<D/32;++r) {
                uint dd=lane+32*r;
                qv[r]=float(Q[((size_t)q*H+h)*D+dd]);
                gv[r]=float(dO[((size_t)q*H+h)*D+dd]);
                float kvv=float(Ksh[dd]);
                dot_q+=qv[r]*kvv; dot_g+=gv[r]*kvv;
            }
            dot_q=simd_sum(dot_q); dot_g=simd_sum(dot_g);
            float p=exp(dot_q*scale[0]-LSE[q*H+h]);
            float ds=p*(dot_g-Delta[q*H+h])*scale[0];
            float a=float(T(ds)), bb=float(T(p));
            for (uint r=0;r<D/32;++r) acc[r]+=a*qv[r]+bb*gv[r];
        }
    }
    for (uint r=0;r<D/32;++r) Partial[group*(D+4)+lane+32*r]=acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dd=tid;dd<D;dd+=128) {
        float total=Partial[dd]+Partial[D+4+dd]+Partial[2*(D+4)+dd]+Partial[3*(D+4)+dd];
        // Compressed output is FP32 until all four parts have been reduced.
        dKV[((size_t)part*B*pool_size+key)*D+dd]=OutT(total);
    }
"""

_KEY_REDUCE_PARTS = r"""
    uint i=thread_position_in_grid.x, size=dims[0];
    if (i<size) {
        float total=partial[i]+partial[size+i]+partial[2*size+i]+partial[3*size+i];
        dcompressed[i]=T(total);
    }
"""

# 一个 TG 一个 query：⟨dO,O⟩ 与逐行 sink 贡献。不依赖任何分区假设，
# 阶段一（query-owned dQ）与阶段二（KV）都读这份全局 delta。
_DELTA_SINK = r"""
    uint tid=thread_position_in_grid.x;
    uint query=thread_position_in_grid.y;
    threadgroup T Gs[H*D];
    threadgroup float Partial[NT/32];

    for (uint i=tid;i<H*D;i+=NT) Gs[i]=g[(size_t)query*H*D+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h=0;h<H;++h) {
        float a=0.0f;
        for (uint i=tid;i<D;i+=NT)
            a+=float(Gs[h*D+i])*float(out[(size_t)query*H*D+h*D+i]);
        a=simd_sum(a);
        if (tid%32==0) Partial[tid/32]=a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid==0) {
            float s=0.0f;
            for (uint p=0;p<NT/32;++p) s+=Partial[p];
            delta[query*H+h]=s;
            dsinkrow[query*H+h]=-exp(sinks[h]-lse[query*H+h])*s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


_BWD_Q_SINK = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint B=dims[0], TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    constexpr uint H=16;
    constexpr uint SS=BK+4, SP=DD+8;
    union QWorkspace {
        T stage[2*H*(D+8)];
        float scratch[H*(D+4)];
    };
    threadgroup QWorkspace workspace;
    threadgroup T* Qs=workspace.stage;
    threadgroup T* Gs=workspace.stage+H*(D+8);
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup uint any_key;
    threadgroup float DotPart[NP*H*SS];
    threadgroup float Sfull[H*SS];
    threadgroup T Ds[H*(BK+8)];
    threadgroup float DeltaH[H];

    for (uint i=tid;i<H*D;i+=NT) {
        uint h=i/D, d=i%D;
        Qs[h*(D+8)+d]=q[((size_t)query*H+h)*D+d];
        Gs[h*(D+8)+d]=g[((size_t)query*H+h)*D+d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h=sg;h<H;h+=NT/32) {
        float value=0.0f;
        for (uint d=tid%32;d<D;d+=32)
            value+=float(Gs[h*(D+8)+d])*float(out[((size_t)query*H+h)*D+d]);
        value=simd_sum(value);
        if (tid%32==0) {
            DeltaH[h]=value;
            delta[query*H+h]=value;
            dsinkrow[query*H+h]=-exp(sinks[h]-lse[query*H+h])*value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<T,8,8> Qf[DD/8], Gf[DD/8];
    simdgroup_matrix<float,8,8> DQ[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],Qs+((size_t)(hg*8)*  (D+8))+col+d*8,D+8);
        simdgroup_load(Gf[d],Gs+((size_t)(hg*8)*(D+8))+col+d*8,D+8);
        DQ[d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    }

    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8], Uf[BK/8];
        for (uint j=0;j<BK/8;++j) {
            Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            Uf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        }
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
            simdgroup_multiply_accumulate(Uf[j],Gf[d],Kf,Uf[j]);
        }
        // DotPart: [NP][H][BK+4]；两个 head group 各写自己的 8 行。
        for (uint j=0;j<BK/8;++j) {
            simdgroup_store(Sf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float sd=DotPart[tid*SS+j];
            for (uint p=1;p<NP;++p) sd+=DotPart[(p*H+tid)*SS+j];
            Sfull[tid*SS+j]=sd*scale[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // QK has been merged into Sfull. Reuse the same bounded allocation
        // for dOK only after every reader of QK has finished.
        for (uint j=0;j<BK/8;++j)
            simdgroup_store(Uf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float ud=DotPart[tid*SS+j];
            for (uint p=1;p<NP;++p) ud+=DotPart[(p*H+tid)*SS+j];
            float p=ids[j]>=0 ? exp(Sfull[tid*SS+j]-lse[query*H+tid]) : 0.0f;
            Ds[tid*(BK+8)+j]=T(p*(ud-DeltaH[tid])*scale[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Df,Kf;
            simdgroup_load(Df,Ds+hg*8*(BK+8)+j*8,BK+8);
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8);
            simdgroup_multiply_accumulate(DQ[d],Df,Kf,DQ[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Qs/Gs 已死亡；Scratch 复用为 fp32 dQ store，再唯一写回全局。
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=0;d<DD/8;++d)
        simdgroup_store(DQ[d],workspace.scratch+hg*8*(D+4)+col+d*8,D+4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<H*D;i+=NT)
        dq[(size_t)query*H*D+i]=T(workspace.scratch[(i/D)*(D+4)+i%D]);
"""

_BWD_KV = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint B=dims[0], TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    constexpr uint H=16;
    constexpr uint SS=BK+4, SP=DD+8;
    threadgroup T Qs[H*(D+8)], Gs[H*(D+8)];
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup uint any_key;
    threadgroup float DotPart[NP*H*SS];
    threadgroup float Sfull[H*SS];
    threadgroup T Ds[H*(BK+8)], Ps[H*(BK+8)];
    threadgroup float Grad8[2*8*(D+4)];
    threadgroup float DeltaH[H];

    for (uint i=tid;i<H*D;i+=NT) {
        uint h=i/D, d=i%D;
        Qs[h*(D+8)+d]=q[((size_t)query*H+h)*D+d];
        Gs[h*(D+8)+d]=g[((size_t)query*H+h)*D+d];
    }
    if (tid<H) DeltaH[tid]=deltaglobal[query*H+tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<T,8,8> Qf[DD/8], Gf[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],Qs+((size_t)(hg*8)*  (D+8))+col+d*8,D+8);
        simdgroup_load(Gf[d],Gs+((size_t)(hg*8)*(D+8))+col+d*8,D+8);
    }

    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8], Uf[BK/8];
        for (uint j=0;j<BK/8;++j) {
            Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            Uf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        }
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
            simdgroup_multiply_accumulate(Uf[j],Gf[d],Kf,Uf[j]);
        }
        // DotPart: [NP][H][BK+4]；两个 head group 各写自己的 8 行。
        for (uint j=0;j<BK/8;++j) {
            simdgroup_store(Sf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float sd=DotPart[tid*SS+j];
            for (uint p=1;p<NP;++p) sd+=DotPart[(p*H+tid)*SS+j];
            Sfull[tid*SS+j]=sd*scale[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // QK has been merged into Sfull. Reuse the same bounded allocation
        // for dOK only after every reader of QK has finished.
        for (uint j=0;j<BK/8;++j)
            simdgroup_store(Uf[j],DotPart+(dp*H+hg*8)*SS+j*8,SS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float ud=DotPart[tid*SS+j];
            for (uint p=1;p<NP;++p) ud+=DotPart[(p*H+tid)*SS+j];
            float p=ids[j]>=0 ? exp(Sfull[tid*SS+j]-lse[query*H+tid]) : 0.0f;
            Ds[tid*(BK+8)+j]=T(p*(ud-DeltaH[tid])*scale[0]);
            Ps[tid*(BK+8)+j]=T(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint key8=0;key8<BK;key8+=8) {
            for (uint d=0;d<DD/8;++d) {
                simdgroup_matrix<T,8,8> Df,Pf;
                simdgroup_matrix<float,8,8> DK=make_filled_simdgroup_matrix<float,8,8>(0.0f);
                simdgroup_load(Df,Ds+hg*8*(BK+8)+key8,BK+8,ulong2(0,0),true);
                simdgroup_load(Pf,Ps+hg*8*(BK+8)+key8,BK+8,ulong2(0,0),true);
                simdgroup_multiply_accumulate(DK,Df,Qf[d],DK);
                simdgroup_multiply_accumulate(DK,Pf,Gf[d],DK);
                simdgroup_store(DK,Grad8+hg*8*(D+4)+col+d*8,D+4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tid;i<8*D;i+=NT) {
                uint j=i/D,d=i%D; int key=ids[key8+j];
                if (key>=0) {
                    float value=Grad8[j*(D+4)+d]+Grad8[(8+j)*(D+4)+d];
                    uint shard=(uint(key)<TQ ? t : (t^(t>>4)^(t>>8))) & uint(SHARDS-1);
                    atomic_fetch_add_explicit(
                        dpoolshard+(((size_t)shard*B+b)*(TQ+NC)+uint(key))*D+d,
                        value,memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""


# 阶段三：唯一写，无 atomic。dPoolShard 是唯一需要清零的浮点缓冲。
_FINALIZE_POOL = r"""
    uint d=thread_position_in_grid.x;
    uint pool=thread_position_in_grid.y;
    uint b=thread_position_in_grid.z;
    uint TQ=dims[0], NC=dims[1], B=dims[2];
    if (d>=D) return;
    float acc=0.0f;
    for (uint s=0;s<SHARDS;++s)
        acc+=poolshard[(((size_t)s*B+b)*(TQ+NC)+pool)*D+d];
    if (pool<TQ) dwindow[((size_t)b*TQ+pool)*D+d]=T(acc);
    else         dcompressed[((size_t)b*NC+pool-TQ)*D+d]=T(acc);
"""

_FINALIZE_SINK = r"""
    uint tid=thread_position_in_grid.x, h=thread_position_in_grid.y;
    uint TQ=dims[0], B=dims[1];
    threadgroup float partial[4];
    float acc=0.0f;
    for (uint q=tid;q<B*TQ;q+=128) acc+=dsinkrow[(size_t)q*H+h];
    acc=simd_sum(acc);
    if (tid%32==0) partial[tid/32]=acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) dsink[h]=partial[0]+partial[1]+partial[2]+partial[3];
"""

_BWD = r"""
    uint tid=thread_position_in_grid.x, sg=tid/32;
    constexpr uint NT=64*NP, DD=D/NP;
    uint hg=sg/NP, dp=sg%NP, col=dp*DD;
    uint query=thread_position_in_grid.y;
    uint B=dims[0], TQ=dims[1], NC=dims[2], b=query/TQ, t=query%TQ;
    uint shard=(t/16)%SHARDS;
    constexpr uint H=16;
    constexpr uint SS=BK+(NP==2 ? 2 : 4), SP=BK+8;
    threadgroup T Ks[BK*(D+8)];
    threadgroup int ids[BK];
    threadgroup uint any_key;
    threadgroup float scores[NP*H*SS], dprob[NP*H*SS];
    threadgroup T Ps[H*SP], Ds[H*SP];
    threadgroup float tile_grad[16*(D+4)];
    threadgroup float delta[H];
    if (tid<H) {
        float a=0.0f;
        for (uint d=0;d<D;++d) a+=float(g[((size_t)query*H+tid)*D+d])*out[((size_t)query*H+tid)*D+d];
        delta[tid]=a;
    }
    simdgroup_matrix<T,8,8> Qf[DD/8], Gf[DD/8];
    simdgroup_matrix<float,8,8> DQ[DD/8];
    for (uint d=0;d<DD/8;++d) {
        simdgroup_load(Qf[d],q+((size_t)query*H+hg*8)*D+col+d*8,D);
        simdgroup_load(Gf[d],g+((size_t)query*H+hg*8)*D+col+d*8,D);
        DQ[d]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    }
    for (uint base=0;base<W+uint(lengths[query]);base+=BK) {
        /*LOAD_TILE*/
        simdgroup_matrix<float,8,8> Sf[BK/8], DP[BK/8];
        for (uint j=0;j<BK/8;++j) {
            Sf[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            DP[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        }
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Kf;
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8,ulong2(0,0),true);
            simdgroup_multiply_accumulate(Sf[j],Qf[d],Kf,Sf[j]);
            simdgroup_multiply_accumulate(DP[j],Gf[d],Kf,DP[j]);
        }
        for (uint j=0;j<BK/8;++j) {
            simdgroup_store(Sf[j],scores+(dp*H+hg*8)*SS+j*8,SS);
            simdgroup_store(DP[j],dprob+(dp*H+hg*8)*SS+j*8,SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<H) for (uint j=0;j<BK;++j) {
            float dot=scores[tid*SS+j], dg=dprob[tid*SS+j];
            for (uint part=1;part<NP;++part) {
                dot+=scores[(part*H+tid)*SS+j];
                dg+=dprob[(part*H+tid)*SS+j];
            }
            float p=ids[j]>=0 ? exp(dot*scale[0]-lse[query*H+tid]) : 0.0f;
            Ps[tid*SP+j]=T(p);
            Ds[tid*SP+j]=T(p*(dg-delta[tid])*scale[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=0;d<DD/8;++d) for (uint j=0;j<BK/8;++j) {
            simdgroup_matrix<T,8,8> Df,Kf;
            simdgroup_load(Df,Ds+hg*8*SP+j*8,SP);
            simdgroup_load(Kf,Ks+j*8*(D+8)+col+d*8,D+8);
            simdgroup_multiply_accumulate(DQ[d],Df,Kf,DQ[d]);
        }
        // Each SIMD group handles eight key rows. A 32-key tile is drained in
        // two 16-row stripes: keeping only 16 rows of tile_grad avoids crossing
        // Metal's 32 KiB limit (D=128, BK=32, NP=2: 32,608 bytes).
        for (uint key_base=0;key_base<BK;key_base+=16) {
            for (uint d=0;d<DD/8;++d) {
                simdgroup_matrix<float,8,8> DK=make_filled_simdgroup_matrix<float,8,8>(0.0f);
                for (uint h=0;h<2;++h) {
                    simdgroup_matrix<T,8,8> Df,Pf,Q,G;
                    simdgroup_load(Df,Ds+h*8*SP+key_base+hg*8,SP,ulong2(0,0),true);
                    simdgroup_load(Pf,Ps+h*8*SP+key_base+hg*8,SP,ulong2(0,0),true);
                    simdgroup_load(Q,q+((size_t)query*H+h*8)*D+col+d*8,D);
                    simdgroup_load(G,g+((size_t)query*H+h*8)*D+col+d*8,D);
                    simdgroup_multiply_accumulate(DK,Df,Q,DK);
                    simdgroup_multiply_accumulate(DK,Pf,G,DK);
                }
                simdgroup_store(DK,tile_grad+hg*8*(D+4)+col+d*8,D+4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tid;i<16*D;i+=NT) {
                uint r=i/D,d=i%D;
                int key=ids[key_base+r];
                float value=tile_grad[r*(D+4)+d];
                if (key>=0 && uint(key)<TQ)
                    atomic_fetch_add_explicit(dwindow+(((size_t)shard*B+b)*TQ+uint(key))*D+d,value,memory_order_relaxed);
                else if (key>=0)
                    atomic_fetch_add_explicit(dcompressed+(((size_t)shard*B+b)*NC+uint(key)-TQ)*D+d,value,memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // 最后一个 tile 的 atomic 读者必须先结束，之后 tile_grad 才能复用为 dQ store。
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d=0;d<DD/8;++d) simdgroup_store(DQ[d],tile_grad+hg*8*(D+4)+col+d*8,D+4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<H*D;i+=NT)
        atomic_store_explicit(dq+(size_t)query*H*D+i,tile_grad[(i/D)*(D+4)+i%D],memory_order_relaxed);
    if (tid<H)
        atomic_store_explicit(dsink+query*H+tid,-exp(sinks[tid]-lse[query*H+tid])*delta[tid],memory_order_relaxed);
"""


@lru_cache(None)
def _kernels():
    compact = mx.fast.metal_kernel(
        name="mqa_compact_visible", input_names=["visible", "dims"],
        output_names=["indices", "lengths"], source=_COMPACT,
    )
    inputs = ["q", "window", "compressed", "indices", "lengths", "segment", "pad", "sinks", "dims", "scale"]
    fwd = mx.fast.metal_kernel(
        name="mqa_indexed_flash_fwd", input_names=inputs, output_names=["out", "lse"],
        source=_FWD.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
    )
    legacy = mx.fast.metal_kernel(
        name="mqa_indexed_flash_bwd", input_names=inputs+["g", "out", "lse"],
        output_names=["dq", "dwindow", "dcompressed", "dsink"],
        source=_BWD.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
        atomic_outputs=True,
    )
    delta = mx.fast.metal_kernel(
        name="mqa_indexed_flash_delta_sink",
        input_names=["g", "out", "sinks", "lse"],
        output_names=["delta", "dsinkrow"], source=_DELTA_SINK, header=_HEADER,
    )
    q_sink = mx.fast.metal_kernel(
        name="mqa_indexed_flash_bwd_q_sink",
        input_names=inputs+["g", "lse", "out"],
        output_names=["dq", "delta", "dsinkrow"],
        source=_BWD_Q_SINK.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
    )
    kv = mx.fast.metal_kernel(
        name="mqa_indexed_flash_bwd_kv",
        input_names=inputs+["g", "lse", "deltaglobal"],
        output_names=["dpoolshard"],
        source=_BWD_KV.replace("/*LOAD_TILE*/", _LOAD_TILE), header=_HEADER,
        atomic_outputs=True,
    )
    pool = mx.fast.metal_kernel(
        name="mqa_finalize_pool_grad",
        input_names=["poolshard", "dims"],
        output_names=["dwindow", "dcompressed"], source=_FINALIZE_POOL,
        header=_HEADER,
    )
    sink = mx.fast.metal_kernel(
        name="mqa_finalize_sink_grad",
        input_names=["dsinkrow", "dims"],
        output_names=["dsink"], source=_FINALIZE_SINK,
        header=_HEADER,
    )
    return compact, fwd, legacy, delta, q_sink, kv, pool, sink


@lru_cache(None)
def _key_owned_kernels():
    csr_count = mx.fast.metal_kernel(
        name="mqa_count_compressed_edges",
        input_names=["indices", "lengths", "dims"],
        output_names=["counts"], source=_CSR_COUNT, header=_HEADER,
        atomic_outputs=True,
    )
    scan_block = mx.fast.metal_kernel(
        name="mqa_csr_scan_block", input_names=["counts", "dims"],
        output_names=["local_prefix", "block_total"], source=_CSR_SCAN_BLOCK,
        header=_HEADER,
    )
    scan_add = mx.fast.metal_kernel(
        name="mqa_csr_scan_add", input_names=["local_prefix", "block_prefix", "dims"],
        output_names=["row_ptr"], source=_CSR_SCAN_ADD, header=_HEADER,
    )
    csr_fill = mx.fast.metal_kernel(
        name="mqa_fill_compressed_edges",
        input_names=["indices", "lengths", "row_ptr", "dims"],
        output_names=["cursor", "edge_q", "edge_slot"], source=_CSR_FILL,
        header=_HEADER, atomic_outputs=True,
    )
    one_key = mx.fast.metal_kernel(
        name="mqa_backward_one_key",
        input_names=["Q", "window", "compressed", "LSE", "dO", "Delta",
                     "row_ptr", "edge_q", "edge_slot", "segment", "pad",
                     "dims", "scale"],
        output_names=["dKV"],
        source=_ONE_KEY, header=_HEADER,
    )
    reduce_parts = mx.fast.metal_kernel(
        name="mqa_key_reduce_parts", input_names=["partial", "dims"],
        output_names=["dcompressed"], source=_KEY_REDUCE_PARTS, header=_HEADER,
    )
    return csr_count, scan_block, scan_add, csr_fill, one_key, reduce_parts


def _csr_exclusive_scan(counts):
    """Return int32 [exclusive prefixes, total], with no host read of GPU data."""
    n = counts.size
    if n == 0:
        return mx.zeros((1,), mx.int32)
    _, scan_block, scan_add, _, _, _ = _key_owned_kernels()
    blocks = (n + 255) // 256
    dims = mx.array([n], mx.uint32)
    local, totals = scan_block(
        inputs=[counts, dims], grid=(blocks * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n + 1,), (blocks,)], output_dtypes=[mx.int32, mx.int32],
    )
    if blocks == 1:
        return local
    block_prefix = _csr_exclusive_scan(totals)
    (row_ptr,) = scan_add(
        inputs=[local, block_prefix, dims],
        grid=(n + 1, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n + 1,)], output_dtypes=[mx.int32],
    )
    return row_ptr


def _compressed_occurrence_csr(indices, lengths, b, t, n):
    """Invert the existing row-major selection, preserving each (query, slot)."""
    if n == 0:
        return mx.zeros((1,), mx.int32), mx.zeros((0,), mx.int32), mx.zeros((0,), mx.int32)
    # This is the selection's static storage capacity, including possible ties,
    # not the nominal top-k. Unused capacity is never read by the consumer.
    edge_cap = indices.size
    if edge_cap > 2**31 - 1 or b * n >= 2**31:
        raise ValueError("compressed occurrence CSR exceeds int32 capacity")
    csr_count, _, _, csr_fill, _, _ = _key_owned_kernels()
    dims = mx.array([b, t, n], mx.uint32)
    (counts,) = csr_count(
        inputs=[indices, lengths, dims],
        grid=(128, b * t, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b * n,)], output_dtypes=[mx.int32], init_value=0,
    )
    row_ptr = _csr_exclusive_scan(counts)
    _, edge_q, edge_slot = csr_fill(
        inputs=[indices, lengths, row_ptr, dims],
        grid=(128, b * t, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b * n,), (edge_cap,), (edge_cap,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32], init_value=0,
    )
    return row_ptr, edge_q, edge_slot


def _key_owned_pool_grad(q, window, compressed, indices, lengths, segment, pad,
                         dims, scale, gcast, lse, deltas, window_size):
    """Implicit window adjacency + occurrence CSR; no floating-point atomics."""
    b, t, h, d = q.shape
    n = compressed.shape[1]
    _, _, _, _, one_key, reduce_parts = _key_owned_kernels()
    row_ptr, edge_q, edge_slot = _compressed_occurrence_csr(indices, lengths, b, t, n)
    # The wrapper's transpose has already been differentiated: gcast is BTHD.
    inputs = [q, window, compressed, lse, gcast, deltas, row_ptr, edge_q, edge_slot,
              segment, pad, dims, scale]
    template = [("T", q.dtype), ("D", d), ("H", h), ("WIN", window_size)]
    (dw,) = one_key(
        inputs=inputs, template=template + [("COMPRESSED", False), ("OutT", window.dtype)],
        grid=(128, b * t, 1), threadgroup=(128, 1, 1),
        output_shapes=[window.shape], output_dtypes=[window.dtype],
    )
    if n:
        (partial,) = one_key(
            inputs=inputs, template=template + [("COMPRESSED", True), ("OutT", mx.float32)],
            grid=(128, b * n, 4), threadgroup=(128, 1, 1),
            output_shapes=[(4, b, n, d)], output_dtypes=[mx.float32],
        )
        (dc,) = reduce_parts(
            inputs=[partial, mx.array([b * n * d], mx.uint32)],
            template=[("T", compressed.dtype)],
            grid=(b * n * d, 1, 1), threadgroup=(128, 1, 1),
            output_shapes=[compressed.shape], output_dtypes=[compressed.dtype],
        )
    else:
        dc = mx.zeros((b, 0, d), compressed.dtype)
    return dw, dc


def enabled_for(q, window_size):
    return (_ENABLED and mx.default_device() == mx.gpu and q.shape[-2] == 16
            and q.shape[-1] in (64, 128) and window_size > 0
            and q.dtype in (mx.bfloat16, mx.float16))


def _compact_mask(mask):
    b, t, n = mask.shape
    return _kernels()[0](
        inputs=[mask, mx.array([n], mx.uint32)],
        grid=(32, b*t, 1), threadgroup=(32, 1, 1),
        output_shapes=[(b*t, n), (b*t,)], output_dtypes=[mx.int32, mx.int32],
    )


def compact_visible(mask):
    """把 [B,T,N] 布尔可见性压成 indexed_attention 用的 (indices, lengths)。

    CED 解码段 N=T：源层（Full/Reindex）算完 keep 后压一次，Reuse 层直接
    拿这份 metadata，不再每层扫一遍 [B,T,T]。
    """
    return _compact_mask(mx.stop_gradient(mask))


# Forward BK=32 is supported, but backward always uses BK=16 to stay inside
# Metal's 32 KiB shared-memory limit. NP=D/64 for both supported dimensions.


@lru_cache(None)
def _operation(window_size, softmax_scale, key_tile):
    if key_tile not in (16, 32):
        raise ValueError("forward key tile must be 16 or 32")
    (compact, forward, legacy, delta_k, q_sink, kv, finalize_pool,
     finalize_sink) = _kernels()

    def constants(q, compressed):
        b, t, _, d = q.shape
        return mx.array([b, t, compressed.shape[1]], mx.uint32), mx.array([softmax_scale], mx.float32)

    @mx.custom_function
    def op(q, window, compressed, indices, lengths, segment, pad, sinks):
        b, t, h, d = q.shape
        parts, threads = d // 64, d
        dims, scale = constants(q, compressed)
        out, lse = forward(
            inputs=[q, window, compressed, indices, lengths, segment, pad, sinks, dims, scale],
            template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", key_tile), ("NP", parts)],
            grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
            output_shapes=[q.shape, (b, t, h)], output_dtypes=[mx.float32, mx.float32],
        )
        return out, lse

    @op.vjp
    def vjp(primals, cotangent, output):
        q, window, compressed, indices, lengths, segment, pad, sinks = primals
        g, _ = cotangent  # LSE is internal state and is never exposed by the wrapper.
        out, lse = output
        b, t, h, d = q.shape
        parts, threads = d // 64, d
        backward_tile = 16
        dims, scale = constants(q, compressed)
        if _SPLIT_BWD:
            (compact, forward, legacy, delta_k, q_sink, kv, finalize_pool,
             finalize_sink) = _kernels()
            gcast = g.astype(q.dtype)
            dq, deltas, dsinkrow = q_sink(
                inputs=[q, window, compressed, indices, lengths, segment, pad, sinks,
                        dims, scale, gcast, lse, out],
                template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", backward_tile), ("NP", parts)],
                grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
                output_shapes=[q.shape, (b, t, h), (b, t, h)],
                output_dtypes=[q.dtype, mx.float32, mx.float32],
            )
            (ds,) = finalize_sink(
                inputs=[dsinkrow, mx.array([t, b], mx.uint32)],
                template=[("H", h)],
                grid=(128, h, 1), threadgroup=(128, 1, 1),
                output_shapes=[sinks.shape], output_dtypes=[sinks.dtype],
            )
            n_comp = compressed.shape[1]
            if _KEY_OWNED_BWD:
                dw, dc = _key_owned_pool_grad(
                    q, window, compressed, indices, lengths, segment, pad,
                    dims, scale, gcast, lse, deltas, window_size)
            else:
                (poolshard,) = kv(
                    inputs=[q, window, compressed, indices, lengths, segment, pad, sinks,
                            dims, scale, gcast, lse, deltas],
                    template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", backward_tile),
                              ("NP", parts), ("SHARDS", _SHARDS)],
                    grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
                    output_shapes=[(_SHARDS,) + (b, t + n_comp, d)],
                    output_dtypes=[mx.float32], init_value=0,
                )
                dw, dc = finalize_pool(
                    inputs=[poolshard, mx.array([t, n_comp, b], mx.uint32)],
                    template=[("T", window.dtype), ("D", d), ("SHARDS", _SHARDS)],
                    grid=(d, t + n_comp, b), threadgroup=(d, 1, 1),
                    output_shapes=[window.shape, compressed.shape],
                    output_dtypes=[window.dtype, compressed.dtype],
                )
        else:
            (compact, forward, legacy, delta_k, q_sink, kv, finalize_pool,
             finalize_sink) = _kernels()
            dq, dw, dc, ds = legacy(
                inputs=[q, window, compressed, indices, lengths, segment, pad, sinks,
                        dims, scale, g.astype(q.dtype), out, lse],
                template=[("T", q.dtype), ("D", d), ("W", window_size), ("BK", backward_tile),
                          ("NP", parts), ("SHARDS", _SHARDS)],
                grid=(threads, b*t, 1), threadgroup=(threads, 1, 1),
                output_shapes=[q.shape, (_SHARDS,)+window.shape,
                               (_SHARDS,)+compressed.shape, (b, t, h)],
                output_dtypes=[mx.float32]*4, init_value=0,
            )
            dw = mx.sum(dw, axis=0)
            dc = mx.sum(dc, axis=0)
            ds = mx.sum(ds, axis=(0, 1))
        # MLX 0.32.2 flattens a custom VJP with tree_flatten(..., strict=False):
        # None leaves disappear rather than reserving argument positions.
        # CustomTransforms::vjp then indexes that vector with the ORIGINAL
        # argnums. In particular sinks is argument 7; four None leaves here
        # shrink eight input VJPs to four and make its lookup out of bounds.
        # Return one array per primal, even for nondifferentiable metadata.
        # These zero leaves are discarded when MLX selects trainable argnums.
        return (
            dq.astype(q.dtype),
            dw.astype(window.dtype),
            dc.astype(compressed.dtype),
            mx.zeros_like(indices),
            mx.zeros_like(lengths),
            mx.zeros_like(segment),
            mx.zeros_like(pad),
            ds.astype(sinks.dtype),
        )

    return op


def indexed_attention(q, window, compressed, visible, segment_ids, pad_mask, sinks,
                      window_size, softmax_scale, selection=None, key_tile=None):
    """Returns [B,H,T,D]. Caller selects the supported no-cache training path."""
    b, t, h, d = q.shape
    if selection is None:
        visible = mx.broadcast_to(visible, (b, t, compressed.shape[1]))
        indices, lengths = _compact_mask(mx.stop_gradient(visible))
    else:
        indices, lengths = selection
    segment = (mx.zeros((b, t), mx.int32) if segment_ids is None
               else segment_ids.astype(mx.int32))
    pad = (mx.ones((b, t), mx.bool_) if pad_mask is None else pad_mask.astype(mx.bool_))
    out, _ = _operation(window_size, softmax_scale, _KEY_TILE if key_tile is None else key_tile)(
        q, window.astype(q.dtype), compressed.astype(q.dtype), indices, lengths,
        segment, pad, sinks.astype(mx.float32),
    )
    return out.astype(q.dtype).transpose(0, 2, 1, 3)


@lru_cache(None)
def prewarm_sparse_attention(head_dim, window_size, softmax_scale, dtype=mx.bfloat16, key_tile=None):
    """Materialize JIT libraries eagerly; no correctness or performance checks."""
    q = mx.zeros((1, 1, 16, head_dim), dtype)
    if not enabled_for(q, window_size):
        return
    kv = mx.zeros((1, 1, head_dim), dtype)
    mask = mx.ones((1, 1, 1), mx.bool_)
    sinks = mx.zeros((16,), mx.float32)

    def fn(a, b, c, s):
        return indexed_attention(a, b, c, mask, None, None, s, window_size, softmax_scale, key_tile=key_tile)

    out, grads = mx.vjp(fn, [q, kv, kv, sinks], [mx.ones((1, 16, 1, head_dim), dtype)])
    mx.eval(out, grads)
