"""MLA 整层 decode 融合 Metal kernel（仅推理前向，B=T=1）。

一条 prenorm 层：GatedNorm → MLA（QKV / RoPE / SDPA / gate / o_proj）→
AttnRes → GatedNorm → Latent-MoE + shared → AttnRes，收成 1 个 kernel。
新 K/V 写回输出，由调用方 ``cache.update``；不在核内改 KV 池。

适用范围见 ``try_layer_decode``。JIT / 在线校验失败按 key 回退 eager，
不把别的形状株连进全局禁用（与 attn_res 相同）；Metal 编译本身失败
才置 ``_DISABLED``。
"""

from __future__ import annotations

import math
import os
import time

import mlx.core as mx

from ..cache import KVCache

_METAL_TYPE = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "float16_t",
    mx.float32: "float",
}
_KERNELS: dict = {}
_VERIFIED: set = set()
_FAILED: set = set()
_DISABLED = os.environ.get("VIBY_LAYER_DECODE", "1") == "0"

_NT_SMALL = 256
_NT_LARGE = 1024
_TG_LIMIT = 30720
_ATTN_RES_EPS = 1e-6


def _nt_for(D, E, inter):
    return _NT_LARGE if max(D, E, inter) >= 256 else _NT_SMALL


def _tg_bytes(D, tmpn, wide, qk, hd, inter, nt):
    return (D + tmpn + wide + D + nt // 32 + qk + qk + hd + inter + hd) * 4 + 128


def _pack_gn(gn, dtype):
    wn = gn.norm.weight.astype(dtype)
    wd = gn.gate_down.astype(dtype)
    wu = gn.gate_up.astype(dtype)
    return mx.concatenate([wn, wd.reshape(-1), wu.reshape(-1)])


def _rope_cs(position_embeddings, T, dtype, attn, cache):
    if (
        position_embeddings is not None
        and len(position_embeddings) > 2
        and position_embeddings[2] is not None
    ):
        freqs, offset, af = position_embeddings[2]
        pos = mx.arange(int(offset), int(offset) + int(T))
        angles = pos[:, None].astype(mx.float32) / freqs.astype(mx.float32)
        c, s = mx.cos(angles), mx.sin(angles)
        af = float(af)
        c = mx.concatenate([c, c], axis=-1) * af
        s = mx.concatenate([s, s], axis=-1) * af
        return c.astype(dtype), s.astype(dtype)
    if position_embeddings is not None:
        return (
            position_embeddings[0].astype(dtype),
            position_embeddings[1].astype(dtype),
        )
    start = int(getattr(cache, "offset", 0) or 0)
    return _rope_cs(attn._fallback_pos(start, T, dtype), T, dtype, attn, cache)


def _shared_w(shared, dtype):
    gus, dws = [], []
    for ff in shared:
        srcs = (ff.gate_proj.weight, ff.up_proj.weight)
        c = ff.__dict__.get("_gu_w_cache")
        if c is not None and c[1] is srcs[0] and c[2] is srcs[1]:
            gu = c[0]
        else:
            gu = mx.concatenate(srcs, axis=0)
            if not ff.training:
                object.__setattr__(ff, "_gu_w_cache", (gu, srcs[0], srcs[1]))
        gus.append(gu.astype(dtype))
        dws.append(ff.down_proj.weight.astype(dtype))
    return mx.stack(gus, 0), mx.stack(dws, 0)


def _pack_residuals(residuals, window, D, dtype):
    n = min(len(residuals), max(int(window) - 1, 0))
    if n <= 0:
        return mx.zeros((1, D), dtype=dtype), 0
    rows = [r.reshape(-1)[-D:].astype(dtype) for r in residuals[-n:]]
    return mx.stack(rows, 0), n


def _build(
    D,
    H,
    HD,
    RD,
    KR,
    RANK,
    I,  # noqa: E741
    E,
    TOPK,
    DE,
    S,
    NIN,
    WIN,
    has_gate,
    has_lat,
    has_lnorm,
    logit_temp,
    norm_topk,
    scaling,
    eps,
    dtype,
    NT,
):
    QK = HD + RD
    QKV = H * QK + KR + RD
    KVUP = 2 * H * HD
    TMPN = max(D, H * HD, KR + RD)
    WIDE = max(D, QKV, KVUP, E, I, RANK, H * QK)
    key = (
        D,
        H,
        HD,
        RD,
        KR,
        RANK,
        I,
        E,
        TOPK,
        DE,
        S,
        NIN,
        WIN,
        has_gate,
        has_lat,
        has_lnorm,
        logit_temp,
        norm_topk,
        scaling,
        eps,
        dtype,
        NT,
    )
    kern = _KERNELS.get(key)
    if kern is not None:
        return kern
    mt = _METAL_TYPE[dtype]
    scale = 1.0 / math.sqrt(QK)
    header = f"""
        constexpr uint D = {D};
        constexpr uint H = {H};
        constexpr uint HD = {HD};
        constexpr uint RD = {RD};
        constexpr uint QK = {QK};
        constexpr uint KR = {KR};
        constexpr uint RANK = {RANK};
        constexpr uint I = {I};
        constexpr uint E = {E};
        constexpr uint TOPK = {TOPK};
        constexpr uint DE = {DE};
        constexpr uint S = {S};
        constexpr uint NIN = {NIN};
        constexpr uint WIN = {WIN};
        constexpr uint HAS_GATE = {int(has_gate)};
        constexpr uint HAS_LAT = {int(has_lat)};
        constexpr uint HAS_LNORM = {int(has_lnorm)};
        constexpr uint NORM_TOPK = {int(norm_topk)};
        constexpr uint NT = {NT};
        constexpr uint QKV = {QKV};
        constexpr uint TMPN = {TMPN};
        constexpr uint WIDE = {WIDE};
        constexpr uint CA = (WIDE / HD < 16u) ? (WIDE / HD) : 16u;
        // split-K 块数 CA 可大于 simd 组数 NT/32；red/red2 必须按两者取大，
        // 否则宽 QKV（大 kv_lora_rank）会写爆 threadgroup、past attention 出错。
        constexpr uint NRED = (NT / 32 > CA) ? (NT / 32) : CA;
        constexpr float EPS = {float(eps)}f;
        constexpr float AR_EPS = {_ATTN_RES_EPS}f;
        constexpr float SCALE = {scale}f;
        constexpr float SCALING = {float(scaling)}f;
        constexpr float LOGIT_TEMP = {float(logit_temp)}f;
        #define MT {mt}
"""
    body = r"""
        uint tid = thread_position_in_threadgroup.x;
        threadgroup float xsh[D];
        threadgroup float tmp[TMPN];
        threadgroup float wide[WIDE];
        threadgroup float vat[D];
        threadgroup float attn_acc[CA * HD];
        threadgroup float red[NRED];
        threadgroup float red2[NRED];
        threadgroup float qh[QK];
        threadgroup float kh[QK];
        threadgroup float vh[HD];
        threadgroup float eh[I];
        threadgroup uint topk[TOPK];
        threadgroup float topw[TOPK];
        threadgroup float alpha[8];
        uint Tp = uint(meta[0]);

        for (uint i = tid; i < D; i += NT) xsh[i] = float(x[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

#define REDUCE_TO_RED0(ACC) \
        ACC = simd_sum(ACC); \
        if ((tid & 31u) == 0u) red[tid >> 5] = ACC; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (tid == 0) { \
            float s = 0.0f; \
            for (uint zi = 0; zi < NT / 32; zi++) s += red[zi]; \
            red[0] = s; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup);

#define GATED_NORM(LN) \
        { \
            float acc = 0.0f; \
            for (uint i = tid; i < D; i += NT) { \
                float t = xsh[i]; \
                acc += t * t; \
            } \
            REDUCE_TO_RED0(acc); \
            if (tid == 0) red[0] = metal::rsqrt(red[0] / float(D) + EPS); \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            float rms = red[0]; \
            for (uint i = tid; i < D; i += NT) { \
                tmp[i] = float((MT)(xsh[i] * rms * float(LN[i]))); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (tid < RANK) { \
                float a2 = 0.0f; \
                for (uint d = 0; d < D; d++) { \
                    a2 += tmp[d] * float(LN[D + d * RANK + tid]); \
                } \
                float hb = float((MT)a2); \
                float sg = 1.0f / (1.0f + metal::exp(-hb)); \
                wide[tid] = float((MT)(hb * sg)); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint i = tid; i < D; i += NT) { \
                float a3 = 0.0f; \
                for (uint j = 0; j < RANK; j++) { \
                    a3 += wide[j] * float(LN[D + D * RANK + j * D + i]); \
                } \
                float gb = float((MT)a3); \
                float g2 = float((MT)(2.0f / (1.0f + metal::exp(-gb)))); \
                xsh[i] = float((MT)(tmp[i] * g2)); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        }

#define DOT4_TG_DEV(X, W, WBASE, INN, ACC) \
        { \
            uint n4 = (INN) >> 2; \
            for (uint k = 0; k < n4; k++) { \
                uint kb = k << 2; \
                float4 xv = *((const threadgroup float4*)((X) + kb)); \
                ACC += xv.x * float((W)[(WBASE) + kb]) \
                    + xv.y * float((W)[(WBASE) + kb + 1]) \
                    + xv.z * float((W)[(WBASE) + kb + 2]) \
                    + xv.w * float((W)[(WBASE) + kb + 3]); \
            } \
            for (uint k = n4 << 2; k < (INN); k++) { \
                ACC += (X)[k] * float((W)[(WBASE) + k]); \
            } \
        }

#define GEMV_TG_DEV(X, W, OUTN, INN, DST) \
        for (uint o = tid; o < (OUTN); o += NT) { \
            float acc = 0.0f; \
            DOT4_TG_DEV(X, W, (size_t)o * (INN), INN, acc); \
            DST[o] = acc; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup);

        GATED_NORM(ln0);

        GEMV_TG_DEV(xsh, qkv_w, QKV, D, wide);

        for (uint i = tid; i < KR; i += NT) tmp[i] = wide[H * QK + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < RD; i += NT) {
            float xi = wide[H * QK + KR + i];
            float yi = (i < RD / 2) ? -wide[H * QK + KR + i + RD / 2]
                                    : wide[H * QK + KR + i - RD / 2];
            tmp[KR + i] = xi * float(rope[i]) + yi * float(rope[RD + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h = 0; h < H; h++) {
            {
                float acc = 0.0f;
                for (uint i = tid; i < HD; i += NT) {
                    float t = wide[h * QK + i];
                    acc += t * t;
                }
                REDUCE_TO_RED0(acc);
                if (tid == 0) red[0] = metal::rsqrt(red[0] / float(HD) + EPS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float rms = red[0];
                for (uint i = tid; i < HD; i += NT) {
                    qh[i] = float((MT)(wide[h * QK + i] * rms * float(qn_kn[i])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < RD; i += NT) {
                float xi = wide[h * QK + HD + i];
                float yi = (i < RD / 2) ? -wide[h * QK + HD + i + RD / 2]
                                        : wide[h * QK + HD + i - RD / 2];
                qh[HD + i] = xi * float(rope[i]) + yi * float(rope[RD + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < HD; i += NT) {
                float acc = 0.0f;
                size_t wb = ((size_t)h * HD + i) * KR;
                for (uint r = 0; r < KR; r++) acc += tmp[r] * float(kv_up_w[wb + r]);
                kh[i] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                float acc = 0.0f;
                for (uint i = tid; i < HD; i += NT) acc += kh[i] * kh[i];
                REDUCE_TO_RED0(acc);
                if (tid == 0) red[0] = metal::rsqrt(red[0] / float(HD) + EPS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float rms = red[0];
                for (uint i = tid; i < HD; i += NT) {
                    kh[i] = float((MT)(kh[i] * rms * float(qn_kn[HD + i])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < RD; i += NT) kh[HD + i] = tmp[KR + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < HD; i += NT) {
                float acc = 0.0f;
                size_t wb = ((size_t)H * HD + (size_t)h * HD + i) * KR;
                for (uint r = 0; r < KR; r++) acc += tmp[r] * float(kv_up_w[wb + r]);
                vh[i] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < QK; i += NT) {
                nk[(size_t)h * QK + i] = (MT)kh[i];
            }
            for (uint i = tid; i < HD; i += NT) {
                nv[(size_t)h * HD + i] = (MT)vh[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- flash-2 style split-K attention over the KV cache ----
            // CA chunk-threads each run an online softmax (running max m,
            // running sum l, running weighted-V accumulator acc[HD], stored in
            // the free wide[] region) over a contiguous block of past keys.
            // The current token (kh/vh) seeds chunk 0. This turns the old
            // O(Tp) per-token serial barriers into ~2 barriers + O(Tp/CA) work
            // per chunk, so wall time no longer scales as O(Tp) with barriers.
            if (tid == 0) {
                for (uint c = 0; c < CA; c++) { red[c] = -1.0e30f; red2[c] = 0.0f; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < CA) {
                uint c = tid;
                float m_c = -1.0e30f, l_c = 0.0f;
                for (uint d = 0; d < HD; d++) attn_acc[c * HD + d] = 0.0f;
                if (c == 0) {
                    float s = 0.0f;
                    for (uint d = 0; d < QK; d++) s += qh[d] * kh[d];
                    s *= SCALE;
                    m_c = s;
                    l_c = 1.0f;
                    for (uint d = 0; d < HD; d++) attn_acc[d] = vh[d];
                }
                uint bs = (Tp + CA - 1) / CA;
                uint start = c * bs;
                uint end = metal::min((c + 1) * bs, Tp);
                for (uint t = start; t < end; t++) {
                    float s = 0.0f;
                    size_t kb = ((size_t)h * Tp + t) * QK;
                    for (uint d = 0; d < QK; d++) s += qh[d] * float(k_cache[kb + d]);
                    s *= SCALE;
                    float om = m_c;
                    m_c = metal::max(m_c, s);
                    float scale = metal::exp(om - m_c);
                    float e = metal::exp(s - m_c);
                    l_c = l_c * scale + e;
                    size_t vb = ((size_t)h * Tp + t) * HD;
                    for (uint d = 0; d < HD; d++) {
                        attn_acc[c * HD + d] = attn_acc[c * HD + d] * scale + e * float(v_cache[vb + d]);
                    }
                }
                red[c] = m_c;
                red2[c] = l_c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float M = red[0];
                for (uint c = 1; c < CA; c++) M = metal::max(M, red[c]);
                for (uint c = 0; c < CA; c++) eh[c] = metal::exp(red[c] - M);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < HD; i += NT) {
                float acc = 0.0f, L = 0.0f;
                for (uint c = 0; c < CA; c++) {
                    L += red2[c] * eh[c];
                    acc += attn_acc[c * HD + i] * eh[c];
                }
                vat[h * HD + i] = (L > 0.0f) ? (acc / L) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (HAS_GATE) {
            for (uint h = tid; h < H; h += NT) {
                float acc = float(ag[H * D + h]);
                size_t wb = (size_t)h * D;
                for (uint d = 0; d < D; d++) acc += xsh[d] * float(ag[wb + d]);
                float g = float((MT)(2.0f / (1.0f + metal::exp(-acc))));
                wide[h] = g;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = 0; h < H; h++) {
                float g = wide[h];
                for (uint i = tid; i < HD; i += NT) vat[h * HD + i] *= g;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        GEMV_TG_DEV(vat, o_w, D, H * HD, wide);
        for (uint i = tid; i < D; i += NT) {
            vat[i] = wide[i];
            v_attn[i] = (MT)wide[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#define LOAD_RES(J, DIDX) \
        (((J) < NIN) ? float(res_in[(size_t)(J) * D + (DIDX)]) \
                     : (((J) == NIN) ? vat[DIDX] : wide[DIDX]))

#define ATTN_RES_MERGE(WOFF, J0, J1, DST) \
        { \
            for (uint j = (J0); j < (J1); j++) { \
                float acc = 0.0f, ss = 0.0f; \
                for (uint d = tid; d < D; d += NT) { \
                    float vd = LOAD_RES(j, d); \
                    acc += float(rq[(WOFF) + d]) * vd; \
                    ss += vd * vd; \
                } \
                acc = simd_sum(acc); \
                ss = simd_sum(ss); \
                if ((tid & 31u) == 0u) { red[tid >> 5] = acc; red2[tid >> 5] = ss; } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                if (tid == 0) { \
                    float t = 0.0f, t2 = 0.0f; \
                    for (uint zi = 0; zi < NT / 32; zi++) { \
                        t += red[zi]; t2 += red2[zi]; \
                    } \
                    alpha[j] = t * metal::rsqrt(t2 / float(D) + AR_EPS); \
                } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            if (tid == 0) { \
                float m = alpha[J0]; \
                for (uint j = (J0) + 1; j < (J1); j++) m = metal::max(m, alpha[j]); \
                float sm = 0.0f; \
                for (uint j = (J0); j < (J1); j++) { \
                    alpha[j] = metal::exp(alpha[j] - m); \
                    sm += alpha[j]; \
                } \
                float inv = 1.0f / sm; \
                for (uint j = (J0); j < (J1); j++) alpha[j] *= inv; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint d = tid; d < D; d += NT) { \
                float acc = 0.0f; \
                for (uint j = (J0); j < (J1); j++) acc += alpha[j] * LOAD_RES(j, d); \
                DST[d] = acc; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        }

        {
            uint j1 = NIN + 1;
            uint j0 = (WIN > 0 && j1 > WIN) ? j1 - WIN : 0;
            ATTN_RES_MERGE(0, j0, j1, xsh);
        }

        GATED_NORM(ln1);

        GEMV_TG_DEV(xsh, router_w, E, D, wide);
        if (HAS_LNORM) {
            float acc = 0.0f;
            for (uint e = tid; e < E; e += NT) acc += wide[e];
            REDUCE_TO_RED0(acc);
            if (tid == 0) red[0] = red[0] / float(E);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mean = red[0];
            for (uint e = tid; e < E; e += NT) wide[e] -= mean;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = 0.0f;
            for (uint e = tid; e < E; e += NT) acc += wide[e] * wide[e];
            REDUCE_TO_RED0(acc);
            if (tid == 0) {
                red[0] = metal::rsqrt(red[0] / float(E) + 1.0e-6f) * LOGIT_TEMP;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sc = red[0];
            for (uint e = tid; e < E; e += NT) wide[e] *= sc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint e = tid; e < E; e += NT) {
            wide[e] = 1.0f / (1.0f + metal::exp(-wide[e]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- O(E*TOPK) single-pass selection (replaces O(E*K^2) scan) ----
        // wide[e] = raw sigmoid score; selection value = score + bias.
        // Thread 0 keeps a descending top-TOPK of (sel, idx); ties prefer the
        // lower index. The quadratic "used" inner loop is gone entirely, so at
        // E=384/TOPK=6 this is ~384*6 comparisons instead of ~384*6*6/2.
        if (tid == 0) {
            for (uint q = 0; q < TOPK; q++) { topk[q] = E; topw[q] = -1.0e30f; }
            for (uint e = 0; e < E; e++) {
                float sel = wide[e] + float(router_b[e]);
                uint pos = TOPK;
                for (uint i = 0; i < TOPK; i++) {
                    if (sel > topw[i] || (sel == topw[i] && e < topk[i])) { pos = i; break; }
                }
                if (pos < TOPK) {
                    for (uint i = TOPK - 1; i > pos; i--) {
                        topw[i] = topw[i - 1];
                        topk[i] = topk[i - 1];
                    }
                    topw[pos] = sel;
                    topk[pos] = e;
                }
            }
            for (uint k = 0; k < TOPK; k++) {
                topw[k] = (topk[k] < E) ? wide[topk[k]] : 0.0f;
            }
            if (NORM_TOPK && TOPK > 1) {
                float sm = 0.0f;
                for (uint k = 0; k < TOPK; k++) sm += topw[k];
                sm = metal::max(sm, 1.0e-9f);
                for (uint k = 0; k < TOPK; k++) topw[k] = topw[k] / sm * SCALING;
            } else {
                for (uint k = 0; k < TOPK; k++) topw[k] *= SCALING;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (HAS_LAT) {
            GEMV_TG_DEV(xsh, lat_down, DE, D, tmp);
            {
                float acc = 0.0f;
                for (uint i = tid; i < DE; i += NT) acc += tmp[i] * tmp[i];
                REDUCE_TO_RED0(acc);
                if (tid == 0) red[0] = metal::rsqrt(red[0] / float(DE) + EPS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float rms = red[0];
                for (uint i = tid; i < DE; i += NT) {
                    tmp[i] = float((MT)(tmp[i] * rms * float(lat_n[i])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint i = tid; i < D; i += NT) tmp[i] = xsh[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = tid; i < DE; i += NT) wide[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TOPK; k++) {
            uint e = topk[k];
            float wk = topw[k];
            for (uint j = tid; j < I; j += NT) {
                float g = 0.0f, u = 0.0f;
                size_t gb = ((size_t)e * (2 * I) + j) * DE;
                size_t ub = ((size_t)e * (2 * I) + I + j) * DE;
                DOT4_TG_DEV(tmp, exp_gu, gb, DE, g);
                DOT4_TG_DEV(tmp, exp_gu, ub, DE, u);
                float sg = 1.0f / (1.0f + metal::exp(-g));
                float gate = 4.0f * metal::tanh(g / 4.0f) * sg;
                float up = 25.0f * metal::tanh(u / 25.0f);
                eh[j] = float((MT)(float((MT)(gate * up))));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = tid; d < DE; d += NT) {
                float acc = 0.0f;
                size_t wb = ((size_t)e * DE + d) * I;
                DOT4_TG_DEV(eh, exp_dw, wb, I, acc);
                wide[d] += wk * acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (HAS_LAT) {
            {
                float acc = 0.0f;
                for (uint i = tid; i < DE; i += NT) acc += wide[i] * wide[i];
                REDUCE_TO_RED0(acc);
                if (tid == 0) red[0] = metal::rsqrt(red[0] / float(DE) + EPS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float rms = red[0];
                for (uint i = tid; i < DE; i += NT) {
                    tmp[i] = float((MT)(wide[i] * rms * float(lat_n[DE + i])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            GEMV_TG_DEV(tmp, lat_up, D, DE, wide);
        }

        for (uint s = 0; s < S; s++) {
            for (uint j = tid; j < I; j += NT) {
                float g = 0.0f, u = 0.0f;
                size_t gb = ((size_t)s * (2 * I) + j) * D;
                size_t ub = ((size_t)s * (2 * I) + I + j) * D;
                DOT4_TG_DEV(xsh, sh_gu, gb, D, g);
                DOT4_TG_DEV(xsh, sh_gu, ub, D, u);
                float sg = 1.0f / (1.0f + metal::exp(-g));
                float gate = 4.0f * metal::tanh(g / 4.0f) * sg;
                float up = 25.0f * metal::tanh(u / 25.0f);
                eh[j] = float((MT)(float((MT)(gate * up))));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = tid; d < D; d += NT) {
                float acc = 0.0f;
                size_t wb = ((size_t)s * D + d) * I;
                DOT4_TG_DEV(eh, sh_dw, wb, I, acc);
                wide[d] += acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = tid; i < D; i += NT) v_mlp[i] = (MT)wide[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint j1 = NIN + 2;
            uint j0 = (WIN > 0 && j1 > WIN) ? j1 - WIN : 0;
            ATTN_RES_MERGE(D, j0, j1, tmp);
        }
        for (uint i = tid; i < D; i += NT) y[i] = (MT)tmp[i];
"""
    kern = mx.fast.metal_kernel(
        name=(
            f"layer_dec_D{D}H{H}hd{HD}rd{RD}kr{KR}R{RANK}I{I}E{E}K{TOPK}"
            f"DE{DE}S{S}N{NIN}W{WIN}G{int(has_gate)}L{int(has_lat)}"
            f"Ln{int(has_lnorm)}Ntk{int(norm_topk)}nt{NT}_{mt}"
        ),
        input_names=[
            "x",
            "res_in",
            "ln0",
            "ln1",
            "qkv_w",
            "kv_up_w",
            "qn_kn",
            "rope",
            "k_cache",
            "v_cache",
            "o_w",
            "ag",
            "rq",
            "router_w",
            "router_b",
            "lat_down",
            "lat_up",
            "lat_n",
            "exp_gu",
            "exp_dw",
            "sh_gu",
            "sh_dw",
            "meta",
        ],
        output_names=["y", "v_attn", "v_mlp", "nk", "nv"],
        source=header + body,
    )
    _KERNELS[key] = kern
    return kern


def _eager_ref(block, x, incoming, cache, position_embeddings):
    from ..block import _attn_res_merge

    tmp = KVCache()
    off = int(cache.offset)
    if off > 0 and cache.keys is not None:
        tmp.update(
            cache.keys[:, :, :off].transpose(0, 2, 1, 3),
            cache.values[:, :, :off].transpose(0, 2, 1, 3),
        )
    residuals = list(incoming)
    h = block.input_layernorm(x)
    va, tmp = block.self_attn(
        h,
        position_embeddings=position_embeddings,
        past_key_value=tmp,
        use_cache=True,
        mask_is_full=True,
    )
    residuals.append(va)
    h = _attn_res_merge(block.attn_res_q_attn, residuals, block.attn_res_window)
    vm = block.mlp(block.post_attention_layernorm(h))
    residuals.append(vm)
    y = _attn_res_merge(block.attn_res_q_mlp, residuals, block.attn_res_window)
    nk = tmp.keys[:, :, off : off + 1].transpose(0, 2, 1, 3)
    nv = tmp.values[:, :, off : off + 1].transpose(0, 2, 1, 3)
    return y, va, vm, nk, nv


def _close(a, b, tol):
    d = (a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()
    return (d <= tol), d


def _eligible(block, x, cache, attention_mask, causal_bias, mask_is_full, segment_ids):
    if _DISABLED or x.dtype not in _METAL_TYPE:
        return False
    if block.training or block.use_linear_attn:
        return False
    if not isinstance(cache, KVCache):
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 1:
        return False
    if causal_bias is not None or segment_ids is not None:
        return False
    if attention_mask is not None:
        return False
    if mask_is_full is False:
        return False
    if getattr(block, "attn_res_register", False):
        # 含 attn_res_read_h：读侧接线不同，核按替换式 AttnRes 写。
        return False
    if getattr(block, "ihc_streams", 0):
        return False
    if getattr(block, "ngram_conf_gate", False):
        return False
    win = int(block.attn_res_window or 0)
    if win <= 0 or win > 4:
        return False
    gn = block.input_layernorm
    rank = int(gn.gate_down.shape[1])
    if rank > _NT_LARGE:
        return False
    mlp = block.mlp
    if (
        int(getattr(mlp, "n_routed", 0) or 0) <= 0
        or int(getattr(mlp, "top_k", 0) or 0) <= 0
    ):
        return False
    if bool(getattr(mlp.router, "collect_stats", False)):
        return False
    if getattr(mlp, "write_scale", None) is not None:
        return False
    if bool(getattr(mlp, "route_scale", False)):
        return False
    return True


def try_layer_decode(
    block,
    hidden_states,
    residuals,
    cache,
    position_embeddings,
    attention_mask=None,
    causal_bias=None,
    mask_is_full=None,
    segment_ids=None,
):
    """满足 decode 条件时跑融合核并 ``cache.update`` / append residuals。

    成功返回 ``(hidden, cache)``；否则 ``None``（调用方走 eager）。
    """
    global _DISABLED
    if not _eligible(
        block,
        hidden_states,
        cache,
        attention_mask,
        causal_bias,
        mask_is_full,
        segment_ids,
    ):
        return None
    attn = block.self_attn
    mlp = block.mlp
    x = hidden_states
    dtype = x.dtype
    D = int(x.shape[-1])
    H = int(attn.n_heads)
    HD = int(attn.head_dim)
    RD = int(attn.rope_dim)
    KR = int(attn.kv_rank)
    RANK = int(block.input_layernorm.gate_down.shape[1])
    I = int(mlp.moe_in)  # noqa: E741
    E = int(mlp.n_routed)
    TOPK = int(mlp.top_k)
    has_lat = mlp.lat_down is not None
    DE = int(mlp.latent_dim) if has_lat else D
    S = len(mlp.shared)
    WIN = int(block.attn_res_window)
    has_gate = attn.attn_gate is not None
    QK = HD + RD
    QKV = H * QK + KR + RD
    KVUP = 2 * H * HD
    TMPN = max(D, H * HD, KR + RD)
    WIDE = max(D, QKV, KVUP, E, I, RANK, H * QK)
    NT = _nt_for(D, E, I)
    if _tg_bytes(D, TMPN, WIDE, QK, HD, I, NT) > _TG_LIMIT:
        return None
    if block.post_attention_layernorm.gate_down.shape[1] != RANK:
        return None
    incoming = list(residuals)
    res_in, NIN = _pack_residuals(incoming, WIN, D, dtype)
    key = (
        D,
        H,
        HD,
        RD,
        KR,
        RANK,
        I,
        E,
        TOPK,
        DE,
        S,
        NIN,
        WIN,
        has_gate,
        has_lat,
        bool(mlp.router.norm_logits),
        float(mlp.router.logit_temp),
        bool(mlp.router.norm_topk_prob),
        float(mlp.router.scaling),
        float(block.input_layernorm.norm.eps),
        dtype,
        NT,
    )
    has_past = int(getattr(cache, "offset", 0) or 0) > 0
    # 空 cache 的 dummy KV 与有 past 的路径不同，不能用 Tp=0 的校验放行 Tp>0。
    vkey = key + (has_past,)
    if vkey in _FAILED:
        return None
    try:
        cos, sin = _rope_cs(position_embeddings, 1, dtype, attn, cache)
        cos = cos.reshape(-1)[-RD:]
        sin = sin.reshape(-1)[-RD:]
        rope = mx.concatenate([cos, sin], axis=0)
        Tp = int(cache.offset)
        if Tp > 0 and cache.keys is not None:
            k_cache = cache.keys[0, :, :Tp, :]
            v_cache = cache.values[0, :, :Tp, :]
        else:
            k_cache = mx.zeros((H, 1, QK), dtype=dtype)
            v_cache = mx.zeros((H, 1, HD), dtype=dtype)
            Tp = 0
        if has_gate:
            ag = mx.concatenate(
                [
                    attn.attn_gate.weight.astype(dtype).reshape(-1),
                    attn.attn_gate.bias.astype(dtype).reshape(-1),
                ]
            )
        else:
            ag = mx.zeros((H * D + H,), dtype=dtype)
        if has_lat:
            lat_down = mlp.lat_down.weight.astype(dtype)
            lat_up = mlp.lat_up.weight.astype(dtype)
            lat_n = mx.concatenate(
                [
                    mlp.latent_norm.weight.astype(dtype),
                    mlp.latent_out_norm.weight.astype(dtype),
                ]
            )
        else:
            lat_down = mx.zeros((1, 1), dtype=dtype)
            lat_up = mx.zeros((1, 1), dtype=dtype)
            lat_n = mx.zeros((2,), dtype=dtype)
        if S > 0:
            sh_gu, sh_dw = _shared_w(mlp.shared, dtype)
        else:
            sh_gu = mx.zeros((1, 2, 1), dtype=dtype)
            sh_dw = mx.zeros((1, 1, 1), dtype=dtype)
        meta = mx.array([Tp], dtype=mx.int32)
        kern = _build(
            D,
            H,
            HD,
            RD,
            KR,
            RANK,
            I,
            E,
            TOPK,
            DE,
            S,
            NIN,
            WIN,
            has_gate,
            has_lat,
            bool(mlp.router.norm_logits),
            float(mlp.router.logit_temp),
            bool(mlp.router.norm_topk_prob),
            float(mlp.router.scaling),
            float(block.input_layernorm.norm.eps),
            dtype,
            NT,
        )
        y, va, vm, nk, nv = kern(
            inputs=[
                x.reshape(D),
                res_in,
                _pack_gn(block.input_layernorm, dtype),
                _pack_gn(block.post_attention_layernorm, dtype),
                attn.qkv_proj.weight.astype(dtype),
                attn.kv_up_proj.weight.astype(dtype),
                mx.concatenate(
                    [
                        attn.q_norm.weight.astype(dtype),
                        attn.k_norm.weight.astype(dtype),
                    ]
                ),
                rope,
                k_cache,
                v_cache,
                attn.o_proj.weight.astype(dtype),
                ag,
                mx.concatenate(
                    [
                        block.attn_res_q_attn.astype(dtype),
                        block.attn_res_q_mlp.astype(dtype),
                    ]
                ),
                mlp.router.weight.astype(dtype),
                mlp.router.expert_bias.astype(mx.float32),
                lat_down,
                lat_up,
                lat_n,
                mlp.experts.gate_up_w.astype(dtype),
                mlp.experts.down_w.astype(dtype),
                sh_gu,
                sh_dw,
                meta,
            ],
            output_shapes=[
                (1, 1, D),
                (1, 1, D),
                (1, 1, D),
                (1, 1, H, QK),
                (1, 1, H, HD),
            ],
            output_dtypes=[dtype, dtype, dtype, dtype, dtype],
            grid=(NT, 1, 1),
            threadgroup=(NT, 1, 1),
        )
        if vkey not in _VERIFIED:
            y_r, va_r, vm_r, nk_r, nv_r = _eager_ref(
                block, x, incoming, cache, position_embeddings
            )
            mx.eval(y, va, vm, nk, nv, y_r, va_r, vm_r, nk_r, nv_r)
            tol = 1e-4 if dtype == mx.float32 else 5e-2
            kv_tol = 1e-4 if dtype == mx.float32 else 2e-2
            checks = (
                _close(y, y_r, tol),
                _close(va, va_r, tol),
                _close(vm, vm_r, tol),
                _close(nk, nk_r, kv_tol),
                _close(nv, nv_r, kv_tol),
            )
            if not all(ok for ok, _ in checks):
                detail = ", ".join(f"{d:.2e}" for _, d in checks)
                raise RuntimeError(f"layer_decode 校验失败 |Δ|={detail}")

            # 大层单 TG 可能慢于 mx.matmul；小层 launch 占主导，融合通常更快，
            # 不做对拍以免短采样噪声把正确核打回 eager。
            if max(D, E) >= 256:

                def _min_t(fn, warm=2, iters=5):
                    for _ in range(warm):
                        mx.eval(fn())
                    ts = []
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        mx.eval(fn())
                        ts.append(time.perf_counter() - t0)
                    return min(ts)

                fused_t = _min_t(
                    lambda: kern(
                        inputs=[
                            x.reshape(D),
                            res_in,
                            _pack_gn(block.input_layernorm, dtype),
                            _pack_gn(block.post_attention_layernorm, dtype),
                            attn.qkv_proj.weight.astype(dtype),
                            attn.kv_up_proj.weight.astype(dtype),
                            mx.concatenate(
                                [
                                    attn.q_norm.weight.astype(dtype),
                                    attn.k_norm.weight.astype(dtype),
                                ]
                            ),
                            rope,
                            k_cache,
                            v_cache,
                            attn.o_proj.weight.astype(dtype),
                            ag,
                            mx.concatenate(
                                [
                                    block.attn_res_q_attn.astype(dtype),
                                    block.attn_res_q_mlp.astype(dtype),
                                ]
                            ),
                            mlp.router.weight.astype(dtype),
                            mlp.router.expert_bias.astype(mx.float32),
                            lat_down,
                            lat_up,
                            lat_n,
                            mlp.experts.gate_up_w.astype(dtype),
                            mlp.experts.down_w.astype(dtype),
                            sh_gu,
                            sh_dw,
                            meta,
                        ],
                        output_shapes=[
                            (1, 1, D),
                            (1, 1, D),
                            (1, 1, D),
                            (1, 1, H, QK),
                            (1, 1, H, HD),
                        ],
                        output_dtypes=[dtype, dtype, dtype, dtype, dtype],
                        grid=(NT, 1, 1),
                        threadgroup=(NT, 1, 1),
                    )[0]
                )
                eager_t = _min_t(
                    lambda: _eager_ref(block, x, incoming, cache, position_embeddings)[
                        0
                    ]
                )
                if fused_t > eager_t * 1.3:
                    raise RuntimeError(
                        f"layer_decode 慢于 eager {fused_t * 1e3:.2f}>{eager_t * 1e3:.2f}ms"
                    )
            _VERIFIED.add(vkey)
        residuals.append(va)
        residuals.append(vm)
        cache.update(nk, nv)
        return y, cache
    except Exception:
        if vkey not in _VERIFIED:
            _FAILED.add(vkey)
        else:
            _DISABLED = True
        return None


def prewarm(block, dtype=None):
    """编译并校验一层 decode 核。返回是否成功。"""
    if _DISABLED:
        return False
    was = block.training
    block.eval()
    router = getattr(getattr(block, "mlp", None), "router", None)
    saved_stats = None
    if router is not None:
        saved_stats = router.collect_stats
        router.collect_stats = False
    try:
        attn = block.self_attn
        if type(attn).__name__ != "MLAAttention":
            return False
        D = int(block.input_layernorm.norm.weight.shape[0])
        if dtype is None:
            dtype = block.input_layernorm.norm.weight.dtype
        x = mx.random.normal((1, 1, D)).astype(dtype)
        cache = KVCache()
        t_past = 8
        cache.update(
            mx.random.normal((1, t_past, attn.n_heads, attn.qk_dim)).astype(dtype),
            mx.random.normal((1, t_past, attn.n_heads, attn.head_dim)).astype(dtype),
        )
        residuals = [mx.random.normal((1, 1, D)).astype(dtype) for _ in range(3)]
        pe = attn._fallback_pos(t_past, 1, dtype)
        r = try_layer_decode(block, x, residuals, cache, pe, mask_is_full=True)
        return r is not None
    finally:
        if router is not None and saved_stats is not None:
            router.collect_stats = saved_stats
        if was:
            block.train()


def prewarm_from_model(model, dtype=None):
    try:
        mods = model.named_modules()
    except Exception:
        return False
    for _, mod in mods:
        if type(mod).__name__ != "VibyBlock":
            continue
        if getattr(mod, "use_linear_attn", True):
            continue
        return prewarm(mod, dtype=dtype)
    return False
