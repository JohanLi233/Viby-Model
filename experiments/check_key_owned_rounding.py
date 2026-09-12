"""Inspect the bf16 dot/operand rounding residual for window key (b=0,p=0).

Run: .venv/bin/python experiments/check_key_owned_rounding.py
This probes the two dot-product orders, then reduces both with the same MLX
formula so any difference cannot be attributed to CSR or key ownership.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from model.kernels import sparse_attention as sa
from test_sparse_attention_kernel import _inputs


_DOTS = r"""
    uint tid=thread_position_in_grid.x, group=tid/32, lane=tid%32;
    constexpr uint NP=D/64, NT=64*NP;
    uint hg=group/NP, dp=group%NP, col=dp*64;
    uint query=thread_position_in_grid.y;
    threadgroup T Ks[8*(D+8)];
    threadgroup float S[NP*16*8], U[NP*16*8];
    for (uint i=tid;i<8*D;i+=NT)
        Ks[(i/D)*(D+8)+i%D]=i/D==0 ? k[i%D] : T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float,8,8> sf=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_matrix<float,8,8> uf=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint i=0;i<64/8;++i) {
        simdgroup_matrix<T,8,8> Q,G,K;
        simdgroup_load(Q,q+((size_t)query*16+hg*8)*D+col+i*8,D);
        simdgroup_load(G,g+((size_t)query*16+hg*8)*D+col+i*8,D);
        simdgroup_load(K,Ks+col+i*8,D+8,ulong2(0,0),true);
        simdgroup_multiply_accumulate(sf,Q,K,sf);
        simdgroup_multiply_accumulate(uf,G,K,uf);
    }
    simdgroup_store(sf,S+(dp*16+hg*8)*8,8);
    simdgroup_store(uf,U+(dp*16+hg*8)*8,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<16) {
        float s=S[tid*8], u=U[tid*8];
        for (uint p=1;p<NP;++p) { s+=S[(p*16+tid)*8]; u+=U[(p*16+tid)*8]; }
        mma[(query*16+tid)*2]=s;
        mma[(query*16+tid)*2+1]=u;
    }
    for (uint h=group;h<16;h+=NT/32) {
        float s=0,u=0;
        for (uint r=0;r<D/32;++r) {
            uint dd=lane+32*r;
            s+=float(q[((size_t)query*16+h)*D+dd])*float(k[dd]);
            u+=float(g[((size_t)query*16+h)*D+dd])*float(k[dd]);
        }
        s=simd_sum(s); u=simd_sum(u);
        if (lane==0) {
            scalar[(query*16+h)*2]=s;
            scalar[(query*16+h)*2+1]=u;
        }
    }
"""


def main():
    b, t, n, d, W = 2, 33, 0, 128, 16
    q, w, c, vis, seg, pad, sinks, _, scale = _inputs(b, t, n, d, W, 21)
    cot = mx.random.normal((b, 16, t, d)).astype(q.dtype)
    g = cot.transpose(0, 2, 1, 3)
    idx, lens = sa.compact_visible(vis)
    out, lse = sa._operation(W, scale, 16)(q, w, c, idx, lens, seg, pad, sinks)
    _, delta, _ = sa._kernels()[4](
        inputs=[
            q,
            w,
            c,
            idx,
            lens,
            seg,
            pad,
            sinks,
            mx.array([b, t, n], mx.uint32),
            mx.array([scale], mx.float32),
            g,
            lse,
            out,
        ],
        template=[("T", q.dtype), ("D", d), ("W", W), ("BK", 16), ("NP", d // 64)],
        grid=(d, b * t, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[q.shape, (b, t, 16), (b, t, 16)],
        output_dtypes=[q.dtype, mx.float32, mx.float32],
    )
    kernel = mx.fast.metal_kernel(
        name="diagnose_key_dot",
        input_names=["q", "g", "k"],
        output_names=["mma", "scalar"],
        source=_DOTS,
        header=sa._HEADER,
    )
    a, z = kernel(
        inputs=[q, g, w[0, 0]],
        template=[("T", q.dtype), ("D", d)],
        grid=(d, 16, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(16, 16, 2)] * 2,
        output_dtypes=[mx.float32] * 2,
    )

    def coeff(x):
        p = mx.exp(x[..., 0] * scale - lse[0, :16])
        ds = p * (x[..., 1] - delta[0, :16]) * scale
        return p.astype(q.dtype).astype(mx.float32), ds.astype(q.dtype).astype(
            mx.float32
        )

    ap, ads = coeff(a)
    zp, zds = coeff(z)
    qa, ga = q[0, :16].astype(mx.float32), g[0, :16].astype(mx.float32)
    ka = mx.sum(ads[..., None] * qa + ap[..., None] * ga, axis=(0, 1))
    kz = mx.sum(zds[..., None] * qa + zp[..., None] * ga, axis=(0, 1))
    mx.eval(a, z, ap, ads, zp, zds, ka, kz)
    print(
        "dot max abs difference [QK, dOK]:",
        np.max(abs(np.asarray(a) - np.asarray(z)), axis=(0, 1)),
    )
    pwhere = np.argwhere(np.asarray(ap) != np.asarray(zp))
    print("P rounding differences (query, head):", pwhere.tolist())
    for i, j in pwhere:
        print(
            "P at", (int(i), int(j)), "MMA:", ap[i, j].item(), "SIMD:", zp[i, j].item()
        )
    where = np.argwhere(np.asarray(ads) != np.asarray(zds))
    print("Ds rounding differences (query, head):", where.tolist())
    for i, j in where:
        print(
            "Ds at",
            (int(i), int(j)),
            "MMA:",
            ads[i, j].item(),
            "SIMD:",
            zds[i, j].item(),
        )
    dimension = int(np.argmax(abs(np.asarray(ka) - np.asarray(kz))))
    print(
        "largest-difference dimension",
        dimension,
        "common FP32 reduction: MMA:",
        ka[dimension].item(),
        "SIMD:",
        kz[dimension].item(),
    )
    print(
        "final bf16: MMA:",
        ka.astype(q.dtype)[dimension].item(),
        "SIMD:",
        kz.astype(q.dtype)[dimension].item(),
    )


if __name__ == "__main__":
    main()
