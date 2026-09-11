# Key-owned sparse-attention dKV acceptance

2026-09-11, local Metal GPU, MLX 0.32.2.

`VIBY_SPARSE_ATTN_KEY_BWD=1` now selects the occurrence-preserving dKV
backend when `VIBY_SPARSE_ATTN_BWD_SPLIT=1` (the default). Numerical acceptance
passes. Subsequent direct ABBA with the completed CSA2 stack found key-owned
fwd+bwd about 1.4% slower than sharded dKV at B=4,T=1024, so it remains opt-in.
See [the current full-stack acceptance](CSA2_KERNEL_ACCEPTANCE.md) for measured
training/inference results and defaults. The numerical record below describes
the earlier standalone repair, before the full stack switched to NP=D/64.

## Correctness repairs

- The old CSR adapter read `dims[0]` as sequence length, although its caller
  supplied `[B,T,N]`. Its global-id intermediate also indexed window metadata
  with `query*T+p`. That intermediate has been removed; count/fill consume
  the same row-major selection and `[B,T,N]` dimensions as the forward kernel.
- `gcast` inside the custom VJP is already `[B,T,H,D]`: differentiation of
  the public output transpose has converted the cotangent. The extra
  transpose in the old key-owned adapter permuted query/head data again.
- Window visibility checks `pad[b*T+p]` for the **key**, together with
  `segment[b*T+t] == segment[b*T+p]`. It does not check the query's pad bit.
  Compressed visibility is already encoded by selection; no additional
  window predicate is applied to it.
- Each SIMD group now writes a separate `Partial[group,d]`. After all head
  and occurrence loops end, one barrier precedes a four-group reduction.
  Threads cooperatively write every dimension exactly once. The previous
  code raced on shared slots and wrote only dimensions 0 through 31.
- There is no fixed neighbor cache or degree limit. Compressed adjacency
  streams from CSR; window adjacency is generated from its bounded interval.

## Layout and dispatch

CSR contains `row_ptr[B*N+1]`, `edge_q[indices.size]`, and
`edge_slot[indices.size]`, all int32. Only slots below `lengths[q]` are counted
and filled, and every repeated `(query,slot)` occurrence is retained. Capacity
comes from selection storage, including ties, rather than nominal top-k.
Count uses integer atomic additions. Fill declares **all three** outputs
atomic (`cursor`, `edge_q`, `edge_slot`) and initializes them with the same
`init_value=0` interface. Edge writes are unique-position atomic stores.

The scan uses blocks of 256 integers, recursively scans block totals, and
adds preceding block prefixes. Its recursion depends only on shapes; no
edge count is read back to the host. The final row pointer holds logical E.
No padded global-id adjacency or gathered per-edge KV tensor is created.

Window dispatch is one 128-thread group per key. Compressed dispatch is
four groups per key, with active parts computed on GPU as
`min(4,max(1,(degree+31)//32))`. Each active part walks strided CSR entries;
inactive and empty parts explicitly write zeros. Partial gradients remain
FP32 until the four-part reduction casts to the primal dtype. There are no
floating-point atomic operations, shard buffers, or shard clearing in this
dKV backend.

At D=128 with a 16-bit input dtype, the explicit shared arrays use 2,368
bytes: 256 bytes for K and 2,112 for `Partial[4,132]`. No auxiliary neighbor
storage is needed. Barriers are outside the independent head/occurrence loops.

CSR is currently built per VJP. Reuse of CSR across layers is a possible
future optimization; this implementation does not cache layer-dependent
P, Ds, Delta, or gradients, and does not change compressed-primal ownership.

## Numerical evidence

The first minimal test failed on the previous code: a single head/key with
D=64 missed all 32 upper dimensions. Binary-exact tests now cover both
`Ds*Q` and `P*dO`, H=1/16, and D=64/128, with exact equality.

The combined targeted suites passed **41 tests**:

```sh
.venv/bin/python -m pytest -q tests/test_sparse_attention_key_owned.py tests/test_sparse_attention_kernel.py
VIBY_SPARSE_ATTN_KEY_BWD=1 .venv/bin/python experiments/repro_sparse_bwd_crash.py --bwd --b 2 --t 17 --n 7
VIBY_SPARSE_ATTN_KEY_BWD=1 .venv/bin/python experiments/repro_sparse_bwd_crash.py --prewarm
```

Both smoke commands passed. Coverage includes fp16/bf16; unequal B/T/N;
document/pad visibility; zero-degree and empty pools; duplicate selection;
scan block boundaries and three recursive levels (65,537 counts); compiled
scan with changing counts; degrees 0/1/32/33/64/65/96/97/4101; all four partial
slots; and compiled two-layer attention sharing the same compressed primal.
The shared-primal test is a local gradient-composition check, not a full CED
training-step acceptance run.

For seed 21 and `(B,T,N,D,W)` equal to `(1,1,1,64,1)`, `(2,17,7,64,8)`,
`(3,48,24,128,16)`, and `(2,33,0,128,16)`, the largest observed pool-gradient
relative L2 errors were:

| Input dtype | Versus query-owned MMA dKV | Versus FP32 eager formula |
| --- | ---: | ---: |
| fp16 | 5.14e-5 | 3.67e-4 |
| bf16 | 3.25e-4 | 2.98e-3 |

Forward output, dQ, and dSink are elementwise identical between the two
dKV backends in these paired tests. dKV is not guaranteed bitwise equal:
SIMD and MMA use different FP32 dot/reduction orders, and both then round
P/Ds to the input dtype at the existing backward operand boundary.

The initially failing strict bf16 comparison used window key `(b=0,p=0)`,
dimension 52, with the earlier NP=1 backward. The diagnostic now follows the
current NP=D/64 layout and prints its observed coefficient differences:

```sh
.venv/bin/python experiments/check_key_owned_rounding.py
```

In the earlier NP=1 record, the maximum QK dot difference was 7.6293945e-6 and the maximum dOK difference
is 3.8146973e-6. For query 2/head 3, P rounds to 0.31640625 with MMA and
0.314453125 with SIMD. Rounded Ds is identical. Applying the **same**
independent FP32 reduction to each set of coefficients gives -0.13788700
and -0.13312626, which round to the observed bf16 gradients -0.1376953125
and -0.1328125. This remaining difference is a coefficient-rounding effect,
not a missing or duplicated occurrence.

The tests bound both per-element and relative L2 errors rather than requiring
bitwise dKV equality. The subsequent performance evidence is in the linked
full-stack report; no long-run convergence or full-repository regression is claimed.
