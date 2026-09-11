# Sparse attention VJP SIGSEGV — source-level repair, 2026-09-10

`model/kernels/sparse_attention.py` now returns eight array leaves from its VJP,
one for each array primal. The four metadata inputs receive `zeros_like` leaves
instead of `None`. Indexed forward/backward Metal code, the learned sink, masks,
sharded KV gradient accumulation and the default enabled state are unchanged.

The existing macOS reports (including `python3.13-2026-09-10-210212.000.ips` and
`python3.13-2026-09-10-210306.0003.ips`) place the fault on CPU thread 0 in
`mlx::core::array::ArrayDesc::init()`, reading near-null address `0x108` during
`eval_impl`. Another report faults during `compile_dfs`. This points to an invalid
MLX graph array, rather than a Metal execution fault.

The positional mismatch in the MLX 0.32.2 source explains these reports:

1. Python `InnerVJPFunction` calls `tree_flatten(..., false)` on the VJP result.
   [Binding source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/transforms.cpp#L734-L780)
2. `tree_flatten` skips non-array leaves, including `None`, when strict is false.
   [Tree source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/trees.cpp#L247-L259)
3. `CustomTransforms::vjp` appends the two output cotangents and then accesses
   `all_vjps[arg]` using the original argument indices, without a bounds check.
   [Primitive source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/primitives.cpp#L1745-L1766)

Old input-gradient structure:
`(dq, dw, dc, None, None, None, None, dsinks)`.
It becomes four array leaves, or six after the two output cotangents are appended.
The trainable `sinks` argument still has index **7**, so its lookup is out of bounds.
Returning all eight array leaves preserves the slots. Metadata zeros are not
selected when only q/window/compressed/sinks are differentiated.

Per the user's instruction, no tests, repro executions, model constructions,
Metal dispatches or performance measurements were run for this repair. The
existing `experiments/repro_sparse_bwd_crash.py` remains available; the runtime
repair and any MFU improvement have not been verified in this turn.
