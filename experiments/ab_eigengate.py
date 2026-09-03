"""同进程 A/B：KDA chunk 扫描 ± EigenGate。

默认关时 B 应与 A 逐位一致（短序列 / 无门控点）。打开后（每 K 个
chunk 一次谱高通）测 fwd+bwd 墙钟与输出差。两条路在同一进程里逐次
交替、取中位数。

用法:
  uv run experiments/ab_eigengate.py
  VIBY_EIGENGATE_K=16 uv run experiments/ab_eigengate.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from model.kda import KDA_CHUNK, _chunk_kda, _scan_prewarm, _scan_prewarm_gated
from model import eigengate as _eg

B = int(os.environ.get("VIBY_BENCH_B", 12))
H = int(os.environ.get("VIBY_BENCH_H", 8))
T = int(os.environ.get("VIBY_BENCH_T", 1024))
D = int(os.environ.get("VIBY_BENCH_D", 96))
K = int(os.environ.get("VIBY_EIGENGATE_K", "16"))
ITERS = int(os.environ.get("VIBY_AB_ITERS", "8"))


def _inputs(seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, T, D)) * 0.3
    k = mx.random.normal((B, H, T, D)) * (D**-0.5)
    v = mx.random.normal((B, H, T, D)) * 0.5
    log_g = -mx.random.uniform(0.0, 0.3, (B, H, T, D))
    beta = mx.sigmoid(mx.random.normal((B, H, T)))
    mx.eval(q, k, v, log_g, beta)
    return q, k, v, log_g, beta


def _loss_fn(q, k, v, log_g, beta):
    o, s = _chunk_kda(q, k, v, log_g, beta)
    return (o**2).sum() + (s**2).sum()


def main():
    q, k, v, log_g, beta = _inputs()
    C = KDA_CHUNK
    NC = (T + C - 1) // C
    _scan_prewarm(NC, C, D, D, B=1, H=1)
    os.environ["VIBY_EIGENGATE_K"] = str(K)
    mask = tuple(_eg.chunk_gate_mask(NC, T, C, 0))
    os.environ["VIBY_EIGENGATE"] = "1"
    os.environ.setdefault("VIBY_EIGENGATE_LAM", "1")
    if any(mask):
        _scan_prewarm_gated(NC, C, D, D, mask, B=1, H=1)

    vg = mx.value_and_grad(_loss_fn, argnums=(0, 1, 2))

    os.environ["VIBY_EIGENGATE"] = "0"
    va, ga = vg(q, k, v, log_g, beta)
    mx.eval(va, *ga)

    os.environ["VIBY_EIGENGATE"] = "1"
    os.environ["VIBY_EIGENGATE_K"] = str(K)
    os.environ.setdefault("VIBY_EIGENGATE_LAM", "1")
    vb, gb = vg(q, k, v, log_g, beta)
    mx.eval(vb, *gb)

    dloss = abs(va.item() - vb.item())
    dgrad = max((a - b).abs().max().item() for a, b in zip(ga, gb))
    print(f"shape B={B} H={H} T={T} D={D}  K={K} chunk={C}")
    print(f"Δloss={dloss:.3e}  Δ∂ max={dgrad:.3e}  (off vs on)")

    def _once(on: bool, gate_k: int):
        os.environ["VIBY_EIGENGATE"] = "1" if on else "0"
        os.environ["VIBY_EIGENGATE_K"] = str(gate_k)
        val, g = vg(q, k, v, log_g, beta)
        t0 = time.perf_counter()
        mx.eval(val, *g)
        return (time.perf_counter() - t0) * 1e3

    # 两条路都先预热（compile / 融合核），再交替计时
    for gate_k in (K, NC):
        for _ in range(3):
            _once(False, gate_k)
            _once(True, gate_k)

    def _report(label, samples):
        med = statistics.median(samples)
        mn = min(samples)
        return med, mn

    times = {"off": [], "on": [], "on_end": []}
    for _ in range(ITERS):
        times["off"].append(_once(False, K))
        times["on"].append(_once(True, K))
        times["on_end"].append(_once(True, NC))  # 只在序列末门控：不切扫描

    med_a, min_a = _report("off", times["off"])
    med_b, min_b = _report("on", times["on"])
    med_c, min_c = _report("on_end", times["on_end"])
    print(f"fwd+bwd  off     median {med_a:.2f} ms  min {min_a:.2f} ms")
    print(
        f"fwd+bwd  on K={K:<3} median {med_b:.2f} ms  min {min_b:.2f} ms  "
        f"ratio {med_b / med_a:.3f} (min {min_b / min_a:.3f})"
    )
    print(
        f"fwd+bwd  on K={NC:<3} median {med_c:.2f} ms  min {min_c:.2f} ms  "
        f"ratio {med_c / med_a:.3f} (min {min_c / min_a:.3f})  [末态一次门控]"
    )


if __name__ == "__main__":
    main()
