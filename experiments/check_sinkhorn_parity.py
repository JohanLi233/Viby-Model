"""Fixed-seed full-model Sinkhorn reference/fused parity, run serially."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from model.config import VibyConfig
from model.model import VibyForCausalLM

mx.random.seed(1234)
cfg = VibyConfig(n_mtp_layers=0)
model = VibyForCausalLM(cfg)
mx.random.seed(7)
X = mx.random.randint(0, cfg.vocab_size, (2, 256))
Y = mx.random.randint(0, cfg.vocab_size, (2, 256))
M = mx.ones((2, 256), dtype=mx.float32)
seg = mx.cumsum((mx.random.uniform(shape=(2,256)) < 1/50).astype(mx.int32), axis=1)

from model.kernels import sinkhorn_fused as sk
from mlx import nn
from mlx.utils import tree_flatten
mx.set_cache_limit(8 * 1024**3)
for enabled in (False, True):
    sk._ENABLED = enabled
    def loss_fn(m):
        return m(X, labels=Y, loss_mask=M, segment_ids=seg, use_mtp=False).loss
    l, g = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(l, g)
    gnorm = mx.sqrt(sum(mx.sum(v * v) for _, v in tree_flatten(g)))
    mx.eval(gnorm)
    print(json.dumps({"fused": enabled, "loss": round(float(l), 6), "grad_norm": round(float(gnorm), 6)}), flush=True)
    del l, g, gnorm
