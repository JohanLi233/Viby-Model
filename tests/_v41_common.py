"""DeepSeek-V4.1（缩放版）测试公共设施。

所有测试都用 tiny 配置（<50M 参数、T<=32）跑，随机性一律由
VIBY_TEST_*_SEED 环境变量（默认 20260910）驱动，保证可复现。

语义依据：/tmp/dsv41_ref/v41.txt（DeepSeek-V4.1 技术报告）与 model/ 下实现。
"""

import contextlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from model.config import VibyConfig  # noqa: E402
from model.model import VibyForCausalLM  # noqa: E402

DEFAULT_SEED = 20260910


def seed_of(name: str, default: int = 0) -> int:
    """VIBY_TEST_<NAME>_SEED 覆盖（NAME 大写），默认固定种子。"""
    env = os.environ.get(f"VIBY_TEST_{name.upper()}_SEED")
    if env is not None:
        return int(env)
    return DEFAULT_SEED + int(default)


# ---------------------------------------------------------------- 配置预设


# These fixtures explicitly retain plain CED for legacy component contracts.
# Default NCP construction and its integration are tested in test_ncp_ced.py.
def cfg_tiny(**kw) -> VibyConfig:
    """tiny 预设：4 层 / dim 256 / 16 专家 top-4，默认关 Engram（省 tokenizer）。"""
    base = dict(
        preset="tiny",
        ncp_enabled=False,
        engram_layer_ids=(),
        vocab_size=256,
        max_seq_len=128,
    )
    base.update(kw)
    return VibyConfig(**base)


def cfg_mix(**kw) -> VibyConfig:
    """混合 CSA2 模式配置（CED 边界 = 第 3 层）。

    ratios = [0,0,2,1,1,1,0]（n_mtp=1）、kv 源 (2,3)、index 源 (2,3,5)、
    candidate 源 3 → 逐层模式 = sliding / sliding / full(r=2) / full(r=1)
    / reuse / reindex(用候选池)。
    """
    base = dict(
        preset="tiny",
        ncp_enabled=False,
        n_layers=6,
        compress_ratios=(0, 0, 2, 1, 1, 1, 0),
        kv_source_layers=(2, 3),
        index_source_layers=(2, 3, 5),
        candidate_source_layer=3,
        engram_layer_ids=(),
        vocab_size=256,
        max_seq_len=128,
    )
    base.update(kw)
    return VibyConfig(**base)


def cfg_ced(**kw) -> VibyConfig:
    """纯 CED 配置：编码段 r=2（第 2 层），解码段 r=1（第 3 层起），无 reindex。"""
    base = dict(
        preset="tiny",
        ncp_enabled=False,
        n_layers=6,
        engram_layer_ids=(),
        vocab_size=256,
        max_seq_len=128,
    )
    base.update(kw)
    return VibyConfig(**base)


def cfg_engram(**kw) -> VibyConfig:
    """带 Engram 的 tiny 配置（挂第 1、2 层，用仓库自带 tokenizer 建压缩表）。"""
    base = dict(
        preset="tiny",
        ncp_enabled=False,
        vocab_size=256,
        max_seq_len=64,
        engram_layer_ids=(1, 2),
        engram_max_ngram_size=4,
    )
    base.update(kw)
    return VibyConfig(**base)


# ---------------------------------------------------------------- 模型缓存

_MODELS: dict = {}


def build(
    cfg: VibyConfig, seed: int = DEFAULT_SEED, skip_init: bool = True
) -> VibyForCausalLM:
    """固定种子构造模型；skip_init 跳过截断正态覆盖（结构测试不需要真初始化）。"""
    mx.random.seed(seed)
    model = VibyForCausalLM(cfg, skip_init=skip_init)
    mx.eval(model.parameters())
    return model


def cached(key: str, factory) -> VibyForCausalLM:
    """按 key 缓存模型，避免每个用例重建（构建 Engram 要读 tokenizer）。"""
    if key not in _MODELS:
        _MODELS[key] = factory()
    return _MODELS[key]


def tiny_model(tag: str = "", **kw) -> VibyForCausalLM:
    key = "tiny:" + tag + ":" + repr(sorted(kw.items()))
    return cached(key, lambda: build(cfg_tiny(**kw), seed=seed_of("tiny")))


def mix_model(**kw) -> VibyForCausalLM:
    key = "mix:" + repr(sorted(kw.items()))
    return cached(key, lambda: build(cfg_mix(**kw), seed=seed_of("mix", 1)))


def ced_model(**kw) -> VibyForCausalLM:
    key = "ced:" + repr(sorted(kw.items()))
    return cached(key, lambda: build(cfg_ced(**kw), seed=seed_of("ced", 2)))


def engram_model(tag: str = "", **kw) -> VibyForCausalLM:
    key = "engram:" + tag + ":" + repr(sorted(kw.items()))
    return cached(key, lambda: build(cfg_engram(**kw), seed=seed_of("engram", 3)))


# ---------------------------------------------------------------- 参数工具


def param(model, path: str) -> mx.array:
    return dict(tree_flatten(model.parameters()))[path]


def set_param(model, path: str, value: mx.array):
    model.update(tree_unflatten([(path, value)]))
    mx.eval(model.parameters())


@contextlib.contextmanager
def perturbed(model, path: str, delta):
    """临时给某个叶子参数加 delta（广播），退出时精确还原。

    用来做"改这条通路是否影响那个输出"的语义探针。
    """
    if isinstance(delta, (int, float)):
        arr = param(model, path)
        new = arr + mx.array(delta, dtype=arr.dtype)
    else:
        new = mx.array(delta, dtype=param(model, path).dtype)
    old = param(model, path)
    set_param(model, path, new)
    try:
        yield new
    finally:
        set_param(model, path, old)


def tree_paths(model) -> list:
    return sorted(k for k, _ in tree_flatten(model.parameters()))


def merge_caches(cfg: VibyConfig, caches: list, start_pos: int):
    """把若干条单序列的 VibyCache 合并成一条 batch cache（连续 batch 解码用）。

    逐层搬运 window / compress_kv / index_k，并把压缩器的 kv_state 按 batch
    维拼接（各序列的分组进度 filled 可以不同）。注意：decode 会**原地**推进
    cache，所以这里的 caches 必须是解码前的快照。
    """
    from model.cache import VibyCache

    bc = VibyCache(cfg, len(caches))
    for i in range(cfg.n_layers):
        for b, c in enumerate(caches):
            src = c[i]
            bc[i].window[b] = src.window[0]
            if src.compress_kv is not None:
                bc[i].compress_kv[b] = src.compress_kv[0]
            if src.index_k is not None:
                bc[i].index_k[b] = src.index_k[0]
            if src.kv_state is not None:
                cur = bc[i].kv_state
                if cur is None:
                    bc[i].kv_state = (
                        src.kv_state[0],
                        src.kv_state[1],
                        src.kv_state[2],
                    )
                else:
                    bc[i].kv_state = tuple(
                        mx.concatenate([cur[j], src.kv_state[j]], axis=0)
                        for j in range(3)
                    )
    bc.start_pos = int(start_pos)
    return bc


# ---------------------------------------------------------------- 数值工具


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.max(d)) if d.size else 0.0


def bit_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.all(a == b).item())


def row_diffs(a: mx.array, b: mx.array):
    """[B,T,D] 逐 token 的最大绝对差（numpy [B,T]），用于可见性探针。"""
    import numpy as np

    d = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)), axis=-1)
    mx.eval(d)
    return np.asarray(d)


def assert_changed_exactly(
    a: mx.array, b: mx.array, expected, changed_tol=1e-9, same_tol=1e-6
):
    """扰动探针：expected 里的位置必须变，其余位置必须（近似）bit 不变。"""

    d = row_diffs(a, b)[0]
    expected = set(int(i) for i in expected)
    for i in range(d.shape[0]):
        if i in expected:
            assert d[i] > changed_tol, f"位置 {i} 应受影响但 Δ={d[i]}"
        else:
            assert d[i] < same_tol, f"位置 {i} 不应受影响但 Δ={d[i]}"
