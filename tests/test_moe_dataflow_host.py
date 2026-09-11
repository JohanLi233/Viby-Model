"""Host-only dispatch/fallback/adjoint checks. These do NOT execute Metal.

The small fake MLX module executes the production native-count fallback with
NumPy and records Metal launch contracts. GPU numerical tests live separately.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Array(np.ndarray):
    @property
    def at(self):
        parent = self

        class At:
            def __getitem__(self, index):
                self.index = index
                return self

            def add(self, values):
                out = parent.copy()
                np.add.at(out, self.index, values)
                return out

        return At()


def array(value, dtype=np.int32):
    return np.asarray(value, dtype=dtype).view(_Array)


@pytest.fixture
def load(monkeypatch):
    mx = types.ModuleType("mlx.core")
    mx.float32, mx.float16 = np.float32, np.float16
    mx.int32, mx.uint32 = np.int32, np.uint32
    mx.bfloat16 = object()  # not emulated
    mx.cpu, mx.gpu, mx.device = "cpu", "gpu", "cpu"
    mx.default_device = lambda: mx.device
    mx.metal = types.SimpleNamespace(is_available=lambda: True)
    mx.stop_gradient = lambda a: a
    mx.zeros_like = np.zeros_like
    mx.sum = np.sum
    mx.allocations, mx.launches = [], []

    def zeros(shape, dtype=np.float32):
        mx.allocations.append(tuple(shape))
        return np.zeros(shape, dtype=dtype).view(_Array)

    mx.zeros = zeros
    mx.ones = lambda shape, dtype: np.ones(shape, dtype=dtype).view(_Array)
    mx.arange = lambda n, dtype: np.arange(n, dtype=dtype).view(_Array)

    def custom(fn):
        def attach(vjp):
            fn.recorded_vjp = vjp
            return vjp
        fn.vjp = attach
        return fn

    def metal_kernel(**spec):
        def launch(**kw):
            mx.launches.append((spec, kw))
            return [np.zeros(shape, dtype=dtype) for shape, dtype in
                    zip(kw["output_shapes"], kw["output_dtypes"])]
        return launch

    mx.custom_function = custom
    mx.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    package = types.ModuleType("mlx")
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    def module(name):
        path = ROOT / "model" / "kernels" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"model.kernels._host_{name}", path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result, mx

    return module


@pytest.mark.parametrize("m,d,k", [(0, 65, 6), (1, 1, 1), (9, 127, 6), (33, 1031, 1), (7, 0, 6)])
def test_production_gather_cpu_fallback_and_adjoint(load, m, d, k):
    mod, _ = load("moe_gather")
    rng = np.random.default_rng(7)
    order = array(rng.permutation(m * k))
    inverse = array(np.argsort(order))
    x = array(rng.normal(size=(m, d)), np.float32)
    g = array(rng.normal(size=(m * k, d)), np.float32)
    got = mod.gather_routes(x, order, inverse, k)
    np.testing.assert_array_equal(got, x[order // k])
    # Independent adjoint identity for the proposed inverse-permutation VJP.
    dx = g[inverse].reshape(m, k, d).sum(axis=1, dtype=np.float64)
    np.testing.assert_allclose(np.sum(got.astype(np.float64) * g),
                               np.sum(x.astype(np.float64) * dx), rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("d", [1, 127, 128, 129, 1024, 1031])
def test_actual_vjp_launch_has_unique_output_owners(load, d):
    mod, mx = load("moe_gather")
    mx.device, mod._ENABLED = mx.gpu, True
    m, k = 9, 6
    x = array(np.zeros((m, d)), np.float32)
    order = array(np.arange(m * k)[::-1])
    inv = array(np.argsort(order))
    mod.gather_routes(x, order, inv, k)
    grads = mod._op(k).recorded_vjp((x, order, inv), np.ones((m*k, d), np.float32), None)
    assert len(grads) == 3 and grads[0].shape == x.shape
    np.testing.assert_array_equal(grads[1], np.zeros_like(order))
    np.testing.assert_array_equal(grads[2], np.zeros_like(inv))
    spec, kw = mx.launches[-1]
    assert not spec.get("atomic_outputs", False) and "init_value" not in kw
    assert kw["threadgroup"] == (128, 1, 1)
    assert dict(kw["template"])["D"] == d and dict(kw["template"])["K"] == k
    owners = [(t, f) for t in range(kw["grid"][1]) for f in range(kw["grid"][0]) if f < d]
    assert len(owners) == len(set(owners)) == m * d
    assert "threadgroup_barrier" not in spec["source"]


@pytest.mark.parametrize("b,t,e,k", [(0, 3, 8, 1), (2, 0, 8, 6), (1, 1, 8, 1),
                                    (3, 17, 96, 6), (4, 1024, 96, 6), (2, 5, 513, 6)])
def test_production_compact_counts_cpu_fallback(load, b, t, e, k):
    mod, mx = load("moe_counts")
    rng = np.random.default_rng(6)
    ids = array(rng.integers(e, size=(b*t, k)))
    got = mod.sequence_route_counts(ids, b, t, e)
    dense = np.zeros((b*t, e), np.float32)
    for j in range(k):
        np.add.at(dense, (np.arange(b*t), ids[:, j]), 1)
    expected = dense.reshape(b, t, e).sum(axis=1)
    np.testing.assert_array_equal(got, expected)
    assert all(shape in ((b*e,), (b, e)) for shape in mx.allocations)
    assert not mx.launches


@pytest.mark.parametrize("t,k", [(1, 1), (17, 6), (43, 6), (1024, 6)])
def test_actual_histogram_grid_visits_every_occurrence_once(load, t, k):
    mod, mx = load("moe_counts")
    mx.device = mx.gpu
    b, e = 3, 96
    mod.sequence_route_counts(array(np.zeros((b*t, k))), b, t, e)
    spec, kw = mx.launches[-1]
    cfg = dict(kw["template"])
    assert cfg["E"] * 4 <= 32768
    assert not spec.get("atomic_outputs", False)
    assert "init_value" not in kw
    assert kw["grid"] == (cfg["TILES"] * cfg["NT"], b, 1)
    visited = []
    for tile in range(cfg["TILES"]):
        for tid in range(cfg["NT"]):
            for i in range(tid, cfg["RT"], cfg["NT"]):
                r = tile * cfg["RT"] + i
                if r < cfg["R"]:
                    visited.append(r)
    assert sorted(visited) == list(range(t*k))
    assert "threadgroup atomic_uint counts[E]" in spec["source"]


def test_shape_validation_precedes_dispatch(load):
    gather, mx = load("moe_gather")
    x = array(np.zeros((2, 3)), np.float32)
    with pytest.raises(ValueError, match="positive"):
        gather.gather_routes(x, array([]), array([]), 0)
    with pytest.raises(ValueError, match="M\\*K"):
        gather.gather_routes(x, array([0]), array([0]), 1)
    counts, mx = load("moe_counts")
    with pytest.raises(ValueError, match="B\\*T"):
        counts.sequence_route_counts(array([[0]]), 2, 1, 3)
    assert not mx.launches


def test_flags_default_to_opt_in(load, monkeypatch):
    for name, flag in [("moe_gather", "VIBY_MOE_GATHER_VJP"),
                       ("moe_counts", "VIBY_MOE_COMPACT_AUX"),
                       ("moe_decode", "VIBY_MOE_DECODE_COMPILE")]:
        monkeypatch.delenv(flag, raising=False)
        mod, _ = load(name)
        assert not mod._ENABLED


def test_compact_count_exactness_guard(load):
    mod, _ = load("moe_counts")
    mod._ENABLED = True
    assert mod.enabled_for(types.SimpleNamespace(size=2**24))
    assert not mod.enabled_for(types.SimpleNamespace(size=2**24 + 1))


def test_native_sorted_gemms_keep_single_index():
    import ast
    tree = ast.parse((ROOT / "model/moe.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sparse_forward")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "gather_mm"]
    assert len(calls) == 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "rhs_indices" in kw and "lhs_indices" not in kw
        assert isinstance(kw["sorted_indices"], ast.Constant) and kw["sorted_indices"].value is True


def test_train_benchmark_independent_switches():
    import ast
    tree = ast.parse((ROOT / "experiments/bench_csa2_plan.py").read_text())
    nodes = []
    for node in tree.body:
        text = ast.unparse(node)
        if (isinstance(node, ast.Assign) and text.startswith("VARIANTS =")) or (
                isinstance(node, ast.Expr) and text.startswith("VARIANTS.update(")) or (
                isinstance(node, ast.For) and text.startswith("for variant in VARIANTS.values():")) or (
                isinstance(node, ast.FunctionDef) and node.name == "configure"):
            nodes.append(node)
    ns = {k: types.SimpleNamespace() for k in ("sa", "fs", "moe", "hc", "moe_counts", "moe_decode", "moe_gather")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "bench-config", "exec"), ns)
    for name, expected in [("fused_combine", (False, False)), ("dataflow_gather", (True, False)),
                           ("dataflow_counts", (False, True)), ("dataflow_combined", (True, True))]:
        ns["configure"](name)
        assert (ns["moe_gather"]._ENABLED, ns["moe_counts"]._ENABLED) == expected
        assert ns["moe"]._COMBINE_ENABLED and ns["fs"]._ENABLED
        assert not ns["sa"]._KEY_OWNED_BWD and not ns["moe_decode"]._ENABLED


def test_decode_benchmark_is_incremental_not_all_experts_baseline():
    import ast
    tree = ast.parse((ROOT / "experiments/bench_csa2_inference.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure")
    ns = {k: types.SimpleNamespace() for k in ("dm", "moe", "fs", "moe_dispatch", "moe_decode")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "decode-config", "exec"), ns)
    for name, flag in [("combined", False), ("compiled_moe", True)]:
        ns["configure"](name)
        assert ns["moe"]._DECODE_GATHER and ns["dm"]._ENABLED and ns["fs"]._ENABLED
        assert ns["moe_dispatch"]._COMBINE_ENABLED and ns["moe_decode"]._ENABLED == flag


def test_decode_guard_rejects_training_large_batches_and_mixed_dtype(load):
    mod, mx = load("moe_decode")
    mod._ENABLED, mx.device = True, mx.gpu
    x = array(np.zeros((1, 1, 64)), np.float32)
    weights = (x,) * 5
    assert mod.enabled_for(x, False, True, 8, weights)
    assert not mod.enabled_for(x, True, True, 8, weights)
    assert not mod.enabled_for(x, False, False, 8, weights)
    assert not mod.enabled_for(array(np.zeros((9, 1, 64)), np.float32), False, True, 8, weights)
    assert not mod.enabled_for(x, False, True, 8, (x.astype(np.float16),) + weights[1:])
    mx.device = mx.cpu
    assert not mod.enabled_for(x, False, True, 8, weights)


def test_decode_cache_forwards_new_parameter_arrays(load, monkeypatch):
    mod, mx = load("moe_decode")
    for name, member in [("model.moe", "expert_act"),
                         ("model.kernels.decode_metadata", "combine_selected_experts")]:
        stub = types.ModuleType(name)
        setattr(stub, member, lambda *args: None)
        monkeypatch.setitem(sys.modules, name, stub)
    # Record explicit inputs, not GPU numerics. A changed weight must be passed
    # through a cache hit, never captured as a constant by the wrapper.
    traced = []
    def compile_fn(fn):
        traced.append(fn)
        return lambda *args: args
    mx.compile = compile_fn
    args0 = tuple(object() for _ in range(8))
    args1 = args0[:3] + tuple(object() for _ in range(5))
    out0 = mod.compiled_decode(*args0, 96, 6, 10.0, 10.0)
    out1 = mod.compiled_decode(*args1, 96, 6, 10.0, 10.0)
    assert out0 == args0 and out1 == args1 and len(traced) == 1
    assert traced[0].__code__.co_argcount == 8
