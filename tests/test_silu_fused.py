"""SiLU-GLU（SwiGLU）融合核的数值与梯度对齐（对照 situ.py 的同构测试）。

默认 `hidden_act=silu`（config 默认值）训练/解码都走融合 Metal kernel，
这里钉死 fwd+bwd、打包/两参数入口一致，以及失败回退不会全局禁用。
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))


import unittest

import mlx.core as mx

from model.acts import silu_glu_eager
from model.kernels.silu import prewarm as silu_prewarm
from model.kernels.silu import prewarm_packed as silu_prewarm_packed
from model.kernels.silu import silu_glu as silu_two_arg
from model.kernels.silu import silu_glu_packed
from model.kernels import silu as silu_mod


class SiluFusedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert silu_prewarm(mx.float32), "silu f32 预热失败"
        assert silu_prewarm(mx.bfloat16), "silu bf16 预热失败"

    def _pair(self, shape, dtype, seed=0):
        mx.random.seed(seed)
        I = shape[-1]  # noqa: E741
        h = (mx.random.normal((*shape[:-1], 2 * I)) * 3.0).astype(dtype)
        mx.eval(h)
        return h, h[..., :I], h[..., I:]

    def test_packed_matches_eager_formula(self):
        for dtype in (mx.float32, mx.bfloat16):
            h, g, u = self._pair((7, 1, 96), dtype)
            got = silu_glu_packed(h)
            ref = silu_glu_eager(g, u)
            mx.eval(got, ref)
            self.assertEqual(got.shape, g.shape)
            rel = (
                (got.astype(mx.float32) - ref.astype(mx.float32)).abs().max()
                / (ref.astype(mx.float32).abs().max() + 1e-12)
            ).item()
            self.assertLess(rel, 1e-2 if dtype == mx.bfloat16 else 1e-5, str(dtype))

    def test_packed_matches_two_arg_kernel_bitwise(self):
        for dtype in (mx.float32, mx.bfloat16):
            h, g, u = self._pair((5, 3, 64), dtype, seed=1)
            got = silu_glu_packed(h)
            ref = silu_two_arg(g, u)
            mx.eval(got, ref)
            self.assertTrue(bool((got == ref).all().item()), str(dtype))

    def test_packed_gradient_splits_into_dg_and_du(self):
        I = 48  # noqa: E741
        h, g, u = self._pair((11, 1, I), mx.float32, seed=2)
        cot = mx.random.normal((11, 1, I))
        mx.eval(cot)

        def f_packed(h_):
            return (silu_glu_packed(h_) * cot).sum()

        def f_ref(g_, u_):
            return (silu_glu_eager(g_, u_) * cot).sum()

        dh = mx.grad(f_packed)(h)
        dg, du = mx.grad(f_ref, argnums=(0, 1))(g, u)
        mx.eval(dh, dg, du)
        for name, got, ref in (("dg", dh[..., :I], dg), ("du", dh[..., I:], du)):
            rel = ((got - ref).abs().max() / (ref.abs().max() + 1e-12)).item()
            self.assertLess(rel, 1e-5, name)

    def test_odd_last_dim_falls_back_without_error(self):
        h = mx.random.normal((4, 7)).astype(mx.float32)
        mx.eval(h)
        with self.assertRaises(ValueError):
            silu_glu_packed(h)

    def test_prewarm_packed_i512_stable(self):
        silu_mod._PACKED_FAILED.clear()
        for seed in range(12):
            mx.random.seed(seed)
            silu_mod._PACKED_VERIFIED.discard((mx.bfloat16, 512))
            self.assertTrue(
                silu_prewarm_packed(mx.bfloat16, [512]),
                f"seed={seed} failed={silu_mod._PACKED_FAILED}",
            )
        self.assertNotIn((mx.bfloat16, 512), silu_mod._PACKED_FAILED)
        self.assertFalse(silu_mod._DISABLED)

    def test_packed_fail_does_not_disable_two_arg(self):
        silu_mod._PACKED_FAILED.add((mx.bfloat16, 32))
        try:
            self.assertFalse(silu_mod._DISABLED)
            h = (mx.random.normal((4, 64)) * 0.5).astype(mx.bfloat16)
            y = silu_glu_packed(h)
            mx.eval(y)
            g = (mx.random.normal((32,)) * 0.5).astype(mx.bfloat16)
            u = (mx.random.normal((32,)) * 0.5).astype(mx.bfloat16)
            mx.eval(silu_two_arg(g, u))
            self.assertFalse(silu_mod._DISABLED)
        finally:
            silu_mod._PACKED_FAILED.discard((mx.bfloat16, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
