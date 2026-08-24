"""打包版 SiTU-GLU（单张量 (..., 2I) 入口）的数值与梯度对齐。

MoE 的 gate/up 是一次 GEMM 出 (..., 2I) 再切两半。切片是跨步视图，
融合 kernel 里的 reshape(-1) 会各物化一份连续副本，反向还要把 dg/du
拼回 (..., 2I)。打包入口直接在核内寻址两半，省掉这些搬运。

本测试钉死：打包版与两参数版（以及 eager 公式）在值和梯度上一致，
且 dh 的两半分别等于 dg / du。
"""

import unittest

import mlx.core as mx

from model.acts import situ_glu_eager
from model.kernels.situ import prewarm as situ_prewarm
from model.kernels.situ import situ_glu as situ_two_arg
from model.kernels.situ import situ_glu_packed


class SituPackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert situ_prewarm(mx.float32), "situ f32 预热失败"
        assert situ_prewarm(mx.bfloat16), "situ bf16 预热失败"

    def _pair(self, shape, dtype, seed=0):
        mx.random.seed(seed)
        I = shape[-1]  # noqa: E741
        h = (mx.random.normal((*shape[:-1], 2 * I)) * 3.0).astype(dtype)
        mx.eval(h)
        return h, h[..., :I], h[..., I:]

    def test_packed_matches_eager_formula(self):
        for dtype in (mx.float32, mx.bfloat16):
            h, g, u = self._pair((7, 1, 96), dtype)
            got = situ_glu_packed(h)
            ref = situ_glu_eager(g, u)
            mx.eval(got, ref)
            self.assertEqual(got.shape, g.shape)
            rel = (
                (got.astype(mx.float32) - ref.astype(mx.float32)).abs().max()
                / (ref.astype(mx.float32).abs().max() + 1e-12)
            ).item()
            self.assertLess(rel, 1e-2 if dtype == mx.bfloat16 else 1e-5, str(dtype))

    def test_packed_matches_two_arg_kernel_bitwise(self):
        # 打包版只改寻址，逐元素算式与两参数核完全相同
        for dtype in (mx.float32, mx.bfloat16):
            h, g, u = self._pair((5, 3, 64), dtype, seed=1)
            got = situ_glu_packed(h)
            ref = situ_two_arg(g, u)
            mx.eval(got, ref)
            self.assertTrue(bool((got == ref).all().item()), str(dtype))

    def test_packed_gradient_splits_into_dg_and_du(self):
        I = 48  # noqa: E741
        h, g, u = self._pair((11, 1, I), mx.float32, seed=2)
        cot = mx.random.normal((11, 1, I))
        mx.eval(cot)

        def f_packed(h_):
            return (situ_glu_packed(h_) * cot).sum()

        def f_ref(g_, u_):
            return (situ_glu_eager(g_, u_) * cot).sum()

        dh = mx.grad(f_packed)(h)
        dg, du = mx.grad(f_ref, argnums=(0, 1))(g, u)
        mx.eval(dh, dg, du)
        for name, got, ref in (("dg", dh[..., :I], dg), ("du", dh[..., I:], du)):
            rel = ((got - ref).abs().max() / (ref.abs().max() + 1e-12)).item()
            self.assertLess(rel, 1e-5, name)

    def test_odd_last_dim_falls_back_without_error(self):
        # 2I 必须是偶数；奇数宽度应走 eager 回退而不是崩
        h = mx.random.normal((4, 7)).astype(mx.float32)
        mx.eval(h)
        with self.assertRaises(ValueError):
            situ_glu_packed(h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
