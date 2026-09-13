"""Moment diagnostics use the CPU and leave the optimization objective intact."""

import unittest
import mlx.core as mx
import numpy as np

from model.dpr import target_moments


class MomentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_isotropic_shifted_and_constant_targets(self):
        y = mx.concatenate([mx.eye(4), -mx.eye(4)], axis=0) * 2
        mask = mx.ones((8,))
        a = [float(v) for v in target_moments(y, mask)]
        np.testing.assert_allclose(a, [0, 1, 0, 1, 4], atol=1e-6)
        a = [float(v) for v in target_moments(y + 3, mask)]
        np.testing.assert_allclose(a, [9, 1, 9, 10, 4], atol=1e-6)
        a = [float(v) for v in target_moments(mx.full((8, 4), 3.0), mask)]
        np.testing.assert_allclose(a, [10, 0, 9, 9, 0], atol=1e-6)

    def test_empty_mask_and_original_regularizer_gradient(self):
        y = mx.array(np.arange(24, dtype=np.float32).reshape(6, 4) / 13)
        mask = mx.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        def old_reg(a):
            mu = mx.sum(a * mask[:, None], axis=0) / mask.sum()
            c = (a - mu) * mask[:, None]
            cov = c.T @ c / mask.sum()
            return mx.mean(mx.square(mu)) + mx.sum(mx.square(cov - mx.eye(4))) / 4

        np.testing.assert_allclose(
            mx.grad(old_reg)(y),
            mx.grad(lambda a: target_moments(a, mask)[0])(y),
            atol=1e-6,
        )
        a = [float(v) for v in target_moments(y, mx.zeros((6,)))]
        np.testing.assert_array_equal(a, [0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
