"""
Unit tests for from-scratch NumPy Multi-Layer Perceptron (Task T2).

Tests:
1. Architecture and parameter shapes (84->32->6), float64 dtype, zero biases.
2. Initial loss close to theoretical random guessing ln(6) ≈ 1.7918.
3. Vectorised analytical gradients vs central-difference numerical gradients (tol < 1e-6).
4. Controlled overfitting on tiny batch (32 samples) driving loss near zero and acc=100%.
5. Parameter get/set mechanisms preserve state.
"""

from __future__ import annotations

import unittest
import numpy as np

from src.models.numpy_mlp import NumpyMLP, gradient_check, overfit_tiny_batch


class TestNumpyMLP(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 42
        self.model = NumpyMLP(n_in=84, n_hidden=32, n_out=6, seed=self.seed)

    def test_parameter_shapes_and_dtypes(self) -> None:
        """Verify parameter matrices have correct dimensions and float64 dtype."""
        self.assertEqual(self.model.W1.shape, (84, 32))
        self.assertEqual(self.model.b1.shape, (32,))
        self.assertEqual(self.model.W2.shape, (32, 6))
        self.assertEqual(self.model.b2.shape, (6,))

        self.assertEqual(self.model.W1.dtype, np.float64)
        self.assertEqual(self.model.b1.dtype, np.float64)
        self.assertEqual(self.model.W2.dtype, np.float64)
        self.assertEqual(self.model.b2.dtype, np.float64)

        # Biases initialized to zero
        np.testing.assert_array_equal(self.model.b1, np.zeros(32, dtype=np.float64))
        np.testing.assert_array_equal(self.model.b2, np.zeros(6, dtype=np.float64))

    def test_initial_loss_near_ln6(self) -> None:
        """Sanity check: Initial cross-entropy for 6 balanced classes ≈ ln(6) ≈ 1.7918."""
        rng = np.random.default_rng(self.seed)
        # Uniform zero-mean data
        X = rng.standard_normal((100, 84)).astype(np.float64)
        y = rng.integers(0, 6, size=100)

        P = self.model.forward(X)
        loss = self.model.loss(P, y)

        expected = np.log(6.0)
        # Should be within 0.3 of theoretical ln(6) for freshly initialized network
        self.assertAlmostEqual(loss, expected, delta=0.35)

    def test_gradient_checking_passes_all_parameters(self) -> None:
        """Verify analytical gradients match central differences with relative error < 1e-6."""
        rng = np.random.default_rng(self.seed)
        X_batch = rng.standard_normal((8, 84)).astype(np.float64)
        y_batch = rng.integers(0, 6, size=8)

        results = gradient_check(self.model, X_batch, y_batch, eps=1e-5, tol=1e-6)
        self.assertEqual(len(results), 8, "Expected 8 checked parameter entries.")

        for r in results:
            self.assertTrue(
                r["passed"],
                f"Grad check failed for {r['parameter']} index {r['index']}: "
                f"rel_diff={r['rel_diff']:.2e} > 1e-6 (ana={r['analytical']:.6e}, num={r['numerical']:.6e})",
            )
            self.assertLess(
                r["rel_diff"],
                1e-6,
                f"Relative error exceeds 1e-6: {r['rel_diff']}",
            )

    def test_overfit_tiny_batch_reaches_near_zero_loss(self) -> None:
        """Verify model memorizes 32 samples with loss < 0.01 and 100% accuracy."""
        rng = np.random.default_rng(self.seed)
        X_batch = rng.standard_normal((32, 84)).astype(np.float64)
        y_batch = rng.integers(0, 6, size=32)

        history = overfit_tiny_batch(self.model, X_batch, y_batch, lr=0.05, epochs=1000)
        self.assertLess(history["final_loss"], 0.01, "Loss must approach near-zero on 32 samples.")
        self.assertEqual(history["final_accuracy"], 1.0, "Accuracy must reach 100% on 32 samples.")

    def test_parameter_get_and_set(self) -> None:
        """Verify parameter cloning and restoring."""
        params_orig = self.model.get_params()
        self.model.W1 += 1.0
        self.assertFalse(np.allclose(self.model.W1, params_orig["W1"]))

        self.model.set_params(params_orig)
        np.testing.assert_array_equal(self.model.W1, params_orig["W1"])


if __name__ == "__main__":
    unittest.main()

