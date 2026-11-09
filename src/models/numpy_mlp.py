"""
Module: src.models.numpy_mlp
Purpose: T2 From-scratch Multi-Layer Perceptron built purely in NumPy (no PyTorch, no autograd).

Responsibilities:
- Implement class `NumpyMLP(n_in=84, n_hidden=32, n_out=6, seed)` using float64 precision:
    * Parameters: W1 (84, 32), b1 (32,), W2 (32, 6), b2 (6,).
    * Xavier / Glorot weight initialisation tailored for tanh activation.
    * Forward pass: Z1 = X @ W1 + b1; H = tanh(Z1); Z2 = H @ W2 + b2; P = softmax(Z2).
    * Loss: Numerically stable multiclass cross-entropy.
    * Backward pass: Vectorised analytical gradients for dW2, db2, dW1, db1 without autograd.
    * Parameter update: Mini-batch gradient descent step with specified learning rate.
- Implement numerical gradient checking via central differences (eps=1e-5) verifying analytical
  derivatives to relative error < 1e-6 across W1, b1, W2, b2.
- Implement controlled overfitting verification on tiny mini-batch (approaching near-zero loss).

Rule compliance:
- Rule 1: Use NumPy float64 for from-scratch mathematical precision.
- Rule 10: Log exact numerical check outputs to outputs/metrics/numpy_mlp.json.
- Rule 13: Document all array dimensions and mathematical calculus derivations in comments.
"""

