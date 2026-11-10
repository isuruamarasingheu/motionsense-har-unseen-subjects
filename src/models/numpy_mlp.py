"""
Module: src.models.numpy_mlp
Purpose: From-scratch Multi-Layer Perceptron (MLP) implemented purely in NumPy
         (no PyTorch, no autograd), adhering to float64 precision.

Coursework: NB6057CEM Artificial Neural Networks — CW1 Practical (Task T2)

Mathematical Foundations:
-------------------------
1. Parameters and Architecture:
   - Input dimension: n_in = 84 (hand-crafted time-domain statistical features)
   - Hidden layer dimension: n_hidden = 32
   - Output classes: n_out = 6 (dws, ups, wlk, jog, std, sit)
   - Parameters:
       * W1: shape (84, 32), float64, Xavier / Glorot uniform initialization for tanh
       * b1: shape (32,),    float64, initialized to zeros
       * W2: shape (32, 6),  float64, Xavier / Glorot uniform initialization for tanh
       * b2: shape (6,),     float64, initialized to zeros

2. Forward Propagation:
   - Z1 = X @ W1 + b1          [Shape: (N, 32)]
   - H  = tanh(Z1)             [Shape: (N, 32)]
   - Z2 = H @ W2 + b2          [Shape: (N, 6)]
   - P  = softmax(Z2)          [Shape: (N, 6)]
     Numerically stable softmax: P_ik = exp(Z2_ik - max_j Z2_ij) / sum_m exp(Z2_im - max_j Z2_ij)

3. Loss Function (Multiclass Cross-Entropy):
   - L = - (1 / N) * sum_{i=1}^N sum_{c=0}^5 Y_onehot_{i, c} * ln(P_{ic})
   - Initial loss sanity check: For 6 equally probable classes, L_init ≈ ln(6) ≈ 1.7918.

4. Backward Propagation (Analytical Vectorised Gradients):
   - dZ2 = (P - Y_onehot) / N              [Shape: (N, 6)]  (includes 1/N factor)
   - dW2 = H.T @ dZ2                       [Shape: (32, 6)]
   - db2 = sum_{rows}(dZ2)                 [Shape: (6,)]
   - dH  = dZ2 @ W2.T                      [Shape: (N, 32)]
   - dZ1 = dH * (1 - H**2)                 [Shape: (N, 32)] (derivative of tanh: d/dz tanh(z) = 1 - tanh^2(z))
   - dW1 = X.T @ dZ1                       [Shape: (84, 32)]
   - db1 = sum_{rows}(dZ1)                 [Shape: (32,)]

5. Parameter Update (Mini-batch Gradient Descent):
   - theta = theta - lr * dtheta for theta in {W1, b1, W2, b2}

Design constraints:
- Built strictly with NumPy (float64) without autograd or torch dependencies.
- Numerical verification logged to outputs/metrics/numpy_mlp.json.
- Docstrings explicitly document array dimensions and mathematical calculus derivations.
- Clear, readable, and explainable structure.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("src.models.numpy_mlp")


class NumpyMLP:
    """
    Single-hidden-layer Multi-Layer Perceptron built strictly from first principles
    using NumPy float64 arithmetic.

    Architecture:
        Input (84) -> Dense(32) + tanh -> Dense(6) + softmax -> Cross-Entropy Loss
    """

    def __init__(
        self,
        n_in: int = 84,
        n_hidden: int = 32,
        n_out: int = 6,
        seed: int = 42,
    ) -> None:
        """
        Initialize the MLP parameters with Xavier/Glorot uniform initialization.

        Why Xavier/Glorot for tanh:
            The hyperbolic tangent activation function is symmetric around zero and has
            a derivative of 1 at z = 0. To prevent the variance of activations and gradients
            from exploding or vanishing as signals propagate forward and backward through
            the network, Glorot & Bengio (2010) proved that weights should be drawn from a
            distribution with variance Var(W) = 2 / (fan_in + fan_out). For a uniform
            distribution U(-a, a), Var(W) = a^2 / 3, yielding the limit:
                limit = sqrt(6 / (fan_in + fan_out))

        Args:
            n_in: Number of input features (default: 84).
            n_hidden: Number of hidden neurons (default: 32).
            n_out: Number of target classes (default: 6).
            seed: Random seed for parameter initialization.
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Xavier / Glorot uniform initialization for W1 (fan_in=84, fan_out=32)
        limit1 = np.sqrt(6.0 / (n_in + n_hidden))
        self.W1: np.ndarray = rng.uniform(-limit1, limit1, size=(n_in, n_hidden)).astype(np.float64)
        self.b1: np.ndarray = np.zeros(n_hidden, dtype=np.float64)

        # Xavier / Glorot uniform initialization for W2 (fan_in=32, fan_out=6)
        limit2 = np.sqrt(6.0 / (n_hidden + n_out))
        self.W2: np.ndarray = rng.uniform(-limit2, limit2, size=(n_hidden, n_out)).astype(np.float64)
        self.b2: np.ndarray = np.zeros(n_out, dtype=np.float64)

        # Cache variables for forward activations and backprop gradients
        self.X: np.ndarray | None = None
        self.Z1: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.Z2: np.ndarray | None = None
        self.P: np.ndarray | None = None

        self.grads: dict[str, np.ndarray] = {
            "dW1": np.zeros_like(self.W1),
            "db1": np.zeros_like(self.b1),
            "dW2": np.zeros_like(self.W2),
            "db2": np.zeros_like(self.b2),
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Execute the forward propagation pass.

        Mathematical steps:
            1. Z1 = X @ W1 + b1
               Linear combination of input features.
               Shape: (N, 84) @ (84, 32) + (32,) -> (N, 32)
            2. H = tanh(Z1)
               Hyperbolic tangent non-linearity, mapping values to (-1, 1).
               Shape: (N, 32)
            3. Z2 = H @ W2 + b2
               Linear projection to class logits.
               Shape: (N, 32) @ (32, 6) + (6,) -> (N, 6)
            4. P = softmax(Z2)
               Numerically stable softmax normalisation. Subtracting max(Z2) prevents
               floating-point overflow in exp(Z2).
               Shape: (N, 6)

        Args:
            X: Input feature matrix of shape (N, 84), dtype float64.

        Returns:
            P: Predicted class probability distribution of shape (N, 6), dtype float64.
        """
        # Ensure input is float64
        self.X = np.asarray(X, dtype=np.float64)

        # Hidden layer pre-activation: (N, 32)
        self.Z1 = self.X @ self.W1 + self.b1

        # Hidden layer activation: (N, 32)
        self.H = np.tanh(self.Z1)

        # Output layer pre-activation (logits): (N, 6)
        self.Z2 = self.H @ self.W2 + self.b2

        # Numerically stable softmax:
        # Shift logits by subtracting row-wise maximum before exponentiating
        logits_max = np.max(self.Z2, axis=1, keepdims=True)
        exp_z = np.exp(self.Z2 - logits_max)
        self.P = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return self.P

    def loss(self, P: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean multiclass cross-entropy loss.

        Formula:
            L = - (1 / N) * sum_{i=1}^N ln(P_{i, y_i})

        Numerical Stability:
            Probabilities are clipped to [1e-15, 1.0 - 1e-15] to prevent log(0)
            underflow producing NaN / Inf.

        Args:
            P: Predicted probabilities of shape (N, 6), dtype float64.
            y: Integer class target indices of shape (N,), values in {0, ..., 5}.

        Returns:
            Scalar cross-entropy loss, float.
        """
        N = y.shape[0]
        if N == 0:
            return 0.0

        eps = 1e-15
        P_clipped = np.clip(P, eps, 1.0 - eps)
        log_likelihood = -np.log(P_clipped[np.arange(N), y])
        return float(np.mean(log_likelihood))

    def backward(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute analytical vectorised gradients via the chain rule (backpropagation).

        Mathematical Derivation:
            Let N be the batch size.
            1. Loss derivative with respect to output logits Z2:
               dL / dZ2_ik = (1 / N) * (P_ik - Y_onehot_ik)
               Vectorised: dZ2 = (P - Y_onehot) / N
               Shape: (N, 6)
            2. Gradients for output layer parameters:
               dW2 = H.T @ dZ2
               Shape: (32, N) @ (N, 6) -> (32, 6)
               db2 = sum_{rows}(dZ2)
               Shape: (6,)
            3. Error propagated back to hidden activations:
               dH = dZ2 @ W2.T
               Shape: (N, 6) @ (6, 32) -> (N, 32)
            4. Gradient through the tanh activation:
               Since d/dz tanh(z) = 1 - tanh^2(z) = 1 - H^2,
               dZ1 = dH * (1 - H**2)
               Shape: (N, 32)
            5. Gradients for hidden layer parameters:
               dW1 = X.T @ dZ1
               Shape: (84, N) @ (N, 32) -> (84, 32)
               db1 = sum_{rows}(dZ1)
               Shape: (32,)

        Args:
            y: Ground-truth class labels of shape (N,), integer indices in {0, ..., 5}.

        Returns:
            Dictionary containing computed parameter gradients:
                {'dW1': (84, 32), 'db1': (32,), 'dW2': (32, 6), 'db2': (6,)}.
        """
        if self.X is None or self.H is None or self.P is None:
            raise RuntimeError("Must call forward(X) before calling backward(y).")

        N = self.X.shape[0]

        # Construct one-hot target matrix: shape (N, n_out)
        Y_onehot = np.zeros((N, self.n_out), dtype=np.float64)
        Y_onehot[np.arange(N), y] = 1.0

        # Vectorised output gradient: (N, 6)
        # Note: 1/N is included directly here so that all parameter gradients
        # are automatically normalized by the batch size.
        dZ2 = (self.P - Y_onehot) / N

        # Output layer parameter gradients
        dW2 = self.H.T @ dZ2               # Shape: (32, 6)
        db2 = np.sum(dZ2, axis=0)          # Shape: (6,)

        # Backpropagate through linear combination W2: (N, 32)
        dH = dZ2 @ self.W2.T

        # Backpropagate through tanh non-linearity: d/dz tanh(z) = 1 - tanh^2(z)
        dZ1 = dH * (1.0 - self.H ** 2)     # Shape: (N, 32)

        # Hidden layer parameter gradients
        dW1 = self.X.T @ dZ1               # Shape: (84, 32)
        db1 = np.sum(dZ1, axis=0)          # Shape: (32,)

        # Store in grads dictionary
        self.grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }
        return self.grads

    def step(self, lr: float) -> None:
        """
        Execute mini-batch gradient descent update on all trainable parameters.

        Formulas:
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2

        Args:
            lr: Learning rate (float, e.g. 0.05).
        """
        self.W1 -= lr * self.grads["dW1"]
        self.b1 -= lr * self.grads["db1"]
        self.W2 -= lr * self.grads["dW2"]
        self.b2 -= lr * self.grads["db2"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted class probabilities for input samples.

        Args:
            X: Input feature matrix of shape (N, 84).

        Returns:
            Probability matrix of shape (N, 6).
        """
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discrete class indices by selecting the argmax probability.

        Args:
            X: Input feature matrix of shape (N, 84).

        Returns:
            Class predictions of shape (N,), integer indices in {0, ..., 5}.
        """
        P = self.forward(X)
        return np.argmax(P, axis=1)

    def get_params(self) -> dict[str, np.ndarray]:
        """Return a deep copy of all current model weights and biases."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Load weights and biases from a parameter dictionary."""
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()


def gradient_check(
    model: NumpyMLP,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-6,
    check_entries: list[tuple[str, tuple[int, ...]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform rigorous numerical gradient checking via two-sided central differences.

    Why Central Differences:
        Taylor theorem guarantees that central differences have quadratic truncation
        error O(eps^2), compared to one-sided differences which have linear error O(eps):
            f'(theta) = [f(theta + eps) - f(theta - eps)] / (2 * eps) + O(eps^2)
        With eps = 1e-5 in float64, numerical error is on the order of 1e-10, enabling
        precise validation against analytical derivatives.
        Because tanh is infinitely differentiable (smooth C^inf), there are no ReLU
        non-differentiable kink points.

    Relative Error Metric:
        rel_diff = |analytical - numerical| / max(|analytical| + |numerical|, 1e-12)
        Tolerance threshold: rel_diff < 1e-6 for all checked parameters.

    Args:
        model: NumpyMLP instance with initialized or trained parameters.
        X_batch: Sample batch features of shape (N, 84), e.g. N = 8.
        y_batch: Sample batch targets of shape (N,), values in {0, ..., 5}.
        eps: Perturbation step size (default: 1e-5).
        tol: Acceptance relative difference threshold (default: 1e-6).
        check_entries: Optional list of tuples (param_name, index_tuple).
                       Defaults to at least 2 entries each for W1, b1, W2, b2 (8 total).

    Returns:
        List of dictionaries detailing analytical vs numerical values and pass/fail status.
    """
    if check_entries is None:
        # Default 8 entries across all 4 parameter arrays
        check_entries = [
            ("W1", (0, 0)),
            ("W1", (42, 16)),
            ("b1", (0,)),
            ("b1", (15,)),
            ("W2", (0, 0)),
            ("W2", (16, 3)),
            ("b2", (0,)),
            ("b2", (3,)),
        ]

    # 1. Forward and backward passes to obtain analytical gradients
    P = model.forward(X_batch)
    _ = model.loss(P, y_batch)
    grads = model.backward(y_batch)

    results: list[dict[str, Any]] = []

    # Helper function to compute scalar loss for perturbed parameters
    def compute_loss_at_param(param_name: str, index: tuple[int, ...], delta: float) -> float:
        param_arr = getattr(model, param_name)
        original_val = param_arr[index]
        param_arr[index] = original_val + delta

        # Recompute forward pass without updating self.X or cached gradients
        Z1 = model.X @ model.W1 + model.b1
        H = np.tanh(Z1)
        Z2 = H @ model.W2 + model.b2
        logits_max = np.max(Z2, axis=1, keepdims=True)
        exp_z = np.exp(Z2 - logits_max)
        P_perturbed = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        loss_val = model.loss(P_perturbed, y_batch)

        # Restore original parameter value
        param_arr[index] = original_val
        return loss_val

    for param_name, index in check_entries:
        ana_val = float(grads[f"d{param_name}"][index])

        # Central difference: [L(theta + eps) - L(theta - eps)] / (2 * eps)
        loss_pos = compute_loss_at_param(param_name, index, +eps)
        loss_neg = compute_loss_at_param(param_name, index, -eps)
        num_val = float((loss_pos - loss_neg) / (2.0 * eps))

        abs_diff = float(abs(ana_val - num_val))
        denom = max(abs(ana_val) + abs(num_val), 1e-12)
        rel_diff = float(abs_diff / denom)
        passed = bool(rel_diff < tol)

        entry_record = {
            "parameter": param_name,
            "index": list(index),
            "analytical": ana_val,
            "numerical": num_val,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "passed": passed,
        }
        results.append(entry_record)

    return results


def overfit_tiny_batch(
    model: NumpyMLP,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    lr: float = 0.05,
    epochs: int = 2000,
) -> dict[str, Any]:
    """
    Controlled sanity check: overfit a tiny batch of 32 samples for 2000 epochs.

    Why this check is essential:
        Before running full-scale training on thousands of samples, verifying that
        a model can drive the loss to near zero (< 1e-3) and reach 100% training
        accuracy on a tiny subset proves that:
        1. Forward and backward computations are mathematically consistent.
        2. Gradient descent updates are stepping in the true descent direction.
        3. Model capacity is sufficient to memorize complex sensor patterns.

    Args:
        model: NumpyMLP instance.
        X_batch: Tiny subset features of shape (32, 84).
        y_batch: Tiny subset class targets of shape (32,).
        lr: Learning rate (default: 0.05).
        epochs: Number of iterations without early stopping (default: 2000).

    Returns:
        Dictionary containing epochs list, loss history, accuracy history,
        initial loss, and final loss.
    """
    N = y_batch.shape[0]
    epoch_list: list[int] = []
    loss_history: list[float] = []
    acc_history: list[float] = []

    for ep in range(epochs):
        P = model.forward(X_batch)
        curr_loss = model.loss(P, y_batch)
        preds = np.argmax(P, axis=1)
        curr_acc = float(np.mean(preds == y_batch))

        model.backward(y_batch)
        model.step(lr)

        epoch_list.append(ep + 1)
        loss_history.append(curr_loss)
        acc_history.append(curr_acc)

    return {
        "epochs": epoch_list,
        "loss_history": loss_history,
        "acc_history": acc_history,
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "final_accuracy": float(acc_history[-1]),
    }


def train_numpy_mlp(
    model: NumpyMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    lr: float = 0.05,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Mini-batch gradient descent training loop with early stopping monitoring val loss.

    Strategy:
        - Mini-batch shuffling at each epoch using a fixed random generator.
        - Training on stratified subset (up to 3000 windows) and evaluation on full val set.
        - Early stopping: stops when validation loss fails to decrease for `patience`
          consecutive epochs.
        - Restores best model parameters encountered during training.
        - Tracks macro-averaged F1 score on validation set for diagnostic tracking.

    Args:
        model: NumpyMLP instance.
        X_train: Training features, shape (N_train, 84).
        y_train: Training labels, shape (N_train,).
        X_val: Validation features, shape (N_val, 84).
        y_val: Validation labels, shape (N_val,).
        batch_size: Size of mini-batches (default: 64).
        lr: Learning rate (default: 0.05).
        max_epochs: Maximum epochs (default: 200).
        patience: Consecutive non-improving epochs before early stopping (default: 15).
        seed: Shuffling seed for reproducibility.
        verbose: If True, prints epoch progress.

    Returns:
        Dictionary containing training and validation progression metrics,
        best epoch, actual completed epochs, and final evaluation scores.
    """
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    N_train = X_train.shape[0]
    n_batches = int(np.ceil(N_train / batch_size))

    # Evaluate initial loss before any parameter update
    init_P_tr = model.forward(X_train)
    init_loss_tr = model.loss(init_P_tr, y_train)
    init_P_va = model.forward(X_val)
    init_loss_va = model.loss(init_P_va, y_val)
    init_val_preds = np.argmax(init_P_va, axis=1)
    init_val_acc = float(np.mean(init_val_preds == y_val))
    init_val_f1 = float(f1_score(y_val, init_val_preds, average="macro"))

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = model.get_params()
    best_epoch = 0

    history: dict[str, list[Any]] = {
        "epoch": [0],
        "train_loss": [float(init_loss_tr)],
        "train_loss_batch_avg": [float(init_loss_tr)],
        "val_loss": [float(init_loss_va)],
        "val_acc": [init_val_acc],
        "val_macro_f1": [init_val_f1],
    }

    if verbose:
        logger.info(
            f"Starting NumpyMLP Training | N_train={N_train}, N_val={len(y_val)}, "
            f"batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}, patience={patience}"
        )
        logger.info(f"Initial State (Epoch 0): Train Loss={init_loss_tr:.4f}, Val Loss={init_loss_va:.4f}")

    actual_epochs = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch

        # Mini-batch shuffle
        perm = rng.permutation(N_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss_accum = 0.0

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, N_train)
            X_b = X_shuffled[start_idx:end_idx]
            y_b = y_shuffled[start_idx:end_idx]

            # Forward pass
            P_b = model.forward(X_b)
            batch_loss = model.loss(P_b, y_b)
            epoch_loss_accum += batch_loss * len(y_b)

            # Backward pass & parameter step
            model.backward(y_b)
            model.step(lr)

        # Running average of mini-batch losses during the epoch
        batch_avg_loss = epoch_loss_accum / N_train

        # Full forward pass over train subset at the END of each epoch (comparable to val)
        P_tr_end = model.forward(X_train)
        train_loss = model.loss(P_tr_end, y_train)

        # Validation evaluation at the end of each epoch
        val_P = model.forward(X_val)
        val_loss = model.loss(val_P, y_val)
        val_preds = np.argmax(val_P, axis=1)
        val_acc = float(np.mean(val_preds == y_val))
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_loss_batch_avg"].append(float(batch_avg_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.get_params()
            best_epoch = epoch
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or patience_counter >= patience):
            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | Train Loss (end): {train_loss:.4f} | "
                f"Batch Avg Loss: {batch_avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            stopped_early = True
            if verbose:
                logger.info(f"Early stopping triggered at epoch {epoch}! Restoring best epoch {best_epoch}.")
            break

    # Restore parameters from the epoch with lowest validation loss
    model.set_params(best_weights)

    # Re-evaluate final performance using best weights
    final_val_P = model.forward(X_val)
    final_val_loss = model.loss(final_val_P, y_val)
    final_val_preds = np.argmax(final_val_P, axis=1)
    final_val_acc = float(np.mean(final_val_preds == y_val))
    final_val_f1 = float(f1_score(y_val, final_val_preds, average="macro"))

    return {
        "initial_train_loss": float(init_loss_tr),
        "initial_val_loss": float(init_loss_va),
        "final_train_loss": float(history["train_loss"][best_epoch]),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": final_val_acc,
        "final_val_macro_f1": final_val_f1,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(actual_epochs),
        "stopped_early": stopped_early,
        "history": history,
    }
