"""
Module: src.models.mlp
Purpose: T3b PyTorch Multi-Layer Perceptron for hand-crafted feature vectors.

Architecture:
- Input dimension: 84 tabular summary features (mean, std, min, max, median, IQR
  over 14 signals: 12 motion channels + |userAcc| + |rotRate|; no skewness).
- Hidden blocks:
    * Linear(84 -> h1) -> BatchNorm1d(h1) -> ReLU -> Dropout(p)
    * Linear(h1 -> h2) -> BatchNorm1d(h2) -> ReLU -> Dropout(p)
    * Linear(h2 -> num_classes) logits (default 6 classes).
- Weight initialisation: Kaiming (He) normal initialisation for linear layers followed
  by ReLU, with zeroed bias. Xavier initialisation for the classification head.
- Output: 6 unnormalised activity logits for CrossEntropyLoss.

Design constraints:
- PyTorch CPU framework implementation.
- Architecture hyperparameters specified via YAML configuration.
- Docstrings explain architectural choices and initialisation schemes.
- Clear modular PyTorch nn.Module design.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PyTorchMLP(nn.Module):
    """
    Two-hidden-layer Multi-Layer Perceptron with Batch Normalisation and Dropout.

    Why:
        Processes 84 hand-crafted statistical features (mean, std, min, max,
        median, IQR over 14 signals: 12 channels + |userAcc| + |rotRate|; no skewness)
        for human activity recognition. Dropout prevents overfitting on smaller activity classes.
    """

    def __init__(
        self,
        input_dim: int = 84,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.dropout_rate = dropout
        self.num_classes = num_classes

        if len(self.hidden_sizes) != 2:
            raise ValueError(
                f"Expected exactly 2 hidden layer sizes, got {self.hidden_sizes}"
            )

        h1, h2 = self.hidden_sizes

        # Block 1: Linear(84 -> h1) -> BatchNorm1d(h1) -> ReLU -> Dropout(p)
        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)

        # Block 2: Linear(h1 -> h2) -> BatchNorm1d(h2) -> ReLU -> Dropout(p)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout)

        # Output head: Linear(h2 -> num_classes)
        self.classifier = nn.Linear(h2, num_classes)

        # Initialise weights using Kaiming normal
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Kaiming (He) normal initialisation to hidden layers and Xavier to classifier.

        Why:
            Kaiming initialisation preserves activation variance across layers with ReLU
            non-linearities, avoiding vanishing or exploding gradients at step 0.
        """
        for fc in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(fc.weight, mode="fan_in", nonlinearity="relu")
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

        # Classifier head (linear output, no ReLU after)
        nn.init.xavier_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # Batch Normalisation default initialization (gamma=1, beta=0)
        for bn in (self.bn1, self.bn2):
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # Block 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        # Classification logits
        out = self.classifier(x)
        return out


def generate_layer_table_mlp(
    model: PyTorchMLP,
    batch_size: int = 1,
) -> str:
    """
    Generate a Markdown table describing the architecture, layer dimensions, activations,
    and parameter counts.

    Why:
        A verifiable layer table is saved to outputs/metrics/<model>_layers.md,
        confirming exact parameter counts and architectural compliance.
    """
    input_dim = model.input_dim
    h1, h2 = model.hidden_sizes
    num_classes = model.num_classes

    # Explicit layer specifications
    layers = [
        ("Input", f"({batch_size}, {input_dim})", "None", 0),
        ("fc1 (Linear)", f"({batch_size}, {h1})", "None", sum(p.numel() for p in model.fc1.parameters() if p.requires_grad)),
        ("bn1 (BatchNorm1d)", f"({batch_size}, {h1})", "None", sum(p.numel() for p in model.bn1.parameters() if p.requires_grad)),
        ("act1 (ReLU)", f"({batch_size}, {h1})", "ReLU", 0),
        ("drop1 (Dropout)", f"({batch_size}, {h1})", f"Dropout(p={model.dropout_rate})", 0),
        ("fc2 (Linear)", f"({batch_size}, {h2})", "None", sum(p.numel() for p in model.fc2.parameters() if p.requires_grad)),
        ("bn2 (BatchNorm1d)", f"({batch_size}, {h2})", "None", sum(p.numel() for p in model.bn2.parameters() if p.requires_grad)),
        ("act2 (ReLU)", f"({batch_size}, {h2})", "ReLU", 0),
        ("drop2 (Dropout)", f"({batch_size}, {h2})", f"Dropout(p={model.dropout_rate})", 0),
        ("classifier (Linear)", f"({batch_size}, {num_classes})", "Linear (Logits)", sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)),
    ]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    table_lines = [
        "# PyTorch MLP Layer Specification",
        "",
        f"- **Model Type**: Multi-Layer Perceptron (`PyTorchMLP`)",
        f"- **Input Dimension**: {input_dim} tabular features",
        f"- **Hidden Architecture**: [{h1}, {h2}]",
        f"- **Dropout Rate**: {model.dropout_rate}",
        f"- **Output Classes**: {num_classes}",
        f"- **Total Trainable Parameters**: {total_params:,}",
        "",
        "| Layer | Layer Type | Output Shape | Activation | Trainable Params |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, out_shape, act, params in layers:
        table_lines.append(f"| {name} | {name.split()[0]} | `{out_shape}` | {act} | {params:,} |")

    table_lines.append(f"| **Total** | | | | **{total_params:,}** |")
    table_lines.append("")

    return "\n".join(table_lines)
