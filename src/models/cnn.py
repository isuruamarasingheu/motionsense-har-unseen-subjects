"""
Module: src.models.cnn
Purpose: T3b PyTorch 1D Temporal Convolutional Neural Network for raw sensor windows.

Architecture:
- Input shape: (batch_size, 12, 128) representing 12 sensor channels across 128 time steps.
- Feature extraction:
    * 3 cascaded Conv1D blocks:
      [Conv1d(in_c, out_c, kernel_size, padding="same") -> BatchNorm1d(out_c) -> ReLU -> MaxPool1d(2)] x 3
    * Global Average Pooling (AdaptiveAvgPool1d(1)) reducing temporal dimension to 1.
- Classification head:
    * Flatten -> Dropout(p) -> Linear(last_channels -> num_classes) logits (default 6).
- Weight initialisation: Kaiming (He) normal initialisation for convolutional and linear layers.

Rule compliance:
- Rule 1: Meaningfully distinct temporal architecture from the static tabular MLP.
- Rule 8: Filter sizes, kernel sizes, and dropout configured via YAML files.
- Rule 13: Docstrings explain architectural choices and initialisation schemes.
- Rule 14: Clear modular PyTorch nn.Module design.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PyTorchCNN1D(nn.Module):
    """
    Three-block 1D Convolutional Neural Network with Batch Normalisation,
    Max Pooling, Global Average Pooling, and Dropout for temporal motion classification.

    Why:
        Operates directly on the 12-channel continuous temporal sensor signals (attitude,
        rotation rate, acceleration, gravity over 128 time steps @ 50 Hz).
        1D temporal convolutions extract invariant local wavelets and frequency motifs,
        avoiding reliance on hand-engineered summary features.
    """

    def __init__(
        self,
        in_channels: int = 12,
        sequence_length: int = 128,
        filters: Sequence[int] = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.3,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.filters = list(filters)
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.num_classes = num_classes

        if len(self.filters) != 3:
            raise ValueError(
                f"Expected exactly 3 filter counts for the 3 Conv blocks, got {self.filters}"
            )

        f1, f2, f3 = self.filters

        # Block 1: Conv1d(12 -> f1) -> BatchNorm1d(f1) -> ReLU -> MaxPool1d(2)
        # padding='same' preserves length 128 before MaxPool -> 64
        self.conv1 = nn.Conv1d(in_channels, f1, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(f1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Block 2: Conv1d(f1 -> f2) -> BatchNorm1d(f2) -> ReLU -> MaxPool1d(2)
        # length 64 -> 32
        self.conv2 = nn.Conv1d(f1, f2, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(f2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Block 3: Conv1d(f2 -> f3) -> BatchNorm1d(f3) -> ReLU -> MaxPool1d(2)
        # length 32 -> 16
        self.conv3 = nn.Conv1d(f2, f3, kernel_size=kernel_size, padding="same")
        self.bn3 = nn.BatchNorm1d(f3)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global Average Pooling: (batch, f3, 16) -> (batch, f3, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Classification Head: Dropout(p) -> Linear(f3 -> num_classes)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(f3, num_classes)

        # Apply Kaiming initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Kaiming (He) normal initialisation to Conv1d layers and Xavier to Linear.

        Why:
            Kaiming initialisation is optimal for layers followed by ReLU activations,
            preventing vanishing gradients across the 3 stacked temporal conv blocks.
        """
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # BatchNorm initialization
        for bn in (self.bn1, self.bn2, self.bn3):
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

        # Classifier head
        nn.init.xavier_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 1D-CNN.

        Args:
            x: Input tensor. Expected shape (batch_size, in_channels, sequence_length),
               or (batch_size, sequence_length, in_channels).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # If input has shape (batch, sequence_length, in_channels), transpose to (batch, in_channels, sequence_length)
        if x.ndim == 3 and x.shape[1] == self.sequence_length and x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        # Pooling & Head
        x = self.gap(x)
        x = self.flatten(x)
        x = self.drop(x)
        out = self.classifier(x)
        return out


def generate_layer_table_cnn(
    model: PyTorchCNN1D,
    batch_size: int = 1,
) -> str:
    """
    Generate a Markdown table describing the architecture, layer dimensions, activations,
    and parameter counts for PyTorchCNN1D.

    Why:
        Plan Section 3b requires a verifiable layer table saved to outputs/metrics/<model>_layers.md,
        confirming exact parameter counts and architectural compliance.
    """
    c_in = model.in_channels
    seq_len = model.sequence_length
    f1, f2, f3 = model.filters
    k = model.kernel_size
    num_classes = model.num_classes

    # Track temporal dimensions through the 3 blocks
    len_b1 = seq_len // 2  # 64
    len_b2 = len_b1 // 2    # 32
    len_b3 = len_b2 // 2    # 16

    layers = [
        ("Input", f"({batch_size}, {c_in}, {seq_len})", "None", 0),
        ("conv1 (Conv1d)", f"({batch_size}, {f1}, {seq_len})", "None", sum(p.numel() for p in model.conv1.parameters() if p.requires_grad)),
        ("bn1 (BatchNorm1d)", f"({batch_size}, {f1}, {seq_len})", "None", sum(p.numel() for p in model.bn1.parameters() if p.requires_grad)),
        ("act1 (ReLU)", f"({batch_size}, {f1}, {seq_len})", "ReLU", 0),
        ("pool1 (MaxPool1d)", f"({batch_size}, {f1}, {len_b1})", "MaxPool(s=2)", 0),
        ("conv2 (Conv1d)", f"({batch_size}, {f2}, {len_b1})", "None", sum(p.numel() for p in model.conv2.parameters() if p.requires_grad)),
        ("bn2 (BatchNorm1d)", f"({batch_size}, {f2}, {len_b1})", "None", sum(p.numel() for p in model.bn2.parameters() if p.requires_grad)),
        ("act2 (ReLU)", f"({batch_size}, {f2}, {len_b1})", "ReLU", 0),
        ("pool2 (MaxPool1d)", f"({batch_size}, {f2}, {len_b2})", "MaxPool(s=2)", 0),
        ("conv3 (Conv1d)", f"({batch_size}, {f3}, {len_b2})", "None", sum(p.numel() for p in model.conv3.parameters() if p.requires_grad)),
        ("bn3 (BatchNorm1d)", f"({batch_size}, {f3}, {len_b2})", "None", sum(p.numel() for p in model.bn3.parameters() if p.requires_grad)),
        ("act3 (ReLU)", f"({batch_size}, {f3}, {len_b2})", "ReLU", 0),
        ("pool3 (MaxPool1d)", f"({batch_size}, {f3}, {len_b3})", "MaxPool(s=2)", 0),
        ("gap (AdaptiveAvgPool1d)", f"({batch_size}, {f3}, 1)", "GlobalAvgPool", 0),
        ("flatten (Flatten)", f"({batch_size}, {f3})", "None", 0),
        ("drop (Dropout)", f"({batch_size}, {f3})", f"Dropout(p={model.dropout_rate})", 0),
        ("classifier (Linear)", f"({batch_size}, {num_classes})", "Linear (Logits)", sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)),
    ]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    table_lines = [
        "# PyTorch 1D-CNN Layer Specification",
        "",
        f"- **Model Type**: 1D Temporal Convolutional Neural Network (`PyTorchCNN1D`)",
        f"- **Input Shape**: ({batch_size}, {c_in}, {seq_len}) [12 channels x 128 timesteps]",
        f"- **Filter Architecture**: [{f1}, {f2}, {f3}]",
        f"- **Kernel Size**: {k}",
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
