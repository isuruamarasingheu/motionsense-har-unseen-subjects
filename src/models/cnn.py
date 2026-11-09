"""
Module: src.models.cnn
Purpose: T3b PyTorch 1D Temporal Convolutional Neural Network for raw sensor windows.

Architecture:
- Input shape: (batch_size, 12, 128) representing 12 sensor channels across 128 time steps.
- Feature extraction:
    * 3 cascaded Conv1D blocks:
      [Conv1d(in_c, out_c, kernel_size, padding="same") -> BatchNorm1d -> ReLU -> MaxPool1d(2)] x 3
    * Global Average Pooling (AdaptiveAvgPool1d(1)) reducing temporal dimension to 1.
- Classification head:
    * Flatten -> Dropout(p) -> Linear(last_channels -> 6) logits.
- Weight initialisation: Kaiming (He) normal initialisation for convolutional and linear layers.

Rule compliance:
- Rule 1: Meaningfully distinct temporal architecture from the static tabular MLP.
- Rule 8: Filter sizes, kernel sizes, and dropout configured via YAML files.
"""

