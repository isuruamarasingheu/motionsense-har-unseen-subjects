"""
Module: src.models.mlp
Purpose: T3b PyTorch Multi-Layer Perceptron for hand-crafted feature vectors.

Architecture:
- Input dimension: 84 tabular summary features.
- Hidden blocks:
    * Linear(84 -> h1) -> BatchNorm1d(h1) -> ReLU -> Dropout(p1)
    * Linear(h1 -> h2) -> BatchNorm1d(h2) -> ReLU -> Dropout(p2)
    * Linear(h2 -> 6) logits.
- Weight initialisation: Kaiming (He) normal initialisation for ReLU layers.
- Output: 6 unnormalised activity logits for CrossEntropyLoss.

Rule compliance:
- Rule 1: PyTorch CPU framework implementation.
- Rule 8: Architecture hyperparameters specified via YAML configuration.
- Rule 14: Clear modular PyTorch nn.Module design.
"""

