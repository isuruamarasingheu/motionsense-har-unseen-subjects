"""
Models package for Human Activity Recognition (HAR).

Architectures:
- numpy_mlp: From-scratch 1-hidden-layer MLP implemented purely with NumPy (tanh, cross-entropy, analytical backprop).
- mlp: Deep PyTorch MLP with BatchNorm, ReLU, and Dropout for 84 tabular features.
- cnn: 1D temporal Convolutional Neural Network processing raw (12, 128) multi-channel sensor sequences.
"""

from src.models.cnn import PyTorchCNN1D, generate_layer_table_cnn
from src.models.mlp import PyTorchMLP, generate_layer_table_mlp
from src.models.numpy_mlp import NumpyMLP

__all__ = [
    "NumpyMLP",
    "PyTorchMLP",
    "generate_layer_table_mlp",
    "PyTorchCNN1D",
    "generate_layer_table_cnn",
]
