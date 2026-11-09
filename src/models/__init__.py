"""
Models package for Human Activity Recognition (HAR).

Architectures:
- numpy_mlp: From-scratch 1-hidden-layer MLP implemented purely with NumPy (tanh, cross-entropy, analytical backprop).
- mlp: Deep PyTorch MLP with BatchNorm, ReLU, and Dropout for 84 tabular features.
- cnn: 1D temporal Convolutional Neural Network processing raw (12, 128) multi-channel sensor sequences.
"""

