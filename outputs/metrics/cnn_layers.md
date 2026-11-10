# PyTorch 1D-CNN Layer Specification

- **Model Type**: 1D Temporal Convolutional Neural Network (`PyTorchCNN1D`)
- **Input Shape**: (1, 12, 128) [12 channels x 128 timesteps]
- **Filter Architecture**: [64, 128, 128]
- **Kernel Size**: 5
- **Dropout Rate**: 0.3
- **Output Classes**: 6
- **Total Trainable Parameters**: 128,454

| Layer | Layer Type | Output Shape | Activation | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| Input | Input | `(1, 12, 128)` | None | 0 |
| conv1 (Conv1d) | conv1 | `(1, 64, 128)` | None | 3,904 |
| bn1 (BatchNorm1d) | bn1 | `(1, 64, 128)` | None | 128 |
| act1 (ReLU) | act1 | `(1, 64, 128)` | ReLU | 0 |
| pool1 (MaxPool1d) | pool1 | `(1, 64, 64)` | MaxPool(s=2) | 0 |
| conv2 (Conv1d) | conv2 | `(1, 128, 64)` | None | 41,088 |
| bn2 (BatchNorm1d) | bn2 | `(1, 128, 64)` | None | 256 |
| act2 (ReLU) | act2 | `(1, 128, 64)` | ReLU | 0 |
| pool2 (MaxPool1d) | pool2 | `(1, 128, 32)` | MaxPool(s=2) | 0 |
| conv3 (Conv1d) | conv3 | `(1, 128, 32)` | None | 82,048 |
| bn3 (BatchNorm1d) | bn3 | `(1, 128, 32)` | None | 256 |
| act3 (ReLU) | act3 | `(1, 128, 32)` | ReLU | 0 |
| pool3 (MaxPool1d) | pool3 | `(1, 128, 16)` | MaxPool(s=2) | 0 |
| gap (AdaptiveAvgPool1d) | gap | `(1, 128, 1)` | GlobalAvgPool | 0 |
| flatten (Flatten) | flatten | `(1, 128)` | None | 0 |
| drop (Dropout) | drop | `(1, 128)` | Dropout(p=0.3) | 0 |
| classifier (Linear) | classifier | `(1, 6)` | Linear (Logits) | 774 |
| **Total** | | | | **128,454** |
