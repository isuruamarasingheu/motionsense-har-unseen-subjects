# PyTorch MLP Layer Specification

- **Model Type**: Multi-Layer Perceptron (`PyTorchMLP`)
- **Input Dimension**: 84 tabular features
- **Hidden Architecture**: [128, 64]
- **Dropout Rate**: 0.2
- **Output Classes**: 6
- **Total Trainable Parameters**: 19,910

| Layer | Layer Type | Output Shape | Activation | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| Input | Input | `(1, 84)` | None | 0 |
| fc1 (Linear) | fc1 | `(1, 128)` | None | 10,880 |
| bn1 (BatchNorm1d) | bn1 | `(1, 128)` | None | 256 |
| act1 (ReLU) | act1 | `(1, 128)` | ReLU | 0 |
| drop1 (Dropout) | drop1 | `(1, 128)` | Dropout(p=0.2) | 0 |
| fc2 (Linear) | fc2 | `(1, 64)` | None | 8,256 |
| bn2 (BatchNorm1d) | bn2 | `(1, 64)` | None | 128 |
| act2 (ReLU) | act2 | `(1, 64)` | ReLU | 0 |
| drop2 (Dropout) | drop2 | `(1, 64)` | Dropout(p=0.2) | 0 |
| classifier (Linear) | classifier | `(1, 6)` | Linear (Logits) | 390 |
| **Total** | | | | **19,910** |
