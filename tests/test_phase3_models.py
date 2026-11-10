"""
Unit tests for Phase 3 models and tuning pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.cnn import PyTorchCNN1D, generate_layer_table_cnn
from src.models.mlp import PyTorchMLP, generate_layer_table_mlp
from src.utils import load_config


def test_pytorch_mlp_forward_and_params() -> None:
    """Test PyTorch MLP forward pass dimensions and exact parameter count matching."""
    model = PyTorchMLP(input_dim=84, hidden_sizes=[128, 64], dropout=0.2, num_classes=6)
    x = torch.randn(8, 84)
    out = model(x)
    assert out.shape == (8, 6), f"Expected shape (8, 6), got {out.shape}"

    # Verify parameter count: (84*128 + 128) + (128*2) + (128*64 + 64) + (64*2) + (64*6 + 6)
    # fc1: 10,880; bn1: 256; fc2: 8,256; bn2: 128; classifier: 390 -> 19,910
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 19910, f"Expected 19,910 parameters, got {total_params}"

    table_md = generate_layer_table_mlp(model)
    assert "**Total Trainable Parameters**: 19,910" in table_md


def test_pytorch_cnn_forward_and_params() -> None:
    """Test PyTorch 1D-CNN with both channel-first and length-first inputs and param count."""
    model = PyTorchCNN1D(
        in_channels=12,
        sequence_length=128,
        filters=[64, 128, 128],
        kernel_size=5,
        dropout=0.3,
        num_classes=6,
    )
    # Channel-first: (batch, 12, 128)
    x1 = torch.randn(4, 12, 128)
    out1 = model(x1)
    assert out1.shape == (4, 6)

    # Length-first: (batch, 128, 12) auto-permuted
    x2 = torch.randn(4, 128, 12)
    out2 = model(x2)
    assert out2.shape == (4, 6)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 128454, f"Expected 128,454 parameters, got {total_params}"

    table_md = generate_layer_table_cnn(model)
    assert "**Total Trainable Parameters**: 128,454" in table_md


def test_tuning_log_completeness() -> None:
    """Verify that tuning_log.csv has all 18 declared trials without missing values in key metrics."""
    log_path = Path("outputs/metrics/tuning_log.csv")
    assert log_path.exists(), "tuning_log.csv does not exist"

    df = pd.read_csv(log_path)
    assert len(df) == 18, f"Expected 18 trials, got {len(df)}"

    expected_trials = {
        f"rf_{i:02d}" for i in range(1, 7)
    } | {
        f"mlp_{i:02d}" for i in range(1, 7)
    } | {
        f"cnn_{i:02d}" for i in range(1, 7)
    }

    actual_trials = set(df["trial_id"].tolist())
    assert actual_trials == expected_trials, f"Mismatch in trials: {expected_trials - actual_trials}"
    assert (df["val_macro_f1"] > 0.85).all(), "All trials should achieve > 0.85 Val Macro-F1"


def test_no_test_set_access_in_phase3_scripts() -> None:
    """Rule 7 check: Phase 3 scripts must never load or reference test data arrays."""
    phase3_files = [
        Path("src/train_baseline.py"),
        Path("src/train.py"),
        Path("src/tune.py"),
        Path("src/models/mlp.py"),
        Path("src/models/cnn.py"),
    ]

    forbidden_patterns = [
        r'np\.load\([\'"].*X_test',
        r'np\.load\([\'"].*y_test',
        r'windows_test\.csv',
    ]

    for p in phase3_files:
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            matches = re.findall(pat, content)
            assert len(matches) == 0, f"Found forbidden test set loading pattern '{pat}' in {p}: {matches}"
