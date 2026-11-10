"""
Module: src.data.features
Purpose: Hand-crafted statistical feature extraction for tabular and MLP models.

Responsibilities:
- Transform raw window arrays (N, 128, 12) into tabular feature matrices (N, 84).
- Derive 2 composite magnitude channels from the 12 primary channels:
    * |userAcceleration| = sqrt(userAcceleration.x^2 + y^2 + z^2) (channels 9, 10, 11)
    * |rotationRate| = sqrt(rotationRate.x^2 + y^2 + z^2) (channels 6, 7, 8)
  yielding 14 total time-series signals per window.
- Compute 6 summary statistics per signal over the window duration (128 samples):
    1. Mean
    2. Standard deviation (std)
    3. Minimum (min)
    4. Maximum (max)
    5. Median (median)
    6. Interquartile Range (IQR = Q75 - Q25)
  yielding exactly 14 signals x 6 statistics = 84 features per window.
- Ensure identical feature representation across Random Forest, PyTorch MLP, and NumPy MLP.

Design constraints:
- Feature definitions and mathematical derivations documented clearly.
- Vectorised, clean, and explainable NumPy implementation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.data.load import SENSOR_COLUMNS

logger = logging.getLogger(__name__)

# Names of the 14 signals (12 sensor channels + 2 composite magnitudes)
SIGNAL_NAMES: list[str] = list(SENSOR_COLUMNS) + [
    "userAcceleration.mag",
    "rotationRate.mag",
]

# Names of the 6 summary statistics
STAT_NAMES: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
]


def get_feature_names() -> list[str]:
    """
    Return the ordered list of all 84 feature names.

    Why:
        Preserves feature interpretability and column tracking when feeding tabular
        features into Random Forest (feature importances) or inspecting learned weights.

    Returns:
        List of 84 string feature names formatted as '<signal>_<statistic>'.
    """
    feature_names: list[str] = []
    for sig in SIGNAL_NAMES:
        for stat in STAT_NAMES:
            feature_names.append(f"{sig}_{stat}")
    return feature_names


def extract_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute 84 hand-crafted statistical features from raw sensor windows.

    Mathematical derivations:
        1. Magnitude channels:
           * |userAcceleration| = sqrt(userAcceleration.x^2 + y^2 + z^2)
           * |rotationRate| = sqrt(rotationRate.x^2 + y^2 + z^2)
           Magnitudes provide orientation-invariant measures of body movement intensity.

        2. Summary statistics (for each of 14 signals over 128 samples):
           * mean:   mu = (1/T) * sum(x_t)
           * std:    sigma = sqrt((1/T) * sum((x_t - mu)^2))
           * min:    min(x_t)
           * max:    max(x_t)
           * median: 50th percentile of x_t
           * IQR:    75th percentile - 25th percentile (robust dispersion metric)

    Args:
        X_raw: NumPy array of shape (N, 128, 12) or (128, 12).

    Returns:
        NumPy array of shape (N, 84) (or (1, 84) for single window) of dtype float64.
    """
    arr = np.asarray(X_raw)

    is_single_window = False
    if arr.ndim == 2:
        # Single window (128, 12) -> expand to (1, 128, 12)
        arr = np.expand_dims(arr, axis=0)
        is_single_window = True
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D array (N, 128, 12) or 2D array (128, 12), got shape {arr.shape}")

    n_windows, window_len, n_channels = arr.shape
    if n_channels != 12:
        raise ValueError(f"Expected 12 sensor channels, got {n_channels}")

    if n_windows == 0:
        return np.empty((0, len(SIGNAL_NAMES) * len(STAT_NAMES)), dtype=np.float64)

    # Convert to float64 for precision in statistics
    arr = arr.astype(np.float64)

    # 1. Compute composite magnitudes
    # userAcceleration: channels 9, 10, 11
    user_acc = arr[:, :, 9:12]
    user_acc_mag = np.linalg.norm(user_acc, axis=2, keepdims=True)  # (N, 128, 1)

    # rotationRate: channels 6, 7, 8
    rot_rate = arr[:, :, 6:9]
    rot_rate_mag = np.linalg.norm(rot_rate, axis=2, keepdims=True)  # (N, 128, 1)

    # Concatenate to form 14 time-series signals
    # Shape: (N, 128, 14)
    signals = np.concatenate([arr, user_acc_mag, rot_rate_mag], axis=2)

    # 2. Compute 6 summary statistics across axis 1 (time dimension, 128 samples)
    stat_mean = np.mean(signals, axis=1)                                # (N, 14)
    stat_std = np.std(signals, axis=1, ddof=0)                          # (N, 14)
    stat_min = np.min(signals, axis=1)                                  # (N, 14)
    stat_max = np.max(signals, axis=1)                                  # (N, 14)
    q25, stat_median, q75 = np.percentile(signals, [25, 50, 75], axis=1) # each (N, 14)
    stat_iqr = q75 - q25                                                # (N, 14)

    # 3. Stack statistics per signal: for each signal, [mean, std, min, max, median, iqr]
    # Shape: (N, 14, 6) -> reshape to (N, 84)
    stats_stacked = np.stack(
        [stat_mean, stat_std, stat_min, stat_max, stat_median, stat_iqr],
        axis=2,
    )
    features = stats_stacked.reshape(n_windows, len(SIGNAL_NAMES) * len(STAT_NAMES))

    # Check for NaNs or Infs
    if np.isnan(features).any():
        raise ValueError("NaN values encountered during feature extraction")
    if np.isinf(features).any():
        raise ValueError("Infinite values encountered during feature extraction")

    return features


if __name__ == "__main__":
    names = get_feature_names()
    print(f"Total features: {len(names)} (expected 84)")
    print(f"First 12 feature names: {names[:12]}")
    print(f"Last 12 feature names:  {names[-12:]}")

    # Test on synthetic data
    x_test = np.random.randn(10, 128, 12)
    feat = extract_features(x_test)
    print(f"\nSynthetic test: input shape {x_test.shape} -> output shape {feat.shape}")
    assert feat.shape == (10, 84), f"Expected (10, 84), got {feat.shape}"
    print("[PASS] Feature extraction verified.")
