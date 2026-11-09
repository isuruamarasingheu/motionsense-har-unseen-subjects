"""
Module: src.data.preprocess
Purpose: Leakage-free feature scaling and sequence normalisation.

Responsibilities:
- Fit standardisation transformers (StandardScaler for 84 tabular features;
  per-channel mean/std for raw 12-channel windows) EXCLUSIVELY on training data.
- Never include validation or test data during fitting.
- Persist fitted transformers and statistics to `outputs/preprocessing/`
  using both joblib (.joblib / .pkl) and human-readable JSON formats.
- Provide transform and inverse_transform utilities for val, test, and inference inputs.
- Verify via assertions that the scaler's fitted mean equals the training feature mean only.

Rule compliance:
- Rule 5: Any scaler/normaliser is fitted on TRAIN data only, then saved and
  reused for val/test/inference.
- Rule 8: Serialises all preprocessing parameters for complete auditability.
- Rule 13: Docstrings explain why fitting on training data only is mandatory.
- Rule 14: Clear, robust, and easily explainable implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.features import get_feature_names
from src.data.load import SENSOR_COLUMNS
from src.utils import load_json, save_json

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    StandardScaler wrapper for 84 tabular window features, fitted strictly on training data.

    Why fit on train only:
        Fitting on the combined dataset (or including validation/test samples) causes
        data leakage: summary statistics (mean, variance) from unseen test subjects
        influence the scaling parameters, leading to over-optimistic performance estimates.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = get_feature_names()

    @property
    def mean_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet.")
        return self.scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet.")
        return self.scaler.scale_

    @property
    def var_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet.")
        return self.scaler.var_

    def fit(self, X_train: np.ndarray) -> FeatureScaler:
        """Fit StandardScaler on training features only."""
        arr = np.asarray(X_train, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (N, n_features), got shape {arr.shape}")
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {arr.shape[1]}"
            )

        self.scaler.fit(arr)
        self.is_fitted = True
        logger.info(
            "FeatureScaler fitted on %d training windows (%d features).",
            arr.shape[0],
            arr.shape[1],
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted standardisation to feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transforming data.")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.scaler.transform(arr)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training features and return transformed array."""
        return self.fit(X_train).transform(X_train)

    def save(
        self,
        joblib_path: str | Path = "outputs/preprocessing/feature_scaler.joblib",
        json_path: str | Path = "outputs/preprocessing/feature_scaler.json",
    ) -> tuple[Path, Path]:
        """Save fitted scaler to both joblib binary and JSON format."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted FeatureScaler.")

        job_file = Path(joblib_path)
        json_file = Path(json_path)
        job_file.parent.mkdir(parents=True, exist_ok=True)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save binary via joblib
        joblib.dump(self.scaler, job_file)

        # 2. Save human-readable JSON parameters
        params = {
            "scaler_type": "StandardScaler",
            "n_features": int(self.scaler.n_features_in_),
            "n_samples_seen": int(self.scaler.n_samples_seen_),
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
            "feature_names": self.feature_names,
        }
        save_json(params, json_file)
        logger.info("Saved FeatureScaler to %s and %s", job_file, json_file)
        return job_file, json_file

    @classmethod
    def load(
        cls,
        joblib_path: str | Path = "outputs/preprocessing/feature_scaler.joblib",
    ) -> FeatureScaler:
        """Load fitted scaler from joblib binary."""
        job_file = Path(joblib_path)
        if not job_file.is_file():
            raise FileNotFoundError(f"FeatureScaler file not found: {job_file}")

        instance = cls()
        instance.scaler = joblib.load(job_file)
        instance.is_fitted = True
        return instance


class RawWindowScaler:
    """
    Per-channel sequence normaliser for raw sensor windows of shape (N, 128, 12).

    Why per-channel normalisation:
        Raw sensor channels possess widely disparate units and dynamic ranges:
        - attitude angles (roll, pitch, yaw) range roughly -pi to +pi radians;
        - gravity components range roughly -1 to +1 g;
        - rotation rates range roughly -10 to +10 rad/s;
        - user accelerations range roughly -5 to +5 g.
        Standardising each channel across all training samples (mu=0, sigma=1)
        ensures gradient flow in 1D-CNNs is well-balanced across channels.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.channel_names = list(SENSOR_COLUMNS)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> RawWindowScaler:
        """
        Compute per-channel mean and standard deviation across all training windows and time steps.

        Args:
            X_train: NumPy array of shape (N, 128, 12).
        """
        arr = np.asarray(X_train, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (N, 128, 12), got shape {arr.shape}")
        if arr.shape[2] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got {arr.shape[2]}"
            )

        # Compute mean and std over (batch, time) dimensions: axis=(0, 1)
        self.mean = np.mean(arr, axis=(0, 1))  # (12,)
        self.std = np.std(arr, axis=(0, 1))    # (12,)
        self.is_fitted = True

        logger.info(
            "RawWindowScaler fitted on %d windows (%d channels).",
            arr.shape[0],
            arr.shape[2],
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply per-channel standardisation to 3D windows or single 2D window."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("RawWindowScaler must be fitted before transforming data.")

        arr = np.asarray(X, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        std = np.maximum(self.std.astype(np.float32), self.eps)

        if arr.ndim == 2:
            # Single window (128, 12)
            return (arr - mean) / std
        elif arr.ndim == 3:
            # Batch of windows (N, 128, 12)
            return (arr - mean[None, None, :]) / std[None, None, :]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training windows and return transformed array."""
        return self.fit(X_train).transform(X_train)

    def save(
        self,
        joblib_path: str | Path = "outputs/preprocessing/raw_scaler.joblib",
        json_path: str | Path = "outputs/preprocessing/raw_scaler.json",
    ) -> tuple[Path, Path]:
        """Save fitted normaliser parameters to joblib and JSON."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("Cannot save an unfitted RawWindowScaler.")

        job_file = Path(joblib_path)
        json_file = Path(json_path)
        job_file.parent.mkdir(parents=True, exist_ok=True)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
            "channel_names": self.channel_names,
        }
        joblib.dump(payload, job_file)

        json_payload = {
            "scaler_type": "PerChannelRawScaler",
            "n_channels": len(self.channel_names),
            "channel_names": self.channel_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": float(self.eps),
        }
        save_json(json_payload, json_file)
        logger.info("Saved RawWindowScaler to %s and %s", job_file, json_file)
        return job_file, json_file

    @classmethod
    def load(
        cls,
        joblib_path: str | Path = "outputs/preprocessing/raw_scaler.joblib",
    ) -> RawWindowScaler:
        """Load fitted normaliser from joblib binary."""
        job_file = Path(joblib_path)
        if not job_file.is_file():
            raise FileNotFoundError(f"RawWindowScaler file not found: {job_file}")

        payload = joblib.load(job_file)
        instance = cls(eps=payload.get("eps", 1e-8))
        instance.mean = payload["mean"]
        instance.std = payload["std"]
        instance.channel_names = payload.get("channel_names", list(SENSOR_COLUMNS))
        instance.is_fitted = True
        return instance


def assert_scaler_fitted_on_train_only(
    feature_scaler: FeatureScaler,
    X_train_feat: np.ndarray,
    X_val_feat: np.ndarray,
    raw_scaler: RawWindowScaler | None = None,
    X_train_raw: np.ndarray | None = None,
    X_val_raw: np.ndarray | None = None,
) -> None:
    """
    Assert that scalers were fitted exclusively on training data and not on validation data.

    Why:
        Enforces Rule 5. Tests that:
        1. Scaler mean exactly matches training feature mean.
        2. Scaler mean does not equal validation feature mean (ruling out test/val leakage).
        3. If raw scaler is provided, raw channel mean matches training raw window mean only.

    Args:
        feature_scaler: Fitted FeatureScaler.
        X_train_feat: Unscaled training feature array (N_train, 84).
        X_val_feat: Unscaled validation feature array (N_val, 84).
        raw_scaler: Optional fitted RawWindowScaler.
        X_train_raw: Optional unscaled training raw window array (N_train, 128, 12).
        X_val_raw: Optional unscaled validation raw window array (N_val, 128, 12).
    """
    # 1. Feature scaler assertions
    train_feat_mean = np.mean(X_train_feat, axis=0)
    assert np.allclose(
        feature_scaler.mean_, train_feat_mean, atol=1e-7
    ), "FeatureScaler mean does NOT match training feature mean!"

    val_feat_mean = np.mean(X_val_feat, axis=0)
    assert not np.allclose(
        feature_scaler.mean_, val_feat_mean, atol=1e-3
    ), "FeatureScaler mean erroneously matches validation feature mean! Leakage detected."

    # 2. Raw window scaler assertions
    if raw_scaler is not None and X_train_raw is not None and X_val_raw is not None:
        assert raw_scaler.mean is not None and raw_scaler.std is not None
        train_raw_mean = np.mean(X_train_raw, axis=(0, 1))
        assert np.allclose(
            raw_scaler.mean, train_raw_mean, atol=1e-7
        ), "RawWindowScaler mean does NOT match training raw window mean!"

        val_raw_mean = np.mean(X_val_raw, axis=(0, 1))
        assert not np.allclose(
            raw_scaler.mean, val_raw_mean, atol=1e-3
        ), "RawWindowScaler mean erroneously matches validation raw window mean! Leakage detected."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Quick self-test with synthetic arrays
    x_tr_feat = np.random.randn(100, 84) + 2.0
    x_va_feat = np.random.randn(30, 84) - 1.0

    fs = FeatureScaler().fit(x_tr_feat)
    fs.save()
    fs_loaded = FeatureScaler.load()

    x_tr_raw = np.random.randn(100, 128, 12) + 5.0
    x_va_raw = np.random.randn(30, 128, 12) - 3.0

    rs = RawWindowScaler().fit(x_tr_raw)
    rs.save()
    rs_loaded = RawWindowScaler.load()

    assert_scaler_fitted_on_train_only(fs_loaded, x_tr_feat, x_va_feat, rs_loaded, x_tr_raw, x_va_raw)
    print("\n[PASS] FeatureScaler and RawWindowScaler verified and assertions passed.")
