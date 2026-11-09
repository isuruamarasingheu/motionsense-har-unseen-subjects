"""
Unit tests for MotionSense data pipeline.

Tests:
1. Deduplication: subject 1 trial 13 dropped.
2. Subject split: train ∩ val ∩ test = empty, 16/4/4, gender-balanced.
3. Windows: no window spans two trials, shape (N, 128, 12).
4. Features: exactly 84 hand-crafted features per window, no NaNs.
5. Scalers: fitted on TRAIN only; scaler mean equals train feature mean.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import extract_features, get_feature_names
from src.data.load import load_motionsense
from src.data.preprocess import (
    FeatureScaler,
    RawWindowScaler,
    assert_scaler_fitted_on_train_only,
)
from src.data.split import assert_disjoint_splits, create_subject_split
from src.data.windows import assert_no_window_spans_trials, create_windows_from_df
from src.utils import load_json


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_motionsense(use_cache=True)
        cls.split_dict = create_subject_split(
            seed=42,
            output_path="outputs/splits/split.json",
        )

    def test_deduplication_removes_subject1_trial13(self) -> None:
        """Verify subject 1 trial 13 (duplicate of trial 5) is completely excluded."""
        dup = self.df[(self.df["subject"] == 1) & (self.df["trial"] == 13)]
        self.assertEqual(len(dup), 0, "Subject 1 trial 13 must be removed by deduplication.")
        # Ensure subject 1 trial 5 is preserved
        trial5 = self.df[(self.df["subject"] == 1) & (self.df["trial"] == 5)]
        self.assertGreater(len(trial5), 0, "Subject 1 trial 5 must be preserved.")

    def test_subject_split_disjoint_and_gender_balanced(self) -> None:
        """Verify subjects in train ∩ val ∩ test = empty and gender balance holds."""
        train_set = set(self.split_dict["train"])
        val_set = set(self.split_dict["val"])
        test_set = set(self.split_dict["test"])

        # Disjointness
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)
        self.assertEqual(len(train_set & val_set & test_set), 0)

        # Sizes
        self.assertEqual(len(train_set), 16)
        self.assertEqual(len(val_set), 4)
        self.assertEqual(len(test_set), 4)
        self.assertEqual(train_set | val_set | test_set, set(range(1, 25)))

        # Demographics
        demo = self.split_dict["demographics"]
        self.assertEqual(demo["val"]["male"], 2)
        self.assertEqual(demo["val"]["female"], 2)
        self.assertEqual(demo["test"]["male"], 2)
        self.assertEqual(demo["test"]["female"], 2)
        self.assertEqual(demo["train"]["male"], 10)
        self.assertEqual(demo["train"]["female"], 6)

    def test_windows_do_not_span_trials(self) -> None:
        """Verify all generated windows are strictly bounded within a single trial recording."""
        # Test on a subset of 2 subjects for speed
        df_sub = self.df[self.df["subject"].isin([2, 3])].copy()
        X_raw, y, meta = create_windows_from_df(df_sub, window_size=128, stride=64)

        self.assertGreater(len(X_raw), 0)
        self.assertEqual(X_raw.shape[1], 128)
        self.assertEqual(X_raw.shape[2], 12)
        self.assertEqual(len(y), len(X_raw))
        self.assertEqual(len(meta), len(X_raw))

        # Assertion function must pass without exception
        assert_no_window_spans_trials(meta, df_sub, window_size=128)

        # Directly verify start_index + 128 <= recording_length for every window
        rec_lens = df_sub.groupby(["subject", "trial"]).size().to_dict()
        for _, row in meta.iterrows():
            sub = int(row["subject"])
            tr = int(row["trial"])
            start = int(row["start_index"])
            total = rec_lens[(sub, tr)]
            self.assertLessEqual(start + 128, total)

    def test_feature_extraction_84_features(self) -> None:
        """Verify feature extraction outputs (N, 84) with no NaN or Inf values."""
        names = get_feature_names()
        self.assertEqual(len(names), 84)

        # Synthetic window test
        X_dummy = np.random.randn(5, 128, 12).astype(np.float32)
        feats = extract_features(X_dummy)
        self.assertEqual(feats.shape, (5, 84))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_scaler_mean_equals_train_mean_only(self) -> None:
        """Verify scaler mean equals mean of TRAIN features only (leakage test)."""
        rng = np.random.RandomState(42)
        X_tr = rng.randn(100, 84) + 3.0
        X_va = rng.randn(40, 84) - 2.0

        fs = FeatureScaler().fit(X_tr)
        self.assertTrue(np.allclose(fs.mean_, np.mean(X_tr, axis=0), atol=1e-7))
        self.assertFalse(np.allclose(fs.mean_, np.mean(X_va, axis=0), atol=1e-2))

        X_tr_raw = rng.randn(50, 128, 12).astype(np.float32) + 1.0
        X_va_raw = rng.randn(20, 128, 12).astype(np.float32) - 1.0

        rs = RawWindowScaler().fit(X_tr_raw)
        self.assertTrue(np.allclose(rs.mean, np.mean(X_tr_raw, axis=(0, 1)), atol=1e-6))
        self.assertFalse(np.allclose(rs.mean, np.mean(X_va_raw, axis=(0, 1)), atol=1e-2))

        # Check assertion helper
        assert_scaler_fitted_on_train_only(fs, X_tr, X_va, rs, X_tr_raw, X_va_raw)


if __name__ == "__main__":
    unittest.main()

