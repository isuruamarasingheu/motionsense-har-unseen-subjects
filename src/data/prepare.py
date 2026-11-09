"""
Module: src.data.prepare
Purpose: End-to-end data preparation pipeline execution.

Responsibilities:
- Orchestrate loading, subject-wise splitting, trial-bounded windowing,
  hand-crafted feature extraction, and train-only scaling.
- Ensure strict adherence to the project rules:
    * Rule 2: Raw data is read-only.
    * Rule 3: macOS artifacts dropped; unnamed index columns removed.
    * Rule 4: Data splitting is by subject only; no subject overlap.
    * Rule 5: Splitting happens BEFORE windowing; scalers fitted on TRAIN only.
    * Rule 9: Random seed loaded from configs/data.yaml.
    * Rule 10: Retain truthful logging and deduplicated dataset records.
- Persist all processed artifacts:
    * `outputs/splits/split.json`
    * `outputs/splits/windows_<split>.csv`
    * `outputs/preprocessing/` (joblib and JSON for feature and raw scalers)
    * `outputs/cache/` (NumPy arrays: raw, raw_scaled, feat, feat_scaled, y)
- Run rigorous assertions:
    * subjects in train ∩ val ∩ test = empty
    * scaler mean equals mean of TRAIN features only
    * no window spans two trials
- Output formatted summary tables of window counts per split and per class.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.features import extract_features, get_feature_names
from src.data.load import LABEL_MAP, load_motionsense
from src.data.preprocess import (
    FeatureScaler,
    RawWindowScaler,
    assert_scaler_fitted_on_train_only,
)
from src.data.split import assert_disjoint_splits, create_subject_split
from src.data.windows import (
    assert_no_window_spans_trials,
    create_windows_from_df,
    save_window_meta,
)
from src.utils import load_config, save_json, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.data.prepare")


def run_pipeline(
    config_path: str | Path = "configs/data.yaml",
) -> dict[str, Any]:
    """
    Execute the complete MotionSense preparation pipeline.

    Args:
        config_path: Path to configs/data.yaml.

    Returns:
        Dictionary containing counts, shapes, and pipeline metadata.
    """
    logger.info("=== Starting Data Preparation Pipeline ===")
    cfg = load_config(config_path)

    # 1. Reproducibility setup
    split_cfg = cfg.get("split", {})
    seed = split_cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Random seed initialized to %d", seed)

    # 2. Load deduplicated dataset
    dataset_cfg = cfg.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data")
    subjects_info = dataset_cfg.get("subjects_info", "MotionSense_dataset/data_subjects_info.csv")
    cache_dir = Path(dataset_cfg.get("cache_dir", "outputs/cache"))
    splits_dir = Path(dataset_cfg.get("splits_dir", "outputs/splits"))
    prep_dir = Path(dataset_cfg.get("preprocessing_dir", "outputs/preprocessing"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MotionSense dataset (with deduplication)...")
    df = load_motionsense(
        data_dir=raw_dir,
        subjects_info_path=subjects_info,
        use_cache=True,
    )

    # Verify duplicate subject 1 trial 13 is removed
    dup_check = df[(df["subject"] == 1) & (df["trial"] == 13)]
    assert len(dup_check) == 0, "Duplicate recording (subject 1 trial 13) was not removed!"
    logger.info("Confirmed: Duplicate recording (subject 1 trial 13) is dropped.")

    # 3. Subject-wise, gender-balanced split (Rule 4 & 5)
    split_json_path = splits_dir / "split.json"
    logger.info("Creating subject-wise split...")
    split_dict = create_subject_split(
        subjects_info_path=subjects_info,
        seed=seed,
        val_male=split_cfg.get("val_composition", {}).get("male", 2),
        val_female=split_cfg.get("val_composition", {}).get("female", 2),
        test_male=split_cfg.get("test_composition", {}).get("male", 2),
        test_female=split_cfg.get("test_composition", {}).get("female", 2),
        output_path=split_json_path,
    )

    # Assertion 1: Subjects in train ∩ val ∩ test = empty
    assert_disjoint_splits(split_dict, all_subjects=sorted(df["subject"].unique().tolist()))
    logger.info("[ASSERTION 1 PASS] subjects in train ∩ val ∩ test = empty (disjoint sets).")

    # 4. Partition DataFrame by subject BEFORE windowing (Rule 5)
    train_subjects = split_dict["train"]
    val_subjects = split_dict["val"]
    test_subjects = split_dict["test"]

    df_train = df[df["subject"].isin(train_subjects)].copy()
    df_val = df[df["subject"].isin(val_subjects)].copy()
    df_test = df[df["subject"].isin(test_subjects)].copy()

    logger.info(
        "Data split by subject: Train (%d rows, %d subs), Val (%d rows, %d subs), Test (%d rows, %d subs)",
        len(df_train),
        len(train_subjects),
        len(df_val),
        len(val_subjects),
        len(df_test),
        len(test_subjects),
    )

    # 5. Extract windows inside each (subject, trial) recording (Rule 4)
    win_cfg = cfg.get("windowing", {})
    window_size = win_cfg.get("window_size", 128)
    stride = win_cfg.get("stride", 64)

    logger.info("Generating windows (length=%d, stride=%d)...", window_size, stride)
    X_train_raw, y_train, meta_train = create_windows_from_df(df_train, window_size, stride)
    X_val_raw, y_val, meta_val = create_windows_from_df(df_val, window_size, stride)
    X_test_raw, y_test, meta_test = create_windows_from_df(df_test, window_size, stride)

    # Save window metadata per split
    save_window_meta(meta_train, "train", splits_dir)
    save_window_meta(meta_val, "val", splits_dir)
    save_window_meta(meta_test, "test", splits_dir)

    # Assertion 2: No window spans two trials
    assert_no_window_spans_trials(meta_train, df_train, window_size)
    assert_no_window_spans_trials(meta_val, df_val, window_size)
    assert_no_window_spans_trials(meta_test, df_test, window_size)
    logger.info("[ASSERTION 2 PASS] no window spans two trials or subject boundaries.")

    # 6. Extract 84 hand-crafted features per window
    logger.info("Extracting 84 hand-crafted statistical features...")
    X_train_feat = extract_features(X_train_raw)
    X_val_feat = extract_features(X_val_raw)
    X_test_feat = extract_features(X_test_raw)

    feature_names = get_feature_names()
    assert X_train_feat.shape[1] == 84, f"Expected 84 features, got {X_train_feat.shape[1]}"
    assert len(feature_names) == 84, f"Expected 84 feature names, got {len(feature_names)}"

    # 7. Fit preprocessing scalers FITTED ON TRAIN ONLY (Rule 5)
    logger.info("Fitting FeatureScaler and RawWindowScaler on TRAIN split only...")
    feat_scaler = FeatureScaler().fit(X_train_feat)
    raw_scaler = RawWindowScaler().fit(X_train_raw)

    # Save scalers to outputs/preprocessing/
    feat_scaler.save(
        joblib_path=prep_dir / "feature_scaler.joblib",
        json_path=prep_dir / "feature_scaler.json",
    )
    raw_scaler.save(
        joblib_path=prep_dir / "raw_scaler.joblib",
        json_path=prep_dir / "raw_scaler.json",
    )

    # Assertion 3: Scaler mean equals mean of TRAIN features only
    assert_scaler_fitted_on_train_only(
        feat_scaler,
        X_train_feat,
        X_val_feat,
        raw_scaler,
        X_train_raw,
        X_val_raw,
    )
    logger.info("[ASSERTION 3 PASS] scaler mean equals mean of TRAIN features only.")

    # Apply transformations
    logger.info("Standardising features and raw windows using fitted scalers...")
    X_train_feat_scaled = feat_scaler.transform(X_train_feat)
    X_val_feat_scaled = feat_scaler.transform(X_val_feat)
    X_test_feat_scaled = feat_scaler.transform(X_test_feat)

    X_train_raw_scaled = raw_scaler.transform(X_train_raw)
    X_val_raw_scaled = raw_scaler.transform(X_val_raw)
    X_test_raw_scaled = raw_scaler.transform(X_test_raw)

    # 8. Save arrays to outputs/cache/
    logger.info("Saving processed NumPy arrays to %s...", cache_dir)
    arrays_to_save: dict[str, np.ndarray] = {
        "X_train_raw.npy": X_train_raw,
        "X_val_raw.npy": X_val_raw,
        "X_test_raw.npy": X_test_raw,
        "X_train_raw_scaled.npy": X_train_raw_scaled,
        "X_val_raw_scaled.npy": X_val_raw_scaled,
        "X_test_raw_scaled.npy": X_test_raw_scaled,
        "X_train_feat.npy": X_train_feat,
        "X_val_feat.npy": X_val_feat,
        "X_test_feat.npy": X_test_feat,
        "X_train_feat_scaled.npy": X_train_feat_scaled,
        "X_val_feat_scaled.npy": X_val_feat_scaled,
        "X_test_feat_scaled.npy": X_test_feat_scaled,
        "y_train.npy": y_train,
        "y_val.npy": y_val,
        "y_test.npy": y_test,
    }

    for fname, arr in arrays_to_save.items():
        np.save(cache_dir / fname, arr)

    save_json({"feature_names": feature_names}, cache_dir / "feature_names.json")

    # 9. Format and print window counts per split and per class
    print_summary_tables(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        split_dict=split_dict,
        labels_map=cfg.get("labels", LABEL_MAP),
    )

    summary_info = {
        "train_windows": len(y_train),
        "val_windows": len(y_val),
        "test_windows": len(y_test),
        "total_windows": len(y_train) + len(y_val) + len(y_test),
        "raw_shape": list(X_train_raw.shape),
        "feat_shape": list(X_train_feat.shape),
    }

    logger.info("=== Data Preparation Pipeline Completed Successfully ===")
    return summary_info


def print_summary_tables(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    split_dict: dict[str, Any],
    labels_map: dict[int | str, str],
) -> None:
    """Print readable, formatted tables of window counts per split and per class."""
    total_train = len(y_train)
    total_val = len(y_val)
    total_test = len(y_test)
    total_all = total_train + total_val + total_test

    # 1. Split Demographics Table
    print("\n" + "=" * 76)
    print("                      DATASET SPLIT & DEMOGRAPHICS")
    print("=" * 76)
    print(f"{'Split':<8} | {'Subjects':<10} | {'Males':<6} | {'Females':<8} | {'Windows':<10} | {'Percentage':<10}")
    print("-" * 76)

    demo = split_dict.get("demographics", {})
    for split_name, total_cnt in [("train", total_train), ("val", total_val), ("test", total_test)]:
        d = demo.get(split_name, {})
        n_subs = len(split_dict.get(split_name, []))
        pct = (total_cnt / total_all) * 100 if total_all > 0 else 0
        print(
            f"{split_name:<8} | {n_subs:<10} | {d.get('male', '-'):<6} | {d.get('female', '-'):<8} | "
            f"{total_cnt:<10,d} | {pct:>9.2f}%"
        )
    print("-" * 76)
    print(
        f"{'Total':<8} | {24:<10} | {14:<6} | {10:<8} | {total_all:<10,d} | {100.00:>9.2f}%"
    )
    print("=" * 76)

    # 2. Window Counts per Class per Split
    # Invert labels_map if keys are int
    code_names: dict[int, str] = {
        0: "downstairs (dws)",
        1: "upstairs (ups)",
        2: "walking (wlk)",
        3: "jogging (jog)",
        4: "standing (std)",
        5: "sitting (sit)",
    }

    print("\n" + "=" * 80)
    print("                 WINDOW COUNTS PER CLASS AND PER SPLIT")
    print("=" * 80)
    print(
        f"{'Label':<5} | {'Activity Class':<18} | {'Train':<9} | {'Val':<8} | {'Test':<8} | {'Total':<9} | {'Class %':<8}"
    )
    print("-" * 80)

    for lbl in range(6):
        c_tr = int(np.sum(y_train == lbl))
        c_va = int(np.sum(y_val == lbl))
        c_te = int(np.sum(y_test == lbl))
        c_tot = c_tr + c_va + c_te
        pct = (c_tot / total_all) * 100 if total_all > 0 else 0
        name = code_names.get(lbl, str(lbl))
        print(
            f"{lbl:<5} | {name:<18} | {c_tr:<9,d} | {c_va:<8,d} | {c_te:<8,d} | {c_tot:<9,d} | {pct:>7.2f}%"
        )

    print("-" * 80)
    print(
        f"{'Total':<5} | {'All Activities':<18} | {total_train:<9,d} | {total_val:<8,d} | {total_test:<8,d} | {total_all:<9,d} | {100.00:>7.2f}%"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline()

