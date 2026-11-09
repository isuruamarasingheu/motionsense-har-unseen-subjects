"""
Module: src.data.windows
Purpose: Trial-bounded fixed-length sliding window segmenter.

Responsibilities:
- Segment continuous sensor streams into windows of length 128 samples (2.56 s at 50 Hz)
  with a stride of 64 samples (50% overlap).
- Constrain window generation strictly WITHIN each individual (subject, trial) recording;
  windows must never cross trial boundaries or subject boundaries.
- Generate raw 3D tensor arrays X_raw of shape (N, 128, 12), activity targets y of shape (N,),
  and per-window tracking metadata (subject, trial, start_index).
- Save window metadata records per split to `outputs/splits/windows_<split>.csv`.

Rule compliance:
- Rule 4: Windows must never cross trial boundaries.
- Rule 5: Splitting happens BEFORE windowing.
- Rule 13: Docstrings explain why window boundaries must not cross trials.
- Rule 14: Clean, explainable, and vectorised operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import SENSOR_COLUMNS

logger = logging.getLogger(__name__)


def create_windows_from_df(
    df: pd.DataFrame,
    window_size: int = 128,
    stride: int = 64,
    sensor_columns: list[str] = SENSOR_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate sliding windows strictly within each (subject, trial) recording.

    Why trial-bounded windowing:
        In continuous sensor data, different trials correspond to different physical activities,
        environments, or sessions. Creating a window that spans the end of one trial and the start
        of another would mix distinct physical movements and corrupt the activity label. Thus,
        each window is strictly contained inside a single (subject, trial) recording stream.

    Args:
        df: Input DataFrame containing sensor columns, 'subject', 'trial', 'activity', and 'label'.
        window_size: Window length in samples (default: 128 = 2.56 s at 50 Hz).
        stride: Stride between successive window starts (default: 64 = 50% overlap).
        sensor_columns: List of the 12 sensor channel names.

    Returns:
        Tuple of:
            - X_raw: NumPy array of shape (N, window_size, len(sensor_columns)), dtype float32.
            - y: NumPy array of shape (N,), integer class labels (0..5), dtype int64.
            - meta: pandas DataFrame containing per-window metadata:
                    ['subject', 'trial', 'start_index', 'activity', 'label'].
    """
    required_cols = set(sensor_columns + ["subject", "trial", "activity", "label"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    window_list: list[np.ndarray] = []
    label_list: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    # Group strictly by subject and trial recording
    # Sorting ensures deterministic ordering
    grouped = df.groupby(["subject", "trial"], sort=True)

    for (subject, trial), group in grouped:
        recording_length = len(group)
        if recording_length < window_size:
            logger.warning(
                "Recording for subject %s trial %s has length %d < window_size %d. Skipping.",
                subject,
                trial,
                recording_length,
                window_size,
            )
            continue

        # Extract continuous sensor array for this trial recording
        sensor_data = group[sensor_columns].to_numpy(dtype=np.float32)
        trial_activity = group["activity"].iloc[0]
        trial_label = int(group["label"].iloc[0])

        # Generate window start indices strictly within this trial
        start_indices = range(0, recording_length - window_size + 1, stride)
        for start_idx in start_indices:
            win = sensor_data[start_idx : start_idx + window_size]
            window_list.append(win)
            label_list.append(trial_label)
            meta_rows.append({
                "subject": int(subject),
                "trial": int(trial),
                "start_index": int(start_idx),
                "activity": str(trial_activity),
                "label": int(trial_label),
            })

    if not window_list:
        X_raw = np.empty((0, window_size, len(sensor_columns)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        meta = pd.DataFrame(columns=["subject", "trial", "start_index", "activity", "label"])
    else:
        X_raw = np.stack(window_list, axis=0)
        y = np.array(label_list, dtype=np.int64)
        meta = pd.DataFrame(meta_rows)

    logger.info(
        "Extracted %d windows of shape (%d, %d). Unique classes: %s",
        len(X_raw),
        window_size,
        len(sensor_columns),
        np.unique(y).tolist() if len(y) > 0 else [],
    )

    return X_raw, y, meta


def save_window_meta(
    meta: pd.DataFrame,
    split_name: str,
    output_dir: str | Path = "outputs/splits",
) -> Path:
    """
    Save window tracking metadata to `outputs/splits/windows_<split>.csv`.

    Args:
        meta: DataFrame containing window metadata records.
        split_name: Name of the split ('train', 'val', or 'test').
        output_dir: Destination folder path.

    Returns:
        Path to the saved CSV file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"windows_{split_name}.csv"
    meta.to_csv(out_path, index=False)
    logger.info("Saved window metadata for '%s' split to %s (%d records)", split_name, out_path, len(meta))
    return out_path


def assert_no_window_spans_trials(
    meta: pd.DataFrame,
    df_source: pd.DataFrame,
    window_size: int = 128,
) -> None:
    """
    Verify and assert that no window spans two trials or subject boundaries.

    Why:
        Enforces Rule 4 and Rule 5. A window spanning multiple trials causes
        label ambiguity and temporal discontinuity corruption.

    Args:
        meta: Metadata DataFrame containing 'subject', 'trial', 'start_index'.
        df_source: Source DataFrame with continuous trial recordings.
        window_size: Window length in samples (default: 128).
    """
    if len(meta) == 0:
        return

    # Map each (subject, trial) recording to its continuous length
    recording_lengths: dict[tuple[int, int], int] = (
        df_source.groupby(["subject", "trial"]).size().to_dict()
    )

    for idx, row in meta.iterrows():
        sub = int(row["subject"])
        trial = int(row["trial"])
        start = int(row["start_index"])

        rec_key = (sub, trial)
        assert rec_key in recording_lengths, (
            f"Window {idx} references unknown recording (subject {sub}, trial {trial})"
        )

        total_len = recording_lengths[rec_key]
        assert start >= 0, f"Window {idx} has negative start index {start}"
        assert start + window_size <= total_len, (
            f"Window {idx} spans beyond recording boundary! "
            f"Subject {sub}, trial {trial}, start {start} + {window_size} > {total_len}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.data.load import load_motionsense

    df = load_motionsense()
    X_raw, y, meta = create_windows_from_df(df, window_size=128, stride=64)
    assert_no_window_spans_trials(meta, df, window_size=128)
    print("\n--- Window Extraction Summary ---")
    print(f"X_raw shape:  {X_raw.shape}")
    print(f"y shape:      {y.shape}")
    print(f"meta rows:    {len(meta)}")
    print(f"Class distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    print("[PASS] All windows strictly contained inside individual trials.")
