"""
Module: src.data.load
Purpose: Raw MotionSense dataset loader, demographic merger, duplicate recording detector,
and pickle cache manager.

Dataset reference:
    Malekzadeh et al., "Mobile Sensor Data Anonymization", IoTDI 2019 /
    "Protecting Sensory Data against Sensitive Inference Attacks", WSDM 2018.
    https://github.com/mmalekzadeh/motion-sense

Responsibilities:
- Walk `MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data/` recursively.
- Skip any `__MACOSX` metadata folders left by macOS archive extraction.
- Extract activity code and trial number from folder name `<act>_<trial>`.
- Extract subject ID from CSV file name `sub_<n>.csv`.
- Drop unnamed index column present in raw CSV files.
- Map activity codes to integer labels 0..5 according to the trial-to-label map:
    0: dws (downstairs) - trials 1, 2, 11
    1: ups (upstairs)   - trials 3, 4, 12
    2: wlk (walking)    - trials 7, 8, 15
    3: jog (jogging)    - trials 9, 16
    4: std (standing)   - trials 6, 14
    5: sit (sitting)    - trials 5, 13
- Detect byte-identical (subject, trial) recordings by comparing sensor arrays (specifically,
  Subject 1's sit_5 and sit_13 recordings are byte-identical in the published dataset).
  Keep the lower trial number, drop the higher trial number, and log the action.
- Merge participant demographic info (weight, height, age, gender) from
  `MotionSense_dataset/data_subjects_info.csv` (code -> subject).
- Cache the merged DataFrame to `outputs/cache/motionsense.pickle` using pandas `to_pickle`
  (retaining fast binary serialization without requiring pyarrow).

Rule compliance:
- Rule 2: Raw data in MotionSense_dataset/ is strictly READ-ONLY.
- Rule 3: Ignore __MACOSX folders and drop the unnamed index column.
- Rule 10: Retain truthful logging of data curation decisions; no hidden filtering.
- Rule 13: Detailed docstrings and provenance tracking.
- Rule 14: Clear, explainable Python code.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Primary 12 motion sensor channels recorded by iPhone 6s at 50 Hz
SENSOR_COLUMNS: list[str] = [
    "attitude.roll",
    "attitude.pitch",
    "attitude.yaw",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
]

# Activity code to integer label map (CW1_PLAN.md Section 1)
LABEL_MAP: dict[str, int] = {
    "dws": 0,  # downstairs (trials 1, 2, 11)
    "ups": 1,  # upstairs   (trials 3, 4, 12)
    "wlk": 2,  # walking    (trials 7, 8, 15)
    "jog": 3,  # jogging    (trials 9, 16)
    "std": 4,  # standing   (trials 6, 14)
    "sit": 5,  # sitting    (trials 5, 13)
}

# Human-readable activity names
ACTIVITY_NAMES: dict[str, str] = {
    "dws": "downstairs",
    "ups": "upstairs",
    "wlk": "walking",
    "jog": "jogging",
    "std": "standing",
    "sit": "sitting",
}


def load_motionsense(
    data_dir: str | Path = "MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data",
    subjects_info_path: str | Path = "MotionSense_dataset/data_subjects_info.csv",
    cache_path: str | Path = "outputs/cache/motionsense.pickle",
    use_cache: bool = True,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load all MotionSense sensor trials, deduplicate identical recordings, merge demographics, and cache.

    Why:
        Consolidates 360 raw trial recordings into a unified DataFrame.
        Importantly, detects dataset artifacts (such as Subject 1's sit_5 and sit_13
        being duplicate copies in the original publication) and eliminates them to prevent
        double-weighting and evaluation score inflation.

    Args:
        data_dir: Directory containing activity trial subfolders (e.g. `dws_1/`).
        subjects_info_path: CSV file path containing participant demographics.
        cache_path: File path to save/load the pickle cache.
        use_cache: If True, loads existing pickle cache if present.
        force_reload: If True, re-parses raw CSVs even if cache file exists.

    Returns:
        Consolidated pandas DataFrame containing sensor channels, trial metadata,
        activity labels (0..5), and participant demographics with duplicates removed.
    """
    cache_file = Path(cache_path)

    # 1. Check existing pickle cache
    if use_cache and not force_reload and cache_file.is_file():
        logger.info("Loading cached MotionSense dataset from %s", cache_file)
        df_cached = pd.read_pickle(cache_file)
        return df_cached

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"MotionSense data directory not found: {data_path.resolve()}")

    sub_info_file = Path(subjects_info_path)
    if not sub_info_file.is_file():
        raise FileNotFoundError(f"Subjects demographic file not found: {sub_info_file.resolve()}")

    logger.info("Loading participant demographic metadata from %s", sub_info_file)
    df_subjects = pd.read_csv(sub_info_file).rename(columns={"code": "subject"})

    trial_folder_pattern = re.compile(r"^([a-z]+)_(\d+)$")
    sub_file_pattern = re.compile(r"^sub_(\d+)\.csv$")

    # Temporary storage for parsed individual recording files
    raw_recordings: list[dict[str, Any]] = []

    # 2. Iterate across activity folders
    folder_names = sorted(os.listdir(data_path))
    for folder_name in folder_names:
        if "__MACOSX" in folder_name:
            continue

        folder_full_path = data_path / folder_name
        if not folder_full_path.is_dir():
            continue

        match_folder = trial_folder_pattern.match(folder_name)
        if not match_folder:
            continue

        act_code = match_folder.group(1)
        trial_id = int(match_folder.group(2))

        if act_code not in LABEL_MAP:
            logger.warning("Unrecognised activity code '%s' in folder '%s'. Skipping.", act_code, folder_name)
            continue

        label = LABEL_MAP[act_code]

        # 3. Read participant CSV files inside current trial folder
        for file_name in sorted(os.listdir(folder_full_path)):
            if "__MACOSX" in file_name or not file_name.endswith(".csv"):
                continue

            match_file = sub_file_pattern.match(file_name)
            if not match_file:
                continue

            sub_id = int(match_file.group(1))
            file_full_path = folder_full_path / file_name

            # Read raw trial recording
            trial_df = pd.read_csv(file_full_path)

            # Drop unnamed index column if present (Rule 3)
            if trial_df.columns[0].startswith("Unnamed") or trial_df.columns[0] == "":
                trial_df = trial_df.iloc[:, 1:]

            missing_cols = set(SENSOR_COLUMNS) - set(trial_df.columns)
            if missing_cols:
                raise ValueError(
                    f"File {file_full_path} is missing expected sensor columns: {missing_cols}"
                )

            trial_df = trial_df[SENSOR_COLUMNS].copy()
            trial_df["subject"] = sub_id
            trial_df["activity"] = act_code
            trial_df["label"] = label
            trial_df["trial"] = trial_id

            raw_recordings.append({
                "subject": sub_id,
                "activity": act_code,
                "trial": trial_id,
                "label": label,
                "df": trial_df,
                "file_path": str(file_full_path),
            })

    if not raw_recordings:
        raise RuntimeError(f"No trial CSV files successfully parsed from {data_path}")

    logger.info("Parsed %d total trial recordings. Checking for duplicate recordings...", len(raw_recordings))

    # 4. Detect and remove identical (subject, activity) recordings
    # Group recordings by subject and sort by trial number ascending
    subject_map: dict[int, list[dict[str, Any]]] = {}
    for rec in raw_recordings:
        subject_map.setdefault(rec["subject"], []).append(rec)

    deduped_dfs: list[pd.DataFrame] = []
    dropped_count = 0

    for sub_id in sorted(subject_map.keys()):
        # Sort trials ascending so lower trial number is encountered first
        recs = sorted(subject_map[sub_id], key=lambda r: r["trial"])
        kept_recs: list[dict[str, Any]] = []

        for rec in recs:
            rec_sensors = rec["df"][SENSOR_COLUMNS].values
            is_dup = False

            for kept in kept_recs:
                if rec["activity"] == kept["activity"]:
                    kept_sensors = kept["df"][SENSOR_COLUMNS].values
                    if rec_sensors.shape == kept_sensors.shape and np.array_equal(rec_sensors, kept_sensors):
                        is_dup = True
                        dropped_count += 1
                        logger.warning(
                            "Duplicate recording detected: Subject %d activity '%s' trial %d is byte-identical "
                            "to trial %d. Dropping trial %d (%d rows) and keeping lower trial %d.",
                            sub_id,
                            rec["activity"],
                            rec["trial"],
                            kept["trial"],
                            rec["trial"],
                            len(rec["df"]),
                            kept["trial"],
                        )
                        break

            if not is_dup:
                kept_recs.append(rec)

        for kept in kept_recs:
            deduped_dfs.append(kept["df"])

    logger.info(
        "Deduplication complete: %d recordings kept, %d duplicate recordings dropped.",
        len(deduped_dfs),
        dropped_count,
    )

    # 5. Concatenate all clean recordings
    full_df = pd.concat(deduped_dfs, ignore_index=True)

    # 6. Merge participant demographics on 'subject'
    full_df = full_df.merge(df_subjects, on="subject", how="left")

    # 7. Ensure cache directory exists and save pickle
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving deduplicated dataset to pickle cache: %s (%d rows)", cache_file, len(full_df))
    full_df.to_pickle(cache_file)

    return full_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = load_motionsense(force_reload=True)
    print("\n--- MotionSense Dataset Load Summary (Deduplicated) ---")
    print(f"Total rows:        {len(df):,}")
    print(f"Total columns:     {len(df.columns)}")
    print(f"Unique subjects:   {df['subject'].nunique()} (expected 24)")
    print(f"Unique trials:     {df['trial'].nunique()} (expected 15)")
    print(f"Unique activities: {df['activity'].nunique()} (expected 6)")
    print("\nRows per activity:")
    print(df["activity"].value_counts())
    print("\nMemory usage:")
    df.info(memory_usage="deep")
