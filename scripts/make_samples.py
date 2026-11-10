"""
Script: scripts/make_samples.py
Purpose: T5b Generate representative and stress-test sample CSV recordings
from the local MotionSense dataset for inference CLI demonstrations and tests.

Coursework requirements:
- Raw dataset is not redistributed; sample recordings are extracted dynamically
  from the local MotionSense dataset (MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data/).
- Samples created in samples/:
  1. valid_walking_subject8.csv:
     10 seconds (500 rows) from held-out TEST subject 8, walking trial 7, rows 500-999.
     Expected prediction: WALKING (high confidence).
  2. failure_walking_subject1.csv:
     10 seconds (500 rows) from held-out TEST subject 1, walking trial 7, rows 500-999.
     Demonstrates the known physical phone-orientation failure discovered in Phase 4 error analysis.
  3. invalid_short.csv:
     First 60 rows (<128 rows requirement) of the valid sample.
     Must be rejected with a clean non-zero error code.
  4. invalid_columns.csv:
     Valid sample with userAcceleration.z removed.
     Must be rejected with a clean non-zero error code.
  5. invalid_nan.csv:
     Valid sample injected with 3 NaN entries.
     Must be rejected with a clean non-zero error code listing row numbers.
  6. out_of_scope_scaled.csv:
     Valid sample with userAcceleration multiplied by 20x.
     Must trigger the out-of-scope warning (>10% outside training bounds) while still predicting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import SENSOR_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scripts.make_samples")

DATA_DIR = Path("MotionSense_dataset/A_DeviceMotion_data/A_DeviceMotion_data")
SAMPLES_DIR = Path("samples")


def make_samples() -> None:
    """Extract sample recordings and write them to samples/ directory."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Valid sample: Subject 8, wlk_7, rows 500-999 (500 samples = 10.0 s @ 50 Hz)
    sub8_path = DATA_DIR / "wlk_7" / "sub_8.csv"
    if not sub8_path.is_file():
        raise FileNotFoundError(f"Raw recording not found: {sub8_path}")

    logger.info("Loading Subject 8 wlk_7 from %s ...", sub8_path)
    df_sub8_raw = pd.read_csv(sub8_path)
    drop_cols = [c for c in df_sub8_raw.columns if str(c).startswith("Unnamed:")]
    df_sub8 = df_sub8_raw.drop(columns=drop_cols)

    # Extract 500 rows starting at row 500
    df_valid = df_sub8.iloc[500:1000][SENSOR_COLUMNS].reset_index(drop=True)
    valid_csv_path = SAMPLES_DIR / "valid_walking_subject8.csv"

    header_valid = (
        "# Source: MotionSense A_DeviceMotion_data/wlk_7/sub_8.csv\n"
        "# Description: Held-out Test Subject 8, Trial wlk_7, rows 500-999 (500 samples = 10 s @ 50 Hz)\n"
        "# Expected: Valid input, majority predicted activity = WALKING with high confidence\n"
    )
    with open(valid_csv_path, "w", encoding="utf-8") as f:
        f.write(header_valid)
    df_valid.to_csv(valid_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", valid_csv_path, df_valid.shape)

    # 2. Failure demo sample: Subject 1, wlk_7, rows 500-999
    # In Phase 4 error analysis, Subject 1 had a tilted phone orientation during wlk_7
    sub1_path = DATA_DIR / "wlk_7" / "sub_1.csv"
    if not sub1_path.is_file():
        raise FileNotFoundError(f"Raw recording not found: {sub1_path}")

    logger.info("Loading Subject 1 wlk_7 from %s ...", sub1_path)
    df_sub1_raw = pd.read_csv(sub1_path)
    drop_cols = [c for c in df_sub1_raw.columns if str(c).startswith("Unnamed:")]
    df_sub1 = df_sub1_raw.drop(columns=drop_cols)

    # Compute Subject 1 median pitch in wlk_7 vs other walking trials (wlk_8, wlk_15) dynamically (do not hard-code)
    sub1_pitch_wlk7 = float(df_sub1_raw["attitude.pitch"].median())
    sub1_wlk8_path = DATA_DIR / "wlk_8" / "sub_1.csv"
    sub1_wlk15_path = DATA_DIR / "wlk_15" / "sub_1.csv"
    other_wlk_pitches = pd.concat([
        pd.read_csv(sub1_wlk8_path)["attitude.pitch"],
        pd.read_csv(sub1_wlk15_path)["attitude.pitch"]
    ])
    sub1_pitch_other = float(other_wlk_pitches.median())
    logger.info(
        "Computed Subject 1 median pitch: wlk_7 = %.2f rad vs other walking trials (wlk_8, wlk_15) = %.2f rad",
        sub1_pitch_wlk7,
        sub1_pitch_other,
    )
    print(
        f"Subject 1 median pitch: wlk_7 = {sub1_pitch_wlk7:.2f} rad vs "
        f"other walking trials (wlk_8, wlk_15) = {sub1_pitch_other:.2f} rad"
    )

    df_failure = df_sub1.iloc[500:1000][SENSOR_COLUMNS].reset_index(drop=True)
    failure_csv_path = SAMPLES_DIR / "failure_walking_subject1.csv"

    header_failure = (
        "# Source: MotionSense A_DeviceMotion_data/wlk_7/sub_1.csv\n"
        "# Description: Held-out Test Subject 1, Trial wlk_7, rows 500-999 (500 samples = 10 s @ 50 Hz)\n"
        f"# Note: Demonstrates tilted phone orientation (median pitch {sub1_pitch_wlk7:.2f} rad vs {sub1_pitch_other:.2f} rad in the same subject's other walking trials)\n"
    )
    with open(failure_csv_path, "w", encoding="utf-8") as f:
        f.write(header_failure)
    df_failure.to_csv(failure_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", failure_csv_path, df_failure.shape)

    # 3. Invalid short sample: first 60 rows of the valid sample (<128 rows required)
    df_short = df_valid.iloc[:60].copy()
    short_csv_path = SAMPLES_DIR / "invalid_short.csv"

    header_short = (
        "# Source: Derived from valid_walking_subject8.csv\n"
        "# Description: Only 60 rows (< 128 rows / 2.56 s minimum requirement at 50 Hz)\n"
        "# Expected: Non-zero exit code (1), error message stating fewer than 128 rows\n"
    )
    with open(short_csv_path, "w", encoding="utf-8") as f:
        f.write(header_short)
    df_short.to_csv(short_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", short_csv_path, df_short.shape)

    # 4. Invalid columns sample: valid sample with userAcceleration.z column dropped
    df_missing_col = df_valid.drop(columns=["userAcceleration.z"]).copy()
    missing_col_csv_path = SAMPLES_DIR / "invalid_columns.csv"

    header_missing_col = (
        "# Source: Derived from valid_walking_subject8.csv\n"
        "# Description: Missing required sensor column 'userAcceleration.z'\n"
        "# Expected: Non-zero exit code (1), error message listing missing column\n"
    )
    with open(missing_col_csv_path, "w", encoding="utf-8") as f:
        f.write(header_missing_col)
    df_missing_col.to_csv(missing_col_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", missing_col_csv_path, df_missing_col.shape)

    # 5. Invalid NaN sample: valid sample with 3 NaN values injected
    df_nan = df_valid.copy()
    df_nan.loc[12, "attitude.roll"] = np.nan
    df_nan.loc[25, "gravity.y"] = np.nan
    df_nan.loc[50, "userAcceleration.x"] = np.nan
    nan_csv_path = SAMPLES_DIR / "invalid_nan.csv"

    header_nan = (
        "# Source: Derived from valid_walking_subject8.csv\n"
        "# Description: Corrupted with 3 NaN entries at row indices 12, 25, 50\n"
        "# Expected: Non-zero exit code (1), error message reporting corrupted row numbers\n"
    )
    with open(nan_csv_path, "w", encoding="utf-8") as f:
        f.write(header_nan)
    df_nan.to_csv(nan_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", nan_csv_path, df_nan.shape)

    # 6. Out-of-scope sample: valid sample with userAcceleration values scaled by 20x
    df_out_of_scope = df_valid.copy()
    acc_cols = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
    df_out_of_scope[acc_cols] = df_out_of_scope[acc_cols] * 20.0
    out_of_scope_csv_path = SAMPLES_DIR / "out_of_scope_scaled.csv"

    header_out_of_scope = (
        "# Source: Derived from valid_walking_subject8.csv\n"
        "# Description: userAcceleration channels magnified by 20x, simulating extreme domain shift\n"
        "# Expected: Out-of-scope warning printed (>10% outside training bounds), prediction proceeds\n"
    )
    with open(out_of_scope_csv_path, "w", encoding="utf-8") as f:
        f.write(header_out_of_scope)
    df_out_of_scope.to_csv(out_of_scope_csv_path, mode="a", index=False)
    logger.info("Saved %s (shape: %s)", out_of_scope_csv_path, df_out_of_scope.shape)

    # 7. Write samples/README.md documentation
    readme_path = SAMPLES_DIR / "README.md"
    readme_content = f"""# MotionSense Sample Recordings (`samples/`)

This directory contains demonstration and testing sensor recordings for the `src.predict` CLI inference engine (Task T5b). Raw data is not committed; all samples are created dynamically from the local dataset via `scripts/make_samples.py`.

To rebuild samples from the local dataset at any time, run:
```bash
python -m scripts.make_samples
```

---

## Sample Inventory

| File Name | Duration / Rows | Provenance | Purpose & Expected Behavior |
| :--- | :--- | :--- | :--- |
| `valid_walking_subject8.csv` | 10.0 s (500 rows) | Test Subject 8, `wlk_7`, rows 500–999 | **Normal operational input**: Valid 12 sensor channels at 50 Hz. Exits with code `0`. Predicted activity: `WALKING` with high confidence (~0.90+). |
| `failure_walking_subject1.csv` | 10.0 s (500 rows) | Test Subject 1, `wlk_7`, rows 500–999 | **Physical phone-orientation failure demo**: tilted phone orientation (median pitch {sub1_pitch_wlk7:.2f} rad vs {sub1_pitch_other:.2f} rad in the same subject's other walking trials). Demonstrates how physical placement mismatch alters predictions. |
| `invalid_short.csv` | 1.2 s (60 rows) | Derived from `valid_walking_subject8.csv` | **Input validation stress-test**: Rejects recordings shorter than 128 samples (2.56 s). Exits with non-zero code `1` and clean message: `need at least 2.56 s of data at 50 Hz`. |
| `invalid_columns.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Schema validation stress-test**: Missing `userAcceleration.z`. Exits with non-zero code `1` and lists missing required sensor columns. |
| `invalid_nan.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Data corruption stress-test**: Injected 3 NaN entries (rows 12, 25, 50). Exits with non-zero code `1` and identifies corrupted row numbers. |
| `out_of_scope_scaled.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Out-of-scope domain-shift warning**: `userAcceleration` channels scaled by 20×. Triggers out-of-scope range warnings (>10% outside training bounds) but allows prediction to complete (exit code `0`). |

---

## Reproduction Command
Run `python -m scripts.make_samples` to rebuild samples from the local dataset.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Saved %s", readme_path)


if __name__ == "__main__":
    make_samples()

