# MotionSense Sample Recordings (`samples/`)

This directory contains demonstration and testing sensor recordings for the `src.predict` CLI inference engine (Task T5b). In compliance with Project Rule 2, raw data is not committed; all samples are created dynamically from the local dataset via `scripts/make_samples.py`.

To rebuild samples from the local dataset at any time, run:
```bash
python -m scripts.make_samples
```

---

## Sample Inventory

| File Name | Duration / Rows | Provenance | Purpose & Expected Behavior |
| :--- | :--- | :--- | :--- |
| `valid_walking_subject8.csv` | 10.0 s (500 rows) | Test Subject 8, `wlk_7`, rows 500–999 | **Normal operational input**: Valid 12 sensor channels at 50 Hz. Exits with code `0`. Predicted activity: `WALKING` with high confidence (~0.90+). |
| `failure_walking_subject1.csv` | 10.0 s (500 rows) | Test Subject 1, `wlk_7`, rows 500–999 | **Physical phone-orientation failure demo**: tilted phone orientation (median pitch -0.86 rad vs -1.28 rad in the same subject's other walking trials). Demonstrates how physical placement mismatch alters predictions. |
| `invalid_short.csv` | 1.2 s (60 rows) | Derived from `valid_walking_subject8.csv` | **Input validation stress-test**: Rejects recordings shorter than 128 samples (2.56 s). Exits with non-zero code `1` and clean message: `need at least 2.56 s of data at 50 Hz`. |
| `invalid_columns.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Schema validation stress-test**: Missing `userAcceleration.z`. Exits with non-zero code `1` and lists missing required sensor columns. |
| `invalid_nan.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Data corruption stress-test**: Injected 3 NaN entries (rows 12, 25, 50). Exits with non-zero code `1` and identifies corrupted row numbers. |
| `out_of_scope_scaled.csv` | 10.0 s (500 rows) | Derived from `valid_walking_subject8.csv` | **Out-of-scope domain-shift warning**: `userAcceleration` channels scaled by 20×. Triggers out-of-scope range warnings (>10% outside training bounds) but allows prediction to complete (exit code `0`). |

---

## Reproduction Command
Run `python -m scripts.make_samples` to rebuild samples from the local dataset.
