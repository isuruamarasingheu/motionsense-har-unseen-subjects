# Human Activity Recognition with MotionSense

**Module:** Artificial Neural Networks  
**Task:** Classify phone sensor data into 6 activities (`dws`: downstairs, `ups`: upstairs, `wlk`: walking, `jog`: jogging, `sit`: sitting, `std`: standing)  
**Dataset:** MotionSense (A_DeviceMotion_data) — Malekzadeh et al. (2018/2019)  
**Runs on:** CPU only (Python 3.11, PyTorch CPU, scikit-learn, NumPy)

---

## 1. Overview

This project builds and compares four models that recognise what a person is doing (walking, jogging, sitting, standing, going upstairs, or going downstairs) from 2.56 seconds of iPhone sensor data. The phone records accelerometer, gyroscope, and attitude readings at 50 Hz across 12 channels. Each 128-sample window is one example. 24 people took part.

The key question is whether the models work on **people they have never seen**. To test this fairly, the 24 participants are split by person: 16 for training, 4 for validation, and 4 for testing. No person's data ever appears in more than one group.

The four models are:

1. **NumPy MLP** — a simple one-hidden-layer neural network built from scratch in NumPy, with hand-written backpropagation and gradient checks to prove the maths is correct.
2. **Random Forest** — a non-neural baseline that uses 84 hand-crafted time-domain summary statistics (14 signals × 6 stats: mean, std, min, max, median, IQR) as input.
3. **PyTorch MLP** — a deeper neural network in PyTorch with Batch Normalisation, ReLU, and Dropout, also trained on the 84 features.
4. **1D-CNN** — a convolutional neural network in PyTorch that reads the raw 12-channel sensor windows directly, without needing hand-crafted features.

---

## 2. Data

### Citation
> Malekzadeh, M., Clegg, R. G., Cavallaro, A., and Haddadi, H. (2019). Mobile Sensor Data Anonymization. In *Proceedings of the 2019 ACM/IEEE International Conference on Information Processing in Sensor Networks (IoTDI 2019)*. ACM.  
> Dataset: [https://github.com/mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense)

### How to get the data
1. Download or clone the dataset from [https://github.com/mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense).
2. Put `data_subjects_info.csv` and `A_DeviceMotion_data/` inside a folder called `MotionSense_dataset/`. The zip extracts to a nested folder structure — keep it as is:
   ```
   MotionSense_dataset/
   ├── data_subjects_info.csv
   └── A_DeviceMotion_data/
       └── A_DeviceMotion_data/
           ├── dws_1/
           ├── dws_2/
           └── ... (15 trial folders)
   ```
3. **Not included in this repo**: The raw data and sample CSVs (`samples/*.csv`) are not shipped with this repository (excluded via `.gitignore`). To create the sample files, run `python -m scripts.make_samples`.
4. **Duplicate recording**: Subject 1 has two identical sitting recordings (`sit_5` and `sit_13`, 10,292 rows each). The data loader (`src/data/load.py`) detects this automatically, keeps `sit_5`, and drops the duplicate `sit_13` so that no data is counted twice.

---

## 3. Setup

Requires **Python 3.11**. Everything runs on CPU — no GPU needed. The CPU version of PyTorch is installed automatically from the extra index URL in `requirements.txt`.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Smoke Test (< 1 min)

Run these two commands to check that everything works:

```bash
# Run the test suite (23 tests)
pytest -q

# Run a quick training check (1 epoch, 2 train subjects, nothing saved)
python -m src.train --config configs/cnn.yaml --smoke-test
```

### Measured times
- **`pytest -q`**: 30.58 s (23 passed).
- **Smoke test**: 5.18 s.
- **Total**: ~35.8 s.

---

## 5. Full Run Order

Run these commands in order. Each line has a short explanation.

```bash
# 1. Prepare data: remove duplicates, split by person, create windows, extract features, fit scalers on training data only (~7 s)
python -m src.data.prepare

# 2. Train the NumPy MLP from scratch with gradient checks and learning rate sweeps (~8 s)
python -m src.train_numpy_mlp

# 3. Train Random Forest baseline with 3 different seeds (not timed)
python -m src.train_baseline --config configs/baseline_rf_final.yaml --seed 0
python -m src.train_baseline --config configs/baseline_rf_final.yaml --seed 1
python -m src.train_baseline --config configs/baseline_rf_final.yaml --seed 2

# 4. Tune hyperparameters on the validation set only — test data is never touched here (not timed)
python -m src.tune

# 5. Train final MLP models with seeds 0, 1, 2 (not timed)
python -m src.train --config configs/mlp_final.yaml --seed 0
python -m src.train --config configs/mlp_final.yaml --seed 1
python -m src.train --config configs/mlp_final.yaml --seed 2

# 6. Train final CNN models with seeds 0, 1, 2 (not timed)
python -m src.train --config configs/cnn_final.yaml --seed 0
python -m src.train --config configs/cnn_final.yaml --seed 1
python -m src.train --config configs/cnn_final.yaml --seed 2

# 7. Evaluate all frozen models on the test set ONCE (~5 s)
python -m src.evaluate

# 8. Run experiments: ablation, error analysis, noise stress test, subgroup analysis (not timed)
# Use --regenerate-noise to rebuild the noise summary and fig17 from saved values without re-running evaluations
python -m src.experiments --all

# 9. Benchmark speed and model sizes on validation data (~48 s)
python -m src.benchmark

# 10. Generate sample CSV files from the local dataset (~1 s)
python -m scripts.make_samples

# 11. Predict on a valid walking recording (subject 8, expects: WALKING)
python -m src.predict --model cnn --input samples/valid_walking_subject8.csv

# 12. Predict on a known failure case — tilted phone (subject 1, predicts UPSTAIRS instead of WALKING)
python -m src.predict --model cnn --input samples/failure_walking_subject1.csv

# 13. Try an invalid file that is too short (should exit with code 1 and an error message)
python -m src.predict --model cnn --input samples/invalid_short.csv
```

---

## 6. Where Things Live

| What | Folder | Details |
| :--- | :--- | :--- |
| **Splits** | `outputs/splits/` | `split.json` records which people go in train (16), val (4), and test (4). Window metadata CSVs list every window's trial, position, and label. |
| **Seeds** | `configs/` | All random seeds are declared in config files: split seed = 42 (`data.yaml`), NumPy MLP seed = 42 (`numpy_mlp.yaml`), tuning seed = 0 (`tuning_budget.yaml`), final model seeds = 0, 1, 2. |
| **Configs** | `configs/` | YAML files for every model. **Frozen** (locked before test evaluation): `tuning_budget.yaml`, `baseline_rf_final.yaml`, `mlp_final.yaml`, `cnn_final.yaml`, `ablation_cnn_no_dropout.yaml`. |
| **Checkpoints** | `outputs/checkpoints/` | Saved model weights (`.pt`, `.joblib`, `.npz`), config snapshots, and `run_info_*.json` files. **Committed to git** — they can be regenerated by retraining, but are included so evaluation works immediately. |
| **Metrics** | `outputs/metrics/` | `results_table.csv` (9-row test results), `results_summary.json`, `tuning_log.csv`, `ablation.json`, `subgroup.json`, `noise_robustness.json`, `benchmark.json`, `recommendation_evidence.csv`, error CSVs, and per-window predictions in `predictions/`. |
| **Figures** | `outputs/figures/` | 19 figures (`fig01` through `fig19`), saved at 150 DPI with titles, axis labels, and legends. |
| **Logs** | `outputs/logs/` | Training logs (loss per epoch) for each model/seed, plus `predict_demo.txt`. |

---

## 7. Notebooks

All notebooks are **read-only views** of saved results. They load files from `outputs/` and display them — they do not retrain anything.

| Notebook | Task | What it shows |
| :--- | :--- | :--- |
| `01_eda.ipynb` | **T1b** | Data exploration: class counts (`fig01`), example signals (`fig02`), data per subject (`fig03`), channel ranges (`fig04`), sampling rate checks. |
| `01b_data_pipeline.ipynb` | **T1c** | How data is prepared: person-based splitting (16/4/4), windowing (128 samples, 50% overlap), 84 time-domain features (14 signals × 6 stats), train-only scaling, and checks that no person leaks across splits. |
| `02_numpy_mlp.ipynb` | **T2** | The hand-built NumPy MLP: forward/backward maths, gradient check (error < $10^{-6}$), overfitting test on 32 samples (`fig06`), training curves (`fig05`), and learning rate sweep (`fig07`). |
| `03_models_and_tuning.ipynb` | **T3a & T3b** | Random Forest, PyTorch MLP, and 1D-CNN architectures. Tuning log (validation only). Loss curves (`fig08`, `fig09`). |
| `04_evaluation.ipynb` | **T4a & T4b** | Test results (run once): 9-row results table, confusion matrices (`fig10`–`fig12`), per-class F1 comparison (`fig13`), validation vs test gap (`fig14`). |
| `05_experiments.ipynb` | **T4c & T4d** | Dropout ablation on the validation set (`fig15`), error analysis showing failure cases (`fig16`), noise robustness curves (`fig17`), and gender subgroup table (`subgroup.json`). *(Note: `fig18_per_subject_f1.png` is generated by `src.experiments` into `outputs/figures/` but is not displayed in this notebook.)* |
| `06_recommendation_inference.ipynb` | **T5a & T5b** | Speed and size benchmarks (`fig19`), model recommendation, CLI inference demos, input validation checks, and the tilted-phone failure case. |

---

## 8. Project Structure

```
ANN/
├── configs/                             # YAML configuration files
│   ├── ablation_cnn_no_dropout.yaml     # Ablation: CNN without dropout [FROZEN]
│   ├── baseline_rf.yaml                 # Random Forest template
│   ├── baseline_rf_final.yaml           # Random Forest final config [FROZEN]
│   ├── cnn.yaml                         # 1D-CNN template
│   ├── cnn_final.yaml                   # 1D-CNN final config [FROZEN]
│   ├── data.yaml                        # Data split and windowing settings (seed 42)
│   ├── mlp.yaml                         # PyTorch MLP template
│   ├── mlp_final.yaml                   # PyTorch MLP final config [FROZEN]
│   ├── numpy_mlp.yaml                   # NumPy MLP settings (seed 42)
│   └── tuning_budget.yaml               # Tuning search grid [FROZEN]
├── MotionSense_dataset/                 # Raw data (not in git)
│   ├── data_subjects_info.csv           # Info about 24 participants
│   └── A_DeviceMotion_data/
│       └── A_DeviceMotion_data/         # 15 trial folders (360 CSVs total)
├── notebooks/                           # Read-only Jupyter notebooks
│   ├── 01_eda.ipynb                     # T1b: Data exploration and plots
│   ├── 01b_data_pipeline.ipynb          # T1c: Data preparation pipeline
│   ├── 02_numpy_mlp.ipynb               # T2: Hand-built NumPy MLP
│   ├── 03_models_and_tuning.ipynb       # T3: Models and tuning
│   ├── 04_evaluation.ipynb              # T4a/T4b: Test evaluation
│   ├── 05_experiments.ipynb             # T4c/T4d: Ablation, errors, noise, subgroups
│   └── 06_recommendation_inference.ipynb # T5: Benchmarks and inference demo
├── outputs/                             # All generated files
│   ├── cache/                           # Processed arrays (not in git)
│   ├── checkpoints/                     # Saved model weights and run info
│   ├── figures/                         # Figures fig01–fig19
│   ├── logs/                            # Training logs and demo output
│   ├── metrics/                         # Results, tuning logs, predictions
│   ├── preprocessing/                   # Fitted scalers (train-only)
│   └── splits/                          # Subject split and window metadata
├── samples/                             # Sample CSVs for inference demos (not in git)
│   ├── README.md                        # What each sample file is for
│   ├── valid_walking_subject8.csv       # Good input: subject 8 walking (10 s, 500 rows)
│   ├── failure_walking_subject1.csv     # Known failure: tilted phone
│   ├── invalid_short.csv                # Too short (60 rows, needs 128)
│   ├── invalid_columns.csv              # Missing a sensor column
│   ├── invalid_nan.csv                  # Has NaN values
│   └── out_of_scope_scaled.csv          # Extreme values (triggers warning)
├── scripts/                             # Helper scripts
│   ├── build_notebook_05.py             # Builds and runs notebook 05
│   ├── build_notebook_06.py             # Builds and runs notebook 06
│   └── make_samples.py                  # Creates sample CSVs from raw data
├── src/                                 # Main code
│   ├── data/
│   │   ├── features.py                  # 84 time-domain features (14 signals × 6 stats)
│   │   ├── load.py                      # Loads dataset, removes duplicate recordings
│   │   ├── prepare.py                   # Runs the full data preparation pipeline
│   │   ├── preprocess.py                # Standardisation scalers (fitted on training data only)
│   │   ├── split.py                     # Splits data by person into train/val/test
│   │   └── windows.py                   # Cuts recordings into fixed-length windows
│   ├── models/
│   │   ├── cnn.py                       # 1D-CNN with global average pooling (PyTorch)
│   │   ├── mlp.py                       # MLP with BatchNorm and Dropout (PyTorch)
│   │   └── numpy_mlp.py                 # One-layer MLP built from scratch (NumPy)
│   ├── benchmark.py                     # Measures speed and model size
│   ├── evaluate.py                      # Tests models on the held-out test set (run once)
│   ├── experiments.py                   # Ablation, error analysis, noise test, subgroup analysis
│   ├── predict.py                       # Command-line tool to classify a CSV file
│   ├── train.py                         # Trains PyTorch models (MLP and CNN)
│   ├── train_baseline.py                # Trains the Random Forest
│   ├── train_numpy_mlp.py               # Trains the NumPy MLP with gradient checks
│   ├── tune.py                          # Tries different hyperparameters on the validation set
│   └── utils.py                         # Seed setting, JSON helpers, run info tracking
├── tests/                               # Automated tests
│   ├── test_data_pipeline.py            # Checks data pipeline and leakage prevention
│   ├── test_numpy_mlp.py                # Checks gradient computation and MLP logic
│   ├── test_phase3_models.py            # Checks PyTorch models and Random Forest
│   └── test_predict.py                  # Checks prediction CLI and error handling
├── .gitignore                           # Files excluded from git
├── CW1_PLAN.md                          # Coursework plan
├── README.md                            # This file
└── requirements.txt                     # Pinned dependencies (includes CPU PyTorch)
```

---

## 9. Reproducibility Notes

1. **Seeds**: Every random operation (splitting, weight initialisation, shuffling, noise sampling) uses a fixed seed from the config files, so results can be reproduced.
2. **CPU differences**: Even with the same seeds, small floating-point differences (around $10^{-5}$ to $10^{-6}$) can appear on different CPUs because of how multi-threaded matrix maths rounds numbers. The overall rankings and conclusions stay the same.
3. **Test set used once**: The test subjects (1, 8, 18, 21) were evaluated **once**, after all configs and models were frozen. The validation subjects (3, 15, 20, 23) were used for tuning and ablation only. `src.evaluate` is the only code that touches the test set.
4. **Noise test**: The noise experiment (`src.experiments --noise`) adds Gaussian noise at five levels ($\sigma \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$, 5 random draws per level) to the test data after models are frozen, to see how accuracy degrades. In the noise tables, $\pm$ is the standard deviation across 3 model seeds, each averaged over 5 noise draws. Running `python -m src.experiments --regenerate-noise` rebuilds the summary and `fig17` from saved values without re-running the evaluations.

---

## 10. References and Borrowed Code

### Dataset
- **MotionSense**:  
  Malekzadeh, M., Clegg, R. G., Cavallaro, A., and Haddadi, H. (2019). Mobile Sensor Data Anonymization. In *Proceedings of the 2019 ACM/IEEE International Conference on Information Processing in Sensor Networks (IoTDI 2019)*, Montreal, Canada.  
  URL: [https://github.com/mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense)

### Libraries used
- **PyTorch** (`torch` 2.15.0+cpu) — CNN and MLP training.
- **scikit-learn** (`sklearn` 1.7.2) — Random Forest, F1 metrics, scalers.
- **NumPy** (`numpy` 1.26.4) — hand-built MLP, array maths.
- **pandas** (`pandas` 2.3.2) — data loading, tables, results.
- **matplotlib** (`matplotlib` 3.8.3) & **seaborn** (`seaborn` 0.13.2) — all figures.
- **joblib** (`joblib` 1.5.2) — saving scalers and Random Forest models.
- **PyYAML** (`yaml` 6.0.2) — reading config files.
- **pytest** (`pytest` 9.2.0) — automated tests.

### Borrowed code
No code was copied from external repositories; libraries used are listed above.
