# Human Activity Recognition with MotionSense (CW1 Practical)

**Module:** NB6057CEM Artificial Neural Networks  
**Target:** 6-class Human Activity Recognition (`dws`, `ups`, `wlk`, `jog`, `sit`, `std`)  
**Data:** MotionSense Dataset (A_DeviceMotion_data) — Malekzadeh et al. (2018/2019)  

---

## 1. Project Overview

This repository develops, evaluates, and compares multiple machine learning and deep learning models for smartphone-based Human Activity Recognition (HAR) using 50 Hz tri-axial accelerometer, gyroscope, and attitude telemetry from iPhone 6s sensors.

Models investigated:
1. **From-Scratch MLP (Task T2):** 1-hidden-layer MLP built purely in NumPy (float64) with analytical backpropagation and gradient verification.
2. **Non-Neural Baseline (Task T3a):** Random Forest classifier on 84 statistical time-series summary features.
3. **Deep Framework MLP (Task T3b):** Multi-layer perceptron with Batch Normalisation, ReLU, and Dropout in PyTorch.
4. **1D Temporal CNN (Task T3b):** 1D Convolutional Neural Network learning spatial-temporal representations directly from raw 12-channel sequences.

---

## 2. Environment Setup

### Prerequisites
- Python 3.11
- Windows / Linux / macOS (CPU execution)

### Installation
```bash
# Install pinned dependencies
pip install -r requirements.txt

# PyTorch CPU (if installing independently)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Project Structure

```
ANN/
├── CW1_PLAN.md                  # Comprehensive implementation plan & rules
├── README.md                    # Project overview & reproduction guide
├── requirements.txt             # Pinned package versions
├── .gitignore                   # Excludes raw data, pycache, checkpoints
├── MotionSense_dataset/         # Raw dataset (READ-ONLY, not redistributed)
│   ├── data_subjects_info.csv
│   └── A_DeviceMotion_data/
├── configs/                     # Declarative YAML configurations
│   ├── data.yaml
│   ├── tuning_budget.yaml
│   ├── numpy_mlp.yaml
│   ├── baseline_rf.yaml
│   ├── mlp.yaml
│   └── cnn.yaml
├── src/                         # Modular codebase
│   ├── data/                    # Loading, splitting, windowing, features, scaling
│   ├── models/                  # NumPy MLP, PyTorch MLP, PyTorch 1D-CNN
│   ├── utils.py                 # Seeding, JSON helpers, run_info tracking
│   ├── train.py                 # Neural network training loop
│   ├── train_baseline.py        # Random Forest trainer
│   ├── tune.py                  # Hyperparameter tuning on validation split
│   ├── evaluate.py              # Test evaluation (run once on frozen models)
│   ├── experiments.py           # Ablation, noise stress, subgroup analysis
│   ├── benchmark.py             # Model parameter & inference latency benchmark
│   └── predict.py               # CLI inference interface
├── notebooks/                   # Jupyter analysis notebooks
│   ├── 01_eda.ipynb             # Exploratory data audit & visualisations
│   └── 02_numpy_mlp.ipynb       # First-principles NumPy MLP demonstration
├── outputs/                     # Generated experiment artifacts
│   ├── splits/                  # Grouped subject splits & window indices
│   ├── preprocessing/           # Fitted scalers (train-only)
│   ├── checkpoints/             # Trained model weights & config snapshots
│   ├── metrics/                 # Evaluation tables & JSON metrics
│   ├── figures/                 # Publication-ready figures (fig01–fig15)
│   └── logs/                    # Training execution logs & run info
├── samples/                     # Test sample CSVs for CLI inference
└── report/                      # Academic report
    └── technical_record.md      # Structured technical record
```

---

## 4. Execution Pipeline

```bash
# 1. Data preparation (subject-wise split, windowing, feature extraction, train-only scalers)
python -m src.data.prepare

# 2. Smoke test (verify CNN pipeline quickly)
python -m src.train --config configs/cnn.yaml --smoke-test

# 3. Train Random Forest baseline
python -m src.train_baseline --config configs/baseline_rf.yaml

# 4. Hyperparameter tuning (validation split only)
python -m src.tune

# 5. Train final neural models across 3 seeds (0, 1, 2)
python -m src.train --config configs/mlp_final.yaml --seed 0
python -m src.train --config configs/cnn_final.yaml --seed 0

# 6. Final evaluation on test split (run once)
python -m src.evaluate

# 7. Experiments (ablation, noise stress, subgroup analysis)
python -m src.experiments --all

# 8. Complexity and latency benchmarks
python -m src.benchmark

# 9. CLI inference demo
python -m src.predict --model cnn --input samples/walking_example.csv
```

