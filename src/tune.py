"""
Module: src.tune
Purpose: Systematic hyperparameter tuning within the pre-declared budget.

Responsibilities:
- Read the pre-declared tuning budget from `configs/tuning_budget.yaml`.
- Execute tuning trials strictly on the VALIDATION split (never test split) using seed 0:
    * Random Forest: 6 trials across n_estimators and max_depth.
    * PyTorch MLP: 6 trials across hidden layer sizes, dropout rates, and class weighting.
    * PyTorch 1D-CNN: 6 trials across filter configurations, dropout rates, and kernel sizes.
- Record every trial's outcomes into `outputs/metrics/tuning_log.csv` (including failed or suboptimal runs).
- Identify best configurations based on highest validation macro-F1 score.
- Output finalised configuration files: `configs/mlp_final.yaml` and `configs/cnn_final.yaml`.

Rule compliance:
- Rule 7: Never use test results to select hyperparameters or choose architectures.
- Rule 10: Retain all runs in tuning_log.csv; never hide unsuccessful trials.
"""

