"""
Module: src.train
Purpose: Generic PyTorch model training and validation pipeline.

Responsibilities:
- Parse CLI arguments (e.g. `--config configs/cnn.yaml --seed 0 --smoke-test`).
- Set random seeds across Python, NumPy, and PyTorch using `src.utils.set_seed`.
- Instantiate DataLoaders (batch size 64, shuffle train split only).
- Setup Adam optimiser (learning rate, weight decay from config) and CrossEntropyLoss
  (with optional class weights computed solely from training labels).
- Training loop:
    * Log train loss, validation loss, validation macro-F1, and validation accuracy per epoch.
    * Early stopping tracked on validation macro-F1 with patience = 8 epochs.
    * Checkpoint the best model to `outputs/checkpoints/<model_name>_seed<seed>.pt`.
    * Save copy of active config and write `run_info.json`.
    * Save training history log (CSV) and plot loss curves to `outputs/figures/`.
- Provide `--smoke-test` mode: subset of 2 train subjects, 1 val subject, 1 epoch (< 1 min execution).

Rule compliance:
- Rule 7: Never touch or evaluate the test split in this training script.
- Rule 8: Save checkpoints, config copy, training log, and run_info.json for every run.
- Rule 9: Set all seeds from config.
- Rule 10: Log all training progression metrics directly from execution.
"""

