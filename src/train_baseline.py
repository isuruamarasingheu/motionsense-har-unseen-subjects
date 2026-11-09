"""
Module: src.train_baseline
Purpose: T3a Non-neural Random Forest baseline trainer.

Responsibilities:
- Train a scikit-learn `RandomForestClassifier` on the 84 scaled tabular features.
- Support `class_weight="balanced"` to address activity class imbalances.
- Evaluate performance on the validation split using macro-F1 and accuracy.
- Save trained baseline model using `joblib` into `outputs/checkpoints/baseline_rf_seed<seed>.joblib`.
- Save validation metrics, configuration copy, and execution provenance via `write_run_info`.

Rule compliance:
- Rule 7: Do not evaluate or load the test split during training/validation.
- Rule 8: Save checkpoints, config copy, metrics JSON, and run_info.json.
- Rule 9: Seed Random Forest via `random_state`.
"""

