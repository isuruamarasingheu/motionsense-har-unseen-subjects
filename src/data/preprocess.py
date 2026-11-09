"""
Module: src.data.preprocess
Purpose: Leakage-free feature scaling and sequence normalisation.

Responsibilities:
- Fit standardisation transformers (e.g., StandardScaler for 84 tabular features;
  per-channel mean/std for raw 12-channel windows) EXCLUSIVELY on training data.
- Never use validation or test data during fitting.
- Persist fitted transformers and statistics to `outputs/preprocessing/`
  using joblib (.pkl) and human-readable JSON formats.
- Apply saved transformers consistently to validation, test, and inference inputs.

Rule compliance:
- Rule 5: Any scaler/normaliser is fitted on TRAIN data only, then saved and
  reused for val/test/inference.
"""

