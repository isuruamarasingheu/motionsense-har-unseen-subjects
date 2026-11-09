"""
Module: src.predict
Purpose: T5 CLI inference engine for practical single-file activity prediction.

Usage:
    python -m src.predict --model cnn --input samples/walking_example.csv

Responsibilities:
- Load the chosen model checkpoint (.pt or .joblib) and corresponding saved preprocessor scalers.
- Validate input CSV:
    * Assert presence of all 12 required sensor columns:
      attitude.roll, attitude.pitch, attitude.yaw,
      gravity.x, gravity.y, gravity.z,
      rotationRate.x, rotationRate.y, rotationRate.z,
      userAcceleration.x, userAcceleration.y, userAcceleration.z
    * Check minimum length requirement (>= 128 rows for at least 1 valid window).
    * Validate numeric integrity (reject NaNs, infs, non-numeric values).
    * Warn if sensor readings fall outside standard operational ranges.
- Window input data into 128-sample segments (stride 64).
- Execute model inference and output:
    * Majority predicted activity class with human-readable label.
    * Confidence score and mean posterior probability distribution across all 6 classes.
- Return exit code 0 on success; clean, non-traceback error messages and non-zero exit codes on invalid inputs.

Rule compliance:
- Rule 5: Reuses fitted train scaler without refitting.
- Rule 14: Clear terminal output for demo and user presentation.
"""

