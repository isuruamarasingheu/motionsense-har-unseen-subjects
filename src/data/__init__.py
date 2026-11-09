"""
Data processing package for Human Activity Recognition (HAR) using MotionSense.

Submodules:
- load: Raw dataset loading and metadata merging.
- split: Grouped, subject-wise splitting to prevent identity leakage.
- windows: Fixed-length sliding window generation strictly bounded by trial.
- features: Hand-crafted statistical time-series feature extraction.
- preprocess: Train-only feature scaling and raw sequence standardisation.
"""

