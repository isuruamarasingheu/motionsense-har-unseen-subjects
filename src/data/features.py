"""
Module: src.data.features
Purpose: Hand-crafted statistical feature extraction for tabular and MLP models.

Responsibilities:
- Transform raw window arrays (N, 128, 12) into tabular feature matrices (N, 84).
- Derive 2 composite magnitude channels from the 12 primary channels:
    * |userAcceleration| = sqrt(userAcceleration.x^2 + y^2 + z^2)
    * |rotationRate| = sqrt(rotationRate.x^2 + y^2 + z^2)
  giving 14 total time-series signals per window.
- Compute 6 summary statistics per signal:
    1. Mean
    2. Standard deviation
    3. Minimum
    4. Maximum
    5. Median
    6. Interquartile Range (IQR = Q75 - Q25)
  yielding exactly 14 x 6 = 84 features per window.
- Ensure identical feature representation across Random Forest, PyTorch MLP, and NumPy MLP.

Rule compliance:
- Rule 13: Feature definitions and mathematical derivations documented clearly.
- Rule 14: Keep vectorised calculations straightforward and explainable.
"""

