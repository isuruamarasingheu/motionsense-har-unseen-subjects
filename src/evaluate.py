"""
Module: src.evaluate
Purpose: T4 Final evaluation on the unseen test split (executed strictly once).

Responsibilities:
- Sole entry point allowed to access the test split, after configurations and checkpoints are frozen.
- Load trained checkpoints across 3 seeds (seeds 0, 1, 2) for:
    * Random Forest baseline
    * PyTorch MLP (final config)
    * PyTorch 1D-CNN (final config)
- Compute test metrics: Macro-F1, overall accuracy, per-class precision, recall, and F1 scores.
- Aggregate runs into `outputs/metrics/results_table.csv` reporting mean +/- std per model.
- Generate evaluation visualisations:
    * Confusion matrices (row-normalised): `fig09_cm_rf.png`, `fig10_cm_mlp.png`, `fig11_cm_cnn.png`
    * Per-class F1 comparison bar chart: `fig12_per_class_f1.png`

Rule compliance:
- Rule 7: The test set is used ONCE, only by src/evaluate.py, only after configs are frozen.
- Rule 10: Every number in the report derives from saved evaluation artifacts.
- Rule 12: Every figure has clear titles, axis labels with units, and legends.
"""

