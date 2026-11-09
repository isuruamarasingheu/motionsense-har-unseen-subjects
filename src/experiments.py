"""
Module: src.experiments
Purpose: T4 Ablation studies, noise stress testing, and subgroup analysis.

Responsibilities:
- Ablation study (validation split only):
    * Evaluate impact of removing Dropout (dropout = 0) on the final CNN architecture across seeds 0, 1, 2.
    * Save outputs to `outputs/metrics/ablation.json` and `outputs/figures/fig13_ablation.png`.
- Error analysis:
    * Identify misclassified test windows from best CNN checkpoint.
    * Visualise at least 5 representative failure cases (e.g., sit vs std, ups vs dws) to
      `outputs/figures/fig14_error_examples.png` and save error logs to `errors.csv`.
- Noise stress test:
    * Pre-declare test protocol in `outputs/metrics/noise_protocol.json`.
    * Inject Gaussian noise N(0, sigma^2) with sigma in {0.0, 0.1, 0.25, 0.5, 1.0} to standardized inputs.
    * Re-extract features for RF/MLP and evaluate macro-F1 over 5 repeats per sigma level across 3 seeds.
    * Plot degradation curves to `outputs/figures/fig15_noise_robustness.png`.
- Subgroup analysis:
    * Disaggregate test macro-F1 across male vs female subjects with window counts to `outputs/metrics/subgroup.json`.

Rule compliance:
- Rule 10: Log all experimental data to metrics files; do not fabricate or smooth figures.
- Rule 12: Proper axis labels, units, legends, and dpi=150 formatting.
"""

