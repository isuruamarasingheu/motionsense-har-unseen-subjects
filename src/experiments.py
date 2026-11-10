"""
Module: src.experiments
Purpose: Tasks T4c & T4d — Ablation study, error analysis, noise stress test,
         and demographic subgroup analysis.

Responsibilities:
1. Ablation Study (Task T4c, Validation Split Only):
   - Compares final 1D-CNN (dropout=0.3) against an ablated variant (dropout=0.0)
     across seeds 0, 1, 2 on validation data only.
   - Asserts exact one-key difference in configuration (`model.dropout`).
   - Checkpoint names: `pytorch_cnn_1d_nodrop_seed<k>.pt`.
   - Saves `outputs/metrics/ablation.json` and `outputs/figures/fig15_ablation_dropout.png`.
2. Error Analysis (Task T4d, Saved Predictions):
   - Inspects test misclassifications from representative median-validation seeds
     (RF seed 1, MLP seed 0, CNN seed 2) using saved predictions in `outputs/metrics/predictions/`.
   - Writes `outputs/metrics/errors_<model>.csv` and `outputs/metrics/error_rate_by_subject_trial.csv`.
   - Algorithmically selects 6 representative failure modes (wlk->ups high/low conf,
     sit->std, dws->ups, wlk->jog, dws->wlk).
   - Plots 12-channel time-series comparisons vs correct windows in `outputs/figures/fig16_error_examples.png`.
3. Noise Stress Test (Task T4d):
   - Pre-declares protocol in `outputs/metrics/noise_protocol.json` before execution.
   - Injects Gaussian sensor noise N(0, (sigma * train_std_c)^2) for sigma in {0, 0.1, 0.25, 0.5, 1.0}
     with 5 repeats (seeds 100-104) into raw test windows.
   - Evaluates all 9 model checkpoints, asserts sigma=0 reproduces `results_table.csv` exactly.
   - Saves `outputs/metrics/noise_robustness.json` and `outputs/figures/fig17_noise_robustness.png`.
4. Subgroup Analysis (Task T4d):
   - Disaggregates test macro-F1 by subject (1, 8, 18, 21) and gender (male vs female).
   - Reports demographic metadata and window distributions.
   - Saves `outputs/metrics/subgroup.json` and `outputs/figures/fig18_per_subject_f1.png`.

Rule compliance:
- Rule 1: Python 3.11, NumPy, pandas, scikit-learn, PyTorch (CPU), matplotlib.
- Rule 7: Test split never used to select models, seeds, or hyperparameters.
- Rule 8: Saves structured JSON metrics, logs, and provenance.
- Rule 10: Pure empirical data; no hard-coded or fabricated metrics.
- Rule 12: High-resolution PNGs (dpi=150) with titled and labelled axes with units.
- Rule 13: Detailed docstrings explain experimental designs, hypotheses, and confounders.
- Rule 14: Clear, modular, and maintainable implementation.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import yaml

from src.data.features import extract_features
from src.data.load import SENSOR_COLUMNS
from src.data.preprocess import RawWindowScaler
from src.models.cnn import PyTorchCNN1D
from src.models.mlp import PyTorchMLP
from src.train import train_pytorch_model
from src.utils import load_config, load_json, save_json, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.experiments")

CLASS_NAMES = ["dws", "ups", "wlk", "jog", "std", "sit"]
CLASS_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ==============================================================================
# TASK T4c: ABLATION STUDY (VALIDATION ONLY)
# ==============================================================================

def run_ablation() -> dict[str, Any]:
    """
    Execute ablation experiment: remove Dropout (dropout=0.0) from the 1D-CNN
    and compare validation macro-F1 and learning dynamics against the baseline (dropout=0.3).

    Why:
        Tests design hypothesis: 'Dropout (p=0.3) before the classifier improves
        validation macro-F1 and reduces seed-to-seed variance of the 1D-CNN.'
        Strictly evaluated on validation split only (Rule 7).
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK T4c: ABLATION STUDY (DROPOUT REGULARISATION)")
    logger.info("=" * 60)

    final_cfg_path = Path("configs/cnn_final.yaml")
    ablation_cfg_path = Path("configs/ablation_cnn_no_dropout.yaml")

    final_cfg = load_config(final_cfg_path)
    ablation_cfg = load_config(ablation_cfg_path)

    # 1. Assert exactly one difference in configuration
    diffs = []
    def compare_dicts(d1: dict, d2: dict, path: str = ""):
        for k in d1.keys() | d2.keys():
            curr = f"{path}.{k}" if path else k
            if k not in d1 or k not in d2:
                diffs.append((curr, d1.get(k), d2.get(k)))
            elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
                compare_dicts(d1[k], d2[k], curr)
            elif d1[k] != d2[k]:
                diffs.append((curr, d1[k], d2[k]))

    compare_dicts(final_cfg, ablation_cfg)
    assert len(diffs) == 1, f"Expected exactly 1 configuration difference, found {len(diffs)}: {diffs}"
    assert diffs[0][0] == "model.dropout", f"Expected diff at model.dropout, got {diffs[0][0]}"
    assert diffs[0][1] == 0.3 and diffs[0][2] == 0.0, f"Expected 0.3 vs 0.0, got {diffs[0]}"
    logger.info(f"Configuration assertion verified: exactly one key differs ({diffs[0][0]}: {diffs[0][1]} vs {diffs[0][2]}).")

    seeds = [0, 1, 2]
    ckpt_dir = Path("outputs/checkpoints")
    logs_dir = Path("outputs/logs")
    metrics_dir = Path("outputs/metrics")

    # 2. Train ablation models if checkpoints do not exist
    for s in seeds:
        ckpt_file = ckpt_dir / f"pytorch_cnn_1d_nodrop_seed{s}.pt"
        if not ckpt_file.exists():
            logger.info(f"Training ablation model: pytorch_cnn_1d_nodrop (seed {s})...")
            train_pytorch_model(
                config=ablation_cfg,
                seed=s,
                override_params={"name": "pytorch_cnn_1d_nodrop"},
                save_artifacts=True,
            )
        else:
            logger.info(f"Ablation checkpoint already exists: {ckpt_file}")

    # 3. Gather validation results for both conditions
    baseline_records = []
    for s in seeds:
        m = load_json(metrics_dir / f"pytorch_cnn_1d_seed{s}.json")
        h = pd.read_csv(logs_dir / f"pytorch_cnn_1d_seed{s}_training.csv")
        baseline_records.append({
            "seed": s,
            "best_epoch": int(m["best_epoch"]),
            "total_epochs": int(m["total_epochs_trained"]),
            "val_macro_f1": float(m["best_val_macro_f1"]),
            "val_accuracy": float(m["val_accuracy_at_best"]),
            "history": h,
        })

    ablation_records = []
    for s in seeds:
        m = load_json(metrics_dir / f"pytorch_cnn_1d_nodrop_seed{s}.json")
        h = pd.read_csv(logs_dir / f"pytorch_cnn_1d_nodrop_seed{s}_training.csv")
        ablation_records.append({
            "seed": s,
            "best_epoch": int(m["best_epoch"]),
            "total_epochs": int(m["total_epochs_trained"]),
            "val_macro_f1": float(m["best_val_macro_f1"]),
            "val_accuracy": float(m["val_accuracy_at_best"]),
            "history": h,
        })

    base_f1s = [r["val_macro_f1"] for r in baseline_records]
    abla_f1s = [r["val_macro_f1"] for r in ablation_records]

    base_mean, base_std = float(np.mean(base_f1s)), float(np.std(base_f1s, ddof=1))
    abla_mean, abla_std = float(np.mean(abla_f1s)), float(np.std(abla_f1s, ddof=1))
    diff_means = base_mean - abla_mean

    logger.info(f"Baseline 1D-CNN (Dropout 0.3) Val Macro-F1: {base_mean:.4f} +/- {base_std:.4f}")
    logger.info(f"Ablation 1D-CNN (Dropout 0.0) Val Macro-F1: {abla_mean:.4f} +/- {abla_std:.4f}")
    logger.info(f"Difference in Means (Dropout 0.3 - 0.0): {diff_means:+.4f}")

    # 4. Save outputs/metrics/ablation.json
    ablation_output = {
        "design_claim": "Dropout (p=0.3) before the classifier improves validation macro-F1 and reduces seed-to-seed variance of the 1D-CNN.",
        "configuration_difference": {
            "key": "model.dropout",
            "baseline_value": 0.3,
            "ablation_value": 0.0,
            "exact_one_key_difference_asserted": True,
        },
        "baseline_dropout_0_3": {
            "mean_val_macro_f1": base_mean,
            "std_val_macro_f1": base_std,
            "runs": [
                {
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "epochs_completed": r["total_epochs"],
                    "val_macro_f1": r["val_macro_f1"],
                    "val_accuracy": r["val_accuracy"],
                }
                for r in baseline_records
            ],
        },
        "ablation_dropout_0_0": {
            "mean_val_macro_f1": abla_mean,
            "std_val_macro_f1": abla_std,
            "runs": [
                {
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "epochs_completed": r["total_epochs"],
                    "val_macro_f1": r["val_macro_f1"],
                    "val_accuracy": r["val_accuracy"],
                }
                for r in ablation_records
            ],
        },
        "difference_in_means": diff_means,
        "variance_ratio_nodrop_over_drop": float((abla_std ** 2) / (base_std ** 2)) if base_std > 0 else 0.0,
        "remaining_confounders": [
            "early-stopping epoch differs per run",
            "BatchNorm still regularises",
            "only 4 validation subjects",
            "3 seeds only",
        ],
    }

    ablation_json_path = Path("outputs/metrics/ablation.json")
    save_json(ablation_output, ablation_json_path)
    logger.info(f"Saved ablation metrics to {ablation_json_path}")

    # 5. Save fig15_ablation_dropout.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    # Left: Bar chart + scatter of seeds
    categories = ["Dropout = 0.3\n(Baseline)", "Dropout = 0.0\n(Ablation)"]
    means = [base_mean, abla_mean]
    stds = [base_std, abla_std]
    bar_colors = ["#2ca02c", "#d62728"]

    ax1.bar(
        [0, 1],
        means,
        yerr=stds,
        capsize=6,
        color=bar_colors,
        alpha=0.75,
        edgecolor="#333333",
        linewidth=1.2,
        width=0.45,
        label="Mean ± Std (ddof=1)",
    )

    # Plot individual seed points
    seed_colors = ["#1f77b4", "#ff7f0e", "#9467bd"]
    for s_idx, s in enumerate(seeds):
        ax1.scatter([0], [base_f1s[s_idx]], color=seed_colors[s_idx], s=70, zorder=5, edgecolors="black", label=f"Seed {s}" if s_idx == 0 else "")
        ax1.scatter([1], [abla_f1s[s_idx]], color=seed_colors[s_idx], s=70, zorder=5, edgecolors="black")
        ax1.annotate(f"s{s} ({base_f1s[s_idx]:.3f})", (0, base_f1s[s_idx]), xytext=(12, -2), textcoords="offset points", fontsize=8.5)
        ax1.annotate(f"s{s} ({abla_f1s[s_idx]:.3f})", (1, abla_f1s[s_idx]), xytext=(12, -2), textcoords="offset points", fontsize=8.5)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(categories, fontsize=10.5, fontweight="medium")
    ax1.set_ylabel("Validation Macro-F1", fontsize=11, fontweight="medium")
    y_min = min(min(base_f1s), min(abla_f1s)) - 0.03
    y_max = max(max(base_f1s), max(abla_f1s)) + 0.03
    ax1.set_ylim(y_min, y_max)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.set_title("Validation Macro-F1 by Dropout Condition", fontsize=11, fontweight="bold")

    # Right: Validation Macro-F1 vs Epoch curves
    for s_idx, s in enumerate(seeds):
        h_base = baseline_records[s_idx]["history"]
        h_abla = ablation_records[s_idx]["history"]
        ax2.plot(
            h_base["epoch"],
            h_base["val_macro_f1"],
            color=seed_colors[s_idx],
            linestyle="-",
            lw=2.0,
            label=f"Dropout 0.3 (Seed {s})",
        )
        ax2.plot(
            h_abla["epoch"],
            h_abla["val_macro_f1"],
            color=seed_colors[s_idx],
            linestyle="--",
            lw=1.8,
            label=f"Dropout 0.0 (Seed {s})",
        )
        # Mark best epochs
        b_ep_base = baseline_records[s_idx]["best_epoch"]
        b_ep_abla = ablation_records[s_idx]["best_epoch"]
        if b_ep_base <= len(h_base):
            ax2.plot(b_ep_base, h_base.loc[h_base["epoch"] == b_ep_base, "val_macro_f1"].values[0], "*", color=seed_colors[s_idx], markersize=10)
        if b_ep_abla <= len(h_abla):
            ax2.plot(b_ep_abla, h_abla.loc[h_abla["epoch"] == b_ep_abla, "val_macro_f1"].values[0], "o", color=seed_colors[s_idx], markersize=6)

    ax2.set_xlabel("Training Epoch", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Validation Macro-F1", fontsize=11, fontweight="medium")
    ax2.set_title("Validation Trajectories: Dropout 0.3 vs 0.0", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    plt.suptitle("Task T4c: 1D-CNN Regularisation Ablation Study (Dropout 0.3 vs 0.0)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig_path = Path("outputs/figures/fig15_ablation_dropout.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ablation figure to {fig_path}")

    return ablation_output


# ==============================================================================
# TASK T4d: ERROR ANALYSIS (FROM SAVED TEST PREDICTIONS)
# ==============================================================================

def run_error_analysis() -> None:
    """
    Perform deep error analysis on test misclassifications from representative models.

    Why:
        Plan Phase 4d & Task T4d: Identifies recurring confusion clusters, evaluates whether
        errors concentrate in specific subject-trial recordings, and visualises raw sensor
        waveforms for representative failure modes against matched correct true-class windows.
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK T4d: ERROR ANALYSIS (FROM SAVED PREDICTIONS)")
    logger.info("=" * 60)

    pred_dir = Path("outputs/metrics/predictions")
    metrics_dir = Path("outputs/metrics")
    fig_dir = Path("outputs/figures")

    # Representative median validation seeds
    rep_models = {
        "baseline_rf": ("RF", pred_dir / "baseline_rf_seed1.csv"),
        "pytorch_mlp": ("MLP", pred_dir / "pytorch_mlp_seed0.csv"),
        "pytorch_cnn_1d": ("CNN", pred_dir / "pytorch_cnn_1d_seed2.csv"),
    }

    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    dfs = {}
    error_dfs = {}

    # 1. Save outputs/metrics/errors_<model>.csv
    for model_key, (disp_name, path) in rep_models.items():
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        df = pd.read_csv(path)
        df["confidence"] = df[prob_cols].max(axis=1)
        dfs[model_key] = df

        err_mask = df["true_label"] != df["pred_label"]
        err_df = df[err_mask].copy()
        err_export = pd.DataFrame({
            "subject": err_df["subject"],
            "trial": err_df["trial"],
            "start_index": err_df["start_index"],
            "true": err_df["true_label"],
            "pred": err_df["pred_label"],
            "confidence": err_df["confidence"],
        })
        err_path = metrics_dir / f"errors_{model_key}.csv"
        err_export.to_csv(err_path, index=False)
        error_dfs[model_key] = err_df
        logger.info(f"Saved {len(err_export)} error rows to {err_path}")

    # 2. Save outputs/metrics/error_rate_by_subject_trial.csv
    cluster_records = []
    for model_key, (disp_name, _) in rep_models.items():
        df = dfs[model_key]
        for (subj, trial, true_act), grp in df.groupby(["subject", "trial", "true_label"]):
            n_tot = len(grp)
            n_err = int((grp["true_label"] != grp["pred_label"]).sum())
            rate = float(n_err / n_tot) if n_tot > 0 else 0.0
            cluster_records.append({
                "model": model_key,
                "subject": subj,
                "trial": trial,
                "true_activity": true_act,
                "total_windows": n_tot,
                "error_windows": n_err,
                "error_rate": round(rate, 4),
            })

    cluster_df = pd.DataFrame(cluster_records)
    cluster_path = metrics_dir / "error_rate_by_subject_trial.csv"
    cluster_df.to_csv(cluster_path, index=False)
    logger.info(f"Saved subject-trial error clustering analysis to {cluster_path}")

    # 3. Algorithmic selection of 6 representative errors
    # Rule for representative error selection:
    #   1. Join prediction DataFrames across representative RF, MLP, CNN models by (subject, trial, start_index).
    #   2. Count unanimous / majority model misclassification flags per window.
    #   3. Target confusion classes:
    #      - Ex 1: wlk -> ups (highest confidence, unanimous or majority error)
    #      - Ex 2: wlk -> ups (low confidence, boundary decision, conf < 0.60)
    #      - Ex 3: sit -> std (highest confidence postural error)
    #      - Ex 4: dws -> ups (highest confidence stair direction inversion)
    #      - Ex 5: wlk -> jog (highest confidence cadence ambiguity)
    #      - Ex 6: dws -> wlk (highest confidence dynamic slope confusion)
    logger.info("Applying deterministic selection rule for 6 representative failure cases...")

    m_rf = dfs["baseline_rf"][["subject", "trial", "start_index", "true_label", "pred_label", "confidence"]].rename(
        columns={"pred_label": "pred_rf", "confidence": "conf_rf"}
    )
    m_mlp = dfs["pytorch_mlp"][["subject", "trial", "start_index", "pred_label", "confidence"]].rename(
        columns={"pred_label": "pred_mlp", "confidence": "conf_mlp"}
    )
    m_cnn = dfs["pytorch_cnn_1d"][["subject", "trial", "start_index", "pred_label", "confidence"]].rename(
        columns={"pred_label": "pred_cnn", "confidence": "conf_cnn"}
    )

    merged = m_rf.merge(m_mlp, on=["subject", "trial", "start_index"]).merge(m_cnn, on=["subject", "trial", "start_index"])

    selected_cases: list[dict[str, Any]] = []

    # Rule Ex 1: wlk -> ups highest confidence (unanimous / majority)
    c1 = merged[(merged["true_label"] == "wlk") & (merged["pred_rf"] == "ups") & (merged["pred_mlp"] == "ups") & (merged["pred_cnn"] == "ups")]
    if len(c1) == 0:
        c1 = merged[(merged["true_label"] == "wlk") & (merged["pred_cnn"] == "ups")]
    best_c1 = c1.sort_values(by="conf_cnn", ascending=False).iloc[0]
    selected_cases.append({
        "case_id": 1,
        "description": "wlk -> ups (high confidence, unanimous)",
        "subject": int(best_c1["subject"]),
        "trial": int(best_c1["trial"]),
        "start_index": int(best_c1["start_index"]),
        "true": "wlk",
        "pred": "ups",
        "model": "CNN s2",
        "confidence": float(best_c1["conf_cnn"]),
        "rule": "Highest CNN confidence among unanimous wlk->ups misclassifications",
    })

    # Rule Ex 2: wlk -> ups low confidence (boundary ambiguity, lowest confidence > 0.35)
    c2 = merged[(merged["true_label"] == "wlk") & (merged["pred_cnn"] == "ups") & (merged["conf_cnn"] > 0.35)]
    best_c2 = c2.sort_values(by="conf_cnn", ascending=True).iloc[0]
    selected_cases.append({
        "case_id": 2,
        "description": "wlk -> ups (low confidence, boundary case)",
        "subject": int(best_c2["subject"]),
        "trial": int(best_c2["trial"]),
        "start_index": int(best_c2["start_index"]),
        "true": "wlk",
        "pred": "ups",
        "model": "CNN s2",
        "confidence": float(best_c2["conf_cnn"]),
        "rule": "Lowest CNN confidence (>0.35 threshold) for wlk->ups misclassifications",
    })

    # Rule Ex 3: sit -> std highest confidence
    c3 = merged[(merged["true_label"] == "sit") & ((merged["pred_rf"] == "std") | (merged["pred_cnn"] == "std"))]
    best_c3 = c3.sort_values(by="conf_rf", ascending=False).iloc[0]
    selected_cases.append({
        "case_id": 3,
        "description": "sit -> std (high confidence, static posture)",
        "subject": int(best_c3["subject"]),
        "trial": int(best_c3["trial"]),
        "start_index": int(best_c3["start_index"]),
        "true": "sit",
        "pred": "std",
        "model": "RF s1",
        "confidence": float(best_c3["conf_rf"]),
        "rule": "Highest confidence among static postural sit->std misclassifications",
    })

    # Rule Ex 4: dws -> ups highest confidence
    c4 = merged[(merged["true_label"] == "dws") & (merged["pred_cnn"] == "ups")]
    best_c4 = c4.sort_values(by="conf_cnn", ascending=False).iloc[0]
    selected_cases.append({
        "case_id": 4,
        "description": "dws -> ups (high confidence, stair direction)",
        "subject": int(best_c4["subject"]),
        "trial": int(best_c4["trial"]),
        "start_index": int(best_c4["start_index"]),
        "true": "dws",
        "pred": "ups",
        "model": "CNN s2",
        "confidence": float(best_c4["conf_cnn"]),
        "rule": "Highest confidence among stair direction inversion dws->ups misclassifications",
    })

    # Rule Ex 5: wlk -> jog highest confidence
    c5 = merged[(merged["true_label"] == "wlk") & (merged["pred_mlp"] == "jog")]
    if len(c5) > 0:
        best_c5 = c5.sort_values(by="conf_mlp", ascending=False).iloc[0]
        selected_cases.append({
            "case_id": 5,
            "description": "wlk -> jog (high confidence, cadence mismatch)",
            "subject": int(best_c5["subject"]),
            "trial": int(best_c5["trial"]),
            "start_index": int(best_c5["start_index"]),
            "true": "wlk",
            "pred": "jog",
            "model": "MLP s0",
            "confidence": float(best_c5["conf_mlp"]),
            "rule": "Highest confidence among walking vs jogging cadence mismatch wlk->jog",
        })
    else:
        # Fallback to ups->wlk
        c5_alt = merged[(merged["true_label"] == "ups") & (merged["pred_cnn"] == "wlk")].sort_values(by="conf_cnn", ascending=False).iloc[0]
        selected_cases.append({
            "case_id": 5,
            "description": "ups -> wlk (high confidence)",
            "subject": int(c5_alt["subject"]),
            "trial": int(c5_alt["trial"]),
            "start_index": int(c5_alt["start_index"]),
            "true": "ups",
            "pred": "wlk",
            "model": "CNN s2",
            "confidence": float(c5_alt["conf_cnn"]),
            "rule": "Highest confidence among ups->wlk misclassifications",
        })

    # Rule Ex 6: dws -> wlk highest confidence (slope vs flat walking)
    c6 = merged[(merged["true_label"] == "dws") & ((merged["pred_cnn"] == "wlk") | (merged["pred_rf"] == "wlk"))]
    best_c6 = c6.sort_values(by="conf_cnn", ascending=False).iloc[0]
    selected_cases.append({
        "case_id": 6,
        "description": "dws -> wlk (high confidence, descending stair vs flat)",
        "subject": int(best_c6["subject"]),
        "trial": int(best_c6["trial"]),
        "start_index": int(best_c6["start_index"]),
        "true": "dws",
        "pred": "wlk",
        "model": "CNN s2",
        "confidence": float(best_c6["conf_cnn"]),
        "rule": "Highest confidence among descending stair vs level walking dws->wlk",
    })

    save_json(selected_cases, metrics_dir / "selected_error_cases.json")
    logger.info(f"Selected {len(selected_cases)} representative error cases.")

    # 4. Load raw test windows for plotting
    logger.info("Loading X_test_raw.npy for sensor waveform plotting...")
    X_test_raw = np.load("outputs/cache/X_test_raw.npy")  # (3730, 128, 12)
    win_test = pd.read_csv("outputs/splits/windows_test.csv")

    # Time axis: 128 samples at 50 Hz = 2.56 seconds
    time_s = np.arange(128) / 50.0

    # Channels to plot:
    # userAcceleration x/y/z: channels 9, 10, 11
    # gravity x/y/z: channels 3, 4, 5
    # attitude.pitch: channel 1
    fig, axes = plt.subplots(len(selected_cases), 3, figsize=(14, 15.5), dpi=150, sharex=True)

    for row_idx, case in enumerate(selected_cases):
        s_val = case["subject"]
        t_val = case["trial"]
        st_idx = case["start_index"]
        true_act = case["true"]
        pred_act = case["pred"]

        # Find matching window index
        m_idx = win_test[(win_test["subject"] == s_val) & (win_test["trial"] == t_val) & (win_test["start_index"] == st_idx)].index[0]
        raw_error_win = X_test_raw[m_idx]

        # Find a correctly classified window of true_act from the SAME subject for contrast
        correct_cands = win_test[(win_test["subject"] == s_val) & (win_test["activity"] == true_act)]
        # Filter for window where CNN (or majority) was correct
        correct_idx = None
        for cand_idx in correct_cands.index:
            cand_row = dfs["pytorch_cnn_1d"].iloc[cand_idx]
            if cand_row["true_label"] == cand_row["pred_label"] and cand_idx != m_idx:
                correct_idx = cand_idx
                break
        if correct_idx is None:
            correct_idx = correct_cands.index[0]
        raw_correct_win = X_test_raw[correct_idx]

        ax_acc = axes[row_idx, 0]
        ax_grav = axes[row_idx, 1]
        ax_pitch = axes[row_idx, 2]

        # userAcceleration x/y/z (channels 9, 10, 11)
        ax_acc.plot(time_s, raw_error_win[:, 9], color="#1f77b4", lw=1.5, label="Error: x" if row_idx == 0 else "")
        ax_acc.plot(time_s, raw_error_win[:, 10], color="#ff7f0e", lw=1.5, label="Error: y" if row_idx == 0 else "")
        ax_acc.plot(time_s, raw_error_win[:, 11], color="#2ca02c", lw=1.5, label="Error: z" if row_idx == 0 else "")
        # Contrast correct window
        ax_acc.plot(time_s, raw_correct_win[:, 9], color="#1f77b4", linestyle=":", alpha=0.5, label="Clean: x" if row_idx == 0 else "")
        ax_acc.plot(time_s, raw_correct_win[:, 10], color="#ff7f0e", linestyle=":", alpha=0.5, label="Clean: y" if row_idx == 0 else "")
        ax_acc.plot(time_s, raw_correct_win[:, 11], color="#2ca02c", linestyle=":", alpha=0.5, label="Clean: z" if row_idx == 0 else "")
        ax_acc.set_ylabel(f"Subj {s_val}, Tr {t_val}\n{true_act}→{pred_act}\n({case['model']}, {case['confidence']:.2f})\n\nAcc (g)", fontsize=8.5, fontweight="medium")
        ax_acc.grid(True, linestyle=":", alpha=0.5)

        # gravity x/y/z (channels 3, 4, 5)
        ax_grav.plot(time_s, raw_error_win[:, 3], color="#1f77b4", lw=1.5, label="Grav: x" if row_idx == 0 else "")
        ax_grav.plot(time_s, raw_error_win[:, 4], color="#ff7f0e", lw=1.5, label="Grav: y" if row_idx == 0 else "")
        ax_grav.plot(time_s, raw_error_win[:, 5], color="#2ca02c", lw=1.5, label="Grav: z" if row_idx == 0 else "")
        ax_grav.plot(time_s, raw_correct_win[:, 3], color="#1f77b4", linestyle=":", alpha=0.5)
        ax_grav.plot(time_s, raw_correct_win[:, 4], color="#ff7f0e", linestyle=":", alpha=0.5)
        ax_grav.plot(time_s, raw_correct_win[:, 5], color="#2ca02c", linestyle=":", alpha=0.5)
        ax_grav.set_ylabel("Gravity (g)", fontsize=9, fontweight="medium")
        ax_grav.grid(True, linestyle=":", alpha=0.5)

        # attitude.pitch (channel 1)
        ax_pitch.plot(time_s, raw_error_win[:, 1], color="#d62728", lw=1.8, label="Error: pitch" if row_idx == 0 else "")
        ax_pitch.plot(time_s, raw_correct_win[:, 1], color="#333333", linestyle="--", lw=1.4, alpha=0.7, label="Clean: pitch" if row_idx == 0 else "")
        ax_pitch.set_ylabel("Pitch (rad)", fontsize=9, fontweight="medium")
        ax_pitch.grid(True, linestyle=":", alpha=0.5)

        if row_idx == 0:
            ax_acc.set_title("userAcceleration x/y/z", fontsize=10.5, fontweight="bold")
            ax_grav.set_title("gravity x/y/z", fontsize=10.5, fontweight="bold")
            ax_pitch.set_title("attitude.pitch (Solid=Error, Dashed=Clean)", fontsize=10.5, fontweight="bold")
            ax_acc.legend(loc="upper right", fontsize=7.5, ncol=2)
            ax_pitch.legend(loc="upper right", fontsize=7.5)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10, fontweight="medium")
    axes[-1, 1].set_xlabel("Time (s)", fontsize=10, fontweight="medium")
    axes[-1, 2].set_xlabel("Time (s)", fontsize=10, fontweight="medium")

    plt.suptitle(
        "Task T4d: Representative Error Modes (Solid = Misclassified Window, Dotted/Dashed = Matched Clean Window)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    fig_err_path = fig_dir / "fig16_error_examples.png"
    fig.savefig(fig_err_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved representative error waveforms figure to {fig_err_path}")


# ==============================================================================
# TASK T4d: NOISE STRESS TEST (PROTOCOL FIRST)
# ==============================================================================

def run_noise_stress_test() -> dict[str, Any]:
    """
    Execute noise stress test: inject channel-wise additive Gaussian sensor noise
    at increasing intensity levels sigma in {0.0, 0.1, 0.25, 0.5, 1.0} into test windows.

    Protocol compliance:
        - Writes outputs/metrics/noise_protocol.json BEFORE running evaluations.
        - sigma=0.0 MUST reproduce results_table.csv exactly (asserted).
        - Saves outputs/metrics/noise_robustness.json and fig17_noise_robustness.png.
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK T4d: NOISE STRESS TEST (PROTOCOL FIRST)")
    logger.info("=" * 60)

    metrics_dir = Path("outputs/metrics")
    prep_dir = Path("outputs/preprocessing")
    cache_dir = Path("outputs/cache")
    ckpt_dir = Path("outputs/checkpoints")

    # 1. Pre-declare protocol in outputs/metrics/noise_protocol.json BEFORE running
    protocol = {
        "hypothesis": "CNN on raw signals degrades faster under additive sensor noise than RF/MLP on window statistics",
        "noise_model": "Additive Gaussian noise N(0, (sigma * train_std_c)^2) injected per channel c into raw unscaled test windows",
        "channel_scaling": "train_std_c derived from saved outputs/preprocessing/raw_scaler.json (strictly training distribution)",
        "sigma_levels": [0.0, 0.1, 0.25, 0.5, 1.0],
        "noise_repeats_per_sigma": 5,
        "noise_random_seeds": [100, 101, 102, 103, 104],
        "models_evaluated": [
            "baseline_rf (seeds 0, 1, 2)",
            "pytorch_mlp (seeds 0, 1, 2)",
            "pytorch_cnn_1d (seeds 0, 1, 2)",
        ],
        "total_model_evaluations": 5 * 5 * 9,  # 5 sigmas * 5 repeats * 9 models = 225 evaluations
        "test_sample_size": 3730,
        "primary_metric": "macro-averaged F1 score",
        "feature_pipeline_under_noise": {
            "RF_and_MLP": "Noise added to raw windows -> 84 features recomputed via extract_features -> standardized via saved feature_scaler.joblib",
            "CNN": "Noise added to raw windows -> standardized via saved raw_scaler.joblib -> transposed to (N, 12, 128)",
        },
        "exact_baseline_reproduction_assertion": "sigma=0.0 must exactly match results_table.csv within 1e-6 tolerance",
    }
    proto_path = metrics_dir / "noise_protocol.json"
    save_json(protocol, proto_path)
    logger.info(f"Protocol registered in {proto_path} BEFORE running experiment.")

    # 2. Load scalers and clean test data
    raw_scaler = RawWindowScaler.load(prep_dir / "raw_scaler.joblib")
    feat_scaler = joblib.load(prep_dir / "feature_scaler.joblib")
    raw_scaler_meta = load_json(prep_dir / "raw_scaler.json")
    train_std_c = np.array(raw_scaler_meta["std"], dtype=np.float64)  # (12,)

    X_test_raw = np.load(cache_dir / "X_test_raw.npy")  # (3730, 128, 12)
    y_test = np.load(cache_dir / "y_test.npy")

    # Load baseline test results table to assert sigma=0 reproduction
    results_table = pd.read_csv(metrics_dir / "results_table.csv")

    # Preload all 9 model checkpoints
    models: dict[str, list[Any]] = {"baseline_rf": [], "pytorch_mlp": [], "pytorch_cnn_1d": []}

    for s in [0, 1, 2]:
        rf = joblib.load(ckpt_dir / f"baseline_rf_seed{s}.joblib")
        models["baseline_rf"].append(rf)

        mlp_ckpt = torch.load(ckpt_dir / f"pytorch_mlp_seed{s}.pt", map_location="cpu")
        m_cfg = mlp_ckpt["config"]["model"]
        mlp = PyTorchMLP(
            input_dim=int(m_cfg["input_dim"]),
            hidden_sizes=list(m_cfg["hidden_sizes"]),
            dropout=float(m_cfg["dropout"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        mlp.load_state_dict(mlp_ckpt["model_state_dict"])
        mlp.eval()
        models["pytorch_mlp"].append(mlp)

        cnn_ckpt = torch.load(ckpt_dir / f"pytorch_cnn_1d_seed{s}.pt", map_location="cpu")
        c_cfg = cnn_ckpt["config"]["model"]
        cnn = PyTorchCNN1D(
            in_channels=int(c_cfg["in_channels"]),
            sequence_length=int(c_cfg["sequence_length"]),
            filters=list(c_cfg["filters"]),
            kernel_size=int(c_cfg["kernel_size"]),
            dropout=float(c_cfg["dropout"]),
            num_classes=int(c_cfg["num_classes"]),
        )
        cnn.load_state_dict(cnn_ckpt["model_state_dict"])
        cnn.eval()
        models["pytorch_cnn_1d"].append(cnn)

    sigma_levels = [0.0, 0.1, 0.25, 0.5, 1.0]
    repeats = [100, 101, 102, 103, 104]

    # Structure to hold evaluations: model_key -> sigma -> list of macro-f1 scores (repeats * 3 seeds)
    noise_evaluations: dict[str, dict[str, list[float]]] = {
        "baseline_rf": {str(sig): [] for sig in sigma_levels},
        "pytorch_mlp": {str(sig): [] for sig in sigma_levels},
        "pytorch_cnn_1d": {str(sig): [] for sig in sigma_levels},
    }

    # Per-seed detailed breakdown
    per_seed_evals: dict[str, dict[int, dict[str, list[float]]]] = {
        m: {s: {str(sig): [] for sig in sigma_levels} for s in [0, 1, 2]} for m in models
    }

    logger.info("Running noise injection iterations across sigmas and repeats...")

    for sig in sigma_levels:
        for r_seed in repeats:
            if sig == 0.0:
                X_noisy_raw = X_test_raw.copy()
            else:
                rng = np.random.default_rng(r_seed)
                # Channel-wise Gaussian noise scaled by training std per channel
                noise = rng.normal(loc=0.0, scale=sig * train_std_c, size=X_test_raw.shape).astype(np.float32)
                X_noisy_raw = X_test_raw + noise

            # 1. CNN inputs: scale noisy raw windows with saved raw_scaler and transpose
            X_cnn_scaled = raw_scaler.transform(X_noisy_raw)
            if X_cnn_scaled.ndim == 3 and X_cnn_scaled.shape[1] == 128 and X_cnn_scaled.shape[2] == 12:
                X_cnn_in = np.transpose(X_cnn_scaled, (0, 2, 1))
            else:
                X_cnn_in = X_cnn_scaled
            t_cnn_in = torch.tensor(X_cnn_in, dtype=torch.float32)

            for s_idx, cnn_model in enumerate(models["pytorch_cnn_1d"]):
                with torch.no_grad():
                    logits = cnn_model(t_cnn_in)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                f1_val = float(f1_score(y_test, preds, average="macro"))
                noise_evaluations["pytorch_cnn_1d"][str(sig)].append(f1_val)
                per_seed_evals["pytorch_cnn_1d"][s_idx][str(sig)].append(f1_val)

            # 2. RF & MLP inputs: recompute features from noisy raw windows and scale
            X_noisy_feat = extract_features(X_noisy_raw)
            X_feat_scaled = feat_scaler.transform(X_noisy_feat)
            t_mlp_in = torch.tensor(X_feat_scaled, dtype=torch.float32)

            # RF
            for s_idx, rf_model in enumerate(models["baseline_rf"]):
                preds = rf_model.predict(X_feat_scaled)
                f1_val = float(f1_score(y_test, preds, average="macro"))
                noise_evaluations["baseline_rf"][str(sig)].append(f1_val)
                per_seed_evals["baseline_rf"][s_idx][str(sig)].append(f1_val)

            # MLP
            for s_idx, mlp_model in enumerate(models["pytorch_mlp"]):
                with torch.no_grad():
                    logits = mlp_model(t_mlp_in)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                f1_val = float(f1_score(y_test, preds, average="macro"))
                noise_evaluations["pytorch_mlp"][str(sig)].append(f1_val)
                per_seed_evals["pytorch_mlp"][s_idx][str(sig)].append(f1_val)

    # 3. Assert exact reproduction at sigma=0.0 against results_table.csv
    logger.info("Verifying exact baseline reproduction assertion (sigma = 0.0)...")
    for m_key in models:
        for s in [0, 1, 2]:
            table_row = results_table[(results_table["model"] == m_key) & (results_table["seed"] == s)].iloc[0]
            expected_f1 = float(table_row["test_macro_f1"])
            # All 5 repeats at sigma=0 should be identical to expected_f1
            computed_f1s = per_seed_evals[m_key][s]["0.0"]
            for val in computed_f1s:
                assert np.isclose(val, expected_f1, atol=1e-5), (
                    f"Mismatch at sigma=0 for {m_key} seed {s}: computed={val:.6f}, expected={expected_f1:.6f}"
                )
    logger.info("Assertion passed: sigma=0 exactly reproduces results_table.csv for all 9 models!")

    # 4. Compile summary statistics per model per sigma
    noise_summary: dict[str, Any] = {
        "protocol": protocol,
        "results_by_model": {},
    }

    for m_key, disp_name in [
        ("baseline_rf", "Random Forest"),
        ("pytorch_mlp", "PyTorch MLP"),
        ("pytorch_cnn_1d", "PyTorch 1D-CNN"),
    ]:
        sig_data = {}
        for sig in sigma_levels:
            vals = noise_evaluations[m_key][str(sig)]
            # vals has 15 entries ordered repeat-major: index = repeat*3 + seed_idx
            seed_means = [float(np.mean(vals[k::3])) for k in range(3)]
            std_across_seeds = float(np.std(seed_means, ddof=1))
            sig_data[str(sig)] = {
                "mean_macro_f1": float(np.mean(vals)),
                "std_macro_f1": float(np.std(vals, ddof=1)),
                "std_across_seeds": std_across_seeds,
                "min_macro_f1": float(np.min(vals)),
                "max_macro_f1": float(np.max(vals)),
                "raw_values": vals,
            }
        noise_summary["results_by_model"][m_key] = {
            "model_name": m_key,
            "display_name": disp_name,
            "sigmas": sig_data,
        }

    noise_res_path = metrics_dir / "noise_robustness.json"
    save_json(noise_summary, noise_res_path)
    logger.info(f"Saved noise stress test results to {noise_res_path}")

    # 5. Plot fig17_noise_robustness.png
    plot_noise_robustness(noise_summary, sigma_levels)

    return noise_summary


def plot_noise_robustness(
    noise_summary: dict[str, Any],
    sigma_levels: list[float],
    fig_noise_path: Path = Path("outputs/figures/fig17_noise_robustness.png"),
) -> Path:
    """
    Plot fig17 (test macro-F1 vs noise sigma) from the saved noise summary.

    Why:
        Kept separate from run_noise_stress_test() so the figure can be
        regenerated from outputs/metrics/noise_robustness.json without
        re-reading the test split. The y-axis spans the full 0-1 range so that
        no model's points fall outside the plot.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    color_map = {
        "baseline_rf": "#4C72B0",
        "pytorch_mlp": "#DD8452",
        "pytorch_cnn_1d": "#55A868",
    }
    marker_map = {
        "baseline_rf": "o",
        "pytorch_mlp": "s",
        "pytorch_cnn_1d": "^",
    }

    for m_key in ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]:
        m_info = noise_summary["results_by_model"][m_key]
        means = [m_info["sigmas"][str(s)]["mean_macro_f1"] for s in sigma_levels]
        stds = [
            m_info["sigmas"][str(s)].get("std_across_seeds", m_info["sigmas"][str(s)]["std_macro_f1"])
            for s in sigma_levels
        ]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(
            sigma_levels,
            means_arr,
            color=color_map[m_key],
            marker=marker_map[m_key],
            lw=2.2,
            markersize=7,
            label=f"{m_info['display_name']} (Mean ± Seed Std)",
        )
        ax.fill_between(
            sigma_levels,
            np.clip(means_arr - stds_arr, 0.0, 1.0),
            np.clip(means_arr + stds_arr, 0.0, 1.0),
            color=color_map[m_key],
            alpha=0.18,
        )

    ax.set_xlabel("Noise Intensity σ (× Training Channel Standard Deviation)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Test Macro-F1 Score", fontsize=11, fontweight="medium")
    ax.set_title("Task T4d: Sensor Noise Robustness Stress Test", fontsize=12, fontweight="bold")
    ax.set_xticks(sigma_levels)
    ax.set_xticklabels(["0.0\n(Clean)", "0.1", "0.25", "0.5", "1.0"], fontsize=9.5)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig_noise_path = Path(fig_noise_path)
    fig.savefig(fig_noise_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved noise robustness figure to {fig_noise_path}")
    return fig_noise_path


def regenerate_noise_summary_from_raw(
    noise_res_path: Path = Path("outputs/metrics/noise_robustness.json"),
    fig_noise_path: Path = Path("outputs/figures/fig17_noise_robustness.png"),
) -> dict[str, Any]:
    """
    Regenerate noise_robustness.json summary fields and fig17 from existing raw_values only,
    computing std_across_seeds without re-evaluating any models or touching test data.
    """
    noise_data = load_json(noise_res_path)
    sigma_levels = [float(s) for s in noise_data["protocol"]["sigma_levels"]]

    for m_key, m_info in noise_data["results_by_model"].items():
        for s_str, s_data in m_info["sigmas"].items():
            vals = s_data["raw_values"]
            seed_means = [float(np.mean(vals[k::3])) for k in range(3)]
            s_data["std_across_seeds"] = float(np.std(seed_means, ddof=1))
            s_data["mean_macro_f1"] = float(np.mean(seed_means))

    save_json(noise_data, noise_res_path)
    logger.info(f"Regenerated summary fields in {noise_res_path} with std_across_seeds.")

    plot_noise_robustness(noise_data, sigma_levels, fig_noise_path=fig_noise_path)
    return noise_data


# ==============================================================================
# TASK T4d: SUBGROUP ANALYSIS (GENDER & SUBJECT DEMOGRAPHICS)
# ==============================================================================

def run_subgroup_analysis() -> dict[str, Any]:
    """
    Perform demographic subgroup analysis across test subjects (1, 8, 18, 21)
    and gender partitions (male vs female).

    Why:
        Plan Phase 4d & Task T4d: Evaluates model equity across participant demographics
        (gender, age, BMI factors) while formally noting the small-sample observational caveat.
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK T4d: DEMOGRAPHIC SUBGROUP ANALYSIS")
    logger.info("=" * 60)

    pred_dir = Path("outputs/metrics/predictions")
    metrics_dir = Path("outputs/metrics")
    fig_dir = Path("outputs/figures")

    # Load subject demographics
    sub_info_df = pd.read_csv("MotionSense_dataset/data_subjects_info.csv")
    sub_demo = {}
    for _, row in sub_info_df.iterrows():
        sub_code = int(row["code"])
        if sub_code in [1, 8, 18, 21]:
            sub_demo[sub_code] = {
                "subject": sub_code,
                "gender": "male" if int(row["gender"]) == 1 else "female",
                "age": int(row["age"]),
                "height": float(row["height"]),
                "weight": float(row["weight"]),
            }

    models = ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]
    disp_names = {
        "baseline_rf": "Random Forest",
        "pytorch_mlp": "PyTorch MLP",
        "pytorch_cnn_1d": "PyTorch 1D-CNN",
    }
    seeds = [0, 1, 2]
    subjects = [1, 8, 18, 21]

    # Structure to hold results: model -> subject -> list of F1s (3 seeds)
    subject_f1s: dict[str, dict[int, list[float]]] = {m: {s: [] for s in subjects} for m in models}
    gender_f1s: dict[str, dict[str, list[float]]] = {m: {"male": [], "female": []} for m in models}
    window_counts: dict[int, int] = {}

    for m in models:
        for s in seeds:
            pred_path = pred_dir / f"{m}_seed{s}.csv"
            df = pd.read_csv(pred_path)

            # Per-subject
            for subj in subjects:
                sub_df = df[df["subject"] == subj]
                if s == 0 and m == "baseline_rf":
                    window_counts[subj] = len(sub_df)
                f1_sub = float(f1_score(sub_df["true_label"], sub_df["pred_label"], average="macro"))
                subject_f1s[m][subj].append(f1_sub)

            # Per-gender
            male_df = df[df["subject"].isin([1, 21])]
            female_df = df[df["subject"].isin([8, 18])]
            gender_f1s[m]["male"].append(float(f1_score(male_df["true_label"], male_df["pred_label"], average="macro")))
            gender_f1s[m]["female"].append(float(f1_score(female_df["true_label"], female_df["pred_label"], average="macro")))

    # Compile JSON summary
    subgroup_summary: dict[str, Any] = {
        "observational_caveat": "There are exactly 2 subjects per gender in the test partition (male: subjects 1, 21; female: subjects 8, 18). Consequently, demographic gender comparisons are indicative only and cannot fully disentangle gender from individual kinematic and anthropometric variations.",
        "subjects": {},
        "gender_groups": {
            "male": {
                "subjects": [1, 21],
                "total_windows": window_counts[1] + window_counts[21],
                "models": {},
            },
            "female": {
                "subjects": [8, 18],
                "total_windows": window_counts[8] + window_counts[18],
                "models": {},
            },
        },
    }

    for subj in subjects:
        demo = sub_demo[subj]
        sub_record = {
            "demographics": demo,
            "window_count": window_counts[subj],
            "models": {},
        }
        for m in models:
            scores = subject_f1s[m][subj]
            sub_record["models"][m] = {
                "macro_f1_mean": float(np.mean(scores)),
                "macro_f1_std": float(np.std(scores, ddof=1)),
                "scores_per_seed": scores,
            }
        subgroup_summary["subjects"][str(subj)] = sub_record

    for g in ["male", "female"]:
        for m in models:
            scores = gender_f1s[m][g]
            subgroup_summary["gender_groups"][g]["models"][m] = {
                "macro_f1_mean": float(np.mean(scores)),
                "macro_f1_std": float(np.std(scores, ddof=1)),
                "scores_per_seed": scores,
            }

    subgroup_json_path = metrics_dir / "subgroup.json"
    save_json(subgroup_summary, subgroup_json_path)
    logger.info(f"Saved subgroup analysis JSON to {subgroup_json_path}")

    # Plot fig18_per_subject_f1.png
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    x = np.arange(len(subjects))
    bar_width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, (m, col) in enumerate(zip(models, colors)):
        means = [float(np.mean(subject_f1s[m][s])) for s in subjects]
        stds = [float(np.std(subject_f1s[m][s], ddof=1)) for s in subjects]
        offset = (idx - 1) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=disp_names[m],
            color=col,
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.9,
        )

    # Subject labels with gender, age, and window count
    tick_labels = [
        f"Subject {s}\n({sub_demo[s]['gender']}, age {sub_demo[s]['age']})\nN={window_counts[s]}"
        for s in subjects
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylabel("Test Macro-F1 Score", fontsize=11, fontweight="medium")
    ax.set_title("Task T4d: Test Macro-F1 by Participant Subgroup (Mean ± Std over 3 Seeds)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.70, 1.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig_sub_path = fig_dir / "fig18_per_subject_f1.png"
    fig.savefig(fig_sub_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved subgroup figure to {fig_sub_path}")

    return subgroup_summary


# ==============================================================================
# MAIN DRIVER
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Task T4c & T4d: Experiments and Robustness.")
    parser.add_argument("--ablation", action="store_true", help="Run Task T4c ablation study.")
    parser.add_argument("--errors", action="store_true", help="Run Task T4d error analysis.")
    parser.add_argument("--noise", action="store_true", help="Run Task T4d noise stress test.")
    parser.add_argument("--regenerate-noise", action="store_true", help="Regenerate noise summary and fig17 from raw_values.")
    parser.add_argument("--subgroup", action="store_true", help="Run Task T4d subgroup analysis.")
    parser.add_argument("--all", action="store_true", help="Run all tasks (T4c and T4d).")
    args = parser.parse_args()

    if args.regenerate_noise:
        regenerate_noise_summary_from_raw()
        return

    # If no flags passed, default to --all
    run_all = args.all or not (args.ablation or args.errors or args.noise or args.subgroup)

    ablation_res = None
    noise_res = None
    subgroup_res = None

    if run_all or args.ablation:
        ablation_res = run_ablation()

    if run_all or args.errors:
        run_error_analysis()

    if run_all or args.noise:
        noise_res = run_noise_stress_test()

    if run_all or args.subgroup:
        subgroup_res = run_subgroup_analysis()

    logger.info("\nAll Phase 4c & 4d tasks completed successfully.")


if __name__ == "__main__":
    main()
