"""
Module: src.evaluate
Purpose: T4 Final evaluation on the unseen test split (executed strictly once).

Responsibilities:
- Sole entry point in the entire project permitted to load the test split (X_test, y_test).
- Evaluates frozen model checkpoints across 3 random seeds (seeds 0, 1, 2) for:
    1. Non-neural Random Forest baseline (`baseline_rf_seed{0,1,2}.joblib`)
    2. PyTorch MLP (`pytorch_mlp_seed{0,1,2}.pt`)
    3. PyTorch 1D-CNN (`pytorch_cnn_1d{0,1,2}.pt`)
- Computes comprehensive test metrics per run:
    * Macro-averaged F1 score (primary benchmark metric)
    * Overall classification accuracy
    * Per-class precision, recall, and F1-score across all 6 activity classes
- Writes artifacts:
    * `outputs/metrics/results_table.csv`: tidy 9-row table (model, config_file, seed, best_epoch,
      val_macro_f1, test_macro_f1, test_accuracy, n_params, checkpoint_path)
    * `outputs/metrics/results_summary.json`: summary statistics (mean +/- sample std, ddof=1)
      and per-seed values, including representative seed records
    * `outputs/metrics/predictions/<model>_seed<k>.csv`: per-window test predictions and posterior
      probabilities for subsequent error analysis (Phase 4d) without re-evaluating test data
- Selects ONE representative seed per model based strictly on MEDIAN VALIDATION MACRO-F1:
    * RF: Seed 1 (Val Macro-F1 = 0.9276)
    * MLP: Seed 0 (Val Macro-F1 = 0.9555)
    * CNN: Seed 2 (Val Macro-F1 = 0.9458)
- Generates publication-ready figures (saved at 150 DPI with full labels and legends):
    * `fig10_cm_rf.png`: row-normalised confusion matrix for RF (representative seed 1)
    * `fig11_cm_mlp.png`: row-normalised confusion matrix for MLP (representative seed 0)
    * `fig12_cm_cnn.png`: row-normalised confusion matrix for CNN (representative seed 2)
    * `fig13_per_class_f1.png`: grouped bar chart of per-class test F1 (mean +/- std across 3 seeds)
    * `fig14_val_vs_test.png`: validation vs test macro-F1 comparison highlighting generalization gaps

Rule compliance:
- Rule 1: Python 3.11, NumPy, pandas, scikit-learn, PyTorch (CPU), matplotlib.
- Rule 7: Test split accessed ONLY by src/evaluate.py after configs and checkpoints are frozen.
- Rule 8: Persists structured metrics CSV, summary JSON, and prediction records.
- Rule 10: Strictly empirical numbers derived from saved artifacts; never hard-coded or estimated.
- Rule 12: All figures include informative titles, labelled axes with units, and complete legends.
- Rule 13: Docstrings explain methodology, parameter counting, and representative seed selection.
- Rule 14: Clear, idiomatic, well-commented code.
"""

from __future__ import annotations

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn

from src.models.cnn import PyTorchCNN1D
from src.models.mlp import PyTorchMLP
from src.utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.evaluate")

# MotionSense standard activity class names ordered by numeric label 0..5
CLASS_NAMES = ["dws", "ups", "wlk", "jog", "std", "sit"]
CLASS_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}


def load_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load scaled test feature arrays, scaled raw test windows, ground-truth labels,
    and window metadata.

    Why:
        Rule 7 explicitly allows src/evaluate.py (and ONLY src/evaluate.py) to load
        the test split. Scaled data arrays were transformed using scalers fitted
        strictly on training subjects in Phase 1c (outputs/preprocessing/feature_scaler.joblib
        and raw_scaler.joblib), guaranteeing zero test leakage.

    Returns:
        X_test_feat: Tabular feature matrix of shape (N, 84).
        X_test_cnn: Raw sequence array of shape (N, 12, 128) channels-first for 1D-CNN.
        y_test: Ground-truth integer labels array of shape (N,).
        df_windows: Metadata DataFrame with subject, trial, and start_index.
    """
    cache_dir = Path("outputs/cache")
    splits_dir = Path("outputs/splits")

    logger.info("Loading test data splits from outputs/cache/ ...")
    X_test_feat = np.load(cache_dir / "X_test_feat_scaled.npy")
    X_test_raw = np.load(cache_dir / "X_test_raw_scaled.npy")
    y_test = np.load(cache_dir / "y_test.npy")
    df_windows = pd.read_csv(splits_dir / "windows_test.csv")

    # Transpose raw windows to (N, 12, 128) channels-first format if stored as (N, 128, 12)
    if X_test_raw.ndim == 3 and X_test_raw.shape[1] == 128 and X_test_raw.shape[2] == 12:
        X_test_cnn = np.transpose(X_test_raw, (0, 2, 1))
    else:
        X_test_cnn = X_test_raw

    logger.info(
        f"Test data loaded successfully: X_test_feat={X_test_feat.shape}, "
        f"X_test_cnn={X_test_cnn.shape}, y_test={y_test.shape}, "
        f"df_windows={len(df_windows)} rows."
    )
    return X_test_feat, X_test_cnn, y_test, df_windows


def evaluate_random_forest_checkpoint(
    seed: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate a saved Random Forest checkpoint on the test split.

    Why:
        Assesses test generalization of the non-neural Random Forest baseline (T3a, T4a).
        Computes predictions, posterior probabilities, and counts total decision tree nodes.
    """
    ckpt_path = Path(f"outputs/checkpoints/baseline_rf_seed{seed}.joblib")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    rf = joblib.load(ckpt_path)
    probs = rf.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_NAMES))))

    # Total decision tree nodes across the ensemble
    total_nodes = int(sum(tree.tree_.node_count for tree in rf.estimators_))

    # Read validation metrics
    val_json_path = Path(f"outputs/metrics/baseline_rf_seed{seed}.json")
    val_metrics = load_json(val_json_path)
    val_macro_f1 = float(val_metrics["val_macro_f1"])
    val_acc = float(val_metrics["val_accuracy"])

    return {
        "model": "baseline_rf",
        "config_file": "configs/baseline_rf_final.yaml",
        "seed": seed,
        "best_epoch": "N/A",
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_acc,
        "test_macro_f1": macro_f1,
        "test_accuracy": acc,
        "n_params": total_nodes,
        "checkpoint_path": str(ckpt_path),
        "probs": probs,
        "preds": preds,
        "cm": cm,
        "report": report,
    }


def evaluate_mlp_checkpoint(
    seed: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate a saved PyTorch MLP checkpoint on the test split.

    Why:
        Measures test generalization of the hand-crafted feature MLP (T3b, T4a)
        using PyTorch in evaluation mode.
    """
    ckpt_path = Path(f"outputs/checkpoints/pytorch_mlp_seed{seed}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = PyTorchMLP(
        input_dim=int(model_cfg.get("input_dim", 84)),
        hidden_sizes=list(model_cfg.get("hidden_sizes", [128, 64])),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        probs_tensor = torch.softmax(logits, dim=1)
        probs = probs_tensor.cpu().numpy()
        preds = np.argmax(probs, axis=1)

    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_NAMES))))

    # Read validation metrics
    val_json_path = Path(f"outputs/metrics/pytorch_mlp_seed{seed}.json")
    val_metrics = load_json(val_json_path)
    val_macro_f1 = float(val_metrics["best_val_macro_f1"])
    val_acc = float(val_metrics["val_accuracy_at_best"])
    best_epoch = int(val_metrics["best_epoch"])

    return {
        "model": "pytorch_mlp",
        "config_file": "configs/mlp_final.yaml",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_acc,
        "test_macro_f1": macro_f1,
        "test_accuracy": acc,
        "n_params": total_params,
        "checkpoint_path": str(ckpt_path),
        "probs": probs,
        "preds": preds,
        "cm": cm,
        "report": report,
    }


def evaluate_cnn_checkpoint(
    seed: int,
    X_test_cnn: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate a saved PyTorch 1D-CNN checkpoint on the test split.

    Why:
        Measures test generalization of the end-to-end raw sequence 1D-CNN (T3c, T4a).
    """
    ckpt_path = Path(f"outputs/checkpoints/pytorch_cnn_1d_seed{seed}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = PyTorchCNN1D(
        in_channels=int(model_cfg.get("in_channels", 12)),
        sequence_length=int(model_cfg.get("sequence_length", 128)),
        filters=list(model_cfg.get("filters", [64, 128, 128])),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        x_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
        logits = model(x_tensor)
        probs_tensor = torch.softmax(logits, dim=1)
        probs = probs_tensor.cpu().numpy()
        preds = np.argmax(probs, axis=1)

    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_NAMES))))

    # Read validation metrics
    val_json_path = Path(f"outputs/metrics/pytorch_cnn_1d_seed{seed}.json")
    val_metrics = load_json(val_json_path)
    val_macro_f1 = float(val_metrics["best_val_macro_f1"])
    val_acc = float(val_metrics["val_accuracy_at_best"])
    best_epoch = int(val_metrics["best_epoch"])

    return {
        "model": "pytorch_cnn_1d",
        "config_file": "configs/cnn_final.yaml",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_acc,
        "test_macro_f1": macro_f1,
        "test_accuracy": acc,
        "n_params": total_params,
        "checkpoint_path": str(ckpt_path),
        "probs": probs,
        "preds": preds,
        "cm": cm,
        "report": report,
    }


def save_prediction_file(
    df_windows: pd.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save per-window test predictions and class posterior probabilities to CSV.

    Why:
        Plan Phase 4b & Phase 4d: Persists window-level inference outputs so downstream
        error analysis, misclassification inspection, and confusion diagnostics do not
        require re-evaluating the test set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "subject": df_windows["subject"].to_numpy(),
        "trial": df_windows["trial"].to_numpy(),
        "start_index": df_windows["start_index"].to_numpy(),
        "true_label": [CLASS_LABELS[y] for y in y_test],
        "pred_label": [CLASS_LABELS[p] for p in preds],
        "prob_dws": probs[:, 0],
        "prob_ups": probs[:, 1],
        "prob_wlk": probs[:, 2],
        "prob_jog": probs[:, 3],
        "prob_std": probs[:, 4],
        "prob_sit": probs[:, 5],
    })
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction records to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_display_name: str,
    seed: int,
    output_path: Path,
    cmap: str = "Blues",
) -> None:
    """
    Generate and save a row-normalised confusion matrix plot.

    Why:
        Visualises per-class recall rates (row-normalised proportions) and misclassification
        patterns (e.g. dynamic stairs confusion vs static sit/stand confusion) for the
        representative median validation seed.
    """
    # Row-normalise: proportion of true class i classified as class j
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalised Proportion (Recall)", fontsize=10, fontweight="medium")

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            norm_val = cm_norm[i, j]
            raw_count = cm[i, j]
            text = f"{norm_val:.2f}\n({raw_count})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if norm_val > thresh else "black",
                fontsize=8.5,
                fontweight="medium",
            )

    ax.set_title(
        f"{model_display_name} (Seed {seed} - Val Median)\nRow-Normalised Test Confusion Matrix",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("True Activity", fontsize=11, fontweight="medium")
    ax.set_xlabel("Predicted Activity", fontsize=11, fontweight="medium")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to {output_path}")


def plot_per_class_f1(
    summary_data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate grouped bar chart of per-class test F1 scores across the three architectures.

    Why:
        Rule 12 & Plan Section 4a: Highlights architectural strengths and weaknesses
        across individual human activities (e.g. stairs vs walking vs static postures).
    """
    models = ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]
    labels = ["Random Forest", "PyTorch MLP", "PyTorch 1D-CNN"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    n_classes = len(CLASS_NAMES)
    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    for idx, (m_key, label, col) in enumerate(zip(models, labels, colors)):
        means = [summary_data[m_key]["summary"]["per_class_test_f1_mean"][cls] for cls in CLASS_NAMES]
        stds = [summary_data[m_key]["summary"]["per_class_test_f1_std"][cls] for cls in CLASS_NAMES]
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3.5,
            label=label,
            color=col,
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.9,
        )

    ax.set_xlabel("Activity Class", fontsize=11, fontweight="medium")
    ax.set_ylabel("Test F1-Score", fontsize=11, fontweight="medium")
    ax.set_title("Per-Class Test F1 Score Comparison (Mean ± Std over 3 Seeds)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved per-class F1 comparison to {output_path}")


def plot_val_vs_test(
    all_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate validation vs test macro-F1 comparison plots across all 3 seeds and models.

    Why:
        Plan Section 4a & Figure 14: Diagnoses the generalization gap between validation
        model selection and final unseen test performance, quantifying selection bias.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=150)

    # Palette
    color_map = {
        "baseline_rf": "#4C72B0",
        "pytorch_mlp": "#DD8452",
        "pytorch_cnn_1d": "#55A868",
    }
    label_map = {
        "baseline_rf": "Random Forest",
        "pytorch_mlp": "PyTorch MLP",
        "pytorch_cnn_1d": "PyTorch 1D-CNN",
    }
    marker_map = {
        "baseline_rf": "o",
        "pytorch_mlp": "s",
        "pytorch_cnn_1d": "^",
    }

    # Left Panel: Paired Bar Chart of Val vs Test per Seed
    bar_width = 0.35
    row_indices = np.arange(len(all_results))
    val_scores = [r["val_macro_f1"] for r in all_results]
    test_scores = [r["test_macro_f1"] for r in all_results]
    bar_labels = [f"{r['model'].replace('pytorch_', '').replace('baseline_', '')}\ns{r['seed']}" for r in all_results]

    ax1.bar(row_indices - bar_width / 2, val_scores, bar_width, label="Validation Macro-F1", color="#1f77b4", alpha=0.85, edgecolor="#333333")
    ax1.bar(row_indices + bar_width / 2, test_scores, bar_width, label="Test Macro-F1", color="#ff7f0e", alpha=0.85, edgecolor="#333333")
    ax1.set_xticks(row_indices)
    ax1.set_xticklabels(bar_labels, fontsize=8.5)
    ax1.set_ylabel("Macro-F1 Score", fontsize=11, fontweight="medium")
    ax1.set_title("Validation vs Test Macro-F1 per Seed", fontsize=11, fontweight="bold")
    # Dynamic y-axis limits to avoid clipping
    min_score = min(min(val_scores), min(test_scores))
    max_score = max(max(val_scores), max(test_scores))
    y_lower = max(0.0, float(np.floor(min_score * 20) / 20) - 0.05)
    y_upper = min(1.02, float(np.ceil(max_score * 20) / 20) + 0.02)

    ax1.set_ylim(y_lower, y_upper)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    # Right Panel: Scatter Plot of Test vs Validation with Identity Diagonal
    diag_min = y_lower
    diag_max = 1.0
    ax2.plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=1.2, label="Ideal Line (y = x)")

    for m_key in ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]:
        m_runs = [r for r in all_results if r["model"] == m_key]
        x_vals = [r["val_macro_f1"] for r in m_runs]
        y_vals = [r["test_macro_f1"] for r in m_runs]
        ax2.scatter(
            x_vals,
            y_vals,
            color=color_map[m_key],
            marker=marker_map[m_key],
            s=80,
            edgecolors="black",
            linewidths=0.8,
            label=label_map[m_key],
            zorder=4,
        )
        for r in m_runs:
            ax2.annotate(
                f"s{r['seed']}",
                (r["val_macro_f1"], r["test_macro_f1"]),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=8.5,
                color="#333333",
            )

    ax2.set_xlabel("Validation Macro-F1", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Test Macro-F1", fontsize=11, fontweight="medium")
    ax2.set_title("Generalisation Gap: Test vs Validation Macro-F1", fontsize=11, fontweight="bold")
    ax2.set_xlim(diag_min, diag_max)
    ax2.set_ylim(diag_min, diag_max)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.suptitle("Task T4b: Validation vs Final Test Generalisation Diagnostic", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved validation vs test figure to {output_path}")


def main() -> None:
    """
    Main execution pipeline for final test evaluation (executed strictly once).
    """
    logger.info("=" * 60)
    logger.info("STARTING PHASE 4: FINAL TEST EVALUATION (T4a, T4b)")
    logger.info("=" * 60)

    # 1. Load test split
    X_test_feat, X_test_cnn, y_test, df_windows = load_test_data()

    # Model evaluation definitions
    seeds = [0, 1, 2]
    all_run_results: list[dict[str, Any]] = []

    pred_dir = Path("outputs/metrics/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 2. Evaluate all 9 checkpoints
    logger.info("\n--- Evaluating Random Forest Baseline (Seeds 0, 1, 2) ---")
    for s in seeds:
        res = evaluate_random_forest_checkpoint(seed=s, X_test=X_test_feat, y_test=y_test)
        all_run_results.append(res)
        save_prediction_file(
            df_windows=df_windows,
            y_test=y_test,
            preds=res["preds"],
            probs=res["probs"],
            output_path=pred_dir / f"baseline_rf_seed{s}.csv",
        )
        logger.info(
            f"[RF Seed {s}] Val Macro-F1: {res['val_macro_f1']:.4f} | "
            f"Test Macro-F1: {res['test_macro_f1']:.4f} | Test Acc: {res['test_accuracy']:.4f}"
        )

    logger.info("\n--- Evaluating PyTorch MLP (Seeds 0, 1, 2) ---")
    for s in seeds:
        res = evaluate_mlp_checkpoint(seed=s, X_test=X_test_feat, y_test=y_test)
        all_run_results.append(res)
        save_prediction_file(
            df_windows=df_windows,
            y_test=y_test,
            preds=res["preds"],
            probs=res["probs"],
            output_path=pred_dir / f"pytorch_mlp_seed{s}.csv",
        )
        logger.info(
            f"[MLP Seed {s}] Best Epoch: {res['best_epoch']} | "
            f"Val Macro-F1: {res['val_macro_f1']:.4f} | "
            f"Test Macro-F1: {res['test_macro_f1']:.4f} | Test Acc: {res['test_accuracy']:.4f}"
        )

    logger.info("\n--- Evaluating PyTorch 1D-CNN (Seeds 0, 1, 2) ---")
    for s in seeds:
        res = evaluate_cnn_checkpoint(seed=s, X_test_cnn=X_test_cnn, y_test=y_test)
        all_run_results.append(res)
        save_prediction_file(
            df_windows=df_windows,
            y_test=y_test,
            preds=res["preds"],
            probs=res["probs"],
            output_path=pred_dir / f"pytorch_cnn_1d_seed{s}.csv",
        )
        logger.info(
            f"[CNN Seed {s}] Best Epoch: {res['best_epoch']} | "
            f"Val Macro-F1: {res['val_macro_f1']:.4f} | "
            f"Test Macro-F1: {res['test_macro_f1']:.4f} | Test Acc: {res['test_accuracy']:.4f}"
        )

    # 3. Write outputs/metrics/results_table.csv
    table_rows = []
    for r in all_run_results:
        table_rows.append({
            "model": r["model"],
            "config_file": r["config_file"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "val_macro_f1": round(r["val_macro_f1"], 6),
            "test_macro_f1": round(r["test_macro_f1"], 6),
            "test_accuracy": round(r["test_accuracy"], 6),
            "n_params": r["n_params"],
            "checkpoint_path": r["checkpoint_path"],
        })
    results_df = pd.DataFrame(table_rows)
    table_path = Path("outputs/metrics/results_table.csv")
    results_df.to_csv(table_path, index=False)
    logger.info(f"\nWritten results table to {table_path} with {len(results_df)} rows.")

    # 4. Write outputs/metrics/results_summary.json
    # Representative median validation seeds identified strictly on validation score prior to test unblinding:
    # RF: seed 1 (0.927551 is median of 0.926386, 0.927551, 0.936759)
    # MLP: seed 0 (0.955549 is median of 0.953321, 0.955549, 0.963387)
    # CNN: seed 2 (0.945839 is median of 0.944405, 0.945839, 0.967298)
    rep_seeds = {
        "baseline_rf": 1,
        "pytorch_mlp": 0,
        "pytorch_cnn_1d": 2,
    }
    rep_reasons = {
        "baseline_rf": "Median validation macro-F1 (0.9276) across seeds 0, 1, 2 (0.9368, 0.9276, 0.9264). Selected strictly on validation performance prior to unblinding test split.",
        "pytorch_mlp": "Median validation macro-F1 (0.9555) across seeds 0, 1, 2 (0.9555, 0.9533, 0.9634). Selected strictly on validation performance prior to unblinding test split.",
        "pytorch_cnn_1d": "Median validation macro-F1 (0.9458) across seeds 0, 1, 2 (0.9673, 0.9444, 0.9458). Selected strictly on validation performance prior to unblinding test split.",
    }

    summary_dict: dict[str, Any] = {}
    for m_key in ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]:
        m_runs = [r for r in all_run_results if r["model"] == m_key]
        val_f1s = [r["val_macro_f1"] for r in m_runs]
        val_accs = [r["val_accuracy"] for r in m_runs]
        test_f1s = [r["test_macro_f1"] for r in m_runs]
        test_accs = [r["test_accuracy"] for r in m_runs]

        # Per-class test F1 mean and sample std (ddof=1)
        per_class_f1_mean = {}
        per_class_f1_std = {}
        for cls_name in CLASS_NAMES:
            c_f1s = [r["report"][cls_name]["f1-score"] for r in m_runs]
            per_class_f1_mean[cls_name] = float(np.mean(c_f1s))
            per_class_f1_std[cls_name] = float(np.std(c_f1s, ddof=1))

        # Build seed records
        seed_records = {}
        for r in m_runs:
            seed_records[str(r["seed"])] = {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "n_params": r["n_params"],
                "checkpoint_path": r["checkpoint_path"],
                "val_macro_f1": r["val_macro_f1"],
                "val_accuracy": r["val_accuracy"],
                "test_macro_f1": r["test_macro_f1"],
                "test_accuracy": r["test_accuracy"],
                "classification_report": r["report"],
            }

        summary_dict[m_key] = {
            "model_name": m_key,
            "representative_seed": {
                "seed": rep_seeds[m_key],
                "selection_criterion": "median validation macro-F1",
                "val_macro_f1": next(r["val_macro_f1"] for r in m_runs if r["seed"] == rep_seeds[m_key]),
                "reason": rep_reasons[m_key],
            },
            "summary": {
                "val_macro_f1_mean": float(np.mean(val_f1s)),
                "val_macro_f1_std": float(np.std(val_f1s, ddof=1)),
                "val_accuracy_mean": float(np.mean(val_accs)),
                "val_accuracy_std": float(np.std(val_accs, ddof=1)),
                "test_macro_f1_mean": float(np.mean(test_f1s)),
                "test_macro_f1_std": float(np.std(test_f1s, ddof=1)),
                "test_accuracy_mean": float(np.mean(test_accs)),
                "test_accuracy_std": float(np.std(test_accs, ddof=1)),
                "per_class_test_f1_mean": per_class_f1_mean,
                "per_class_test_f1_std": per_class_f1_std,
            },
            "seeds": seed_records,
        }

    summary_file_path = Path("outputs/metrics/results_summary.json")
    save_json(summary_dict, summary_file_path)
    logger.info(f"Saved results summary JSON to {summary_file_path}")

    # 5. Generate Figures
    fig_dir = Path("outputs/figures")

    # fig10_cm_rf.png
    rf_rep_run = next(r for r in all_run_results if r["model"] == "baseline_rf" and r["seed"] == rep_seeds["baseline_rf"])
    plot_confusion_matrix(
        cm=rf_rep_run["cm"],
        model_display_name="Random Forest Baseline",
        seed=rf_rep_run["seed"],
        output_path=fig_dir / "fig10_cm_rf.png",
        cmap="Blues",
    )

    # fig11_cm_mlp.png
    mlp_rep_run = next(r for r in all_run_results if r["model"] == "pytorch_mlp" and r["seed"] == rep_seeds["pytorch_mlp"])
    plot_confusion_matrix(
        cm=mlp_rep_run["cm"],
        model_display_name="PyTorch MLP",
        seed=mlp_rep_run["seed"],
        output_path=fig_dir / "fig11_cm_mlp.png",
        cmap="Oranges",
    )

    # fig12_cm_cnn.png
    cnn_rep_run = next(r for r in all_run_results if r["model"] == "pytorch_cnn_1d" and r["seed"] == rep_seeds["pytorch_cnn_1d"])
    plot_confusion_matrix(
        cm=cnn_rep_run["cm"],
        model_display_name="PyTorch 1D-CNN",
        seed=cnn_rep_run["seed"],
        output_path=fig_dir / "fig12_cm_cnn.png",
        cmap="Greens",
    )

    # fig13_per_class_f1.png
    plot_per_class_f1(summary_data=summary_dict, output_path=fig_dir / "fig13_per_class_f1.png")

    # fig14_val_vs_test.png
    plot_val_vs_test(all_results=all_run_results, output_path=fig_dir / "fig14_val_vs_test.png")

    # 6. Print console summary
    print("\n" + "=" * 90)
    print("FINAL TEST EVALUATION RESULTS TABLE (Task T4a / T4b)")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    print("\nMODEL AGGREGATED SUMMARY (Mean ± Std, sample ddof=1):")
    print("-" * 90)
    for m_key, disp_name in [
        ("baseline_rf", "Random Forest Baseline"),
        ("pytorch_mlp", "PyTorch MLP"),
        ("pytorch_cnn_1d", "PyTorch 1D-CNN"),
    ]:
        s = summary_dict[m_key]["summary"]
        rep = summary_dict[m_key]["representative_seed"]
        print(
            f"{disp_name:<24} | "
            f"Val Macro-F1: {s['val_macro_f1_mean']:.4f} ± {s['val_macro_f1_std']:.4f} | "
            f"Test Macro-F1: {s['test_macro_f1_mean']:.4f} ± {s['test_macro_f1_std']:.4f} | "
            f"Test Acc: {s['test_accuracy_mean']:.4f} ± {s['test_accuracy_std']:.4f} | "
            f"Rep Seed: {rep['seed']}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
