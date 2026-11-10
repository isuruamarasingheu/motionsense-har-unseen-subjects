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
- Rule 10: Never invent or estimate metrics; compute strictly from model predictions.
- Rule 13: Docstrings explain methodology and hyperparameters.
- Rule 14: Clear, idiomatic scikit-learn code.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.utils import load_config, save_json, set_seed, write_run_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.train_baseline")


def train_random_forest(
    config: dict[str, Any] | None = None,
    seed: int = 0,
    override_params: dict[str, Any] | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Train and evaluate a Random Forest classifier on the 84 scaled tabular features.

    Why:
        Serves as the strong non-neural benchmark (Task T3a) to compare against
        the NumPy MLP, PyTorch MLP, and PyTorch 1D-CNN.

    Args:
        config: Full configuration dict (defaults to configs/baseline_rf.yaml if None).
        seed: Random state seed for reproducibility.
        override_params: Dictionary of hyperparameter overrides (e.g. from tuning budget).
        save_artifacts: Whether to save model checkpoint, config copy, and metrics to disk.

    Returns:
        Dictionary containing validation metrics, model hyperparameters, and training time.
    """
    if config is None:
        config = load_config("configs/baseline_rf.yaml")

    active_cfg = copy.deepcopy(config)
    model_cfg = active_cfg.get("model", {})

    # Extract model hyperparameters
    n_estimators = int(model_cfg.get("n_estimators", 100))
    max_depth = model_cfg.get("max_depth", None)
    if max_depth is not None:
        max_depth = int(max_depth)
    class_weight = model_cfg.get("class_weight", "balanced")
    n_jobs = int(model_cfg.get("n_jobs", -1))

    # Apply overrides if provided (e.g. from tuning trials)
    if override_params:
        if "n_estimators" in override_params:
            n_estimators = int(override_params["n_estimators"])
        if "max_depth" in override_params:
            max_depth = override_params["max_depth"]
            if max_depth is not None:
                max_depth = int(max_depth)
        if "class_weight" in override_params:
            class_weight = override_params["class_weight"]

    set_seed(seed)

    # Output directories
    ckpt_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")
    for d in (ckpt_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load scaled tabular features and labels (Strictly train and validation only)
    logger.info("Loading tabular training and validation feature sets...")
    X_train = np.load("outputs/cache/X_train_feat_scaled.npy")
    y_train = np.load("outputs/cache/y_train.npy")
    X_val = np.load("outputs/cache/X_val_feat_scaled.npy")
    y_val = np.load("outputs/cache/y_val.npy")

    logger.info(
        f"Data loaded: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_val={X_val.shape}, y_val={y_val.shape}"
    )

    # Initialise model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=n_jobs,
    )

    logger.info(
        f"Training RandomForestClassifier(n_estimators={n_estimators}, "
        f"max_depth={max_depth}, class_weight='{class_weight}', seed={seed})..."
    )
    start_time = time.perf_counter()
    rf.fit(X_train, y_train)
    fit_duration = time.perf_counter() - start_time
    logger.info(f"Model fitting complete in {fit_duration:.2f} seconds.")

    # Validation evaluation
    y_pred_val = rf.predict(X_val)
    val_macro_f1 = float(f1_score(y_val, y_pred_val, average="macro"))
    val_acc = float(accuracy_score(y_val, y_pred_val))
    report = classification_report(y_val, y_pred_val, output_dict=True)

    logger.info(
        f"Validation Results: Macro-F1 = {val_macro_f1:.4f}, Accuracy = {val_acc:.4f}"
    )

    results: dict[str, Any] = {
        "model_name": "baseline_rf",
        "seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": class_weight,
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_acc,
        "fit_duration_seconds": fit_duration,
        "classification_report": report,
    }

    if save_artifacts:
        ckpt_path = ckpt_dir / f"baseline_rf_seed{seed}.joblib"
        joblib.dump(rf, ckpt_path)
        logger.info(f"Saved model checkpoint to {ckpt_path}")

        # Save config copy
        config_copy_path = ckpt_dir / f"baseline_rf_seed{seed}_config.yaml"
        active_cfg["model"]["n_estimators"] = n_estimators
        active_cfg["model"]["max_depth"] = max_depth
        active_cfg["model"]["class_weight"] = class_weight
        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.dump(active_cfg, f, default_flow_style=False)

        # Save metrics JSON
        metrics_path = metrics_dir / f"baseline_rf_seed{seed}.json"
        save_json(results, metrics_path)
        if seed == 0:
            save_json(results, metrics_dir / "baseline_rf.json")
        logger.info(f"Saved metrics to {metrics_path}")

        # Provenance tracking
        run_info_path = ckpt_dir / f"run_info_baseline_rf_seed{seed}.json"
        write_run_info(
            run_info_path,
            extra_info={
                "model": "baseline_rf",
                "seed": seed,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            },
        )
        logger.info(f"Saved execution provenance to {run_info_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest baseline model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_rf.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random state seed (default: 0).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Override number of trees.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Override max tree depth (None for unlimited).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth

    train_random_forest(config=cfg, seed=args.seed, override_params=overrides)


if __name__ == "__main__":
    main()
