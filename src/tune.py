"""
Module: src.tune
Purpose: Systematic hyperparameter tuning within the pre-declared budget.

Responsibilities:
- Read the pre-declared tuning budget from `configs/tuning_budget.yaml`.
- Execute tuning trials strictly on the VALIDATION split (never test split) using seed 0:
    * Random Forest: 6 trials across n_estimators and max_depth.
    * PyTorch MLP: 6 trials across hidden layer sizes, dropout rates, and class weighting.
    * PyTorch 1D-CNN: 6 trials across filter configurations, dropout rates, and kernel sizes.
- Record every trial's outcomes into `outputs/metrics/tuning_log.csv` (including suboptimal runs).
- Identify best configurations based on highest validation macro-F1 score.
- Output finalised configuration files: `configs/mlp_final.yaml` and `configs/cnn_final.yaml`.
- Generate final loss curves (fig08_mlp_loss_curves.png, fig09_cnn_loss_curves.png) and layer tables.

Design constraints:
- Never touch or evaluate the test split during tuning or model selection.
- Save checkpoints, final configs, logs, metrics, and run_info.json.
- All tuning uses seed 0 as declared in tuning_budget.yaml.
- Retain all runs in tuning_log.csv; never hide unsuccessful trials.
- Loss curve plots saved with titled axes, units, and dpi 150.
- Clear, structured, and reproducible tuning orchestration.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.train import train_pytorch_model
from src.train_baseline import train_random_forest
from src.utils import load_config, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.tune")


def run_tuning(budget_path: str = "configs/tuning_budget.yaml") -> dict[str, Any]:
    """
    Execute the systematic tuning protocol declared in configs/tuning_budget.yaml.

    Why:
        Declares tuning budget before any tuning runs,
        executes trials on validation data only using seed 0, and logs all outcomes.
    """
    logger.info(f"Loading tuning budget from {budget_path}...")
    budget_cfg = load_config(budget_path)

    selection_rule = budget_cfg.get("selection_rule", {})
    tuning_seed = int(selection_rule.get("tuning_seed", 0))
    metric_target = selection_rule.get("metric", "val_macro_f1")

    logger.info(
        f"Tuning protocol: target='{metric_target}', seed={tuning_seed}, test split=EXCLUDED."
    )

    metrics_dir = Path("outputs/metrics")
    configs_dir = Path("configs")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tuning_records: list[dict[str, Any]] = []

    # =========================================================================
    # 1. Random Forest Tuning (6 trials)
    # =========================================================================
    rf_trials = budget_cfg.get("random_forest", {}).get("trials", [])
    logger.info(f"\n{'='*30} Running {len(rf_trials)} Random Forest Trials {'='*30}")
    base_rf_cfg = load_config("configs/baseline_rf.yaml")

    for trial in rf_trials:
        t_id = trial["trial_id"]
        logger.info(f"--- Running Trial {t_id}: {trial} ---")
        t_start = time.perf_counter()
        res = train_random_forest(
            config=base_rf_cfg,
            seed=tuning_seed,
            override_params=trial,
            save_artifacts=False,
        )
        duration = time.perf_counter() - t_start

        tuning_records.append({
            "trial_id": t_id,
            "model_type": "random_forest",
            "hyperparameters": json.dumps({
                "n_estimators": trial.get("n_estimators"),
                "max_depth": trial.get("max_depth"),
                "class_weight": trial.get("class_weight"),
            }),
            "val_macro_f1": res["val_macro_f1"],
            "val_accuracy": res["val_accuracy"],
            "val_loss": None,
            "best_epoch": None,
            "n_params": None,
            "duration_seconds": round(duration, 2),
        })

    # =========================================================================
    # 2. PyTorch MLP Tuning (6 trials)
    # =========================================================================
    mlp_trials = budget_cfg.get("mlp", {}).get("trials", [])
    logger.info(f"\n{'='*30} Running {len(mlp_trials)} PyTorch MLP Trials {'='*30}")
    base_mlp_cfg = load_config("configs/mlp.yaml")

    for trial in mlp_trials:
        t_id = trial["trial_id"]
        logger.info(f"--- Running Trial {t_id}: {trial} ---")
        t_start = time.perf_counter()
        trial_overrides = {
            "hidden_sizes": trial["hidden_sizes"],
            "dropout": trial["dropout"],
            "learning_rate": trial["lr"],
            "class_weight": trial["class_weight"],
        }
        res = train_pytorch_model(
            config=base_mlp_cfg,
            seed=tuning_seed,
            smoke_test=False,
            override_params=trial_overrides,
            save_artifacts=False,
        )
        duration = time.perf_counter() - t_start

        tuning_records.append({
            "trial_id": t_id,
            "model_type": "pytorch_mlp",
            "hyperparameters": json.dumps(trial_overrides),
            "val_macro_f1": res["best_val_macro_f1"],
            "val_accuracy": res["val_accuracy_at_best"],
            "val_loss": round(res["val_loss_at_best"], 4),
            "best_epoch": res["best_epoch"],
            "n_params": res["total_trainable_parameters"],
            "duration_seconds": round(duration, 2),
        })

    # =========================================================================
    # 3. PyTorch 1D-CNN Tuning (6 trials)
    # =========================================================================
    cnn_trials = budget_cfg.get("cnn", {}).get("trials", [])
    logger.info(f"\n{'='*30} Running {len(cnn_trials)} PyTorch 1D-CNN Trials {'='*30}")
    base_cnn_cfg = load_config("configs/cnn.yaml")

    for trial in cnn_trials:
        t_id = trial["trial_id"]
        logger.info(f"--- Running Trial {t_id}: {trial} ---")
        t_start = time.perf_counter()
        trial_overrides = {
            "filters": trial["filters"],
            "kernel_size": trial["kernel_size"],
            "dropout": trial["dropout"],
            "learning_rate": trial["lr"],
        }
        res = train_pytorch_model(
            config=base_cnn_cfg,
            seed=tuning_seed,
            smoke_test=False,
            override_params=trial_overrides,
            save_artifacts=False,
        )
        duration = time.perf_counter() - t_start

        tuning_records.append({
            "trial_id": t_id,
            "model_type": "pytorch_cnn_1d",
            "hyperparameters": json.dumps(trial_overrides),
            "val_macro_f1": res["best_val_macro_f1"],
            "val_accuracy": res["val_accuracy_at_best"],
            "val_loss": round(res["val_loss_at_best"], 4),
            "best_epoch": res["best_epoch"],
            "n_params": res["total_trainable_parameters"],
            "duration_seconds": round(duration, 2),
        })

    # =========================================================================
    # 4. Save Complete Tuning Log
    # =========================================================================
    tuning_df = pd.DataFrame(tuning_records)
    log_csv_path = metrics_dir / "tuning_log.csv"
    tuning_df.to_csv(log_csv_path, index=False)
    logger.info(f"\nSaved complete tuning log with {len(tuning_df)} trials to {log_csv_path}")

    # =========================================================================
    # 5. Model Selection & Final Config Emission
    # =========================================================================
    rf_df = tuning_df[tuning_df["model_type"] == "random_forest"]
    mlp_df = tuning_df[tuning_df["model_type"] == "pytorch_mlp"]
    cnn_df = tuning_df[tuning_df["model_type"] == "pytorch_cnn_1d"]

    best_rf_idx = rf_df["val_macro_f1"].idxmax()
    best_mlp_idx = mlp_df["val_macro_f1"].idxmax()
    best_cnn_idx = cnn_df["val_macro_f1"].idxmax()

    best_rf_record = tuning_df.loc[best_rf_idx]
    best_mlp_record = tuning_df.loc[best_mlp_idx]
    best_cnn_record = tuning_df.loc[best_cnn_idx]

    logger.info("\n" + "=" * 60)
    logger.info("TUNING SELECTION RESULTS (Seed 0 Validation Macro-F1):")
    logger.info(
        f"  Random Forest: {best_rf_record['trial_id']} | "
        f"Val Macro-F1 = {best_rf_record['val_macro_f1']:.4f} | "
        f"Params: {best_rf_record['hyperparameters']}"
    )
    logger.info(
        f"  PyTorch MLP:   {best_mlp_record['trial_id']} | "
        f"Val Macro-F1 = {best_mlp_record['val_macro_f1']:.4f} | "
        f"Params: {best_mlp_record['hyperparameters']}"
    )
    logger.info(
        f"  PyTorch 1D-CNN:{best_cnn_record['trial_id']} | "
        f"Val Macro-F1 = {best_cnn_record['val_macro_f1']:.4f} | "
        f"Params: {best_cnn_record['hyperparameters']}"
    )
    logger.info("=" * 60 + "\n")

    # Generate configs/baseline_rf_final.yaml
    rf_best_params = json.loads(best_rf_record["hyperparameters"])
    rf_final_cfg = copy.deepcopy(base_rf_cfg)
    rf_final_cfg["model"]["n_estimators"] = rf_best_params["n_estimators"]
    rf_final_cfg["model"]["max_depth"] = rf_best_params["max_depth"]
    rf_final_cfg["model"]["class_weight"] = rf_best_params["class_weight"]
    rf_final_path = configs_dir / "baseline_rf_final.yaml"
    with open(rf_final_path, "w", encoding="utf-8") as f:
        yaml.dump(rf_final_cfg, f, default_flow_style=False)
    logger.info(f"Generated finalised config: {rf_final_path}")

    # Generate configs/mlp_final.yaml
    mlp_best_params = json.loads(best_mlp_record["hyperparameters"])
    mlp_final_cfg = copy.deepcopy(base_mlp_cfg)
    mlp_final_cfg["model"]["hidden_sizes"] = mlp_best_params["hidden_sizes"]
    mlp_final_cfg["model"]["dropout"] = mlp_best_params["dropout"]
    mlp_final_cfg["training"]["learning_rate"] = mlp_best_params["learning_rate"]
    mlp_final_cfg["training"]["use_class_weights"] = mlp_best_params["class_weight"]
    mlp_final_path = configs_dir / "mlp_final.yaml"
    with open(mlp_final_path, "w", encoding="utf-8") as f:
        yaml.dump(mlp_final_cfg, f, default_flow_style=False)
    logger.info(f"Generated finalised config: {mlp_final_path}")

    # Generate configs/cnn_final.yaml
    cnn_best_params = json.loads(best_cnn_record["hyperparameters"])
    cnn_final_cfg = copy.deepcopy(base_cnn_cfg)
    cnn_final_cfg["model"]["filters"] = cnn_best_params["filters"]
    cnn_final_cfg["model"]["kernel_size"] = cnn_best_params["kernel_size"]
    cnn_final_cfg["model"]["dropout"] = cnn_best_params["dropout"]
    cnn_final_cfg["training"]["learning_rate"] = cnn_best_params["learning_rate"]
    cnn_final_path = configs_dir / "cnn_final.yaml"
    with open(cnn_final_path, "w", encoding="utf-8") as f:
        yaml.dump(cnn_final_cfg, f, default_flow_style=False)
    logger.info(f"Generated finalised config: {cnn_final_path}")

    # Save tuning summary JSON
    summary = {
        "selection_rule": selection_rule,
        "best_models": {
            "random_forest": {
                "trial_id": best_rf_record["trial_id"],
                "val_macro_f1": best_rf_record["val_macro_f1"],
                "val_accuracy": best_rf_record["val_accuracy"],
                "hyperparameters": json.loads(best_rf_record["hyperparameters"]),
                "final_config_path": str(rf_final_path),
            },
            "mlp": {
                "trial_id": best_mlp_record["trial_id"],
                "val_macro_f1": best_mlp_record["val_macro_f1"],
                "val_accuracy": best_mlp_record["val_accuracy"],
                "val_loss": best_mlp_record["val_loss"],
                "best_epoch": int(best_mlp_record["best_epoch"]),
                "n_params": int(best_mlp_record["n_params"]),
                "hyperparameters": mlp_best_params,
                "final_config_path": str(mlp_final_path),
            },
            "cnn": {
                "trial_id": best_cnn_record["trial_id"],
                "val_macro_f1": best_cnn_record["val_macro_f1"],
                "val_accuracy": best_cnn_record["val_accuracy"],
                "val_loss": best_cnn_record["val_loss"],
                "best_epoch": int(best_cnn_record["best_epoch"]),
                "n_params": int(best_cnn_record["n_params"]),
                "hyperparameters": cnn_best_params,
                "final_config_path": str(cnn_final_path),
            },
        },
    }
    save_json(summary, metrics_dir / "tuning_summary.json")

    # =========================================================================
    # 6. Final Training & Artifact Generation for Best Models (Seed 0)
    # =========================================================================
    logger.info("\nTraining and generating persistent artifacts for finalised models...")
    # 1. Best RF (seed 0)
    rf_best_params = json.loads(best_rf_record["hyperparameters"])
    train_random_forest(
        config=base_rf_cfg,
        seed=tuning_seed,
        override_params=rf_best_params,
        save_artifacts=True,
    )

    # 2. Best MLP (seed 0) -> writes fig08_mlp_loss_curves.png, mlp_layers.md, checkpoints
    logger.info("Training finalised MLP (seed 0)...")
    train_pytorch_model(
        config=mlp_final_cfg,
        seed=tuning_seed,
        smoke_test=False,
        save_artifacts=True,
    )

    # 3. Best CNN (seed 0) -> writes fig09_cnn_loss_curves.png, cnn_layers.md, checkpoints
    logger.info("Training finalised 1D-CNN (seed 0)...")
    train_pytorch_model(
        config=cnn_final_cfg,
        seed=tuning_seed,
        smoke_test=False,
        save_artifacts=True,
    )

    logger.info("\nPhase 3 Tuning & Model Finalisation successfully completed!")
    return summary


def main() -> None:
    run_tuning()


if __name__ == "__main__":
    main()
