"""
Module: src.train
Purpose: Generic PyTorch model training and validation pipeline.

Responsibilities:
- Parse CLI arguments (e.g. `--config configs/cnn.yaml --seed 0 --smoke-test`).
- Set random seeds across Python, NumPy, and PyTorch using `src.utils.set_seed`.
- Instantiate DataLoaders (batch size 64, shuffle train split only).
- Setup Adam optimiser (learning rate, weight decay from config) and CrossEntropyLoss
  (with optional class weights computed solely from training labels).
- Training loop:
    * Log train loss, validation loss, validation macro-F1, and validation accuracy per epoch.
    * Early stopping tracked on validation macro-F1 with patience = 8 epochs.
    * Checkpoint the best model to `outputs/checkpoints/<model_name>_seed<seed>.pt`.
    * Save copy of active config and write `run_info.json`.
    * Save training history log (CSV) and plot loss curves to `outputs/figures/`.
- Provide `--smoke-test` mode: subset of 2 train subjects, 1 val subject, 1 epoch (< 1 min execution).
- Save layer table markdown to `outputs/metrics/<model_name>_layers.md`.

Rule compliance:
- Rule 1: Python 3.11, PyTorch (CPU).
- Rule 7: Never touch or evaluate the test split in this training script.
- Rule 8: Save checkpoints, config copy, training log, and run_info.json for every run.
- Rule 9: Set all seeds from config.
- Rule 10: Log all training progression metrics directly from execution.
- Rule 12: Loss curve plots saved as PNG (dpi 150) with titled axes and units.
- Rule 13: Docstrings explain architecture, loss, and training loop details.
- Rule 14: Clean, readable, modular implementation.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import PyTorchCNN1D, generate_layer_table_cnn
from src.models.mlp import PyTorchMLP, generate_layer_table_mlp
from src.utils import load_config, save_json, set_seed, write_run_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.train")


def create_model_from_config(config: dict[str, Any]) -> nn.Module:
    """
    Instantiate PyTorchMLP or PyTorchCNN1D based on configuration specifications.

    Why:
        Unified model factory ensuring hyperparameter decoupling across YAML configs.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "").lower()

    if "mlp" in name:
        input_dim = int(model_cfg.get("input_dim", 84))
        hidden_sizes = list(model_cfg.get("hidden_sizes", [128, 64]))
        dropout = float(model_cfg.get("dropout", 0.2))
        num_classes = int(model_cfg.get("num_classes", 6))
        return PyTorchMLP(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            num_classes=num_classes,
        )
    elif "cnn" in name:
        in_channels = int(model_cfg.get("in_channels", 12))
        seq_len = int(model_cfg.get("sequence_length", 128))
        filters = list(model_cfg.get("filters", [32, 64, 64]))
        kernel_size = int(model_cfg.get("kernel_size", 5))
        dropout = float(model_cfg.get("dropout", 0.3))
        num_classes = int(model_cfg.get("num_classes", 6))
        return PyTorchCNN1D(
            in_channels=in_channels,
            sequence_length=seq_len,
            filters=filters,
            kernel_size=kernel_size,
            dropout=dropout,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model name in configuration: {name}")


def load_dataset_splits(
    model_name: str,
    smoke_test: bool = False,
    n_train_subjects: int = 2,
    n_val_subjects: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached training and validation arrays.

    Why:
        Rule 7 strictly restricts data loading to train and validation splits.
        Never loads X_test or y_test.
        For CNN, transposes raw arrays to (N, 12, 128) channels-first format.
    """
    is_cnn = "cnn" in model_name.lower()

    if is_cnn:
        X_train = np.load("outputs/cache/X_train_raw_scaled.npy")
        X_val = np.load("outputs/cache/X_val_raw_scaled.npy")
        # Ensure shape is (N, channels, sequence_length) -> (N, 12, 128)
        if X_train.ndim == 3 and X_train.shape[1] == 128 and X_train.shape[2] == 12:
            X_train = np.transpose(X_train, (0, 2, 1))
        if X_val.ndim == 3 and X_val.shape[1] == 128 and X_val.shape[2] == 12:
            X_val = np.transpose(X_val, (0, 2, 1))
    else:
        X_train = np.load("outputs/cache/X_train_feat_scaled.npy")
        X_val = np.load("outputs/cache/X_val_feat_scaled.npy")

    y_train = np.load("outputs/cache/y_train.npy")
    y_val = np.load("outputs/cache/y_val.npy")

    # Smoke test subsetting: filter by subject boundaries
    if smoke_test:
        win_tr = pd.read_csv("outputs/splits/windows_train.csv")
        win_va = pd.read_csv("outputs/splits/windows_val.csv")

        train_subjs = win_tr["subject"].unique()[:n_train_subjects]
        val_subjs = win_va["subject"].unique()[:n_val_subjects]

        tr_mask = win_tr["subject"].isin(train_subjs).to_numpy()
        va_mask = win_va["subject"].isin(val_subjs).to_numpy()

        X_train = X_train[tr_mask]
        y_train = y_train[tr_mask]
        X_val = X_val[va_mask]
        y_val = y_val[va_mask]

        logger.info(
            f"[SMOKE-TEST] Filtered to subjects: train={train_subjs.tolist()}, val={val_subjs.tolist()}. "
            f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}"
        )

    return X_train, y_train, X_val, y_val


def plot_loss_curves(
    history: list[dict[str, Any]],
    best_epoch: int,
    best_macro_f1: float,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Generate and save training and validation loss curves alongside validation macro-F1
    on a secondary y-axis, with marked best epoch and integer epoch ticks.

    Why:
        Plan Section 3b & Rule 12: visual diagnosis of learning convergence,
        overfitting inflection points, and tracking selection noise on validation macro-F1.
    """
    epochs = [int(h["epoch"]) for h in history]
    train_losses = [float(h["train_loss"]) for h in history]
    val_losses = [float(h["val_loss"]) for h in history]
    val_f1s = [float(h["val_macro_f1"]) for h in history]

    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=150)

    # Primary axis: Loss
    line1 = ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", lw=2)
    line2 = ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11, color="#333333", fontweight="medium")
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Secondary axis: Validation Macro-F1
    ax2 = ax1.twinx()
    line3 = ax2.plot(
        epochs,
        val_f1s,
        label="Val Macro-F1",
        color="#2ca02c",
        linestyle="-.",
        lw=1.8,
        marker="o",
        markersize=4,
    )
    ax2.set_ylabel("Validation Macro-F1", fontsize=11, color="#2ca02c", fontweight="medium")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # Mark the selected best checkpoint epoch
    line_best = ax1.axvline(
        x=best_epoch,
        color="#d62728",
        linestyle="--",
        lw=1.5,
        label=f"Best Model (Epoch {best_epoch}, Val F1={best_macro_f1:.4f})",
    )
    if 1 <= best_epoch <= len(epochs):
        best_f1_val = val_f1s[best_epoch - 1]
        ax2.plot(best_epoch, best_f1_val, "*", color="#d62728", markersize=12, zorder=5)

    # Combine legends from both axes
    lines = line1 + line2 + line3 + [line_best]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    model_display = "PyTorch 1D-CNN" if "cnn" in model_name.lower() else "PyTorch MLP"
    ax1.set_title(
        f"{model_display}: Loss & Validation Macro-F1 vs Training Epoch",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved to {output_path}")


def train_pytorch_model(
    config: dict[str, Any],
    seed: int = 0,
    smoke_test: bool = False,
    override_params: dict[str, Any] | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Train and validate a PyTorch model according to configuration.

    Why:
        Generic trainer supporting both MLP and 1D-CNN architectures, with
        deterministic seeding, early stopping on val macro-F1, checkpointing,
        and comprehensive artifact logging.
    """
    set_seed(seed)
    cfg = copy.deepcopy(config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    smoke_cfg = cfg.get("smoke_test", {})

    # Apply overrides (e.g. from tuning trial)
    if override_params:
        for k, v in override_params.items():
            if k in model_cfg:
                model_cfg[k] = v
            elif k in train_cfg:
                train_cfg[k] = v
            elif k == "class_weight":
                train_cfg["use_class_weights"] = bool(v)

    model_name = model_cfg.get("name", "pytorch_model")
    is_cnn = "cnn" in model_name.lower()

    # Training parameters
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    max_epochs = int(train_cfg.get("max_epochs", 60))
    patience = int(train_cfg.get("patience", 8))
    use_class_weights = bool(train_cfg.get("use_class_weights", False))

    if smoke_test:
        max_epochs = int(smoke_cfg.get("epochs", 1))
        n_train_sub = int(smoke_cfg.get("n_train_subjects", 2))
        n_val_sub = int(smoke_cfg.get("n_val_subjects", 1))
    else:
        n_train_sub, n_val_sub = 16, 4

    # Setup directories
    ckpt_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")
    logs_dir = Path("outputs/logs")
    fig_dir = Path("outputs/figures")
    for d in (ckpt_dir, metrics_dir, logs_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info(f"Loading data for {model_name} (smoke_test={smoke_test})...")
    X_train_np, y_train_np, X_val_np, y_val_np = load_dataset_splits(
        model_name=model_name,
        smoke_test=smoke_test,
        n_train_subjects=n_train_sub,
        n_val_subjects=n_val_sub,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_np, dtype=torch.float32),
        torch.tensor(y_val_np, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Build model
    model = create_model_from_config(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Instantiated {model_name} with {total_params:,} trainable parameters.")

    # 3. Loss function with optional class weighting computed solely from TRAIN labels
    if use_class_weights:
        classes = np.unique(y_train_np)
        weights = compute_class_weight("balanced", classes=classes, y=y_train_np)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Using class-weighted CrossEntropyLoss: {weights.round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training loop with early stopping on validation macro-F1
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    start_train_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_y)
            train_batches += len(batch_y)

        train_loss = train_loss_sum / train_batches

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_sum += loss.item() * len(batch_y)
                val_batches += len(batch_y)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        val_loss = val_loss_sum / val_batches
        val_macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
        val_acc = float(accuracy_score(all_targets, all_preds))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_acc,
        })

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping tracking on validation macro-F1
        if val_macro_f1 > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs). "
                    f"Best epoch was {best_epoch} with Val Macro-F1 = {best_val_macro_f1:.4f}."
                )
                break

    total_duration = time.perf_counter() - start_train_time
    logger.info(
        f"Training finished in {total_duration:.2f}s. "
        f"Best Val Macro-F1: {best_val_macro_f1:.4f} at epoch {best_epoch}."
    )

    # Compile results
    results: dict[str, Any] = {
        "model_name": model_name,
        "seed": seed,
        "total_trainable_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "val_loss_at_best": best_val_loss,
        "val_accuracy_at_best": best_val_acc,
        "total_epochs_trained": len(history),
        "total_duration_seconds": total_duration,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "use_class_weights": use_class_weights,
            "dropout": model_cfg.get("dropout"),
            "hidden_sizes": model_cfg.get("hidden_sizes") if not is_cnn else None,
            "filters": model_cfg.get("filters") if is_cnn else None,
            "kernel_size": model_cfg.get("kernel_size") if is_cnn else None,
        },
    }

    if save_artifacts and not smoke_test:
        # Save best model checkpoint
        ckpt_filename = f"{model_name}_seed{seed}.pt"
        ckpt_path = ckpt_dir / ckpt_filename
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state_dict,
                "best_val_macro_f1": best_val_macro_f1,
                "config": cfg,
                "seed": seed,
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save active config copy
        cfg_copy_path = ckpt_dir / f"{model_name}_seed{seed}_config.yaml"
        with open(cfg_copy_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(cfg, f, default_flow_style=False)

        # Save training history CSV
        history_df = pd.DataFrame(history)
        history_path = logs_dir / f"{model_name}_seed{seed}_training.csv"
        history_df.to_csv(history_path, index=False)
        if seed == 0:
            history_df.to_csv(logs_dir / f"{model_name}_training.csv", index=False)
        logger.info(f"Saved training log to {history_path}")

        # Save metrics JSON
        metrics_path = metrics_dir / f"{model_name}_seed{seed}.json"
        save_json(results, metrics_path)
        if seed == 0:
            save_json(results, metrics_dir / f"{model_name}.json")

        # Save execution provenance
        run_info_path = ckpt_dir / f"run_info_{model_name}_seed{seed}.json"
        write_run_info(
            run_info_path,
            extra_info={
                "model": model_name,
                "seed": seed,
                "best_val_macro_f1": best_val_macro_f1,
                "val_accuracy": best_val_acc,
                "best_epoch": best_epoch,
                "total_trainable_parameters": total_params,
            },
        )

        # Save loss curves (seed 0 only to avoid overwriting fig08/fig09; skip for ablation runs)
        if seed == 0 and "nodrop" not in model_name.lower():
            fig_name = "fig09_cnn_loss_curves.png" if is_cnn else "fig08_mlp_loss_curves.png"
            fig_path = fig_dir / fig_name
            plot_loss_curves(
                history=history,
                best_epoch=best_epoch,
                best_macro_f1=best_val_macro_f1,
                model_name=model_name,
                output_path=fig_path,
            )

        # Save layer table markdown
        if seed == 0 and "nodrop" not in model_name.lower():
            if is_cnn:
                layer_table_md = generate_layer_table_cnn(model)
                table_path = metrics_dir / "cnn_layers.md"
            else:
                layer_table_md = generate_layer_table_mlp(model)
                table_path = metrics_dir / "mlp_layers.md"
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(layer_table_md)
            logger.info(f"Saved layer table to {table_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic PyTorch HAR trainer.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/cnn.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for weight init and shuffling (default: 0).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test mode: 2 train subjects, 1 val subject, 1 epoch.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_pytorch_model(
        config=cfg,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
