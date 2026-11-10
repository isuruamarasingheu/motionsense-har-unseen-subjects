"""
Script: src.train_numpy_mlp
Purpose: Standalone execution script for Task T2: NumPy MLP from first principles.

Tasks:
1. Load dataset cache and config (configs/numpy_mlp.yaml).
2. Stratified sample up to 3000 train windows; evaluate on full val set.
3. Verify initial loss sanity check (ln(6) ≈ 1.7918).
4. Run numerical gradient check via central differences (tol < 1e-6 across 8 entries).
5. Run controlled overfitting check on 32 samples (2000 epochs) -> fig06_numpy_mlp_overfit.png.
6. Train on 3000 samples with mini-batches (size 64), lr 0.05, early stopping patience 15 -> fig05_numpy_mlp_loss.png.
7. Save outputs/metrics/numpy_mlp.json, training log CSV, checkpoint NPZ, and run_info.json.

Design constraints:
- Never touch test set.
- Save checkpoint, config copy, metrics JSON, training log, run_info.json.
- Set all seeds from config.
- Never invent or estimate numbers; log all metrics from execution.
- Figures saved to outputs/figures/ with labelled axes and units.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.models.numpy_mlp import (
    NumpyMLP,
    gradient_check,
    overfit_tiny_batch,
    train_numpy_mlp,
)
from src.utils import load_config, save_json, set_seed, write_run_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.train_numpy_mlp")


def run_numpy_mlp_pipeline(
    config_path: str = "configs/numpy_mlp.yaml",
) -> dict[str, Any]:
    """Execute the end-to-end NumPy MLP training, gradient check, and evaluation pipeline."""
    # 1. Load configuration
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    grad_cfg = cfg.get("gradient_check", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    n_in = int(model_cfg.get("n_in", 84))
    n_hidden = int(model_cfg.get("n_hidden", 32))
    n_out = int(model_cfg.get("n_out", 6))

    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 0.05))
    max_epochs = int(train_cfg.get("max_epochs", 200))
    patience = int(train_cfg.get("patience", 15))
    train_subset_size = int(train_cfg.get("train_subset_size", 3000))

    eps = float(grad_cfg.get("epsilon", 1e-5))
    tol = float(grad_cfg.get("tolerance", 1e-6))
    grad_batch_size = int(grad_cfg.get("batch_size", 8))

    # Ensure output directories exist
    fig_dir = Path("outputs/figures")
    metrics_dir = Path("outputs/metrics")
    logs_dir = Path("outputs/logs")
    ckpt_dir = Path("outputs/checkpoints")
    for d in (fig_dir, metrics_dir, logs_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load cached features and targets
    logger.info("Loading preprocessed feature data from outputs/cache/...")
    X_train_full = np.load("outputs/cache/X_train_feat_scaled.npy")
    y_train_full = np.load("outputs/cache/y_train.npy")
    X_val = np.load("outputs/cache/X_val_feat_scaled.npy")
    y_val = np.load("outputs/cache/y_val.npy")

    logger.info(f"Loaded: X_train_full={X_train_full.shape}, X_val={X_val.shape}")

    # Stratified subset of training data (up to 3000 windows)
    if len(y_train_full) > train_subset_size:
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=train_subset_size,
            stratify=y_train_full,
            random_state=seed,
        )
    else:
        X_train_sub, y_train_sub = X_train_full, y_train_full

    logger.info(
        f"Training subset selected: shape={X_train_sub.shape}, "
        f"class distribution={np.bincount(y_train_sub)}"
    )

    # 3. Model instantiation and initial loss sanity check
    model = NumpyMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=seed)
    P_init = model.forward(X_train_sub)
    init_loss = model.loss(P_init, y_train_sub)
    theoretical_loss = np.log(n_out)
    logger.info(
        f"Initial Loss Sanity Check: measured={init_loss:.4f}, "
        f"theoretical ln({n_out})={theoretical_loss:.4f}"
    )

    # 4. Numerical Gradient Checking (Batch of 8 samples)
    logger.info("Executing numerical gradient checking via central differences...")
    rng_check = np.random.default_rng(seed)
    indices_check = rng_check.choice(len(y_train_sub), size=grad_batch_size, replace=False)
    X_grad_batch = X_train_sub[indices_check]
    y_grad_batch = y_train_sub[indices_check]

    grad_entries = [
        ("W1", (0, 0)),
        ("W1", (42, 16)),
        ("b1", (0,)),
        ("b1", (15,)),
        ("W2", (0, 0)),
        ("W2", (16, 3)),
        ("b2", (0,)),
        ("b2", (3,)),
    ]
    grad_results = gradient_check(
        model,
        X_grad_batch,
        y_grad_batch,
        eps=eps,
        tol=tol,
        check_entries=grad_entries,
    )

    all_passed = True
    for r in grad_results:
        status_str = "PASS" if r["passed"] else "FAIL"
        logger.info(
            f"  [{status_str}] {r['parameter']:<3} index={str(r['index']):<9} "
            f"ana={r['analytical']:+.6e} num={r['numerical']:+.6e} "
            f"abs_diff={r['abs_diff']:.2e} rel_diff={r['rel_diff']:.2e}"
        )
        if not r["passed"]:
            all_passed = False

    if not all_passed:
        logger.error("Gradient check failed! Numerical derivatives do not match analytical ones.")
    else:
        logger.info("All 8 gradient checks passed with relative difference < 1e-6!")

    # 5. Controlled Sanity Check: Overfitting a tiny batch (32 samples, 2000 epochs)
    logger.info("Running controlled overfitting check on 32 samples (2000 epochs)...")
    overfit_model = NumpyMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=seed)
    X_tiny = X_train_sub[:32]
    y_tiny = y_train_sub[:32]
    overfit_res = overfit_tiny_batch(overfit_model, X_tiny, y_tiny, lr=lr, epochs=2000)

    logger.info(
        f"Overfitting completed: Initial Loss={overfit_res['initial_loss']:.4f}, "
        f"Final Loss={overfit_res['final_loss']:.6f}, "
        f"Final Accuracy={overfit_res['final_accuracy']*100:.1f}%"
    )

    # Plot fig06_numpy_mlp_overfit.png
    fig_overfit, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    color = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss (log scale)", color=color, fontsize=11, fontweight="bold")
    ax1.semilogy(
        overfit_res["epochs"],
        overfit_res["loss_history"],
        color=color,
        linewidth=2,
        label="Training Loss (32 samples)",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_acc = "tab:green"
    ax2.set_ylabel("Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    ax2.plot(
        overfit_res["epochs"],
        [a * 100.0 for a in overfit_res["acc_history"]],
        color=color_acc,
        linewidth=1.5,
        linestyle="--",
        label="Training Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(-5, 105)
    ax2.grid(False)

    fig_overfit.suptitle(
        "NumPy MLP Sanity Check: Memorising Tiny Batch (N=32, 2000 Epochs)",
        fontsize=12,
        fontweight="bold",
    )
    fig_overfit.tight_layout()
    overfit_fig_path = fig_dir / "fig06_numpy_mlp_overfit.png"
    fig_overfit.savefig(overfit_fig_path, dpi=150)
    plt.close(fig_overfit)
    logger.info(f"Saved overfitting plot to {overfit_fig_path}")

    # 6. Full Training on 3000-Sample Stratified Subset with Early Stopping
    logger.info("Executing main training loop on 3000-sample subset with early stopping...")
    train_model = NumpyMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=seed)
    train_res = train_numpy_mlp(
        train_model,
        X_train_sub,
        y_train_sub,
        X_val,
        y_val,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        verbose=True,
    )

    logger.info(
        f"Training finished: Completed Epochs={train_res['epochs_completed']}, "
        f"Best Epoch={train_res['best_epoch']}, "
        f"Final Val Loss={train_res['final_val_loss']:.4f}, "
        f"Final Val Macro-F1={train_res['final_val_macro_f1']:.4f}"
    )

    # Plot fig05_numpy_mlp_loss.png
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5), dpi=150)
    epochs_range = train_res["history"]["epoch"]
    ax_loss.plot(
        epochs_range,
        train_res["history"]["train_loss"],
        color="tab:blue",
        linewidth=2,
        label="Train Loss (3000 subset)",
    )
    ax_loss.plot(
        epochs_range,
        train_res["history"]["val_loss"],
        color="tab:orange",
        linewidth=2,
        linestyle="-",
        label="Validation Loss (3534 windows)",
    )
    ax_loss.axvline(
        train_res["best_epoch"],
        color="tab:green",
        linestyle=":",
        linewidth=2,
        label=f"Best Epoch ({train_res['best_epoch']})",
    )
    ax_loss.set_title(
        "NumPy MLP (84-32-6 tanh): Cross-Entropy Loss vs Epoch",
        fontsize=12,
        fontweight="bold",
    )
    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Mean Cross-Entropy Loss (per window)", fontsize=11, fontweight="bold")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right", frameon=True)

    # Text box with key metrics (placed upper-middle to avoid overlapping early epoch curves)
    summary_box = (
        f"Epoch 0 Train Loss: {train_res['initial_train_loss']:.3f}\n"
        f"Epoch 0 Val Loss:   {train_res['initial_val_loss']:.3f}\n"
        f"Best Epoch (ep {train_res['best_epoch']}):\n"
        f"  Train Loss:       {train_res['final_train_loss']:.3f}\n"
        f"  Val Loss:         {train_res['final_val_loss']:.3f}\n"
        f"  Val Macro-F1:     {train_res['final_val_macro_f1']:.3f}\n"
        f"Completed Epochs:   {train_res['epochs_completed']}"
    )
    ax_loss.text(
        0.38,
        0.95,
        summary_box,
        transform=ax_loss.transAxes,
        fontsize=8.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
    )

    fig_loss.tight_layout()
    loss_fig_path = fig_dir / "fig05_numpy_mlp_loss.png"
    fig_loss.savefig(loss_fig_path, dpi=150)
    plt.close(fig_loss)
    logger.info(f"Saved loss curve plot to {loss_fig_path}")

    # 7. Learning-rate diagnosis for T2d: lr in {0.005, 0.05, 0.5, 2.0}, 100 epochs, no early stopping
    logger.info("Executing learning rate sweep over {0.005, 0.05, 0.5, 2.0} (100 epochs)...")
    sweep_lrs = [0.005, 0.05, 0.5, 2.0]
    sweep_results: dict[str, Any] = {}
    sweep_epochs = 100

    fig_lr, ax_lr = plt.subplots(figsize=(8.5, 5), dpi=150)
    lr_colors = {0.005: "tab:blue", 0.05: "tab:green", 0.5: "tab:orange", 2.0: "tab:red"}

    for sweep_lr in sweep_lrs:
        lr_model = NumpyMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=seed)
        lr_rng = np.random.default_rng(seed)

        # Initial untrained loss at epoch 0
        init_lr_loss = float(lr_model.loss(lr_model.forward(X_train_sub), y_train_sub))
        lr_train_losses = [init_lr_loss]

        N_sub = len(y_train_sub)
        n_b = int(np.ceil(N_sub / batch_size))

        for ep in range(1, sweep_epochs + 1):
            perm = lr_rng.permutation(N_sub)
            X_p = X_train_sub[perm]
            y_p = y_train_sub[perm]

            for b in range(n_b):
                s = b * batch_size
                e = min(s + batch_size, N_sub)
                P_b = lr_model.forward(X_p[s:e])
                lr_model.backward(y_p[s:e])
                lr_model.step(sweep_lr)

            full_loss = float(lr_model.loss(lr_model.forward(X_train_sub), y_train_sub))
            lr_train_losses.append(full_loss)

        # Validation performance at epoch 100
        val_P_lr = lr_model.forward(X_val)
        val_loss_lr = float(lr_model.loss(val_P_lr, y_val))
        val_preds_lr = np.argmax(val_P_lr, axis=1)
        val_f1_lr = float(f1_score(y_val, val_preds_lr, average="macro"))

        sweep_results[str(sweep_lr)] = {
            "learning_rate": sweep_lr,
            "initial_train_loss": init_lr_loss,
            "loss_epoch_10": float(lr_train_losses[10]),
            "loss_epoch_50": float(lr_train_losses[50]),
            "final_train_loss_epoch_100": float(lr_train_losses[100]),
            "final_val_loss_epoch_100": val_loss_lr,
            "final_val_macro_f1_epoch_100": val_f1_lr,
            "min_train_loss": float(min(lr_train_losses)),
        }

        ax_lr.semilogy(
            range(0, sweep_epochs + 1),
            lr_train_losses,
            label=f"lr = {sweep_lr}",
            color=lr_colors[sweep_lr],
            linewidth=1.8,
        )

    ax_lr.set_title("NumPy MLP Learning Rate Sweep: Training Loss vs Epoch (Log Scale)", fontsize=12, fontweight="bold")
    ax_lr.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_lr.set_ylabel("Training Cross-Entropy Loss (log scale)", fontsize=11, fontweight="bold")
    ax_lr.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_lr.legend(loc="upper right", frameon=True, fontsize=10)
    fig_lr.tight_layout()

    lr_fig_path = fig_dir / "fig07_numpy_mlp_lr_sweep.png"
    fig_lr.savefig(lr_fig_path, dpi=150)
    plt.close(fig_lr)
    logger.info(f"Saved learning rate sweep plot to {lr_fig_path}")

    # 8. Persist Training Log (CSV)
    df_history = pd.DataFrame(train_res["history"])
    log_csv_path = logs_dir / "numpy_mlp_training.csv"
    df_history.to_csv(log_csv_path, index=False)
    logger.info(f"Saved training log CSV to {log_csv_path}")

    # 9. Save Metrics JSON
    metrics_data = {
        "model_name": "numpy_mlp",
        "architecture": f"{n_in}-{n_hidden}-{n_out}",
        "activation": "tanh",
        "seed": seed,
        "initial_train_loss": train_res["initial_train_loss"],
        "initial_val_loss": train_res["initial_val_loss"],
        "theoretical_initial_loss": float(theoretical_loss),
        "final_train_loss": train_res["final_train_loss"],
        "final_val_loss": train_res["final_val_loss"],
        "final_val_accuracy": train_res["final_val_acc"],
        "val_macro_f1": train_res["final_val_macro_f1"],
        "best_epoch": train_res["best_epoch"],
        "epochs_completed": train_res["epochs_completed"],
        "stopped_early": train_res["stopped_early"],
        "learning_rate": lr,
        "batch_size": batch_size,
        "patience": patience,
        "train_subset_size": train_subset_size,
        "val_set_size": int(len(y_val)),
        "tiny_batch_overfit": {
            "batch_size": 32,
            "epochs": 2000,
            "initial_loss": overfit_res["initial_loss"],
            "final_loss": overfit_res["final_loss"],
            "final_accuracy": overfit_res["final_accuracy"],
            "memorization_success": bool(
                overfit_res["final_loss"] < 0.01 and overfit_res["final_accuracy"] == 1.0
            ),
        },
        "learning_rate_sweep": sweep_results,
        "gradient_check": {
            "epsilon": eps,
            "tolerance": tol,
            "batch_size": grad_batch_size,
            "all_passed": all_passed,
            "entries_checked": grad_results,
        },
    }
    metrics_path = metrics_dir / "numpy_mlp.json"
    save_json(metrics_data, metrics_path)
    logger.info(f"Saved metrics JSON to {metrics_path}")

    # 10. Save Model Parameters Checkpoint
    checkpoint_path = ckpt_dir / f"numpy_mlp_seed{seed}.npz"
    np.savez_compressed(
        checkpoint_path,
        W1=train_model.W1,
        b1=train_model.b1,
        W2=train_model.W2,
        b2=train_model.b2,
        seed=seed,
        val_macro_f1=train_res["final_val_macro_f1"],
        epochs_completed=train_res["epochs_completed"],
    )
    logger.info(f"Saved weights checkpoint to {checkpoint_path}")

    # 10. Save Run Info
    run_info_path = ckpt_dir / "run_info_numpy_mlp.json"
    write_run_info(
        run_info_path,
        extra_info={
            "model": "numpy_mlp",
            "seed": seed,
            "config_path": config_path,
            "val_macro_f1": train_res["final_val_macro_f1"],
            "final_val_loss": train_res["final_val_loss"],
            "best_epoch": train_res["best_epoch"],
        },
    )
    logger.info(f"Saved run_info to {run_info_path}")

    return metrics_data


if __name__ == "__main__":
    run_numpy_mlp_pipeline()

