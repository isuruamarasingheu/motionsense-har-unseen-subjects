"""
Module: src.benchmark
Purpose: T5a Benchmark inference latency, parameter counts, memory footprint,
and deployment trade-offs for representative model checkpoints.

Coursework requirements:
- Representative seeds: Random Forest seed 1, PyTorch MLP seed 0, PyTorch 1D-CNN seed 2.
- Measurements must use VALIDATION windows only (outputs/cache/X_val_raw.npy, etc.).
  Never touch or unblind the test set.
- Latency benchmarks:
  * Model-only (preprocessed features or preprocessed raw windows).
  * End-to-end (from raw 128x12 sensor windows through feature extraction/scaling to prediction).
  * Batch sizes: 1 (real-time stream) and 256 (batched offline throughput).
  * Warm-up: 10 runs, followed by 100 timed runs using time.perf_counter().
  * Report: mean, standard deviation, median (in milliseconds), and per-window latency for batch 256.
- Size metrics:
  * Trainable parameter count for PyTorch MLP and PyTorch 1D-CNN.
  * Total tree node count for Random Forest (labelled explicitly as 'tree nodes').
  * Checkpoint file size in megabytes (MB).
- Artifacts:
  * outputs/metrics/benchmark.json
  * outputs/metrics/recommendation_evidence.csv
  * outputs/figures/fig19_tradeoff.png (x = latency batch 1 log scale, y = test macro-F1 mean +/- std,
    marker size proportional to checkpoint file size, labelled models).
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn

from src.data.features import extract_features
from src.data.preprocess import FeatureScaler, RawWindowScaler
from src.models.cnn import PyTorchCNN1D
from src.models.mlp import PyTorchMLP
from src.utils import get_cpu_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.benchmark")

# CPU thread configuration
NUM_THREADS = 6
torch.set_num_threads(NUM_THREADS)

# Representative checkpoint paths selected strictly on validation set performance
REPRESENTATIVE_MODELS = {
    "baseline_rf": {
        "display_name": "Random Forest",
        "seed": 1,
        "checkpoint_path": Path("outputs/checkpoints/baseline_rf_seed1.joblib"),
        "config_path": Path("outputs/checkpoints/baseline_rf_seed1_config.yaml"),
    },
    "pytorch_mlp": {
        "display_name": "PyTorch MLP",
        "seed": 0,
        "checkpoint_path": Path("outputs/checkpoints/pytorch_mlp_seed0.pt"),
        "config_path": Path("outputs/checkpoints/pytorch_mlp_seed0_config.yaml"),
    },
    "pytorch_cnn_1d": {
        "display_name": "PyTorch 1D-CNN",
        "seed": 2,
        "checkpoint_path": Path("outputs/checkpoints/pytorch_cnn_1d_seed2.pt"),
        "config_path": Path("outputs/checkpoints/pytorch_cnn_1d_seed2_config.yaml"),
    },
}


def get_cpu_info() -> str:
    """Detect human-readable CPU name from environment or fallback to platform processor."""
    try:
        import subprocess

        cmd = "wmic cpu get name"
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=False)
        lines = [line.strip() for line in res.stdout.split("\n") if line.strip() and "Name" not in line]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes (MB)."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path.stat().st_size / (1024.0 * 1024.0)


def measure_timing(
    fn: Callable[[], Any],
    warm_up_runs: int = 10,
    timed_runs: int = 100,
) -> dict[str, float]:
    """
    Measure execution time of a callable function over repeated runs with warm-up.

    Args:
        fn: Zero-argument callable to time.
        warm_up_runs: Number of un-timed executions to prime caches.
        timed_runs: Number of timed executions.

    Returns:
        Dictionary with mean, std, median, min, and max latency in milliseconds.
    """
    # 1. Warm-up
    for _ in range(warm_up_runs):
        fn()

    # 2. Timed runs with high-resolution performance counter
    latencies_ms: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms, ddof=1)),
        "median_ms": float(np.median(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "timed_runs": timed_runs,
        "warm_up_runs": warm_up_runs,
    }


def benchmark_rf(
    checkpoint_path: Path,
    feat_scaler: FeatureScaler,
    X_val_feat_scaled: np.ndarray,
    X_val_raw: np.ndarray,
) -> dict[str, Any]:
    """Benchmark Random Forest model-only and end-to-end latency for both n_jobs=-1 and n_jobs=1."""
    logger.info("Benchmarking Random Forest from %s ...", checkpoint_path)
    rf = joblib.load(checkpoint_path)
    file_size_mb = get_file_size_mb(checkpoint_path)

    # Count total decision tree nodes across ensemble
    total_nodes = int(sum(tree.tree_.node_count for tree in rf.estimators_))

    # Slice validation inputs
    x_feat_b1 = X_val_feat_scaled[:1]
    x_feat_b256 = X_val_feat_scaled[:256]

    x_raw_b1 = X_val_raw[:1]
    x_raw_b256 = X_val_raw[:256]

    def make_e2e_fn(rf_model, raw_batch):
        def fn():
            f = extract_features(raw_batch)
            f_scaled = feat_scaler.transform(f)
            probs = rf_model.predict_proba(f_scaled)
            return np.argmax(probs, axis=1)
        return fn

    # 1. Benchmark with n_jobs=-1 (multi-threaded pool)
    rf.set_params(n_jobs=-1)
    timing_model_b1_neg1 = measure_timing(lambda: rf.predict_proba(x_feat_b1))
    timing_model_b256_neg1 = measure_timing(lambda: rf.predict_proba(x_feat_b256))
    timing_e2e_b1_neg1 = measure_timing(make_e2e_fn(rf, x_raw_b1))
    timing_e2e_b256_neg1 = measure_timing(make_e2e_fn(rf, x_raw_b256))

    timing_model_b256_neg1["per_window_mean_ms"] = timing_model_b256_neg1["mean_ms"] / 256.0
    timing_model_b256_neg1["per_window_median_ms"] = timing_model_b256_neg1["median_ms"] / 256.0
    timing_e2e_b256_neg1["per_window_mean_ms"] = timing_e2e_b256_neg1["mean_ms"] / 256.0
    timing_e2e_b256_neg1["per_window_median_ms"] = timing_e2e_b256_neg1["median_ms"] / 256.0

    # 2. Benchmark with n_jobs=1 (single-threaded, avoids thread pool synchronization overhead for batch 1)
    rf.set_params(n_jobs=1)
    timing_model_b1_job1 = measure_timing(lambda: rf.predict_proba(x_feat_b1))
    timing_model_b256_job1 = measure_timing(lambda: rf.predict_proba(x_feat_b256))
    timing_e2e_b1_job1 = measure_timing(make_e2e_fn(rf, x_raw_b1))
    timing_e2e_b256_job1 = measure_timing(make_e2e_fn(rf, x_raw_b256))

    timing_model_b256_job1["per_window_mean_ms"] = timing_model_b256_job1["mean_ms"] / 256.0
    timing_model_b256_job1["per_window_median_ms"] = timing_model_b256_job1["median_ms"] / 256.0
    timing_e2e_b256_job1["per_window_mean_ms"] = timing_e2e_b256_job1["mean_ms"] / 256.0
    timing_e2e_b256_job1["per_window_median_ms"] = timing_e2e_b256_job1["median_ms"] / 256.0

    return {
        "model_key": "baseline_rf",
        "display_name": "Random Forest",
        "size_metric_name": "tree nodes",
        "size_metric_value": total_nodes,
        "trainable_parameters": None,
        "tree_nodes": total_nodes,
        "checkpoint_file_size_mb": float(file_size_mb),
        "latency": {
            # Report both n_jobs=-1 and n_jobs=1 in benchmark.json
            "n_jobs_neg1": {
                "model_only": {
                    "batch_1": timing_model_b1_neg1,
                    "batch_256": timing_model_b256_neg1,
                },
                "end_to_end": {
                    "batch_1": timing_e2e_b1_neg1,
                    "batch_256": timing_e2e_b256_neg1,
                },
            },
            "n_jobs_1": {
                "model_only": {
                    "batch_1": timing_model_b1_job1,
                    "batch_256": timing_model_b256_job1,
                },
                "end_to_end": {
                    "batch_1": timing_e2e_b1_job1,
                    "batch_256": timing_e2e_b256_job1,
                },
            },
            # Top-level latency profile: uses n_jobs=1 for batch 1 (per brief) and n_jobs=-1 for batch 256
            "model_only": {
                "batch_1": timing_model_b1_job1,
                "batch_256": timing_model_b256_neg1,
            },
            "end_to_end": {
                "batch_1": timing_e2e_b1_job1,
                "batch_256": timing_e2e_b256_neg1,
            },
        },
    }


def benchmark_mlp(
    checkpoint_path: Path,
    feat_scaler: FeatureScaler,
    X_val_feat_scaled: np.ndarray,
    X_val_raw: np.ndarray,
) -> dict[str, Any]:
    """Benchmark PyTorch MLP model-only and end-to-end latency."""
    logger.info("Benchmarking PyTorch MLP from %s ...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    model = PyTorchMLP(
        input_dim=int(model_cfg.get("input_dim", 84)),
        hidden_sizes=list(model_cfg.get("hidden_sizes", [128, 64])),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    file_size_mb = get_file_size_mb(checkpoint_path)
    total_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Preprocessed tensors for model-only
    t_feat_b1 = torch.from_numpy(X_val_feat_scaled[:1]).float()
    t_feat_b256 = torch.from_numpy(X_val_feat_scaled[:256]).float()

    x_raw_b1 = X_val_raw[:1]
    x_raw_b256 = X_val_raw[:256]

    @torch.no_grad()
    def fn_model_b1():
        logits = model(t_feat_b1)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def fn_model_b256():
        logits = model(t_feat_b256)
        return torch.softmax(logits, dim=1)

    # End-to-end callables
    @torch.no_grad()
    def fn_e2e_b1():
        f = extract_features(x_raw_b1)
        f_scaled = feat_scaler.transform(f)
        t = torch.from_numpy(f_scaled).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

    @torch.no_grad()
    def fn_e2e_b256():
        f = extract_features(x_raw_b256)
        f_scaled = feat_scaler.transform(f)
        t = torch.from_numpy(f_scaled).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

    timing_model_b1 = measure_timing(fn_model_b1)
    timing_model_b256 = measure_timing(fn_model_b256)
    timing_e2e_b1 = measure_timing(fn_e2e_b1)
    timing_e2e_b256 = measure_timing(fn_e2e_b256)

    timing_model_b256["per_window_mean_ms"] = timing_model_b256["mean_ms"] / 256.0
    timing_model_b256["per_window_median_ms"] = timing_model_b256["median_ms"] / 256.0

    timing_e2e_b256["per_window_mean_ms"] = timing_e2e_b256["mean_ms"] / 256.0
    timing_e2e_b256["per_window_median_ms"] = timing_e2e_b256["median_ms"] / 256.0

    return {
        "model_key": "pytorch_mlp",
        "display_name": "PyTorch MLP",
        "size_metric_name": "trainable parameters",
        "size_metric_value": total_params,
        "trainable_parameters": total_params,
        "tree_nodes": None,
        "checkpoint_file_size_mb": float(file_size_mb),
        "latency": {
            "model_only": {
                "batch_1": timing_model_b1,
                "batch_256": timing_model_b256,
            },
            "end_to_end": {
                "batch_1": timing_e2e_b1,
                "batch_256": timing_e2e_b256,
            },
        },
    }


def benchmark_cnn(
    checkpoint_path: Path,
    raw_scaler: RawWindowScaler,
    X_val_raw_scaled: np.ndarray,
    X_val_raw: np.ndarray,
) -> dict[str, Any]:
    """Benchmark PyTorch 1D-CNN model-only and end-to-end latency."""
    logger.info("Benchmarking PyTorch 1D-CNN from %s ...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
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

    file_size_mb = get_file_size_mb(checkpoint_path)
    total_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Preprocessed raw tensors (transposed to channels-first (B, 12, 128))
    raw_scaled_b1 = np.transpose(X_val_raw_scaled[:1], (0, 2, 1))
    raw_scaled_b256 = np.transpose(X_val_raw_scaled[:256], (0, 2, 1))

    t_raw_b1 = torch.from_numpy(raw_scaled_b1).float()
    t_raw_b256 = torch.from_numpy(raw_scaled_b256).float()

    x_raw_b1 = X_val_raw[:1]
    x_raw_b256 = X_val_raw[:256]

    @torch.no_grad()
    def fn_model_b1():
        logits = model(t_raw_b1)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def fn_model_b256():
        logits = model(t_raw_b256)
        return torch.softmax(logits, dim=1)

    # End-to-end callables: raw window(s) -> channel normalisation -> transpose -> CNN
    @torch.no_grad()
    def fn_e2e_b1():
        scaled = raw_scaler.transform(x_raw_b1)
        ch_first = np.transpose(scaled, (0, 2, 1))
        t = torch.from_numpy(ch_first).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

    @torch.no_grad()
    def fn_e2e_b256():
        scaled = raw_scaler.transform(x_raw_b256)
        ch_first = np.transpose(scaled, (0, 2, 1))
        t = torch.from_numpy(ch_first).float()
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

    timing_model_b1 = measure_timing(fn_model_b1)
    timing_model_b256 = measure_timing(fn_model_b256)
    timing_e2e_b1 = measure_timing(fn_e2e_b1)
    timing_e2e_b256 = measure_timing(fn_e2e_b256)

    timing_model_b256["per_window_mean_ms"] = timing_model_b256["mean_ms"] / 256.0
    timing_model_b256["per_window_median_ms"] = timing_model_b256["median_ms"] / 256.0

    timing_e2e_b256["per_window_mean_ms"] = timing_e2e_b256["mean_ms"] / 256.0
    timing_e2e_b256["per_window_median_ms"] = timing_e2e_b256["median_ms"] / 256.0

    return {
        "model_key": "pytorch_cnn_1d",
        "display_name": "PyTorch 1D-CNN",
        "size_metric_name": "trainable parameters",
        "size_metric_value": total_params,
        "trainable_parameters": total_params,
        "tree_nodes": None,
        "checkpoint_file_size_mb": float(file_size_mb),
        "latency": {
            "model_only": {
                "batch_1": timing_model_b1,
                "batch_256": timing_model_b256,
            },
            "end_to_end": {
                "batch_1": timing_e2e_b1,
                "batch_256": timing_e2e_b256,
            },
        },
    }


def load_frozen_quality_metrics() -> dict[str, dict[str, Any]]:
    """
    Read frozen quality numbers strictly from saved files:
    results_summary.json, noise_robustness.json, subgroup.json.

    Rule: Never invent, estimate, or hard-code quality numbers.
    """
    results_path = Path("outputs/metrics/results_summary.json")
    noise_path = Path("outputs/metrics/noise_robustness.json")
    subgroup_path = Path("outputs/metrics/subgroup.json")

    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    with open(noise_path, "r", encoding="utf-8") as f:
        noise_data = json.load(f)

    with open(subgroup_path, "r", encoding="utf-8") as f:
        subgroup_data = json.load(f)

    evidence: dict[str, dict[str, Any]] = {}

    for model_key in ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]:
        res_summary = results_data[model_key]["summary"]
        test_f1_mean = float(res_summary["test_macro_f1_mean"])
        test_f1_std = float(res_summary["test_macro_f1_std"])
        val_f1_mean = float(res_summary["val_macro_f1_mean"])
        val_f1_std = float(res_summary["val_macro_f1_std"])

        # Worst-subject macro-F1 from subgroup.json
        worst_sub_id: str = ""
        worst_sub_f1: float = 1.0
        for sid, sinfo in subgroup_data["subjects"].items():
            mf1 = float(sinfo["models"][model_key]["macro_f1_mean"])
            if mf1 < worst_sub_f1:
                worst_sub_f1 = mf1
                worst_sub_id = sid

        # Noise robustness at sigma = 0.25 and 0.50
        n025 = noise_data["results_by_model"][model_key]["sigmas"]["0.25"]
        n050 = noise_data["results_by_model"][model_key]["sigmas"]["0.5"]

        evidence[model_key] = {
            "test_macro_f1_mean": test_f1_mean,
            "test_macro_f1_std": test_f1_std,
            "val_macro_f1_mean": val_f1_mean,
            "val_macro_f1_std": val_f1_std,
            "worst_subject_id": worst_sub_id,
            "worst_subject_f1": worst_sub_f1,
            "noise_025_mean": float(n025["mean_macro_f1"]),
            "noise_025_std": float(n025.get("std_across_seeds", n025["std_macro_f1"])),
            "noise_050_mean": float(n050["mean_macro_f1"]),
            "noise_050_std": float(n050.get("std_across_seeds", n050["std_macro_f1"])),
        }

    return evidence


def plot_fig19_tradeoff(
    benchmarks: dict[str, dict[str, Any]],
    quality_evidence: dict[str, dict[str, Any]],
    output_path: Path = Path("outputs/figures/fig19_tradeoff.png"),
) -> None:
    """
    Generate Figure 19: Latency vs Test Macro-F1 Trade-off.
    x = end-to-end latency batch 1 (ms, log scale),
    y = test macro-F1 mean with +/- std error bars,
    marker size = proportional to checkpoint file size (MB),
    label each model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)

    model_colors = {
        "baseline_rf": "#2b5c8f",
        "pytorch_mlp": "#d95f02",
        "pytorch_cnn_1d": "#2ca02c",
    }
    model_labels = {
        "baseline_rf": "Random Forest (seed 1)",
        "pytorch_mlp": "PyTorch MLP (seed 0)",
        "pytorch_cnn_1d": "PyTorch 1D-CNN (seed 2)",
    }

    xs = []
    ys = []
    y_errs = []
    sizes = []
    keys = ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]

    for key in keys:
        b = benchmarks[key]
        q = quality_evidence[key]

        x_lat = b["latency"]["end_to_end"]["batch_1"]["mean_ms"]
        y_f1 = q["test_macro_f1_mean"]
        y_err = q["test_macro_f1_std"]
        fsize = b["checkpoint_file_size_mb"]

        xs.append(x_lat)
        ys.append(y_f1)
        y_errs.append(y_err)

        # Scale marker size proportionally (RF is ~22 MB, MLP ~0.08 MB, CNN ~0.5 MB)
        # Visual area scaling: size proportional to sqrt(MB) with min floor
        bubble_size = 120.0 + 35.0 * np.sqrt(fsize * 10.0)
        sizes.append(bubble_size)

        # Plot point with error bar
        ax.errorbar(
            x_lat,
            y_f1,
            yerr=y_err,
            fmt="none",
            ecolor=model_colors[key],
            elinewidth=2.0,
            capsize=6,
            capthick=2.0,
            zorder=3,
        )

        ax.scatter(
            x_lat,
            y_f1,
            s=bubble_size,
            color=model_colors[key],
            alpha=0.75,
            edgecolors="black",
            linewidth=1.5,
            label=f"{model_labels[key]} ({fsize:.2f} MB)",
            zorder=4,
        )

    # Annotate points with clean callouts
    offsets = {
        "baseline_rf": (12, 10),
        "pytorch_mlp": (12, -15),
        "pytorch_cnn_1d": (-25, 12),
    }

    for key, x, y in zip(keys, xs, ys):
        dx, dy = offsets[key]
        b = benchmarks[key]
        q = quality_evidence[key]
        txt = (
            f"{model_labels[key].split(' (')[0]}\n"
            f"F1: {q['test_macro_f1_mean']:.4f}\n"
            f"Lat: {x:.2f} ms | {b['checkpoint_file_size_mb']:.2f} MB"
        )
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#555555", lw=1.0),
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("End-to-End CPU Latency Batch 1 [ms] (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Test Macro-F1 (Mean ± Std)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Figure 19: Latency vs Test Macro-F1 Trade-off\n"
        "(CPU Single-Window Latency vs Generalisation; Bubble Size = Checkpoint File Size)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    # Set generous margins for readability
    ax.set_ylim(0.82, 0.91)
    ax.legend(
        title="Model & Checkpoint Footprint",
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        fontsize=9.5,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 19 trade-off plot to %s", output_path)


def run_benchmarks() -> tuple[dict[str, Any], pd.DataFrame]:
    """Execute complete benchmarking suite and produce all artifacts."""
    logger.info("=== Starting Task T5a Model Benchmarking Suite ===")

    # 1. Environment metadata
    cpu_name = get_cpu_name()
    os_name = f"{platform.system()} {platform.release()} ({platform.version()}, {platform.machine()})"
    python_ver = platform.python_version()
    torch_ver = torch.__version__
    sklearn_ver = sklearn.__version__

    logger.info("Environment: CPU=%s, OS=%s, Python=%s, PyTorch=%s, scikit-learn=%s, Threads=%d",
                cpu_name, os_name, python_ver, torch_ver, sklearn_ver, NUM_THREADS)

    # 2. Load validation windows strictly (Rule compliance: never touch test set for timing)
    cache_dir = Path("outputs/cache")
    logger.info("Loading VALIDATION windows from outputs/cache/ ...")
    X_val_raw = np.load(cache_dir / "X_val_raw.npy")
    X_val_feat_scaled = np.load(cache_dir / "X_val_feat_scaled.npy")
    X_val_raw_scaled = np.load(cache_dir / "X_val_raw_scaled.npy")

    logger.info("Validation inputs loaded: X_val_raw=%s, X_val_feat_scaled=%s, X_val_raw_scaled=%s",
                X_val_raw.shape, X_val_feat_scaled.shape, X_val_raw_scaled.shape)

    # 3. Load saved scalers
    feat_scaler = FeatureScaler.load()
    raw_scaler = RawWindowScaler.load()

    # 4. Execute benchmarks
    rf_bench = benchmark_rf(
        checkpoint_path=REPRESENTATIVE_MODELS["baseline_rf"]["checkpoint_path"],
        feat_scaler=feat_scaler,
        X_val_feat_scaled=X_val_feat_scaled,
        X_val_raw=X_val_raw,
    )

    mlp_bench = benchmark_mlp(
        checkpoint_path=REPRESENTATIVE_MODELS["pytorch_mlp"]["checkpoint_path"],
        feat_scaler=feat_scaler,
        X_val_feat_scaled=X_val_feat_scaled,
        X_val_raw=X_val_raw,
    )

    cnn_bench = benchmark_cnn(
        checkpoint_path=REPRESENTATIVE_MODELS["pytorch_cnn_1d"]["checkpoint_path"],
        raw_scaler=raw_scaler,
        X_val_raw_scaled=X_val_raw_scaled,
        X_val_raw=X_val_raw,
    )

    benchmarks = {
        "baseline_rf": rf_bench,
        "pytorch_mlp": mlp_bench,
        "pytorch_cnn_1d": cnn_bench,
    }

    # 5. Load quality evidence from saved files
    quality_evidence = load_frozen_quality_metrics()

    # 6. Build outputs/metrics/benchmark.json
    benchmark_payload = {
        "environment": {
            "cpu_name": cpu_name,
            "os": os_name,
            "python_version": python_ver,
            "torch_version": torch_ver,
            "sklearn_version": sklearn_ver,
            "torch_num_threads": NUM_THREADS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "models": benchmarks,
    }

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json_path = metrics_dir / "benchmark.json"

    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    logger.info("Saved benchmark results to %s", benchmark_json_path)

    # 7. Build outputs/metrics/recommendation_evidence.csv
    # Requirement: one row per model: test macro-F1 mean +/- std (3 seeds), val macro-F1 mean +/- std,
    # worst-subject test F1 (from subgroup.json), F1 at noise sigma 0.25 and 0.5, params/nodes,
    # file size, end-to-end latency batch 1, per-window latency batch 256.
    evidence_rows = []
    for model_key in ["baseline_rf", "pytorch_mlp", "pytorch_cnn_1d"]:
        b = benchmarks[model_key]
        q = quality_evidence[model_key]

        if model_key == "baseline_rf":
            size_repr = f"{b['tree_nodes']:,} tree nodes"
        else:
            size_repr = f"{b['trainable_parameters']:,} params"

        e2e_b1_mean = b["latency"]["end_to_end"]["batch_1"]["mean_ms"]
        e2e_b1_std = b["latency"]["end_to_end"]["batch_1"]["std_ms"]
        e2e_b256_per_win = b["latency"]["end_to_end"]["batch_256"]["per_window_mean_ms"]

        if model_key == "baseline_rf":
            thread_setting = "n_jobs=1 (b1) / n_jobs=-1 (b256)"
        else:
            thread_setting = f"torch_num_threads={NUM_THREADS}"

        evidence_rows.append({
            "model": model_key,
            "display_name": b["display_name"],
            "representative_seed": REPRESENTATIVE_MODELS[model_key]["seed"],
            "test_macro_f1": f"{q['test_macro_f1_mean']:.4f} ± {q['test_macro_f1_std']:.4f}",
            "val_macro_f1": f"{q['val_macro_f1_mean']:.4f} ± {q['val_macro_f1_std']:.4f}",
            "worst_subject_test_f1": f"{q['worst_subject_f1']:.4f} (Sub {q['worst_subject_id']})",
            "f1_noise_sigma_0_25": f"{q['noise_025_mean']:.4f} ± {q['noise_025_std']:.4f}",
            "f1_noise_sigma_0_50": f"{q['noise_050_mean']:.4f} ± {q['noise_050_std']:.4f}",
            "size_metric": size_repr,
            "file_size_mb": f"{b['checkpoint_file_size_mb']:.2f}",
            "e2e_latency_batch_1_ms": f"{e2e_b1_mean:.2f} ± {e2e_b1_std:.2f}",
            "e2e_latency_batch_256_per_window_ms": f"{e2e_b256_per_win:.4f}",
            "threading_setting": thread_setting,
        })

    df_evidence = pd.DataFrame(evidence_rows)
    evidence_csv_path = metrics_dir / "recommendation_evidence.csv"
    df_evidence.to_csv(evidence_csv_path, index=False)
    logger.info("Saved recommendation evidence table to %s", evidence_csv_path)

    # 8. Plot Figure 19: Trade-off plot
    plot_fig19_tradeoff(benchmarks, quality_evidence)

    return benchmark_payload, df_evidence


if __name__ == "__main__":
    run_benchmarks()
